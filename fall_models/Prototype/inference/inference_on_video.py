#!/usr/bin/env python3
"""
MP4 -> YOLOv11 pose (yolo11l-pose.pt) -> temporal model inference -> popup display.

Preprocessing mirrors training:
  - dataset_helpers/dataset.py: fill/interp missing joints, optional normalize/vel/acc/global + mask channel
  - training/train_models.py: temporal model architectures + checkpoint metadata

Usage:
  python -m inference.inference_on_video --video path\\to\\clip.mp4 --model models\\tcn\\<run>\\tcn_best.pt
  python -m inference.inference_on_video --video path\\to\\clip.mp4 --model models\\tcn --save out.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

# Allow running as a script from any working directory (mirrors `python -m ...`).
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import dataset_helpers.dataset as ds

from models.tcn.simple_tcn import TCNBaseline
from models.lstm.simple_lstm import LSTMBaseline
from models.gru.simple_gru import GRUBaseline
from models.gcn.simple_gcn import GCNBaseline
from models.mlp.simple_mlp import MLPBaseline
from models.stgcn.simple_stgcn import STGCNBaseline

try:
    from models.cnnlstm.cnn_lstm import CNNLSTMTwoHead
except Exception:
    CNNLSTMTwoHead = None

from models.rf.train_rf import windows_to_sklearn_features


K = 17  # COCO-17 joints for Ultralytics pose models

SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12),
]

KNOWN_ARCHES = ["tcn", "lstm", "gru", "gcn", "mlp", "stgcn", "cnnlstm", "rf"]

# Updated fall-merged label names (7 classes)
# 0: Fall (all fall subclasses merged)
# 1..6: ADLs
FALL_MERGED_CLASS_NAMES = [
    "Fall",
    "Walking",
    "Standing",
    "Sitting",
    "Picking up an object",
    "Jumping",
    "Laying",
]


def pick_device(device: Optional[str]) -> str:
    if not device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    d = device.lower().strip()
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def infer_arch_from_path(p: Path) -> Optional[str]:
    tokens = [p.name.lower(), p.stem.lower()] + [x.lower() for x in p.parts]
    for arch in sorted(KNOWN_ARCHES, key=len, reverse=True):
        if any(tok == arch for tok in tokens):
            return arch
        if any(tok.startswith(arch + "_") for tok in tokens):
            return arch
        if arch == "rf":
            # Avoid substring matches like "pe[r f]ormance" -> "rf".
            if any(tok.startswith("rf") for tok in tokens):
                return arch
            continue
        if any(arch in tok for tok in tokens):
            return arch
    return None


def resolve_ckpt_and_arch(model_arg: str, arch_arg: Optional[str]) -> Tuple[Path, str]:
    """
    --model can be:
      - a checkpoint file (*.pt or *.pkl)
      - a model folder containing checkpoints (picks newest *best*.pt / *best*.pkl)
      - a model python file under models/<arch>/...py (picks newest *best*.pt / *best*.pkl under that folder)
    """
    p = Path(model_arg).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"--model not found: {p}")

    arch = (arch_arg or "").lower().strip() or infer_arch_from_path(p)

    if p.is_file():
        suf = p.suffix.lower()
        if suf in {".pt", ".pth", ".bin"}:
            if not arch:
                arch = infer_arch_from_path(p)
            if not arch:
                raise ValueError("Could not infer --arch from checkpoint path. Pass --arch explicitly.")
            if arch == "rf":
                raise ValueError(
                    "Inferred/selected --arch rf for a .pt checkpoint. RF checkpoints are expected to be .pkl/.pickle.\n"
                    "Pass the correct --arch (tcn/lstm/...) or provide an RF .pkl checkpoint."
                )
            return p, arch

        if suf in {".pkl", ".pickle"}:
            if not arch:
                arch = infer_arch_from_path(p) or "rf"
            return p, arch

        if suf == ".py":
            if not arch:
                raise ValueError("Could not infer --arch from model .py path. Pass --arch explicitly.")
            model_dir = p.parent
            if arch == "rf":
                ckpts = sorted(model_dir.glob("**/*best*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
                if not ckpts:
                    ckpts = sorted(model_dir.glob("**/*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
            else:
                ckpts = sorted(model_dir.glob("**/*best*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
                if not ckpts:
                    ckpts = sorted(model_dir.glob("**/*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found under: {model_dir}")
            return ckpts[0], arch

        raise ValueError(f"Unsupported --model file type: {p.suffix}")

    if arch == "rf":
        ckpts = sorted(p.glob("**/*best*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not ckpts:
            ckpts = sorted(p.glob("**/*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    else:
        ckpts = sorted(p.glob("**/*best*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not ckpts:
            # Allow pointing to a folder containing an RF run folder without passing --arch rf.
            ckpts = sorted(p.glob("**/*best*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not ckpts:
            ckpts = sorted(p.glob("**/*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not ckpts:
            ckpts = sorted(p.glob("**/*.pkl"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint *.pt/*.pkl files found under: {p}")

    ckpt = ckpts[0]
    if not arch:
        arch = infer_arch_from_path(ckpt)
    if not arch:
        if ckpt.suffix.lower() in {".pkl", ".pickle"}:
            arch = "rf"
        else:
            raise ValueError("Could not infer --arch from checkpoint path. Pass --arch explicitly.")
    return ckpt, arch


def load_checkpoint(ckpt_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    # PyTorch >=2.6 defaults `weights_only=True`, which can fail on our training checkpoints
    # because they include non-tensor metadata (e.g., NumPy scalars). We trained these
    # checkpoints ourselves, so we opt into the legacy behavior for compatibility.
    try:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"], ckpt_obj
    if isinstance(ckpt_obj, dict):
        return ckpt_obj, {}
    raise TypeError("Unsupported checkpoint format (expected dict or dict with 'state_dict').")


def clean_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(state.keys())
    if any(k.startswith("module.") for k in keys):
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_rf_checkpoint(ckpt_path: Path) -> Dict[str, object]:
    """
    Loads a Random Forest checkpoint saved by `models/rf/train_rf.py`.
    """
    try:
        with Path(ckpt_path).open("rb") as f:
            obj = pickle.load(f)
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Failed to load RF checkpoint. You likely need scikit-learn installed.\n"
            "Install it with: pip install scikit-learn\n"
            f"Import error: {e}"
        )

    if not isinstance(obj, dict) or "model" not in obj:
        raise TypeError(f"Unsupported RF checkpoint format: expected a dict with key 'model'. Got: {type(obj)}")
    return obj


def load_class_names(num_classes: int, meta: Dict[str, object], labels_file: Optional[str]) -> List[str]:
    names: List[str] = []
    if labels_file:
        p = Path(labels_file).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"--labels-file not found: {p}")
        names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    # Default to updated fall-merged names when displaying 7-class outputs.
    if not names and int(num_classes) == 7:
        return list(FALL_MERGED_CLASS_NAMES)

    if not names:
        for key in ("new_label_names", "class_names", "classes", "labels"):
            v = meta.get(key, None)
            if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                names = list(v)
                break

    if not names:
        names = [f"class_{i}" for i in range(int(num_classes))]

    if len(names) != int(num_classes):
        names = names[: int(num_classes)] + [f"class_{i}" for i in range(len(names), int(num_classes))]
    return names


def build_temporal_model(
    arch: str,
    in_features: int,
    num_classes: int,
    device: str,
    T_used: int,
    node_features: Optional[int],
) -> nn.Module:
    arch = arch.lower().strip()
    if arch == "tcn":
        model = TCNBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            kernel_size=3,
            dropout=0.1,
        )
    elif arch == "lstm":
        model = LSTMBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
            pool="last",
        )
    elif arch == "gru":
        model = GRUBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
            pool="last",
        )
    elif arch == "gcn":
        if node_features is None:
            raise ValueError("GCN requires node_features (in_features must be divisible by 17).")
        model = GCNBaseline(
            num_nodes=17,
            node_features=int(node_features),
            num_classes=num_classes,
            hidden_size=64,
            dropout=0.1,
        )
    elif arch == "mlp":
        model = MLPBaseline(
            T=int(T_used),
            in_features=in_features,
            num_classes=num_classes,
            hidden_sizes=(256, 128),
            dropout=0.2,
        )
    elif arch == "stgcn":
        if node_features is None:
            raise ValueError("STGCN requires node_features (in_features must be divisible by 17).")
        model = STGCNBaseline(
            num_nodes=17,
            node_features=int(node_features),
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            t_kernel=9,
            dropout=0.1,
        )
    elif arch == "cnnlstm":
        if CNNLSTMTwoHead is None:
            raise RuntimeError("CNNLSTMTwoHead import failed (models/cnnlstm).")
        model = CNNLSTMTwoHead(
            in_features=in_features,
            num_classes=num_classes,
            embed_dim=128,
            hidden_size=128,
            lstm_layers=1,
            dropout=0.2,
            num_keypoints=17 if node_features is not None else None,
            kp_channels=node_features,
            pool="last",
        )
    else:
        raise ValueError(f"Unknown --arch: {arch} (expected one of {KNOWN_ARCHES})")
    model = model.to(device)
    print("Model device:", next(model.parameters()).device)
    return model


def draw_hud(frame: np.ndarray, lines: List[str], org: Tuple[int, int] = (10, 10)) -> np.ndarray:
    if not lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    pad = 8
    line_gap = 6
    bg_alpha = 0.6

    x0, y0 = org
    sizes = [cv2.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
    max_w = max(w for (w, h) in sizes)
    total_h = sum(h for (w, h) in sizes) + line_gap * (len(lines) - 1)

    box_w = max_w + 2 * pad
    box_h = total_h + 2 * pad
    h_img, w_img = frame.shape[:2]
    x1 = min(w_img - 1, x0 + box_w)
    y1 = min(h_img - 1, y0 + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0)

    y = y0 + pad
    for (w, h), s in zip(sizes, lines):
        y += h
        cv2.putText(frame, s, (x0 + pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_gap
    return frame


def draw_pose(frame: np.ndarray, xy: np.ndarray, conf: np.ndarray, conf_thres: float) -> np.ndarray:
    for i in range(K):
        if float(conf[i]) > float(conf_thres):
            x, y = int(xy[i, 0]), int(xy[i, 1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    for a, b in SKELETON:
        if float(conf[a]) > float(conf_thres) and float(conf[b]) > float(conf_thres):
            ax, ay = int(xy[a, 0]), int(xy[a, 1])
            bx, by = int(xy[b, 0]), int(xy[b, 1])
            cv2.line(frame, (ax, ay), (bx, by), (0, 255, 255), 2)
    return frame


def open_video_writer(save_path: Path, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = save_path.suffix.lower()
    if suffix in {".avi"}:
        codecs = ["XVID", "MJPG", "mp4v"]
    elif suffix in {".mp4", ".m4v", ".mov"}:
        codecs = ["mp4v", "avc1", "H264", "MJPG"]
    else:
        codecs = ["mp4v", "MJPG"]

    w, h = int(frame_size[0]), int(frame_size[1])
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(save_path), fourcc, float(fps), (w, h))
        if writer.isOpened():
            print(f"[save] writing: {save_path.as_posix()} ({w}x{h} @{float(fps):.2f}fps, codec={codec})")
            return writer

    raise RuntimeError(f"Could not open VideoWriter for: {save_path} (tried codecs={codecs})")


def feature_layout(use_conf: bool, add_vel: bool, add_acc: bool) -> Dict[str, Optional[object]]:
    idx = 2  # xy
    conf_idx = None
    if use_conf:
        conf_idx = idx
        idx += 1
    vel_slice = None
    if add_vel:
        vel_slice = slice(idx, idx + 2)
        idx += 2
    acc_slice = None
    if add_acc:
        acc_slice = slice(idx, idx + 2)
        idx += 2
    return {"conf_idx": conf_idx, "vel_slice": vel_slice, "acc_slice": acc_slice}


def expected_in_features(
    use_conf: bool,
    add_vel: bool,
    add_acc: bool,
    add_global: bool,
    add_mask: bool,
) -> int:
    if add_acc and not add_vel:
        raise ValueError("add_acc=True requires add_vel=True (acc is computed from vel).")
    c = 2
    if use_conf:
        c += 1
    if add_vel:
        c += 2
    if add_acc:
        c += 2
    if add_global:
        c += 4
    if add_mask:
        c += 1
    return int(K * c)


def pose_on_frame(
    pose_model: YOLO,
    frame_bgr: np.ndarray,
    imgsz: int,
    yolo_conf: float,
    device: str,
    max_people: int,
    use_half: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    xy_zeros = np.zeros((K, 2), dtype=np.float32)
    cf_zeros = np.zeros((K,), dtype=np.float32)

    results = pose_model.predict(
        source=frame_bgr,
        imgsz=int(imgsz),
        conf=float(yolo_conf),
        verbose=False,
        device=device,
        half=bool(use_half),
        max_det=max(1, int(max_people)),
    )
    if not results or len(results) == 0 or results[0].keypoints is None:
        return xy_zeros, cf_zeros

    kpts = results[0].keypoints
    xy_all = kpts.xy.cpu().numpy() if hasattr(kpts.xy, "cpu") else np.array(kpts.xy)
    cf_all = kpts.conf.cpu().numpy() if (hasattr(kpts, "conf") and hasattr(kpts.conf, "cpu")) else None

    if xy_all.ndim != 3 or xy_all.shape[0] == 0:
        return xy_zeros, cf_zeros
    if xy_all.shape[1] != K:
        raise ValueError(f"Expected {K} keypoints, got {xy_all.shape[1]}")

    if cf_all is not None and cf_all.ndim == 2:
        scores = cf_all.sum(axis=1)
        best = int(np.argmax(scores))
        xy = xy_all[best].astype(np.float32, copy=False)
        cf = cf_all[best].astype(np.float32, copy=False)
        return xy, cf

    # No confidences available: treat as all-ones (model will likely ignore if use_conf=False)
    best = 0
    xy = xy_all[best].astype(np.float32, copy=False)
    cf = np.ones((K,), dtype=np.float32)
    return xy, cf


def make_window_features(
    xy_seq: np.ndarray,      # (L,K,2) in pixel coords
    conf_seq: np.ndarray,    # (L,K)
    T: int,
    use_conf: bool,
    normalize: bool,
    normalize_mode: str,
    add_vel: bool,
    add_acc: bool,
    add_global: bool,
    add_mask: bool,
    conf_thres: float,
    max_interp_gap: int,
    missing_mode: str,
    interp_mode: str,
    interp_group: int,
    rp_center_mode: str,
    rp_img_w: Optional[int],
    rp_img_h: Optional[int],
    min_valid_frac: float,
) -> np.ndarray:
    """
    Build a single window feature tensor (T,F) matching dataset_helpers/dataset.py + training/train_models.py.
    """
    T = int(T)
    L = int(xy_seq.shape[0])
    if L <= 0:
        feat_dim = expected_in_features(use_conf, add_vel, add_acc, add_global, add_mask)
        return np.zeros((T, feat_dim), dtype=np.float32)

    missing_mode = str(missing_mode).lower().strip()
    interp_mode = str(interp_mode).lower().strip()
    if missing_mode == "conf_thres" and interp_mode == "short_gap_hold":
        xy_filled, conf_filled = ds._fill_and_mask_kpts(
            xy_seq.astype(np.float32, copy=False),
            conf_seq.astype(np.float32, copy=False),
            conf_thres=float(conf_thres),
            max_interp_gap=int(max_interp_gap),
        )
    else:
        xy_filled, conf_filled = ds._fill_and_mask_kpts_paper(
            xy_seq.astype(np.float32, copy=False),
            conf_seq.astype(np.float32, copy=False),
            conf_thres=float(conf_thres),
            missing_mode=str(missing_mode),
            interp_mode=str(interp_mode),
            max_interp_gap=int(max_interp_gap),
            interp_group=int(interp_group),
        )

    if not bool(normalize):
        xy_used = xy_filled.astype(np.float32, copy=False)
    else:
        nm = str(normalize_mode).lower().strip()
        if nm == "center_scale":
            xy_used = ds._normalize_xy(xy_filled, conf_filled)
        elif nm == "paper_rp":
            center = ds._compute_image_center(
                xy=xy_filled,
                rp_center_mode=str(rp_center_mode),
                rp_img_w=rp_img_w,
                rp_img_h=rp_img_h,
            )
            xy_used = ds._normalize_xy_paper_rp(xy_filled, conf_filled, center=center)
        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

    parts = [xy_used]
    if use_conf:
        parts.append(conf_filled[..., None])

    vel = None
    if add_vel:
        vel = ds._add_velocity_channels(xy_used)
        parts.append(vel)
    if add_acc:
        if vel is None:
            vel = ds._add_velocity_channels(xy_used)
        parts.append(ds._add_acceleration_channels(vel))
    if add_global:
        g = ds._global_features(xy_used, conf_filled)  # (L,4)
        parts.append(np.repeat(g[:, None, :], repeats=K, axis=1))

    Xf = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)  # (L,K,C)

    frac_valid = (conf_filled > float(conf_thres)).mean(axis=1)
    valid = frac_valid >= float(min_valid_frac)  # (L,)

    layout = feature_layout(use_conf=use_conf, add_vel=add_vel, add_acc=add_acc)
    seq = Xf

    if L < T:
        pad = np.repeat(seq[-1:, :, :], repeats=(T - L), axis=0) if L > 0 else np.zeros((T, K, Xf.shape[2]), np.float32)
        if layout["conf_idx"] is not None:
            pad[:, :, int(layout["conf_idx"])] = 0.0
        if layout["vel_slice"] is not None:
            pad[:, :, layout["vel_slice"]] = 0.0
        if layout["acc_slice"] is not None:
            pad[:, :, layout["acc_slice"]] = 0.0

        seq = np.concatenate([seq, pad], axis=0)
        valid = np.concatenate([valid, np.zeros((T - L,), dtype=bool)], axis=0)

    seq = seq.copy()
    seq[~valid] = 0.0
    if add_mask:
        m = np.repeat(valid.astype(np.float32)[:, None, None], repeats=K, axis=1)
        seq = np.concatenate([seq, m], axis=-1)

    # Flatten to (T, F)
    return seq.reshape(T, int(seq.shape[1]) * int(seq.shape[2])).astype(np.float32, copy=False)


@torch.no_grad()
def infer_one_window(
    model: nn.Module,
    window_feat: np.ndarray,  # (T,F)
    device: str,
    use_half: bool,
    merge_fall_11_to_7: bool,
) -> Tuple[int, float, Optional[float]]:
    model.eval()

    xb = torch.from_numpy(window_feat[None, ...]).to(device)
    xb = xb.half() if use_half else xb.float()

    out = model(xb)
    fall_logit = None
    if isinstance(out, (tuple, list)) and len(out) == 2:
        logits, fall_logit = out[0], out[1]
    else:
        logits = out

    if logits.ndim == 3:
        logits = logits[:, -1, :]

    prob = torch.softmax(logits, dim=-1)
    if merge_fall_11_to_7:
        if int(prob.shape[-1]) != 11:
            raise ValueError(f"merge_fall_11_to_7=True expects 11 classes, got {int(prob.shape[-1])}")
        prob = torch.cat([prob[:, :5].sum(dim=1, keepdim=True), prob[:, 5:]], dim=1)  # (1,7)

    pconf, pred = torch.max(prob, dim=-1)

    p_fall = None
    if fall_logit is not None:
        p_fall = float(torch.sigmoid(fall_logit.view(-1))[0].item())

    return int(pred.item()), float(pconf.item()), p_fall


def _rf_predict_proba_aligned(clf, X_feat: np.ndarray, num_classes: int) -> np.ndarray:
    if not hasattr(clf, "predict_proba"):
        raise TypeError("RF checkpoint model does not implement predict_proba().")
    raw = clf.predict_proba(X_feat)
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"Expected RF predict_proba output (N,C). Got shape {getattr(raw, 'shape', None)}")

    num_classes = int(num_classes)
    out = np.zeros((int(raw.shape[0]), int(num_classes)), dtype=np.float32)

    classes = getattr(clf, "classes_", None)
    if classes is None:
        if int(raw.shape[1]) != int(num_classes):
            raise ValueError(f"RF predict_proba returned C={int(raw.shape[1])}, expected num_classes={int(num_classes)}")
        return raw.astype(np.float32, copy=False)

    classes_np = np.asarray(classes).astype(np.int64, copy=False).reshape(-1)
    for j, cls_id in enumerate(classes_np.tolist()):
        if 0 <= int(cls_id) < int(num_classes) and j < int(raw.shape[1]):
            out[:, int(cls_id)] = raw[:, int(j)]

    return out


def infer_one_window_rf(
    clf,
    window_feat: np.ndarray,  # (T,F)
    feature_mode: str,
    num_classes: int,
    expected_feature_dim: Optional[int] = None,
) -> Tuple[int, float, Optional[float]]:
    X_feat = windows_to_sklearn_features(window_feat[None, ...], mode=str(feature_mode))
    if expected_feature_dim is not None and int(expected_feature_dim) > 0 and int(X_feat.shape[1]) != int(expected_feature_dim):
        raise ValueError(
            f"RF feature_dim mismatch: extracted={int(X_feat.shape[1])}, ckpt feature_dim={int(expected_feature_dim)} "
            f"(mode={str(feature_mode)})"
        )

    probs = _rf_predict_proba_aligned(clf, X_feat, num_classes=int(num_classes))[0]
    pred = int(np.argmax(probs))
    pconf = float(probs[pred]) if probs.size > 0 else 0.0
    p_fall = float(probs[0]) if probs.size > 0 else None
    return pred, pconf, p_fall


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream windowed pose inference on an MP4 with YOLO pose overlay.")
    ap.add_argument("--video", type=str, required=True, help="Path to input .mp4")
    ap.add_argument("--model", type=str, required=True, help="Checkpoint *.pt/*.pkl OR model folder OR model .py")
    ap.add_argument("--arch", type=str, default=None, choices=KNOWN_ARCHES, help="Override model architecture if needed")
    ap.add_argument("--yolo-weights", type=str, default="yolo11l-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--max-people", type=int, default=1)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--half", type=int, default=0, help="Use FP16 on CUDA for YOLO+temporal model (0/1)")
    ap.add_argument("--T", type=int, default=0, help="0 => use ckpt T_used/T, else override")
    ap.add_argument("--stride", type=int, default=0, help="0 => use ckpt stride, else override")
    ap.add_argument(
        "--normalize-mode",
        type=str,
        default=None,
        choices=["center_scale", "paper_rp"],
        help="Override checkpoint normalize_mode when --normalize 1 (center_scale or paper_rp).",
    )
    ap.add_argument(
        "--missing-mode",
        type=str,
        default=None,
        choices=["conf_thres", "zeros_only", "conf_or_zeros"],
        help="Override checkpoint missing_mode (conf_thres, zeros_only, conf_or_zeros).",
    )
    ap.add_argument(
        "--interp-mode",
        type=str,
        default=None,
        choices=["short_gap_hold", "paper_group_linear"],
        help="Override checkpoint interp_mode (short_gap_hold or paper_group_linear).",
    )
    ap.add_argument(
        "--interp-group",
        type=int,
        default=0,
        help="Override checkpoint interp_group (>0). Only used for --interp-mode paper_group_linear.",
    )
    ap.add_argument(
        "--rp-center-mode",
        type=str,
        default=None,
        choices=["auto", "normalized_01", "pixel"],
        help="Override checkpoint rp_center_mode for --normalize-mode paper_rp.",
    )
    ap.add_argument("--rp-img-w", type=int, default=0, help="Image width W for paper_rp when using pixel coordinates (0 => auto from video).")
    ap.add_argument("--rp-img-h", type=int, default=0, help="Image height H for paper_rp when using pixel coordinates (0 => auto from video).")
    ap.add_argument("--labels-file", type=str, default=None)
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save annotated output video (e.g. out.mp4). If a directory, writes <video_stem>_annotated.mp4 inside.",
    )
    ap.add_argument("--display-fps", type=float, default=0.0, help="0 => source fps")
    args = ap.parse_args()

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"--video not found: {video_path}")

    save_path: Optional[Path] = None
    if args.save:
        save_arg = Path(args.save).expanduser()
        if str(args.save).endswith(("/", "\\")) or (save_arg.exists() and save_arg.is_dir()):
            save_path = save_arg / f"{video_path.stem}_annotated.mp4"
        else:
            save_path = save_arg
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".mp4")

    device = pick_device(args.device)
    use_half = bool(int(args.half)) and device.startswith("cuda")

    ckpt_path, arch = resolve_ckpt_and_arch(args.model, args.arch)
    print(f"[model] arch={arch} ckpt={ckpt_path.as_posix()}")

    is_rf = str(arch).lower().strip() == "rf" or ckpt_path.suffix.lower() in {".pkl", ".pickle"}
    rf_model = None
    rf_feature_mode = "flatten"
    rf_feature_dim = None

    if is_rf:
        meta = load_rf_checkpoint(ckpt_path)
        rf_model = meta.get("model", None)
        rf_feature_mode = str(meta.get("feature_mode", "flatten"))
        rf_feature_dim = meta.get("feature_dim", None)
        state = None
    else:
        state, meta = load_checkpoint(ckpt_path)
        state = clean_state_dict(state)

    # Preproc config (prefer checkpoint meta)
    T_final = int(args.T) if int(args.T) > 0 else int(meta.get("T", meta.get("T_used", 64)) or 64)
    stride_final = int(args.stride) if int(args.stride) > 0 else int(meta.get("stride", 16) or 16)
    T_final = max(1, int(T_final))
    stride_final = max(1, int(stride_final))

    use_conf = bool(meta.get("use_conf", True))
    normalize = bool(meta.get("normalize", True))
    normalize_mode = str(args.normalize_mode) if args.normalize_mode else str(meta.get("normalize_mode") or "center_scale")
    add_vel = bool(meta.get("add_vel", True))
    add_acc = bool(meta.get("add_acc", True))
    add_global = bool(meta.get("add_global", True))
    add_mask = bool(meta.get("add_mask_channel", True))
    conf_thres = float(meta.get("conf_thres", 0.2))
    max_interp_gap = int(meta.get("max_interp_gap", 5))
    missing_mode = str(args.missing_mode) if args.missing_mode else str(meta.get("missing_mode") or "conf_thres")
    interp_mode = str(args.interp_mode) if args.interp_mode else str(meta.get("interp_mode") or "short_gap_hold")
    interp_group = int(args.interp_group) if int(args.interp_group) > 0 else int(meta.get("interp_group", 100) or 100)
    rp_center_mode = str(args.rp_center_mode) if args.rp_center_mode else str(meta.get("rp_center_mode") or "auto")
    rp_img_w: Optional[int] = None
    rp_img_h: Optional[int] = None
    if int(args.rp_img_w) > 0:
        rp_img_w = int(args.rp_img_w)
    elif meta.get("rp_img_w", None) is not None:
        rp_img_w = int(meta.get("rp_img_w"))  # type: ignore[arg-type]
    if int(args.rp_img_h) > 0:
        rp_img_h = int(args.rp_img_h)
    elif meta.get("rp_img_h", None) is not None:
        rp_img_h = int(meta.get("rp_img_h"))  # type: ignore[arg-type]
    min_valid_frac = float(meta.get("min_valid_frac", 0.3))

    num_classes = int(meta.get("num_classes", 0) or 0)
    in_features_meta = int(meta.get("in_features", 0) or 0)
    if num_classes <= 0:
        if is_rf:
            nln = meta.get("new_label_names", None)
            if isinstance(nln, (list, tuple)):
                num_classes = int(len(nln))
            if int(num_classes) <= 0:
                num_classes = 7
        else:
            raise ValueError("Checkpoint missing num_classes. Use a checkpoint from training/train_models.py.")

    merge_fall_11_to_7 = int(num_classes) == 11
    display_num_classes = 7 if merge_fall_11_to_7 else int(num_classes)
    class_names = load_class_names(num_classes=display_num_classes, meta=meta, labels_file=args.labels_file)

    in_features = expected_in_features(
        use_conf=use_conf,
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        add_mask=add_mask,
    )
    if in_features_meta > 0 and int(in_features) != int(in_features_meta):
        raise ValueError(f"Feature mismatch: expected in_features={in_features}, ckpt expects {in_features_meta}")

    if is_rf:
        if rf_model is None:
            raise ValueError("RF checkpoint missing 'model'.")
    else:
        node_features_meta = meta.get("node_features", None)
        if node_features_meta is None:
            nf = int(in_features // K)
            node_features_meta = nf if nf * K == int(in_features) else None

        model = build_temporal_model(
            arch=arch,
            in_features=int(in_features),
            num_classes=int(num_classes),
            device=device,
            T_used=int(T_final),
            node_features=int(node_features_meta) if node_features_meta is not None else None,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print("[WARN] missing keys:", missing[:8], "..." if len(missing) > 8 else "")
        if unexpected:
            print("[WARN] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")
        model.eval()

    pose_model = YOLO(str(Path(args.yolo_weights).expanduser()))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    writer: Optional[cv2.VideoWriter] = None
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(src_fps) or src_fps <= 1e-3:
            src_fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fps_play = float(args.display_fps) if float(args.display_fps) > 1e-3 else float(src_fps)
        delay_ms = max(1, int(round(1000.0 / max(1e-6, fps_play))))

        frames_buf: deque[np.ndarray] = deque()
        xy_buf: deque[np.ndarray] = deque()
        cf_buf: deque[np.ndarray] = deque()

        base_idx = 0        # absolute frame index of frames_buf[0]
        display_idx = 0     # absolute frame index being displayed
        processed_total = 0 # absolute count of frames processed by YOLO
        cap_done = False

        window_preds: Dict[int, Tuple[int, float, Optional[float]]] = {}
        next_win_start = 0

        t_pose0 = time.time()

        def process_next_frame() -> bool:
            nonlocal processed_total, cap_done

            ok, frame = cap.read()
            if not ok:
                cap_done = True
                return False

            xy, cf = pose_on_frame(
                pose_model=pose_model,
                frame_bgr=frame,
                imgsz=int(args.imgsz),
                yolo_conf=float(args.yolo_conf),
                device=device,
                max_people=int(args.max_people),
                use_half=use_half,
            )

            frames_buf.append(frame)
            xy_buf.append(xy)
            cf_buf.append(cf)
            processed_total += 1

            if processed_total % 200 == 0:
                dt = time.time() - t_pose0
                if frame_count > 0:
                    pct = 100.0 * float(processed_total) / float(frame_count)
                    print(f"[pose] {processed_total}/{frame_count} ({pct:.1f}%) | {dt:.1f}s")
                else:
                    print(f"[pose] {processed_total} frames | {dt:.1f}s")

            return True

        def compute_window_pred(start: int) -> Tuple[int, float, Optional[float]]:
            if start in window_preds:
                return window_preds[start]
            if start < base_idx:
                raise RuntimeError(f"Cannot compute window {start}: frames already dropped (base_idx={base_idx}).")
            if start >= processed_total:
                raise RuntimeError(f"Cannot compute window {start}: frame not processed yet (processed_total={processed_total}).")

            off = int(start - base_idx)
            avail = int(processed_total - start)
            L = int(min(int(T_final), max(0, avail)))
            if L <= 0:
                raise RuntimeError(f"Window start {start} has no available frames (processed_total={processed_total}).")

            xy_seq = np.stack([xy_buf[off + i] for i in range(L)], axis=0)
            conf_seq = np.stack([cf_buf[off + i] for i in range(L)], axis=0)

            window_feat = make_window_features(
                xy_seq=xy_seq,
                conf_seq=conf_seq,
                T=int(T_final),
                use_conf=use_conf,
                normalize=normalize,
                normalize_mode=normalize_mode,
                add_vel=add_vel,
                add_acc=add_acc,
                add_global=add_global,
                add_mask=add_mask,
                conf_thres=conf_thres,
                max_interp_gap=max_interp_gap,
                missing_mode=missing_mode,
                interp_mode=interp_mode,
                interp_group=int(interp_group),
                rp_center_mode=rp_center_mode,
                rp_img_w=rp_img_w,
                rp_img_h=rp_img_h,
                min_valid_frac=min_valid_frac,
            )
            if is_rf:
                pred, pconf, p_fall = infer_one_window_rf(
                    clf=rf_model,
                    window_feat=window_feat,
                    feature_mode=rf_feature_mode,
                    num_classes=display_num_classes,
                    expected_feature_dim=int(rf_feature_dim) if rf_feature_dim is not None else None,
                )
            else:
                pred, pconf, p_fall = infer_one_window(
                    model=model,
                    window_feat=window_feat,
                    device=device,
                    use_half=use_half,
                    merge_fall_11_to_7=merge_fall_11_to_7,
                )
            window_preds[start] = (pred, pconf, p_fall)
            return window_preds[start]

        def compute_ready_windows() -> None:
            nonlocal next_win_start

            while True:
                if next_win_start in window_preds:
                    next_win_start += int(stride_final)
                    continue

                if cap_done:
                    if next_win_start >= processed_total:
                        break
                    compute_window_pred(int(next_win_start))
                    next_win_start += int(stride_final)
                    continue

                if processed_total >= int(next_win_start) + int(T_final):
                    compute_window_pred(int(next_win_start))
                    next_win_start += int(stride_final)
                    continue

                break

        # Warm up: read enough frames to make the FIRST window prediction, then start display.
        while processed_total < int(T_final) and not cap_done:
            process_next_frame()
        if processed_total <= 0:
            raise RuntimeError("Video had 0 frames.")

        # For paper_rp normalisation, default image dims to the video frame size.
        if str(normalize_mode).lower().strip() == "paper_rp" and frames_buf:
            h_img, w_img = frames_buf[0].shape[:2]
            if rp_img_w is None:
                rp_img_w = int(w_img)
            if rp_img_h is None:
                rp_img_h = int(h_img)

        compute_window_pred(0)
        next_win_start = int(stride_final)

        if save_path is not None:
            h0, w0 = frames_buf[0].shape[:2]
            writer = open_video_writer(save_path=save_path, fps=float(src_fps), frame_size=(w0, h0))

        # Main loop: keep a small lookahead so each next-window prediction is ready in time.
        window_name = "inference_on_video"
        fps_ema: Optional[float] = None
        ema_alpha = 0.1
        t_prev = time.perf_counter()
        while True:
            if not frames_buf and cap_done:
                break

            # Keep a lead of ~T frames (plus 1) so we can predict the next window before it is displayed.
            target_processed = int(display_idx) + int(T_final) + 1
            while not cap_done and processed_total < target_processed:
                process_next_frame()

            compute_ready_windows()

            if not frames_buf:
                continue

            win_start = (int(display_idx) // int(stride_final)) * int(stride_final)
            if win_start not in window_preds:
                # If we're behind (slow device), wait until we can compute it, then continue.
                while not cap_done and processed_total < int(win_start) + int(T_final):
                    process_next_frame()
                    compute_ready_windows()
                if win_start not in window_preds:
                    compute_window_pred(int(win_start))

            pred, pconf, p_fall = window_preds.get(int(win_start), (-1, 0.0, None))
            label = class_names[pred] if 0 <= int(pred) < len(class_names) else "..."

            frame = frames_buf[0].copy()
            xy = xy_buf[0]
            cf = cf_buf[0]

            frame = draw_pose(frame, xy, cf, conf_thres=conf_thres)

            frame_info = f"frame {int(display_idx) + 1}"
            if frame_count > 0:
                frame_info += f"/{frame_count}"

            t_now = time.perf_counter()
            dt = max(1e-6, float(t_now - t_prev))
            inst_fps = 1.0 / dt
            fps_ema = inst_fps if fps_ema is None else (1.0 - ema_alpha) * float(fps_ema) + ema_alpha * inst_fps
            t_prev = t_now

            win_id = int(win_start) // max(1, int(stride_final))
            hud = [
                frame_info,
                f"fps: {float(fps_ema):.1f} (target {float(fps_play):.1f})",
                f"window {win_id} (start={win_start})",
                f"pred: {label} ({float(pconf):.2f})" if int(pred) >= 0 else "pred: ...",
                f"T={int(T_final)} stride={int(stride_final)}",
            ]
            if p_fall is not None:
                hud.append(f"fall_prob: {float(p_fall):.2f}")

            frame = draw_hud(frame, hud)

            if writer is not None:
                frame_h, frame_w = frame.shape[:2]
                frame_to_write = frame
                if frame_h != h0 or frame_w != w0:
                    frame_to_write = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_LINEAR)
                writer.write(frame_to_write)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break

            frames_buf.popleft()
            xy_buf.popleft()
            cf_buf.popleft()
            base_idx += 1
            display_idx += 1

        return 0
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
