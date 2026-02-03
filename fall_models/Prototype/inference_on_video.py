#!/usr/bin/env python3
"""
MP4 -> YOLOv11 pose (yolo11l-pose.pt) -> temporal model inference -> popup display.

Preprocessing mirrors training:
  - dataset.py: fill/interp missing joints, optional normalize/vel/acc/global + mask channel
  - models/train_models.py: temporal model architectures + checkpoint metadata

Usage:
  python inference_on_video.py --video path\\to\\clip.mp4 --model path\\to\\tcn_best.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

import dataset as ds

from models.tcn.simple_tcn import TCNBaseline
from models.lstm.simple_lstm import LSTMBaseline
from models.gru.simple_gru import GRUBaseline
from models.gcn.simple_gcn import GCNBaseline
from models.mlp.simple_mlp import MLPBaseline
from models.stgcn.simple_stgcn import STGCNBaseline

try:
    from models.cnnlstm.cnn_lstm_two_head import CNNLSTMTwoHead
except Exception:
    CNNLSTMTwoHead = None


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

KNOWN_ARCHES = ["tcn", "lstm", "gru", "gcn", "mlp", "stgcn", "cnnlstm"]

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
        if any(arch in tok for tok in tokens):
            return arch
    return None


def resolve_ckpt_and_arch(model_arg: str, arch_arg: Optional[str]) -> Tuple[Path, str]:
    """
    --model can be:
      - a checkpoint file (*.pt)
      - a model folder containing checkpoints (picks newest *best*.pt)
      - a model python file under models/<arch>/...py (picks newest *best*.pt under that folder)
    """
    p = Path(model_arg).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"--model not found: {p}")

    arch = (arch_arg or "").lower().strip() or infer_arch_from_path(p)

    if p.is_file():
        if p.suffix.lower() in {".pt", ".pth", ".bin"}:
            if not arch:
                arch = infer_arch_from_path(p)
            if not arch:
                raise ValueError("Could not infer --arch from checkpoint path. Pass --arch explicitly.")
            return p, arch

        if p.suffix.lower() == ".py":
            if not arch:
                raise ValueError("Could not infer --arch from model .py path. Pass --arch explicitly.")
            model_dir = p.parent
            ckpts = sorted(model_dir.glob("**/*best*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not ckpts:
                ckpts = sorted(model_dir.glob("**/*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found under: {model_dir}")
            return ckpts[0], arch

        raise ValueError(f"Unsupported --model file type: {p.suffix}")

    ckpts = sorted(p.glob("**/*best*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not ckpts:
        ckpts = sorted(p.glob("**/*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint *.pt files found under: {p}")

    ckpt = ckpts[0]
    if not arch:
        arch = infer_arch_from_path(ckpt)
    if not arch:
        raise ValueError("Could not infer --arch from checkpoint path. Pass --arch explicitly.")
    return ckpt, arch


def load_checkpoint(ckpt_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
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

    return model.to(device)


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


def extract_pose_video(
    video_path: Path,
    pose_model: YOLO,
    imgsz: int,
    yolo_conf: float,
    device: str,
    max_people: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-3:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    prealloc = frame_count > 0 and frame_count < 10_000_000

    if prealloc:
        xy = np.full((frame_count, K, 2), np.nan, dtype=np.float32)
        cf = np.full((frame_count, K), np.nan, dtype=np.float32)
    else:
        xy_list: List[np.ndarray] = []
        cf_list: List[np.ndarray] = []

    def pick_top_person(r) -> Tuple[np.ndarray, np.ndarray]:
        if r.keypoints is None or r.keypoints.xy is None:
            return np.zeros((K, 2), np.float32), np.zeros((K,), np.float32)
        xy_all = r.keypoints.xy.detach().cpu().numpy()  # (P,K,2)
        if xy_all.shape[0] == 0:
            return np.zeros((K, 2), np.float32), np.zeros((K,), np.float32)
        if xy_all.shape[1] != K:
            raise ValueError(f"Expected {K} keypoints, got {xy_all.shape[1]}")

        if r.boxes is not None and r.boxes.conf is not None:
            pc = r.boxes.conf.detach().cpu().numpy()
            order = np.argsort(-pc)
        elif r.keypoints.conf is not None:
            kc_all = r.keypoints.conf.detach().cpu().numpy()
            order = np.argsort(-np.nanmean(kc_all, axis=1))
        else:
            order = np.arange(xy_all.shape[0])

        idx = int(order[0])
        xy0 = xy_all[idx].astype(np.float32, copy=False)
        if r.keypoints.conf is not None:
            kc = r.keypoints.conf.detach().cpu().numpy()[idx].astype(np.float32, copy=False)
        else:
            kc = np.ones((K,), dtype=np.float32)
        return xy0, kc

    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        r = pose_model.predict(
            frame,
            imgsz=int(imgsz),
            conf=float(yolo_conf),
            device=device,
            verbose=False,
            max_det=max(1, int(max_people)),
        )[0]
        xy0, kc0 = pick_top_person(r)

        if prealloc:
            xy[i] = xy0
            cf[i] = kc0
        else:
            xy_list.append(xy0)
            cf_list.append(kc0)

        i += 1
        if i % 100 == 0:
            dt = time.time() - t0
            if prealloc:
                pct = 100.0 * i / max(1, frame_count)
                print(f"[pose] {i}/{frame_count} ({pct:.1f}%) | {dt:.1f}s")
            else:
                print(f"[pose] {i} frames | {dt:.1f}s")

    cap.release()

    if prealloc:
        xy = xy[:i]
        cf = cf[:i]
    else:
        if not xy_list:
            raise RuntimeError("No frames read.")
        xy = np.stack(xy_list, axis=0)
        cf = np.stack(cf_list, axis=0)

    if xy.shape[0] == 0:
        raise RuntimeError("Video had 0 frames.")

    print(f"[pose] Done. frames={xy.shape[0]} time={time.time() - t0:.1f}s")
    return xy, cf, fps


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


def make_windows(
    xy_raw: np.ndarray,
    conf_raw: np.ndarray,
    T: int,
    stride: int,
    use_conf: bool,
    normalize: bool,
    add_vel: bool,
    add_acc: bool,
    add_global: bool,
    add_mask: bool,
    conf_thres: float,
    max_interp_gap: int,
    min_valid_frac: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      Xw_flat: (W,T,F)
      win_starts: (W,)
      xy_draw, conf_draw: filled pose for drawing (pixel coords)
    """
    xy_filled, conf_filled = ds._fill_and_mask_kpts(
        xy_raw.astype(np.float32, copy=False),
        conf_raw.astype(np.float32, copy=False),
        conf_thres=float(conf_thres),
        max_interp_gap=int(max_interp_gap),
    )
    xy_used = ds._normalize_xy(xy_filled, conf_filled) if normalize else xy_filled.astype(np.float32, copy=False)

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
        g = ds._global_features(xy_used, conf_filled)  # (N,4)
        parts.append(np.repeat(g[:, None, :], repeats=K, axis=1))

    Xf = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)  # (N,K,C)
    N = int(Xf.shape[0])

    frac_valid = (conf_filled > float(conf_thres)).mean(axis=1)
    frame_valid = frac_valid >= float(min_valid_frac)

    layout = feature_layout(use_conf=use_conf, add_vel=add_vel, add_acc=add_acc)
    stride = max(1, int(stride))
    starts = list(range(0, max(1, N), stride))

    wins: List[np.ndarray] = []
    win_starts: List[int] = []

    for s in starts:
        seq = Xf[s : s + int(T)].copy()  # (L,K,C)
        valid = frame_valid[s : s + int(T)].copy()  # (L,)
        L = int(seq.shape[0])

        if L < int(T):
            pad = (
                np.repeat(seq[-1:, :, :], repeats=(int(T) - L), axis=0)
                if L > 0
                else np.zeros((int(T), K, Xf.shape[2]), np.float32)
            )
            if layout["conf_idx"] is not None:
                pad[:, :, int(layout["conf_idx"])] = 0.0
            if layout["vel_slice"] is not None:
                pad[:, :, layout["vel_slice"]] = 0.0
            if layout["acc_slice"] is not None:
                pad[:, :, layout["acc_slice"]] = 0.0
            seq = np.concatenate([seq, pad], axis=0)
            valid = np.concatenate([valid, np.zeros((int(T) - L,), dtype=bool)], axis=0)

        seq[~valid] = 0.0
        if add_mask:
            m = np.repeat(valid.astype(np.float32)[:, None, None], repeats=K, axis=1)
            seq = np.concatenate([seq, m], axis=-1)

        wins.append(seq)
        win_starts.append(int(s))

        if s + stride >= N and s >= N - 1:
            break

    Xw = np.stack(wins, axis=0)  # (W,T,K,C')
    Xw_flat = Xw.reshape(int(Xw.shape[0]), int(T), int(Xw.shape[2]) * int(Xw.shape[3])).astype(np.float32, copy=False)
    return Xw_flat, np.array(win_starts, dtype=np.int64), xy_filled, conf_filled


@torch.no_grad()
def infer_windows(
    model: nn.Module,
    Xw: np.ndarray,
    device: str,
    batch_size: int,
    use_half: bool,
    merge_fall_11_to_7: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    W = int(Xw.shape[0])
    preds = np.empty((W,), dtype=np.int64)
    confs = np.empty((W,), dtype=np.float32)
    fall_probs: Optional[np.ndarray] = None

    for s in range(0, W, int(batch_size)):
        e = min(W, s + int(batch_size))
        xb = torch.from_numpy(Xw[s:e]).to(device)
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
            prob = torch.cat([prob[:, :5].sum(dim=1, keepdim=True), prob[:, 5:]], dim=1)  # (B,7)

        pconf, pred = torch.max(prob, dim=-1)
        preds[s:e] = pred.detach().cpu().numpy()
        confs[s:e] = pconf.detach().cpu().numpy()

        if fall_logit is not None:
            if fall_probs is None:
                fall_probs = np.zeros((W,), dtype=np.float32)
            fall_probs[s:e] = torch.sigmoid(fall_logit.view(-1)).detach().cpu().numpy()

    return preds, confs, fall_probs


def assign_to_frames(
    n_frames: int,
    win_starts: np.ndarray,
    win_preds: np.ndarray,
    win_confs: np.ndarray,
    win_fall_probs: Optional[np.ndarray],
    T: int,
    align: str,
    backfill: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    pred_f = np.full((n_frames,), -1, dtype=np.int64)
    conf_f = np.zeros((n_frames,), dtype=np.float32)
    fall_f = np.zeros((n_frames,), dtype=np.float32) if win_fall_probs is not None else None

    for i, s in enumerate(win_starts.tolist()):
        idx = min(int(s) + (int(T) - 1 if align == "end" else int(T) // 2), n_frames - 1)
        pred_f[idx] = int(win_preds[i])
        conf_f[idx] = float(win_confs[i])
        if fall_f is not None and win_fall_probs is not None:
            fall_f[idx] = float(win_fall_probs[i])

    # forward-fill for display
    last_p, last_c, last_f = -1, 0.0, 0.0
    for i in range(n_frames):
        if pred_f[i] >= 0:
            last_p, last_c = int(pred_f[i]), float(conf_f[i])
            if fall_f is not None:
                last_f = float(fall_f[i])
        else:
            pred_f[i], conf_f[i] = last_p, last_c
            if fall_f is not None:
                fall_f[i] = last_f

    if backfill and bool(np.any(pred_f >= 0)):
        first = int(np.argmax(pred_f >= 0))
        if first > 0:
            pred_f[:first] = pred_f[first]
            conf_f[:first] = conf_f[first]
            if fall_f is not None:
                fall_f[:first] = fall_f[first]

    return pred_f, conf_f, fall_f


def main() -> int:
    ap = argparse.ArgumentParser(description="Infer a temporal pose model on an MP4 with YOLO pose overlay.")
    ap.add_argument("--video", type=str, required=True, help="Path to input .mp4")
    ap.add_argument("--model", type=str, required=True, help="Checkpoint *.pt OR model folder OR model .py")
    ap.add_argument("--arch", type=str, default=None, choices=KNOWN_ARCHES, help="Override model architecture if needed")
    ap.add_argument("--yolo-weights", type=str, default="yolo11l-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--max-people", type=int, default=1)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--half", type=int, default=0, help="Use FP16 on CUDA (0/1)")
    ap.add_argument("--T", type=int, default=0, help="0 => use ckpt T_used/T, else override")
    ap.add_argument("--stride", type=int, default=0, help="0 => use ckpt stride, else override")
    ap.add_argument("--pred-align", type=str, default="auto", choices=["auto", "center", "end"])
    ap.add_argument("--labels-file", type=str, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--display-fps", type=float, default=0.0, help="0 => source fps")
    ap.add_argument("--backfill", type=int, default=1, help="Backfill early frames with first prediction (0/1)")
    args = ap.parse_args()

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"--video not found: {video_path}")

    device = pick_device(args.device)
    use_half = bool(int(args.half)) and device.startswith("cuda")

    ckpt_path, arch = resolve_ckpt_and_arch(args.model, args.arch)
    print(f"[model] arch={arch} ckpt={ckpt_path.as_posix()}")

    state, meta = load_checkpoint(ckpt_path)
    state = clean_state_dict(state)

    # Preproc config (prefer checkpoint meta)
    T_final = int(args.T) if int(args.T) > 0 else int(meta.get("T", meta.get("T_used", 64)) or 64)
    stride_final = int(args.stride) if int(args.stride) > 0 else int(meta.get("stride", 18) or 18)

    use_conf = bool(meta.get("use_conf", True))
    normalize = bool(meta.get("normalize", True))
    add_vel = bool(meta.get("add_vel", True))
    add_acc = bool(meta.get("add_acc", True))
    add_global = bool(meta.get("add_global", True))
    add_mask = bool(meta.get("add_mask_channel", True))
    conf_thres = float(meta.get("conf_thres", 0.2))
    max_interp_gap = int(meta.get("max_interp_gap", 5))
    min_valid_frac = float(meta.get("min_valid_frac", 0.3))

    label_mode = str(meta.get("label_mode", "center"))
    if args.pred_align == "auto":
        pred_align = "center" if label_mode in {"center", "hybrid_center_fallpct"} else "end"
    else:
        pred_align = str(args.pred_align)

    num_classes = int(meta.get("num_classes", 0) or 0)
    in_features_meta = int(meta.get("in_features", 0) or 0)
    if num_classes <= 0:
        raise ValueError("Checkpoint missing num_classes. Use a checkpoint from models/train_models.py.")

    merge_fall_11_to_7 = int(num_classes) == 11
    display_num_classes = 7 if merge_fall_11_to_7 else int(num_classes)
    class_names = load_class_names(num_classes=display_num_classes, meta=meta, labels_file=args.labels_file)

    pose_model = YOLO(str(Path(args.yolo_weights).expanduser()))
    xy_raw, conf_raw, src_fps = extract_pose_video(
        video_path=video_path,
        pose_model=pose_model,
        imgsz=int(args.imgsz),
        yolo_conf=float(args.yolo_conf),
        device=device,
        max_people=int(args.max_people),
    )
    n_frames = int(xy_raw.shape[0])

    Xw, win_starts, xy_draw, conf_draw = make_windows(
        xy_raw=xy_raw,
        conf_raw=conf_raw,
        T=int(T_final),
        stride=int(stride_final),
        use_conf=use_conf,
        normalize=normalize,
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        add_mask=add_mask,
        conf_thres=conf_thres,
        max_interp_gap=max_interp_gap,
        min_valid_frac=min_valid_frac,
    )
    in_features = int(Xw.shape[-1])
    if in_features_meta > 0 and in_features != in_features_meta:
        raise ValueError(f"Feature mismatch: built in_features={in_features}, ckpt expects {in_features_meta}")

    node_features_meta = meta.get("node_features", None)
    if node_features_meta is None:
        nf = int(in_features // K)
        node_features_meta = nf if nf * K == in_features else None

    model = build_temporal_model(
        arch=arch,
        in_features=in_features,
        num_classes=num_classes,
        device=device,
        T_used=int(T_final),
        node_features=int(node_features_meta) if node_features_meta is not None else None,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[WARN] missing keys:", missing[:8], "..." if len(missing) > 8 else "")
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:8], "..." if len(unexpected) > 8 else "")

    t0 = time.time()
    win_preds, win_confs, win_fall = infer_windows(
        model=model,
        Xw=Xw,
        device=device,
        batch_size=int(args.batch_size),
        use_half=use_half,
        merge_fall_11_to_7=merge_fall_11_to_7,
    )
    print(f"[infer] windows={len(win_preds)} time={time.time() - t0:.2f}s")

    pred_f, conf_f, fall_f = assign_to_frames(
        n_frames=n_frames,
        win_starts=win_starts,
        win_preds=win_preds,
        win_confs=win_confs,
        win_fall_probs=win_fall,
        T=int(T_final),
        align=pred_align,
        backfill=bool(int(args.backfill)),
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for display: {video_path}")

    fps_play = float(args.display_fps) if float(args.display_fps) > 1e-3 else float(src_fps)
    delay_ms = max(1, int(round(1000.0 / max(1e-6, fps_play))))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or idx >= n_frames:
            break

        p = int(pred_f[idx])
        pconf = float(conf_f[idx])
        label = class_names[p] if 0 <= p < len(class_names) else "..."

        hud = [
            f"frame {idx + 1}/{n_frames}",
            f"pred: {label} ({pconf:.2f})" if p >= 0 else "pred: ...",
            f"T={T_final} stride={stride_final} align={pred_align}",
        ]
        if fall_f is not None:
            hud.append(f"fall_prob: {float(fall_f[idx]):.2f}")

        frame = draw_pose(frame, xy_draw[idx], conf_draw[idx], conf_thres=conf_thres)
        frame = draw_hud(frame, hud)

        cv2.imshow("inference_on_video", frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord("q"), 27):
            break

        idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
