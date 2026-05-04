#!/usr/bin/env python3
"""
Offline video -> YOLOv11 pose -> MotionBERT ActionNet inference.

This script:
1) Runs YOLOv11 pose on a video
2) Builds a MotionBERT action .pkl (same schema as prepare_motionbert_dataset.py)
3) Loads a MotionBERT ActionNet checkpoint (*.bin)
4) Runs per-window predictions and saves a CSV

Display mode (--display)
- Runs YOLO pose + MotionBERT inference while the video is being displayed (similar to inference_on_video.py).
- If processing is slower than the source FPS, playback will automatically slow down (no frame skipping).
- Shows the effective FPS in an on-screen HUD.

Notes
- Preprocessing mirrors MotionBERT training (models/MotionBERT/lib/data/dataset_action.py):
  make_cam -> human_tracking -> coco2h36m -> concat conf -> resample -> crop_scale
- You can pass paths relative to:
    - your current working directory
    - this repo root (fall_models/Prototype)
    - MotionBERT root (fall_models/Prototype/models/MotionBERT)
"""

from __future__ import annotations

import argparse
import base64
import csv
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from inference.helpers.keypoint_runtime import KeypointRuntime

# -----------------------------------------------------------------------------
# MotionBERT imports: add MotionBERT root (contains `lib/`) to sys.path
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]  # fall_models/Prototype

def _find_motionbert_root(start_file: Path) -> Tuple[Path, Path]:
    initial_repo_root = start_file.parents[1]
    candidates = [
        initial_repo_root / "models" / "classification" / "MotionBERT",
        initial_repo_root / "models" / "MotionBERT",
    ]
    for candidate in candidates:
        if candidate.exists():
            return initial_repo_root, candidate

    for parent in start_file.parents:
        for suffix in (
            Path("models") / "classification" / "MotionBERT",
            Path("models") / "MotionBERT",
        ):
            candidate = parent / suffix
            if candidate.exists():
                return parent, candidate

    raise FileNotFoundError("MotionBERT root not found under expected models directories.")


_REPO_ROOT, _MB_ROOT = _find_motionbert_root(_THIS_FILE)

mb_root_str = str(_MB_ROOT)
if mb_root_str not in sys.path:
    sys.path.insert(0, mb_root_str)

# Match MotionBERT training/eval scripts (train_action_weighted_balanced.py) which import via top-level `lib.*`.
from lib.utils.tools import get_config  # noqa: E402
from lib.utils.learning import load_backbone  # noqa: E402
from lib.model.model_action import ActionNet  # noqa: E402
from lib.data.dataset_action import make_cam, coco2h36m, human_tracking  # noqa: E402
from lib.utils.utils_data import crop_scale, resample  # noqa: E402

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
CLASS_NAMES_DEFAULT = [
    "Falling forward using hands",  # 0
    "Falling forward using knees",  # 1
    "Falling backwards",  # 2
    "Falling sideward",  # 3
    "Falling sitting in an empty chair",  # 4
    "Walking",  # 5
    "Standing",  # 6
    "Sitting",  # 7
    "Picking up an object",  # 8
    "Jumping",  # 9
    "Laying",  # 10
]

CLASS_NAMES_MERGED_DEFAULT = [
    "Fall",  # 0 (all fall subclasses merged)
    "Walking",  # 1
    "Standing",  # 2
    "Sitting",  # 3
    "Picking up an object",  # 4
    "Jumping",  # 5
    "Laying",  # 6
]

FALL_CLASS_IDS_DEFAULT = [0, 1, 2, 3, 4]
DEFAULT_CONFIG_CANDIDATES = [
    "configs/action/MB_ft_UPFall_xsub_LITE.yaml",
    "configs/action/MB_ft_UPFall_xsub.yaml",
]


def pick_default_config_relpath() -> str:
    for rel_path in DEFAULT_CONFIG_CANDIDATES:
        if (_MB_ROOT / rel_path).exists():
            return rel_path
    return DEFAULT_CONFIG_CANDIDATES[0]

# COCO keypoint order for Ultralytics pose models (17 joints)
K = 17

SKELETON = [
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
    (5, 6),  # shoulders
    (11, 12),  # hips
    (5, 11),
    (6, 12),  # torso sides
]


def pick_device(device: Optional[str]) -> str:
    if not device:
        return "cuda" if torch.cuda.is_available() else "cpu"
    d = device.lower().strip()
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def ceil_div_pos(a: int, b: int) -> int:
    a_i = int(a)
    b_i = int(b)
    if a_i <= 0:
        raise ValueError(f"Expected positive integer, got {a_i}.")
    if b_i <= 0:
        raise ValueError(f"Expected positive divisor, got {b_i}.")
    return (a_i + b_i - 1) // b_i


def resolve_path(path: str, *, desc: str) -> Path:
    """
    Resolve a user-provided path in a forgiving way:
      1) as given (relative to CWD)
      2) relative to this repo root (where this file lives)
      3) relative to MotionBERT root (repo_root/models/MotionBERT)
    """
    p = Path(path).expanduser()
    if p.exists():
        return p

    repo_rel = (_REPO_ROOT / path).expanduser()
    if repo_rel.exists():
        return repo_rel

    mb_rel = (_MB_ROOT / path).expanduser()
    if mb_rel.exists():
        return mb_rel

    raise FileNotFoundError(f"{desc} not found: {path}")


def resolve_checkpoint_path(ckpt: str) -> Path:
    """
    Accept either:
      - a checkpoint file (e.g. best_epoch.bin)
      - a checkpoint directory containing best_epoch.bin
    """
    p = resolve_path(ckpt, desc="Checkpoint")
    if p.is_file():
        return p

    best = p / "best_epoch.bin"
    if best.exists():
        return best
    latest = p / "latest_epoch.bin"
    if latest.exists():
        return latest

    bins = sorted(p.glob("**/*.bin"), key=lambda x: x.stat().st_mtime, reverse=True)
    if bins:
        return bins[0]

    raise FileNotFoundError(f"No *.bin checkpoints found under: {p.as_posix()}")


def infer_fall_indices(class_names: List[str]) -> List[int]:
    fall_idx = []
    for i, n in enumerate(class_names):
        s = n.lower()
        if s.startswith("fall") or "falling" in s:
            fall_idx.append(i)
    return fall_idx


def interpolate_missing_joints_inplace(
    kxy: np.ndarray,
    ksc: np.ndarray,
    missing_conf_thres: float = 0.0,
) -> None:
    """
    Interpolate missing joints over time within ONE window.
    Missing definition (per joint, per frame): score <= threshold OR non-finite coords.
    """
    T = kxy.shape[0]
    V = kxy.shape[1]
    if V != 17 or kxy.shape[2] != 2:
        raise ValueError(f"Expected kxy (T,17,2), got {kxy.shape}")
    if ksc.shape != (T, 17):
        raise ValueError(f"Expected ksc (T,17), got {ksc.shape}")

    t_idx = np.arange(T, dtype=np.float64)

    for j in range(V):
        finite_joint = np.isfinite(kxy[:, j, 0]) & np.isfinite(kxy[:, j, 1])
        valid_joint = (ksc[:, j] > missing_conf_thres) & finite_joint
        n_valid_joint = int(np.sum(valid_joint))

        if n_valid_joint == 0:
            kxy[:, j, :] = 0.0
            ksc[:, j] = 0.0
            continue

        for a in range(2):
            valid = (ksc[:, j] > missing_conf_thres) & np.isfinite(kxy[:, j, a])
            idx = np.where(valid)[0]
            if idx.size >= 2:
                vals = kxy[idx, j, a].astype(np.float64)
                interp_all = np.interp(t_idx, idx.astype(np.float64), vals)
                invalid = ~valid
                kxy[invalid, j, a] = interp_all[invalid].astype(np.float32)
            elif idx.size == 1:
                kxy[:, j, a] = float(kxy[idx[0], j, a])
            else:
                kxy[:, j, a] = 0.0


def draw_hud(
    frame,
    lines,
    org=(10, 10),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    thickness=2,
    pad=8,
    line_gap=6,
    bg_color=(0, 0, 0),
    bg_alpha=0.6,
    text_color=(255, 255, 255),
):
    if not lines:
        return frame

    x0, y0 = org
    sizes = [cv2.getTextSize(str(s), font, font_scale, thickness)[0] for s in lines]
    max_w = max(w for w, h in sizes)
    total_h = sum(h for w, h in sizes) + line_gap * (len(lines) - 1)

    box_w = max_w + 2 * pad
    box_h = total_h + 2 * pad

    h_img, w_img = frame.shape[:2]
    x1 = min(w_img - 1, x0 + box_w)
    y1 = min(h_img - 1, y0 + box_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, -1)
    frame = cv2.addWeighted(overlay, bg_alpha, frame, 1.0 - bg_alpha, 0)

    y = y0 + pad
    for (w, h), s in zip(sizes, lines):
        y += h
        cv2.putText(frame, str(s), (x0 + pad, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += line_gap

    return frame


def draw_pose(frame, xy: np.ndarray, conf: np.ndarray, conf_thres: float = 0.2, draw_skeleton: bool = True):
    for i in range(K):
        if conf[i] > conf_thres:
            x, y = int(xy[i, 0]), int(xy[i, 1])
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    if draw_skeleton:
        for a, b in SKELETON:
            if conf[a] > conf_thres and conf[b] > conf_thres:
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


def _clean_state_dict_for_model(state: dict, model: nn.Module) -> dict:
    """
    Flexibly handle DataParallel 'module.' prefixes.
    """
    if not isinstance(state, dict):
        return state
    state_keys = list(state.keys())
    has_module_prefix = any(k.startswith("module.") for k in state_keys)
    model_is_dp = isinstance(model, nn.DataParallel)

    if has_module_prefix and not model_is_dp:
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    if (not has_module_prefix) and model_is_dp:
        return {("module." + k): v for k, v in state.items()}
    return state


def _infer_action_classes_from_state_dict(state: dict) -> Optional[int]:
    if not isinstance(state, dict):
        return None
    candidate_keys = (
        "head.fc2.weight",
        "module.head.fc2.weight",
        "head.fc2.bias",
        "module.head.fc2.bias",
    )
    for key in candidate_keys:
        value = state.get(key)
        if value is None:
            continue
        shape = tuple(getattr(value, "shape", ()))
        if len(shape) >= 1 and int(shape[0]) > 0:
            return int(shape[0])
    return None


def build_windows(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    img_shape: Tuple[int, int],
    win_len: int,
    win_step: int,
    pad_tail: bool,
    missing_conf_thres: float,
    drop_empty_windows: bool,
    video_stem: str,
) -> Tuple[List[str], List[dict]]:
    """Return (split_list, annotations) with MotionBERT action pkl schema."""
    annotations: List[dict] = []
    split_list: List[str] = []

    T_total = int(kpts_xy.shape[0])
    if T_total <= 0:
        return split_list, annotations

    for start in range(0, T_total, win_step):
        end = start + win_len
        if end > T_total:
            if not pad_tail:
                break
            pad_n = end - T_total
            if pad_n >= win_len:
                break

        frame_dir = f"{video_stem}_s{start}_len{win_len}"

        raw_kxy = kpts_xy[start : min(end, T_total)].astype(np.float32)
        raw_ksc = kpts_conf[start : min(end, T_total)].astype(np.float32)

        if pad_tail and end > T_total:
            last_xy = raw_kxy[-1:, :, :]
            last_sc = raw_ksc[-1:, :]
            raw_kxy = np.concatenate([raw_kxy, np.repeat(last_xy, pad_n, axis=0)], axis=0)
            raw_ksc = np.concatenate([raw_ksc, np.repeat(last_sc, pad_n, axis=0)], axis=0)

        if raw_kxy.shape[0] != win_len or raw_ksc.shape[0] != win_len:
            continue

        kxy = raw_kxy.copy()
        ksc = raw_ksc.copy()

        nonfinite_xy = ~np.isfinite(kxy)
        nonfinite_sc = ~np.isfinite(ksc)
        if nonfinite_xy.any() or nonfinite_sc.any():
            kxy[nonfinite_xy] = 0.0
            ksc[nonfinite_sc] = 0.0
            nonfinite_joint = nonfinite_xy.any(axis=2) | nonfinite_sc
            ksc[nonfinite_joint] = 0.0

        if (ksc < 0).any() or (ksc > 1).any():
            ksc = np.clip(ksc, 0.0, 1.0)

        interpolate_missing_joints_inplace(kxy, ksc, missing_conf_thres=missing_conf_thres)

        if drop_empty_windows:
            if np.all(ksc <= missing_conf_thres):
                continue
            if (np.ptp(kxy[..., 0]) < 1e-6) and (np.ptp(kxy[..., 1]) < 1e-6):
                continue

        keypoint = kxy[None, ...].astype(np.float32)
        keypoint_score = ksc[None, ...].astype(np.float32)

        annotations.append(
            {
                "frame_dir": frame_dir,
                "total_frames": int(win_len),
                "img_shape": (int(img_shape[0]), int(img_shape[1])),
                "keypoint": keypoint,
                "keypoint_score": keypoint_score,
                "label": 0,
            }
        )
        split_list.append(frame_dir)

        if (not pad_tail) and end >= T_total:
            break

    return split_list, annotations


def build_motion_from_annotation(
    ann: dict,
    clip_len: int,
    scale_range: Optional[List[float]],
) -> np.ndarray:
    """
    Build MotionBERT ActionNet input exactly like MotionBERT's dataset_action.NTURGBD:
      resample -> make_cam -> human_tracking -> coco2h36m -> concat conf -> crop_scale
    """
    keypoint = ann["keypoint"]  # (1, T, 17, 2)
    keypoint_score = ann["keypoint_score"]  # (1, T, 17)
    img_shape = ann["img_shape"]

    resample_id = resample(ori_len=int(ann["total_frames"]), target_len=int(clip_len), randomness=False)
    motion_cam = make_cam(x=keypoint, img_shape=img_shape)
    motion_cam = human_tracking(motion_cam)
    motion_cam = coco2h36m(motion_cam)
    motion_conf = keypoint_score[..., None]
    motion = np.concatenate((motion_cam[:, resample_id], motion_conf[:, resample_id]), axis=-1)

    if motion.shape[0] == 1:
        fake = np.zeros(motion.shape, dtype=motion.dtype)
        motion = np.concatenate((motion, fake), axis=0)

    if scale_range:
        motion = crop_scale(motion, scale_range=scale_range)

    return motion.astype(np.float32)


def load_labels_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    p = resolve_path(path, desc="Labels file")
    names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return names or None


def pad_or_trim(names: List[str], num_classes: int) -> List[str]:
    if len(names) != int(num_classes):
        if len(names) > int(num_classes):
            names = names[: int(num_classes)]
        else:
            for i in range(len(names), int(num_classes)):
                names.append(f"class_{i}")
    return names


def window_start_end_from_frame_dir(frame_dir: str, total_frames: int, frame_step: int = 1) -> Tuple[int, int]:
    start_frame = 0
    try:
        parts = frame_dir.split("_s", 1)
        if len(parts) > 1:
            start_frame = int(parts[1].split("_len", 1)[0])
    except Exception:
        start_frame = 0

    step = max(1, int(frame_step))
    start_raw = int(start_frame) * step
    end_raw = int(start_raw) + (max(1, int(total_frames)) - 1) * step
    return int(start_raw), int(end_raw)


def predict_one_window(
    *,
    model: nn.Module,
    device: str,
    ann: dict,
    frame_dir: str,
    clip_len: int,
    scale_range: Optional[List[float]],
    merge_fall: bool,
    fall_idx: List[int],
    class_names_out: List[str],
    unmerged_len_expected: int,
    merged_len_expected: int,
    frame_step: int = 1,
) -> dict:
    motion = build_motion_from_annotation(ann, clip_len=clip_len, scale_range=scale_range)

    X = torch.from_numpy(motion).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(X)
        probs_t = torch.softmax(out, dim=1).squeeze(0).detach().cpu().numpy()

    if merge_fall:
        if probs_t.shape[0] == merged_len_expected:
            merged_probs = probs_t
            fall_prob = float(merged_probs[0]) if merged_probs.size else 0.0
        elif probs_t.shape[0] == unmerged_len_expected:
            fall_prob = float(np.sum(probs_t[FALL_CLASS_IDS_DEFAULT]))
            nonfall_probs = [probs_t[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
            merged_probs = np.array([fall_prob] + nonfall_probs, dtype=np.float32)
        else:
            merged_probs = probs_t
            fall_prob = float(np.sum(probs_t[fall_idx])) if fall_idx else 0.0
    else:
        merged_probs = probs_t
        fall_prob = float(np.sum(probs_t[fall_idx])) if fall_idx else 0.0

    pred_id = int(np.argmax(merged_probs))
    pred_conf = float(np.max(merged_probs))
    p_fall = float(fall_prob)

    start_frame, end_frame = window_start_end_from_frame_dir(
        frame_dir,
        int(ann["total_frames"]),
        frame_step=int(frame_step),
    )
    pred_name = class_names_out[pred_id] if 0 <= pred_id < len(class_names_out) else str(pred_id)

    return {
        "frame_dir": str(frame_dir),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "pred_id": int(pred_id),
        "pred_name": str(pred_name),
        "pred_conf": float(pred_conf),
        "p_fall": float(p_fall),
    }


def stream_infer_and_display(
    *,
    ckpt_path: Path,
    video_path: Path,
    cfg,
    keypoint_model_path: Path,
    device: str,
    clip_len: int,
    win_step: int,
    frame_step: int,
    clip_len_raw: int,
    win_step_raw: int,
    save_path: Optional[Path],
    args: argparse.Namespace,
) -> int:
    """
    One-pass streaming: read frames -> YOLO pose -> window inference -> imshow.

    Playback targets the source FPS (or --display-fps). If processing can't keep up, the display slows down.
    """

    win_step = max(1, int(win_step))
    frame_step = max(1, int(frame_step))
    missing_conf_thres = float(args.missing_conf_thres)
    drop_empty_windows = not bool(args.keep_empty_windows)
    pad_tail = bool(args.pad_tail)

    # Models
    keypoint_runtime = KeypointRuntime(
        model_path=Path(keypoint_model_path).expanduser(),
        device=device,
        backend=getattr(args, "keypoint_backend", None),
    )
    print(f"[pose] backend={keypoint_runtime.backend} model={Path(keypoint_model_path).expanduser()}")

    model_backbone = load_backbone(cfg)
    model = ActionNet(
        backbone=model_backbone,
        dim_rep=getattr(cfg, "dim_rep", 512),
        num_classes=int(getattr(cfg, "action_classes", 11)),
        dropout_ratio=getattr(cfg, "dropout_ratio", 0.0),
        version=getattr(cfg, "model_version", "class"),
        hidden_dim=getattr(cfg, "hidden_dim", 2048),
        num_joints=getattr(cfg, "num_joints", 17),
    )

    use_dp = device.startswith("cuda") and torch.cuda.device_count() > 1
    if use_dp:
        model = nn.DataParallel(model)
    model = model.to(device)
    print("Model device:", next(model.parameters()).device)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = _clean_state_dict_for_model(state, model)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Labels / inference config
    num_classes = int(getattr(cfg, "action_classes", 11))
    scale_range = getattr(cfg, "scale_range_test", None)

    labels_file_names = load_labels_file(args.labels_file)

    unmerged_len_expected = len(CLASS_NAMES_DEFAULT)
    merged_len_expected = len(CLASS_NAMES_MERGED_DEFAULT)

    merge_fall = (not bool(args.no_merge_fall)) and (num_classes == unmerged_len_expected)

    if labels_file_names is not None:
        if merge_fall and len(labels_file_names) == merged_len_expected:
            class_names_out = list(labels_file_names)
        else:
            base_names = pad_or_trim(list(labels_file_names), num_classes)
            if merge_fall:
                class_names_out = ["Fall"] + [base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
            else:
                class_names_out = base_names
    else:
        if num_classes == merged_len_expected:
            class_names_out = list(CLASS_NAMES_MERGED_DEFAULT)
        else:
            base_names = pad_or_trim(list(CLASS_NAMES_DEFAULT), num_classes)
            if merge_fall:
                class_names_out = ["Fall"] + [base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
            else:
                class_names_out = base_names

    fall_idx = infer_fall_indices(class_names_out)

    predict_kwargs = dict(
        model=model,
        device=device,
        clip_len=int(clip_len),
        scale_range=scale_range,
        merge_fall=merge_fall,
        fall_idx=fall_idx,
        class_names_out=class_names_out,
        unmerged_len_expected=unmerged_len_expected,
        merged_len_expected=merged_len_expected,
        frame_step=frame_step,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_pkl = Path(args.out_pkl)

    def write_header(writer: csv.writer) -> None:
        writer.writerow(
            [
                "frame_dir",
                "start_frame",
                "end_frame",
                "pred_id",
                "pred_name",
                "pred_conf",
                "p_fall",
            ]
        )

    def write_pred_row(writer: csv.writer, pred: dict) -> None:
        writer.writerow(
            [
                pred["frame_dir"],
                pred["start_frame"],
                pred["end_frame"],
                pred["pred_id"],
                pred["pred_name"],
                f"{float(pred['pred_conf']):.6f}",
                f"{float(pred['p_fall']):.6f}",
            ]
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path.as_posix()}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(src_fps) or src_fps <= 1e-3:
        src_fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    user_fps = float(args.display_fps) if args.display_fps is not None else None
    fps_target = float(user_fps) if (user_fps is not None and np.isfinite(user_fps) and user_fps > 0.0) else float(src_fps)
    if not np.isfinite(fps_target) or fps_target <= 1e-3:
        fps_target = 30.0
    frame_period_s = 1.0 / float(fps_target)

    window_name = "MotionBERT Inference"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print(f"[Display] Target FPS={fps_target:.2f} (source={src_fps:.2f}). Press 'q' or Esc to quit.", flush=True)

    video_writer: Optional[cv2.VideoWriter] = None
    out_w: Optional[int] = None
    out_h: Optional[int] = None

    frames_buf: "deque[np.ndarray]" = deque()
    xy_buf: "deque[np.ndarray]" = deque()
    cf_buf: "deque[np.ndarray]" = deque()

    all_xy: List[np.ndarray] = []
    all_cf: List[np.ndarray] = []

    img_shape: Optional[Tuple[int, int]] = None
    processed_total = 0  # raw frames processed
    sampled_total = 0  # sampled frames used for MotionBERT windows
    display_idx = 0
    cap_done = False

    window_preds: dict[int, dict] = {}
    skipped_windows: set[int] = set()
    next_win_start = 0
    last_xy = np.zeros((17, 2), dtype=np.float32)
    last_cf = np.zeros((17,), dtype=np.float32)

    def pose_on_frame(frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        detections = keypoint_runtime.predict(
            frame_bgr=frame_bgr,
            imgsz=int(args.imgsz),
            conf=float(args.conf_thres),
            max_people=1,
            use_half=False,
        )

        kpts_xy = np.zeros((17, 2), dtype=np.float32)
        kpts_conf = np.zeros((17,), dtype=np.float32)

        xy_all = detections.xy
        cf_all = detections.conf
        if xy_all.ndim == 3 and xy_all.shape[0] > 0:
            scores = cf_all.sum(axis=1) if (cf_all.ndim == 2 and cf_all.shape[0] == xy_all.shape[0]) else None
            best = int(np.argmax(scores)) if scores is not None else 0
            kpts_xy = xy_all[best].astype(np.float32)
            if cf_all.ndim == 2 and cf_all.shape[0] == xy_all.shape[0]:
                kpts_conf = cf_all[best].astype(np.float32)
            else:
                kpts_conf = np.ones((17,), dtype=np.float32)

        return kpts_xy, kpts_conf

    def process_next_frame() -> bool:
        nonlocal processed_total, sampled_total, cap_done, img_shape, last_xy, last_cf

        if cap_done:
            return False

        ok, frame = cap.read()
        if not ok:
            cap_done = True
            return False

        if img_shape is None:
            h, w = frame.shape[:2]
            img_shape = (int(h), int(w))

        raw_idx = int(processed_total)
        do_pose = (int(raw_idx) % int(frame_step)) == 0

        if do_pose:
            xy, cf = pose_on_frame(frame)
            all_xy.append(xy)
            all_cf.append(cf)
            sampled_total += 1
            last_xy = xy
            last_cf = cf
        else:
            xy = last_xy
            cf = last_cf

        frames_buf.append(frame)
        xy_buf.append(xy)
        cf_buf.append(cf)
        processed_total += 1

        if args.limit_frames is not None and processed_total >= int(args.limit_frames):
            cap_done = True

        return True

    def make_window_annotation(start: int) -> Optional[Tuple[str, dict]]:
        if img_shape is None:
            return None

        end = int(start) + int(clip_len)
        if end <= int(sampled_total):
            raw_kxy = np.stack(all_xy[start:end], axis=0).astype(np.float32)
            raw_ksc = np.stack(all_cf[start:end], axis=0).astype(np.float32)
        else:
            if not cap_done or (not pad_tail):
                return None
            if start >= int(sampled_total):
                return None
            pad_n = int(end - int(sampled_total))
            if pad_n >= int(clip_len):
                return None
            raw_kxy = np.stack(all_xy[start:sampled_total], axis=0).astype(np.float32)
            raw_ksc = np.stack(all_cf[start:sampled_total], axis=0).astype(np.float32)
            last_xy = raw_kxy[-1:, :, :]
            last_sc = raw_ksc[-1:, :]
            raw_kxy = np.concatenate([raw_kxy, np.repeat(last_xy, pad_n, axis=0)], axis=0)
            raw_ksc = np.concatenate([raw_ksc, np.repeat(last_sc, pad_n, axis=0)], axis=0)

        if raw_kxy.shape != (int(clip_len), 17, 2) or raw_ksc.shape != (int(clip_len), 17):
            return None

        kxy = raw_kxy.copy()
        ksc = raw_ksc.copy()

        nonfinite_xy = ~np.isfinite(kxy)
        nonfinite_sc = ~np.isfinite(ksc)
        if nonfinite_xy.any() or nonfinite_sc.any():
            kxy[nonfinite_xy] = 0.0
            ksc[nonfinite_sc] = 0.0
            nonfinite_joint = nonfinite_xy.any(axis=2) | nonfinite_sc
            ksc[nonfinite_joint] = 0.0

        if (ksc < 0).any() or (ksc > 1).any():
            ksc = np.clip(ksc, 0.0, 1.0)

        interpolate_missing_joints_inplace(kxy, ksc, missing_conf_thres=missing_conf_thres)

        if drop_empty_windows:
            if np.all(ksc <= missing_conf_thres):
                return None
            if (np.ptp(kxy[..., 0]) < 1e-6) and (np.ptp(kxy[..., 1]) < 1e-6):
                return None

        frame_dir = f"{video_path.stem}_s{start}_len{clip_len}"
        ann = {
            "frame_dir": frame_dir,
            "total_frames": int(clip_len),
            "img_shape": (int(img_shape[0]), int(img_shape[1])),
            "keypoint": kxy[None, ...].astype(np.float32),
            "keypoint_score": ksc[None, ...].astype(np.float32),
            "label": 0,
        }
        return frame_dir, ann

    def compute_window_pred(start: int, *, writer: csv.writer) -> Optional[dict]:
        if int(start) in window_preds:
            return window_preds[int(start)]
        if int(start) in skipped_windows:
            return None

        window = make_window_annotation(int(start))
        if window is None:
            # Window is either not ready yet, or it was dropped (empty / too short)
            if cap_done and (not pad_tail) and int(sampled_total) < int(start) + int(clip_len):
                skipped_windows.add(int(start))
            if drop_empty_windows and int(sampled_total) >= int(start) + int(clip_len):
                skipped_windows.add(int(start))
            return None

        frame_dir, ann = window
        pred = predict_one_window(ann=ann, frame_dir=frame_dir, **predict_kwargs)
        window_preds[int(start)] = pred
        write_pred_row(writer, pred)
        return pred

    def compute_ready_windows(*, writer: csv.writer) -> None:
        nonlocal next_win_start

        while True:
            if int(next_win_start) in window_preds or int(next_win_start) in skipped_windows:
                next_win_start = int(next_win_start) + int(win_step)
                continue

            if not cap_done:
                if int(sampled_total) >= int(next_win_start) + int(clip_len):
                    compute_window_pred(int(next_win_start), writer=writer)
                    next_win_start = int(next_win_start) + int(win_step)
                    continue
                break

            # cap_done
            if int(sampled_total) >= int(next_win_start) + int(clip_len):
                compute_window_pred(int(next_win_start), writer=writer)
                next_win_start = int(next_win_start) + int(win_step)
                continue
            if pad_tail and int(next_win_start) < int(sampled_total):
                compute_window_pred(int(next_win_start), writer=writer)
                next_win_start = int(next_win_start) + int(win_step)
                continue
            break

    def get_pred_for_frame(frame_idx: int) -> Optional[dict]:
        if frame_idx < 0:
            return None
        sample_idx = int(frame_idx) // int(frame_step)
        ws = (int(sample_idx) // int(win_step)) * int(win_step)
        pred = window_preds.get(int(ws))
        if pred is not None:
            return pred

        # If the exact window was dropped (empty), fall back to the most recent available window covering frame_idx.
        s = int(ws) - int(win_step)
        while s >= 0:
            p = window_preds.get(int(s))
            if p is not None and int(frame_idx) <= int(p["end_frame"]):
                return p
            s -= int(win_step)
        return None

    try:
        with out_csv.open("w", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            write_header(writer)

            # Warm up: read enough frames to make the FIRST window prediction (if possible), then start display.
            while int(sampled_total) < int(clip_len) and not cap_done:
                process_next_frame()
            if int(processed_total) <= 0:
                raise RuntimeError("Video had 0 frames.")

            compute_window_pred(0, writer=writer)
            next_win_start = int(win_step)

            if save_path is not None and frames_buf:
                out_h, out_w = frames_buf[0].shape[:2]
                video_writer = open_video_writer(
                    save_path=save_path,
                    fps=float(src_fps),
                    frame_size=(int(out_w), int(out_h)),
                )

            fps_ema: Optional[float] = None
            ema_alpha = 0.1
            t_prev = time.perf_counter()
            user_exit = False

            while True:
                if not frames_buf and cap_done:
                    break

                t_frame0 = time.perf_counter()

                display_sample_idx = int(display_idx) // int(frame_step)
                target_sampled = int(display_sample_idx) + int(clip_len) + 1
                while (not cap_done) and int(sampled_total) < int(target_sampled):
                    process_next_frame()

                compute_ready_windows(writer=writer)

                if not frames_buf:
                    continue

                win_start = (int(display_sample_idx) // int(win_step)) * int(win_step)
                if win_start not in window_preds and win_start not in skipped_windows:
                    while (not cap_done) and int(sampled_total) < int(win_start) + int(clip_len):
                        process_next_frame()
                        compute_ready_windows(writer=writer)
                    compute_window_pred(int(win_start), writer=writer)

                pred = get_pred_for_frame(int(display_idx))

                frame = frames_buf[0].copy()
                xy = xy_buf[0]
                cf = cf_buf[0]
                frame = draw_pose(
                    frame,
                    xy,
                    cf,
                    conf_thres=float(args.display_conf_thres),
                    draw_skeleton=True,
                )

                frame_info = f"frame {int(display_idx) + 1}"
                if int(frame_count) > 0:
                    frame_info += f"/{int(frame_count)}"

                win_id = int(win_start) // max(1, int(win_step))
                win_start_raw = int(win_start) * int(frame_step)
                fps_for_hud = float(fps_ema) if fps_ema is not None else float(fps_target)
                hud = [
                    frame_info,
                    f"fps: {float(fps_for_hud):.1f} (target {float(fps_target):.1f})",
                    f"window {win_id} (sample_start={win_start}, raw_start={win_start_raw})",
                ]
                if pred is not None:
                    hud.append(f"pred: {pred['pred_name']} ({float(pred['pred_conf']):.2f})")
                    hud.append(f"fall_prob: {float(pred['p_fall']):.2f}")
                    hud.append(f"win: {int(pred['start_frame'])}-{int(pred['end_frame'])}")
                else:
                    hud.append("pred: ... (warming up)")
                hud.append(
                    f"T={int(clip_len)} stride={int(win_step)} sampled "
                    f"(raw T/stride={int(clip_len_raw)}/{int(win_step_raw)}, k={int(frame_step)})"
                )

                frame = draw_hud(frame, hud)

                if video_writer is not None and out_w is not None and out_h is not None:
                    frame_h, frame_w = frame.shape[:2]
                    frame_to_write = frame
                    if frame_h != int(out_h) or frame_w != int(out_w):
                        frame_to_write = cv2.resize(frame, (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR)
                    video_writer.write(frame_to_write)

                cv2.imshow(window_name, frame)

                elapsed = float(time.perf_counter() - t_frame0)
                wait_s = float(frame_period_s) - elapsed
                wait_ms = max(1, int(round(1000.0 * wait_s))) if wait_s > 0.0 else 1
                key = cv2.waitKey(int(wait_ms)) & 0xFF
                if key in (ord("q"), 27):
                    user_exit = True
                    break

                frames_buf.popleft()
                xy_buf.popleft()
                cf_buf.popleft()
                display_idx += 1

                # Effective display FPS (includes processing + waitKey pacing).
                t_now = time.perf_counter()
                dt = max(1e-6, float(t_now - t_prev))
                inst_fps = 1.0 / dt
                fps_ema = inst_fps if fps_ema is None else (1.0 - ema_alpha) * float(fps_ema) + ema_alpha * inst_fps
                t_prev = t_now

            if (not user_exit) and cap_done:
                while True:
                    before = int(next_win_start)
                    compute_ready_windows(writer=writer)
                    if int(next_win_start) == before:
                        break
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()

    if img_shape is None:
        raise RuntimeError("No frames read from video.")

    if not all_xy or not all_cf:
        raise RuntimeError(
            "No sampled frames were generated. Reduce --k/--frame-step or ensure the video has readable frames."
        )

    # Build and save MotionBERT action pkl from sampled frames only.
    kpts_xy = np.stack(all_xy, axis=0)  # (T_sampled,17,2)
    kpts_conf = np.stack(all_cf, axis=0)  # (T_sampled,17)
    split_list, annotations = build_windows(
        kpts_xy=kpts_xy,
        kpts_conf=kpts_conf,
        img_shape=img_shape,
        win_len=int(clip_len),
        win_step=int(win_step),
        pad_tail=pad_tail,
        missing_conf_thres=missing_conf_thres,
        drop_empty_windows=drop_empty_windows,
        video_stem=video_path.stem,
    )

    dataset = {"split": {"xsub_train": [], "xsub_val": split_list}, "annotations": annotations}
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved pkl: {out_pkl.as_posix()}")
    print(f"Saved predictions: {out_csv.as_posix()}")
    print(f"Windows: {len(split_list)}")
    return 0


def run_inference_stream_packets(
    *,
    video_path: Path,
    classification_model_path: Path,
    keypoint_model_path: Path,
    on_packet: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
    save_path: Optional[Path] = None,
    no_display: bool = True,
    realtime: bool = True,
    display_fps: float = 0.0,
    device: Optional[str] = None,
    keypoint_backend: Optional[str] = None,
    half: int = 0,
    imgsz: int = 640,
    yolo_conf: float = 0.25,
    T: int = 0,
    stride: int = 0,
    frame_step: int = 1,
    config_path: Optional[str] = None,
    labels_file: Optional[str] = None,
    pad_tail: bool = False,
    missing_conf_thres: float = 0.0,
    keep_empty_windows: bool = False,
    display_conf_thres: float = 0.2,
    no_merge_fall: bool = False,
    jpeg_quality: int = 80,
    **_unused_options: Any,
) -> int:
    frame_step = int(frame_step)
    if frame_step <= 0:
        raise ValueError("--frame-step/--k must be >= 1.")

    resolved_video_path = Path(video_path).expanduser()
    if not resolved_video_path.exists():
        raise FileNotFoundError(f"--video not found: {resolved_video_path}")

    resolved_ckpt_path = resolve_checkpoint_path(str(classification_model_path))
    resolved_keypoint_path = resolve_path(str(keypoint_model_path), desc="Keypoint model path")

    config_to_use = str(config_path).strip() if config_path else pick_default_config_relpath()
    resolved_cfg_path = resolve_path(config_to_use, desc="Config")

    run_device = pick_device(device)
    use_half = bool(int(half)) and run_device.startswith("cuda")

    cfg = get_config(str(resolved_cfg_path))

    clip_len_raw = int(T) if int(T) > 0 else int(getattr(cfg, "clip_len", 64))
    win_step_raw = int(stride) if int(stride) > 0 else 16
    if int(clip_len_raw) <= 0:
        raise ValueError(f"Invalid window length: {clip_len_raw}.")
    if int(win_step_raw) <= 0:
        raise ValueError(f"Invalid window stride: {win_step_raw}.")

    clip_len = max(1, int(ceil_div_pos(int(clip_len_raw), int(frame_step))))
    win_step = max(1, int(ceil_div_pos(int(win_step_raw), int(frame_step))))
    if int(frame_step) > 1 and (
        (int(clip_len_raw) % int(frame_step)) != 0 or (int(win_step_raw) % int(frame_step)) != 0
    ):
        print(
            f"[window][WARN] raw clip_len/win_step ({int(clip_len_raw)}/{int(win_step_raw)}) "
            f"are not divisible by frame_step={int(frame_step)}; using ceil division for sampled windows."
        )
    print(
        f"[window] raw clip_len/win_step={int(clip_len_raw)}/{int(win_step_raw)} "
        f"-> sampled clip_len/win_step={int(clip_len)}/{int(win_step)} (k={int(frame_step)})"
    )

    keypoint_runtime = KeypointRuntime(
        model_path=resolved_keypoint_path,
        device=run_device,
        backend=keypoint_backend,
    )
    print(f"[pose] backend={keypoint_runtime.backend} model={resolved_keypoint_path}")

    checkpoint = torch.load(str(resolved_ckpt_path), map_location="cpu")
    raw_state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    cfg_num_classes = int(getattr(cfg, "action_classes", 11))
    ckpt_num_classes = _infer_action_classes_from_state_dict(raw_state)
    num_classes = int(ckpt_num_classes) if ckpt_num_classes is not None else int(cfg_num_classes)
    if ckpt_num_classes is not None and int(ckpt_num_classes) != int(cfg_num_classes):
        print(
            f"[model][WARN] Config action_classes={int(cfg_num_classes)} but checkpoint head expects "
            f"{int(ckpt_num_classes)} classes; using checkpoint value."
        )

    model_backbone = load_backbone(cfg)
    model = ActionNet(
        backbone=model_backbone,
        dim_rep=getattr(cfg, "dim_rep", 512),
        num_classes=int(num_classes),
        dropout_ratio=getattr(cfg, "dropout_ratio", 0.0),
        version=getattr(cfg, "model_version", "class"),
        hidden_dim=getattr(cfg, "hidden_dim", 2048),
        num_joints=getattr(cfg, "num_joints", 17),
    )

    use_dp = run_device.startswith("cuda") and torch.cuda.device_count() > 1
    if use_dp:
        model = nn.DataParallel(model)
    model = model.to(run_device)

    state = _clean_state_dict_for_model(raw_state, model)
    model.load_state_dict(state, strict=True)
    model.eval()

    scale_range = getattr(cfg, "scale_range_test", None)

    labels_file_names = load_labels_file(labels_file)

    unmerged_len_expected = len(CLASS_NAMES_DEFAULT)
    merged_len_expected = len(CLASS_NAMES_MERGED_DEFAULT)
    merge_fall = (not bool(no_merge_fall)) and (num_classes == unmerged_len_expected)

    if labels_file_names is not None:
        if merge_fall and len(labels_file_names) == merged_len_expected:
            class_names_out = list(labels_file_names)
        else:
            base_names = pad_or_trim(list(labels_file_names), num_classes)
            if merge_fall:
                class_names_out = ["Fall"] + [
                    base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT
                ]
            else:
                class_names_out = base_names
    else:
        if num_classes == merged_len_expected:
            class_names_out = list(CLASS_NAMES_MERGED_DEFAULT)
        else:
            base_names = pad_or_trim(list(CLASS_NAMES_DEFAULT), num_classes)
            if merge_fall:
                class_names_out = ["Fall"] + [
                    base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT
                ]
            else:
                class_names_out = base_names

    fall_idx = infer_fall_indices(class_names_out)
    standing_label = next((str(name) for name in class_names_out if "stand" in str(name).lower()), "standing")

    predict_kwargs = dict(
        model=model,
        device=run_device,
        clip_len=int(clip_len),
        scale_range=scale_range,
        merge_fall=merge_fall,
        fall_idx=fall_idx,
        class_names_out=class_names_out,
        unmerged_len_expected=unmerged_len_expected,
        merged_len_expected=merged_len_expected,
        frame_step=frame_step,
    )

    cap = cv2.VideoCapture(str(resolved_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {resolved_video_path.as_posix()}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not np.isfinite(src_fps) or src_fps <= 1e-3:
        src_fps = 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    is_packet_stream = on_packet is not None and bool(no_display)
    fps_play = float(display_fps) if float(display_fps) > 1e-3 else float(src_fps)
    if is_packet_stream and float(display_fps) <= 1e-3:
        fps_play = min(float(src_fps), 18.0)
    frame_period_s = 1.0 / max(1e-6, float(fps_play))

    window_name = "MotionBERT Inference"
    if not bool(no_display):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    writer: Optional[cv2.VideoWriter] = None
    base_w = 0
    base_h = 0

    frames_buf: "deque[np.ndarray]" = deque()
    xy_buf: "deque[np.ndarray]" = deque()
    cf_buf: "deque[np.ndarray]" = deque()

    all_xy: List[np.ndarray] = []
    all_cf: List[np.ndarray] = []

    img_shape: Optional[Tuple[int, int]] = None
    processed_total = 0
    sampled_total = 0
    display_idx = 0
    cap_done = False

    window_preds: Dict[int, dict] = {}
    skipped_windows: set[int] = set()
    next_win_start = 0
    last_xy = np.zeros((17, 2), dtype=np.float32)
    last_cf = np.zeros((17,), dtype=np.float32)

    drop_empty_windows = not bool(keep_empty_windows)
    conf_thres_final = float(display_conf_thres)

    def pose_on_frame(frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        detections = keypoint_runtime.predict(
            frame_bgr=frame_bgr,
            imgsz=int(imgsz),
            conf=float(yolo_conf),
            max_people=1,
            use_half=bool(use_half),
        )

        kpts_xy = np.zeros((17, 2), dtype=np.float32)
        kpts_conf = np.zeros((17,), dtype=np.float32)

        xy_all = detections.xy
        cf_all = detections.conf
        if xy_all.ndim == 3 and xy_all.shape[0] > 0:
            scores = cf_all.sum(axis=1) if (cf_all.ndim == 2 and cf_all.shape[0] == xy_all.shape[0]) else None
            best = int(np.argmax(scores)) if scores is not None else 0
            kpts_xy = xy_all[best].astype(np.float32)
            if cf_all.ndim == 2 and cf_all.shape[0] == xy_all.shape[0]:
                kpts_conf = cf_all[best].astype(np.float32)
            else:
                kpts_conf = np.ones((17,), dtype=np.float32)

        return kpts_xy, kpts_conf

    def process_next_frame() -> bool:
        nonlocal processed_total, sampled_total, cap_done, img_shape, last_xy, last_cf

        if cap_done:
            return False

        ok, frame = cap.read()
        if not ok:
            cap_done = True
            return False

        if img_shape is None:
            h, w = frame.shape[:2]
            img_shape = (int(h), int(w))

        raw_idx = int(processed_total)
        do_pose = (int(raw_idx) % int(frame_step)) == 0

        if do_pose:
            xy, cf = pose_on_frame(frame)
            all_xy.append(xy)
            all_cf.append(cf)
            sampled_total += 1
            last_xy = xy
            last_cf = cf
        else:
            xy = last_xy
            cf = last_cf

        frames_buf.append(frame)
        xy_buf.append(xy)
        cf_buf.append(cf)
        processed_total += 1
        return True

    def make_window_annotation(start: int) -> Optional[Tuple[str, dict]]:
        if img_shape is None:
            return None

        end = int(start) + int(clip_len)
        if end <= int(sampled_total):
            raw_kxy = np.stack(all_xy[start:end], axis=0).astype(np.float32)
            raw_ksc = np.stack(all_cf[start:end], axis=0).astype(np.float32)
        else:
            if not cap_done or (not bool(pad_tail)):
                return None
            if start >= int(sampled_total):
                return None
            pad_n = int(end - int(sampled_total))
            if pad_n >= int(clip_len):
                return None
            raw_kxy = np.stack(all_xy[start:sampled_total], axis=0).astype(np.float32)
            raw_ksc = np.stack(all_cf[start:sampled_total], axis=0).astype(np.float32)
            last_xy_local = raw_kxy[-1:, :, :]
            last_sc_local = raw_ksc[-1:, :]
            raw_kxy = np.concatenate([raw_kxy, np.repeat(last_xy_local, pad_n, axis=0)], axis=0)
            raw_ksc = np.concatenate([raw_ksc, np.repeat(last_sc_local, pad_n, axis=0)], axis=0)

        if raw_kxy.shape != (int(clip_len), 17, 2) or raw_ksc.shape != (int(clip_len), 17):
            return None

        kxy = raw_kxy.copy()
        ksc = raw_ksc.copy()

        nonfinite_xy = ~np.isfinite(kxy)
        nonfinite_sc = ~np.isfinite(ksc)
        if nonfinite_xy.any() or nonfinite_sc.any():
            kxy[nonfinite_xy] = 0.0
            ksc[nonfinite_sc] = 0.0
            nonfinite_joint = nonfinite_xy.any(axis=2) | nonfinite_sc
            ksc[nonfinite_joint] = 0.0

        if (ksc < 0).any() or (ksc > 1).any():
            ksc = np.clip(ksc, 0.0, 1.0)

        interpolate_missing_joints_inplace(kxy, ksc, missing_conf_thres=float(missing_conf_thres))

        if drop_empty_windows:
            if np.all(ksc <= float(missing_conf_thres)):
                return None
            if (np.ptp(kxy[..., 0]) < 1e-6) and (np.ptp(kxy[..., 1]) < 1e-6):
                return None

        frame_dir = f"{resolved_video_path.stem}_s{start}_len{clip_len}"
        ann = {
            "frame_dir": frame_dir,
            "total_frames": int(clip_len),
            "img_shape": (int(img_shape[0]), int(img_shape[1])),
            "keypoint": kxy[None, ...].astype(np.float32),
            "keypoint_score": ksc[None, ...].astype(np.float32),
            "label": 0,
        }
        return frame_dir, ann

    def compute_window_pred(start: int) -> Optional[dict]:
        if int(start) in window_preds:
            return window_preds[int(start)]
        if int(start) in skipped_windows:
            return None

        window = make_window_annotation(int(start))
        if window is None:
            if cap_done and (not bool(pad_tail)) and int(sampled_total) < int(start) + int(clip_len):
                skipped_windows.add(int(start))
            if drop_empty_windows and int(sampled_total) >= int(start) + int(clip_len):
                skipped_windows.add(int(start))
            return None

        frame_dir, ann = window
        pred = predict_one_window(ann=ann, frame_dir=frame_dir, **predict_kwargs)
        window_preds[int(start)] = pred
        return pred

    def compute_ready_windows() -> None:
        nonlocal next_win_start

        while True:
            if int(next_win_start) in window_preds or int(next_win_start) in skipped_windows:
                next_win_start = int(next_win_start) + int(win_step)
                continue

            if not cap_done:
                if int(sampled_total) >= int(next_win_start) + int(clip_len):
                    compute_window_pred(int(next_win_start))
                    next_win_start = int(next_win_start) + int(win_step)
                    continue
                break

            if int(sampled_total) >= int(next_win_start) + int(clip_len):
                compute_window_pred(int(next_win_start))
                next_win_start = int(next_win_start) + int(win_step)
                continue
            if bool(pad_tail) and int(next_win_start) < int(sampled_total):
                compute_window_pred(int(next_win_start))
                next_win_start = int(next_win_start) + int(win_step)
                continue
            break

    def get_pred_for_frame(frame_idx: int) -> Optional[dict]:
        if frame_idx < 0:
            return None
        sample_idx = int(frame_idx) // int(frame_step)
        win_start = (int(sample_idx) // int(win_step)) * int(win_step)
        pred = window_preds.get(int(win_start))
        if pred is not None:
            return pred

        start = int(win_start) - int(win_step)
        while start >= 0:
            candidate = window_preds.get(int(start))
            if candidate is not None and int(frame_idx) <= int(candidate["end_frame"]):
                return candidate
            start -= int(win_step)
        return None

    hud_delay_frames = 32
    hud_delay_buf: "deque[List[str]]" = deque()

    try:
        while int(sampled_total) < int(clip_len) and not cap_done:
            process_next_frame()
        if int(processed_total) <= 0:
            raise RuntimeError("Video had 0 frames.")

        compute_window_pred(0)
        next_win_start = int(win_step)

        if save_path is not None and frames_buf:
            save_path_resolved = Path(save_path).expanduser()
            if save_path_resolved.suffix == "":
                save_path_resolved = save_path_resolved.with_suffix(".mp4")
            base_h, base_w = frames_buf[0].shape[:2]
            writer = open_video_writer(
                save_path=save_path_resolved,
                fps=float(src_fps),
                frame_size=(int(base_w), int(base_h)),
            )

        fps_ema: Optional[float] = None
        ema_alpha = 0.1

        while True:
            if not frames_buf and cap_done:
                break

            t_frame_start = time.perf_counter()

            display_sample_idx = int(display_idx) // int(frame_step)
            target_sampled = int(display_sample_idx) + int(clip_len) + 1
            while (not cap_done) and int(sampled_total) < int(target_sampled):
                process_next_frame()

            compute_ready_windows()

            if not frames_buf:
                continue

            win_start = (int(display_sample_idx) // int(win_step)) * int(win_step)
            if win_start not in window_preds and win_start not in skipped_windows:
                while (not cap_done) and int(sampled_total) < int(win_start) + int(clip_len):
                    process_next_frame()
                    compute_ready_windows()
                compute_window_pred(int(win_start))

            pred = get_pred_for_frame(int(display_idx))
            if pred is None:
                pred_id = -1
                pred_label = "..."
                pred_conf = 0.0
                p_fall: Optional[float] = None
            else:
                pred_id = int(pred["pred_id"])
                pred_label = str(pred["pred_name"])
                pred_conf = float(pred["pred_conf"])
                p_fall = float(pred["p_fall"])

            frame_original = frames_buf[0]
            xy = xy_buf[0]
            cf = cf_buf[0]

            frame_info = f"frame {int(display_idx) + 1}"
            if int(frame_count) > 0:
                frame_info += f"/{int(frame_count)}"

            fps_for_hud = float(fps_ema) if fps_ema is not None else float(fps_play)
            hud = [
                frame_info,
                f"fps: {float(fps_for_hud):.1f}",
                f"pose: {pred_label} ({float(pred_conf):.2f})" if int(pred_id) >= 0 else "pose: ...",
                f"T={int(clip_len)} stride={int(win_step)} sampled (k={int(frame_step)})",
            ]
            if p_fall is not None:
                hud.append(f"fall_prob: {float(p_fall):.2f}")

            hud_delay_buf.append(list(hud))
            show_current_hud = bool(cap_done) and len(frames_buf) <= int(hud_delay_frames)
            if show_current_hud:
                hud_to_draw = list(hud)
            elif len(hud_delay_buf) <= int(hud_delay_frames):
                hud_to_draw = list(hud)
                if len(hud_to_draw) > 2:
                    hud_to_draw[2] = f"pose: {standing_label} (1.00)"
                for i, line in enumerate(hud_to_draw):
                    if line.startswith("fall_prob:"):
                        hud_to_draw[i] = "fall_prob: 0.00"
            else:
                hud_to_draw = hud_delay_buf.popleft()

            if on_packet is not None:
                ok_jpg, encoded = cv2.imencode(
                    ".jpg",
                    frame_original,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if not ok_jpg:
                    raise RuntimeError(f"Failed to encode frame {int(display_idx)} to JPEG.")
                frame_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
                packet: Dict[str, Any] = {
                    "type": "frame",
                    "frame_index": int(display_idx),
                    "frame_number": int(display_idx) + 1,
                    "frame_count": int(frame_count),
                    "fps": float(fps_for_hud),
                    "pred": {
                        "label": str(pred_label),
                        "conf": float(pred_conf),
                        "class_id": int(pred_id),
                    },
                    "params": {
                        "T": int(clip_len),
                        "stride": int(win_step),
                        "k": int(frame_step),
                    },
                    "hud_lines": list(hud_to_draw),
                    "pose": {
                        "format": "coco17",
                        "xy": np.asarray(xy, dtype=np.float32).tolist(),
                        "conf": np.asarray(cf, dtype=np.float32).tolist(),
                        "conf_thres": float(conf_thres_final),
                        "skeleton": [[int(a), int(b)] for a, b in SKELETON],
                    },
                    "frame_jpeg_b64": frame_b64,
                    "size": {"w": int(frame_original.shape[1]), "h": int(frame_original.shape[0])},
                    "overlay": {
                        "hud": {
                            "x": 10,
                            "y": 10,
                            "pad": 8,
                            "line_gap": 6,
                            "bg_alpha": 0.6,
                            "font_px": 20,
                        },
                        "pose": {
                            "keypoint_radius": 3,
                            "skeleton_width": 2,
                        },
                    },
                }
                if p_fall is not None:
                    packet["pred"]["fall_prob"] = float(p_fall)
                on_packet(packet)

            key = -1
            needs_rendered_frame = (on_frame is not None) or (writer is not None) or (not bool(no_display))
            if needs_rendered_frame:
                frame_to_render = draw_pose(
                    frame_original.copy(),
                    xy,
                    cf,
                    conf_thres=float(conf_thres_final),
                    draw_skeleton=True,
                )
                frame_to_render = draw_hud(frame_to_render, hud_to_draw)

                if on_frame is not None:
                    on_frame(frame_to_render)

                if writer is not None:
                    frame_h, frame_w = frame_to_render.shape[:2]
                    frame_to_write = frame_to_render
                    if frame_h != int(base_h) or frame_w != int(base_w):
                        frame_to_write = cv2.resize(frame_to_render, (int(base_w), int(base_h)), interpolation=cv2.INTER_LINEAR)
                    writer.write(frame_to_write)

                if not bool(no_display):
                    cv2.imshow(window_name, frame_to_render)
                    if bool(realtime):
                        elapsed_s = time.perf_counter() - t_frame_start
                        remaining_s = frame_period_s - elapsed_s
                        wait_ms = int(max(1, remaining_s * 1000.0)) if remaining_s > 0 else 1
                    else:
                        wait_ms = 1
                    key = cv2.waitKey(int(wait_ms)) & 0xFF

            if bool(realtime) and is_packet_stream:
                elapsed_s = time.perf_counter() - t_frame_start
                remaining_s = frame_period_s - elapsed_s
                if remaining_s > 0.0:
                    time.sleep(remaining_s)

            total_ms = (time.perf_counter() - t_frame_start) * 1000.0
            inst_fps = 1000.0 / max(1e-6, total_ms)
            fps_ema = inst_fps if fps_ema is None else (1.0 - ema_alpha) * float(fps_ema) + ema_alpha * inst_fps

            should_quit = key in (ord("q"), 27)

            frames_buf.popleft()
            xy_buf.popleft()
            cf_buf.popleft()
            display_idx += 1
            if should_quit:
                break

        return 0
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not bool(no_display):
            cv2.destroyAllWindows()


def run_inference_stream(
    *,
    video_path: Path,
    classification_model_path: Path,
    keypoint_model_path: Path,
    on_frame: Optional[Callable[[np.ndarray], None]] = None,
    save_path: Optional[Path] = None,
    no_display: bool = True,
    realtime: bool = True,
    display_fps: float = 0.0,
    **inference_options: Any,
) -> int:
    return run_inference_stream_packets(
        video_path=video_path,
        classification_model_path=classification_model_path,
        keypoint_model_path=keypoint_model_path,
        on_packet=None,
        on_frame=on_frame,
        save_path=save_path,
        no_display=bool(no_display),
        realtime=bool(realtime),
        display_fps=float(display_fps),
        **dict(inference_options),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to MotionBERT checkpoint (*.bin) OR checkpoint directory (contains best_epoch.bin)",
    )
    ap.add_argument(
        "--config",
        type=str,
        default=pick_default_config_relpath(),
        help="MotionBERT config yaml (can be relative to models/MotionBERT/)",
    )
    ap.add_argument("--video", type=str, required=True, help="Path to input mp4")
    ap.add_argument(
        "--keypoint-model",
        "--yolo-weights",
        dest="keypoint_model",
        type=str,
        default="models/keypoint/ultralytics/yolo11l-pose.pt",
        help="Keypoint model path (YOLO weights file, AlphaPose bundle directory, or ViTPose marker directory).",
    )
    ap.add_argument(
        "--keypoint-backend",
        type=str,
        default=None,
        choices=["yolo", "alphapose", "vitpose"],
        help="Override keypoint backend (auto-detected from --keypoint-model when omitted).",
    )
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument(
        "--win-len",
        type=int,
        default=None,
        help="Raw-frame window length (defaults to config.clip_len, then scaled by --k/--frame-step).",
    )
    ap.add_argument(
        "--win-step",
        type=int,
        default=16,
        help="Raw-frame window stride (scaled by --k/--frame-step for sampled inference).",
    )
    ap.add_argument(
        "--frame-step",
        "--k",
        type=int,
        default=1,
        help=(
            "Run YOLO pose every k raw frames (k>=1). "
            "Window length/stride are defined in raw frames and scaled to sampled frames using ceil division."
        ),
    )
    ap.add_argument("--pad-tail", action="store_true")
    ap.add_argument("--missing-conf-thres", type=float, default=0.0)
    ap.add_argument("--keep-empty-windows", action="store_true", default=False)
    ap.add_argument("--out-pkl", type=str, default="outputs/motionbert_video.pkl")
    ap.add_argument("--out-csv", type=str, default="outputs/motionbert_video_preds.csv")
    ap.add_argument("--labels-file", type=str, default=None)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--display", action="store_true", help="Display video with pose + streaming window prediction (FPS HUD)")
    ap.add_argument("--display-conf-thres", type=float, default=0.2, help="Keypoint conf threshold for drawing")
    ap.add_argument("--display-fps", type=float, default=None, help="Playback FPS for display (default: video FPS)")
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save annotated output video (e.g. out.mp4). If a directory, writes <video_stem>_annotated.mp4 inside.",
    )
    ap.add_argument("--no-merge-fall", action="store_true", help="Disable merging the first five fall labels into one class")
    args = ap.parse_args()

    device = pick_device(args.device)

    ckpt_path = resolve_checkpoint_path(args.model)
    video_path = resolve_path(args.video, desc="Video")
    cfg_path = resolve_path(args.config, desc="Config")
    keypoint_model_path = resolve_path(args.keypoint_model, desc="Keypoint model path")

    cfg = get_config(str(cfg_path))
    clip_len_raw = int(args.win_len) if args.win_len is not None else int(getattr(cfg, "clip_len", 64))
    win_step_raw = max(1, int(args.win_step))
    frame_step = int(args.frame_step)
    if int(frame_step) <= 0:
        raise ValueError("--frame-step/--k must be >= 1.")
    if int(clip_len_raw) <= 0:
        raise ValueError(f"Invalid window length: {clip_len_raw}.")

    clip_len = max(1, int(ceil_div_pos(int(clip_len_raw), int(frame_step))))
    win_step = max(1, int(ceil_div_pos(int(win_step_raw), int(frame_step))))
    if int(frame_step) > 1 and (
        (int(clip_len_raw) % int(frame_step)) != 0 or (int(win_step_raw) % int(frame_step)) != 0
    ):
        print(
            f"[window][WARN] raw clip_len/win_step ({int(clip_len_raw)}/{int(win_step_raw)}) "
            f"are not divisible by frame_step={int(frame_step)}; using ceil division for sampled windows."
        )
    print(
        f"[window] raw clip_len/win_step={int(clip_len_raw)}/{int(win_step_raw)} "
        f"-> sampled clip_len/win_step={int(clip_len)}/{int(win_step)} (k={int(frame_step)})"
    )

    save_path: Optional[Path] = None
    if args.save:
        save_arg = Path(args.save).expanduser()
        if str(args.save).endswith(("/", "\\")) or (save_arg.exists() and save_arg.is_dir()):
            save_path = save_arg / f"{video_path.stem}_annotated.mp4"
        else:
            save_path = save_arg
        if save_path.suffix == "":
            save_path = save_path.with_suffix(".mp4")
        if not args.display:
            print("[WARN] --save provided; enabling --display for annotated video output.")
            args.display = True

    if args.display:
        return stream_infer_and_display(
            ckpt_path=ckpt_path,
            video_path=video_path,
            cfg=cfg,
            keypoint_model_path=keypoint_model_path,
            device=device,
            clip_len=clip_len,
            win_step=win_step,
            frame_step=frame_step,
            clip_len_raw=clip_len_raw,
            win_step_raw=win_step_raw,
            save_path=save_path,
            args=args,
        )

    # ------------------------------------------------------------------
    # 1) Keypoint extraction
    # ------------------------------------------------------------------
    keypoint_runtime = KeypointRuntime(
        model_path=keypoint_model_path,
        device=device,
        backend=args.keypoint_backend,
    )
    print(f"[pose] backend={keypoint_runtime.backend} model={keypoint_model_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path.as_posix()}")

    frames_xy: List[np.ndarray] = []
    frames_cf: List[np.ndarray] = []
    img_shape = None
    frame_idx = 0
    sampled_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if img_shape is None:
            h, w = frame.shape[:2]
            img_shape = (int(h), int(w))

        do_pose = (int(frame_idx) % int(frame_step)) == 0
        if do_pose:
            detections = keypoint_runtime.predict(
                frame_bgr=frame,
                imgsz=int(args.imgsz),
                conf=float(args.conf_thres),
                max_people=1,
                use_half=False,
            )

            kpts_xy = np.zeros((17, 2), dtype=np.float32)
            kpts_conf = np.zeros((17,), dtype=np.float32)

            xy_all = detections.xy
            cf_all = detections.conf
            if xy_all.ndim == 3 and xy_all.shape[0] > 0:
                scores = cf_all.sum(axis=1) if (cf_all.ndim == 2 and cf_all.shape[0] == xy_all.shape[0]) else None
                best = int(np.argmax(scores)) if scores is not None else 0
                kpts_xy = xy_all[best].astype(np.float32)
                if cf_all.ndim == 2 and cf_all.shape[0] == xy_all.shape[0]:
                    kpts_conf = cf_all[best].astype(np.float32)
                else:
                    kpts_conf = np.ones((17,), dtype=np.float32)

            frames_xy.append(kpts_xy)
            frames_cf.append(kpts_conf)
            sampled_idx += 1

        frame_idx += 1
        if args.limit_frames is not None and frame_idx >= int(args.limit_frames):
            break

    cap.release()

    if img_shape is None:
        raise RuntimeError("No frames read from video.")
    if sampled_idx <= 0:
        raise RuntimeError("No sampled frames were generated. Reduce --k/--frame-step or remove --limit-frames.")

    kpts_xy = np.stack(frames_xy, axis=0)  # (T_sampled,17,2)
    kpts_conf = np.stack(frames_cf, axis=0)  # (T_sampled,17)

    # ------------------------------------------------------------------
    # 2) Build and save MotionBERT action pkl
    # ------------------------------------------------------------------
    split_list, annotations = build_windows(
        kpts_xy=kpts_xy,
        kpts_conf=kpts_conf,
        img_shape=img_shape,
        win_len=int(clip_len),
        win_step=int(win_step),
        pad_tail=bool(args.pad_tail),
        missing_conf_thres=float(args.missing_conf_thres),
        drop_empty_windows=not bool(args.keep_empty_windows),
        video_stem=video_path.stem,
    )

    dataset = {"split": {"xsub_train": [], "xsub_val": split_list}, "annotations": annotations}

    out_pkl = Path(args.out_pkl)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    if not annotations:
        print("No windows were generated. Try --pad-tail or smaller --win-len.")
        return 1

    # ------------------------------------------------------------------
    # 3) Load MotionBERT ActionNet
    # ------------------------------------------------------------------
    model_backbone = load_backbone(cfg)
    model = ActionNet(
        backbone=model_backbone,
        dim_rep=getattr(cfg, "dim_rep", 512),
        num_classes=int(getattr(cfg, "action_classes", 11)),
        dropout_ratio=getattr(cfg, "dropout_ratio", 0.0),
        version=getattr(cfg, "model_version", "class"),
        hidden_dim=getattr(cfg, "hidden_dim", 2048),
        num_joints=getattr(cfg, "num_joints", 17),
    )

    use_dp = device.startswith("cuda") and torch.cuda.device_count() > 1
    if use_dp:
        model = nn.DataParallel(model)
    model = model.to(device)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = _clean_state_dict_for_model(state, model)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ------------------------------------------------------------------
    # 4) Run per-window inference
    # ------------------------------------------------------------------
    num_classes = int(getattr(cfg, "action_classes", 11))
    scale_range = getattr(cfg, "scale_range_test", None)

    labels_file_names = load_labels_file(args.labels_file)

    unmerged_len_expected = len(CLASS_NAMES_DEFAULT)
    merged_len_expected = len(CLASS_NAMES_MERGED_DEFAULT)

    # Only merge at inference time when the model is trained with separate fall subclasses (11-class).
    merge_fall = (not bool(args.no_merge_fall)) and (num_classes == unmerged_len_expected)

    if labels_file_names is not None:
        if merge_fall and len(labels_file_names) == merged_len_expected:
            # User provided already-merged display names (7).
            class_names_out = list(labels_file_names)
        else:
            # User provided names matching the model output space.
            base_names = pad_or_trim(list(labels_file_names), num_classes)
            if merge_fall:
                class_names_out = ["Fall"] + [base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
            else:
                class_names_out = base_names
    else:
        # No labels file: choose a sane default taxonomy for the configured model output space.
        if num_classes == merged_len_expected:
            class_names_out = list(CLASS_NAMES_MERGED_DEFAULT)
        else:
            base_names = pad_or_trim(list(CLASS_NAMES_DEFAULT), num_classes)
            if merge_fall:
                class_names_out = ["Fall"] + [base_names[i] for i in range(unmerged_len_expected) if i not in FALL_CLASS_IDS_DEFAULT]
            else:
                class_names_out = base_names

    fall_idx = infer_fall_indices(class_names_out)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ann_map = {ann["frame_dir"]: ann for ann in annotations}

    def write_header(writer: csv.writer) -> None:
        writer.writerow(
            [
                "frame_dir",
                "start_frame",
                "end_frame",
                "pred_id",
                "pred_name",
                "pred_conf",
                "p_fall",
            ]
        )

    def write_pred_row(writer: csv.writer, pred: dict) -> None:
        writer.writerow(
            [
                pred["frame_dir"],
                pred["start_frame"],
                pred["end_frame"],
                pred["pred_id"],
                pred["pred_name"],
                f"{float(pred['pred_conf']):.6f}",
                f"{float(pred['p_fall']):.6f}",
            ]
        )

    # ------------------------------------------------------------------
    # 4) Predict windows (offline)
    # ------------------------------------------------------------------
    predict_kwargs = dict(
        model=model,
        device=device,
        clip_len=clip_len,
        scale_range=scale_range,
        merge_fall=merge_fall,
        fall_idx=fall_idx,
        class_names_out=class_names_out,
        unmerged_len_expected=unmerged_len_expected,
        merged_len_expected=merged_len_expected,
        frame_step=frame_step,
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        write_header(writer)
        for frame_dir in split_list:
            ann = ann_map[frame_dir]
            pred = predict_one_window(ann=ann, frame_dir=frame_dir, **predict_kwargs)
            write_pred_row(writer, pred)

    print(f"Saved pkl: {out_pkl.as_posix()}")
    print(f"Saved predictions: {out_csv.as_posix()}")
    print(f"Windows: {len(split_list)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
