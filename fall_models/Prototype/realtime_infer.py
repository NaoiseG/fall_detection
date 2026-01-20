#!/usr/bin/env python3
"""
Real-time pose -> temporal action/fall inference on video.

This version is updated to MATCH your NEW training pipeline:
- confidence thresholding + short-gap interpolation for missing joints
- per-frame skeleton normalisation (center + scale)
- optional velocity + acceleration
- optional global features (COM speed + aspect)
- optional mask channel (valid-frame indicator)

It also fixes a bug in person selection (conf indexing) and removes hard-coded
node_features=3 for GCN/STGCN. node_features is inferred from in_features.

Run examples (from project root):

  python realtime_infer.py --temporal-model tcn --ckpt-root models --ckpt-subdir <RUN_DIR> --source 0
  python realtime_infer.py --temporal-model stgcn --ckpt-root models --ckpt-subdir <RUN_DIR> --source input.mp4 --out out.mp4 --show 0

Notes:
- If your checkpoint contains preprocessing metadata, this script will use it.
- If not, it falls back to CLI defaults.
"""

from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Tuple, Optional

import numpy as np
import torch
import cv2
from ultralytics import YOLO

# Same model definitions as training/eval
from models.tcn.simple_tcn import TCNBaseline
from models.lstm.simple_lstm import LSTMBaseline
from models.gru.simple_gru import GRUBaseline
from models.gcn.simple_gcn import GCNBaseline
from models.mlp.simple_mlp import MLPBaseline
from models.stgcn.simple_stgcn import STGCNBaseline

# -----------------------------
# Labels (11 classes)
# -----------------------------
CLASS_NAMES = [
    "Falling forward using hands",
    "Falling forward using knees",
    "Falling backwards",
    "Falling sideward",
    "Falling sitting in an empty chair",
    "Walking",
    "Standing",
    "Sitting",
    "Picking up an object",
    "Jumping",
    "Laying",
]

K = 17  # COCO-17 keypoints

# COCO-17 skeleton edges (0-based indices)
COCO17_SKELETON = [
    (5, 7), (7, 9),        # left arm
    (6, 8), (8, 10),       # right arm
    (5, 6),                # shoulders
    (5, 11), (6, 12),      # torso
    (11, 12),              # hips
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
    (0, 1), (0, 2),        # nose to eyes
    (1, 3), (2, 4),        # eyes to ears
    (3, 5), (4, 6),        # ears to shoulders (approx)
]

# For normalisation helpers (COCO indices)
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


# -----------------------------
# Utilities
# -----------------------------
def parse_source(source_arg: str):
    """
    --source can be:
      - "0" (or other integer) for webcam index
      - a file path
      - a GStreamer pipeline string
    """
    s = str(source_arg)
    if s.isdigit() and len(s) <= 2:
        return int(s)
    return s


def pick_device(device_arg: Optional[str]):
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_ckpt_path(args) -> Path:
    ckpt_root = Path(args.ckpt_root)
    ckpt_subdir = Path(args.ckpt_subdir)
    weights_name = args.weights_name.strip() if args.weights_name else ""
    if not weights_name:
        weights_name = f"{args.temporal_model.lower()}_best.pt"
    return ckpt_root / ckpt_subdir / weights_name


def overlay_text(frame, text, fps_text=None):
    x, y = 10, 30
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    if fps_text is not None:
        cv2.putText(frame, fps_text, (x, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_pose(frame, kpts_xy, kpts_conf=None, conf_thres=0.2, draw_skeleton=True):
    """Draw keypoints and optional skeleton. Skips points/edges with low conf."""
    if kpts_xy is None:
        return frame

    h, w = frame.shape[:2]

    def in_bounds(x, y):
        return 0 <= x < w and 0 <= y < h

    if draw_skeleton and kpts_xy.shape[0] >= K:
        for a, b in COCO17_SKELETON:
            xa, ya = kpts_xy[a]
            xb, yb = kpts_xy[b]
            ca = kpts_conf[a] if kpts_conf is not None else 1.0
            cb = kpts_conf[b] if kpts_conf is not None else 1.0
            if ca < conf_thres or cb < conf_thres:
                continue
            xa_i, ya_i = int(round(float(xa))), int(round(float(ya)))
            xb_i, yb_i = int(round(float(xb))), int(round(float(yb)))
            if in_bounds(xa_i, ya_i) and in_bounds(xb_i, yb_i):
                cv2.line(frame, (xa_i, ya_i), (xb_i, yb_i), (0, 255, 0), 2, cv2.LINE_AA)

    for i in range(min(K, kpts_xy.shape[0])):
        x, y = kpts_xy[i]
        c = kpts_conf[i] if kpts_conf is not None else 1.0
        if c < conf_thres:
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if in_bounds(xi, yi):
            cv2.circle(frame, (xi, yi), 3, (0, 0, 255), -1, cv2.LINE_AA)

    return frame


# -----------------------------
# Ultralytics keypoints extraction (FIXED)
# -----------------------------
def extract_top_person_kpts(result, max_people: int = 1):
    """
    Returns:
      kpts_xy: (K,2) float32
      kpts_conf: (K,) float32
      found: bool
    """
    if result is None or getattr(result, "keypoints", None) is None:
        return None, None, False

    kps = result.keypoints
    if kps is None or len(kps) == 0:
        return None, None, False

    if getattr(kps, "xy", None) is None or kps.xy.numel() == 0:
        return None, None, False

    # pick person with highest mean confidence
    if getattr(kps, "conf", None) is not None and kps.conf is not None and kps.conf.numel() > 0:
        idx = int(np.argmax(kps.conf.mean(dim=1).detach().cpu().numpy()))
    else:
        idx = 0

    xy0 = kps.xy[idx].detach().cpu().float().numpy()  # (K,2)

    if getattr(kps, "conf", None) is not None and kps.conf is not None and kps.conf.numel() > 0:
        conf0 = kps.conf[idx].detach().cpu().float().numpy()  # (K,)
    else:
        conf0 = np.ones((xy0.shape[0],), dtype=np.float32)

    # Ensure K=17
    if xy0.shape[0] != K:
        out_xy = np.zeros((K, 2), dtype=np.float32)
        out_conf = np.zeros((K,), dtype=np.float32)
        kk = min(K, xy0.shape[0])
        out_xy[:kk] = xy0[:kk].astype(np.float32, copy=False)
        out_conf[:kk] = conf0[:kk].astype(np.float32, copy=False)
        return out_xy, out_conf, True

    return xy0.astype(np.float32, copy=False), conf0.astype(np.float32, copy=False), True


# -----------------------------
# Preprocessing (self-contained, mirrors dataset.py)
# -----------------------------
def _interp_short_gaps_1d(x: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    """Interpolate short gaps, hold for long gaps."""
    N = x.shape[0]
    out = x.copy()

    idx_valid = np.where(valid)[0]
    if idx_valid.size == 0:
        return np.zeros_like(out)

    # ffill
    last = idx_valid[0]
    for i in range(N):
        if valid[i]:
            last = i
        out[i] = out[last]

    # bfill
    last = idx_valid[-1]
    for i in range(N - 1, -1, -1):
        if valid[i]:
            last = i
        out[i] = out[last]

    # linear interpolate short gaps
    for a, b in zip(idx_valid[:-1], idx_valid[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        if gap <= max_gap:
            ya, yb = out[a], out[b]
            for j in range(1, gap + 1):
                t = j / (gap + 1)
                out[a + j] = (1 - t) * ya + t * yb

    return out


def _fill_and_mask_kpts(
    kxy: np.ndarray, kconf: np.ndarray, conf_thres: float, max_interp_gap: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    kxy: (T,K,2), kconf: (T,K)
    - missing if conf < conf_thres
    - fill xy by interpolation/hold
    - set conf=0 where missing
    """
    T, K_, _ = kxy.shape
    xy = kxy.astype(np.float32, copy=True)
    conf = kconf.astype(np.float32, copy=True)

    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

    missing = conf < conf_thres  # (T,K)

    for j in range(K_):
        v = ~missing[:, j]
        xj = xy[:, j, 0]
        yj = xy[:, j, 1]
        xy[:, j, 0] = _interp_short_gaps_1d(xj, v, max_gap=max_interp_gap)
        xy[:, j, 1] = _interp_short_gaps_1d(yj, v, max_gap=max_interp_gap)
        conf[missing[:, j], j] = 0.0

    return xy, conf


def _frame_center_scale(xy_t: np.ndarray, conf_t: np.ndarray) -> Tuple[np.ndarray, float]:
    """Robust per-frame center+scale."""
    valid = conf_t > 0.0

    def safe_mid(a: int, b: int):
        if a < xy_t.shape[0] and b < xy_t.shape[0] and valid[a] and valid[b]:
            return 0.5 * (xy_t[a] + xy_t[b]), True
        return np.zeros((2,), dtype=np.float32), False

    center, ok = safe_mid(L_HIP, R_HIP)
    if not ok:
        center, ok = safe_mid(L_SHOULDER, R_SHOULDER)
    if not ok:
        center = xy_t[valid].mean(axis=0).astype(np.float32) if valid.any() else np.zeros((2,), np.float32)

    scale = 1.0
    if valid[L_SHOULDER] and valid[R_SHOULDER]:
        scale = float(np.linalg.norm(xy_t[L_SHOULDER] - xy_t[R_SHOULDER]))
    elif valid[L_HIP] and valid[R_HIP]:
        scale = float(np.linalg.norm(xy_t[L_HIP] - xy_t[R_HIP]))
    else:
        sh, ok_sh = safe_mid(L_SHOULDER, R_SHOULDER)
        hp, ok_hp = safe_mid(L_HIP, R_HIP)
        if ok_sh and ok_hp:
            scale = float(np.linalg.norm(sh - hp))

    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    return center, scale


def _normalize_xy(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Per-frame translation + scale normalisation."""
    T, K_, _ = xy.shape
    out = np.empty_like(xy, dtype=np.float32)
    for t in range(T):
        center, scale = _frame_center_scale(xy[t], conf[t])
        out[t] = (xy[t] - center[None, :]) / float(scale)
    return out


def _add_velocity_channels(xy_norm: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(xy_norm, dtype=np.float32)
    vel[1:] = xy_norm[1:] - xy_norm[:-1]
    return vel


def _add_acceleration_channels(vel: np.ndarray) -> np.ndarray:
    acc = np.zeros_like(vel, dtype=np.float32)
    acc[2:] = vel[2:] - vel[1:-1]
    return acc


def _global_features(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """
    Returns g: (T,4): com_x, com_y, com_speed, aspect(h/(w+eps))
    """
    T, K_, _ = xy.shape
    g = np.zeros((T, 4), dtype=np.float32)
    eps = 1e-6

    for t in range(T):
        valid = conf[t] > 0.0
        if not np.any(valid):
            continue
        pts = xy[t, valid]
        com = pts.mean(axis=0)
        g[t, 0:2] = com

        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        w = float(mx[0] - mn[0])
        h = float(mx[1] - mn[1])
        g[t, 3] = h / (w + eps)

    dcom = np.zeros((T, 2), dtype=np.float32)
    dcom[1:] = g[1:, 0:2] - g[:-1, 0:2]
    g[:, 2] = np.linalg.norm(dcom, axis=1)
    return g


# -----------------------------
# Model factory (node_features-aware)
# -----------------------------
def get_model(
    model_name: str,
    in_features: int,
    num_classes: int,
    device: str,
    T_used: int | None = None,
    node_features: int | None = None,
):
    model_name = model_name.lower().strip()

    if model_name == "tcn":
        model = TCNBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            kernel_size=3,
            dropout=0.1,
        )

    elif model_name == "lstm":
        model = LSTMBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
            pool="last",
        )

    elif model_name == "gru":
        model = GRUBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
            pool="last",
        )

    elif model_name == "gcn":
        if node_features is None:
            raise ValueError("node_features must be provided for GCN.")
        model = GCNBaseline(
            num_nodes=17,
            node_features=node_features,
            num_classes=num_classes,
            hidden_size=64,
            dropout=0.1,
        )

    elif model_name == "mlp":
        if T_used is None:
            raise ValueError("T_used must be provided for MLP.")
        model = MLPBaseline(
            T=T_used,
            in_features=in_features,
            num_classes=num_classes,
            hidden_sizes=(256, 128),
            dropout=0.2,
        )

    elif model_name == "stgcn":
        if node_features is None:
            raise ValueError("node_features must be provided for STGCN.")
        model = STGCNBaseline(
            num_nodes=17,
            node_features=node_features,
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            t_kernel=9,
            dropout=0.1,
        )

    else:
        raise ValueError(f"Unknown model '{model_name}'.")

    return model.to(device)


# -----------------------------
# Main
# -----------------------------
def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Real-time pose->temporal inference with overlay + video writing.")
    parser.add_argument("--source", type=str, default="0", help="Webcam index (e.g. 0) or video file path or gstreamer string.")
    parser.add_argument("--yolo-weights", type=str, default="yolo11l-pose.pt", help="Ultralytics YOLO pose weights.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--max-people", type=int, default=1, help="Max people to consider (use 1 for top person).")

    parser.add_argument("--device", type=str, default=None, help='Device string, e.g. "cuda" or "cpu". Default: auto.')
    parser.add_argument("--half", type=int, default=0, help="Use FP16 on CUDA for speed (0/1).")

    parser.add_argument("--temporal-model", type=str, required=True, choices=["tcn", "lstm", "gru", "gcn", "mlp", "stgcn"])
    parser.add_argument("--ckpt-root", type=str, default="models")
    parser.add_argument("--ckpt-subdir", type=str, required=True, help="Timestamped run folder, e.g. 2026-01-20_13-34-50_401623")
    parser.add_argument("--weights-name", type=str, default=None, help='Override checkpoint filename, else "{temporal_model}_best.pt".')

    # Set default to 0 so ckpt T is used by default
    parser.add_argument("--T", type=int, default=0, help="Window length. 0 => use checkpoint T/T_used if present.")
    parser.add_argument("--stride", type=int, default=1, help="Run temporal inference every N frames once buffer full.")

    parser.add_argument("--out", type=str, default="annotated.mp4", help="Output video path.")
    parser.add_argument("--show", type=int, default=1, help="Show live preview window (0/1).")

    # If ckpt has no metadata, these are the fallback defaults
    parser.add_argument("--fallback-use-conf", type=int, default=1, help="Fallback: include conf channel if ckpt lacks metadata.")
    parser.add_argument("--fallback-normalize", type=int, default=1, help="Fallback: normalise pose per frame if ckpt lacks metadata.")
    parser.add_argument("--fallback-add-vel", type=int, default=1, help="Fallback: add velocity channels if ckpt lacks metadata.")
    parser.add_argument("--fallback-add-acc", type=int, default=1, help="Fallback: add acceleration channels if ckpt lacks metadata.")
    parser.add_argument("--fallback-add-global", type=int, default=1, help="Fallback: add global features if ckpt lacks metadata.")
    parser.add_argument("--fallback-add-mask", type=int, default=1, help="Fallback: add mask channel if ckpt lacks metadata.")
    parser.add_argument("--fallback-conf-thres", type=float, default=0.2, help="Fallback: conf threshold for missing joints.")
    parser.add_argument("--fallback-max-interp-gap", type=int, default=5, help="Fallback: max gap for interpolation.")
    parser.add_argument("--fallback-min-valid-frac", type=float, default=0.3, help="Fallback: valid-frame threshold (fraction joints above conf).")

    args = parser.parse_args()

    device = pick_device(args.device)
    args.device = device

    # Load YOLO pose model
    pose_model = YOLO(args.yolo_weights)

    # Load checkpoint
    ckpt_path = resolve_ckpt_path(args)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    has_meta = isinstance(ckpt, dict) and ("state_dict" in ckpt)
    state = ckpt["state_dict"] if has_meta else ckpt

    # ---- Read metadata (or fallback) ----
    # window length
    if has_meta:
        T_ckpt = int(ckpt.get("T", ckpt.get("T_used", 0)) or 0)
    else:
        T_ckpt = 0
    T_final = int(args.T) if int(args.T) > 0 else (T_ckpt if T_ckpt > 0 else 64)

    # preprocessing flags
    def _b(key: str, fallback: bool) -> bool:
        if has_meta:
            return bool(ckpt.get(key, fallback))
        return fallback

    use_conf_final = _b("use_conf", bool(args.fallback_use_conf))
    normalize_final = _b("normalize", bool(args.fallback_normalize))
    add_vel_final = _b("add_vel", bool(args.fallback_add_vel))
    add_acc_final = _b("add_acc", bool(args.fallback_add_acc))
    add_global_final = _b("add_global", bool(args.fallback_add_global))
    add_mask_final = _b("add_mask_channel", bool(args.fallback_add_mask))

    conf_thres_final = float(ckpt.get("conf_thres", args.fallback_conf_thres)) if has_meta else float(args.fallback_conf_thres)
    max_interp_gap_final = int(ckpt.get("max_interp_gap", args.fallback_max_interp_gap)) if has_meta else int(args.fallback_max_interp_gap)
    min_valid_frac_final = float(ckpt.get("min_valid_frac", args.fallback_min_valid_frac)) if has_meta else float(args.fallback_min_valid_frac)

    if add_acc_final and not add_vel_final:
        raise RuntimeError("add_acc=True requires add_vel=True (acc is computed from vel).")

    # model dims
    if has_meta:
        in_features_final = int(ckpt["in_features"])
        num_classes_final = int(ckpt["num_classes"])
    else:
        # infer layout if no metadata
        cj = 2
        if use_conf_final:
            cj += 1
        if add_vel_final:
            cj += 2
        if add_acc_final:
            cj += 2
        if add_global_final:
            cj += 4
        if add_mask_final:
            cj += 1
        in_features_final = int(K * cj)
        num_classes_final = 11

    node_features_final = int(in_features_final // 17) if (in_features_final % 17 == 0) else None

    # Instantiate and load model
    model = get_model(
        args.temporal_model,
        in_features=in_features_final,
        num_classes=num_classes_final,
        device=args.device,
        T_used=T_final,
        node_features=node_features_final,
    )
    model.load_state_dict(state, strict=False)
    model.eval()

    # Optional half precision (CUDA only)
    use_half = bool(args.half) and str(device).startswith("cuda")
    if use_half:
        model = model.half()

    # Video capture
    source = parse_source(args.source)
    if isinstance(source, str) and ("!" in source or source.strip().startswith("gst-")):
        cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    # infer fps/size
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 1e-3 or np.isnan(src_fps):
        src_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        ok, fr = cap.read()
        if not ok:
            raise RuntimeError("Could not read initial frame to infer video size.")
        height, width = fr.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # writer
    out_path = str(args.out)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(src_fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {out_path}")

    # Buffers store RAW xy/conf so we can compute vel/acc/global/mask over sequence
    xy_buf = deque(maxlen=T_final)    # each (K,2)
    cf_buf = deque(maxlen=T_final)    # each (K,)

    # prediction persistence
    last_pred_idx = None
    last_pred_conf = 0.0

    # fps smoothing
    fps_ema = None
    ema_alpha = 0.1
    t_prev = time.perf_counter()

    frame_idx = 0
    stride = max(1, int(args.stride))

    # zeros for missing frames
    xy_zeros = np.zeros((K, 2), dtype=np.float32)
    cf_zeros = np.zeros((K,), dtype=np.float32)

    # For debug overlay: show what preprocessing is active
    preproc_tag = []
    preproc_tag.append("conf" if use_conf_final else "no-conf")
    preproc_tag.append("norm" if normalize_final else "no-norm")
    preproc_tag.append("vel" if add_vel_final else "no-vel")
    preproc_tag.append("acc" if add_acc_final else "no-acc")
    preproc_tag.append("glob" if add_global_final else "no-glob")
    preproc_tag.append("mask" if add_mask_final else "no-mask")
    preproc_str = ",".join(preproc_tag)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO pose inference
        results = pose_model.predict(
            source=frame,
            imgsz=int(args.imgsz),
            conf=float(args.conf_thres),
            verbose=False,
            max_det=int(args.max_people),
            device=0 if str(device).startswith("cuda") else "cpu",
        )
        res0 = results[0] if isinstance(results, (list, tuple)) and len(results) > 0 else results

        kpts_xy, kpts_conf, found = extract_top_person_kpts(res0, max_people=int(args.max_people))

        if found:
            # keep raw xy/conf for window-level preprocessing
            xy_buf.append(np.nan_to_num(kpts_xy, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False))
            cf_buf.append(np.nan_to_num(kpts_conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False))
        else:
            xy_buf.append(xy_zeros)
            cf_buf.append(cf_zeros)

        # Temporal inference
        if len(xy_buf) == T_final and (frame_idx % stride == 0):
            xy_seq = np.stack(xy_buf, axis=0)   # (T,K,2)
            cf_seq = np.stack(cf_buf, axis=0)   # (T,K)

            # (1) fill/mask
            xy_filled, cf_filled = _fill_and_mask_kpts(
                xy_seq, cf_seq,
                conf_thres=conf_thres_final,
                max_interp_gap=max_interp_gap_final,
            )

            # (2) normalize
            xy_used = _normalize_xy(xy_filled, cf_filled) if normalize_final else xy_filled.astype(np.float32, copy=False)

            feats = [xy_used]  # (T,K,2)

            # conf channel
            if use_conf_final:
                feats.append(cf_filled[..., None])  # (T,K,1)

            # vel / acc
            vel = None
            if add_vel_final:
                vel = _add_velocity_channels(xy_used)  # (T,K,2)
                feats.append(vel)
            if add_acc_final:
                acc = _add_acceleration_channels(vel)  # (T,K,2)
                feats.append(acc)

            # global
            if add_global_final:
                g = _global_features(xy_used, cf_filled)  # (T,4)
                gk = np.repeat(g[:, None, :], repeats=K, axis=1)  # (T,K,4)
                feats.append(gk)

            Xf = np.concatenate(feats, axis=-1).astype(np.float32, copy=False)  # (T,K,Cj)

            # mask channel (valid frame)
            if add_mask_final:
                frac_valid = (cf_filled > conf_thres_final).mean(axis=1)  # (T,)
                frame_valid = (frac_valid >= min_valid_frac_final).astype(np.float32)  # (T,)
                m = np.repeat(frame_valid[:, None, None], repeats=K, axis=1)  # (T,K,1)
                Xf = np.concatenate([Xf, m], axis=-1)  # (T,K,Cj+1)

            window = Xf.reshape(T_final, -1)  # (T, F)

            # Safety: feature size should match model expectation
            if window.shape[1] != in_features_final:
                # Don’t crash mid-run: keep last prediction and show mismatch
                last_pred_idx = None
                last_pred_conf = 0.0
            else:
                X = torch.from_numpy(window).unsqueeze(0).to(device)  # (1,T,F)
                X = X.half() if use_half else X.float()

                with torch.no_grad():
                    logits = model(X)
                    probs = torch.softmax(logits, dim=1)
                    pred_idx = int(probs.argmax(dim=1).item())
                    pred_conf = float(probs.max(dim=1).values.item())

                last_pred_idx = pred_idx
                last_pred_conf = pred_conf

        # Overlay pose
        if found:
            frame = draw_pose(frame, kpts_xy, kpts_conf, conf_thres=0.2, draw_skeleton=True)

        # FPS estimate
        t_now = time.perf_counter()
        dt = max(1e-6, t_now - t_prev)
        inst_fps = 1.0 / dt
        t_prev = t_now
        fps_ema = inst_fps if fps_ema is None else (ema_alpha * inst_fps + (1 - ema_alpha) * fps_ema)

        # Text overlay
        if last_pred_idx is not None and 0 <= int(last_pred_idx) < len(CLASS_NAMES):
            label = CLASS_NAMES[int(last_pred_idx)]
            text = f"{label} (p={last_pred_conf:.2f})"
        elif len(xy_buf) < T_final:
            text = f"Buffering... ({len(xy_buf)}/{T_final})"
        else:
            text = "No prediction (feature mismatch)"

        # include preprocessing + T for sanity
        text2 = f"FPS: {fps_ema:.1f} | T={T_final} stride={stride} | {preproc_str}"
        frame = overlay_text(frame, text, fps_text=text2)

        writer.write(frame)

        if int(args.show) == 1:
            cv2.imshow("Pose + Temporal Inference", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if int(args.show) == 1:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
