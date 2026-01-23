#!/usr/bin/env python3
"""
Real-time pose -> temporal action/fall inference on video.

Key fixes vs your previous script:
- CLASS_NAMES now matches your dataset (11 classes in the order shown in your screenshot)
- Optional class-name loading from checkpoint or from a labels txt file
- Safer fall-probability computation (auto fall-class detection if fall ids not provided)
- Skip inference until enough valid frames exist in the T-window (prevents early "confident garbage")
- Robust handling if model outputs logits shaped (B,T,C)
- MLPBaseline now gets the correct T (T_final)

Based on your uploaded script.  :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Tuple, Optional, List

import numpy as np
import torch
import cv2
from ultralytics import YOLO

# Same model definitions as training/
from models.tcn.simple_tcn import TCNBaseline
from models.lstm.simple_lstm import LSTMBaseline
from models.gru.simple_gru import GRUBaseline
from models.gcn.simple_gcn import GCNBaseline
from models.mlp.simple_mlp import MLPBaseline
from models.stgcn.simple_stgcn import STGCNBaseline

# Optional: CNN+LSTM two-head model (activity + fall)
try:
    from models.cnnlstm.cnn_lstm_two_head import CNNLSTMTwoHead
except Exception:
    CNNLSTMTwoHead = None

# -----------------------------
# Labels (DEFAULT)
# -----------------------------
# Must match the training label order.
# Your screenshot shows these 11 classes in this order:
CLASS_NAMES_DEFAULT = [
    "Falling forward using hands",        # 0
    "Falling forward using knees",        # 1
    "Falling backwards",                  # 2
    "Falling sideward",                   # 3
    "Falling sitting in an empty chair",  # 4
    "Walking",                            # 5
    "Standing",                           # 6
    "Sitting",                            # 7
    "Picking up an object",               # 8
    "Jumping",                            # 9
    "Laying",                             # 10
]

# COCO keypoint order for YOLO pose (17 joints)
K = 17

SKELETON = [
    (5, 7), (7, 9),        # left arm
    (6, 8), (8, 10),       # right arm
    (11, 13), (13, 15),    # left leg
    (12, 14), (14, 16),    # right leg
    (5, 6),                # shoulders
    (11, 12),              # hips
    (5, 11), (6, 12),      # torso sides
]


# -----------------------------
# Utilities
# -----------------------------
def pick_device(device: Optional[str]) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    d = device.lower()
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def resolve_ckpt_path(args) -> Path:
    ckpt_root = Path(args.ckpt_root)
    run_dir = ckpt_root / args.ckpt_subdir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir.as_posix()}")

    weights_name = args.weights_name or f"{args.temporal_model}_best.pt"
    ckpt_path = run_dir / weights_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.as_posix()}")
    return ckpt_path


def load_class_names(num_classes: int, ckpt: object, has_meta: bool, labels_file: Optional[str]) -> List[str]:
    """
    Priority:
    1) --labels-file (one label per line)
    2) checkpoint metadata key 'class_names' or 'classes' if present
    3) CLASS_NAMES_DEFAULT
    """
    # (1) CLI file
    if labels_file:
        p = Path(labels_file)
        if not p.exists():
            raise FileNotFoundError(f"--labels-file not found: {p.as_posix()}")
        names = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        names = []

    # (2) checkpoint
    if not names and has_meta and isinstance(ckpt, dict):
        for key in ("class_names", "classes", "labels"):
            v = ckpt.get(key, None)
            if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                names = list(v)
                break

    # (3) default
    if not names:
        names = list(CLASS_NAMES_DEFAULT)

    # Make length match num_classes
    if len(names) != int(num_classes):
        print(f"[WARN] class_names length ({len(names)}) != num_classes ({num_classes}).")
        if len(names) > int(num_classes):
            names = names[: int(num_classes)]
        else:
            # pad generic names
            for i in range(len(names), int(num_classes)):
                names.append(f"class_{i}")

    return names


def infer_fall_indices(class_names: List[str]) -> List[int]:
    """
    Auto-detect fall classes by name. This matches your dataset:
    first 5 are falls, but we also support name-based detection.
    """
    fall_idx = []
    for i, n in enumerate(class_names):
        s = n.lower()
        if s.startswith("fall") or "falling" in s:
            fall_idx.append(i)
    return fall_idx


def draw_hud(frame, lines, org=(10, 10), font=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7, thickness=2, pad=8, line_gap=6,
             bg_color=(0, 0, 0), bg_alpha=0.6, text_color=(255, 255, 255)):
    if not lines:
        return frame

    x0, y0 = org
    sizes = [cv2.getTextSize(s, font, font_scale, thickness)[0] for s in lines]
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
        cv2.putText(frame, s, (x0 + pad, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
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


def _interp_1d(x: np.ndarray, mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = x.copy()
    n = len(x)
    i = 0
    while i < n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        gap = j - i
        left_ok = (i - 1) >= 0 and mask[i - 1]
        right_ok = j < n and mask[j]
        if gap <= max_gap and left_ok and right_ok:
            x0 = out[i - 1]
            x1 = out[j]
            for k in range(gap):
                t = (k + 1) / (gap + 1)
                out[i + k] = (1 - t) * x0 + t * x1
        i = j
    return out


def _fill_and_mask_kpts(xy: np.ndarray, conf: np.ndarray, conf_thres: float, max_interp_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    T, K_, _ = xy.shape
    xy = xy.astype(np.float32, copy=False)
    conf = conf.astype(np.float32, copy=False)

    valid = conf > conf_thres  # (T,K)

    xy_filled = xy.copy()
    conf_filled = conf.copy()
    xy_filled[~valid] = 0.0
    conf_filled[~valid] = 0.0

    if max_interp_gap > 0:
        for j in range(K_):
            m = valid[:, j]
            if m.sum() < 2:
                continue
            for c in range(2):
                seq = xy_filled[:, j, c]
                xy_filled[:, j, c] = _interp_1d(seq, m, max_gap=max_interp_gap)

    return xy_filled, conf_filled


def _normalize_xy(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    xy = xy.astype(np.float32, copy=False)
    conf = conf.astype(np.float32, copy=False)
    T = xy.shape[0]
    out = xy.copy()

    for t in range(T):
        m = conf[t] > 0
        if m.sum() == 0:
            continue

        if conf[t, 11] > 0 and conf[t, 12] > 0:
            center = 0.5 * (xy[t, 11] + xy[t, 12])
        else:
            center = xy[t, m].mean(axis=0)

        out[t] = out[t] - center[None, :]

        scale = None
        if (conf[t, 5] > 0 and conf[t, 6] > 0) and (conf[t, 11] > 0 and conf[t, 12] > 0):
            sh = 0.5 * (xy[t, 5] + xy[t, 6])
            hp = 0.5 * (xy[t, 11] + xy[t, 12])
            scale = np.linalg.norm(sh - hp)
        if scale is None or scale < 1e-6:
            pts = out[t, m]
            scale = np.sqrt((pts**2).sum(axis=1).mean())
        if scale < 1e-6:
            scale = 1.0

        out[t] = out[t] / scale

    return out


def _vel(xy: np.ndarray) -> np.ndarray:
    v = np.zeros_like(xy, dtype=np.float32)
    v[1:] = xy[1:] - xy[:-1]
    return v


def _acc(v: np.ndarray) -> np.ndarray:
    a = np.zeros_like(v, dtype=np.float32)
    a[2:] = v[2:] - v[1:-1]
    return a


def _global_feats(xy: np.ndarray) -> np.ndarray:
    com = xy.mean(axis=1)  # (T,2)
    vcom = np.zeros_like(com, dtype=np.float32)
    vcom[1:] = com[1:] - com[:-1]

    xmin = xy[:, :, 0].min(axis=1)
    xmax = xy[:, :, 0].max(axis=1)
    ymin = xy[:, :, 1].min(axis=1)
    ymax = xy[:, :, 1].max(axis=1)
    w = (xmax - xmin).astype(np.float32)
    h = (ymax - ymin).astype(np.float32)

    g = np.stack([vcom[:, 0], vcom[:, 1], w, h], axis=1).astype(np.float32)  # (T,4)
    return g



def unpack_model_output(out):
    """
    Supports:
      - single head: logits
      - two head: (activity_logits, fall_logit)
    """
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]
    return out, None


def get_model(model_name: str, in_features: int, num_classes: int, T_for_mlp: int, node_features: Optional[int] = None) -> torch.nn.Module:
    model_name = model_name.lower().strip()

    if model_name == "tcn":
        model = TCNBaseline(in_features=in_features, num_classes=num_classes, hidden_channels=128, num_blocks=4, kernel_size=3, dropout=0.1)

    elif model_name == "lstm":
        model = LSTMBaseline(in_features=in_features, num_classes=num_classes, hidden_size=128, num_layers=2, dropout=0.1, bidirectional=False, pool="last")

    elif model_name == "gru":
        model = GRUBaseline(in_features=in_features, num_classes=num_classes, hidden_size=128, num_layers=2, dropout=0.1, bidirectional=False, pool="last")

    elif model_name == "gcn":
        if node_features is None:
            raise ValueError("node_features must be provided for GCN.")
        model = GCNBaseline(num_nodes=17, node_features=node_features, num_classes=num_classes, hidden_size=64, dropout=0.1)

    elif model_name == "mlp":
        model = MLPBaseline(T=int(T_for_mlp), in_features=in_features, num_classes=num_classes, hidden_sizes=(256, 128), dropout=0.2)

    elif model_name == "stgcn":
        if node_features is None:
            raise ValueError("node_features must be provided for STGCN.")
        model = STGCNBaseline(num_nodes=17, node_features=node_features, num_classes=num_classes, hidden_channels=128, num_blocks=4, t_kernel=9, dropout=0.1)


    elif model_name == "cnnlstm":
        if CNNLSTMTwoHead is None:
            raise ImportError("CNNLSTMTwoHead not available. Ensure models/cnnlstm/cnn_lstm_two_head.py exists.")
        # Uses keypoint-CNN path if node_features provided, else falls back to MLP-like path
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
        raise ValueError(f"Unknown temporal model: {model_name}")

    return model


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Real-time pose->temporal inference with overlay + video writing.")
    parser.add_argument("--source", type=str, default="0", help='Webcam index (e.g. 0) or video file path.')
    parser.add_argument("--yolo-weights", type=str, default="yolo11l-pose.pt", help="Ultralytics YOLO pose weights.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--max-people", type=int, default=1, help="Max people to consider (use 1 for top person).")

    parser.add_argument("--device", type=str, default=None, help='Device string, e.g. "cuda" or "cpu". Default: auto.')
    parser.add_argument("--half", type=int, default=0, help="Use FP16 on CUDA for speed (0/1).")

    parser.add_argument("--temporal-model", type=str, required=True, choices=["tcn", "lstm", "gru", "gcn", "mlp", "stgcn", "cnnlstm"])
    parser.add_argument("--ckpt-root", type=str, default="models")
    parser.add_argument("--ckpt-subdir", type=str, required=True)
    parser.add_argument("--weights-name", type=str, default=None)

    parser.add_argument("--labels-file", type=str, default=None, help="Optional: path to a txt file with one class name per line.")

    parser.add_argument("--T", type=int, default=0, help="Window length. 0 => use checkpoint T/T_used if present.")
    parser.add_argument("--stride", type=int, default=1, help="Run temporal inference every N frames once buffer full.")

    # If you don't pass fall ids, we'll try to auto-detect fall classes by name.
    parser.add_argument("--fall-class-ids", nargs="+", type=int, default=None,
                        help="Fall class ids in ORIGINAL label space (1-based). Example: --fall-class-ids 1 2 3 4 5")

    parser.add_argument("--fall-threshold", type=float, default=0.5)
    parser.add_argument("--debounce-consecutive", type=int, default=2)
    parser.add_argument("--debounce-m", type=int, default=3)
    parser.add_argument("--debounce-n", type=int, default=2)

    parser.add_argument("--min-window-valid", type=float, default=0.5,
                        help="Skip inference until at least this fraction of frames in the window are valid (0..1). Helps stop early bogus predictions.")

    parser.add_argument("--out", type=str, default="annotated.mp4")
    parser.add_argument("--show", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0, help="Enable debug prints (0/1).")

    # Fallback preprocessing flags if ckpt lacks metadata
    parser.add_argument("--fallback-use-conf", type=int, default=1)
    parser.add_argument("--fallback-normalize", type=int, default=1)
    parser.add_argument("--fallback-add-vel", type=int, default=1)
    parser.add_argument("--fallback-add-acc", type=int, default=1)
    parser.add_argument("--fallback-add-global", type=int, default=1)
    parser.add_argument("--fallback-add-mask", type=int, default=1)
    parser.add_argument("--fallback-conf-thres", type=float, default=0.2)
    parser.add_argument("--fallback-max-interp-gap", type=int, default=5)
    parser.add_argument("--fallback-min-valid-frac", type=float, default=0.3)

    parser.add_argument("--pred-align", type=str, default="auto", choices=["auto", "end", "center"])
    parser.add_argument("--hud-scale", type=float, default=0.7)
    parser.add_argument("--hud-pos", type=str, default="top", choices=["top", "bottom"])

    args = parser.parse_args()

    device = pick_device(args.device)
    args.device = device

    pose_model = YOLO(args.yolo_weights)

    ckpt_path = resolve_ckpt_path(args)
    print("Loading temporal model checkpoint from:", ckpt_path.as_posix())
    ckpt = torch.load(ckpt_path, map_location="cpu")

    has_meta = isinstance(ckpt, dict) and ("state_dict" in ckpt)
    state = ckpt["state_dict"] if has_meta else ckpt

    def _b(key: str, fallback: bool) -> bool:
        return bool(ckpt.get(key, fallback)) if has_meta else bool(fallback)

    if int(args.T) > 0:
        T_final = int(args.T)
    else:
        T_final = int(ckpt.get("T", ckpt.get("T_used", 64))) if has_meta else 64

    use_conf_final = _b("use_conf", bool(args.fallback_use_conf))
    normalize_final = _b("normalize", bool(args.fallback_normalize))
    add_vel_final = _b("add_vel", bool(args.fallback_add_vel))
    add_acc_final = _b("add_acc", bool(args.fallback_add_acc))
    add_global_final = _b("add_global", bool(args.fallback_add_global))
    add_mask_final = _b("add_mask_channel", bool(args.fallback_add_mask))

    # -----------------------------
    # Prediction-to-frame alignment
    # -----------------------------
    # The temporal model consumes a causal window ending at the current frame index.
    # We must decide which frame inside that window the prediction should be displayed on.
    #
    # - align_mode='end'    : prediction is for the window end (frame t), i.e. clip ending at t
    # - align_mode='center' : prediction is for the window center (frame t - T/2)
    #
    # For 'center', we delay display by label_offset frames so that when we show frame i
    # we have already processed up to i+label_offset and computed its prediction.
    label_mode_ckpt = str(ckpt.get("label_mode", "end")) if has_meta else "end"
    if args.pred_align == "end":
        align_mode = "end"
    elif args.pred_align == "center":
        align_mode = "center"
    else:
        align_mode = "center" if label_mode_ckpt in ("center", "hybrid_center_fallpct") else "end"

    label_offset = 0 if align_mode == "end" else (T_final // 2)
    display_delay = int(label_offset)
    conf_thres_final = float(ckpt.get("conf_thres", args.fallback_conf_thres)) if has_meta else float(args.fallback_conf_thres)
    max_interp_gap_final = int(ckpt.get("max_interp_gap", args.fallback_max_interp_gap)) if has_meta else int(args.fallback_max_interp_gap)
    min_valid_frac_final = float(ckpt.get("min_valid_frac", args.fallback_min_valid_frac)) if has_meta else float(args.fallback_min_valid_frac)

    if add_acc_final and not add_vel_final:
        raise RuntimeError("add_acc=True requires add_vel=True.")

    if has_meta:
        in_features_final = int(ckpt["in_features"])
        num_classes_final = int(ckpt["num_classes"])
    else:
        cj = 2
        if use_conf_final: cj += 1
        if add_vel_final: cj += 2
        if add_acc_final: cj += 2
        if add_global_final: cj += 4
        if add_mask_final: cj += 1
        in_features_final = int(K * cj)
        num_classes_final = 11

    # Load class names in the right order
    CLASS_NAMES = load_class_names(num_classes_final, ckpt, has_meta, args.labels_file)

    # Fall indices
    if args.fall_class_ids:
        fall_ids_0based = [(int(x) - 1) for x in args.fall_class_ids]
        fall_idx = [i for i in fall_ids_0based if 0 <= int(i) < int(num_classes_final)]
        if len(fall_idx) == 0:
            print("[WARN] --fall-class-ids provided but none are within [1..num_classes]. Disabling P(fall).")
    else:
        fall_idx = infer_fall_indices(CLASS_NAMES)

    fall_hist = deque(maxlen=max(1, int(args.debounce_m)))
    fall_consec = 0
    last_p_fall: Optional[float] = None
    last_debounced: Optional[bool] = None

    node_features_final = int(in_features_final // 17) if (in_features_final % 17 == 0) else None

    model = get_model(
        args.temporal_model,
        in_features=in_features_final,
        num_classes=num_classes_final,
        T_for_mlp=T_final,
        node_features=node_features_final,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    model = model.to(device)

    use_half = (int(args.half) == 1) and ("cuda" in device)
    if use_half:
        model = model.half()

    # Warm up the temporal model once to avoid a slow first prediction (CUDA kernel/JIT init).
    try:
        with torch.no_grad():
            dummy = torch.zeros((1, T_final, in_features_final), device=device)
            dummy = dummy.half() if use_half else dummy.float()
            _ = model(dummy)
    except Exception as e:
        print(f"[WARN] temporal-model warmup failed (continuing): {e}")

    src = args.source
    if src.isdigit():
        src = int(src)

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps is None or src_fps <= 1e-3:
        src_fps = 30.0

    out_path = str(Path(args.out))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(src_fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter: {out_path}")

    xy_buf = deque(maxlen=T_final)
    cf_buf = deque(maxlen=T_final)

    # Frames are buffered so we can display them with the correct aligned prediction.
    # Each entry: (frame_idx, frame_bgr, found, kpts_xy, kpts_conf)
    display_q = deque()

    # Cache predictions by the frame index they should be displayed on.
    # Value: (pred_idx, pred_conf, p_fall, debounced, p_fall_source)
    pred_cache = {}
    last_pred = None  # fallback when a frame has no cached prediction (eg. stride > 1)

    # During startup we temporarily ignore --stride until we have a prediction for the first displayable frame.
    startup_ready = False

    fps_ema = None
    ema_alpha = 0.1
    t_prev = time.perf_counter()
    frame_idx = 0
    stride = max(1, int(args.stride))

    xy_zeros = np.zeros((K, 2), dtype=np.float32)
    cf_zeros = np.zeros((K,), dtype=np.float32)

    preproc_tag = []
    preproc_tag.append("conf" if use_conf_final else "no-conf")
    preproc_tag.append("norm" if normalize_final else "no-norm")
    preproc_tag.append("vel" if add_vel_final else "no-vel")
    preproc_tag.append("acc" if add_acc_final else "no-acc")
    preproc_tag.append("glob" if add_global_final else "no-glob")
    preproc_tag.append("mask" if add_mask_final else "no-mask")
    preproc_str = ",".join(preproc_tag)

    # Optional lightweight logging to validate timing/alignment.
    # With --debug, we print (occasionally) the displayed frame index and the index the prediction was produced from.
    def _stack_left_pad(buf: deque, T: int, pad: np.ndarray) -> np.ndarray:
        """Return a (T, ...) array built from buf, left-padding with the oldest element if needed."""
        if len(buf) == 0:
            return np.stack([pad] * T, axis=0)
        if len(buf) >= T:
            return np.stack(list(buf)[-T:], axis=0)
        pad_elem = buf[0]
        pads = [pad_elem] * (T - len(buf))
        return np.stack(pads + list(buf), axis=0)

    def _run_temporal_infer(end_idx: int, warmup: bool):
        """Run temporal inference on a window ending at end_idx (current frame index)."""
        nonlocal fall_consec, last_p_fall, last_debounced

        # Build (T,K,2) and (T,K) sequences. If we're still warming up (<T frames seen),
        # left-pad with the oldest available frame to avoid 'no prediction' on startup.
        xy_seq = _stack_left_pad(xy_buf, T_final, xy_zeros)
        cf_seq = _stack_left_pad(cf_buf, T_final, cf_zeros)

        xy_filled, cf_filled = _fill_and_mask_kpts(
            xy_seq, cf_seq,
            conf_thres=conf_thres_final,
            max_interp_gap=max_interp_gap_final,
        )

        xy_used = _normalize_xy(xy_filled, cf_filled) if normalize_final else xy_filled.astype(np.float32, copy=False)

        feats = [xy_used]
        if use_conf_final:
            feats.append(cf_filled[..., None])

        if add_vel_final:
            v = _vel(xy_used)
            feats.append(v)

        if add_acc_final:
            v = _vel(xy_used)
            a = _acc(v)
            feats.append(a)

        if add_global_final:
            g = _global_feats(xy_used)
            gk = np.repeat(g[:, None, :], repeats=K, axis=1)
            feats.append(gk)

        Xf = np.concatenate(feats, axis=-1).astype(np.float32, copy=False)

        frac_valid = (cf_filled > conf_thres_final).mean(axis=1)  # (T,)
        frame_valid = (frac_valid >= min_valid_frac_final).astype(np.float32)  # (T,)

        # If we're still warming up, don't suppress inference (user wants predictions from frame 0).
        min_window_valid_eff = 0.0 if warmup else float(args.min_window_valid)
        window_valid_ratio = float(frame_valid.mean())
        if window_valid_ratio < min_window_valid_eff:
            return None

        Xf = Xf * frame_valid[:, None, None]

        if add_mask_final:
            m = np.repeat(frame_valid[:, None, None], repeats=K, axis=1)
            Xf = np.concatenate([Xf, m], axis=-1)

        window = Xf.reshape(T_final, -1)
        if window.shape[1] != in_features_final:
            return None

        X = torch.from_numpy(window).unsqueeze(0).to(device)
        X = X.half() if use_half else X.float()

        with torch.no_grad():
            out = model(X)
            activity_logits, fall_logit = unpack_model_output(out)

            # Robust: if activity logits is (B,T,C), pool over T
            if hasattr(activity_logits, "ndim") and activity_logits.ndim == 3:
                activity_logits = activity_logits.mean(dim=1)

            probs = torch.softmax(activity_logits, dim=1)
            pred_idx = int(probs.argmax(dim=1).item())
            pred_conf = float(probs.max(dim=1).values.item())

            # Binary fall score
            p_fall = None
            p_fall_source = None
            if fall_logit is not None:
                # fall_logit may be (B,), (B,1) or (B,T); reduce to (B,)
                fl = fall_logit.view(fall_logit.shape[0], -1).mean(dim=1)
                p_fall = float(torch.sigmoid(fl)[0].item())
                p_fall_source = "fall_head"
            elif len(fall_idx) > 0:
                p_fall = float(probs[0, fall_idx].sum().item())
                p_fall_source = "activity_softmax"

            debounced = None
            if p_fall is not None:
                is_fall = (p_fall >= float(args.fall_threshold))

                fall_hist.append(1 if is_fall else 0)
                fall_consec = (fall_consec + 1) if is_fall else 0

                debounced_now = False
                if int(args.debounce_consecutive) > 1 and fall_consec >= int(args.debounce_consecutive):
                    debounced_now = True
                if int(args.debounce_n) > 0 and sum(fall_hist) >= int(args.debounce_n):
                    debounced_now = True

                last_p_fall = p_fall
                last_debounced = bool(debounced_now)
                debounced = bool(debounced_now)
            else:
                last_p_fall = None
                last_debounced = None

        return (pred_idx, pred_conf, p_fall, debounced, p_fall_source)

    stop = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = pose_model.predict(
            source=frame,
            imgsz=int(args.imgsz),
            conf=float(args.conf_thres),
            verbose=False,
            device=device,
            half=use_half,
        )

        found = False
        kpts_xy = xy_zeros
        kpts_conf = cf_zeros

        if results and len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints
            xy_all = kpts.xy.cpu().numpy() if hasattr(kpts.xy, "cpu") else np.array(kpts.xy)
            cf_all = kpts.conf.cpu().numpy() if hasattr(kpts.conf, "cpu") else np.array(kpts.conf)

            if xy_all.ndim == 3 and xy_all.shape[0] > 0:
                scores = cf_all.sum(axis=1)
                best = int(np.argmax(scores))
                kpts_xy = xy_all[best].astype(np.float32)
                kpts_conf = cf_all[best].astype(np.float32)
                found = True

        # Append for later display (we may delay display for alignment).
        display_q.append((frame_idx, frame.copy(), found, kpts_xy.copy(), kpts_conf.copy()))

        # Append to temporal buffers (causal: window ends at current frame_idx).
        xy_buf.append(kpts_xy if found else xy_zeros)
        cf_buf.append(kpts_conf if found else cf_zeros)

        warmup = (len(xy_buf) < T_final)

        # Inference scheduling:
        # - During startup, force inference each frame until we have a prediction for the first displayable frame.
        # - After startup, follow --stride (run every N frames).
        force_infer = not startup_ready
        do_infer = force_infer or (frame_idx % stride == 0) or (frame_idx == 0)
        if do_infer:
            pred = _run_temporal_infer(end_idx=frame_idx, warmup=warmup)
            if pred is not None:
                pred_idx, pred_conf, p_fall, debounced, p_fall_source = pred
                target_idx = int(frame_idx) if align_mode == "end" else int(frame_idx - label_offset)
                if target_idx < 0:
                    target_idx = 0
                pred_cache[target_idx] = pred
                last_pred = pred

        # FPS estimate (processing FPS)
        t_now = time.perf_counter()
        dt = max(1e-6, t_now - t_prev)
        inst_fps = 1.0 / dt
        t_prev = t_now
        fps_ema = inst_fps if fps_ema is None else (ema_alpha * inst_fps + (1 - ema_alpha) * fps_ema)

        # Decide when we can start emitting frames: we want predictions available from the first displayed frame.
        if not startup_ready and len(display_q) > display_delay:
            first_idx = int(display_q[0][0])
            # In end-mode we also accept last_pred (because it targets the same frame index),
            # but in center-mode we require a cached prediction for that frame to avoid 'future labels on past frames'.
            if align_mode == "end":
                startup_ready = (first_idx in pred_cache) or (first_idx == frame_idx and last_pred is not None)
            else:
                startup_ready = (first_idx in pred_cache)

        # Safety: don't deadlock output if predictions can't be produced (eg. feature mismatch / no detections).
        if not startup_ready and frame_idx >= (display_delay + T_final):
            startup_ready = True

        # Emit as many frames as possible while keeping the desired display_delay.
        while startup_ready and len(display_q) > display_delay:
            out_idx, out_frame, out_found, out_xy, out_conf = display_q.popleft()

            # Pick the prediction that belongs to this specific displayed frame index.
            pred = pred_cache.pop(int(out_idx), None)
            if pred is not None:
                last_pred = pred
            else:
                pred = last_pred

            if out_found:
                out_frame = draw_pose(out_frame, out_xy, out_conf, conf_thres=conf_thres_final, draw_skeleton=True)

            lines = []
            if pred is not None:
                pred_idx, pred_conf, p_fall, debounced, p_fall_source = pred
                if 0 <= int(pred_idx) < len(CLASS_NAMES):
                    activity = CLASS_NAMES[int(pred_idx)]
                else:
                    activity = f"class_{int(pred_idx)}"
                lines.append(f"{activity} (argmax p={pred_conf:.2f})")
                if p_fall is not None:
                    lines.append(f"P(fall)={float(p_fall):.2f} (src={p_fall_source or 'n/a'})  thr={float(args.fall_threshold):.2f}  debounced={int(debounced or 0)}")
            else:
                lines.append("No prediction yet")

            # Explicit warmup note for early frames where we had to left-pad history.
            # Prediction for frame i is produced when the window ends at:
            #   end_idx = i            (end) or i + label_offset (center)
            end_idx_for_this_frame = int(out_idx) if align_mode == "end" else int(out_idx + label_offset)
            if end_idx_for_this_frame < (T_final - 1):
                lines.append(f"Warmup (padded history): end_idx={end_idx_for_this_frame} < {T_final-1}")

            lines.append(f"FPS: {fps_ema:.1f} | T={T_final} | align={align_mode} | display_delay={display_delay} | {preproc_str}")

            hud_x = 10
            hud_y = 10 if args.hud_pos == "top" else max(10, out_frame.shape[0] - 110)
            out_frame = draw_hud(out_frame, lines, org=(hud_x, hud_y), font_scale=float(args.hud_scale))

            if int(args.debug) == 1 and (int(out_idx) % int(max(1, round(src_fps))) == 0):
                src_end = int(out_idx) if align_mode == "end" else int(out_idx + label_offset)
                print(f"[debug] display frame={int(out_idx)} | pred_from_window_end={src_end} | align={align_mode}")

            writer.write(out_frame)
            if int(args.show) == 1:
                cv2.imshow("realtime_infer", out_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    stop = True
                    break

            # Prevent unbounded growth in pred_cache if display stalls for any reason.
            # Keep only a small tail.
            min_keep = int(out_idx) - (2 * T_final + 50)
            if min_keep > 0:
                old_keys = [k for k in pred_cache.keys() if int(k) < min_keep]
                for k in old_keys:
                    pred_cache.pop(k, None)

        if stop:
            break

        frame_idx += 1

    # Flush remaining delayed frames (eg. center alignment at end of file).
    startup_ready = True
    while len(display_q) > 0 and not stop:
        out_idx, out_frame, out_found, out_xy, out_conf = display_q.popleft()

        pred = pred_cache.pop(int(out_idx), None)
        if pred is not None:
            last_pred = pred
        else:
            pred = last_pred

        if out_found:
            out_frame = draw_pose(out_frame, out_xy, out_conf, conf_thres=conf_thres_final, draw_skeleton=True)

        lines = []
        if pred is not None:
            pred_idx, pred_conf, p_fall, debounced, p_fall_source = pred
            activity = CLASS_NAMES[int(pred_idx)] if 0 <= int(pred_idx) < len(CLASS_NAMES) else f"class_{int(pred_idx)}"
            lines.append(f"{activity} (argmax p={pred_conf:.2f})")
            if p_fall is not None:
                lines.append(f"P(fall)={float(p_fall):.2f} (src={p_fall_source or 'n/a'})  thr={float(args.fall_threshold):.2f}  debounced={int(debounced or 0)}")
        else:
            lines.append("No prediction")

        lines.append(f"FPS: {fps_ema:.1f} | T={T_final} | align={align_mode} | display_delay={display_delay} | {preproc_str}")

        hud_x = 10
        hud_y = 10 if args.hud_pos == "top" else max(10, out_frame.shape[0] - 110)
        out_frame = draw_hud(out_frame, lines, org=(hud_x, hud_y), font_scale=float(args.hud_scale))

        writer.write(out_frame)
        if int(args.show) == 1:
            cv2.imshow("realtime_infer", out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    writer.release()
    if int(args.show) == 1:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
