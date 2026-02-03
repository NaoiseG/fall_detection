#!/usr/bin/env python3
"""
Offline video -> YOLOv11 pose -> MotionBERT ActionNet (LITE) inference.

This script:
1) Runs YOLOv11 pose on a video
2) Builds a MotionBERT action .pkl (same schema as prepare_motionbert_dataset.py)
3) Loads MotionBERT ActionNet checkpoint (best_epoch.bin)
4) Runs per-window predictions and saves a CSV


Optional flags you’ll probably care about

--config (defaults to MB_ft_UPFall_xsub_LITE.yaml)
--yolo-weights (defaults to yolo11l-pose.pt)
--win-len (defaults to config.clip_len)
--win-step (default 16)
--pad-tail (pad short tail windows instead of dropping)
--out-pkl (default motionbert_video.pkl)
--out-csv (default motionbert_video_preds.csv)
--labels-file (override class names)
--keep-empty-windows (if you want to keep empty/degenerate windows)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# MotionBERT imports (add repo root for MotionBERT to sys.path)
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_MB_ROOT = _REPO_ROOT / "models" / "MotionBERT"
if str(_MB_ROOT) not in sys.path:
    sys.path.insert(0, str(_MB_ROOT))

from models.MotionBERT.lib.utils.tools import get_config, read_pkl  # noqa: E402
from models.MotionBERT.lib.utils.learning import load_backbone  # noqa: E402
from models.MotionBERT.lib.model.model_action import ActionNet  # noqa: E402
from models.MotionBERT.lib.data.dataset_action import make_cam, coco2h36m, human_tracking  # noqa: E402
from models.MotionBERT.lib.utils.utils_data import crop_scale, resample  # noqa: E402

# -----------------------------------------------------------------------------
# Labels
# -----------------------------------------------------------------------------
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

FALL_CLASS_IDS_DEFAULT = [0, 1, 2, 3, 4]

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


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def pick_device(device: Optional[str]) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    d = device.lower()
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


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

    def compute_num_windows(T_total: int) -> int:
        if pad_tail:
            if T_total <= 0:
                return 0
            return int((max(0, T_total - 1)) // win_step + 1)
        if T_total < win_len:
            return 0
        return int((T_total - win_len) // win_step + 1)

    n_wins = compute_num_windows(T_total)
    if n_wins == 0:
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

        raw_kxy = kpts_xy[start:min(end, T_total)].astype(np.float32)
        raw_ksc = kpts_conf[start:min(end, T_total)].astype(np.float32)

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

        if ((ksc < 0).any() or (ksc > 1).any()):
            ksc = np.clip(ksc, 0.0, 1.0)

        interpolate_missing_joints_inplace(kxy, ksc, missing_conf_thres=missing_conf_thres)

        if drop_empty_windows:
            if np.all(ksc <= missing_conf_thres):
                continue
            if (np.ptp(kxy[..., 0]) < 1e-6) and (np.ptp(kxy[..., 1]) < 1e-6):
                continue

        keypoint = kxy[None, ...].astype(np.float32)
        keypoint_score = ksc[None, ...].astype(np.float32)

        annotations.append({
            "frame_dir": frame_dir,
            "total_frames": int(win_len),
            "img_shape": (int(img_shape[0]), int(img_shape[1])),
            "keypoint": keypoint,
            "keypoint_score": keypoint_score,
            "label": 0,
        })
        split_list.append(frame_dir)

        if (not pad_tail) and end >= T_total:
            break

    return split_list, annotations


def build_motion_from_annotation(
    ann: dict,
    clip_len: int,
    scale_range: Optional[List[float]],
) -> np.ndarray:
    keypoint = ann["keypoint"]  # (1, T, 17, 2)
    keypoint_score = ann["keypoint_score"]  # (1, T, 17)
    img_shape = ann["img_shape"]

    resample_id = resample(ori_len=ann["total_frames"], target_len=clip_len, randomness=False)
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
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--labels-file not found: {p.as_posix()}")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=str, required=True, help="Directory containing best_epoch.bin")
    ap.add_argument("--config", type=str, default="models/MotionBERT/configs/action/MB_ft_UPFall_xsub_LITE.yaml")
    ap.add_argument("--video", type=str, required=True, help="Path to input mp4")
    ap.add_argument("--yolo-weights", type=str, default="yolo11l-pose.pt")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf-thres", type=float, default=0.25)
    ap.add_argument("--win-len", type=int, default=None, help="Window length (defaults to config.clip_len)")
    ap.add_argument("--win-step", type=int, default=16)
    ap.add_argument("--pad-tail", action="store_true")
    ap.add_argument("--missing-conf-thres", type=float, default=0.0)
    ap.add_argument("--keep-empty-windows", action="store_true", default=False)
    ap.add_argument("--out-pkl", type=str, default="outputs/motionbert_video.pkl")
    ap.add_argument("--out-csv", type=str, default="outputs/motionbert_video_preds.csv")
    ap.add_argument("--labels-file", type=str, default=None)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--display", action="store_true", help="Display video with pose + current window prediction")
    ap.add_argument("--display-conf-thres", type=float, default=0.2, help="Keypoint conf threshold for drawing")
    ap.add_argument("--display-fps", type=float, default=None, help="Playback FPS for display (default: video FPS)")
    ap.add_argument("--no-merge-fall", action="store_true", help="Disable merging the first five fall labels into one class")
    args = ap.parse_args()

    device = pick_device(args.device)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = ckpt_dir / "best_epoch.bin"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.as_posix()}")

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path.as_posix()}")

    cfg = get_config(args.config)
    clip_len = int(args.win_len) if args.win_len is not None else int(getattr(cfg, "clip_len", 64))

    # ------------------------------------------------------------------
    # 1) YOLOv11 pose extraction
    # ------------------------------------------------------------------
    pose_model = YOLO(args.yolo_weights)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path.as_posix()}")

    frames_xy: List[np.ndarray] = []
    frames_cf: List[np.ndarray] = []
    img_shape = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if img_shape is None:
            h, w = frame.shape[:2]
            img_shape = (int(h), int(w))

        results = pose_model.predict(
            source=frame,
            imgsz=int(args.imgsz),
            conf=float(args.conf_thres),
            verbose=False,
            device=device,
        )

        kpts_xy = np.zeros((17, 2), dtype=np.float32)
        kpts_conf = np.zeros((17,), dtype=np.float32)

        if results and len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints
            xy_all = kpts.xy.cpu().numpy() if hasattr(kpts.xy, "cpu") else np.array(kpts.xy)
            cf_all = kpts.conf.cpu().numpy() if hasattr(kpts.conf, "cpu") else np.array(kpts.conf)

            if xy_all.ndim == 3 and xy_all.shape[0] > 0:
                scores = cf_all.sum(axis=1)
                best = int(np.argmax(scores))
                kpts_xy = xy_all[best].astype(np.float32)
                kpts_conf = cf_all[best].astype(np.float32)

        frames_xy.append(kpts_xy)
        frames_cf.append(kpts_conf)

        frame_idx += 1
        if args.limit_frames is not None and frame_idx >= int(args.limit_frames):
            break

    cap.release()

    if img_shape is None:
        raise RuntimeError("No frames read from video.")

    kpts_xy = np.stack(frames_xy, axis=0)  # (T,17,2)
    kpts_conf = np.stack(frames_cf, axis=0)  # (T,17)

    # ------------------------------------------------------------------
    # 2) Build and save MotionBERT action pkl
    # ------------------------------------------------------------------
    split_list, annotations = build_windows(
        kpts_xy=kpts_xy,
        kpts_conf=kpts_conf,
        img_shape=img_shape,
        win_len=clip_len,
        win_step=int(args.win_step),
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

    use_dp = (device.startswith("cuda") and torch.cuda.device_count() > 1)
    if use_dp:
        model = nn.DataParallel(model)
    model = model.to(device)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = _clean_state_dict_for_model(state, model)
    model.load_state_dict(state, strict=True)
    model.eval()

    num_classes = int(getattr(cfg, "action_classes", 11))
    merge_fall = not bool(args.no_merge_fall)
    labels_file_names = load_labels_file(args.labels_file)

    merged_len_expected = 1 + (len(CLASS_NAMES_DEFAULT) - len(FALL_CLASS_IDS_DEFAULT))

    if merge_fall:
        # If labels file already looks merged (e.g., 7 classes), use it directly.
        if labels_file_names is not None and len(labels_file_names) in (merged_len_expected, num_classes):
            merged_class_names = list(labels_file_names)
        else:
            raw_names = list(labels_file_names) if labels_file_names is not None else list(CLASS_NAMES_DEFAULT)
            merged_class_names = ["Fall"] + [raw_names[i] for i in range(len(raw_names)) if i not in FALL_CLASS_IDS_DEFAULT]
    else:
        base = list(labels_file_names) if labels_file_names is not None else list(CLASS_NAMES_DEFAULT)
        merged_class_names = pad_or_trim(base, num_classes)

    fall_idx = infer_fall_indices(merged_class_names)

    # ------------------------------------------------------------------
    # 4) Run per-window inference
    # ------------------------------------------------------------------
    dataset_loaded = read_pkl(str(out_pkl))
    split = dataset_loaded["split"]["xsub_val"]
    ann_map = {ann["frame_dir"]: ann for ann in dataset_loaded["annotations"]}

    scale_range = getattr(cfg, "scale_range_test", None)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    preds_for_display = []

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_dir",
            "start_frame",
            "end_frame",
            "pred_id",
            "pred_name",
            "pred_conf",
            "p_fall",
        ])

        for frame_dir in split:
            ann = ann_map[frame_dir]
            motion = build_motion_from_annotation(ann, clip_len=clip_len, scale_range=scale_range)

            X = torch.from_numpy(motion).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(X)
                probs_t = torch.softmax(out, dim=1).squeeze(0).detach().cpu().numpy()

            # Merge first five fall classes into a single "Fall" class for display/output.
            if merge_fall:
                if probs_t.shape[0] == len(merged_class_names):
                    merged_probs = probs_t
                    fall_prob = float(merged_probs[0]) if len(merged_probs) > 0 else 0.0
                elif probs_t.shape[0] == len(CLASS_NAMES_DEFAULT):
                    fall_prob = float(np.sum(probs_t[FALL_CLASS_IDS_DEFAULT]))
                    nonfall_probs = [probs_t[i] for i in range(len(probs_t)) if i not in FALL_CLASS_IDS_DEFAULT]
                    merged_probs = np.array([fall_prob] + nonfall_probs, dtype=np.float32)
                else:
                    merged_probs = probs_t
                    fall_prob = float(np.sum(probs_t[FALL_CLASS_IDS_DEFAULT])) if probs_t.shape[0] > max(FALL_CLASS_IDS_DEFAULT) else 0.0
            else:
                merged_probs = probs_t
                fall_prob = float(np.sum(probs_t[fall_idx])) if fall_idx else 0.0

            pred_id = int(np.argmax(merged_probs))
            pred_conf = float(np.max(merged_probs))
            p_fall = float(fall_prob)

            # parse window start from frame_dir
            start_frame = 0
            end_frame = ann["total_frames"] - 1
            try:
                parts = frame_dir.split("_s")
                if len(parts) > 1:
                    start_frame = int(parts[1].split("_len")[0])
                    end_frame = start_frame + int(ann["total_frames"]) - 1
            except Exception:
                pass

            pred_name = merged_class_names[pred_id] if 0 <= pred_id < len(merged_class_names) else str(pred_id)

            writer.writerow([
                frame_dir,
                start_frame,
                end_frame,
                pred_id,
                pred_name,
                f"{pred_conf:.6f}",
                f"{p_fall:.6f}",
            ])
            preds_for_display.append({
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "pred_id": int(pred_id),
                "pred_name": str(pred_name),
                "pred_conf": float(pred_conf),
                "p_fall": float(p_fall),
            })

    print(f"Saved pkl: {out_pkl.as_posix()}")
    print(f"Saved predictions: {out_csv.as_posix()}")
    print(f"Windows: {len(split)}")

    # ------------------------------------------------------------------
    # 5) Optional display with pose + current window prediction
    # ------------------------------------------------------------------
    if args.display:
        preds_for_display.sort(key=lambda x: x["start_frame"])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for display: {video_path.as_posix()}")

        cv2.namedWindow("MotionBERT Inference", cv2.WINDOW_NORMAL)
        cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        display_fps = float(args.display_fps) if args.display_fps is not None else cap_fps
        if not np.isfinite(display_fps) or display_fps <= 0.0:
            display_fps = 30.0
        delay_ms = max(1, int(1000.0 / display_fps))
        print(f"[Display] Playing at ~{display_fps:.2f} FPS. Press 'q' or Esc to quit.", flush=True)

        pred_idx = 0
        current_pred = None
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx < len(kpts_xy):
                frame = draw_pose(
                    frame,
                    kpts_xy[frame_idx],
                    kpts_conf[frame_idx],
                    conf_thres=float(args.display_conf_thres),
                    draw_skeleton=True,
                )

            while pred_idx < len(preds_for_display) and preds_for_display[pred_idx]["start_frame"] <= frame_idx:
                current_pred = preds_for_display[pred_idx]
                pred_idx += 1

            if current_pred is not None and frame_idx > current_pred["end_frame"]:
                current_pred = None

            lines = []
            if current_pred is not None:
                lines.append(f"Pred: {current_pred['pred_name']} ({current_pred['pred_conf']:.2f})")
                lines.append(f"p_fall: {current_pred['p_fall']:.2f}")
                lines.append(f"Window: {current_pred['start_frame']}-{current_pred['end_frame']}")
            else:
                lines.append("Pred: (warming up)")

            frame = draw_hud(frame, lines)
            cv2.imshow("MotionBERT Inference", frame)

            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q") or key == 27:
                break

            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
