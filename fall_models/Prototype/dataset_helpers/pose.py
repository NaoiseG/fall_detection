import json
import os
import glob
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import torch

import cv2
import numpy as np
from ultralytics import YOLO


# ----------------------------- CONFIG -----------------------------

@dataclass
class PoseExportConfig:
    model_path: str = "pose_models/ultralytics/yolo11l-pose.pt"
    imgsz: Optional[float] = None
    conf_thres: float = 0.25
    conf_min: float = 0.75
    fps: int = 30
    max_people: int = 1
    detector_max_det: int = 10
    num_kpts: int = 17               # COCO keypoints for Ultralytics pose models
    video_codec: str = "mp4v"        # mp4v is widely supported
    save_csv: bool = False
    max_jump_px: Optional[float] = None
    max_jump_diag_frac: float = 0.25
    max_lost: int = 10
    switch_margin_px: float = 20.0
    reset_on_max_lost: bool = False
    lock_first_target: bool = True
    min_iou_same_track: float = 0.1
    max_box_area_ratio: float = 2.0
    strict_reacquire: bool = True
    target_x_frac: float = 0.5
    target_y_frac: float = 0.5
    draw_kpt_threshold: float = 0.30
    draw_no_target_text: bool = True
    no_suspicious: bool = False
    allow_region1_start: bool = False
    allow_region2_start: bool = False
    suspicious_conf_thres: float = 0.30
    suspicious_start_frames: int = 100
    suspicious_region1_xyxy: Tuple[float, float, float, float] = (540.0, 160.0, 660.0, 230.0)
    suspicious_region2_xyxy: Tuple[float, float, float, float] = (260.0, 100.0, 430.0, 190.0)
    suspicious_region3_xyxy: Tuple[float, float, float, float] = (465.0, 105.0, 515.0, 190.0)
    suspicious_switch_min_iou: float = 0.35
    suspicious_switch_max_jump_frac: float = 0.25
    prefer_foreground_on_acquire: bool = False
    acquire_min_box_area_ratio: float = 0.35
    acquire_bottom_margin_px: float = 60.0
    lock_delay_frames: int = 0


POSE_LOCK_SETTINGS_PRESETS: Dict[str, Dict[str, Any]] = {
    # Uses PoseExportConfig defaults.
    "default": {},
    # Preserves the settings that were previously hard-coded in
    # dataset_helpers/get_keypoints_files.py.
    "strict_lock": {
        "conf_thres": 0.01,
        "conf_min": 0.01,
        "detector_max_det": 10,
        "max_jump_px": None,
        "max_jump_diag_frac": 0.12,
        "max_lost": 60,
        "switch_margin_px": 9999.0,
        "reset_on_max_lost": False,
        "lock_first_target": True,
        "strict_reacquire": True,
        "min_iou_same_track": 0.05,
        "max_box_area_ratio": 2.5,
        "target_x_frac": 0.5,
        "target_y_frac": 0.5,
    },
}


def pose_lock_settings_choices() -> Tuple[str, ...]:
    return tuple(sorted(POSE_LOCK_SETTINGS_PRESETS))


def apply_pose_lock_settings(config: PoseExportConfig, preset_name: str) -> PoseExportConfig:
    preset_key = str(preset_name).strip().lower()
    if preset_key not in POSE_LOCK_SETTINGS_PRESETS:
        choices = ", ".join(sorted(POSE_LOCK_SETTINGS_PRESETS))
        raise ValueError(f"Unknown pose lock settings preset '{preset_name}'. Choices: {choices}")

    for field_name, value in POSE_LOCK_SETTINGS_PRESETS[preset_key].items():
        setattr(config, field_name, value)
    return config


def extract_model_stride(model: YOLO) -> int:
    stride = getattr(getattr(model, "model", None), "stride", 32)
    if torch.is_tensor(stride):
        return max(1, int(stride.max().item()))
    if isinstance(stride, (list, tuple)):
        numeric = []
        for value in stride:
            try:
                numeric.append(int(value))
            except (TypeError, ValueError):
                continue
        if numeric:
            return max(1, max(numeric))
    try:
        return max(1, int(stride))
    except (TypeError, ValueError):
        return 32


def resolve_predict_imgsz(
    imgsz_value: Optional[float],
    frame_shape: Tuple[int, int],
    stride: int,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "mode": "default",
        "requested_value": None,
        "applied_hw": None,
        "applied_ratio": 1.0,
    }
    if imgsz_value is None:
        return None, info

    value = float(imgsz_value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"imgsz must be positive, got {imgsz_value!r}")

    frame_h, frame_w = frame_shape
    frame_area = float(frame_h * frame_w)

    if value <= 1.0:
        target_ratio = value
        aspect = float(frame_w) / float(frame_h)
        max_h = max(stride, (frame_h // stride) * stride)
        max_w = max(stride, (frame_w // stride) * stride)
        h_candidates = list(range(stride, max_h + 1, stride))
        w_candidates = list(range(stride, max_w + 1, stride))

        best_hw: Optional[Tuple[int, int]] = None
        best_score: Optional[Tuple[float, int, float, float]] = None
        for cand_h in h_candidates:
            for cand_w in w_candidates:
                cand_ratio = float(cand_h * cand_w) / frame_area
                area_err = abs(cand_ratio - target_ratio)
                overshoot = 1 if cand_ratio > target_ratio + 1e-9 else 0
                aspect_err = abs((float(cand_w) / float(cand_h)) - aspect)
                score = (area_err, overshoot, aspect_err, -float(cand_h * cand_w))
                if best_score is None or score < best_score:
                    best_score = score
                    best_hw = (cand_h, cand_w)

        if best_hw is None:
            raise RuntimeError(
                f"Could not resolve a valid imgsz for frame shape {(frame_h, frame_w)} and stride {stride}."
            )

        applied_ratio = float(best_hw[0] * best_hw[1]) / frame_area
        info.update(
            {
                "mode": "pixel_ratio",
                "requested_value": target_ratio,
                "applied_hw": best_hw,
                "applied_ratio": applied_ratio,
            }
        )
        return list(best_hw), info

    explicit_square = int(round(value))
    if explicit_square <= 0:
        raise ValueError(f"imgsz must be positive, got {imgsz_value!r}")

    info.update(
        {
            "mode": "square",
            "requested_value": explicit_square,
            "applied_hw": (explicit_square, explicit_square),
            "applied_ratio": float(explicit_square * explicit_square) / frame_area,
        }
    )
    return explicit_square, info


# ----------------------------- SORTING -----------------------------

def frame_time_key(path: str) -> pd.Timestamp:
    return parse_frame_timestamp(path)

def list_frames(frames_dir: str, pattern: str = "*.png") -> List[str]:
    paths = glob.glob(os.path.join(frames_dir, pattern))
    paths = sorted(paths, key=frame_time_key)
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir} matching {pattern}")
    return paths

FRAME_TS_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T(?P<h>\d{2})_(?P<m>\d{2})_(?P<s>\d{2}(?:\.\d+)?)"
)

def parse_frame_timestamp(path: str) -> pd.Timestamp:
    name = os.path.splitext(os.path.basename(path))[0]
    m = FRAME_TS_RE.search(name)
    if not m:
        raise ValueError(f"Cannot parse timestamp from {path}")
    dt = f"{m.group('date')}T{m.group('h')}:{m.group('m')}:{m.group('s')}"
    return pd.to_datetime(dt)


# ----------------------------- IO HELPERS -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_video_writer(out_path: str, fps: int, frame_size: Tuple[int, int], codec: str = "mp4v") -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    w, h = frame_size
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img


def letterbox_image_to_shape(
    image: np.ndarray,
    new_shape: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    orig_h, orig_w = image.shape[:2]
    new_h, new_w = int(new_shape[0]), int(new_shape[1])
    if new_h <= 0 or new_w <= 0:
        raise ValueError(f"new_shape must be positive, got {new_shape!r}")

    ratio = min(float(new_h) / float(orig_h), float(new_w) / float(orig_w))
    new_unpad_w = int(round(float(orig_w) * ratio))
    new_unpad_h = int(round(float(orig_h) * ratio))
    dw = float(new_w - new_unpad_w) / 2.0
    dh = float(new_h - new_unpad_h) / 2.0

    resized = image
    if (orig_w, orig_h) != (new_unpad_w, new_unpad_h):
        resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    boxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return boxed, {"ratio": ratio, "pad_left": left, "pad_top": top}


def scale_pose_from_letterbox(
    pose: Dict[str, Any],
    ratio: float,
    pad_left: int,
    pad_top: int,
    orig_shape: Tuple[int, int],
) -> Dict[str, Any]:
    orig_h, orig_w = orig_shape
    scaled = dict(pose)

    xy = np.asarray(pose["xy"], dtype=np.float32).copy()
    xy[..., 0] = (xy[..., 0] - float(pad_left)) / float(ratio)
    xy[..., 1] = (xy[..., 1] - float(pad_top)) / float(ratio)
    xy[..., 0] = np.clip(xy[..., 0], 0.0, float(orig_w - 1))
    xy[..., 1] = np.clip(xy[..., 1], 0.0, float(orig_h - 1))
    scaled["xy"] = xy

    boxes_xyxy = np.asarray(pose["boxes_xyxy"], dtype=np.float32).copy()
    if boxes_xyxy.size:
        boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - float(pad_left)) / float(ratio)
        boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - float(pad_top)) / float(ratio)
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0.0, float(orig_w - 1))
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0.0, float(orig_h - 1))
    scaled["boxes_xyxy"] = boxes_xyxy

    if boxes_xyxy.shape[0]:
        centers = np.column_stack(
            (
                0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]),
                0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]),
            )
        ).astype(np.float32)
    else:
        centers = np.empty((0, 2), dtype=np.float32)
    scaled["box_centers"] = centers
    return scaled


def is_engine_weights_path(weights_path: Path) -> bool:
    return weights_path.suffix.lower() == ".engine"


def resolve_yolo_predict_device(device: str, yolo_is_engine: bool) -> Any:
    """
    Ultralytics accepts string/int device selectors.
    For TensorRT engines, a numeric GPU index is often the most compatible.
    """
    device_str = str(device).strip()
    if not yolo_is_engine:
        return device_str

    d = device_str.lower()
    if d == "cuda":
        return 0
    if d.startswith("cuda:"):
        idx = d.split(":", 1)[1].strip()
        if idx.isdigit():
            return int(idx)
    return device_str


def _has_ultralytics_engine_metadata(engine_path: Path) -> bool:
    """
    Ultralytics TensorRT loader expects engine files to begin with:
      [4-byte little-endian metadata length][JSON metadata][serialized TRT engine]
    """
    try:
        file_size = int(engine_path.stat().st_size)
        if file_size <= 4:
            return False

        with engine_path.open("rb") as f:
            raw_len = f.read(4)
            if len(raw_len) != 4:
                return False
            meta_len = int.from_bytes(raw_len, byteorder="little", signed=False)

            max_reasonable = min(file_size - 4, 8 * 1024 * 1024)
            if meta_len <= 0 or meta_len > int(max_reasonable):
                return False

            meta_raw = f.read(meta_len)
            if len(meta_raw) != int(meta_len):
                return False

        meta = json.loads(meta_raw.decode("utf-8"))
        return isinstance(meta, dict)
    except Exception:
        return False


def ensure_ultralytics_engine_header(engine_path: Path) -> Path:
    """
    Wrap metadata-less TRT engines with an empty Ultralytics metadata header.
    """
    if _has_ultralytics_engine_metadata(engine_path):
        return engine_path

    stem = engine_path.stem
    if not stem.endswith(".ultra"):
        stem = f"{stem}.ultra"
    wrapped_path = engine_path.with_name(f"{stem}.engine")

    try:
        src_stat = engine_path.stat()
        if wrapped_path.exists():
            dst_stat = wrapped_path.stat()
            if int(dst_stat.st_mtime) >= int(src_stat.st_mtime) and int(dst_stat.st_size) > int(src_stat.st_size):
                return wrapped_path
    except OSError:
        pass

    meta_raw = b"{}"
    with engine_path.open("rb") as src, wrapped_path.open("wb") as dst:
        dst.write(int(len(meta_raw)).to_bytes(4, byteorder="little", signed=False))
        dst.write(meta_raw)
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return wrapped_path


def _torch_dtype_from_numpy_dtype(np_dtype: np.dtype) -> Optional[torch.dtype]:
    mapping = {
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float64): torch.float64,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int16): torch.int16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.bool_): torch.bool,
    }
    return mapping.get(np.dtype(np_dtype))


def _tensor_from_numpy_without_bridge(arr: np.ndarray) -> torch.Tensor:
    torch_dtype = _torch_dtype_from_numpy_dtype(arr.dtype)
    if torch_dtype is None:
        return torch.tensor(arr.tolist())
    return torch.tensor(arr.tolist(), dtype=torch_dtype)


@contextmanager
def _temporary_torch_from_numpy_fallback(enabled: bool):
    if not enabled:
        yield
        return

    original_from_numpy = torch.from_numpy

    def patched_from_numpy(arr):
        try:
            return original_from_numpy(arr)
        except TypeError as exc:
            if isinstance(arr, np.ndarray) and "expected np.ndarray" in str(exc):
                # TensorRT binding setup in some Jetson stacks trips over
                # torch.from_numpy(np.empty(...)). Falling back to a pure-Python
                # conversion keeps engine initialization working.
                return _tensor_from_numpy_without_bridge(arr)
            raise

    torch.from_numpy = patched_from_numpy
    try:
        yield
    finally:
        torch.from_numpy = original_from_numpy


# ----------------------------- DATA STRUCTURES -----------------------------

def init_pose_arrays(num_frames: int, max_people: int, num_kpts: int) -> Dict[str, np.ndarray]:
    kpts_xy = np.full((num_frames, max_people, num_kpts, 2), np.nan, dtype=np.float32)
    kpts_conf = np.full((num_frames, max_people, num_kpts), np.nan, dtype=np.float32)
    person_conf = np.full((num_frames, max_people), np.nan, dtype=np.float32)
    return {"kpts_xy": kpts_xy, "kpts_conf": kpts_conf, "person_conf": person_conf}


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12
TORSO_IDXS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)


def _to_numpy_or_none(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def extract_box_centers_conf(r) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Safely extract:
      - box centers: (P, 2)
      - box conf: (P,) or None
      - boxes xyxy: (P, 4)
    """
    if r.boxes is None:
        empty_centers = np.empty((0, 2), dtype=np.float32)
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        return empty_centers, None, empty_boxes

    boxes_xyxy = _to_numpy_or_none(getattr(r.boxes, "xyxy", None))
    if boxes_xyxy is None:
        empty_centers = np.empty((0, 2), dtype=np.float32)
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        return empty_centers, None, empty_boxes

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[0] == 0 or boxes_xyxy.shape[1] < 4:
        empty_centers = np.empty((0, 2), dtype=np.float32)
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        return empty_centers, None, empty_boxes

    boxes_xyxy = boxes_xyxy[:, :4]
    centers = np.column_stack((
        0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]),
        0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]),
    )).astype(np.float32)

    box_conf = _to_numpy_or_none(getattr(r.boxes, "conf", None))
    if box_conf is not None:
        box_conf = np.asarray(box_conf, dtype=np.float32).reshape(-1)
        if box_conf.shape[0] < boxes_xyxy.shape[0]:
            pad = boxes_xyxy.shape[0] - box_conf.shape[0]
            box_conf = np.pad(box_conf, (0, pad), mode="constant", constant_values=np.nan)
        elif box_conf.shape[0] > boxes_xyxy.shape[0]:
            box_conf = box_conf[: boxes_xyxy.shape[0]]

    return centers, box_conf, boxes_xyxy


def _box_area_xyxy(box_xyxy: np.ndarray) -> float:
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)
    if box.shape[0] < 4 or not np.all(np.isfinite(box[:4])):
        return 0.0
    w = float(max(0.0, float(box[2] - box[0])))
    h = float(max(0.0, float(box[3] - box[1])))
    return w * h


def box_iou_xyxy(box1, box2) -> float:
    b1 = np.asarray(box1, dtype=np.float32).reshape(-1)
    b2 = np.asarray(box2, dtype=np.float32).reshape(-1)
    if b1.shape[0] < 4 or b2.shape[0] < 4:
        return 0.0
    if not np.all(np.isfinite(b1[:4])) or not np.all(np.isfinite(b2[:4])):
        return 0.0

    x_left = max(float(b1[0]), float(b2[0]))
    y_top = max(float(b1[1]), float(b2[1]))
    x_right = min(float(b1[2]), float(b2[2]))
    y_bottom = min(float(b1[3]), float(b2[3]))

    inter_w = max(0.0, x_right - x_left)
    inter_h = max(0.0, y_bottom - y_top)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    a1 = _box_area_xyxy(b1[:4])
    a2 = _box_area_xyxy(b2[:4])
    denom = a1 + a2 - inter
    if denom <= 0.0:
        return 0.0
    return float(inter / denom)


def compute_pose_centers(
    xy: np.ndarray,
    kc: Optional[np.ndarray],
    conf_thres: float,
) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        return np.empty((0, 2), dtype=np.float32)

    centers = np.full((xy.shape[0], 2), np.nan, dtype=np.float32)
    visible = np.isfinite(xy[..., 0]) & np.isfinite(xy[..., 1])

    if kc is not None:
        kc_arr = np.asarray(kc, dtype=np.float32)
        if kc_arr.shape[:2] == xy.shape[:2]:
            visible &= np.isfinite(kc_arr) & (kc_arr >= float(conf_thres))

    for person_idx in range(xy.shape[0]):
        torso_vis = [kp_idx for kp_idx in TORSO_IDXS if kp_idx < xy.shape[1] and visible[person_idx, kp_idx]]
        if len(torso_vis) >= 2:
            centers[person_idx] = np.mean(xy[person_idx, torso_vis], axis=0)
            continue

        pts = xy[person_idx, visible[person_idx]]
        if len(pts) >= 4:
            centers[person_idx] = np.mean(pts, axis=0)

    return centers


def point_in_box(center: np.ndarray, box_xyxy: Tuple[float, float, float, float]) -> bool:
    c = np.asarray(center, dtype=np.float32).reshape(-1)
    if c.shape[0] < 2 or not np.all(np.isfinite(c[:2])):
        return False
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return x1 <= float(c[0]) <= x2 and y1 <= float(c[1]) <= y2


def classify_suspicious_candidates(
    pose_centers: np.ndarray,
    frame_idx: int,
    config: PoseExportConfig,
    region2_all_clip: bool = False,
    allow_region1: bool = False,
    allow_region2: bool = False,
) -> np.ndarray:
    centers = np.asarray(pose_centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] < 2:
        return np.zeros((0,), dtype=bool)

    if not bool(config.no_suspicious):
        return np.zeros((centers.shape[0],), dtype=bool)

    region2_active = bool(region2_all_clip) or int(frame_idx) < max(0, int(config.suspicious_start_frames))
    suspicious = np.zeros((centers.shape[0],), dtype=bool)
    for idx in range(centers.shape[0]):
        if (not allow_region1) and point_in_box(centers[idx], config.suspicious_region1_xyxy):
            suspicious[idx] = True
            continue
        if point_in_box(centers[idx], config.suspicious_region3_xyxy):
            suspicious[idx] = True
            continue
        if region2_active and (not allow_region2) and point_in_box(centers[idx], config.suspicious_region2_xyxy):
            suspicious[idx] = True
    return suspicious


def choose_locked_candidate(
    candidate_idx: np.ndarray,
    dists: np.ndarray,
    box_conf: Optional[np.ndarray],
    conf_min: float,
    boxes_xyxy: np.ndarray,
    prev_box_xyxy: np.ndarray,
    min_iou_same_track: float,
    max_box_area_ratio: float,
    strict_reacquire: bool,
) -> Tuple[Optional[int], float, float]:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int32).reshape(-1)
    if candidate_idx.size == 0:
        return None, -1.0, float("inf")

    if box_conf is not None:
        conf_ok = []
        for idx in candidate_idx.tolist():
            if idx < box_conf.shape[0]:
                conf_val = float(box_conf[idx])
                if np.isfinite(conf_val) and conf_val < conf_min:
                    continue
            conf_ok.append(idx)
        candidate_idx = np.asarray(conf_ok, dtype=np.int32)
        if candidate_idx.size == 0:
            return None, -1.0, float("inf")

    if not strict_reacquire:
        ranked = candidate_idx[np.argsort(dists[candidate_idx])]
        best_idx = int(ranked[0])
        return best_idx, -1.0, float(dists[best_idx])

    prev_area = _box_area_xyxy(prev_box_xyxy)
    if prev_area <= 0.0:
        return None, -1.0, float("inf")

    area_ratio_limit = max(1.0, float(max_box_area_ratio))
    min_area_ratio = 1.0 / area_ratio_limit
    min_iou = max(0.0, float(min_iou_same_track))

    best_idx: Optional[int] = None
    best_iou = -1.0
    best_dist = float("inf")

    for idx in candidate_idx.tolist():
        cand_box = boxes_xyxy[idx]
        iou = box_iou_xyxy(cand_box, prev_box_xyxy)
        if iou < min_iou:
            continue

        cand_area = _box_area_xyxy(cand_box)
        if cand_area <= 0.0:
            continue
        area_ratio = cand_area / prev_area
        if area_ratio < min_area_ratio or area_ratio > area_ratio_limit:
            continue

        dist = float(dists[idx])
        if (iou > best_iou + 1e-6) or (abs(iou - best_iou) <= 1e-6 and dist < best_dist):
            best_idx = int(idx)
            best_iou = iou
            best_dist = dist

    if best_idx is None:
        return None, -1.0, float("inf")
    return best_idx, best_iou, best_dist


def choose_foreground_acquire_candidates(
    candidate_idx: np.ndarray,
    boxes_xyxy: np.ndarray,
    box_conf: Optional[np.ndarray],
    conf_min: float,
    min_box_area_ratio: float,
    bottom_margin_px: float,
) -> np.ndarray:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int32).reshape(-1)
    if candidate_idx.size == 0:
        return candidate_idx

    area_ratio = float(max(0.0, min_box_area_ratio))
    bottom_margin = float(max(0.0, bottom_margin_px))

    areas = np.asarray(
        [_box_area_xyxy(boxes_xyxy[idx]) for idx in candidate_idx.tolist()],
        dtype=np.float32,
    )
    bottoms = np.asarray(
        [float(np.asarray(boxes_xyxy[idx], dtype=np.float32).reshape(-1)[3]) for idx in candidate_idx.tolist()],
        dtype=np.float32,
    )

    finite_mask = np.isfinite(areas) & np.isfinite(bottoms)
    if np.any(finite_mask):
        max_area = float(np.max(areas[finite_mask]))
        max_bottom = float(np.max(bottoms[finite_mask]))
        keep_mask = finite_mask.copy()
        if max_area > 0.0:
            keep_mask &= areas >= (max_area * area_ratio)
        keep_mask &= bottoms >= (max_bottom - bottom_margin)
        if np.any(keep_mask):
            candidate_idx = candidate_idx[keep_mask]

    if candidate_idx.size == 0 or box_conf is None:
        return candidate_idx

    conf_ok = []
    for idx in candidate_idx.tolist():
        if idx >= box_conf.shape[0]:
            continue
        conf_val = float(box_conf[idx])
        if np.isfinite(conf_val) and conf_val >= conf_min:
            conf_ok.append(idx)

    if conf_ok:
        return np.asarray(conf_ok, dtype=np.int32)
    return candidate_idx


def select_person_idx(
    box_centers: np.ndarray,
    box_conf: Optional[np.ndarray],
    boxes_xyxy: np.ndarray,
    prev_center: Optional[np.ndarray],
    prev_box_xyxy: Optional[np.ndarray],
    target_center: np.ndarray,
    conf_min: float,
    max_jump_px: float,
    min_iou_same_track: float,
    max_box_area_ratio: float,
    locked: bool,
    strict_reacquire: bool = True,
    candidate_is_suspicious: Optional[np.ndarray] = None,
    prev_selected_was_suspicious: bool = False,
    suspicious_switch_min_iou: float = 0.35,
    suspicious_switch_max_jump_frac: float = 0.25,
    prefer_foreground_on_acquire: bool = False,
    acquire_min_box_area_ratio: float = 0.35,
    acquire_bottom_margin_px: float = 60.0,
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    Single-target selection:
      - Before lock: acquire once, closest to target_center (optionally preferring conf >= conf_min).
      - After lock: never center-reacquire. Match only candidates consistent with previous target.
        Strict mode gates by center jump, IoU to previous box, and box-area ratio.
        Candidates are ranked by highest IoU, then smallest center distance.
    """
    num_people = min(int(box_centers.shape[0]), int(boxes_xyxy.shape[0]))
    if num_people == 0:
        return None, prev_center

    box_centers = box_centers[:num_people]
    boxes_xyxy = boxes_xyxy[:num_people]
    if box_conf is not None:
        box_conf = box_conf[:num_people]
    if candidate_is_suspicious is None:
        candidate_is_suspicious = np.zeros((num_people,), dtype=bool)
    else:
        candidate_is_suspicious = np.asarray(candidate_is_suspicious, dtype=bool).reshape(-1)
        if candidate_is_suspicious.shape[0] < num_people:
            pad = num_people - candidate_is_suspicious.shape[0]
            candidate_is_suspicious = np.pad(candidate_is_suspicious, (0, pad), mode="constant", constant_values=False)
        elif candidate_is_suspicious.shape[0] > num_people:
            candidate_is_suspicious = candidate_is_suspicious[:num_people]

    if not locked:
        candidate_idx = np.arange(num_people, dtype=np.int32)
        non_suspicious = candidate_idx[~candidate_is_suspicious[candidate_idx]]
        if non_suspicious.size == 0:
            return None, prev_center
        candidate_idx = non_suspicious

        if prefer_foreground_on_acquire:
            candidate_idx = choose_foreground_acquire_candidates(
                candidate_idx=candidate_idx,
                boxes_xyxy=boxes_xyxy,
                box_conf=box_conf,
                conf_min=conf_min,
                min_box_area_ratio=acquire_min_box_area_ratio,
                bottom_margin_px=acquire_bottom_margin_px,
            )
            if candidate_idx.size == 0:
                return None, prev_center
        elif box_conf is not None:
            high_conf = candidate_idx[
                np.isfinite(box_conf[candidate_idx]) & (box_conf[candidate_idx] >= conf_min)
            ]
            if high_conf.size > 0:
                candidate_idx = high_conf.astype(np.int32, copy=False)

        dists = np.linalg.norm(box_centers[candidate_idx] - target_center[None, :], axis=1)
        if dists.size == 0:
            return None, prev_center

        best_rel = int(np.argmin(dists))
        best_idx = int(candidate_idx[best_rel])
        return best_idx, box_centers[best_idx].astype(np.float32, copy=True)

    # After lock: never return to generic center acquisition.
    if prev_center is None or prev_box_xyxy is None:
        return None, prev_center

    dists = np.linalg.norm(box_centers - prev_center[None, :], axis=1)
    if dists.size == 0:
        return None, prev_center

    valid_jump = np.where(np.isfinite(dists) & (dists <= max_jump_px))[0]
    if valid_jump.size == 0:
        return None, prev_center

    allowed_idx = valid_jump[~candidate_is_suspicious[valid_jump]]
    suspicious_idx = valid_jump[candidate_is_suspicious[valid_jump]]

    best_idx, _, _ = choose_locked_candidate(
        candidate_idx=allowed_idx,
        dists=dists,
        box_conf=box_conf,
        conf_min=conf_min,
        boxes_xyxy=boxes_xyxy,
        prev_box_xyxy=prev_box_xyxy,
        min_iou_same_track=min_iou_same_track,
        max_box_area_ratio=max_box_area_ratio,
        strict_reacquire=strict_reacquire,
    )
    if best_idx is not None:
        return best_idx, box_centers[best_idx].astype(np.float32, copy=True)

    best_idx, best_iou, best_dist = choose_locked_candidate(
        candidate_idx=suspicious_idx,
        dists=dists,
        box_conf=box_conf,
        conf_min=conf_min,
        boxes_xyxy=boxes_xyxy,
        prev_box_xyxy=prev_box_xyxy,
        min_iou_same_track=min_iou_same_track,
        max_box_area_ratio=max_box_area_ratio,
        strict_reacquire=strict_reacquire,
    )
    if best_idx is None:
        return None, prev_center

    if prev_selected_was_suspicious:
        return best_idx, box_centers[best_idx].astype(np.float32, copy=True)

    required_iou = max(float(suspicious_switch_min_iou), float(min_iou_same_track))
    allowed_jump = max(1.0, float(max_jump_px) * float(suspicious_switch_max_jump_frac))
    if best_dist <= allowed_jump and (not strict_reacquire or best_iou >= required_iou):
        return best_idx, box_centers[best_idx].astype(np.float32, copy=True)

    return None, prev_center


def draw_selected_pose(
    frame: np.ndarray,
    kpts_xy: Optional[np.ndarray],
    kpts_conf: Optional[np.ndarray],
    box_xyxy: Optional[np.ndarray],
    draw_kpt_threshold: float,
    draw_no_target_text: bool,
) -> np.ndarray:
    out = frame.copy()
    if kpts_xy is None:
        if draw_no_target_text:
            cv2.putText(
                out,
                "NO TARGET",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        return out

    xy = np.asarray(kpts_xy, dtype=np.float32)
    conf = None
    if kpts_conf is not None:
        conf = np.asarray(kpts_conf, dtype=np.float32).reshape(-1)

    if box_xyxy is not None:
        box_xyxy = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)
        if box_xyxy.shape[0] >= 4 and np.all(np.isfinite(box_xyxy[:4])):
            x1, y1, x2, y2 = np.round(box_xyxy[:4]).astype(int).tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

    for a, b in COCO_SKELETON:
        if a >= xy.shape[0] or b >= xy.shape[0]:
            continue
        if not np.isfinite(xy[a]).all() or not np.isfinite(xy[b]).all():
            continue
        if conf is not None:
            if a >= conf.shape[0] or b >= conf.shape[0]:
                continue
            if not np.isfinite(conf[a]) or not np.isfinite(conf[b]):
                continue
            if conf[a] < draw_kpt_threshold or conf[b] < draw_kpt_threshold:
                continue

        pt1 = tuple(np.round(xy[a]).astype(int).tolist())
        pt2 = tuple(np.round(xy[b]).astype(int).tolist())
        cv2.line(out, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    for k in range(xy.shape[0]):
        if not np.isfinite(xy[k]).all():
            continue
        if conf is not None:
            if k >= conf.shape[0] or not np.isfinite(conf[k]):
                continue
            if conf[k] < draw_kpt_threshold:
                continue
        pt = tuple(np.round(xy[k]).astype(int).tolist())
        cv2.circle(out, pt, 3, (0, 0, 255), -1, cv2.LINE_AA)

    return out


def extract_pose_for_frame(r) -> Optional[Dict[str, Any]]:
    """
    Return dict with:
      xy: (P, K, 2)
      kc: (P, K)
      box_conf: (P,) or None
      box_centers: (P, 2)
      boxes_xyxy: (P, 4)
    """
    if r.keypoints is None or getattr(r.keypoints, "xy", None) is None:
        return None

    xy = _to_numpy_or_none(r.keypoints.xy)
    if xy is None:
        return None
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 3 or xy.shape[0] == 0:
        return None

    kc = _to_numpy_or_none(getattr(r.keypoints, "conf", None))
    if kc is not None:
        kc = np.asarray(kc, dtype=np.float32)

    box_centers, box_conf, boxes_xyxy = extract_box_centers_conf(r)

    return {
        "xy": xy,
        "kc": kc,
        "box_conf": box_conf,
        "box_centers": box_centers,
        "boxes_xyxy": boxes_xyxy,
    }


# ----------------------------- CORE PIPELINE -----------------------------

def run_pose_on_frames(
    frames_dir: str,
    out_dir: str,
    windows_csv: str,
    config: PoseExportConfig,
    pattern: str = "*.png"
) -> Tuple[str, str, Optional[str]]:
    """
    Processes a directory of frames.
    Writes:
      - annotated video (mp4)
      - keypoints (npz)
      - optional csv
    Returns paths to outputs.
    """
    ensure_dir(out_dir)

    frame_paths = list_frames(frames_dir, pattern=pattern)
    num_frames = len(frame_paths)

    weights_path = Path(config.model_path).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"Pose model not found: {weights_path}")

    yolo_is_engine = is_engine_weights_path(weights_path)
    runtime_weights_path = weights_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if yolo_is_engine:
        if not str(device).lower().startswith("cuda"):
            raise RuntimeError(
                f"TensorRT .engine model requires CUDA, but resolved device is '{device}'."
            )
        runtime_weights_path = ensure_ultralytics_engine_header(weights_path)

    try:
        model = YOLO(str(runtime_weights_path), task="pose")
    except TypeError:
        model = YOLO(str(runtime_weights_path))

    if not yolo_is_engine:
        model.to(device)

    yolo_predict_device = resolve_yolo_predict_device(device=device, yolo_is_engine=yolo_is_engine)
    use_half_yolo = bool(yolo_is_engine and str(device).lower().startswith("cuda"))
    print("Using device:", device)
    if yolo_is_engine:
        print(f"Using TensorRT engine: {weights_path}")
        if runtime_weights_path != weights_path:
            print(f"Wrapped engine path: {runtime_weights_path}")

    first = read_image(frame_paths[0])
    h, w = first.shape[:2]
    _, predict_imgsz_info = resolve_predict_imgsz(
        imgsz_value=config.imgsz,
        frame_shape=(h, w),
        stride=extract_model_stride(model),
    )
    if predict_imgsz_info["mode"] == "pixel_ratio":
        applied_h, applied_w = predict_imgsz_info["applied_hw"]
        print(
            "Using resized predict canvas: "
            f"{applied_w}x{applied_h} "
            f"(requested pixel ratio={predict_imgsz_info['requested_value']:.3f}, "
            f"applied={predict_imgsz_info['applied_ratio']:.3f} relative to {w}x{h})"
        )
    elif predict_imgsz_info["mode"] == "square":
        print(
            "Using explicit square predict imgsz: "
            f"{int(predict_imgsz_info['requested_value'])}"
        )

    out_video = os.path.join(out_dir, "pose_out.mp4")
    out_npz = os.path.join(out_dir, "keypoints.npz")
    out_csv = os.path.join(out_dir, "keypoints.csv") if config.save_csv else None

    writer = make_video_writer(out_video, config.fps, (w, h), codec=config.video_codec)

    arrays = init_pose_arrays(num_frames, config.max_people, config.num_kpts)

    csv_rows = []
    if config.save_csv:
        csv_rows.append(["frame", "person", "kpt", "x", "y", "kpt_conf", "person_conf", "frame_path"])
    

    #Get frame labels
    windows_df = load_window_labels(windows_csv)
    frame_dts = [parse_frame_timestamp(p) for p in frame_paths]

    frames_df = pd.DataFrame({
        "frame_idx": np.arange(num_frames),
        "frame_path": frame_paths,
        "frame_dt": frame_dts,
    }).sort_values("frame_dt").reset_index(drop=True)

    frames_df = pd.merge_asof(
        frames_df.sort_values("frame_dt"),
        windows_df.sort_values("window_start"),
        left_on="frame_dt",
        right_on="window_start",
        direction="backward",
        allow_exact_matches=True,
    )

    frames_df["label"] = frames_df["label"].fillna("unknown")
    frames_df["window_id"] = frames_df["window_id"].fillna(-1).astype(np.int64)

    frame_labels = frames_df["label"].to_numpy()
    frame_window_ids = frames_df["window_id"].to_numpy()

    target_center = np.array(
        [w * config.target_x_frac, h * config.target_y_frac],
        dtype=np.float32,
    )
    frame_diag = float(np.hypot(float(w), float(h)))
    max_jump_px = float(config.max_jump_px) if config.max_jump_px is not None else float(config.max_jump_diag_frac * frame_diag)
    prev_center: Optional[np.ndarray] = None
    prev_box_xyxy: Optional[np.ndarray] = None
    prev_selected_was_suspicious = False
    track_locked = False
    lost_count = 0

    for i, p in enumerate(frame_paths):
        frame = cv2.imread(p)
        if frame is None:
            print(f"Skipping unreadable frame: {p}")
            continue

        frame_for_model = frame
        letterbox_info = None
        if predict_imgsz_info["applied_hw"] is not None:
            frame_for_model, letterbox_info = letterbox_image_to_shape(
                frame,
                predict_imgsz_info["applied_hw"],
            )

        # Keep detector candidate count independent from exported people count.
        max_det = max(1, int(config.detector_max_det))
        predict_kwargs = {
            "source": frame_for_model,
            "conf": config.conf_thres,
            "verbose": False,
            "device": yolo_predict_device,
            "max_det": max_det,
        }
        if predict_imgsz_info["applied_hw"] is not None:
            predict_kwargs["imgsz"] = list(predict_imgsz_info["applied_hw"])
        if yolo_is_engine:
            predict_kwargs["half"] = use_half_yolo
        with _temporary_torch_from_numpy_fallback(enabled=bool(yolo_is_engine)):
            results = model.predict(**predict_kwargs)
        r = results[0]
        selected_xy: Optional[np.ndarray] = None
        selected_kc: Optional[np.ndarray] = None
        selected_box_xyxy: Optional[np.ndarray] = None
        selected_person_conf = float("nan")

        pose = extract_pose_for_frame(r)
        if pose is not None and letterbox_info is not None:
            pose = scale_pose_from_letterbox(
                pose,
                ratio=float(letterbox_info["ratio"]),
                pad_left=int(letterbox_info["pad_left"]),
                pad_top=int(letterbox_info["pad_top"]),
                orig_shape=(h, w),
            )
        if pose is None:
            lost_count += 1
        else:
            xy = pose["xy"]
            kc = pose["kc"]
            box_conf = pose["box_conf"]
            box_centers = pose["box_centers"]
            boxes_xyxy = pose["boxes_xyxy"]

            # Match pose rows to bbox rows before selection.
            num_candidates = min(int(xy.shape[0]), int(box_centers.shape[0]), int(boxes_xyxy.shape[0]))
            if num_candidates <= 0:
                lost_count += 1
            else:
                xy = xy[:num_candidates]
                box_centers = box_centers[:num_candidates]
                boxes_xyxy = boxes_xyxy[:num_candidates]
                if box_conf is not None:
                    box_conf = box_conf[:num_candidates]
                if kc is not None and kc.ndim >= 2:
                    kc = kc[:num_candidates]
                else:
                    kc = None

                locked_for_selection = bool(
                    track_locked or (
                        (not config.lock_first_target) and
                        (prev_center is not None) and
                        (prev_box_xyxy is not None)
                    )
                )
                pose_centers = compute_pose_centers(
                    xy=xy,
                    kc=kc,
                    conf_thres=config.suspicious_conf_thres,
                )
                invalid_pose_centers = ~np.isfinite(pose_centers[:, 0]) | ~np.isfinite(pose_centers[:, 1])
                if np.any(invalid_pose_centers):
                    pose_centers[invalid_pose_centers] = box_centers[invalid_pose_centers]
                candidate_is_suspicious = classify_suspicious_candidates(
                    pose_centers=pose_centers,
                    frame_idx=i,
                    config=config,
                    region2_all_clip=locked_for_selection,
                    allow_region1=(not locked_for_selection) and bool(config.allow_region1_start),
                    allow_region2=(not locked_for_selection) and bool(config.allow_region2_start),
                )
                idx, new_center = select_person_idx(
                    box_centers=box_centers,
                    box_conf=box_conf,
                    boxes_xyxy=boxes_xyxy,
                    prev_center=prev_center,
                    prev_box_xyxy=prev_box_xyxy,
                    target_center=target_center,
                    conf_min=config.conf_min,
                    max_jump_px=max_jump_px,
                    min_iou_same_track=config.min_iou_same_track,
                    max_box_area_ratio=config.max_box_area_ratio,
                    locked=locked_for_selection,
                    strict_reacquire=config.strict_reacquire,
                    candidate_is_suspicious=candidate_is_suspicious,
                    prev_selected_was_suspicious=prev_selected_was_suspicious,
                    suspicious_switch_min_iou=config.suspicious_switch_min_iou,
                    suspicious_switch_max_jump_frac=config.suspicious_switch_max_jump_frac,
                    prefer_foreground_on_acquire=config.prefer_foreground_on_acquire,
                    acquire_min_box_area_ratio=config.acquire_min_box_area_ratio,
                    acquire_bottom_margin_px=config.acquire_bottom_margin_px,
                )

                if idx is None:
                    lost_count += 1
                else:
                    prev_center = new_center
                    prev_box_xyxy = np.asarray(boxes_xyxy[idx], dtype=np.float32).copy()
                    if (
                        config.lock_first_target
                        and not track_locked
                        and i >= max(0, int(config.lock_delay_frames))
                    ):
                        track_locked = True
                    lost_count = 0
                    prev_selected_was_suspicious = bool(
                        idx < candidate_is_suspicious.shape[0] and candidate_is_suspicious[idx]
                    )

                    selected_xy = xy[idx]
                    selected_box_xyxy = prev_box_xyxy
                    if kc is not None and idx < kc.shape[0]:
                        selected_kc = kc[idx]

                    if box_conf is not None and idx < box_conf.shape[0] and np.isfinite(box_conf[idx]):
                        selected_person_conf = float(box_conf[idx])
                    elif selected_kc is not None:
                        selected_person_conf = float(np.nanmean(selected_kc))

                    if config.max_people > 0:
                        j = 0
                        xy_sel = np.asarray(selected_xy, dtype=np.float32)
                        xy_count = min(config.num_kpts, int(xy_sel.shape[0]))
                        arrays["kpts_xy"][i, j, :xy_count] = xy_sel[:xy_count]

                        if selected_kc is not None:
                            kc_sel = np.asarray(selected_kc, dtype=np.float32).reshape(-1)
                            kc_count = min(config.num_kpts, int(kc_sel.shape[0]))
                            arrays["kpts_conf"][i, j, :kc_count] = kc_sel[:kc_count]

                        arrays["person_conf"][i, j] = selected_person_conf

                        if config.save_csv:
                            for k in range(config.num_kpts):
                                x, y = arrays["kpts_xy"][i, j, k]
                                kconf = arrays["kpts_conf"][i, j, k]
                                pconf = arrays["person_conf"][i, j]
                                csv_rows.append([i, j, k, float(x), float(y), float(kconf), float(pconf), p])

        if config.reset_on_max_lost and (not track_locked) and lost_count > config.max_lost:
            prev_center = None
            prev_box_xyxy = None
            prev_selected_was_suspicious = False

        annotated = draw_selected_pose(
            frame=frame,
            kpts_xy=selected_xy,
            kpts_conf=selected_kc,
            box_xyxy=selected_box_xyxy,
            draw_kpt_threshold=config.draw_kpt_threshold,
            draw_no_target_text=config.draw_no_target_text,
        )
        writer.write(annotated)

    writer.release()

    #Save to .npz file
    np.savez_compressed(
        out_npz,
        kpts_xy=arrays["kpts_xy"],
        kpts_conf=arrays["kpts_conf"],
        person_conf=arrays["person_conf"],
        frame_paths=np.array(frame_paths, dtype=object),
        frame_labels=frame_labels,
        window_ids=frame_window_ids,
        frame_timestamps=np.array(frame_dts, dtype="datetime64[ns]"),
        fps=np.array([config.fps], dtype=np.int32),
        model_path=np.array([config.model_path], dtype=object),
        predict_imgsz=np.array(
            predict_imgsz_info["applied_hw"] if predict_imgsz_info["applied_hw"] is not None else (-1, -1),
            dtype=np.int32,
        ),
        predict_pixel_ratio=np.array([predict_imgsz_info["applied_ratio"]], dtype=np.float32),
        predict_imgsz_mode=np.array([predict_imgsz_info["mode"]], dtype=object),
    )

    if config.save_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerows(csv_rows)

    return out_video, out_npz, out_csv

def load_window_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=1, encoding="utf-8-sig")

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    if "Timestamp" not in df.columns:
        raise ValueError(f"'Timestamp' not found. Columns: {df.columns[:10].tolist()}")

    if "Tag" not in df.columns:
        raise ValueError(f"'Tag' not found. Columns: {df.columns[-10:].tolist()}")

    window_start = pd.to_datetime(
        df["Timestamp"],
        format="%Y-%m-%dT%H:%M:%S.%f",
        errors="coerce"
    )

    out = pd.DataFrame({
        "window_start": window_start,
        "label": df["Tag"].astype(str),
    })

    out = (
        out.dropna(subset=["window_start"])
           .sort_values("window_start")
           .reset_index(drop=True)
    )

    out["window_id"] = np.arange(len(out), dtype=np.int64)
    return out

def find_camera_folders(root: str, camera: int = 1) -> List[str]:
    pat = os.path.join(root, "**", f"*Camera{camera}")
    folders = [p for p in glob.glob(pat, recursive=True) if os.path.isdir(p)]
    folders = [p for p in folders if re.search(rf"Camera{camera}$", os.path.basename(p))]
    return folders


def find_features_csv(trial_dir: str, features_suffix: str = "Features1&0.5.csv") -> str:
    cands = glob.glob(os.path.join(trial_dir, f"*{features_suffix}"))
    if not cands:
        raise FileNotFoundError(f"No *{features_suffix} found in {trial_dir}")
    if len(cands) > 1:
        # usually only one; pick first but warn
        print(f"Warning: multiple feature CSVs in {trial_dir}, using {cands[0]}")
    return cands[0]

def batch_process_upfall_tree(
    upfall_root: str,
    output_root: str,
    config: PoseExportConfig,
    camera: int = 1,
    features_suffix: str = "Features1&0.5.csv",
    pattern: str = "*.png",
) -> List[Tuple[str, str, str]]:
    ensure_dir(output_root)

    cam_folders = find_camera_folders(upfall_root, camera=camera)
    results = []

    for frames_dir in cam_folders:
        trial_dir = os.path.dirname(frames_dir)  # .../TrialZ/
        try:
            windows_csv = find_features_csv(trial_dir, features_suffix=features_suffix)
        except FileNotFoundError as e:
            print(f"Skipping {frames_dir}: {e}")
            continue

        # Make a unique output folder name from the relative path
        rel = os.path.relpath(frames_dir, upfall_root)
        out_dir = os.path.join(output_root, rel)
        ensure_dir(out_dir)

        # Skip if already processed (resume-friendly)
        if os.path.exists(os.path.join(out_dir, "keypoints.npz")):
            continue

        out_video, out_npz, _ = run_pose_on_frames(
            frames_dir=frames_dir,
            out_dir=out_dir,
            windows_csv=windows_csv,
            config=config,
            pattern=pattern,
        )
        results.append((frames_dir, out_video, out_npz))

    return results




# ----------------------------- EXAMPLE USAGE -----------------------------

# if __name__ == "__main__":
#     cfg = PoseExportConfig(
#         model_path="yolo11l-pose.pt",
#         conf_thres=0.25,
#         fps=30,
#         max_people=1,
#         save_csv=False
#     )
#     frames = "../../Datasets/UP-Fall/Subject1Activity1Trial1Camera1"
#     windows_csv = "../../Datasets/UP-Fall/Subject1Activity1Trial1Features1&0.5.csv"

#     # Single folder
#     run_pose_on_frames(frames_dir=frames, out_dir="outputs/frames_run", windows_csv=windows_csv, config=cfg)

#     # Many folders
#     # batch_process_folders(input_root="all_sequences", output_root="outputs", config=cfg)

