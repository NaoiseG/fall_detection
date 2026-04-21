from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

os.environ.setdefault("YOLO_OFFLINE", "True")
os.environ.setdefault("ULTRALYTICS_OFFLINE", "True")
os.environ.setdefault("ULTRALYTICS_SYNC", "False")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import cv2
import numpy as np
import torch
from ultralytics import YOLO

K = 17

SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
]


def _maybe_cuda_sync(sync_cuda: bool) -> None:
    if bool(sync_cuda) and torch.cuda.is_available():
        torch.cuda.synchronize()


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def is_engine_weights_path(weights_path: Path) -> bool:
    return Path(weights_path).suffix.lower() == ".engine"


def resolve_yolo_predict_device(device: str, yolo_is_engine: bool) -> Any:
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


def _scale_xy_from_letterbox(
    xy: np.ndarray,
    *,
    ratio: float,
    pad_left: int,
    pad_top: int,
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    orig_h, orig_w = orig_shape
    scaled = np.asarray(xy, dtype=np.float32).copy()
    scaled[..., 0] = (scaled[..., 0] - float(pad_left)) / float(ratio)
    scaled[..., 1] = (scaled[..., 1] - float(pad_top)) / float(ratio)
    scaled[..., 0] = np.clip(scaled[..., 0], 0.0, float(orig_w - 1))
    scaled[..., 1] = np.clip(scaled[..., 1], 0.0, float(orig_h - 1))
    return scaled


def _scale_boxes_from_letterbox(
    boxes_xyxy: np.ndarray,
    *,
    ratio: float,
    pad_left: int,
    pad_top: int,
    orig_shape: Tuple[int, int],
) -> np.ndarray:
    orig_h, orig_w = orig_shape
    scaled = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    if scaled.size == 0:
        return scaled.reshape(0, 4)
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - float(pad_left)) / float(ratio)
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - float(pad_top)) / float(ratio)
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0.0, float(orig_w - 1))
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0.0, float(orig_h - 1))
    return scaled


def _has_ultralytics_engine_metadata(engine_path: Path) -> bool:
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
                if os.environ.get("POSE_DEBUG_NUMPY_FALLBACK", "0") == "1":
                    print(
                        "[pose_debug] torch.from_numpy fallback "
                        f"shape={tuple(arr.shape)} dtype={arr.dtype}",
                        flush=True,
                    )
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


def _to_numpy_or_none(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _extract_boxes_xyxy_conf(r: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if r is None or getattr(r, "boxes", None) is None:
        return np.empty((0, 4), dtype=np.float32), None

    boxes_xyxy = _to_numpy_or_none(getattr(r.boxes, "xyxy", None))
    if boxes_xyxy is None:
        return np.empty((0, 4), dtype=np.float32), None
    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[0] == 0 or boxes_xyxy.shape[1] < 4:
        return np.empty((0, 4), dtype=np.float32), None

    boxes_xyxy = boxes_xyxy[:, :4]

    box_conf = _to_numpy_or_none(getattr(r.boxes, "conf", None))
    if box_conf is not None:
        box_conf = np.asarray(box_conf, dtype=np.float32).reshape(-1)
        if box_conf.shape[0] < boxes_xyxy.shape[0]:
            pad = boxes_xyxy.shape[0] - box_conf.shape[0]
            box_conf = np.pad(box_conf, (0, pad), mode="constant", constant_values=np.nan)
        elif box_conf.shape[0] > boxes_xyxy.shape[0]:
            box_conf = box_conf[: boxes_xyxy.shape[0]]

    return boxes_xyxy, box_conf


def select_person_idx(
    box_centers: np.ndarray,
    box_conf: Optional[np.ndarray],
    prev_center: Optional[np.ndarray],
    target_center: np.ndarray,
    conf_min: float,
    max_jump_px: float,
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    num_people = int(box_centers.shape[0])
    if num_people == 0:
        return None, prev_center

    if prev_center is None:
        candidate_idx = np.arange(num_people, dtype=np.int32)
        if box_conf is not None and box_conf.shape[0] >= num_people:
            high_conf = np.where(np.isfinite(box_conf[:num_people]) & (box_conf[:num_people] >= float(conf_min)))[0]
            if high_conf.size > 0:
                candidate_idx = high_conf.astype(np.int32, copy=False)

        dists = np.linalg.norm(box_centers[candidate_idx] - target_center[None, :], axis=1)
        if dists.size == 0:
            return None, prev_center

        best_rel = int(np.argmin(dists))
        best_idx = int(candidate_idx[best_rel])
        return best_idx, box_centers[best_idx].astype(np.float32, copy=True)

    dists = np.linalg.norm(box_centers - prev_center[None, :], axis=1)
    if dists.size == 0:
        return None, prev_center

    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    if (not np.isfinite(best_dist)) or (best_dist > float(max_jump_px)):
        return None, prev_center
    return best_idx, box_centers[best_idx].astype(np.float32, copy=True)


@dataclass
class PosePipelineConfig:
    yolo_weights: Path
    device: str
    imgsz: Optional[float] = 640
    yolo_conf: float = 0.25
    yolo_iou: Optional[float] = None
    max_det: int = 10
    use_half: bool = False

    frame_step: int = 1
    track_conf_min: float = 0.75
    track_max_jump_px: float = 0.0
    track_max_jump_diag_frac: float = 0.25
    track_max_lost: int = 10
    track_target_x_frac: float = 0.5
    track_target_y_frac: float = 0.5


@dataclass
class PoseFrameOutput:
    raw_frame_idx: int
    sampled: bool
    sampled_frame_idx: Optional[int]
    found: bool
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    pose_infer_ms: float
    track_ms: float


class PosePipeline:
    """
    Shared YOLO + target-lock tracking pipeline used by all temporal classifiers.
    """

    def __init__(self, config: PosePipelineConfig) -> None:
        self.config = config

        yolo_weights = Path(config.yolo_weights).expanduser()
        if not yolo_weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {yolo_weights}")

        self._yolo_is_engine = is_engine_weights_path(yolo_weights)
        if self._yolo_is_engine and (not str(config.device).lower().startswith("cuda")):
            raise ValueError("TensorRT .engine YOLO weights require CUDA.")

        self._yolo_runtime_weights = ensure_ultralytics_engine_header(yolo_weights) if self._yolo_is_engine else yolo_weights
        self._yolo_predict_device = resolve_yolo_predict_device(device=config.device, yolo_is_engine=self._yolo_is_engine)

        print(
            "[pose] yolo_init_start "
            f"weights={self._yolo_runtime_weights.as_posix()} "
            f"engine={bool(self._yolo_is_engine)} device={config.device}",
            flush=True,
        )
        try:
            self._pose_model = YOLO(str(self._yolo_runtime_weights), task="pose")
        except TypeError:
            self._pose_model = YOLO(str(self._yolo_runtime_weights))

        if not self._yolo_is_engine:
            print(f"[pose] yolo_to_device_start device={config.device}", flush=True)
            self._pose_model.to(str(config.device))
            _maybe_cuda_sync(str(config.device).lower().startswith("cuda"))
            print(f"[pose] yolo_to_device_done device={config.device}", flush=True)

        self._model_stride = extract_model_stride(self._pose_model)
        print(f"[pose] yolo_init_done stride={int(self._model_stride)}", flush=True)
        self._predict_imgsz_runtime: Optional[Any] = None
        self._predict_imgsz_info: Dict[str, Any] = {
            "mode": "default",
            "requested_value": None,
            "applied_hw": None,
            "applied_ratio": 1.0,
        }
        self._predict_imgsz_frame_shape: Optional[Tuple[int, int]] = None

        self.reset_tracking_state()

    @property
    def yolo_runtime_weights(self) -> Path:
        return self._yolo_runtime_weights

    @property
    def yolo_predict_device(self) -> Any:
        return self._yolo_predict_device

    @property
    def predict_stride(self) -> int:
        return int(self._model_stride)

    @property
    def predict_imgsz_info(self) -> Dict[str, Any]:
        info = dict(self._predict_imgsz_info)
        applied_hw = info.get("applied_hw")
        if applied_hw is not None:
            info["applied_hw"] = (int(applied_hw[0]), int(applied_hw[1]))
        info["stride"] = int(self._model_stride)
        return info

    def reset_tracking_state(self) -> None:
        self._sampled_count = 0
        self._last_xy = np.zeros((K, 2), dtype=np.float32)
        self._last_cf = np.zeros((K,), dtype=np.float32)
        self._prev_center: Optional[np.ndarray] = None
        self._target_center: Optional[np.ndarray] = None
        self._lost_count = 0
        self._max_jump_px_runtime: Optional[float] = None

    def _init_geometry_if_needed(self, frame: np.ndarray) -> None:
        if self._target_center is not None and self._max_jump_px_runtime is not None:
            return

        h, w = frame.shape[:2]
        frame_diag = float(np.hypot(float(w), float(h)))
        self._target_center = np.array(
            [
                float(self.config.track_target_x_frac) * float(w),
                float(self.config.track_target_y_frac) * float(h),
            ],
            dtype=np.float32,
        )

        if float(self.config.track_max_jump_px) > 0.0:
            self._max_jump_px_runtime = float(self.config.track_max_jump_px)
        else:
            self._max_jump_px_runtime = float(self.config.track_max_jump_diag_frac) * float(frame_diag)

    def _resolve_predict_imgsz_if_needed(self, frame: np.ndarray) -> None:
        frame_shape = (int(frame.shape[0]), int(frame.shape[1]))
        if self._predict_imgsz_frame_shape == frame_shape:
            return

        self._predict_imgsz_runtime, self._predict_imgsz_info = resolve_predict_imgsz(
            imgsz_value=self.config.imgsz,
            frame_shape=frame_shape,
            stride=int(self._model_stride),
        )
        self._predict_imgsz_frame_shape = frame_shape

        mode = str(self._predict_imgsz_info.get("mode", "default"))
        if mode == "pixel_ratio":
            applied_h, applied_w = self._predict_imgsz_info["applied_hw"]
            print(
                "[pose] using resized predict canvas: "
                f"{int(applied_w)}x{int(applied_h)} "
                f"(requested pixel ratio={float(self._predict_imgsz_info['requested_value']):.3f}, "
                f"applied={float(self._predict_imgsz_info['applied_ratio']):.3f} relative to "
                f"{int(frame_shape[1])}x{int(frame_shape[0])})"
            )
        elif mode == "square":
            requested_square = int(self._predict_imgsz_info["requested_value"])
            if requested_square != 640:
                print(f"[pose] using explicit square predict imgsz: {requested_square}")

    def process_frame(self, frame_bgr: np.ndarray, raw_frame_idx: int, sync_cuda_timing: bool = False) -> PoseFrameOutput:
        do_sample = (int(raw_frame_idx) % int(self.config.frame_step)) == 0
        if not do_sample:
            return PoseFrameOutput(
                raw_frame_idx=int(raw_frame_idx),
                sampled=False,
                sampled_frame_idx=None,
                found=bool(np.any(self._last_cf > 0.0)),
                keypoints_xy=self._last_xy.copy(),
                keypoints_conf=self._last_cf.copy(),
                pose_infer_ms=0.0,
                track_ms=0.0,
            )

        self._init_geometry_if_needed(frame_bgr)
        self._resolve_predict_imgsz_if_needed(frame_bgr)
        target_center = self._target_center
        max_jump_px = self._max_jump_px_runtime
        if target_center is None or max_jump_px is None:
            raise RuntimeError("Pose pipeline geometry failed to initialize.")

        frame_for_model = frame_bgr
        letterbox_info: Optional[Dict[str, Any]] = None
        applied_hw = self._predict_imgsz_info.get("applied_hw")
        if applied_hw is not None:
            frame_for_model, letterbox_info = letterbox_image_to_shape(
                frame_bgr,
                (int(applied_hw[0]), int(applied_hw[1])),
            )

        predict_kwargs = {
            "source": frame_for_model,
            "conf": float(self.config.yolo_conf),
            "verbose": False,
            "device": self._yolo_predict_device,
            "half": bool(self.config.use_half),
            "max_det": max(1, int(self.config.max_det)),
        }
        if self._predict_imgsz_runtime is not None:
            predict_kwargs["imgsz"] = self._predict_imgsz_runtime
        if self.config.yolo_iou is not None:
            predict_kwargs["iou"] = float(self.config.yolo_iou)

        trace_predict = int(raw_frame_idx) == 0 or _env_flag("POSE_DEBUG_PREDICT", False)
        if trace_predict:
            print(
                "[pose] predict_pre_sync_start "
                f"frame={int(raw_frame_idx)} sync_cuda={bool(sync_cuda_timing)}",
                flush=True,
            )
        _maybe_cuda_sync(sync_cuda_timing)
        if trace_predict:
            print(
                "[pose] predict_call_start "
                f"frame={int(raw_frame_idx)} source_shape={tuple(frame_for_model.shape)} "
                f"device={self._yolo_predict_device} half={bool(self.config.use_half)} "
                f"imgsz={predict_kwargs.get('imgsz', None)} max_det={predict_kwargs['max_det']}",
                flush=True,
            )
        t_pose0 = time.perf_counter()
        with _temporary_torch_from_numpy_fallback(enabled=bool(self._yolo_is_engine)), torch.inference_mode():
            results = self._pose_model.predict(**predict_kwargs)
        if trace_predict:
            print(
                "[pose] predict_call_done "
                f"frame={int(raw_frame_idx)} elapsed_ms={(time.perf_counter() - t_pose0) * 1000.0:.1f}",
                flush=True,
            )
        _maybe_cuda_sync(sync_cuda_timing)
        if trace_predict:
            print(f"[pose] predict_post_sync_done frame={int(raw_frame_idx)}", flush=True)
        pose_infer_ms = (time.perf_counter() - t_pose0) * 1000.0

        t_track0 = time.perf_counter()
        xy_zeros = np.zeros((K, 2), dtype=np.float32)
        cf_zeros = np.zeros((K,), dtype=np.float32)

        found = False
        out_xy = xy_zeros
        out_cf = cf_zeros

        if results and len(results) > 0 and results[0].keypoints is not None:
            r = results[0]
            kpts = r.keypoints
            xy_all = kpts.xy.cpu().numpy() if hasattr(kpts.xy, "cpu") else np.asarray(kpts.xy)
            cf_all = kpts.conf.cpu().numpy() if (hasattr(kpts, "conf") and hasattr(kpts.conf, "cpu")) else None
            boxes_xyxy, box_conf = _extract_boxes_xyxy_conf(r)

            if letterbox_info is not None:
                xy_all = _scale_xy_from_letterbox(
                    xy_all,
                    ratio=float(letterbox_info["ratio"]),
                    pad_left=int(letterbox_info["pad_left"]),
                    pad_top=int(letterbox_info["pad_top"]),
                    orig_shape=(int(frame_bgr.shape[0]), int(frame_bgr.shape[1])),
                )
                if boxes_xyxy.shape[0] > 0:
                    boxes_xyxy = _scale_boxes_from_letterbox(
                        boxes_xyxy,
                        ratio=float(letterbox_info["ratio"]),
                        pad_left=int(letterbox_info["pad_left"]),
                        pad_top=int(letterbox_info["pad_top"]),
                        orig_shape=(int(frame_bgr.shape[0]), int(frame_bgr.shape[1])),
                    )

            if xy_all.ndim == 3 and xy_all.shape[0] > 0 and xy_all.shape[1] == K:
                num_candidates = int(xy_all.shape[0])
                if boxes_xyxy.shape[0] > 0:
                    num_candidates = min(num_candidates, int(boxes_xyxy.shape[0]))
                    box_centers = np.column_stack(
                        (
                            0.5 * (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]),
                            0.5 * (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]),
                        )
                    ).astype(np.float32, copy=False)
                else:
                    box_centers = np.mean(xy_all, axis=1).astype(np.float32, copy=False)

                if cf_all is not None and cf_all.ndim == 2:
                    num_candidates = min(num_candidates, int(cf_all.shape[0]))
                else:
                    cf_all = None

                if box_conf is not None:
                    num_candidates = min(num_candidates, int(box_conf.shape[0]))

                if num_candidates > 0:
                    xy_all = xy_all[:num_candidates]
                    box_centers = box_centers[:num_candidates]
                    if cf_all is not None:
                        cf_all = cf_all[:num_candidates]
                    if box_conf is not None:
                        box_conf = box_conf[:num_candidates]

                    idx, new_center = select_person_idx(
                        box_centers=box_centers,
                        box_conf=box_conf,
                        prev_center=self._prev_center,
                        target_center=target_center,
                        conf_min=float(self.config.track_conf_min),
                        max_jump_px=float(max_jump_px),
                    )
                    if idx is not None:
                        found = True
                        out_xy = xy_all[idx].astype(np.float32, copy=False)
                        if cf_all is not None and idx < cf_all.shape[0] and cf_all[idx].shape[0] == K:
                            out_cf = cf_all[idx].astype(np.float32, copy=False)
                        else:
                            out_cf = np.ones((K,), dtype=np.float32)
                        self._prev_center = new_center

        if found:
            self._lost_count = 0
        else:
            self._lost_count += 1
            if int(self._lost_count) > int(self.config.track_max_lost):
                self._prev_center = None

        track_ms = (time.perf_counter() - t_track0) * 1000.0

        self._last_xy = out_xy.copy()
        self._last_cf = out_cf.copy()

        sampled_idx = int(self._sampled_count)
        self._sampled_count += 1

        return PoseFrameOutput(
            raw_frame_idx=int(raw_frame_idx),
            sampled=True,
            sampled_frame_idx=int(sampled_idx),
            found=bool(found),
            keypoints_xy=out_xy,
            keypoints_conf=out_cf,
            pose_infer_ms=float(pose_infer_ms),
            track_ms=float(track_ms),
        )
