import json
import glob
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch


# ----------------------------- CONFIG -----------------------------

@dataclass
class AlphaPoseExportConfig:
    # AlphaPose repo root (relative to project root by default)
    alphapose_root: str = "pose_models/AlphaPose"

    # FastPose COCO-17 config + checkpoint/engine
    cfg_path: str = "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
    checkpoint: str = "pretrained_models/fast_res50_256x192.pth"

    # YOLOv3-SPP detector cfg + weights/engine
    detector_cfg: str = "detector/yolo/cfg/yolov3-spp.cfg"
    detector_weights: str = "detector/yolo/data/yolov3-spp.weights"

    conf_thres: float = 0.1
    conf_min: float = 0.75
    nms_thres: float = 0.6
    fps: int = 30
    max_people: int = 1
    num_kpts: int = 17
    video_codec: str = "mp4v"
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
    draw_confidence_text: bool = True
    min_box_area: int = 0
    flip: bool = False
    vis_fast: bool = True
    render_video: bool = True
    device: Optional[str] = None  # "cuda" or "cpu"


# ----------------------------- SORTING -----------------------------

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


def frame_time_key(path: str) -> pd.Timestamp:
    return parse_frame_timestamp(path)


def list_frames(frames_dir: str, pattern: str = "*.png") -> List[str]:
    paths = glob.glob(os.path.join(frames_dir, pattern))
    paths = sorted(paths, key=frame_time_key)
    if not paths:
        raise FileNotFoundError(f"No frames found in {frames_dir} matching {pattern}")
    return paths


# ----------------------------- IO HELPERS -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def make_video_writer(out_path: str, fps: int, frame_size: Tuple[int, int], codec: str = "mp4v") -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    w, h = frame_size
    return cv2.VideoWriter(out_path, fourcc, fps, (w, h))


def is_engine_weights_path(weights_path: str) -> bool:
    return Path(weights_path).suffix.lower() == ".engine"


def _strip_tensorrt_engine_header(raw: bytes) -> Optional[bytes]:
    """
    Some exporters prepend serialized engines with:
      [4-byte metadata length][JSON metadata][engine bytes]
    Strip that wrapper if present.
    """
    if len(raw) <= 4:
        return None

    meta_len = int.from_bytes(raw[:4], byteorder="little", signed=False)
    max_reasonable = min(len(raw) - 4, 8 * 1024 * 1024)
    if meta_len <= 0 or meta_len > int(max_reasonable):
        return None

    meta_raw = raw[4 : 4 + meta_len]
    try:
        meta = json.loads(meta_raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    return raw[4 + meta_len :]


def _torch_dtype_from_numpy(np_dtype: np.dtype) -> torch.dtype:
    d = np.dtype(np_dtype)
    if d == np.float16:
        return torch.float16
    if d == np.float32:
        return torch.float32
    if d == np.float64:
        return torch.float64
    if d == np.int8:
        return torch.int8
    if d == np.int16:
        return torch.int16
    if d == np.int32:
        return torch.int32
    if d == np.int64:
        return torch.int64
    if d == np.uint8:
        return torch.uint8
    if d == np.bool_:
        return torch.bool
    raise TypeError(f"Unsupported numpy dtype for TensorRT tensor IO: {d}")


class TensorRTEngineRunner:
    """
    Minimal TensorRT runtime wrapper for CUDA inference on numpy batches.
    Supports both legacy binding APIs and newer name-based TensorRT APIs.
    """

    def __init__(self, engine_path: Path, device: str = "cuda"):
        self.engine_path = Path(engine_path)
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"TensorRT .engine requires CUDA, but CUDA is unavailable. Cannot load: {self.engine_path}"
            )

        device_str = str(device).strip().lower()
        if not device_str.startswith("cuda"):
            raise RuntimeError(
                f"TensorRT .engine requires a CUDA device, but got {device!r} for {self.engine_path}."
            )
        self.device = torch.device(device)

        try:
            import tensorrt as trt  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "TensorRT python package is required for .engine inference. "
                "Install the matching `tensorrt` wheel for your CUDA/TensorRT stack."
            ) from exc

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        raw = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(raw)
        if self.engine is None:
            stripped = _strip_tensorrt_engine_header(raw)
            if stripped is not None:
                self.engine = self.runtime.deserialize_cuda_engine(stripped)
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {self.engine_path}. "
                "Check that the engine matches this TensorRT runtime."
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context for {self.engine_path}")

        self._name_api = hasattr(self.engine, "num_io_tensors") and hasattr(self.engine, "get_tensor_name")

        self.input_name: str
        self.input_index: Optional[int]
        self.input_shape_template: Tuple[int, ...]
        self.input_np_dtype: np.dtype
        self.output_names: List[str] = []
        self.output_indices: List[int] = []

        if self._name_api:
            n_io = int(self.engine.num_io_tensors)
            input_names: List[str] = []
            output_names: List[str] = []
            for i in range(n_io):
                name = str(self.engine.get_tensor_name(i))
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.INPUT:
                    input_names.append(name)
                elif mode == self.trt.TensorIOMode.OUTPUT:
                    output_names.append(name)

            if len(input_names) != 1:
                raise RuntimeError(
                    f"Expected exactly 1 TensorRT input tensor, found {len(input_names)} in {self.engine_path}"
                )
            if not output_names:
                raise RuntimeError(f"No TensorRT output tensors found in {self.engine_path}")

            self.input_name = input_names[0]
            self.input_index = None
            self.output_names = output_names
            self.input_shape_template = tuple(int(v) for v in self.engine.get_tensor_shape(self.input_name))
            self.input_np_dtype = self._trt_dtype_to_numpy(self.engine.get_tensor_dtype(self.input_name))
        else:
            n_bindings = int(self.engine.num_bindings)
            input_indices = [i for i in range(n_bindings) if bool(self.engine.binding_is_input(i))]
            output_indices = [i for i in range(n_bindings) if not bool(self.engine.binding_is_input(i))]
            if len(input_indices) != 1:
                raise RuntimeError(
                    f"Expected exactly 1 TensorRT input binding, found {len(input_indices)} in {self.engine_path}"
                )
            if not output_indices:
                raise RuntimeError(f"No TensorRT output bindings found in {self.engine_path}")

            self.input_index = int(input_indices[0])
            self.output_indices = [int(i) for i in output_indices]
            self.input_name = str(self.engine.get_binding_name(self.input_index))
            self.output_names = [str(self.engine.get_binding_name(i)) for i in self.output_indices]
            self.input_shape_template = tuple(int(v) for v in self.engine.get_binding_shape(self.input_index))
            self.input_np_dtype = self._trt_dtype_to_numpy(self.engine.get_binding_dtype(self.input_index))

        self.static_batch_size: Optional[int]
        if self.input_shape_template and int(self.input_shape_template[0]) > 0:
            self.static_batch_size = int(self.input_shape_template[0])
        else:
            self.static_batch_size = None

    def _trt_dtype_to_numpy(self, trt_dtype) -> np.dtype:
        try:
            return np.dtype(self.trt.nptype(trt_dtype))
        except Exception:
            s = str(trt_dtype).lower()
            if "float16" in s or "half" in s:
                return np.dtype(np.float16)
            if "float32" in s:
                return np.dtype(np.float32)
            if "int8" in s:
                return np.dtype(np.int8)
            if "int32" in s:
                return np.dtype(np.int32)
            if "bool" in s:
                return np.dtype(np.bool_)
            return np.dtype(np.float32)

    def _set_input_shape_if_needed(self, shape: Tuple[int, ...]) -> None:
        shape_t = tuple(int(v) for v in shape)
        if len(shape_t) != len(self.input_shape_template):
            raise ValueError(
                f"TensorRT input rank mismatch for {self.engine_path}: "
                f"expected rank={len(self.input_shape_template)}, got shape={shape_t}"
            )

        for dim_i, (expect, got) in enumerate(zip(self.input_shape_template, shape_t)):
            if int(expect) >= 0 and int(expect) != int(got):
                raise ValueError(
                    f"Input shape mismatch at dim={dim_i} for {self.engine_path}: "
                    f"engine expects {self.input_shape_template}, got {shape_t}"
                )

        if any(int(v) < 0 for v in self.input_shape_template):
            if self._name_api and hasattr(self.context, "set_input_shape"):
                ok = self.context.set_input_shape(self.input_name, shape_t)
                if ok is False:
                    raise RuntimeError(
                        f"Failed to set TensorRT dynamic input shape {shape_t} on tensor '{self.input_name}'"
                    )
            elif self.input_index is not None and hasattr(self.context, "set_binding_shape"):
                ok = self.context.set_binding_shape(int(self.input_index), shape_t)
                if ok is False:
                    raise RuntimeError(
                        f"Failed to set TensorRT dynamic input shape {shape_t} on binding index {self.input_index}"
                    )
            else:
                raise RuntimeError(
                    "TensorRT engine uses dynamic shapes, but this runtime/context does not expose "
                    "set_input_shape or set_binding_shape."
                )

    def infer(self, x_batch: np.ndarray) -> List[np.ndarray]:
        x_np = np.asarray(x_batch)
        if x_np.ndim != len(self.input_shape_template):
            raise ValueError(
                f"TensorRT input rank mismatch for {self.engine_path}: "
                f"expected rank={len(self.input_shape_template)}, got shape={tuple(x_np.shape)}"
            )

        if x_np.dtype != self.input_np_dtype:
            x_np = x_np.astype(self.input_np_dtype, copy=False)
        if not x_np.flags.c_contiguous:
            x_np = np.ascontiguousarray(x_np)

        x_t = torch.as_tensor(
            x_np,
            device=self.device,
            dtype=_torch_dtype_from_numpy(self.input_np_dtype),
        )
        self._set_input_shape_if_needed(tuple(int(v) for v in x_t.shape))

        if self._name_api and hasattr(self.context, "set_tensor_address"):
            output_tensors: Dict[str, torch.Tensor] = {}
            self.context.set_tensor_address(self.input_name, int(x_t.data_ptr()))
            for out_name in self.output_names:
                out_shape = tuple(int(v) for v in self.context.get_tensor_shape(out_name))
                if any(int(v) < 0 for v in out_shape):
                    raise RuntimeError(
                        f"Unresolved dynamic output shape for tensor '{out_name}' in {self.engine_path}: {out_shape}"
                    )
                out_dtype = self._trt_dtype_to_numpy(self.engine.get_tensor_dtype(out_name))
                out_t = torch.empty(
                    out_shape,
                    device=self.device,
                    dtype=_torch_dtype_from_numpy(out_dtype),
                )
                self.context.set_tensor_address(out_name, int(out_t.data_ptr()))
                output_tensors[out_name] = out_t

            if hasattr(self.context, "execute_async_v3"):
                stream = torch.cuda.current_stream(device=self.device)
                ok = self.context.execute_async_v3(stream_handle=int(stream.cuda_stream))
            elif hasattr(self.context, "execute_v3"):
                ok = self.context.execute_v3()
            else:
                if not hasattr(self.engine, "num_bindings") or not hasattr(self.engine, "get_binding_index"):
                    raise RuntimeError("TensorRT context lacks supported execution APIs.")
                bindings = [0] * int(self.engine.num_bindings)
                in_idx = int(self.engine.get_binding_index(self.input_name))
                bindings[in_idx] = int(x_t.data_ptr())
                for out_name, out_t in output_tensors.items():
                    out_idx = int(self.engine.get_binding_index(out_name))
                    bindings[out_idx] = int(out_t.data_ptr())
                ok = self.context.execute_v2(bindings)

            if not ok:
                raise RuntimeError(f"TensorRT execution failed for {self.engine_path}")
            torch.cuda.synchronize(self.device)
            return [output_tensors[name].detach().cpu().numpy() for name in self.output_names]

        if self.input_index is None or not hasattr(self.engine, "num_bindings"):
            raise RuntimeError("TensorRT runtime does not expose supported tensor-address or binding APIs.")

        n_bindings = int(self.engine.num_bindings)
        bindings = [0] * n_bindings
        bindings[int(self.input_index)] = int(x_t.data_ptr())

        out_tensors: List[torch.Tensor] = []
        for out_idx in self.output_indices:
            out_shape = tuple(int(v) for v in self.context.get_binding_shape(int(out_idx)))
            if any(int(v) < 0 for v in out_shape):
                raise RuntimeError(
                    f"Unresolved dynamic output shape for binding index {out_idx} in {self.engine_path}: {out_shape}"
                )
            out_dtype = self._trt_dtype_to_numpy(self.engine.get_binding_dtype(int(out_idx)))
            out_t = torch.empty(
                out_shape,
                device=self.device,
                dtype=_torch_dtype_from_numpy(out_dtype),
            )
            bindings[int(out_idx)] = int(out_t.data_ptr())
            out_tensors.append(out_t)

        ok = self.context.execute_v2(bindings)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path}")
        torch.cuda.synchronize(self.device)
        return [t.detach().cpu().numpy() for t in out_tensors]


def _infer_engine_outputs_batched(engine_runner: TensorRTEngineRunner, x_batch: np.ndarray) -> List[np.ndarray]:
    x_np = np.asarray(x_batch)
    if x_np.ndim < 1:
        raise ValueError("TensorRT inference batch must have at least one dimension.")

    static_bs = engine_runner.static_batch_size
    batch_size = int(x_np.shape[0])
    if static_bs is None or static_bs <= 0 or batch_size == static_bs:
        try:
            return engine_runner.infer(x_np)
        except RuntimeError as exc:
            msg = str(exc).lower()
            profile_mismatch = (
                "satisfyprofile" in msg
                or "optimization profile" in msg
                or "failed to set tensorrt dynamic input shape" in msg
            )
            if not profile_mismatch or batch_size <= 1:
                raise

            mid = max(1, batch_size // 2)
            left_outputs = _infer_engine_outputs_batched(engine_runner, x_np[:mid])
            right_outputs = _infer_engine_outputs_batched(engine_runner, x_np[mid:])
            if len(left_outputs) != len(right_outputs):
                raise RuntimeError(
                    "TensorRT engine returned an inconsistent number of outputs across split batches."
                ) from exc

            merged: List[np.ndarray] = []
            for left_arr, right_arr in zip(left_outputs, right_outputs):
                left_np = np.asarray(left_arr)
                right_np = np.asarray(right_arr)
                if left_np.ndim == 0 or right_np.ndim == 0:
                    raise RuntimeError(
                        "TensorRT engine output cannot be merged after split-batch fallback."
                    ) from exc
                merged.append(np.concatenate((left_np, right_np), axis=0))
            return merged

    out_parts: Optional[List[List[np.ndarray]]] = None
    start = 0
    while start < batch_size:
        end = min(start + int(static_bs), batch_size)
        chunk = x_np[start:end]
        chunk_len = int(end - start)
        if chunk_len < int(static_bs):
            pad_shape = (int(static_bs) - chunk_len, *chunk.shape[1:])
            pad = np.zeros(pad_shape, dtype=chunk.dtype)
            chunk = np.concatenate((chunk, pad), axis=0)

        chunk_outputs = engine_runner.infer(chunk)
        if out_parts is None:
            out_parts = [[] for _ in chunk_outputs]
        elif len(chunk_outputs) != len(out_parts):
            raise RuntimeError("TensorRT engine returned an inconsistent number of outputs across batches.")

        for out_idx, arr in enumerate(chunk_outputs):
            arr_np = np.asarray(arr)
            if arr_np.ndim >= 1 and arr_np.shape[0] == int(static_bs):
                arr_np = arr_np[:chunk_len]
            elif arr_np.ndim == 0 and chunk_len != 1:
                raise RuntimeError(
                    f"TensorRT engine output shape {arr_np.shape} cannot be split into batch chunks."
                )
            out_parts[out_idx].append(arr_np)
        start = end

    if out_parts is None:
        return []

    merged: List[np.ndarray] = []
    for parts in out_parts:
        first = np.asarray(parts[0])
        if first.ndim == 0:
            merged.append(first)
        else:
            merged.append(np.concatenate(parts, axis=0))
    return merged


def _coerce_int_id(value: Any) -> int:
    if value is None:
        return 0
    if torch.is_tensor(value):
        if value.numel() == 0:
            return 0
        return int(value.view(-1)[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0
        return _coerce_int_id(value[0])
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _normalize_ids(ids: Any, count: int) -> List[int]:
    if ids is None:
        return [0] * count
    if torch.is_tensor(ids):
        if ids.numel() == 0:
            return [0] * count
        values = [_coerce_int_id(v) for v in ids.view(-1)]
    elif isinstance(ids, np.ndarray):
        if ids.size == 0:
            return [0] * count
        values = [_coerce_int_id(v) for v in ids.reshape(-1)]
    elif isinstance(ids, (list, tuple)):
        values = [_coerce_int_id(v) for v in ids]
    else:
        values = [_coerce_int_id(ids)]

    if len(values) < count:
        values.extend([0] * (count - len(values)))
    elif len(values) > count:
        values = values[:count]
    return values


# ----------------------------- LABELS -----------------------------

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

    if not locked:
        candidate_idx = np.arange(num_people, dtype=np.int32)
        if box_conf is not None:
            high_conf = np.where(np.isfinite(box_conf[:num_people]) & (box_conf[:num_people] >= conf_min))[0]
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

    if not strict_reacquire:
        ranked = valid_jump[np.argsort(dists[valid_jump])]
        best_idx = int(ranked[0])
        return best_idx, box_centers[best_idx].astype(np.float32, copy=True)

    prev_area = _box_area_xyxy(prev_box_xyxy)
    if prev_area <= 0.0:
        return None, prev_center

    area_ratio_limit = max(1.0, float(max_box_area_ratio))
    min_area_ratio = 1.0 / area_ratio_limit
    min_iou = max(0.0, float(min_iou_same_track))

    best_idx: Optional[int] = None
    best_iou = -1.0
    best_dist = float("inf")

    for idx in valid_jump.tolist():
        if box_conf is not None and idx < box_conf.shape[0]:
            conf_val = float(box_conf[idx])
            if np.isfinite(conf_val) and conf_val < conf_min:
                continue

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
        return None, prev_center

    return best_idx, box_centers[best_idx].astype(np.float32, copy=True)


def draw_selected_pose(
    frame: np.ndarray,
    kpts_xy: Optional[np.ndarray],
    kpts_conf: Optional[np.ndarray],
    box_xyxy: Optional[np.ndarray],
    person_conf: float,
    draw_kpt_threshold: float,
    draw_no_target_text: bool,
    draw_confidence_text: bool,
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

    text_origin = (12, 30)
    if box_xyxy is not None:
        box_xyxy = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)
        if box_xyxy.shape[0] >= 4 and np.all(np.isfinite(box_xyxy[:4])):
            x1, y1, x2, y2 = np.round(box_xyxy[:4]).astype(int).tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            text_origin = (x1, max(20, y1 - 8))

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

    if draw_confidence_text and np.isfinite(person_conf):
        cv2.putText(
            out,
            f"CONF {float(person_conf):.3f}",
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


class TensorRTYOLODetector:
    """
    TensorRT-backed replacement for AlphaPose's YOLOv3 detector.
    Assumes the engine preserves Darknet YOLOv3-SPP input preprocessing and either:
      - returns a single transformed detection tensor shaped like (B, N, 5 + num_classes), or
      - returns one raw YOLO head tensor per detection scale.
    """

    def __init__(self, model_cfg: str, engine_path: str, opt: "_Opt", confidence: float, nms_thres: float):
        from detector.yolo.bbox import bbox_iou
        from detector.yolo.darknet import Darknet
        from detector.yolo.preprocess import prep_frame, prep_image
        from detector.yolo.util import predict_transform

        self.model_cfg = str(model_cfg)
        self.model_weights = str(engine_path)
        self.detector_opt = opt
        self.confidence = float(confidence)
        self.nms_thres = float(nms_thres)
        self.model = None

        darknet = Darknet(self.model_cfg)
        net_info = getattr(darknet, "net_info", {})
        self.inp_dim = int(net_info.get("height", 608))
        self.num_classes = self._extract_num_classes(getattr(darknet, "blocks", []))
        self.yolo_anchors = self._extract_yolo_anchors(darknet)
        del darknet

        self._bbox_iou = bbox_iou
        self._predict_transform = predict_transform
        self._prep_frame = prep_frame
        self._prep_image = prep_image
        self._nms_wrapper = None
        if platform.system() != "Windows":
            try:
                from detector.nms import nms_wrapper
                self._nms_wrapper = nms_wrapper
            except Exception:
                self._nms_wrapper = None

        self._engine_runner = TensorRTEngineRunner(Path(self.model_weights), device=str(opt.device))

    def _extract_num_classes(self, blocks: List[Dict[str, Any]]) -> int:
        for block in reversed(blocks):
            if str(block.get("type", "")).lower() == "yolo" and "classes" in block:
                return int(block["classes"])
        return 80

    def _extract_yolo_anchors(self, darknet) -> List[List[Tuple[float, float]]]:
        anchors: List[List[Tuple[float, float]]] = []
        for module in getattr(darknet, "module_list", []):
            if len(module) > 0 and hasattr(module[0], "anchors"):
                anchors.append(list(module[0].anchors))
        return anchors

    def image_preprocess(self, img_source):
        if isinstance(img_source, str):
            img, _, _ = self._prep_image(img_source, self.inp_dim)
            return img
        if isinstance(img_source, (torch.Tensor, np.ndarray)):
            img, _, _ = self._prep_frame(img_source, self.inp_dim)
            return img
        raise IOError(f"Unknown image source type: {type(img_source)}")

    def _outputs_to_prediction(self, outputs: List[np.ndarray], batch_size: int) -> torch.Tensor:
        expected_attrs = 5 + int(self.num_classes)

        for out in outputs:
            arr = np.asarray(out)
            if arr.ndim == 2 and batch_size == 1 and arr.shape[-1] >= expected_attrs:
                arr = arr[np.newaxis, ...]
            if arr.ndim == 3 and arr.shape[0] == batch_size and arr.shape[-1] >= expected_attrs:
                return torch.as_tensor(arr, device=self.detector_opt.device)

        four_d = [np.asarray(out) for out in outputs if np.asarray(out).ndim == 4]
        if four_d and len(four_d) == len(self.yolo_anchors):
            pred_parts: List[torch.Tensor] = []
            for out_arr, anchors in zip(sorted(four_d, key=lambda a: int(a.shape[2])), self.yolo_anchors):
                raw = torch.as_tensor(out_arr, device=self.detector_opt.device)
                pred_parts.append(
                    self._predict_transform(
                        raw,
                        self.inp_dim,
                        anchors,
                        self.num_classes,
                        self.detector_opt,
                    )
                )
            return torch.cat(pred_parts, dim=1)

        output_shapes = ", ".join(str(tuple(np.asarray(out).shape)) for out in outputs)
        raise RuntimeError(
            "Unsupported TensorRT detector outputs for AlphaPose YOLOv3-SPP. "
            f"Expected one transformed output or {len(self.yolo_anchors)} raw YOLO heads, got: {output_shapes}"
        )

    def images_detection(self, imgs, orig_dim_list):
        args = self.detector_opt
        if isinstance(imgs, torch.Tensor):
            x_batch = imgs.detach().cpu().numpy()
        else:
            x_batch = np.asarray(imgs)
        if x_batch.ndim == 3:
            x_batch = np.expand_dims(x_batch, axis=0)

        outputs = _infer_engine_outputs_batched(self._engine_runner, x_batch)
        prediction = self._outputs_to_prediction(outputs, batch_size=int(x_batch.shape[0]))

        dets = self.dynamic_write_results(
            prediction,
            self.confidence,
            self.num_classes,
            nms=True,
            nms_conf=self.nms_thres,
        )
        if isinstance(dets, int) or dets.shape[0] == 0:
            return 0

        dets = dets.cpu()
        orig_dim_list = torch.index_select(orig_dim_list, 0, dets[:, 0].long())
        scaling_factor = torch.min(self.inp_dim / orig_dim_list, 1)[0].view(-1, 1)
        dets[:, [1, 3]] -= (self.inp_dim - scaling_factor * orig_dim_list[:, 0].view(-1, 1)) / 2
        dets[:, [2, 4]] -= (self.inp_dim - scaling_factor * orig_dim_list[:, 1].view(-1, 1)) / 2
        dets[:, 1:5] /= scaling_factor
        for i in range(dets.shape[0]):
            dets[i, [1, 3]] = torch.clamp(dets[i, [1, 3]], 0.0, orig_dim_list[i, 0])
            dets[i, [2, 4]] = torch.clamp(dets[i, [2, 4]], 0.0, orig_dim_list[i, 1])

        return dets

    def dynamic_write_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):
        prediction_bak = prediction.clone()
        dets = self.write_results(prediction.clone(), confidence, num_classes, nms, nms_conf)
        if isinstance(dets, int):
            return dets

        if dets.shape[0] > 100:
            nms_conf -= 0.05
            dets = self.write_results(prediction_bak.clone(), confidence, num_classes, nms, nms_conf)

        return dets

    def write_results(self, prediction, confidence, num_classes, nms=True, nms_conf=0.4):
        args = self.detector_opt
        conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
        prediction = prediction * conf_mask

        try:
            torch.nonzero(prediction[:, :, 4]).transpose(0, 1).contiguous()
        except Exception:
            return 0

        box_a = prediction.new(prediction.shape)
        box_a[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_a[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_a[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_a[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_a[:, :, :4]

        batch_size = prediction.size(0)
        output = prediction.new(1, prediction.size(2) + 1)
        write = False
        num = 0
        for ind in range(batch_size):
            image_pred = prediction[ind]

            max_conf, max_conf_score = torch.max(image_pred[:, 5 : 5 + num_classes], 1)
            max_conf = max_conf.float().unsqueeze(1)
            max_conf_score = max_conf_score.float().unsqueeze(1)
            image_pred = torch.cat((image_pred[:, :5], max_conf, max_conf_score), 1)

            non_zero_ind = torch.nonzero(image_pred[:, 4])
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

            try:
                img_classes = torch.unique(image_pred_[:, -1])
            except Exception:
                continue

            for cls in img_classes:
                if cls != 0:
                    continue

                cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
                class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
                image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

                conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
                image_pred_class = image_pred_class[conf_sort_index]

                if nms:
                    if self._nms_wrapper is not None:
                        _, inds = self._nms_wrapper.nms(image_pred_class[:, :5], nms_conf)
                        image_pred_class = image_pred_class[inds]
                    else:
                        max_detections = []
                        while image_pred_class.size(0):
                            max_detections.append(image_pred_class[0].unsqueeze(0))
                            if len(image_pred_class) == 1:
                                break
                            ious = self._bbox_iou(max_detections[-1], image_pred_class[1:], args)
                            image_pred_class = image_pred_class[1:][ious < nms_conf]
                        image_pred_class = torch.cat(max_detections).data

                batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
                seq = batch_ind, image_pred_class
                if not write:
                    output = torch.cat(seq, 1)
                    write = True
                else:
                    out = torch.cat(seq, 1)
                    output = torch.cat((output, out))
                num += 1

        if not num:
            return 0
        return output

    def detect_one_img(self, img_name):
        raise NotImplementedError("TensorRTYOLODetector.detect_one_img is not used by this pipeline.")


# ----------------------------- ALPHAPOSE WRAPPER -----------------------------

class AlphaPoseRunner:
    def __init__(self, config: AlphaPoseExportConfig):
        self.config = config
        self.root = Path(config.alphapose_root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(f"AlphaPose root not found: {self.root}")

        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))

        # Resolve model and detector paths to absolute locations
        self.cfg_path = self._resolve_path(config.cfg_path)
        self.checkpoint_path = self._resolve_path(config.checkpoint)
        self.detector_cfg_path = self._resolve_path(config.detector_cfg)
        self.detector_weights_path = self._resolve_path(config.detector_weights)
        self.pose_is_engine = is_engine_weights_path(self.checkpoint_path)
        self.detector_is_engine = is_engine_weights_path(self.detector_weights_path)

        if not Path(self.cfg_path).exists():
            raise FileNotFoundError(f"AlphaPose cfg not found: {self.cfg_path}")
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"AlphaPose checkpoint not found: {self.checkpoint_path}")
        if not Path(self.detector_cfg_path).exists():
            raise FileNotFoundError(f"YOLOv3-SPP cfg not found: {self.detector_cfg_path}")
        if not Path(self.detector_weights_path).exists():
            raise FileNotFoundError(
                f"YOLOv3-SPP weights/engine not found: {self.detector_weights_path}."
            )

        from alphapose.utils.config import update_config
        from alphapose.models import builder
        from alphapose.utils.transforms import flip, flip_heatmap, get_func_heatmap_to_coord
        from alphapose.utils.pPose_nms import pose_nms
        from alphapose.utils.presets import SimpleTransform
        if not self.detector_is_engine:
            from detector.apis import get_detector
            from detector import yolo_cfg as yolo_cfg_mod

        self.flip = flip
        self.flip_heatmap = flip_heatmap
        self.pose_nms = pose_nms
        self.SimpleTransform = SimpleTransform

        self.cfg = update_config(self.cfg_path)

        # Device selection
        if config.device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            requested_device = str(config.device).strip().lower()
        use_cuda = requested_device.startswith("cuda")
        if use_cuda and not torch.cuda.is_available():
            use_cuda = False
            requested_device = "cpu"
        if (self.pose_is_engine or self.detector_is_engine) and not use_cuda:
            raise RuntimeError(
                "TensorRT .engine weights require CUDA. "
                f"Resolved device: {requested_device!r}, "
                f"pose_engine={self.pose_is_engine}, detector_engine={self.detector_is_engine}"
            )

        gpu_index = 0
        if use_cuda and requested_device.startswith("cuda:"):
            maybe_idx = requested_device.split(":", 1)[1].strip()
            if maybe_idx.isdigit():
                gpu_index = int(maybe_idx)

        gpus = [gpu_index] if use_cuda else [-1]
        device = torch.device(f"cuda:{gpu_index}" if gpus[0] >= 0 else "cpu")
        self.device = device

        # Minimal opt namespace required by AlphaPose detector/vis
        self.opt = _Opt(
            detector="yolo",
            gpus=gpus,
            device=device,
            tracking=False,
            pose_track=False,
            pose_flow=False,
            min_box_area=config.min_box_area,
            flip=config.flip,
            vis_fast=config.vis_fast,
            showbox=False,
        )

        # Build detector backend
        if self.detector_is_engine:
            self.detector = TensorRTYOLODetector(
                model_cfg=self.detector_cfg_path,
                engine_path=self.detector_weights_path,
                opt=self.opt,
                confidence=float(config.conf_thres),
                nms_thres=float(config.nms_thres),
            )
        else:
            # Override detector cfg to absolute paths and requested thresholds.
            yolo_cfg_mod.cfg.CONFIG = self.detector_cfg_path
            yolo_cfg_mod.cfg.WEIGHTS = self.detector_weights_path
            yolo_cfg_mod.cfg.CONFIDENCE = float(config.conf_thres)
            yolo_cfg_mod.cfg.NMS_THRES = float(config.nms_thres)
            self.detector = get_detector(self.opt)

        # Build pose backend
        self.pose_engine_runner: Optional[TensorRTEngineRunner] = None
        self.pose_model = None
        if self.pose_is_engine:
            self.pose_engine_runner = TensorRTEngineRunner(Path(self.checkpoint_path), device=str(device))
        else:
            self.pose_model = builder.build_sppe(self.cfg.MODEL, preset_cfg=self.cfg.DATA_PRESET)
            self.pose_model.load_state_dict(torch.load(self.checkpoint_path, map_location=device))
            self.pose_model.to(device).eval()

        self.pose_dataset = builder.retrieve_dataset(self.cfg.DATASET.TRAIN)
        self.heatmap_to_coord = get_func_heatmap_to_coord(self.cfg)
        self.norm_type = self.cfg.LOSS.get("NORM_TYPE", None)
        self.hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        self.eval_joints = list(range(self.cfg.DATA_PRESET.NUM_JOINTS))

        if config.num_kpts != self.cfg.DATA_PRESET.NUM_JOINTS:
            raise ValueError(
                f"config.num_kpts={config.num_kpts} does not match model joints "
                f"{self.cfg.DATA_PRESET.NUM_JOINTS}"
            )

        # Preprocess transform for crops
        self.transformation = self.SimpleTransform(
            self.pose_dataset,
            scale_factor=0,
            input_size=self.cfg.DATA_PRESET.IMAGE_SIZE,
            output_size=self.cfg.DATA_PRESET.HEATMAP_SIZE,
            rot=0,
            sigma=self.cfg.DATA_PRESET.SIGMA,
            train=False,
            add_dpg=False,
            gpu_device=device,
        )

        loss_type = self.cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss")
        num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
        self.hand_face_num = None
        if loss_type == "MSELoss":
            self.vis_thres = [0.4] * num_joints
        elif "JointRegression" in loss_type:
            self.vis_thres = [0.05] * num_joints
        elif loss_type == "Combined":
            if num_joints == 68:
                self.hand_face_num = 42
            else:
                self.hand_face_num = 110
            self.vis_thres = [0.4] * (num_joints - self.hand_face_num) + [0.05] * self.hand_face_num

        self.use_heatmap_loss = (loss_type == "MSELoss")

    def _resolve_path(self, p: str) -> str:
        path = Path(p).expanduser()
        if path.is_absolute():
            return str(path.resolve())

        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return str(cwd_path)

        return str((self.root / path).resolve())

    def _infer_pose_engine(self, inps: torch.Tensor) -> torch.Tensor:
        if self.pose_engine_runner is None:
            raise RuntimeError("Pose engine runner is not initialised.")

        x_batch = inps.detach().cpu().numpy()
        outputs = _infer_engine_outputs_batched(self.pose_engine_runner, x_batch)
        if not outputs:
            raise RuntimeError("TensorRT pose engine returned no outputs.")

        arr: Optional[np.ndarray] = None
        for out in outputs:
            out_np = np.asarray(out)
            if out_np.ndim >= 3 and out_np.shape[0] == int(x_batch.shape[0]):
                arr = out_np
                break
        if arr is None:
            arr = np.asarray(outputs[0])

        if arr.ndim == 3 and int(x_batch.shape[0]) == 1:
            arr = arr[np.newaxis, ...]
        if arr.ndim < 3:
            raise RuntimeError(
                "Unsupported TensorRT FastPose output. "
                f"Expected rank >= 3, got shape={tuple(arr.shape)} from {self.checkpoint_path}"
            )

        return torch.from_numpy(np.ascontiguousarray(arr))

    def infer(self, image_rgb: np.ndarray, image_name: str) -> Dict[str, Any]:
        """
        Returns a dict with:
          - im_res: AlphaPose-style result for visualization
          - people: list of per-person dicts with numpy keypoints for export
        """
        if image_rgb is None:
            return {"im_res": {"imgname": image_name, "result": []}, "people": []}

        img = self.detector.image_preprocess(image_rgb)
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.dim() == 3:
            img = img.unsqueeze(0)

        im_dim = image_rgb.shape[1], image_rgb.shape[0]
        im_dim_list = torch.FloatTensor(im_dim).repeat(1, 2)

        with torch.no_grad():
            dets = self.detector.images_detection(img, im_dim_list)
            if isinstance(dets, int) or dets is None or dets.shape[0] == 0:
                return {"im_res": {"imgname": image_name, "result": []}, "people": []}
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()

            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = torch.zeros(scores.shape)

            boxes = boxes[dets[:, 0] == 0]
            scores = scores[dets[:, 0] == 0]
            ids = ids[dets[:, 0] == 0]
            if boxes.nelement() == 0:
                return {"im_res": {"imgname": image_name, "result": []}, "people": []}

            inps = torch.zeros(boxes.size(0), 3, *self.cfg.DATA_PRESET.IMAGE_SIZE)
            cropped_boxes = torch.zeros(boxes.size(0), 4)

            for i, box in enumerate(boxes):
                inps[i], cropped_box = self.transformation.test_transform(image_rgb, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)

            inps = inps.to(self.opt.device)
            if self.config.flip:
                inps = torch.cat((inps, self.flip(inps)))

            if self.pose_engine_runner is not None:
                hm = self._infer_pose_engine(inps)
            else:
                if self.pose_model is None:
                    raise RuntimeError("Pose backend is not initialised.")
                hm = self.pose_model(inps)
            if self.config.flip:
                hm_flip = self.flip_heatmap(hm[int(len(hm) / 2):], self.pose_dataset.joint_pairs, shift=True)
                hm = (hm[0:int(len(hm) / 2)] + hm_flip) / 2
            hm = hm.cpu()

            pose_coords = []
            pose_scores = []
            for i in range(hm.shape[0]):
                bbox = cropped_boxes[i].tolist()
                if isinstance(self.heatmap_to_coord, list):
                    if self.hand_face_num is None:
                        raise ValueError("Combined loss requires hand_face_num to be set")
                    pose_coords_body, pose_scores_body = self.heatmap_to_coord[0](
                        hm[i][self.eval_joints[:-self.hand_face_num]],
                        bbox,
                        hm_shape=self.hm_size,
                        norm_type=self.norm_type,
                    )
                    pose_coords_fh, pose_scores_fh = self.heatmap_to_coord[1](
                        hm[i][self.eval_joints[-self.hand_face_num:]],
                        bbox,
                        hm_shape=self.hm_size,
                        norm_type=self.norm_type,
                    )
                    pose_coord = np.concatenate((pose_coords_body, pose_coords_fh), axis=0)
                    pose_score = np.concatenate((pose_scores_body, pose_scores_fh), axis=0)
                else:
                    pose_coord, pose_score = self.heatmap_to_coord(
                        hm[i][self.eval_joints], bbox, hm_shape=self.hm_size, norm_type=self.norm_type
                    )
                pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

            preds_img = torch.cat(pose_coords)
            preds_scores = torch.cat(pose_scores)

            boxes, scores, ids, preds_img, preds_scores, _ = self.pose_nms(
                boxes, scores, ids, preds_img, preds_scores, self.config.min_box_area,
                use_heatmap_loss=self.use_heatmap_loss
            )
            norm_ids = _normalize_ids(ids, len(scores))

            result_list = []
            people = []
            for k in range(len(scores)):
                kp = preds_img[k]
                kp_score = preds_scores[k]
                proposal_score = torch.mean(kp_score) + scores[k] + 1.25 * torch.max(kp_score)
                box = boxes[k]
                box_xyxy = np.asarray(box.cpu().numpy() if torch.is_tensor(box) else box, dtype=np.float32).reshape(-1)
                if box_xyxy.shape[0] < 4:
                    continue
                box_xyxy = box_xyxy[:4]
                score_k = scores[k]
                if torch.is_tensor(score_k):
                    score_k_val = float(score_k.reshape(-1)[0].item())
                else:
                    score_k_val = float(score_k)
                box_xywh = [
                    float(box_xyxy[0]),
                    float(box_xyxy[1]),
                    float(box_xyxy[2] - box_xyxy[0]),
                    float(box_xyxy[3] - box_xyxy[1]),
                ]
                result_list.append(
                    {
                        "keypoints": kp,
                        "kp_score": kp_score,
                        "proposal_score": float(proposal_score),
                        "idx": norm_ids[k],
                        "box": box_xywh,
                    }
                )

                people.append(
                    {
                        "kpts_xy": kp.cpu().numpy().astype(np.float32),
                        "kpts_conf": kp_score.cpu().numpy().astype(np.float32).reshape(-1),
                        "box_xyxy": box_xyxy.astype(np.float32, copy=False),
                        "box_center": np.array(
                            [
                                0.5 * (box_xyxy[0] + box_xyxy[2]),
                                0.5 * (box_xyxy[1] + box_xyxy[3]),
                            ],
                            dtype=np.float32,
                        ),
                        "box_conf": score_k_val,
                        "person_conf": float(proposal_score),
                    }
                )

            im_res = {"imgname": image_name, "result": result_list}
            return {"im_res": im_res, "people": people}

    def render(self, frame_bgr: np.ndarray, im_res: Dict[str, Any]) -> np.ndarray:
        if not im_res or not im_res.get("result"):
            return frame_bgr

        if self.config.vis_fast:
            from alphapose.utils.vis import vis_frame_fast as vis_frame
        else:
            from alphapose.utils.vis import vis_frame

        return vis_frame(frame_bgr, im_res, self.opt, self.vis_thres.copy())


class _Opt:
    def __init__(
        self,
        detector: str,
        gpus: List[int],
        device: torch.device,
        tracking: bool,
        pose_track: bool,
        pose_flow: bool,
        min_box_area: int,
        flip: bool,
        vis_fast: bool,
        showbox: bool,
    ):
        self.detector = detector
        self.gpus = gpus
        self.device = device
        self.tracking = tracking
        self.pose_track = pose_track
        self.pose_flow = pose_flow
        self.min_box_area = min_box_area
        self.flip = flip
        self.vis_fast = vis_fast
        self.showbox = showbox
        self.sp = True
        self.save_img = False
        self.vis = False
        self.format = None
        self.eval = False


# ----------------------------- CORE PIPELINE -----------------------------

def run_pose_on_frames_alphapose(
    frames_dir: str,
    out_dir: str,
    windows_csv: str,
    config: AlphaPoseExportConfig,
    pattern: str = "*.png",
    runner: Optional[AlphaPoseRunner] = None,
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

    if runner is None:
        runner = AlphaPoseRunner(config)

    first_bgr = cv2.imread(frame_paths[0])
    if first_bgr is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")
    h, w = first_bgr.shape[:2]

    out_video = os.path.join(out_dir, "pose_out.mp4")
    out_npz = os.path.join(out_dir, "keypoints.npz")
    out_csv = os.path.join(out_dir, "keypoints.csv") if config.save_csv else None

    writer = make_video_writer(out_video, config.fps, (w, h), codec=config.video_codec)

    arrays = init_pose_arrays(num_frames, config.max_people, config.num_kpts)

    csv_rows = []
    if config.save_csv:
        csv_rows.append(["frame", "person", "kpt", "x", "y", "kpt_conf", "person_conf", "frame_path"])

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
    max_jump_px = (
        float(config.max_jump_px)
        if config.max_jump_px is not None
        else float(config.max_jump_diag_frac * frame_diag)
    )
    prev_center: Optional[np.ndarray] = None
    prev_box_xyxy: Optional[np.ndarray] = None
    track_locked = False
    lost_count = 0

    for i, p in enumerate(frame_paths):
        frame_bgr = cv2.imread(p)
        if frame_bgr is None:
            print(f"Skipping unreadable frame: {p}")
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_name = os.path.basename(p)
        out = runner.infer(frame_rgb, image_name=image_name)
        people = out["people"]
        selected_xy: Optional[np.ndarray] = None
        selected_kc: Optional[np.ndarray] = None
        selected_box_xyxy: Optional[np.ndarray] = None
        selected_person_conf = float("nan")

        if not people:
            lost_count += 1
        else:
            xy = np.asarray([person["kpts_xy"] for person in people], dtype=np.float32)
            kc = np.asarray([person["kpts_conf"] for person in people], dtype=np.float32)
            box_centers = np.asarray([person["box_center"] for person in people], dtype=np.float32)
            boxes_xyxy = np.asarray([person["box_xyxy"] for person in people], dtype=np.float32)
            box_conf = np.asarray(
                [float(person.get("box_conf", np.nan)) for person in people],
                dtype=np.float32,
            )

            num_candidates = min(
                int(xy.shape[0]),
                int(kc.shape[0]),
                int(box_centers.shape[0]),
                int(boxes_xyxy.shape[0]),
            )
            if num_candidates <= 0:
                lost_count += 1
            else:
                xy = xy[:num_candidates]
                kc = kc[:num_candidates]
                box_centers = box_centers[:num_candidates]
                boxes_xyxy = boxes_xyxy[:num_candidates]
                box_conf = box_conf[:num_candidates]

                locked_for_selection = bool(
                    track_locked or (
                        (not config.lock_first_target) and
                        (prev_center is not None) and
                        (prev_box_xyxy is not None)
                    )
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
                )

                if idx is None:
                    lost_count += 1
                else:
                    prev_center = new_center
                    prev_box_xyxy = np.asarray(boxes_xyxy[idx], dtype=np.float32).copy()
                    if config.lock_first_target and not track_locked:
                        track_locked = True
                    lost_count = 0

                    selected_xy = xy[idx]
                    selected_kc = kc[idx]
                    selected_box_xyxy = prev_box_xyxy

                    if idx < box_conf.shape[0] and np.isfinite(box_conf[idx]):
                        selected_person_conf = float(box_conf[idx])
                    elif selected_kc is not None:
                        selected_person_conf = float(np.nanmean(selected_kc))
                    elif idx < len(people):
                        selected_person_conf = float(people[idx].get("person_conf", np.nan))

                    if config.max_people > 0:
                        j = 0
                        xy_sel = np.asarray(selected_xy, dtype=np.float32)
                        xy_count = min(config.num_kpts, int(xy_sel.shape[0]))
                        arrays["kpts_xy"][i, j, :xy_count] = xy_sel[:xy_count]

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

        if config.render_video:
            annotated = draw_selected_pose(
                frame=frame_bgr,
                kpts_xy=selected_xy,
                kpts_conf=selected_kc,
                box_xyxy=selected_box_xyxy,
                person_conf=selected_person_conf,
                draw_kpt_threshold=config.draw_kpt_threshold,
                draw_no_target_text=config.draw_no_target_text,
                draw_confidence_text=config.draw_confidence_text,
            )
            writer.write(annotated)
        else:
            writer.write(frame_bgr)

    writer.release()

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
        model_path=np.array([str(Path(config.checkpoint))], dtype=object),
    )

    if config.save_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerows(csv_rows)

    return out_video, out_npz, out_csv
