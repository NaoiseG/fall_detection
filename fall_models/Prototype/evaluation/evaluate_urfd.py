#!/usr/bin/env python3
"""
URFD one-shot evaluation for the repo's shared final temporal pipeline.

Assumption:
This script targets the same shared YOLO + GenericTemporalAdapter path used by
`inference/inference_on_video.py` and the benchmarked final pipeline runs under
`benchmarks/img_downsize/final_pipelines/...`. It therefore reuses the existing
pose loader, tracking, feature construction, temporal window assembly, and
classifier checkpoint loading already present in the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

IMPORT_ERROR: Optional[Exception] = None

try:
    import numpy as np
    import torch
    import cv2
    from inference.benchmark_core import _assemble_window
    from inference.classifier_adapters import (
        GenericAdapterConfig,
        GenericTemporalAdapter,
        Prediction,
        TemporalClassifierAdapter,
        _rf_predict_proba_aligned,
        pick_device,
    )
    from inference.pose_pipeline import PosePipeline, PosePipelineConfig
except Exception as exc:  # pragma: no cover - import depends on local runtime env
    IMPORT_ERROR = exc
    np = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]
    cv2 = None  # type: ignore[assignment]
    _assemble_window = None  # type: ignore[assignment]
    GenericAdapterConfig = None  # type: ignore[assignment]
    GenericTemporalAdapter = None  # type: ignore[assignment]
    Prediction = None  # type: ignore[assignment]
    TemporalClassifierAdapter = None  # type: ignore[assignment]
    _rf_predict_proba_aligned = None  # type: ignore[assignment]
    pick_device = None  # type: ignore[assignment]
    PosePipeline = None  # type: ignore[assignment]
    PosePipelineConfig = None  # type: ignore[assignment]

if torch is not None:
    _no_grad = torch.no_grad
else:  # pragma: no cover - used only when runtime deps are missing
    def _no_grad():
        def decorator(fn):
            return fn
        return decorator

LOGGER = logging.getLogger("evaluate_urfd")

DATASET_NAME = "URFD"
DEFAULT_FPS = 30.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_CONSECUTIVE_POSITIVE = 3
DEFAULT_OUTPUT_DIR = Path("outputs") / "urfd_eval"
DEFAULT_FRAME_EXTS = (".png", ".jpg", ".jpeg")
DEFAULT_TEST_SEQUENCES_PER_CLASS = 5

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


@dataclass(frozen=True)
class URFDSequence:
    dataset: str
    video_id: str
    video_label: str
    sequence_root: Path
    frame_dir: Path
    frame_paths: Tuple[Path, ...]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-shot inference-only evaluation of the existing final keypoint + "
            "temporal classifier pipeline on the URFD dataset."
        )
    )
    parser.add_argument(
        "--urfd-root",
        type=Path,
        required=True,
        help="Path to the URFD root containing ADLs/, Falls/, and optionally cvat/.",
    )
    parser.add_argument(
        "--keypoint-weights",
        type=Path,
        required=True,
        help="Path to existing pose/keypoint weights used by the final pipeline (.pt or .engine).",
    )
    parser.add_argument(
        "--classifier-model",
        type=Path,
        required=True,
        help="Path to the existing trained temporal classifier checkpoint or model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where window_predictions.csv, video_summary.csv, metrics.json, and run_config.json are written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device for both pose and classifier. Defaults to CUDA if available, else CPU.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help=(
            "Optional raw-frame temporal window length override. If omitted, reuse the classifier checkpoint's "
            "existing window length."
        ),
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help=(
            "Optional raw-frame temporal stride override. If omitted, reuse the classifier checkpoint's "
            "existing stride."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Frames-per-second used to convert frame indices to seconds in outputs. Default: 30.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Fall-score threshold for a positive window. Default: 0.5.",
    )
    parser.add_argument(
        "--min-consecutive-positive",
        type=int,
        default=DEFAULT_MIN_CONSECUTIVE_POSITIVE,
        help="Minimum consecutive positive windows required to mark a sequence as detected_fall. Default: 3.",
    )
    parser.add_argument(
        "--frame-exts",
        nargs="*",
        default=None,
        help="Optional frame extensions to search for. Defaults to: png jpg jpeg.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run a small sanity-check subset using up to 5 ADL and 5 Fall sequences "
            f"(or fewer if the dataset contains less)."
        ),
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Optional classifier architecture override. Usually inferred from --classifier-model.",
    )
    parser.add_argument(
        "--imgsz",
        type=float,
        default=None,
        help=(
            "Optional YOLO predict size override. If omitted, the script tries to infer imgsz from the keypoint "
            "weights path and otherwise falls back to 640."
        ),
    )
    parser.add_argument(
        "--half",
        type=int,
        choices=(0, 1),
        default=None,
        help="Optional FP16 toggle for the pose model on CUDA. If omitted, fp16 is inferred from the weights path.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Raw-frame subsampling factor before temporal windowing. Default: 1.",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO pose confidence threshold. Default: 0.25.")
    parser.add_argument("--yolo-iou", type=float, default=None, help="Optional YOLO NMS IoU threshold.")
    parser.add_argument("--max-people", type=int, default=10, help="Maximum people to consider for pose detection/tracking. Default: 10.")
    parser.add_argument(
        "--max-det",
        type=int,
        default=0,
        help="Optional YOLO max_det override. Values <= 0 reuse --max-people.",
    )
    parser.add_argument(
        "--track-conf-min",
        type=float,
        default=0.75,
        help="Minimum box confidence used when initializing person tracking. Default: 0.75.",
    )
    parser.add_argument(
        "--track-max-jump-px",
        type=float,
        default=0.0,
        help="Absolute max per-frame tracking jump in pixels. Default 0 means use diagonal fraction.",
    )
    parser.add_argument(
        "--track-max-jump-diag-frac",
        type=float,
        default=0.25,
        help="Fallback max jump as a fraction of frame diagonal. Default: 0.25.",
    )
    parser.add_argument(
        "--track-max-lost",
        type=int,
        default=10,
        help="How many sampled misses to tolerate before resetting the tracker. Default: 10.",
    )
    parser.add_argument(
        "--track-target-x-frac",
        type=float,
        default=0.5,
        help="Tracking target x-position as a fraction of frame width. Default: 0.5.",
    )
    parser.add_argument(
        "--track-target-y-frac",
        type=float,
        default=0.5,
        help="Tracking target y-position as a fraction of frame height. Default: 0.5.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log verbosity. Default: INFO.",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level).upper(), logging.INFO), format="[%(levelname)s] %(message)s")


def normalize_frame_exts(values: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not values:
        return DEFAULT_FRAME_EXTS
    normalized: List[str] = []
    for raw in values:
        item = str(raw).strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        if item not in normalized:
            normalized.append(item)
    return tuple(normalized or DEFAULT_FRAME_EXTS)


def natural_sort_key(value: Any) -> List[Any]:
    text = str(value)
    parts = re.split(r"(\d+)", text.lower())
    key: List[Any] = []
    for part in parts:
        if not part:
            continue
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def iter_progress(iterable: Iterable[Any], *, desc: str, total: Optional[int] = None, leave: bool = True) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=leave, dynamic_ncols=True)


def path_has_images(path: Path, frame_exts: Sequence[str]) -> bool:
    ext_set = {ext.lower() for ext in frame_exts}
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() in ext_set:
                return True
    except OSError:
        return False
    return False


def collect_frame_paths(frame_dir: Path, frame_exts: Sequence[str]) -> List[Path]:
    ext_set = {ext.lower() for ext in frame_exts}
    frames = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in ext_set]
    frames.sort(key=lambda p: natural_sort_key(p.name))
    return frames


def resolve_sequence_frame_dir(sequence_root: Path, frame_exts: Sequence[str]) -> Optional[Path]:
    current = sequence_root
    seen: set[Path] = set()
    while True:
        repeated = current / current.name
        if repeated in seen or (not repeated.is_dir()):
            break
        seen.add(repeated)
        current = repeated

    if current.is_dir() and path_has_images(current, frame_exts):
        return current
    if sequence_root.is_dir() and path_has_images(sequence_root, frame_exts):
        return sequence_root

    candidates: List[Path] = []
    try:
        for path in sequence_root.rglob("*"):
            if path.is_dir() and path_has_images(path, frame_exts):
                candidates.append(path)
    except OSError:
        return None

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (
            len(p.relative_to(sequence_root).parts),
            1 if p.name == sequence_root.name else 0,
            natural_sort_key(p.as_posix()),
        )
    )
    return candidates[-1]


def discover_urfd_sequences(urfd_root: Path, frame_exts: Sequence[str]) -> List[URFDSequence]:
    label_dirs = (("ADLs", "non_fall"), ("Falls", "fall"))
    sequences: List[URFDSequence] = []

    for dirname, video_label in label_dirs:
        root = urfd_root / dirname
        if not root.exists():
            LOGGER.warning("Missing expected directory: %s", root)
            continue
        if not root.is_dir():
            LOGGER.warning("Expected directory but found non-directory: %s", root)
            continue

        children = [p for p in root.iterdir() if p.is_dir()]
        children.sort(key=lambda p: natural_sort_key(p.name))
        for sequence_root in children:
            frame_dir = resolve_sequence_frame_dir(sequence_root, frame_exts)
            if frame_dir is None:
                LOGGER.warning("No frame directory found for sequence: %s", sequence_root)
                continue
            try:
                frame_paths = tuple(collect_frame_paths(frame_dir, frame_exts))
            except OSError as exc:
                LOGGER.warning("Failed to list frames for %s: %s", frame_dir, exc)
                continue
            sequences.append(
                URFDSequence(
                    dataset=DATASET_NAME,
                    video_id=sequence_root.name,
                    video_label=video_label,
                    sequence_root=sequence_root,
                    frame_dir=frame_dir,
                    frame_paths=frame_paths,
                )
            )

    sequences.sort(key=lambda seq: (0 if seq.video_label == "non_fall" else 1, natural_sort_key(seq.video_id)))
    return sequences


def select_test_sequences(
    sequences: Sequence[URFDSequence],
    *,
    max_per_label: int = DEFAULT_TEST_SEQUENCES_PER_CLASS,
) -> List[URFDSequence]:
    limit = max(1, int(max_per_label))
    selected: List[URFDSequence] = []
    seen_per_label = {"non_fall": 0, "fall": 0}

    for sequence in sequences:
        label = str(sequence.video_label)
        if label not in seen_per_label:
            continue
        if int(seen_per_label[label]) >= limit:
            continue
        selected.append(sequence)
        seen_per_label[label] += 1

    return selected


def infer_imgsz_from_path(weights_path: Path) -> Optional[float]:
    text = weights_path.as_posix().lower()
    match = re.search(r"imgsz[_-]?(\d+)", text)
    if match:
        return float(match.group(1))
    for part in weights_path.parts:
        match = re.fullmatch(r"(\d{3,4})", part)
        if match:
            return float(match.group(1))
    return None


def resolve_pose_half_arg(explicit_half: Optional[int], weights_path: Path, device: str) -> bool:
    if not str(device).lower().startswith("cuda"):
        return False
    if explicit_half is not None:
        return bool(int(explicit_half))
    return "fp16" in weights_path.as_posix().lower()


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float(num) / float(den)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_pose_pipeline(args: argparse.Namespace, *, device: str, keypoint_weights: Path, imgsz: float) -> PosePipeline:
    max_det = int(args.max_det) if int(args.max_det) > 0 else int(args.max_people)
    return PosePipeline(
        PosePipelineConfig(
            yolo_weights=keypoint_weights,
            device=str(device),
            imgsz=float(imgsz),
            yolo_conf=float(args.yolo_conf),
            yolo_iou=float(args.yolo_iou) if args.yolo_iou is not None else None,
            max_det=int(max_det),
            use_half=bool(resolve_pose_half_arg(args.half, keypoint_weights, device)),
            frame_step=max(1, int(args.frame_step)),
            track_conf_min=float(args.track_conf_min),
            track_max_jump_px=float(args.track_max_jump_px),
            track_max_jump_diag_frac=float(args.track_max_jump_diag_frac),
            track_max_lost=int(args.track_max_lost),
            track_target_x_frac=float(args.track_target_x_frac),
            track_target_y_frac=float(args.track_target_y_frac),
        )
    )


def build_classifier_adapter(args: argparse.Namespace, *, device: str) -> GenericTemporalAdapter:
    return GenericTemporalAdapter(
        GenericAdapterConfig(
            model_arg=str(args.classifier_model),
            arch_arg=args.arch,
            device=device,
            frame_step=max(1, int(args.frame_step)),
            half=False,
            T_override=int(args.window_size) if args.window_size is not None else 0,
            stride_override=int(args.stride) if args.stride is not None else 0,
        )
    )


@_no_grad()
def compute_generic_probs(adapter: GenericTemporalAdapter, prepared_input: Any) -> Tuple[np.ndarray, Optional[float]]:
    if adapter._is_rf:  # type: ignore[attr-defined]
        probs = _rf_predict_proba_aligned(
            adapter._rf_model,  # type: ignore[attr-defined]
            prepared_input,
            num_classes=int(adapter._display_num_classes),  # type: ignore[attr-defined]
        )[0]
        return np.asarray(probs, dtype=np.float32), float(probs[0]) if len(probs) > 0 else None

    if adapter._model is None:  # type: ignore[attr-defined]
        raise RuntimeError("Temporal model is not initialized.")

    xb = torch.from_numpy(np.asarray(prepared_input)[None, ...]).to(adapter.device)
    xb = xb.half() if bool(adapter.use_half_temporal) else xb.float()
    out = adapter._model(xb)  # type: ignore[attr-defined]

    fall_logit = None
    if isinstance(out, (tuple, list)) and len(out) == 2:
        logits, fall_logit = out[0], out[1]
    else:
        logits = out

    if logits.ndim == 3:
        logits = logits[:, -1, :]

    probs_t = torch.softmax(logits, dim=-1)
    if bool(adapter.merge_fall_11_to_7):
        if int(probs_t.shape[-1]) != 11:
            raise ValueError(f"Expected 11-class logits before merge, got shape {tuple(probs_t.shape)}")
        probs_t = torch.cat([probs_t[:, :5].sum(dim=1, keepdim=True), probs_t[:, 5:]], dim=1)

    probs = probs_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    explicit_fall_prob = None
    if fall_logit is not None:
        explicit_fall_prob = float(torch.sigmoid(fall_logit.view(-1))[0].item())
    return probs, explicit_fall_prob


def resolve_fall_score(
    adapter: TemporalClassifierAdapter,
    prepared_input: Any,
    prediction: Prediction,
) -> Tuple[float, Optional[np.ndarray], str]:
    extra = prediction.extra if isinstance(prediction.extra, dict) else {}
    probs_raw = extra.get("probs")
    if isinstance(probs_raw, (list, tuple)) and len(probs_raw) > 0:
        probs_np = np.asarray(probs_raw, dtype=np.float32)
        if probs_np.ndim == 1 and probs_np.shape[0] > 0:
            return float(probs_np[0]), probs_np, "prediction.extra.probs"

    p_fall = extra.get("p_fall")
    if isinstance(adapter, GenericTemporalAdapter):
        probs_np, explicit_fall_prob = compute_generic_probs(adapter, prepared_input)
        if probs_np.ndim == 1 and probs_np.shape[0] > 0:
            return float(probs_np[0]), probs_np, "generic_softmax_class0"
        if explicit_fall_prob is not None:
            return float(explicit_fall_prob), None, "generic_fall_head"

    if p_fall is not None and np.isfinite(float(p_fall)):
        return float(p_fall), None, "prediction.extra.p_fall"

    if prediction.pred_id == 0 and prediction.confidence is not None and np.isfinite(float(prediction.confidence)):
        return float(prediction.confidence), None, "predicted_class_confidence_fallback"

    return 0.0, None, "default_zero"


def read_frame_with_fallback(path: Path, last_shape: Optional[Tuple[int, int, int]]) -> Tuple[Optional[np.ndarray], bool]:
    frame = cv2.imread(str(path))
    if frame is not None:
        return frame, False
    if last_shape is not None:
        return np.zeros(last_shape, dtype=np.uint8), True
    return None, True


def frame_time_seconds(frame_idx_0based: int, fps: float) -> float:
    return float(frame_idx_0based) / float(fps)


def to_human_frame(frame_idx_0based: int) -> int:
    return int(frame_idx_0based) + 1


def _append_window_row(
    rows: List[Dict[str, Any]],
    *,
    sequence: URFDSequence,
    window_id: int,
    window_data: Any,
    prediction: Prediction,
    fall_score: float,
    probs_np: Optional[np.ndarray],
    fall_score_source: str,
    threshold: float,
    prep_metrics: Dict[str, float],
    infer_metrics: Dict[str, float],
    assembly_ms: float,
    fps: float,
    conf_thres: float,
) -> None:
    valid_pose_frames = int(np.sum(np.any(window_data.conf_seq > float(conf_thres), axis=1)))
    missing_pose_frames = int(window_data.conf_seq.shape[0] - valid_pose_frames)

    row = {
        "dataset": sequence.dataset,
        "video_id": sequence.video_id,
        "video_path": str(sequence.frame_dir),
        "video_label": sequence.video_label,
        "window_id": int(window_id),
        "start_frame": to_human_frame(int(window_data.raw_start_idx)),
        "end_frame": to_human_frame(int(window_data.raw_end_idx)),
        "start_time_s": frame_time_seconds(int(window_data.raw_start_idx), fps),
        "end_time_s": frame_time_seconds(int(window_data.raw_end_idx), fps),
        "num_frames_in_window": int(window_data.raw_end_idx - window_data.raw_start_idx + 1),
        "fall_score": float(fall_score),
        "predicted_label": "fall" if int(prediction.pred_id) == 0 else "non_fall",
        "threshold": float(threshold),
        "is_positive": bool(float(fall_score) >= float(threshold)),
        "consecutive_positive_count": 0,
        "event_decision": False,
        "predicted_class_id": int(prediction.pred_id),
        "predicted_class_name": str(prediction.pred_label),
        "prediction_confidence": float(prediction.confidence) if prediction.confidence is not None else None,
        "fall_score_source": str(fall_score_source),
        "window_start_raw_idx": int(window_data.raw_start_idx),
        "window_end_raw_idx": int(window_data.raw_end_idx),
        "sampled_start_idx": int(window_data.sampled_start_idx),
        "sampled_end_idx": int(window_data.sampled_end_idx),
        "num_sampled_frames_in_window": int(window_data.xy_seq.shape[0]),
        "valid_pose_frames_in_window": int(valid_pose_frames),
        "missing_pose_frames_in_window": int(missing_pose_frames),
        "window_assembly_ms": float(assembly_ms),
        "temporal_prep_ms": float(prep_metrics.get("temporal_prep_ms", 0.0)),
        "temporal_forward_ms": float(infer_metrics.get("temporal_forward_ms", 0.0)),
        "temporal_total_ms": float(prep_metrics.get("temporal_prep_ms", 0.0)) + float(infer_metrics.get("temporal_forward_ms", 0.0)),
    }
    if probs_np is not None:
        row["class_probs"] = json.dumps([float(x) for x in probs_np.tolist()])
    rows.append(row)


def evaluate_sequence(
    sequence: URFDSequence,
    *,
    pose_pipeline: PosePipeline,
    classifier: GenericTemporalAdapter,
    fps: float,
    threshold: float,
    min_consecutive_positive: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pose_pipeline.reset_tracking_state()

    window_rows: List[Dict[str, Any]] = []
    sampled_xy: List[np.ndarray] = []
    sampled_cf: List[np.ndarray] = []
    sampled_raw_idx: List[int] = []

    last_frame_shape: Optional[Tuple[int, int, int]] = None
    readable_frames = 0
    unreadable_frames = 0
    pose_found_frames = 0
    sampled_pose_found_frames = 0
    next_window_id = 0
    next_window_start = 0
    image_shape_hw: Optional[Tuple[int, int]] = None
    policy = classifier.window_policy
    conf_thres = float(getattr(classifier, "conf_thres", 0.0))

    frame_iter: Iterable[Tuple[int, Path]] = enumerate(sequence.frame_paths)
    frame_iter = iter_progress(frame_iter, desc=f"{sequence.video_id}", total=len(sequence.frame_paths), leave=False)

    for raw_frame_idx, frame_path in frame_iter:
        frame, used_blank_fallback = read_frame_with_fallback(frame_path, last_frame_shape)
        if frame is None:
            unreadable_frames += 1
            LOGGER.warning("Unreadable frame with no known shape; skipping %s", frame_path)
            continue

        if used_blank_fallback:
            unreadable_frames += 1
            LOGGER.warning("Unreadable frame; substituting a blank frame for %s", frame_path)
        else:
            readable_frames += 1
            last_frame_shape = tuple(int(v) for v in frame.shape)
            image_shape_hw = (int(frame.shape[0]), int(frame.shape[1]))

        pose_out = pose_pipeline.process_frame(frame_bgr=frame, raw_frame_idx=int(raw_frame_idx), sync_cuda_timing=False)
        if pose_out.found:
            pose_found_frames += 1
        if pose_out.sampled:
            sampled_xy.append(pose_out.keypoints_xy.copy())
            sampled_cf.append(pose_out.keypoints_conf.copy())
            sampled_raw_idx.append(int(raw_frame_idx))
            if pose_out.found:
                sampled_pose_found_frames += 1

        while True:
            assembled = _assemble_window(
                sampled_xy=sampled_xy,
                sampled_cf=sampled_cf,
                sampled_raw_idx=sampled_raw_idx,
                sampled_start=int(next_window_start),
                sampled_len=int(policy.sampled_window_len),
                cap_done=False,
                pad_tail=False,
                image_shape=(int(image_shape_hw[0]), int(image_shape_hw[1])) if image_shape_hw is not None else (0, 0),
                video_stem=sequence.video_id,
            )
            if assembled is None:
                break

            window_data, assembly_ms = assembled
            prepared_input, prep_metrics = classifier.prepare_window(window_data=window_data, sync_cuda_timing=False)
            prediction, infer_metrics = classifier.infer(prepared_input=prepared_input, sync_cuda_timing=False)
            fall_score, probs_np, fall_score_source = resolve_fall_score(classifier, prepared_input, prediction)
            _append_window_row(
                window_rows,
                sequence=sequence,
                window_id=next_window_id,
                window_data=window_data,
                prediction=prediction,
                fall_score=fall_score,
                probs_np=probs_np,
                fall_score_source=fall_score_source,
                threshold=threshold,
                prep_metrics=prep_metrics,
                infer_metrics=infer_metrics,
                assembly_ms=assembly_ms,
                fps=fps,
                conf_thres=conf_thres,
            )
            next_window_id += 1
            next_window_start += int(policy.sampled_window_stride)

    if image_shape_hw is not None:
        while True:
            assembled = _assemble_window(
                sampled_xy=sampled_xy,
                sampled_cf=sampled_cf,
                sampled_raw_idx=sampled_raw_idx,
                sampled_start=int(next_window_start),
                sampled_len=int(policy.sampled_window_len),
                cap_done=True,
                pad_tail=False,
                image_shape=(int(image_shape_hw[0]), int(image_shape_hw[1])),
                video_stem=sequence.video_id,
            )
            if assembled is None:
                break

            window_data, assembly_ms = assembled
            prepared_input, prep_metrics = classifier.prepare_window(window_data=window_data, sync_cuda_timing=False)
            prediction, infer_metrics = classifier.infer(prepared_input=prepared_input, sync_cuda_timing=False)
            fall_score, probs_np, fall_score_source = resolve_fall_score(classifier, prepared_input, prediction)
            _append_window_row(
                window_rows,
                sequence=sequence,
                window_id=next_window_id,
                window_data=window_data,
                prediction=prediction,
                fall_score=fall_score,
                probs_np=probs_np,
                fall_score_source=fall_score_source,
                threshold=threshold,
                prep_metrics=prep_metrics,
                infer_metrics=infer_metrics,
                assembly_ms=assembly_ms,
                fps=fps,
                conf_thres=conf_thres,
            )
            next_window_id += 1
            next_window_start += int(policy.sampled_window_stride)

    consecutive_positive = 0
    detected_fall = False
    detection_run_start_row: Optional[Dict[str, Any]] = None
    max_fall_score = 0.0
    num_positive_windows = 0

    for idx, row in enumerate(window_rows):
        is_positive = bool(row["is_positive"])
        if is_positive:
            num_positive_windows += 1
            consecutive_positive += 1
        else:
            consecutive_positive = 0

        row["consecutive_positive_count"] = int(consecutive_positive)
        max_fall_score = max(max_fall_score, float(row["fall_score"]))

        if (not detected_fall) and consecutive_positive >= int(min_consecutive_positive):
            detected_fall = True
            start_idx = idx - int(min_consecutive_positive) + 1
            if start_idx < 0:
                start_idx = 0
            detection_run_start_row = window_rows[start_idx]
        row["event_decision"] = bool(detected_fall)

    first_positive_window_id: Optional[int] = None
    for row in window_rows:
        if bool(row["is_positive"]):
            first_positive_window_id = int(row["window_id"])
            break

    first_detection_frame: Optional[int] = None
    first_detection_time_s: Optional[float] = None
    first_detection_window_id: Optional[int] = None
    if detection_run_start_row is not None:
        first_detection_window_id = int(detection_run_start_row["window_id"])
        first_detection_frame = int(detection_run_start_row["start_frame"])
        first_detection_time_s = float(detection_run_start_row["start_time_s"])

    is_true_fall = sequence.video_label == "fall"
    if is_true_fall and detected_fall:
        outcome = "TP"
    elif (not is_true_fall) and (not detected_fall):
        outcome = "TN"
    elif (not is_true_fall) and detected_fall:
        outcome = "FP"
    else:
        outcome = "FN"

    video_summary = {
        "dataset": sequence.dataset,
        "video_id": sequence.video_id,
        "video_path": str(sequence.frame_dir),
        "video_label": sequence.video_label,
        "num_frames": int(len(sequence.frame_paths)),
        "num_windows": int(len(window_rows)),
        "num_positive_windows": int(num_positive_windows),
        "max_fall_score": float(max_fall_score),
        "first_positive_window_id": first_positive_window_id,
        "first_detection_frame": first_detection_frame,
        "first_detection_time_s": first_detection_time_s,
        "detected_fall": bool(detected_fall),
        "outcome": outcome,
        "first_detection_window_id": first_detection_window_id,
        "num_readable_frames": int(readable_frames),
        "num_unreadable_frames": int(unreadable_frames),
        "pose_found_frames": int(pose_found_frames),
        "sampled_pose_found_frames": int(sampled_pose_found_frames),
        "sequence_root": str(sequence.sequence_root),
        "frame_dir": str(sequence.frame_dir),
    }
    return window_rows, video_summary


def compute_metrics(video_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(1 for row in video_rows if row.get("outcome") == "TP")
    tn = sum(1 for row in video_rows if row.get("outcome") == "TN")
    fp = sum(1 for row in video_rows if row.get("outcome") == "FP")
    fn = sum(1 for row in video_rows if row.get("outcome") == "FN")

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (recall + specificity)

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "num_videos": int(len(video_rows)),
        "num_fall_videos": int(sum(1 for row in video_rows if row.get("video_label") == "fall")),
        "num_non_fall_videos": int(sum(1 for row in video_rows if row.get("video_label") == "non_fall")),
    }


def print_metrics(metrics: Dict[str, Any]) -> None:
    print()
    print("URFD video-level metrics")
    print(f"  TP: {metrics['tp']}  TN: {metrics['tn']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f}")
    print(f"  F1:                 {metrics['f1']:.4f}")
    print(f"  Balanced accuracy:  {metrics['balanced_accuracy']:.4f}")


def main() -> int:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing runtime dependency needed for URFD evaluation. "
            "Make sure OpenCV, Ultralytics, PyTorch, and the repo inference dependencies are installed. "
            f"Original import error: {IMPORT_ERROR}"
        ) from IMPORT_ERROR

    if args.window_size is not None and int(args.window_size) <= 0:
        raise ValueError("--window-size must be > 0 when provided.")
    if args.stride is not None and int(args.stride) <= 0:
        raise ValueError("--stride must be > 0 when provided.")
    if float(args.fps) <= 0.0 or (not math.isfinite(float(args.fps))):
        raise ValueError("--fps must be a finite value > 0.")
    if (not math.isfinite(float(args.threshold))) or float(args.threshold) < 0.0 or float(args.threshold) > 1.0:
        raise ValueError("--threshold must be a finite value in [0, 1].")
    if int(args.min_consecutive_positive) <= 0:
        raise ValueError("--min-consecutive-positive must be >= 1.")
    if int(args.frame_step) <= 0:
        raise ValueError("--frame-step must be >= 1.")
    if int(args.max_people) <= 0:
        raise ValueError("--max-people must be >= 1.")
    if int(args.max_det) < 0:
        raise ValueError("--max-det must be >= 0.")

    urfd_root = args.urfd_root.expanduser().resolve()
    keypoint_weights = args.keypoint_weights.expanduser().resolve()
    classifier_model = args.classifier_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    frame_exts = normalize_frame_exts(args.frame_exts)

    args.urfd_root = urfd_root
    args.keypoint_weights = keypoint_weights
    args.classifier_model = classifier_model
    args.output_dir = output_dir

    if not urfd_root.exists():
        raise FileNotFoundError(f"--urfd-root not found: {urfd_root}")
    if not keypoint_weights.exists():
        raise FileNotFoundError(f"--keypoint-weights not found: {keypoint_weights}")
    if not classifier_model.exists():
        raise FileNotFoundError(f"--classifier-model not found: {classifier_model}")

    sequences_all = discover_urfd_sequences(urfd_root, frame_exts)
    if not sequences_all:
        raise RuntimeError(
            f"No URFD sequences were found under {urfd_root}. "
            "Expected nested frame folders under ADLs/ and Falls/."
        )

    num_adl_all = sum(1 for seq in sequences_all if seq.video_label == "non_fall")
    num_fall_all = sum(1 for seq in sequences_all if seq.video_label == "fall")

    sequences = list(sequences_all)
    if bool(args.test):
        sequences = select_test_sequences(sequences_all, max_per_label=DEFAULT_TEST_SEQUENCES_PER_CLASS)

    num_adl = sum(1 for seq in sequences if seq.video_label == "non_fall")
    num_fall = sum(1 for seq in sequences if seq.video_label == "fall")

    print(f"Found {num_adl_all} ADL sequences and {num_fall_all} Fall sequences under {urfd_root}")
    if bool(args.test):
        print(
            "Test mode enabled: "
            f"running {num_adl} ADL and {num_fall} Fall sequences "
            f"(limit {DEFAULT_TEST_SEQUENCES_PER_CLASS} per class)."
        )

    device = pick_device(args.device)
    resolved_imgsz = float(args.imgsz) if args.imgsz is not None else float(infer_imgsz_from_path(keypoint_weights) or 640.0)

    classifier = build_classifier_adapter(args, device=device)
    pose_pipeline = build_pose_pipeline(args, device=device, keypoint_weights=keypoint_weights, imgsz=resolved_imgsz)

    LOGGER.info(
        "Using classifier arch=%s | raw window=%d stride=%d | sampled window=%d stride=%d | frame_step=%d",
        classifier.arch,
        classifier.window_policy.raw_window_len,
        classifier.window_policy.raw_window_stride,
        classifier.window_policy.sampled_window_len,
        classifier.window_policy.sampled_window_stride,
        classifier.window_policy.frame_step,
    )
    LOGGER.info("Using pose imgsz=%s on device=%s", f"{resolved_imgsz:g}", device)

    window_rows_all: List[Dict[str, Any]] = []
    video_rows: List[Dict[str, Any]] = []

    start_time = time.perf_counter()
    seq_iter: Iterable[URFDSequence] = iter_progress(sequences, desc="URFD sequences", total=len(sequences), leave=True)
    for sequence in seq_iter:
        if len(sequence.frame_paths) == 0:
            LOGGER.warning("Sequence has no frames: %s", sequence.frame_dir)
            video_rows.append(
                {
                    "dataset": sequence.dataset,
                    "video_id": sequence.video_id,
                    "video_path": str(sequence.frame_dir),
                    "video_label": sequence.video_label,
                    "num_frames": 0,
                    "num_windows": 0,
                    "num_positive_windows": 0,
                    "max_fall_score": 0.0,
                    "first_positive_window_id": None,
                    "first_detection_frame": None,
                    "first_detection_time_s": None,
                    "detected_fall": False,
                    "outcome": "FN" if sequence.video_label == "fall" else "TN",
                    "first_detection_window_id": None,
                    "num_readable_frames": 0,
                    "num_unreadable_frames": 0,
                    "pose_found_frames": 0,
                    "sampled_pose_found_frames": 0,
                    "sequence_root": str(sequence.sequence_root),
                    "frame_dir": str(sequence.frame_dir),
                }
            )
            continue

        try:
            window_rows, video_summary = evaluate_sequence(
                sequence,
                pose_pipeline=pose_pipeline,
                classifier=classifier,
                fps=float(args.fps),
                threshold=float(args.threshold),
                min_consecutive_positive=int(args.min_consecutive_positive),
            )
        except Exception as exc:
            LOGGER.warning("Failed to evaluate %s: %s", sequence.video_id, exc)
            video_rows.append(
                {
                    "dataset": sequence.dataset,
                    "video_id": sequence.video_id,
                    "video_path": str(sequence.frame_dir),
                    "video_label": sequence.video_label,
                    "num_frames": int(len(sequence.frame_paths)),
                    "num_windows": 0,
                    "num_positive_windows": 0,
                    "max_fall_score": 0.0,
                    "first_positive_window_id": None,
                    "first_detection_frame": None,
                    "first_detection_time_s": None,
                    "detected_fall": False,
                    "outcome": "FN" if sequence.video_label == "fall" else "TN",
                    "first_detection_window_id": None,
                    "num_readable_frames": 0,
                    "num_unreadable_frames": 0,
                    "pose_found_frames": 0,
                    "sampled_pose_found_frames": 0,
                    "sequence_root": str(sequence.sequence_root),
                    "frame_dir": str(sequence.frame_dir),
                    "warning": str(exc),
                }
            )
            continue

        window_rows_all.extend(window_rows)
        video_rows.append(video_summary)

    elapsed_s = time.perf_counter() - start_time

    metrics = compute_metrics(video_rows)
    metrics["elapsed_s"] = float(elapsed_s)

    output_dir.mkdir(parents=True, exist_ok=True)
    window_csv_path = output_dir / "window_predictions.csv"
    video_csv_path = output_dir / "video_summary.csv"
    metrics_json_path = output_dir / "metrics.json"
    run_config_path = output_dir / "run_config.json"

    window_fieldnames = [
        "dataset",
        "video_id",
        "video_path",
        "video_label",
        "window_id",
        "start_frame",
        "end_frame",
        "start_time_s",
        "end_time_s",
        "num_frames_in_window",
        "fall_score",
        "predicted_label",
        "threshold",
        "is_positive",
        "consecutive_positive_count",
        "event_decision",
        "predicted_class_id",
        "predicted_class_name",
        "prediction_confidence",
        "fall_score_source",
        "window_start_raw_idx",
        "window_end_raw_idx",
        "sampled_start_idx",
        "sampled_end_idx",
        "num_sampled_frames_in_window",
        "valid_pose_frames_in_window",
        "missing_pose_frames_in_window",
        "window_assembly_ms",
        "temporal_prep_ms",
        "temporal_forward_ms",
        "temporal_total_ms",
        "class_probs",
    ]
    video_fieldnames = [
        "dataset",
        "video_id",
        "video_path",
        "video_label",
        "num_frames",
        "num_windows",
        "num_positive_windows",
        "max_fall_score",
        "first_positive_window_id",
        "first_detection_frame",
        "first_detection_time_s",
        "detected_fall",
        "outcome",
        "first_detection_window_id",
        "num_readable_frames",
        "num_unreadable_frames",
        "pose_found_frames",
        "sampled_pose_found_frames",
        "sequence_root",
        "frame_dir",
        "warning",
    ]

    write_csv(window_csv_path, window_rows_all, window_fieldnames)
    write_csv(video_csv_path, video_rows, video_fieldnames)

    with metrics_json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    run_config = {
        "urfd_root": str(urfd_root),
        "keypoint_weights": str(keypoint_weights),
        "classifier_model": str(classifier_model),
        "device": str(device),
        "resolved_pose_imgsz": float(resolved_imgsz),
        "resolved_pose_half": bool(resolve_pose_half_arg(args.half, keypoint_weights, device)),
        "frame_exts": list(frame_exts),
        "fps": float(args.fps),
        "threshold": float(args.threshold),
        "min_consecutive_positive": int(args.min_consecutive_positive),
        "arch": args.arch if args.arch is not None else classifier.arch,
        "window_size_arg": args.window_size,
        "stride_arg": args.stride,
        "resolved_raw_window_len": int(classifier.window_policy.raw_window_len),
        "resolved_raw_window_stride": int(classifier.window_policy.raw_window_stride),
        "resolved_sampled_window_len": int(classifier.window_policy.sampled_window_len),
        "resolved_sampled_window_stride": int(classifier.window_policy.sampled_window_stride),
        "frame_step": int(classifier.window_policy.frame_step),
        "yolo_conf": float(args.yolo_conf),
        "yolo_iou": None if args.yolo_iou is None else float(args.yolo_iou),
        "max_people": int(args.max_people),
        "max_det": int(args.max_det),
        "track_conf_min": float(args.track_conf_min),
        "track_max_jump_px": float(args.track_max_jump_px),
        "track_max_jump_diag_frac": float(args.track_max_jump_diag_frac),
        "track_max_lost": int(args.track_max_lost),
        "track_target_x_frac": float(args.track_target_x_frac),
        "track_target_y_frac": float(args.track_target_y_frac),
        "classifier_class_names": list(classifier.class_names),
        "dataset_name": DATASET_NAME,
        "test_mode": bool(args.test),
        "test_sequences_per_class": int(DEFAULT_TEST_SEQUENCES_PER_CLASS),
        "num_sequences_found_total": int(len(sequences_all)),
        "num_adl_sequences_total": int(num_adl_all),
        "num_fall_sequences_total": int(num_fall_all),
        "num_sequences_selected": int(len(sequences)),
        "num_adl_sequences_selected": int(num_adl),
        "num_fall_sequences_selected": int(num_fall),
    }
    with run_config_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(run_config), handle, indent=2)

    print_metrics(metrics)
    print()
    print("Outputs")
    print(f"  Window CSV:  {window_csv_path}")
    print(f"  Video CSV:   {video_csv_path}")
    print(f"  Metrics:     {metrics_json_path}")
    print(f"  Run config:  {run_config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
