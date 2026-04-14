#!/usr/bin/env python3
"""
LE2I one-shot evaluation for the repo's shared final temporal pipeline.

Assumption:
This script targets the same shared YOLO + GenericTemporalAdapter path used by
`inference/inference_on_video.py` and the benchmarked final pipeline runs under
`benchmarks/img_downsize/final_pipelines/...`. It therefore reuses the existing
pose loader, tracking, feature construction, temporal window assembly, and
classifier checkpoint loading already present in the repository.

The evaluator expects the LE2I subset layout:
  <subset>/Videos/video (i).avi
  <subset>/Annotation_files/video (i).txt

The first two annotation lines are interpreted as the annotated fall start and
fall end frames. When both values are `0`, the clip is treated as `non_fall`.
Subsets without annotation files are skipped by default because they cannot be
scored fairly for either loose clip-level or strict timing-aware metrics.

MotionBERT checkpoints are detected by `--arch motionbert` /
`--arch motionbert_action`, by a `.bin` checkpoint suffix, or by `MotionBERT`
appearing in the checkpoint path. Unless overridden, MotionBERT uses the repo's
shared `configs/action/MB_ft_UPFall_xsub.yaml` config.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
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
        MotionBERTAdapter,
        MotionBERTAdapterConfig,
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
    MotionBERTAdapter = None  # type: ignore[assignment]
    MotionBERTAdapterConfig = None  # type: ignore[assignment]
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

LOGGER = logging.getLogger("evaluate_le2i")

DATASET_NAME = "LE2I"
DEFAULT_FPS = 25.0
DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_CONSECUTIVE_POSITIVE = 3
DEFAULT_OUTPUT_DIR = Path("outputs") / "le2i_eval"
DEFAULT_TEST_SEQUENCES_PER_CLASS = 5
DEFAULT_DECISION_SEARCH_MIN_VALUES = (1, 2, 3, 4, 5)
DEFAULT_DECISION_SEARCH_CSV_NAME = "decision_rule_search.csv"
DEFAULT_MOTIONBERT_CONFIG = "configs/action/MB_ft_UPFall_xsub.yaml"
DEFAULT_STRICT_EARLY_TOLERANCE_FRAMES = 0

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


@dataclass(frozen=True)
class LE2ISequence:
    dataset: str
    subset_name: str
    video_id: str
    video_label: str
    subset_root: Path
    video_path: Path
    annotation_path: Path


@dataclass(frozen=True)
class LE2IFallTimingAnnotation:
    annotation_path: Path
    event_start_frame: Optional[int]
    event_end_frame: Optional[int]
    num_lines: int
    source_kind: str
    strict_reliable: bool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-shot inference-only evaluation of the existing final keypoint + "
            "temporal classifier pipeline on the LE2I dataset."
        )
    )
    parser.add_argument(
        "--le2i-root",
        type=Path,
        required=True,
        help=(
            "Path to the LE2I root containing annotated subsets such as Home_01/, Home_02/, "
            "Coffee_room_01/, and Coffee_room_02/."
        ),
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
        help="Path to the existing trained temporal classifier checkpoint or model directory, including MotionBERT *.bin checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where window_predictions.csv, video_summary.csv, metrics.json, and run_config.json are written.",
    )
    parser.add_argument(
        "--ffmpeg-video-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional cache directory for FFmpeg-made video-only copies of LE2I AVIs. "
            "When omitted, the cache defaults to <output-dir>/video_only_cache."
        ),
    )
    parser.add_argument(
        "--disable-ffmpeg-video-only-remux",
        action="store_true",
        help=(
            "Disable the LE2I-specific FFmpeg video-only remux fallback. "
            "By default the evaluator prefers a video-only cached copy when FFmpeg is available."
        ),
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
        "--strict-early-tolerance-frames",
        type=int,
        default=DEFAULT_STRICT_EARLY_TOLERANCE_FRAMES,
        help=(
            "How many frames before the annotated fall start still count as on-time for the strict metric. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--strict-late-tolerance-frames",
        type=int,
        default=None,
        help=(
            "How many frames after the annotated fall end still count as on-time for the strict metric. "
            "If omitted, the script uses raw_window_len - 1 to account for temporal context latency."
        ),
    )
    parser.add_argument(
        "--optimize-video-decision",
        action="store_true",
        help=(
            "After inference, sweep threshold and min-consecutive-positive combinations on the saved window scores "
            "and select the combination with the highest video-level accuracy."
        ),
    )
    parser.add_argument(
        "--search-thresholds",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Optional candidate fall-score thresholds for --optimize-video-decision. "
            "Defaults to 0.00, 0.05, ..., 1.00."
        ),
    )
    parser.add_argument(
        "--search-min-consecutive-values",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional candidate min-consecutive-positive values for --optimize-video-decision. "
            "Defaults to 1 2 3 4 5."
        ),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run a small sanity-check subset using up to 5 non-fall and 5 fall sequences "
            f"(or fewer if the dataset contains less)."
        ),
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Optional classifier architecture override. Use motionbert for MotionBERT *.bin checkpoints if needed.",
    )
    parser.add_argument(
        "--motionbert-config",
        type=str,
        default=DEFAULT_MOTIONBERT_CONFIG,
        help=(
            "MotionBERT config path, resolved relative to the repo root or MotionBERT root when needed. "
            f"Default: {DEFAULT_MOTIONBERT_CONFIG}."
        ),
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


def _safe_int_from_text(value: str) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _read_nonempty_text_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return [str(line).strip() for line in handle if str(line).strip()]


def _parse_le2i_phase_rows(lines: Sequence[str]) -> List[Tuple[int, int]]:
    rows: List[Tuple[int, int]] = []
    for line in lines:
        parts = [part.strip() for part in str(line).split(",")]
        if len(parts) < 2:
            continue
        frame_idx = _safe_int_from_text(parts[0])
        phase_label = _safe_int_from_text(parts[1])
        if frame_idx is None or phase_label is None:
            continue
        rows.append((int(frame_idx), int(phase_label)))
    rows.sort(key=lambda item: item[0])
    return rows


def parse_le2i_annotation(
    annotation_path: Path,
    *,
    warn_on_malformed_timing: bool = True,
) -> Tuple[str, Optional[LE2IFallTimingAnnotation]]:
    lines = _read_nonempty_text_lines(annotation_path)
    if len(lines) >= 2:
        event_start_frame = _safe_int_from_text(lines[0])
        event_end_frame = _safe_int_from_text(lines[1])
        if event_start_frame is not None and event_end_frame is not None:
            if int(event_start_frame) > 0 and int(event_end_frame) >= int(event_start_frame):
                return (
                    "fall",
                    LE2IFallTimingAnnotation(
                        annotation_path=annotation_path,
                        event_start_frame=int(event_start_frame),
                        event_end_frame=int(event_end_frame),
                        num_lines=int(len(lines)),
                        source_kind="header",
                        strict_reliable=True,
                    ),
                )
            return "non_fall", None

    phase_rows = _parse_le2i_phase_rows(lines)
    non_normal_rows = [(frame_idx, phase_label) for frame_idx, phase_label in phase_rows if int(phase_label) != 1]
    if non_normal_rows:
        if warn_on_malformed_timing:
            LOGGER.warning(
                "Missing or malformed LE2I timing header; using loose fall-only labeling and skipping strict timing for %s",
                annotation_path,
            )
        return (
            "fall",
            LE2IFallTimingAnnotation(
                annotation_path=annotation_path,
                event_start_frame=None,
                event_end_frame=None,
                num_lines=int(len(lines)),
                source_kind="phase_only_no_header",
                strict_reliable=False,
            ),
        )

    if phase_rows:
        return "non_fall", None

    if warn_on_malformed_timing:
        LOGGER.warning("Could not parse LE2I annotation file: %s", annotation_path)
    return "non_fall", None


def resolve_le2i_annotation_dir(subset_root: Path) -> Optional[Path]:
    for dirname in ("Annotation_files", "Annotations_files"):
        candidate = subset_root / dirname
        if candidate.is_dir():
            return candidate
    return None


def find_ffmpeg_executable() -> Optional[str]:
    return shutil.which("ffmpeg")


def _iter_video_frames(video_path: Path):
    """Yield (total_frames_hint, frame_generator) for a video file.

    Tries PyAV first (avoids OpenCV's native AVI codec which can corrupt the
    heap on some LE2I files).  Falls back to cv2.CAP_FFMPEG and then the
    default OpenCV backend.  The generator yields BGR uint8 numpy arrays.
    """
    # --- PyAV path -----------------------------------------------------------
    try:
        import av as _av  # type: ignore[import]
        container = _av.open(str(video_path))
        stream = container.streams.video[0]
        hint = int(stream.frames) if stream.frames else 0

        def _av_gen():
            try:
                for f in container.decode(stream):
                    yield f.to_ndarray(format="bgr24")
            finally:
                container.close()

        return hint, _av_gen()
    except ImportError:
        pass
    except Exception as exc:
        LOGGER.debug("PyAV failed for %s (%s); falling back to cv2", video_path, exc)

    # --- OpenCV path ---------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def _cv2_gen():
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    return hint, _cv2_gen()


def remux_video_to_video_only(
    video_path: Path,
    *,
    cache_dir: Path,
    ffmpeg_exe: str,
) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._") or "video"
    suffix = video_path.suffix if video_path.suffix else ".avi"
    path_hash = hashlib.sha1(str(video_path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:16]
    cache_name = f"{safe_stem}__{path_hash}{suffix}"
    remuxed_path = cache_dir / cache_name
    if remuxed_path.is_file():
        try:
            if remuxed_path.stat().st_mtime >= video_path.stat().st_mtime and remuxed_path.stat().st_size > 0:
                return remuxed_path
        except OSError:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg_exe),
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        str(remuxed_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or (not remuxed_path.is_file()) or remuxed_path.stat().st_size <= 0:
        stderr_text = str(result.stderr or "").strip()
        if remuxed_path.exists():
            try:
                remuxed_path.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"FFmpeg video-only remux failed for {video_path}. "
            f"stderr: {stderr_text or 'n/a'}"
        )
    return remuxed_path


def discover_le2i_sequences(le2i_root: Path) -> List[LE2ISequence]:
    sequences: List[LE2ISequence] = []

    subset_roots = [path for path in le2i_root.iterdir() if path.is_dir()]
    subset_roots.sort(key=lambda path: natural_sort_key(path.name))

    for subset_root in subset_roots:
        annotation_dir = resolve_le2i_annotation_dir(subset_root)
        video_dir = subset_root / "Videos"
        if annotation_dir is None:
            LOGGER.info("Skipping unannotated LE2I subset: %s", subset_root)
            continue
        if not video_dir.is_dir():
            LOGGER.warning("Annotated subset is missing Videos/: %s", subset_root)
            continue

        annotation_paths = [path for path in annotation_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"]
        annotation_paths.sort(key=lambda path: natural_sort_key(path.name))

        for annotation_path in annotation_paths:
            video_path = video_dir / f"{annotation_path.stem}.avi"
            if not video_path.is_file():
                LOGGER.warning("Missing LE2I video for annotation %s: expected %s", annotation_path, video_path)
                continue

            video_label, _ = parse_le2i_annotation(annotation_path, warn_on_malformed_timing=True)
            sequences.append(
                LE2ISequence(
                    dataset=DATASET_NAME,
                    subset_name=subset_root.name,
                    video_id=f"{subset_root.name}/{annotation_path.stem}",
                    video_label=video_label,
                    subset_root=subset_root,
                    video_path=video_path,
                    annotation_path=annotation_path,
                )
            )

    sequences.sort(key=lambda seq: (0 if seq.video_label == "non_fall" else 1, natural_sort_key(seq.video_id)))
    return sequences


def select_test_sequences(
    sequences: Sequence[LE2ISequence],
    *,
    max_per_label: int = DEFAULT_TEST_SEQUENCES_PER_CLASS,
) -> List[LE2ISequence]:
    limit = max(1, int(max_per_label))
    selected: List[LE2ISequence] = []
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


def load_le2i_fall_timing_annotations(sequences: Sequence[LE2ISequence]) -> Dict[str, LE2IFallTimingAnnotation]:
    annotations: Dict[str, LE2IFallTimingAnnotation] = {}
    for sequence in sequences:
        video_label, annotation = parse_le2i_annotation(sequence.annotation_path, warn_on_malformed_timing=False)
        if video_label == "fall" and annotation is not None and bool(annotation.strict_reliable):
            annotations[sequence.video_id] = annotation
    return annotations


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


def default_decision_search_thresholds() -> List[float]:
    return [round(0.05 * idx, 4) for idx in range(21)]


def normalize_search_thresholds(values: Optional[Sequence[float]], *, configured_threshold: float) -> List[float]:
    raw_values = list(values) if values else default_decision_search_thresholds()
    raw_values.append(float(configured_threshold))

    deduped: Dict[float, float] = {}
    for value in raw_values:
        if (not math.isfinite(float(value))) or float(value) < 0.0 or float(value) > 1.0:
            raise ValueError("All decision-search thresholds must be finite values in [0, 1].")
        key = round(float(value), 6)
        deduped[key] = float(value)

    normalized = sorted(float(v) for v in deduped.values())
    if not normalized:
        raise ValueError("Decision-search threshold grid is empty.")
    return normalized


def normalize_search_min_consecutive_values(
    values: Optional[Sequence[int]],
    *,
    configured_min_consecutive_positive: int,
) -> List[int]:
    raw_values = list(values) if values else list(DEFAULT_DECISION_SEARCH_MIN_VALUES)
    raw_values.append(int(configured_min_consecutive_positive))

    normalized = sorted({int(value) for value in raw_values})
    if not normalized:
        raise ValueError("Decision-search min-consecutive-positive grid is empty.")
    if any(int(value) <= 0 for value in normalized):
        raise ValueError("All decision-search min-consecutive-positive values must be >= 1.")
    return normalized


def resolve_strict_late_tolerance_frames(explicit_value: Optional[int], raw_window_len: int) -> int:
    if explicit_value is not None:
        return max(0, int(explicit_value))
    return max(0, int(raw_window_len) - 1)


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


def count_field_values(
    rows: Sequence[Dict[str, Any]],
    key: str,
    *,
    eligible_key: Optional[str] = None,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        if eligible_key is not None and not bool(row.get(eligible_key, False)):
            continue
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    return counts


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


def normalize_classifier_arch_name(arch: Optional[str]) -> str:
    return str(arch or "").strip().lower()


def is_motionbert_classifier(model_path: Path, arch: Optional[str]) -> bool:
    arch_name = normalize_classifier_arch_name(arch)
    if arch_name in {"motionbert", "motionbert_action"}:
        return True

    if model_path.suffix.lower() == ".bin":
        return True

    tokens = [model_path.name.lower(), model_path.stem.lower()] + [part.lower() for part in model_path.parts]
    return any("motionbert" in token for token in tokens)


def get_classifier_name(adapter: TemporalClassifierAdapter) -> str:
    name = getattr(adapter, "arch", None)
    if name is not None and str(name).strip():
        return str(name).strip()
    fallback = getattr(adapter, "name", None)
    if fallback is not None and str(fallback).strip():
        return str(fallback).strip()
    return "unknown"


def build_classifier_adapter(args: argparse.Namespace, *, device: str) -> TemporalClassifierAdapter:
    if is_motionbert_classifier(args.classifier_model, args.arch):
        return MotionBERTAdapter(
            MotionBERTAdapterConfig(
                model_arg=str(args.classifier_model),
                config_arg=str(args.motionbert_config),
                device=device,
                frame_step=max(1, int(args.frame_step)),
                win_len_raw=int(args.window_size) if args.window_size is not None else None,
                win_step_raw=int(args.stride) if args.stride is not None else 16,
                labels_file=None,
                no_merge_fall=False,
                missing_conf_thres=0.0,
                repo_root=_REPO_ROOT,
            )
        )

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
    sequence: LE2ISequence,
    timing_annotation: Optional[LE2IFallTimingAnnotation],
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
    start_frame_1based = to_human_frame(int(window_data.raw_start_idx))
    end_frame_1based = to_human_frame(int(window_data.raw_end_idx))

    has_timing_annotation = bool(
        sequence.video_label == "fall"
        and timing_annotation is not None
        and timing_annotation.event_start_frame is not None
        and timing_annotation.event_end_frame is not None
    )
    annotated_event_start_frame = None
    annotated_event_end_frame = None
    overlaps_annotated_fall_event = False
    annotated_positive_frames_in_window = 0
    annotated_phase = "non_event"

    if has_timing_annotation and timing_annotation is not None:
        annotated_event_start_frame = int(timing_annotation.event_start_frame)
        annotated_event_end_frame = int(timing_annotation.event_end_frame)
        overlap_start = max(int(start_frame_1based), int(timing_annotation.event_start_frame))
        overlap_end = min(int(end_frame_1based), int(timing_annotation.event_end_frame))
        if overlap_start <= overlap_end:
            overlaps_annotated_fall_event = True
            annotated_positive_frames_in_window = int(overlap_end - overlap_start + 1)
        if overlaps_annotated_fall_event:
            annotated_phase = "event"

    row = {
        "dataset": sequence.dataset,
        "video_id": sequence.video_id,
        "video_path": str(sequence.video_path),
        "video_label": sequence.video_label,
        "subset_name": sequence.subset_name,
        "annotation_path": str(sequence.annotation_path),
        "window_id": int(window_id),
        "start_frame": int(start_frame_1based),
        "end_frame": int(end_frame_1based),
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
        "timing_annotation_available": bool(has_timing_annotation),
        "annotated_event_start_frame": annotated_event_start_frame,
        "annotated_event_end_frame": annotated_event_end_frame,
        "annotated_event_start_time_s": (
            frame_time_seconds(int(annotated_event_start_frame) - 1, fps) if annotated_event_start_frame is not None else None
        ),
        "annotated_event_end_time_s": (
            frame_time_seconds(int(annotated_event_end_frame) - 1, fps) if annotated_event_end_frame is not None else None
        ),
        "overlaps_annotated_fall_event": bool(overlaps_annotated_fall_event),
        "annotated_positive_frames_in_window": int(annotated_positive_frames_in_window),
        "annotated_phase": str(annotated_phase),
        "overlaps_transition_phase": None,
        "overlaps_post_fall_phase": None,
    }
    if probs_np is not None:
        row["class_probs"] = json.dumps([float(x) for x in probs_np.tolist()])
    rows.append(row)


def compute_video_decision_fields(
    window_rows: Sequence[Dict[str, Any]],
    *,
    video_label: str,
    annotated_event_start_frame: Optional[int],
    annotated_event_start_time_s: Optional[float],
    threshold: float,
    min_consecutive_positive: int,
    apply_to_rows: bool,
) -> Dict[str, Any]:
    ordered_rows = sorted(window_rows, key=lambda row: int(row.get("window_id", 0)))

    consecutive_positive = 0
    detected_fall = False
    detection_run_start_row: Optional[Dict[str, Any]] = None
    confirmed_detection_row: Optional[Dict[str, Any]] = None
    max_fall_score = 0.0
    num_positive_windows = 0
    first_positive_window_id: Optional[int] = None
    timing_hit_row: Optional[Dict[str, Any]] = None

    for idx, row in enumerate(ordered_rows):
        fall_score = float(row.get("fall_score", 0.0))
        is_positive = bool(fall_score >= float(threshold))
        confirmed_detection_alert = False

        if is_positive:
            num_positive_windows += 1
            consecutive_positive += 1
            if first_positive_window_id is None:
                first_positive_window_id = int(row["window_id"])
            if timing_hit_row is None and bool(row.get("overlaps_annotated_fall_event", False)):
                timing_hit_row = row
        else:
            consecutive_positive = 0

        max_fall_score = max(max_fall_score, fall_score)

        if (not detected_fall) and consecutive_positive >= int(min_consecutive_positive):
            detected_fall = True
            start_idx = max(0, idx - int(min_consecutive_positive) + 1)
            detection_run_start_row = ordered_rows[start_idx]
            confirmed_detection_row = row
            confirmed_detection_alert = True

        if apply_to_rows:
            row["threshold"] = float(threshold)
            row["is_positive"] = bool(is_positive)
            row["consecutive_positive_count"] = int(consecutive_positive)
            row["event_decision"] = bool(detected_fall)
            row["confirmed_detection_alert"] = bool(confirmed_detection_alert)

    first_detection_frame: Optional[int] = None
    first_detection_time_s: Optional[float] = None
    first_detection_window_id: Optional[int] = None
    if detection_run_start_row is not None:
        first_detection_window_id = int(detection_run_start_row["window_id"])
        first_detection_frame = int(detection_run_start_row["start_frame"])
        first_detection_time_s = float(detection_run_start_row["start_time_s"])

    first_confirmed_detection_window_id: Optional[int] = None
    first_confirmed_detection_frame: Optional[int] = None
    first_confirmed_detection_time_s: Optional[float] = None
    first_confirmed_detection_delay_frames: Optional[int] = None
    first_confirmed_detection_delay_s: Optional[float] = None
    if confirmed_detection_row is not None:
        first_confirmed_detection_window_id = int(confirmed_detection_row["window_id"])
        first_confirmed_detection_frame = int(confirmed_detection_row["end_frame"])
        first_confirmed_detection_time_s = float(confirmed_detection_row["end_time_s"])
        if annotated_event_start_frame is not None and annotated_event_start_time_s is not None:
            first_confirmed_detection_delay_frames = int(
                first_confirmed_detection_frame - int(annotated_event_start_frame)
            )
            first_confirmed_detection_delay_s = float(
                first_confirmed_detection_time_s - float(annotated_event_start_time_s)
            )

    timing_hit = timing_hit_row is not None
    first_timing_hit_window_id: Optional[int] = None
    first_timing_hit_frame: Optional[int] = None
    first_timing_hit_time_s: Optional[float] = None
    first_timing_hit_delay_frames: Optional[int] = None
    first_timing_hit_delay_s: Optional[float] = None

    if timing_hit_row is not None:
        first_timing_hit_window_id = int(timing_hit_row["window_id"])
        first_timing_hit_frame = int(timing_hit_row["start_frame"])
        first_timing_hit_time_s = float(timing_hit_row["start_time_s"])
        if annotated_event_start_frame is not None and annotated_event_start_time_s is not None:
            first_timing_hit_delay_frames = int(first_timing_hit_frame - int(annotated_event_start_frame))
            first_timing_hit_delay_s = float(first_timing_hit_time_s - float(annotated_event_start_time_s))

    is_true_fall = str(video_label) == "fall"
    if is_true_fall and detected_fall:
        outcome = "TP"
    elif (not is_true_fall) and (not detected_fall):
        outcome = "TN"
    elif (not is_true_fall) and detected_fall:
        outcome = "FP"
    else:
        outcome = "FN"

    return {
        "num_positive_windows": int(num_positive_windows),
        "max_fall_score": float(max_fall_score),
        "first_positive_window_id": first_positive_window_id,
        "first_detection_frame": first_detection_frame,
        "first_detection_time_s": first_detection_time_s,
        "detected_fall": bool(detected_fall),
        "outcome": outcome,
        "first_detection_window_id": first_detection_window_id,
        "first_confirmed_detection_window_id": first_confirmed_detection_window_id,
        "first_confirmed_detection_frame": first_confirmed_detection_frame,
        "first_confirmed_detection_time_s": first_confirmed_detection_time_s,
        "first_confirmed_detection_delay_frames": first_confirmed_detection_delay_frames,
        "first_confirmed_detection_delay_s": first_confirmed_detection_delay_s,
        "timing_hit": bool(timing_hit),
        "first_timing_hit_window_id": first_timing_hit_window_id,
        "first_timing_hit_frame": first_timing_hit_frame,
        "first_timing_hit_time_s": first_timing_hit_time_s,
        "first_timing_hit_delay_frames": first_timing_hit_delay_frames,
        "first_timing_hit_delay_s": first_timing_hit_delay_s,
    }


def compute_strict_event_fields(
    video_summary: Dict[str, Any],
    *,
    strict_early_tolerance_frames: int,
    strict_late_tolerance_frames: int,
) -> Dict[str, Any]:
    video_label = str(video_summary.get("video_label", ""))
    fps = float(video_summary.get("fps", DEFAULT_FPS) or DEFAULT_FPS)
    detected_fall = bool(video_summary.get("detected_fall", False))
    timing_available = bool(video_summary.get("timing_annotation_available", False))
    annotated_event_start_frame = video_summary.get("annotated_event_start_frame")
    annotated_event_end_frame = video_summary.get("annotated_event_end_frame")
    annotated_event_start_time_s = video_summary.get("annotated_event_start_time_s")
    strict_alert_window_id = video_summary.get("first_confirmed_detection_window_id")
    strict_alert_frame = video_summary.get("first_confirmed_detection_frame")
    strict_alert_time_s = video_summary.get("first_confirmed_detection_time_s")

    strict_eligible = bool(
        video_label == "non_fall"
        or (
            video_label == "fall"
            and timing_available
            and annotated_event_start_frame is not None
            and annotated_event_end_frame is not None
        )
    )

    strict_eval_start_frame: Optional[int] = None
    strict_eval_end_frame: Optional[int] = None
    strict_eval_start_time_s: Optional[float] = None
    strict_eval_end_time_s: Optional[float] = None
    strict_detection_within_tolerance: Optional[bool] = None
    strict_outcome: Optional[str] = None
    strict_outcome_reason: Optional[str] = None
    strict_alert_delay_frames: Optional[int] = None
    strict_alert_delay_s: Optional[float] = None

    if strict_eligible and video_label == "fall":
        strict_eval_start_frame = max(1, int(annotated_event_start_frame) - int(strict_early_tolerance_frames))
        strict_eval_end_frame = int(annotated_event_end_frame) + int(strict_late_tolerance_frames)
        strict_eval_start_time_s = frame_time_seconds(int(strict_eval_start_frame) - 1, fps)
        strict_eval_end_time_s = frame_time_seconds(int(strict_eval_end_frame) - 1, fps)

        if strict_alert_frame is not None and annotated_event_start_frame is not None and annotated_event_start_time_s is not None:
            strict_alert_delay_frames = int(strict_alert_frame) - int(annotated_event_start_frame)
            if strict_alert_time_s is not None:
                strict_alert_delay_s = float(strict_alert_time_s) - float(annotated_event_start_time_s)

        if strict_alert_frame is None:
            strict_detection_within_tolerance = False
            strict_outcome = "FN"
            strict_outcome_reason = "missed_fall"
        elif int(strict_alert_frame) < int(strict_eval_start_frame):
            strict_detection_within_tolerance = False
            strict_outcome = "FN"
            strict_outcome_reason = "early_alarm"
        elif int(strict_alert_frame) > int(strict_eval_end_frame):
            strict_detection_within_tolerance = False
            strict_outcome = "FN"
            strict_outcome_reason = "late_detection"
        else:
            strict_detection_within_tolerance = True
            strict_outcome = "TP"
            strict_outcome_reason = "on_time_detection"
    elif strict_eligible:
        strict_detection_within_tolerance = not bool(detected_fall)
        if detected_fall:
            strict_outcome = "FP"
            strict_outcome_reason = "false_alarm_non_fall"
        else:
            strict_outcome = "TN"
            strict_outcome_reason = "correct_non_fall"
    else:
        strict_outcome_reason = "missing_timing_annotation"

    return {
        "strict_eligible": bool(strict_eligible),
        "strict_early_tolerance_frames": int(strict_early_tolerance_frames),
        "strict_late_tolerance_frames": int(strict_late_tolerance_frames),
        "strict_eval_start_frame": strict_eval_start_frame,
        "strict_eval_end_frame": strict_eval_end_frame,
        "strict_eval_start_time_s": strict_eval_start_time_s,
        "strict_eval_end_time_s": strict_eval_end_time_s,
        "strict_alert_window_id": strict_alert_window_id,
        "strict_alert_frame": strict_alert_frame,
        "strict_alert_time_s": strict_alert_time_s,
        "strict_alert_delay_frames": strict_alert_delay_frames,
        "strict_alert_delay_s": strict_alert_delay_s,
        "strict_detection_within_tolerance": strict_detection_within_tolerance,
        "strict_outcome": strict_outcome,
        "strict_outcome_reason": strict_outcome_reason,
    }


def build_video_summary_for_decision(
    base_video_summary: Dict[str, Any],
    *,
    window_rows: Sequence[Dict[str, Any]],
    threshold: float,
    min_consecutive_positive: int,
    strict_early_tolerance_frames: int,
    strict_late_tolerance_frames: int,
    apply_to_rows: bool,
) -> Dict[str, Any]:
    summary = dict(base_video_summary)
    summary.update(
        compute_video_decision_fields(
            window_rows,
            video_label=str(summary.get("video_label", "")),
            annotated_event_start_frame=summary.get("annotated_event_start_frame"),
            annotated_event_start_time_s=summary.get("annotated_event_start_time_s"),
            threshold=float(threshold),
            min_consecutive_positive=int(min_consecutive_positive),
            apply_to_rows=apply_to_rows,
        )
    )
    summary.update(
        compute_strict_event_fields(
            summary,
            strict_early_tolerance_frames=int(strict_early_tolerance_frames),
            strict_late_tolerance_frames=int(strict_late_tolerance_frames),
        )
    )
    return summary


def search_best_video_decision(
    sequence_results: Sequence[Dict[str, Any]],
    *,
    thresholds: Sequence[float],
    min_consecutive_values: Sequence[int],
    configured_threshold: float,
    configured_min_consecutive_positive: int,
    strict_early_tolerance_frames: int,
    strict_late_tolerance_frames: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    search_rows: List[Dict[str, Any]] = []

    for threshold in thresholds:
        for min_consecutive_positive in min_consecutive_values:
            video_rows = [
                build_video_summary_for_decision(
                    dict(result["base_video_summary"]),
                    window_rows=result["window_rows"],
                    threshold=float(threshold),
                    min_consecutive_positive=int(min_consecutive_positive),
                    strict_early_tolerance_frames=int(strict_early_tolerance_frames),
                    strict_late_tolerance_frames=int(strict_late_tolerance_frames),
                    apply_to_rows=False,
                )
                for result in sequence_results
            ]
            metrics = compute_metrics(video_rows)
            search_rows.append(
                {
                    "threshold": float(threshold),
                    "min_consecutive_positive": int(min_consecutive_positive),
                    "accuracy": float(metrics["accuracy"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "f1": float(metrics["f1"]),
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "specificity": float(metrics["specificity"]),
                    "tp": int(metrics["tp"]),
                    "tn": int(metrics["tn"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                    "num_videos": int(metrics["num_videos"]),
                }
            )

    if not search_rows:
        raise RuntimeError("Decision-rule search produced no candidate combinations.")

    best_row = max(
        search_rows,
        key=lambda row: (
            float(row["accuracy"]),
            float(row["balanced_accuracy"]),
            float(row["f1"]),
            -abs(float(row["threshold"]) - float(configured_threshold)),
            -abs(int(row["min_consecutive_positive"]) - int(configured_min_consecutive_positive)),
            -float(row["threshold"]),
            -int(row["min_consecutive_positive"]),
        ),
    )
    return dict(best_row), search_rows


def evaluate_sequence(
    sequence: LE2ISequence,
    *,
    pose_pipeline: PosePipeline,
    classifier: TemporalClassifierAdapter,
    timing_annotation: Optional[LE2IFallTimingAnnotation],
    fps: float,
    threshold: float,
    min_consecutive_positive: int,
    strict_early_tolerance_frames: int,
    strict_late_tolerance_frames: int,
    ffmpeg_exe: Optional[str],
    video_only_cache_dir: Optional[Path],
    remux_video_only: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pose_pipeline.reset_tracking_state()

    window_rows: List[Dict[str, Any]] = []
    sampled_xy: List[np.ndarray] = []
    sampled_cf: List[np.ndarray] = []
    sampled_raw_idx: List[int] = []

    readable_frames = 0
    unreadable_frames = 0
    pose_found_frames = 0
    sampled_pose_found_frames = 0
    next_window_id = 0
    next_window_start = 0
    image_shape_hw: Optional[Tuple[int, int]] = None
    policy = classifier.window_policy
    conf_thres = float(getattr(classifier, "conf_thres", 0.0))
    raw_frame_idx = 0

    # Determine target frame size from checkpoint training dimensions (paper_rp uses pixel coords).
    target_hw: Optional[Tuple[int, int]] = None
    _ckpt_w = getattr(classifier, "rp_img_w", None)
    _ckpt_h = getattr(classifier, "rp_img_h", None)
    if _ckpt_w is not None and _ckpt_h is not None:
        target_hw = (int(_ckpt_h), int(_ckpt_w))

    input_video_path = sequence.video_path
    effective_video_path = sequence.video_path
    if bool(remux_video_only) and ffmpeg_exe is not None and video_only_cache_dir is not None:
        effective_video_path = remux_video_to_video_only(
            sequence.video_path,
            cache_dir=video_only_cache_dir,
            ffmpeg_exe=ffmpeg_exe,
        )

    total_frames_hint, frame_iter = _iter_video_frames(effective_video_path)

    progress = None
    if tqdm is not None:
        progress = tqdm(
            total=total_frames_hint if total_frames_hint > 0 else None,
            desc=f"{sequence.video_id}",
            leave=False,
            dynamic_ncols=True,
        )

    try:
        for frame in frame_iter:
            readable_frames += 1
            if target_hw is not None and (int(frame.shape[0]), int(frame.shape[1])) != target_hw:
                frame = cv2.resize(frame, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
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
                    timing_annotation=timing_annotation,
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

            raw_frame_idx += 1
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()

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
                timing_annotation=timing_annotation,
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

    annotated_event_start_frame: Optional[int] = None
    annotated_event_end_frame: Optional[int] = None
    annotated_event_start_time_s: Optional[float] = None
    annotated_event_end_time_s: Optional[float] = None

    if (
        timing_annotation is not None
        and sequence.video_label == "fall"
        and timing_annotation.event_start_frame is not None
        and timing_annotation.event_end_frame is not None
    ):
        annotated_event_start_frame = int(timing_annotation.event_start_frame)
        annotated_event_end_frame = int(timing_annotation.event_end_frame)
        annotated_event_start_time_s = frame_time_seconds(int(annotated_event_start_frame) - 1, fps)
        annotated_event_end_time_s = frame_time_seconds(int(annotated_event_end_frame) - 1, fps)

    base_video_summary = {
        "dataset": sequence.dataset,
        "video_id": sequence.video_id,
        "video_path": str(input_video_path),
        "video_label": sequence.video_label,
        "subset_name": sequence.subset_name,
        "annotation_path": str(sequence.annotation_path),
        "opened_video_path": str(effective_video_path),
        "fps": float(fps),
        "num_frames": int(raw_frame_idx),
        "num_windows": int(len(window_rows)),
        "timing_annotation_available": bool(
            sequence.video_label == "fall"
            and timing_annotation is not None
            and timing_annotation.event_start_frame is not None
            and timing_annotation.event_end_frame is not None
        ),
        "annotated_event_start_frame": annotated_event_start_frame,
        "annotated_event_end_frame": annotated_event_end_frame,
        "annotated_event_start_time_s": annotated_event_start_time_s,
        "annotated_event_end_time_s": annotated_event_end_time_s,
        "num_readable_frames": int(readable_frames),
        "num_unreadable_frames": int(unreadable_frames),
        "pose_found_frames": int(pose_found_frames),
        "sampled_pose_found_frames": int(sampled_pose_found_frames),
        "sequence_root": str(sequence.subset_root),
        "video_file": str(input_video_path),
    }
    video_summary = build_video_summary_for_decision(
        base_video_summary,
        window_rows=window_rows,
        threshold=float(threshold),
        min_consecutive_positive=int(min_consecutive_positive),
        strict_early_tolerance_frames=int(strict_early_tolerance_frames),
        strict_late_tolerance_frames=int(strict_late_tolerance_frames),
        apply_to_rows=True,
    )
    return window_rows, video_summary


def compute_metrics(
    video_rows: Sequence[Dict[str, Any]],
    *,
    outcome_key: str = "outcome",
    eligible_key: Optional[str] = None,
) -> Dict[str, Any]:
    eligible_rows = [
        row for row in video_rows
        if eligible_key is None or bool(row.get(eligible_key, False))
    ]

    tp = sum(1 for row in eligible_rows if row.get(outcome_key) == "TP")
    tn = sum(1 for row in eligible_rows if row.get(outcome_key) == "TN")
    fp = sum(1 for row in eligible_rows if row.get(outcome_key) == "FP")
    fn = sum(1 for row in eligible_rows if row.get(outcome_key) == "FN")

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
        "num_videos": int(len(eligible_rows)),
        "num_fall_videos": int(sum(1 for row in eligible_rows if row.get("video_label") == "fall")),
        "num_non_fall_videos": int(sum(1 for row in eligible_rows if row.get("video_label") == "non_fall")),
        "num_skipped_videos": int(len(video_rows) - len(eligible_rows)),
    }


def compute_timing_window_metrics(window_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    eligible_rows: List[Dict[str, Any]] = []
    for row in window_rows:
        video_label = str(row.get("video_label", ""))
        if video_label == "non_fall":
            eligible_rows.append(row)
            continue
        if bool(row.get("timing_annotation_available", False)):
            eligible_rows.append(row)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    positive_gt = 0
    negative_gt = 0

    for row in eligible_rows:
        pred_positive = bool(row.get("is_positive", False))
        gt_positive = bool(row.get("overlaps_annotated_fall_event", False))
        if gt_positive:
            positive_gt += 1
        else:
            negative_gt += 1

        if pred_positive and gt_positive:
            tp += 1
        elif (not pred_positive) and (not gt_positive):
            tn += 1
        elif pred_positive and (not gt_positive):
            fp += 1
        else:
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)

    metrics = {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(safe_div(tp + tn, tp + tn + fp + fn)),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(safe_div(2.0 * precision * recall, precision + recall)),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "num_windows_scored": int(len(eligible_rows)),
        "num_positive_gt_windows": int(positive_gt),
        "num_negative_gt_windows": int(negative_gt),
        "num_fall_windows_with_timing_annotations": int(
            sum(
                1
                for row in eligible_rows
                if str(row.get("video_label", "")) == "fall" and bool(row.get("timing_annotation_available", False))
            )
        ),
        "num_non_fall_windows": int(
            sum(1 for row in eligible_rows if str(row.get("video_label", "")) == "non_fall")
        ),
    }
    return metrics


def print_decision_search_summary(
    *,
    configured_threshold: float,
    configured_min_consecutive_positive: int,
    configured_metrics: Dict[str, Any],
    best_search_row: Dict[str, Any],
    num_combinations: int,
) -> None:
    print()
    print("Decision-rule search")
    print(f"  Evaluated {num_combinations} threshold/min-consecutive combinations.")
    print(
        "  Configured: "
        f"threshold={float(configured_threshold):.4f} "
        f"min_consecutive_positive={int(configured_min_consecutive_positive)} "
        f"accuracy={float(configured_metrics['accuracy']):.4f}"
    )
    print(
        "  Selected:   "
        f"threshold={float(best_search_row['threshold']):.4f} "
        f"min_consecutive_positive={int(best_search_row['min_consecutive_positive'])} "
        f"accuracy={float(best_search_row['accuracy']):.4f}"
    )
    print(
        "  Tie-breaks: balanced_accuracy, f1, closeness to the configured values, "
        "then smaller threshold/min-consecutive."
    )


def print_metric_block(title: str, metrics: Dict[str, Any]) -> None:
    print()
    print(title)
    print(f"  TP: {metrics['tp']}  TN: {metrics['tn']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f}")
    print(f"  F1:                 {metrics['f1']:.4f}")
    print(f"  Balanced accuracy:  {metrics['balanced_accuracy']:.4f}")


def print_metrics(metrics: Dict[str, Any], *, title: Optional[str] = None) -> None:
    print_metric_block(title or f"{DATASET_NAME} video-level metrics", metrics)


def main() -> int:
    args = build_arg_parser().parse_args()
    configure_logging(args.log_level)

    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing runtime dependency needed for LE2I evaluation. "
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
    if args.search_thresholds is not None:
        for value in args.search_thresholds:
            if (not math.isfinite(float(value))) or float(value) < 0.0 or float(value) > 1.0:
                raise ValueError("--search-thresholds values must be finite values in [0, 1].")
    if args.search_min_consecutive_values is not None:
        for value in args.search_min_consecutive_values:
            if int(value) <= 0:
                raise ValueError("--search-min-consecutive-values values must be >= 1.")
    if int(args.strict_early_tolerance_frames) < 0:
        raise ValueError("--strict-early-tolerance-frames must be >= 0.")
    if args.strict_late_tolerance_frames is not None and int(args.strict_late_tolerance_frames) < 0:
        raise ValueError("--strict-late-tolerance-frames must be >= 0 when provided.")
    if int(args.frame_step) <= 0:
        raise ValueError("--frame-step must be >= 1.")
    if int(args.max_people) <= 0:
        raise ValueError("--max-people must be >= 1.")
    if int(args.max_det) < 0:
        raise ValueError("--max-det must be >= 0.")

    le2i_root = args.le2i_root.expanduser().resolve()
    keypoint_weights = args.keypoint_weights.expanduser().resolve()
    classifier_model = args.classifier_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    args.le2i_root = le2i_root
    args.keypoint_weights = keypoint_weights
    args.classifier_model = classifier_model
    args.output_dir = output_dir

    if not le2i_root.exists():
        raise FileNotFoundError(f"--le2i-root not found: {le2i_root}")
    if not keypoint_weights.exists():
        raise FileNotFoundError(f"--keypoint-weights not found: {keypoint_weights}")
    if not classifier_model.exists():
        raise FileNotFoundError(f"--classifier-model not found: {classifier_model}")

    sequences_all = discover_le2i_sequences(le2i_root)
    if not sequences_all:
        raise RuntimeError(
            f"No annotated LE2I sequences were found under {le2i_root}. "
            "Expected subsets containing both Videos/ and Annotation_files/."
        )

    num_adl_all = sum(1 for seq in sequences_all if seq.video_label == "non_fall")
    num_fall_all = sum(1 for seq in sequences_all if seq.video_label == "fall")

    sequences = list(sequences_all)
    if bool(args.test):
        sequences = select_test_sequences(sequences_all, max_per_label=DEFAULT_TEST_SEQUENCES_PER_CLASS)

    num_adl = sum(1 for seq in sequences if seq.video_label == "non_fall")
    num_fall = sum(1 for seq in sequences if seq.video_label == "fall")

    print(f"Found {num_adl_all} non-fall videos and {num_fall_all} fall videos under {le2i_root}")
    if bool(args.test):
        print(
            "Test mode enabled: "
            f"running {num_adl} non-fall and {num_fall} fall sequences "
            f"(limit {DEFAULT_TEST_SEQUENCES_PER_CLASS} per class)."
        )

    timing_annotations = load_le2i_fall_timing_annotations(sequences_all)
    timing_annotation_source = "Annotation_files/video (i).txt headers"
    matched_timing_annotations_all = sum(
        1
        for seq in sequences_all
        if seq.video_label == "fall" and seq.video_id in timing_annotations
    )
    matched_timing_annotations_selected = sum(
        1
        for seq in sequences
        if seq.video_label == "fall" and seq.video_id in timing_annotations
    )
    print(
        "Loaded fall timing annotations for "
        f"{len(timing_annotations)} fall videos from LE2I annotation headers."
    )
    print(
        "Matched timing annotations for "
        f"{matched_timing_annotations_selected}/{num_fall} selected fall videos "
        f"({matched_timing_annotations_all}/{num_fall_all} across the full discovery set)."
    )
    malformed_header_falls_all = int(num_fall_all - matched_timing_annotations_all)
    if malformed_header_falls_all > 0:
        print(
            "Loose-only fall videos with malformed or headerless timing annotations: "
            f"{malformed_header_falls_all}. These remain in clip-level metrics but are excluded from strict timing metrics."
        )

    device = pick_device(args.device)
    resolved_imgsz = float(args.imgsz) if args.imgsz is not None else float(infer_imgsz_from_path(keypoint_weights) or 640.0)

    classifier = build_classifier_adapter(args, device=device)
    classifier_name = get_classifier_name(classifier)
    strict_early_tolerance_frames = int(args.strict_early_tolerance_frames)
    strict_late_tolerance_frames = resolve_strict_late_tolerance_frames(
        args.strict_late_tolerance_frames,
        classifier.window_policy.raw_window_len,
    )
    ffmpeg_exe = None if bool(args.disable_ffmpeg_video_only_remux) else find_ffmpeg_executable()
    video_only_cache_dir = None
    remux_video_only = bool(ffmpeg_exe is not None)
    if remux_video_only:
        video_only_cache_dir = (
            args.ffmpeg_video_cache_dir.expanduser().resolve()
            if args.ffmpeg_video_cache_dir is not None
            else (output_dir / "video_only_cache").resolve()
        )
    pose_pipeline = build_pose_pipeline(args, device=device, keypoint_weights=keypoint_weights, imgsz=resolved_imgsz)

    LOGGER.info(
        "Using classifier arch=%s | raw window=%d stride=%d | sampled window=%d stride=%d | frame_step=%d",
        classifier_name,
        classifier.window_policy.raw_window_len,
        classifier.window_policy.raw_window_stride,
        classifier.window_policy.sampled_window_len,
        classifier.window_policy.sampled_window_stride,
        classifier.window_policy.frame_step,
    )
    LOGGER.info("Using pose imgsz=%s on device=%s", f"{resolved_imgsz:g}", device)
    if remux_video_only and video_only_cache_dir is not None:
        LOGGER.info("Using FFmpeg video-only cache at %s", video_only_cache_dir)
    else:
        LOGGER.info("FFmpeg video-only remux is disabled or FFmpeg is unavailable; opening source AVIs directly.")
    LOGGER.info(
        "Strict timing metric: alert uses first confirmed window end frame with allowed interval [fall_start - %d, fall_end + %d].",
        strict_early_tolerance_frames,
        strict_late_tolerance_frames,
    )

    sequence_results: List[Dict[str, Any]] = []

    start_time = time.perf_counter()
    seq_iter: Iterable[LE2ISequence] = iter_progress(sequences, desc="LE2I videos", total=len(sequences), leave=True)
    for sequence in seq_iter:
        timing_annotation = timing_annotations.get(sequence.video_id)
        annotated_event_start_frame: Optional[int] = None
        annotated_event_end_frame: Optional[int] = None
        annotated_event_start_time_s: Optional[float] = None
        annotated_event_end_time_s: Optional[float] = None
        if (
            timing_annotation is not None
            and sequence.video_label == "fall"
            and timing_annotation.event_start_frame is not None
            and timing_annotation.event_end_frame is not None
        ):
            annotated_event_start_frame = int(timing_annotation.event_start_frame)
            annotated_event_end_frame = int(timing_annotation.event_end_frame)
            annotated_event_start_time_s = frame_time_seconds(int(annotated_event_start_frame) - 1, float(args.fps))
            annotated_event_end_time_s = frame_time_seconds(int(annotated_event_end_frame) - 1, float(args.fps))

        try:
            window_rows, video_summary = evaluate_sequence(
                sequence,
                pose_pipeline=pose_pipeline,
                classifier=classifier,
                timing_annotation=timing_annotation,
                fps=float(args.fps),
                threshold=float(args.threshold),
                min_consecutive_positive=int(args.min_consecutive_positive),
                strict_early_tolerance_frames=int(strict_early_tolerance_frames),
                strict_late_tolerance_frames=int(strict_late_tolerance_frames),
                ffmpeg_exe=ffmpeg_exe,
                video_only_cache_dir=video_only_cache_dir,
                remux_video_only=bool(remux_video_only),
            )
        except Exception as exc:
            LOGGER.warning("Failed to evaluate %s: %s", sequence.video_id, exc)
            sequence_results.append(
                {
                    "window_rows": [],
                    "base_video_summary": build_video_summary_for_decision(
                        {
                            "dataset": sequence.dataset,
                            "video_id": sequence.video_id,
                            "video_path": str(sequence.video_path),
                            "video_label": sequence.video_label,
                            "subset_name": sequence.subset_name,
                            "annotation_path": str(sequence.annotation_path),
                            "fps": float(args.fps),
                            "num_frames": 0,
                            "num_windows": 0,
                            "timing_annotation_available": bool(
                                sequence.video_label == "fall"
                                and timing_annotation is not None
                                and timing_annotation.event_start_frame is not None
                                and timing_annotation.event_end_frame is not None
                            ),
                            "annotated_event_start_frame": annotated_event_start_frame,
                            "annotated_event_end_frame": annotated_event_end_frame,
                            "annotated_event_start_time_s": annotated_event_start_time_s,
                            "annotated_event_end_time_s": annotated_event_end_time_s,
                            "num_readable_frames": 0,
                            "num_unreadable_frames": 0,
                            "pose_found_frames": 0,
                            "sampled_pose_found_frames": 0,
                            "sequence_root": str(sequence.subset_root),
                            "video_file": str(sequence.video_path),
                            "opened_video_path": str(sequence.video_path),
                            "warning": str(exc),
                        },
                        window_rows=[],
                        threshold=float(args.threshold),
                        min_consecutive_positive=int(args.min_consecutive_positive),
                        strict_early_tolerance_frames=int(strict_early_tolerance_frames),
                        strict_late_tolerance_frames=int(strict_late_tolerance_frames),
                        apply_to_rows=True,
                    ),
                }
            )
            continue

        sequence_results.append(
            {
                "window_rows": window_rows,
                "base_video_summary": dict(video_summary),
            }
        )

    elapsed_s = time.perf_counter() - start_time

    window_rows_all = [row for result in sequence_results for row in result["window_rows"]]
    video_rows = [dict(result["base_video_summary"]) for result in sequence_results]
    configured_metrics = compute_metrics(video_rows)
    configured_metrics["elapsed_s"] = float(elapsed_s)
    configured_strict_video_metrics: Optional[Dict[str, Any]] = None
    if any(
        bool(row.get("strict_eligible", False)) and str(row.get("video_label", "")) == "fall"
        for row in video_rows
    ):
        configured_strict_video_metrics = compute_metrics(
            video_rows,
            outcome_key="strict_outcome",
            eligible_key="strict_eligible",
        )
        configured_strict_video_metrics["elapsed_s"] = float(elapsed_s)
        configured_strict_video_metrics["outcome_reason_counts"] = count_field_values(
            video_rows,
            "strict_outcome_reason",
            eligible_key="strict_eligible",
        )

    selected_threshold = float(args.threshold)
    selected_min_consecutive_positive = int(args.min_consecutive_positive)
    decision_search_rows: Optional[List[Dict[str, Any]]] = None
    decision_search_best_row: Optional[Dict[str, Any]] = None
    decision_search_thresholds: List[float] = normalize_search_thresholds(
        args.search_thresholds,
        configured_threshold=float(args.threshold),
    )
    decision_search_min_values: List[int] = normalize_search_min_consecutive_values(
        args.search_min_consecutive_values,
        configured_min_consecutive_positive=int(args.min_consecutive_positive),
    )

    if bool(args.optimize_video_decision):
        decision_search_best_row, decision_search_rows = search_best_video_decision(
            sequence_results,
            thresholds=decision_search_thresholds,
            min_consecutive_values=decision_search_min_values,
            configured_threshold=float(args.threshold),
            configured_min_consecutive_positive=int(args.min_consecutive_positive),
            strict_early_tolerance_frames=int(strict_early_tolerance_frames),
            strict_late_tolerance_frames=int(strict_late_tolerance_frames),
        )
        selected_threshold = float(decision_search_best_row["threshold"])
        selected_min_consecutive_positive = int(decision_search_best_row["min_consecutive_positive"])

        video_rows = []
        for result in sequence_results:
            video_rows.append(
                build_video_summary_for_decision(
                    dict(result["base_video_summary"]),
                    window_rows=result["window_rows"],
                    threshold=float(selected_threshold),
                    min_consecutive_positive=int(selected_min_consecutive_positive),
                    strict_early_tolerance_frames=int(strict_early_tolerance_frames),
                    strict_late_tolerance_frames=int(strict_late_tolerance_frames),
                    apply_to_rows=True,
                )
            )
        window_rows_all = [row for result in sequence_results for row in result["window_rows"]]
    else:
        for result in sequence_results:
            build_video_summary_for_decision(
                dict(result["base_video_summary"]),
                window_rows=result["window_rows"],
                threshold=float(selected_threshold),
                min_consecutive_positive=int(selected_min_consecutive_positive),
                strict_early_tolerance_frames=int(strict_early_tolerance_frames),
                strict_late_tolerance_frames=int(strict_late_tolerance_frames),
                apply_to_rows=True,
            )

    metrics = compute_metrics(video_rows)
    metrics["elapsed_s"] = float(elapsed_s)
    strict_video_metrics: Optional[Dict[str, Any]] = None
    if any(
        bool(row.get("strict_eligible", False)) and str(row.get("video_label", "")) == "fall"
        for row in video_rows
    ):
        strict_video_metrics = compute_metrics(
            video_rows,
            outcome_key="strict_outcome",
            eligible_key="strict_eligible",
        )
        strict_video_metrics["elapsed_s"] = float(elapsed_s)
        strict_video_metrics["outcome_reason_counts"] = count_field_values(
            video_rows,
            "strict_outcome_reason",
            eligible_key="strict_eligible",
        )
    timing_window_metrics: Optional[Dict[str, Any]] = None
    if timing_annotations:
        timing_window_metrics = compute_timing_window_metrics(window_rows_all)

    output_dir.mkdir(parents=True, exist_ok=True)
    window_csv_path = output_dir / "window_predictions.csv"
    video_csv_path = output_dir / "video_summary.csv"
    metrics_json_path = output_dir / "metrics.json"
    run_config_path = output_dir / "run_config.json"
    decision_search_csv_path = output_dir / DEFAULT_DECISION_SEARCH_CSV_NAME

    window_fieldnames = [
        "dataset",
        "video_id",
        "video_path",
        "video_label",
        "subset_name",
        "annotation_path",
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
        "confirmed_detection_alert",
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
        "timing_annotation_available",
        "annotated_event_start_frame",
        "annotated_event_end_frame",
        "annotated_event_start_time_s",
        "annotated_event_end_time_s",
        "overlaps_annotated_fall_event",
        "annotated_positive_frames_in_window",
        "annotated_phase",
        "overlaps_transition_phase",
        "overlaps_post_fall_phase",
    ]
    video_fieldnames = [
        "dataset",
        "video_id",
        "video_path",
        "video_label",
        "subset_name",
        "annotation_path",
        "num_frames",
        "num_windows",
        "num_positive_windows",
        "max_fall_score",
        "first_positive_window_id",
        "first_detection_frame",
        "first_detection_time_s",
        "first_confirmed_detection_window_id",
        "first_confirmed_detection_frame",
        "first_confirmed_detection_time_s",
        "first_confirmed_detection_delay_frames",
        "first_confirmed_detection_delay_s",
        "detected_fall",
        "outcome",
        "first_detection_window_id",
        "timing_annotation_available",
        "annotated_event_start_frame",
        "annotated_event_end_frame",
        "annotated_event_start_time_s",
        "annotated_event_end_time_s",
        "timing_hit",
        "first_timing_hit_window_id",
        "first_timing_hit_frame",
        "first_timing_hit_time_s",
        "first_timing_hit_delay_frames",
        "first_timing_hit_delay_s",
        "strict_eligible",
        "strict_early_tolerance_frames",
        "strict_late_tolerance_frames",
        "strict_eval_start_frame",
        "strict_eval_end_frame",
        "strict_eval_start_time_s",
        "strict_eval_end_time_s",
        "strict_alert_window_id",
        "strict_alert_frame",
        "strict_alert_time_s",
        "strict_alert_delay_frames",
        "strict_alert_delay_s",
        "strict_detection_within_tolerance",
        "strict_outcome",
        "strict_outcome_reason",
        "num_readable_frames",
        "num_unreadable_frames",
        "pose_found_frames",
        "sampled_pose_found_frames",
        "sequence_root",
        "video_file",
        "opened_video_path",
        "warning",
    ]

    write_csv(window_csv_path, window_rows_all, window_fieldnames)
    write_csv(video_csv_path, video_rows, video_fieldnames)
    if decision_search_rows is not None:
        write_csv(
            decision_search_csv_path,
            decision_search_rows,
            [
                "threshold",
                "min_consecutive_positive",
                "accuracy",
                "balanced_accuracy",
                "f1",
                "precision",
                "recall",
                "specificity",
                "tp",
                "tn",
                "fp",
                "fn",
                "num_videos",
            ],
        )

    with metrics_json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "video_level_metrics": metrics,
                "strict_timing_video_metrics": strict_video_metrics,
                "timing_window_metrics": timing_window_metrics,
                "configured_video_level_metrics": configured_metrics if bool(args.optimize_video_decision) else None,
                "configured_strict_timing_video_metrics": (
                    configured_strict_video_metrics if bool(args.optimize_video_decision) else None
                ),
                "decision_rule_search": {
                    "enabled": bool(args.optimize_video_decision),
                    "selection_metric": "accuracy",
                    "search_thresholds": decision_search_thresholds,
                    "search_min_consecutive_values": decision_search_min_values,
                    "configured_threshold": float(args.threshold),
                    "configured_min_consecutive_positive": int(args.min_consecutive_positive),
                    "selected_threshold": float(selected_threshold),
                    "selected_min_consecutive_positive": int(selected_min_consecutive_positive),
                    "num_combinations_evaluated": int(len(decision_search_rows or [])),
                    "search_results_csv": (
                        str(decision_search_csv_path) if decision_search_rows is not None else None
                    ),
                    "best_result": decision_search_best_row,
                    "tie_breakers": [
                        "balanced_accuracy",
                        "f1",
                        "closeness_to_configured_threshold",
                        "closeness_to_configured_min_consecutive_positive",
                        "lower_threshold",
                        "lower_min_consecutive_positive",
                    ],
                },
            },
            handle,
            indent=2,
        )

    run_config = {
        "le2i_root": str(le2i_root),
        "keypoint_weights": str(keypoint_weights),
        "classifier_model": str(classifier_model),
        "device": str(device),
        "resolved_pose_imgsz": float(resolved_imgsz),
        "resolved_pose_half": bool(resolve_pose_half_arg(args.half, keypoint_weights, device)),
        "fps": float(args.fps),
        "threshold": float(args.threshold),
        "min_consecutive_positive": int(args.min_consecutive_positive),
        "strict_metric_alert_anchor": "first_confirmed_positive_window_end_frame",
        "strict_early_tolerance_frames": int(strict_early_tolerance_frames),
        "strict_late_tolerance_frames": int(strict_late_tolerance_frames),
        "ffmpeg_video_only_remux_enabled": bool(remux_video_only),
        "ffmpeg_executable": ffmpeg_exe,
        "ffmpeg_video_cache_dir": None if video_only_cache_dir is None else str(video_only_cache_dir),
        "selected_threshold": float(selected_threshold),
        "selected_min_consecutive_positive": int(selected_min_consecutive_positive),
        "optimize_video_decision": bool(args.optimize_video_decision),
        "decision_search_thresholds": decision_search_thresholds,
        "decision_search_min_consecutive_values": decision_search_min_values,
        "arch": args.arch if args.arch is not None else classifier_name,
        "classifier_backend": classifier_name,
        "motionbert_config": str(args.motionbert_config) if is_motionbert_classifier(classifier_model, args.arch) else None,
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
        "timing_annotation_source": str(timing_annotation_source),
        "num_timing_annotations_loaded": int(len(timing_annotations)),
        "num_timing_annotations_matched_selected_fall_sequences": int(
            sum(
                1
                for seq in sequences
                if seq.video_label == "fall" and seq.video_id in timing_annotations
            )
        ),
        "num_timing_annotations_matched_all_fall_sequences": int(
            sum(
                1
                for seq in sequences_all
                if seq.video_label == "fall" and seq.video_id in timing_annotations
            )
        ),
        "num_fall_sequences_without_strict_timing_annotation": int(
            sum(1 for seq in sequences_all if seq.video_label == "fall") - len(timing_annotations)
        ),
        "timing_ground_truth_positive_rule": "window overlaps the LE2I annotated fall interval from the annotation header",
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

    if bool(args.optimize_video_decision) and decision_search_best_row is not None:
        print_decision_search_summary(
            configured_threshold=float(args.threshold),
            configured_min_consecutive_positive=int(args.min_consecutive_positive),
            configured_metrics=configured_metrics,
            best_search_row=decision_search_best_row,
            num_combinations=int(len(decision_search_rows or [])),
        )
    print_metrics(metrics, title=f"{DATASET_NAME} clip-level video metrics")
    if strict_video_metrics is not None:
        print_metrics(strict_video_metrics, title=f"{DATASET_NAME} strict timing-aware video metrics")
        print(
            "  Strict alert time uses the end frame of the first confirmed positive window."
        )
        print(
            "  Allowed alert interval: "
            f"[fall_start - {int(strict_early_tolerance_frames)}, fall_end + {int(strict_late_tolerance_frames)}] frames."
        )
        print(f"  Strict outcome reasons: {json.dumps(strict_video_metrics.get('outcome_reason_counts', {}), sort_keys=True)}")
    if timing_window_metrics is not None:
        print_metric_block(f"{DATASET_NAME} timing-aware window metrics", timing_window_metrics)
        print(
            "  Timing GT positives are windows overlapping the annotated fall interval "
            "from the LE2I annotation header."
        )
    print()
    print("Outputs")
    print(f"  Window CSV:  {window_csv_path}")
    print(f"  Video CSV:   {video_csv_path}")
    print(f"  Metrics:     {metrics_json_path}")
    print(f"  Run config:  {run_config_path}")
    if decision_search_rows is not None:
        print(f"  Search CSV:  {decision_search_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
