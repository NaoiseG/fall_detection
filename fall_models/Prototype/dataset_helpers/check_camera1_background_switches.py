#!/usr/bin/env python3
"""
Flag Camera1 .npz files where the selected pose track either:
- switches onto a small person in the upper/background area, or
- locks onto a small upper/background reflection/person from the start and
  never meaningfully acquires the foreground subject.

Behavior:
- Only scans Camera1 files
- Uses absolute pixel coordinates, not normalized coordinates
- Measures both pose center and body scale from the selected track
- Looks for sustained background-like runs after an initial guard period
- Also looks for persistent reflection/background-like locking from the start
- CSV output includes suspicious entries only
- No CSV unless --output is provided

Expected .npz keys:
- kpts_xy      [T, P, 17, 2]
- kpts_conf    [T, P, 17]
- person_conf  [T, P]

 python $PROTOTYPE_DIR/dataset_helpers/check_camera1_background_switches.py --root ~/scratch/keypoints/UPFall_keypoints/yolo11l/base --output ~/scratch/keypoints/camera1_background_report.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12
TORSO_IDXS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
Run = Tuple[int, int]


def choose_person_track(person_conf: np.ndarray) -> int:
    mean_conf = np.nanmean(person_conf, axis=0)
    return int(np.nanargmax(mean_conf))


def compute_centers(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thres: float = 0.3,
) -> np.ndarray:
    """
    xy:   [T, 17, 2]
    conf: [T, 17]
    returns centers: [T, 2] with NaN where unavailable
    """
    frame_count = xy.shape[0]
    centers = np.full((frame_count, 2), np.nan, dtype=np.float32)

    for frame_idx in range(frame_count):
        visible = conf[frame_idx] >= conf_thres

        torso_vis = [joint_idx for joint_idx in TORSO_IDXS if visible[joint_idx]]
        if len(torso_vis) >= 2:
            centers[frame_idx] = np.mean(xy[frame_idx, torso_vis], axis=0)
            continue

        pts = xy[frame_idx, visible]
        if len(pts) >= 4:
            centers[frame_idx] = np.mean(pts, axis=0)

    return centers


def compute_body_scales(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thres: float = 0.3,
) -> np.ndarray:
    """
    xy:   [T, 17, 2]
    conf: [T, 17]
    returns scale: [T] using the visible-keypoint bbox diagonal, NaN if unavailable
    """
    frame_count = xy.shape[0]
    scales = np.full(frame_count, np.nan, dtype=np.float32)

    for frame_idx in range(frame_count):
        visible = conf[frame_idx] >= conf_thres
        pts = xy[frame_idx, visible]
        if len(pts) < 2:
            continue

        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        scales[frame_idx] = float(np.linalg.norm(maxs - mins))

    return scales


def parse_path_metadata(npz_path: Path) -> Dict[str, str]:
    parts = list(npz_path.parts)

    def extract_token(text: str, prefix: str) -> str:
        match = re.search(rf"({prefix}\d+)", text)
        return match.group(1) if match else ""

    def find_prefix(prefix: str) -> str:
        for part in parts:
            if part.startswith(prefix):
                return part
        return ""

    camera_dir = ""
    for part in parts:
        if "Camera1" in part or "Camera2" in part:
            camera_dir = part
            break

    metadata = {
        "subject": find_prefix("Subject"),
        "activity": find_prefix("Activity"),
        "trial": find_prefix("Trial"),
        "variant": "",
        "model": "",
        "camera_dir": camera_dir,
        }

    for idx, part in enumerate(parts):
        if part.startswith("yolo"):
            metadata["model"] = part
            if idx + 1 < len(parts):
                metadata["variant"] = parts[idx + 1]
            break

    for key in ("subject", "activity", "trial"):
        prefix = key.capitalize()
        metadata[key] = extract_token(metadata[key], prefix) or metadata[key]

    if camera_dir:
        for key in ("subject", "activity", "trial"):
            if metadata[key]:
                continue
            metadata[key] = extract_token(camera_dir, key.capitalize())

    return metadata


def close_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0 or mask.size == 0:
        return mask.copy()

    closed = mask.copy()
    idx = 0
    while idx < len(closed):
        if closed[idx]:
            idx += 1
            continue

        gap_start = idx
        while idx < len(closed) and not closed[idx]:
            idx += 1
        gap_end = idx

        if gap_start == 0 or gap_end == len(closed):
            continue

        if gap_end - gap_start <= max_gap:
            closed[gap_start:gap_end] = True

    return closed


def find_runs(mask: np.ndarray, min_len: int) -> List[Run]:
    runs: List[Run] = []
    start = None

    for idx, is_on in enumerate(mask):
        if is_on and start is None:
            start = idx
            continue

        if not is_on and start is not None:
            if idx - start >= min_len:
                runs.append((start, idx - 1))
            start = None

    if start is not None and len(mask) - start >= min_len:
        runs.append((start, len(mask) - 1))

    return runs


def spans_to_text(runs: Sequence[Run]) -> str:
    if not runs:
        return ""
    return ";".join(f"{start}-{end}" for start, end in runs)


def is_camera1_npz(npz_path: str | Path) -> bool:
    return any("Camera1" in str(part) for part in Path(npz_path).parts)


def runs_to_mask(frame_count: int, runs: Sequence[Run]) -> np.ndarray:
    mask = np.zeros((max(0, int(frame_count)),), dtype=bool)
    if mask.size == 0:
        return mask

    last_idx = int(mask.size - 1)
    for start, end in runs:
        start_i = max(0, int(start))
        end_i = min(int(end), last_idx)
        if start_i <= end_i:
            mask[start_i : end_i + 1] = True
    return mask


def _scalar_text(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif int(value.size) == 1:
            value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return str(value).strip()


def summarize_reference(
    centers: np.ndarray,
    scales: np.ndarray,
    valid_mask: np.ndarray,
    reference_scale_percentile: float,
) -> Tuple[float, float, int]:
    valid_scales = scales[valid_mask]
    if valid_scales.size == 0:
        return np.nan, np.nan, 0

    cutoff = float(np.nanpercentile(valid_scales, reference_scale_percentile))
    reference_mask = valid_mask & (scales >= cutoff)
    reference_count = int(np.sum(reference_mask))

    if reference_count == 0:
        reference_mask = valid_mask
        reference_count = int(np.sum(reference_mask))

    reference_scale = float(np.nanmedian(scales[reference_mask]))
    reference_cy = float(np.nanmedian(centers[reference_mask, 1]))
    return reference_scale, reference_cy, reference_count


def run_lengths(runs: Sequence[Run]) -> List[int]:
    return [end - start + 1 for start, end in runs]


def fraction_from_runs(runs: Sequence[Run], denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    frame_total = sum(end - start + 1 for start, end in runs)
    return frame_total / denominator


def first_run_start(runs: Sequence[Run]) -> int:
    return runs[0][0] if runs else -1


def fraction_true(mask: np.ndarray, valid_mask: np.ndarray) -> float:
    denom = int(np.sum(valid_mask))
    if denom <= 0:
        return 0.0
    return float(np.sum(mask & valid_mask)) / float(denom)


def compute_camera1_flagged_frame_mask(
    npz_path: str | Path,
    *,
    conf_thres: float = 0.3,
    reference_scale_percentile: float = 65.0,
    background_scale_ratio: float = 0.6,
    background_y_margin: float = 45.0,
    gap_frames: int = 3,
    min_run_frames: int = 6,
    start_guard_frames: int = 45,
    reflection_max_scale_px: float = 140.0,
    reflection_max_cy_px: float = 240.0,
    min_reflection_run_frames: int = 6,
) -> np.ndarray:
    """
    Returns a per-frame boolean mask for frames that look like the Camera1
    background-switch / reflection failure modes used by this script.

    The returned mask marks frames belonging to:
    - background-like runs that meet the minimum run length and start after the
      background-switch guard period, or
    - reflection-like runs that meet the minimum run length.
    """
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as data:
        required = {"kpts_xy", "kpts_conf", "person_conf"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"Missing keys {missing}")

        kpts_xy = data["kpts_xy"]
        kpts_conf = data["kpts_conf"]
        person_conf = data["person_conf"]
        source_npz_path = _scalar_text(data["source_npz_path"]) if "source_npz_path" in data.files else ""

    if not is_camera1_npz(npz_path) and not is_camera1_npz(source_npz_path):
        return np.zeros((int(kpts_xy.shape[0]),), dtype=bool)

    if kpts_xy.ndim != 4 or kpts_conf.ndim != 3 or person_conf.ndim != 2:
        raise ValueError(
            f"Unexpected shapes: "
            f"kpts_xy={kpts_xy.shape}, "
            f"kpts_conf={kpts_conf.shape}, "
            f"person_conf={person_conf.shape}"
        )

    person_idx = choose_person_track(person_conf)
    xy = kpts_xy[:, person_idx, :, :]
    conf = kpts_conf[:, person_idx, :]

    centers = compute_centers(xy, conf, conf_thres=conf_thres)
    scales = compute_body_scales(xy, conf, conf_thres=conf_thres)

    valid_mask = np.isfinite(centers[:, 0]) & np.isfinite(scales)
    reference_scale, reference_cy, _reference_count = summarize_reference(
        centers=centers,
        scales=scales,
        valid_mask=valid_mask,
        reference_scale_percentile=reference_scale_percentile,
    )

    background_like_mask = np.zeros((len(centers),), dtype=bool)
    if np.isfinite(reference_scale) and np.isfinite(reference_cy):
        background_like_mask = (
            valid_mask
            & (scales <= float(reference_scale) * float(background_scale_ratio))
            & (centers[:, 1] <= float(reference_cy) - float(background_y_margin))
        )
    background_like_mask = close_short_gaps(background_like_mask, int(gap_frames))
    background_runs = find_runs(background_like_mask, int(min_run_frames))
    qualifying_background_runs = [run for run in background_runs if int(run[0]) >= int(start_guard_frames)]
    background_flag_mask = runs_to_mask(len(centers), qualifying_background_runs)

    reflection_like_mask = (
        valid_mask
        & (scales <= float(reflection_max_scale_px))
        & (centers[:, 1] <= float(reflection_max_cy_px))
    )
    reflection_like_mask = close_short_gaps(reflection_like_mask, int(gap_frames))
    reflection_runs = find_runs(reflection_like_mask, int(min_reflection_run_frames))
    reflection_flag_mask = runs_to_mask(len(centers), reflection_runs)

    return np.asarray(background_flag_mask | reflection_flag_mask, dtype=bool)


def build_camera1_majority_flagged_window_mask(
    *,
    meta: Dict[str, object],
    frame_mask_cache: Dict[str, np.ndarray] | None = None,
    majority_fraction: float = 0.33,
    conf_thres: float = 0.3,
    reference_scale_percentile: float = 65.0,
    background_scale_ratio: float = 0.6,
    background_y_margin: float = 45.0,
    gap_frames: int = 3,
    min_run_frames: int = 6,
    start_guard_frames: int = 45,
    reflection_max_scale_px: float = 140.0,
    reflection_max_cy_px: float = 240.0,
    min_reflection_run_frames: int = 6,
) -> np.ndarray:
    """
    Returns a boolean mask aligned to dataset window metadata.

    A window is marked True when strictly more than ``majority_fraction`` of its
    sampled frames belong to the Camera1 background/reflection failure mask.
    """
    source_indices = np.asarray(meta.get("window_source_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    start_frames = np.asarray(meta.get("window_start_frames_sampled", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    frame_counts = np.asarray(meta.get("window_frame_counts_sampled", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    source_paths = [str(p) for p in meta.get("source_npz_paths", [])]

    n_windows = int(source_indices.shape[0])
    if start_frames.shape[0] != n_windows or frame_counts.shape[0] != n_windows:
        raise RuntimeError(
            "Window metadata length mismatch while applying Camera1 background filtering: "
            f"source_indices={n_windows}, start_frames={int(start_frames.shape[0])}, "
            f"frame_counts={int(frame_counts.shape[0])}"
        )

    if frame_mask_cache is None:
        frame_mask_cache = {}

    skip_mask = np.zeros((n_windows,), dtype=bool)
    for idx in range(n_windows):
        src_idx = int(source_indices[idx])
        if not (0 <= src_idx < len(source_paths)):
            raise RuntimeError(
                "Window source index out of range while applying Camera1 background filtering: "
                f"window={idx}, source_index={src_idx}, num_sources={len(source_paths)}"
            )

        src_path = source_paths[src_idx]
        frame_count = int(frame_counts[idx])
        if frame_count <= 0:
            continue

        cache_key = Path(src_path).resolve().as_posix()
        frame_mask = frame_mask_cache.get(cache_key)
        if frame_mask is None:
            frame_mask = compute_camera1_flagged_frame_mask(
                src_path,
                conf_thres=float(conf_thres),
                reference_scale_percentile=float(reference_scale_percentile),
                background_scale_ratio=float(background_scale_ratio),
                background_y_margin=float(background_y_margin),
                gap_frames=int(gap_frames),
                min_run_frames=int(min_run_frames),
                start_guard_frames=int(start_guard_frames),
                reflection_max_scale_px=float(reflection_max_scale_px),
                reflection_max_cy_px=float(reflection_max_cy_px),
                min_reflection_run_frames=int(min_reflection_run_frames),
            )
            frame_mask_cache[cache_key] = frame_mask

        start = int(start_frames[idx])
        end = start + frame_count
        if start < 0 or end > int(frame_mask.shape[0]):
            raise RuntimeError(
                "Window frame span is out of range while applying Camera1 background filtering: "
                f"path={src_path}, start={start}, frame_count={frame_count}, num_frames={int(frame_mask.shape[0])}"
            )

        flagged_count = int(np.count_nonzero(frame_mask[start:end]))
        if flagged_count > (float(frame_count) * float(majority_fraction)):
            skip_mask[idx] = True

    return skip_mask


def analyze_npz(
    npz_path: Path,
    conf_thres: float,
    reference_scale_percentile: float,
    background_scale_ratio: float,
    background_y_margin: float,
    gap_frames: int,
    min_run_frames: int,
    start_guard_frames: int,
    min_valid_fraction: float,
    min_background_fraction: float,
    min_background_runs: int,
    reflection_start_frames: int,
    reflection_max_scale_px: float,
    reflection_max_cy_px: float,
    min_reflection_fraction: float,
    min_reflection_start_fraction: float,
    min_reflection_run_frames: int,
    min_reflection_start_valid_frames: int,
) -> Dict[str, object]:
    data = np.load(npz_path, allow_pickle=True)

    required = {"kpts_xy", "kpts_conf", "person_conf"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Missing keys {missing}")

    kpts_xy = data["kpts_xy"]
    kpts_conf = data["kpts_conf"]
    person_conf = data["person_conf"]

    if kpts_xy.ndim != 4 or kpts_conf.ndim != 3 or person_conf.ndim != 2:
        raise ValueError(
            f"Unexpected shapes: "
            f"kpts_xy={kpts_xy.shape}, "
            f"kpts_conf={kpts_conf.shape}, "
            f"person_conf={person_conf.shape}"
        )

    fps = float(np.ravel(data["fps"])[0]) if "fps" in data.files else 30.0

    person_idx = choose_person_track(person_conf)
    xy = kpts_xy[:, person_idx, :, :]
    conf = kpts_conf[:, person_idx, :]

    centers = compute_centers(xy, conf, conf_thres=conf_thres)
    scales = compute_body_scales(xy, conf, conf_thres=conf_thres)

    valid_mask = np.isfinite(centers[:, 0]) & np.isfinite(scales)
    valid_fraction = float(np.mean(valid_mask)) if len(valid_mask) else 0.0
    valid_count = int(np.sum(valid_mask))

    valid_centers = centers[valid_mask]
    valid_scales = scales[valid_mask]
    if len(valid_centers) > 0:
        mean_center = np.mean(valid_centers, axis=0)
        min_center = np.min(valid_centers, axis=0)
        max_center = np.max(valid_centers, axis=0)
        mean_scale = float(np.mean(valid_scales))
        min_scale = float(np.min(valid_scales))
        max_scale = float(np.max(valid_scales))
    else:
        mean_center = np.array([np.nan, np.nan], dtype=np.float32)
        min_center = np.array([np.nan, np.nan], dtype=np.float32)
        max_center = np.array([np.nan, np.nan], dtype=np.float32)
        mean_scale = np.nan
        min_scale = np.nan
        max_scale = np.nan

    reference_scale, reference_cy, reference_count = summarize_reference(
        centers=centers,
        scales=scales,
        valid_mask=valid_mask,
        reference_scale_percentile=reference_scale_percentile,
    )

    background_mask = np.zeros(len(centers), dtype=bool)
    if np.isfinite(reference_scale) and np.isfinite(reference_cy):
        background_mask = (
            valid_mask
            & (scales <= reference_scale * background_scale_ratio)
            & (centers[:, 1] <= reference_cy - background_y_margin)
        )

    background_mask = close_short_gaps(background_mask, gap_frames)
    background_runs = find_runs(background_mask, min_run_frames)
    qualifying_runs = [
        run for run in background_runs if run[0] >= start_guard_frames
    ]

    background_fraction = fraction_from_runs(background_runs, valid_count)
    qualifying_background_fraction = fraction_from_runs(qualifying_runs, valid_count)
    background_lengths = run_lengths(background_runs)
    qualifying_lengths = run_lengths(qualifying_runs)

    background_switch_suspicious = (
        valid_fraction >= min_valid_fraction
        and qualifying_background_fraction >= min_background_fraction
        and len(qualifying_runs) >= min_background_runs
    )

    reflection_like_mask = (
        valid_mask
        & (scales <= float(reflection_max_scale_px))
        & (centers[:, 1] <= float(reflection_max_cy_px))
    )
    reflection_like_mask = close_short_gaps(reflection_like_mask, gap_frames)
    reflection_runs = find_runs(reflection_like_mask, min_reflection_run_frames)
    reflection_lengths = run_lengths(reflection_runs)
    reflection_fraction = fraction_from_runs(reflection_runs, valid_count)

    start_n = min(len(reflection_like_mask), max(0, int(reflection_start_frames)))
    start_valid_mask = valid_mask[:start_n]
    start_reflection_mask = reflection_like_mask[:start_n]
    start_valid_count = int(np.sum(start_valid_mask))
    start_reflection_fraction = fraction_true(start_reflection_mask, start_valid_mask)
    reflection_start_runs = find_runs(start_reflection_mask, min_reflection_run_frames)
    reflection_start_lengths = run_lengths(reflection_start_runs)

    persistent_reflection_lock = (
        start_valid_count >= int(min_reflection_start_valid_frames)
        and start_reflection_fraction >= float(min_reflection_start_fraction)
        and reflection_fraction >= float(min_reflection_fraction)
    )

    suspicious = bool(background_switch_suspicious or persistent_reflection_lock)
    suspicious_reasons = []
    if background_switch_suspicious:
        suspicious_reasons.append("background_switch")
    if persistent_reflection_lock:
        suspicious_reasons.append("persistent_reflection_lock")

    meta = parse_path_metadata(npz_path)
    first_background_frame = first_run_start(background_runs)
    first_qualifying_frame = first_run_start(qualifying_runs)
    first_reflection_frame = first_run_start(reflection_runs)

    return {
        "file": str(npz_path),
        "model": meta["model"],
        "variant": meta["variant"],
        "subject": meta["subject"],
        "activity": meta["activity"],
        "trial": meta["trial"],
        "camera_dir": meta["camera_dir"],
        "fps": round(fps, 3),
        "frames": int(len(centers)),
        "valid_fraction": round(valid_fraction, 4),
        "mean_cx_px": round(float(mean_center[0]), 2),
        "mean_cy_px": round(float(mean_center[1]), 2),
        "min_cx_px": round(float(min_center[0]), 2) if np.isfinite(min_center[0]) else np.nan,
        "min_cy_px": round(float(min_center[1]), 2) if np.isfinite(min_center[1]) else np.nan,
        "max_cx_px": round(float(max_center[0]), 2) if np.isfinite(max_center[0]) else np.nan,
        "max_cy_px": round(float(max_center[1]), 2) if np.isfinite(max_center[1]) else np.nan,
        "mean_scale_px": round(mean_scale, 2) if np.isfinite(mean_scale) else np.nan,
        "min_scale_px": round(min_scale, 2) if np.isfinite(min_scale) else np.nan,
        "max_scale_px": round(max_scale, 2) if np.isfinite(max_scale) else np.nan,
        "reference_scale_px": round(reference_scale, 2) if np.isfinite(reference_scale) else np.nan,
        "reference_cy_px": round(reference_cy, 2) if np.isfinite(reference_cy) else np.nan,
        "reference_frame_count": reference_count,
        "background_fraction": round(background_fraction, 4),
        "qualifying_background_fraction": round(qualifying_background_fraction, 4),
        "background_switch_suspicious": bool(background_switch_suspicious),
        "background_run_count": len(background_runs),
        "qualifying_background_run_count": len(qualifying_runs),
        "longest_background_run_frames": max(background_lengths) if background_lengths else 0,
        "longest_qualifying_run_frames": max(qualifying_lengths) if qualifying_lengths else 0,
        "first_background_frame": first_background_frame,
        "first_background_sec": round(first_background_frame / fps, 3) if first_background_frame >= 0 else np.nan,
        "first_qualifying_background_frame": first_qualifying_frame,
        "first_qualifying_background_sec": round(first_qualifying_frame / fps, 3)
        if first_qualifying_frame >= 0 else np.nan,
        "background_run_spans": spans_to_text(background_runs),
        "qualifying_background_run_spans": spans_to_text(qualifying_runs),
        "reflection_like_fraction": round(reflection_fraction, 4),
        "start_reflection_like_fraction": round(start_reflection_fraction, 4),
        "reflection_run_count": len(reflection_runs),
        "longest_reflection_run_frames": max(reflection_lengths) if reflection_lengths else 0,
        "longest_start_reflection_run_frames": max(reflection_start_lengths) if reflection_start_lengths else 0,
        "first_reflection_frame": first_reflection_frame,
        "first_reflection_sec": round(first_reflection_frame / fps, 3) if first_reflection_frame >= 0 else np.nan,
        "reflection_run_spans": spans_to_text(reflection_runs),
        "persistent_reflection_lock": bool(persistent_reflection_lock),
        "suspicious_reasons": ",".join(suspicious_reasons),
        "suspicious": bool(suspicious),
    }


def find_npz_files(root: Path, activities: Sequence[str]) -> List[Path]:
    files = []
    activity_set = set(activities)

    for npz_path in root.rglob("*.npz"):
        if not any("Camera1" in part for part in npz_path.parts):
            continue

        if "all" in activity_set:
            files.append(npz_path)
            continue

        parts = set(npz_path.parts)
        if any(activity in parts for activity in activity_set):
            files.append(npz_path)

    return sorted(files)


def write_csv(rows: List[Dict[str, object]], output: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Root directory to scan")
    parser.add_argument(
        "--activities",
        nargs="+",
        default=["all"],
        help="Activities to scan, e.g. Activity6 or Activity1 Activity6 or all",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="Keypoint confidence threshold")
    parser.add_argument(
        "--reference-scale-percentile",
        type=float,
        default=65.0,
        help="Percentile used to anchor the large/foreground reference body size",
    )
    parser.add_argument(
        "--background-scale-ratio",
        type=float,
        default=0.6,
        help="Background frames must be at or below this fraction of the reference body scale",
    )
    parser.add_argument(
        "--background-y-margin",
        type=float,
        default=45.0,
        help="Background frames must be at least this many pixels above the reference center y",
    )
    parser.add_argument(
        "--gap-frames",
        type=int,
        default=3,
        help="Fill short gaps between background-like frames up to this length",
    )
    parser.add_argument(
        "--min-run-frames",
        type=int,
        default=6,
        help="Minimum contiguous background-like frames required for a run",
    )
    parser.add_argument(
        "--start-guard-frames",
        type=int,
        default=45,
        help="Ignore background runs that start before this many frames",
    )
    parser.add_argument("--min-valid-fraction", type=float, default=0.4)
    parser.add_argument(
        "--min-background-fraction",
        type=float,
        default=0.03,
        help="Minimum qualifying background-frame fraction required to flag a clip",
    )
    parser.add_argument(
        "--min-background-runs",
        type=int,
        default=1,
        help="Minimum number of qualifying background runs required to flag a clip",
    )
    parser.add_argument(
        "--reflection-start-frames",
        type=int,
        default=120,
        help="Frames from the start used to detect a persistent reflection/background lock",
    )
    parser.add_argument(
        "--reflection-max-scale-px",
        type=float,
        default=140.0,
        help="Reflection-like tracks must stay at or below this body-scale threshold",
    )
    parser.add_argument(
        "--reflection-max-cy-px",
        type=float,
        default=240.0,
        help="Reflection-like tracks must stay at or above this upper-frame center-y threshold",
    )
    parser.add_argument(
        "--min-reflection-fraction",
        type=float,
        default=0.8,
        help="Minimum fraction of valid frames that must look reflection-like to flag a persistent lock",
    )
    parser.add_argument(
        "--min-reflection-start-fraction",
        type=float,
        default=0.8,
        help="Minimum reflection-like fraction in the early clip to flag a persistent lock",
    )
    parser.add_argument(
        "--min-reflection-run-frames",
        type=int,
        default=6,
        help="Minimum contiguous reflection-like frames required for a reflection run",
    )
    parser.add_argument(
        "--min-reflection-start-valid-frames",
        type=int,
        default=8,
        help="Minimum number of valid early frames required before reflection-start checks apply",
    )

    args = parser.parse_args()

    files = find_npz_files(args.root, args.activities)
    print(f"Found {len(files)} Camera1 .npz files to scan")

    rows: List[Dict[str, object]] = []
    failed: List[Tuple[str, str]] = []

    for idx, npz_path in enumerate(files, 1):
        try:
            row = analyze_npz(
                npz_path=npz_path,
                conf_thres=args.conf_thres,
                reference_scale_percentile=args.reference_scale_percentile,
                background_scale_ratio=args.background_scale_ratio,
                background_y_margin=args.background_y_margin,
                gap_frames=args.gap_frames,
                min_run_frames=args.min_run_frames,
                start_guard_frames=args.start_guard_frames,
                min_valid_fraction=args.min_valid_fraction,
                min_background_fraction=args.min_background_fraction,
                min_background_runs=args.min_background_runs,
                reflection_start_frames=args.reflection_start_frames,
                reflection_max_scale_px=args.reflection_max_scale_px,
                reflection_max_cy_px=args.reflection_max_cy_px,
                min_reflection_fraction=args.min_reflection_fraction,
                min_reflection_start_fraction=args.min_reflection_start_fraction,
                min_reflection_run_frames=args.min_reflection_run_frames,
                min_reflection_start_valid_frames=args.min_reflection_start_valid_frames,
            )
            rows.append(row)

            if row["suspicious"]:
                print(
                    f"[{idx}/{len(files)}] SUSPICIOUS | "
                    f"reasons={row['suspicious_reasons']} | "
                    f"bg_frac={row['qualifying_background_fraction']:.2f} | "
                    f"refl_frac={row['reflection_like_fraction']:.2f} | "
                    f"first_bg={row['first_qualifying_background_frame']} | "
                    f"first_refl={row['first_reflection_frame']} | "
                    f"{npz_path}"
                )

        except Exception as exc:
            failed.append((str(npz_path), str(exc)))
            print(f"[{idx}/{len(files)}] ERROR | {npz_path} | {exc}")

    suspicious_rows = [row for row in rows if row["suspicious"]]
    print(f"\nSuspicious files: {len(suspicious_rows)} / {len(rows)}")

    if suspicious_rows:
        print("\nFlagged files:")
        for row in suspicious_rows[:100]:
            print(
                f"- {row['file']} | "
                f"reasons={row['suspicious_reasons']} | "
                f"bg_frac={row['qualifying_background_fraction']} | "
                f"refl_frac={row['reflection_like_fraction']} | "
                f"bg_spans={row['qualifying_background_run_spans']} | "
                f"refl_spans={row['reflection_run_spans']}"
            )

    if failed:
        print(f"\nFailed files: {len(failed)}")
        for path, err in failed[:20]:
            print(f"- {path}: {err}")

    if args.output is not None:
        if suspicious_rows:
            write_csv(suspicious_rows, args.output)
            print(f"\nWrote suspicious report: {args.output}")
        else:
            print("\nNo suspicious entries found; no CSV written.")


if __name__ == "__main__":
    main()
