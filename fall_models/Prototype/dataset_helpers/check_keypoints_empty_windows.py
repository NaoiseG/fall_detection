#!/usr/bin/env python3
"""
Flag pose NPZ files where the tracked person is lost for a long tail segment and
not meaningfully recovered, producing empty training windows.

Behavior:
- Scans NPZ files and can filter by activity and camera directory
- Uses the same frame-validity rule as training:
  fraction(conf >= conf_thres) >= min_valid_frac
- Bridges short invalid gaps to ignore brief detector dropouts
- Ignores short valid blips so tiny reappearances do not count as recovery
- Flags clips with an unrecovered invalid tail that produces empty windows
- CSV output includes flagged entries only
- No CSV unless --output is provided

Expected .npz keys:
- kpts_conf    [T, P, 17] or [T, 17]
- person_conf  [T, P] or [T] (optional but preferred)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def choose_person_track(person_conf: np.ndarray) -> int:
    if person_conf.ndim == 1:
        return 0
    mean_conf = np.nanmean(person_conf, axis=0)
    if not np.any(np.isfinite(mean_conf)):
        return 0
    return int(np.nanargmax(mean_conf))


def choose_person_track_from_kpts(kpts_conf: np.ndarray) -> int:
    if kpts_conf.ndim == 2:
        return 0
    mean_conf = np.nanmean(kpts_conf, axis=(0, 2))
    if not np.any(np.isfinite(mean_conf)):
        return 0
    return int(np.nanargmax(mean_conf))


def parse_path_metadata(npz_path: Path) -> Dict[str, str]:
    parts = list(npz_path.parts)

    def find_prefix(prefix: str) -> str:
        for p in parts:
            if p.startswith(prefix):
                return p
        return ""

    camera_dir = ""
    for p in parts:
        if "Camera1" in p or "Camera2" in p:
            camera_dir = p
            break

    metadata = {
        "subject": find_prefix("Subject"),
        "activity": find_prefix("Activity"),
        "trial": find_prefix("Trial"),
        "variant": "",
        "model": "",
        "camera_dir": camera_dir,
    }

    for i, p in enumerate(parts):
        if p.startswith("yolo"):
            metadata["model"] = p
            if i + 1 < len(parts):
                metadata["variant"] = parts[i + 1]
            break

    return metadata


def compute_frame_valid(
    conf: np.ndarray,
    conf_thres: float,
    min_valid_frac: float,
) -> np.ndarray:
    if conf.ndim != 2:
        raise ValueError(f"Expected conf shape [T, K], got {conf.shape}")
    visible = np.nan_to_num(conf, nan=-np.inf) >= float(conf_thres)
    frac_valid = np.mean(visible, axis=1)
    return frac_valid >= float(min_valid_frac)


def bridge_short_invalid_gaps(valid: np.ndarray, max_gap_frames: int) -> np.ndarray:
    out = np.asarray(valid, dtype=bool).copy()
    max_gap_frames = max(0, int(max_gap_frames))
    if out.size == 0 or max_gap_frames <= 0:
        return out

    n = out.size
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap_frames:
            out[i:j] = True
        i = j

    return out


def suppress_short_valid_runs(valid: np.ndarray, min_run_frames: int) -> np.ndarray:
    out = np.asarray(valid, dtype=bool).copy()
    min_run_frames = max(1, int(min_run_frames))
    if out.size == 0 or min_run_frames <= 1:
        return out

    n = out.size
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < n and out[j]:
            j += 1
        if (j - i) < min_run_frames:
            out[i:j] = False
        i = j

    return out


def count_trailing_invalid(valid: np.ndarray) -> int:
    out = np.asarray(valid, dtype=bool)
    if out.size == 0:
        return 0

    trailing = 0
    for idx in range(out.size - 1, -1, -1):
        if out[idx]:
            break
        trailing += 1
    return trailing


def get_window_starts(num_frames: int, stride: int) -> List[int]:
    stride = max(1, int(stride))
    return list(range(0, max(1, int(num_frames)), stride))


def find_empty_window_starts(
    valid: np.ndarray,
    T: int,
    stride: int,
) -> List[int]:
    out = np.asarray(valid, dtype=bool)
    num_frames = int(out.size)
    starts = get_window_starts(num_frames, stride=stride)
    empty_starts: List[int] = []

    for s in starts:
        e = min(num_frames, s + int(T))
        if e <= s or not bool(np.any(out[s:e])):
            empty_starts.append(int(s))

    return empty_starts


def resolve_threshold_frames(
    fps: float,
    frames_arg: Optional[int],
    seconds_arg: Optional[float],
    default_seconds: float,
) -> int:
    if frames_arg is not None:
        return max(1, int(frames_arg))
    seconds = default_seconds if seconds_arg is None else float(seconds_arg)
    return max(1, int(math.ceil(float(fps) * seconds)))


def frame_to_sec(frame_idx: int, fps: float) -> float:
    if frame_idx < 0:
        return float("nan")
    return float(frame_idx) / float(fps)


def analyze_npz(
    npz_path: Path,
    conf_thres: float,
    min_valid_frac: float,
    window_T: int,
    window_stride: int,
    bridge_gap_frames: int,
    loss_tail_frames: Optional[int],
    loss_tail_seconds: Optional[float],
    recovery_frames: Optional[int],
    recovery_seconds: Optional[float],
    min_empty_windows: int,
) -> Dict[str, object]:
    data = np.load(npz_path, allow_pickle=True)

    if "kpts_conf" not in data.files:
        raise ValueError("Missing key {'kpts_conf'}")

    kpts_conf = data["kpts_conf"]
    if kpts_conf.ndim == 2:
        kpts_conf = kpts_conf[:, None, :]
    if kpts_conf.ndim != 3:
        raise ValueError(f"Unexpected kpts_conf shape: {kpts_conf.shape}")

    fps = float(np.ravel(data["fps"])[0]) if "fps" in data.files else 30.0

    if "person_conf" in data.files:
        person_conf = data["person_conf"]
        if person_conf.ndim == 1:
            person_conf = person_conf[:, None]
        if person_conf.ndim != 2:
            raise ValueError(f"Unexpected person_conf shape: {person_conf.shape}")
        person_idx = choose_person_track(person_conf)
    else:
        person_idx = choose_person_track_from_kpts(kpts_conf)

    person_idx = max(0, min(int(person_idx), int(kpts_conf.shape[1]) - 1))
    conf = kpts_conf[:, person_idx, :]

    raw_valid = compute_frame_valid(conf, conf_thres=conf_thres, min_valid_frac=min_valid_frac)
    bridged_valid = bridge_short_invalid_gaps(raw_valid, max_gap_frames=bridge_gap_frames)

    recovery_frames_used = resolve_threshold_frames(
        fps=fps,
        frames_arg=recovery_frames,
        seconds_arg=recovery_seconds,
        default_seconds=0.5,
    )
    effective_valid = suppress_short_valid_runs(bridged_valid, min_run_frames=recovery_frames_used)

    loss_tail_frames_used = resolve_threshold_frames(
        fps=fps,
        frames_arg=loss_tail_frames,
        seconds_arg=loss_tail_seconds,
        default_seconds=2.0,
    )

    raw_valid_fraction = float(np.mean(raw_valid)) if raw_valid.size else 0.0
    effective_valid_fraction = float(np.mean(effective_valid)) if effective_valid.size else 0.0

    tracked_any = bool(np.any(effective_valid))
    all_empty_clip = not tracked_any

    if tracked_any:
        valid_idxs = np.flatnonzero(effective_valid)
        first_valid_frame = int(valid_idxs[0])
        last_valid_frame = int(valid_idxs[-1])
    else:
        first_valid_frame = -1
        last_valid_frame = -1

    final_invalid_run_frames = int(count_trailing_invalid(effective_valid))
    if all_empty_clip:
        loss_start_frame = 0 if effective_valid.size > 0 else -1
    elif final_invalid_run_frames > 0:
        loss_start_frame = int(effective_valid.size - final_invalid_run_frames)
    else:
        loss_start_frame = -1

    empty_window_starts = find_empty_window_starts(
        effective_valid,
        T=int(window_T),
        stride=int(window_stride),
    )

    if loss_start_frame >= 0:
        tail_empty_window_starts = [s for s in empty_window_starts if s >= loss_start_frame]
    else:
        tail_empty_window_starts = []

    first_empty_window_start = empty_window_starts[0] if empty_window_starts else -1

    unrecovered_tail_loss = bool(
        tracked_any
        and final_invalid_run_frames >= int(loss_tail_frames_used)
        and last_valid_frame < (effective_valid.size - 1)
        and len(tail_empty_window_starts) >= int(min_empty_windows)
    )

    flagged = bool(all_empty_clip or unrecovered_tail_loss)
    if all_empty_clip:
        flag_reason = "all_empty_clip"
    elif unrecovered_tail_loss:
        flag_reason = "unrecovered_tail_loss"
    else:
        flag_reason = ""

    meta = parse_path_metadata(npz_path)

    return {
        "file": str(npz_path),
        "model": meta["model"],
        "variant": meta["variant"],
        "subject": meta["subject"],
        "activity": meta["activity"],
        "trial": meta["trial"],
        "camera_dir": meta["camera_dir"],
        "fps": round(fps, 3),
        "frames": int(effective_valid.size),
        "window_T": int(window_T),
        "window_stride": int(window_stride),
        "conf_thres": float(conf_thres),
        "min_valid_frac": float(min_valid_frac),
        "bridge_gap_frames": int(bridge_gap_frames),
        "recovery_frames": int(recovery_frames_used),
        "loss_tail_frames_threshold": int(loss_tail_frames_used),
        "raw_valid_fraction": round(raw_valid_fraction, 4),
        "effective_valid_fraction": round(effective_valid_fraction, 4),
        "tracked_any": bool(tracked_any),
        "all_empty_clip": bool(all_empty_clip),
        "first_valid_frame": first_valid_frame if first_valid_frame >= 0 else np.nan,
        "last_valid_frame": last_valid_frame if last_valid_frame >= 0 else np.nan,
        "loss_start_frame": loss_start_frame if loss_start_frame >= 0 else np.nan,
        "loss_start_sec": round(frame_to_sec(loss_start_frame, fps), 3) if loss_start_frame >= 0 else np.nan,
        "final_invalid_run_frames": int(final_invalid_run_frames),
        "final_invalid_run_sec": round(float(final_invalid_run_frames) / float(fps), 3) if fps > 0 else np.nan,
        "empty_window_count": int(len(empty_window_starts)),
        "tail_empty_window_count": int(len(tail_empty_window_starts)),
        "first_empty_window_start_frame": first_empty_window_start if first_empty_window_start >= 0 else np.nan,
        "first_empty_window_start_sec": round(frame_to_sec(first_empty_window_start, fps), 3) if first_empty_window_start >= 0 else np.nan,
        "flag_reason": flag_reason,
        "flagged": bool(flagged),
    }


def find_npz_files(
    root: Path,
    activities: Sequence[str],
    camera_dirs: Sequence[str],
) -> List[Path]:
    files: List[Path] = []
    activity_set = {str(x) for x in activities}
    camera_set = {str(x) for x in camera_dirs}

    for p in root.rglob("*.npz"):
        if "all" not in activity_set:
            parts = set(p.parts)
            if not any(activity in parts for activity in activity_set):
                continue

        if "all" not in camera_set:
            if not any(any(cam in part for cam in camera_set) for part in p.parts):
                continue

        files.append(p)

    return sorted(files)


def write_csv(rows: List[Dict[str, object]], output: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    parser.add_argument(
        "--camera-dirs",
        nargs="+",
        default=["all"],
        help="Camera directories to scan, e.g. Camera2 or Camera1 Camera2 or all",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--conf-thres", type=float, default=0.2, help="Keypoint confidence threshold")
    parser.add_argument(
        "--min-valid-frac",
        "--min-valid-fraction",
        dest="min_valid_frac",
        type=float,
        default=0.3,
        help="Min fraction of joints above conf_thres for a frame to count as valid",
    )
    parser.add_argument("--window-T", "--T", dest="window_T", type=int, default=64, help="Sliding window length")
    parser.add_argument("--window-stride", "--stride", dest="window_stride", type=int, default=16, help="Sliding window stride")
    parser.add_argument(
        "--bridge-gap-frames",
        type=int,
        default=5,
        help="Bridge invalid gaps up to this many frames so brief dropouts do not count as a loss",
    )
    parser.add_argument(
        "--loss-tail-seconds",
        type=float,
        default=2.0,
        help="Minimum unrecovered invalid tail, in seconds, needed to flag a clip",
    )
    parser.add_argument(
        "--loss-tail-frames",
        type=int,
        default=None,
        help="Optional frame-based override for unrecovered invalid tail threshold",
    )
    parser.add_argument(
        "--recovery-seconds",
        type=float,
        default=0.5,
        help="Minimum sustained valid run, in seconds, that counts as recovery",
    )
    parser.add_argument(
        "--recovery-frames",
        type=int,
        default=None,
        help="Optional frame-based override for sustained recovery threshold",
    )
    parser.add_argument(
        "--min-empty-windows",
        type=int,
        default=1,
        help="Require at least this many empty tail windows before flagging",
    )

    args = parser.parse_args()

    files = find_npz_files(args.root, args.activities, args.camera_dirs)
    print(f"Found {len(files)} .npz files to scan")

    rows: List[Dict[str, object]] = []
    failed: List[Tuple[str, str]] = []

    for i, fpath in enumerate(files, 1):
        try:
            row = analyze_npz(
                npz_path=fpath,
                conf_thres=args.conf_thres,
                min_valid_frac=args.min_valid_frac,
                window_T=args.window_T,
                window_stride=args.window_stride,
                bridge_gap_frames=args.bridge_gap_frames,
                loss_tail_frames=args.loss_tail_frames,
                loss_tail_seconds=args.loss_tail_seconds,
                recovery_frames=args.recovery_frames,
                recovery_seconds=args.recovery_seconds,
                min_empty_windows=args.min_empty_windows,
            )
            rows.append(row)

            if row["flagged"]:
                print(
                    f"[{i}/{len(files)}] FLAGGED | "
                    f"reason={row['flag_reason']} | "
                    f"tail={row['final_invalid_run_sec']:.2f}s | "
                    f"tail_empty_windows={row['tail_empty_window_count']} | "
                    f"{fpath}"
                )

        except Exception as e:
            failed.append((str(fpath), str(e)))
            print(f"[{i}/{len(files)}] ERROR | {fpath} | {e}")

    flagged = [r for r in rows if r["flagged"]]
    print(f"\nFlagged files: {len(flagged)} / {len(rows)}")

    if flagged:
        print("\nTop flagged files:")
        flagged_sorted = sorted(
            flagged,
            key=lambda r: (
                int(r["tail_empty_window_count"]),
                float(r["final_invalid_run_frames"]),
                -float(r["effective_valid_fraction"]),
            ),
            reverse=True,
        )
        for r in flagged_sorted[:100]:
            print(
                f"- {r['file']} | "
                f"reason={r['flag_reason']} | "
                f"tail={r['final_invalid_run_sec']}s | "
                f"tail_empty_windows={r['tail_empty_window_count']} | "
                f"valid_frac={r['effective_valid_fraction']}"
            )

    if failed:
        print(f"\nFailed files: {len(failed)}")
        for path, err in failed[:20]:
            print(f"- {path}: {err}")

    if args.output is not None:
        if flagged:
            write_csv(flagged, args.output)
            print(f"\nWrote flagged report: {args.output}")
        else:
            print("\nNo flagged entries found; no CSV written.")


if __name__ == "__main__":
    main()
