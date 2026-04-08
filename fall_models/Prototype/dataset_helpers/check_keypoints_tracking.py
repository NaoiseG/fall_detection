#!/usr/bin/env python3
"""
Flag Camera2 .npz files where the tracked pose center falls inside one of three
known bad background regions.

Behavior:
- Only scans Camera2 files
- Uses absolute pixel coordinates, not normalized coordinates
- Region 1 is checked over the whole clip
- Region 2 is checked only at the start of the clip
- Region 3 is checked over the whole clip
- CSV output includes suspicious entries only
- No CSV unless --output is provided

Expected .npz keys:
- kpts_xy      [T, P, 17, 2]
- kpts_conf    [T, P, 17]
- person_conf  [T, P]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12
TORSO_IDXS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]


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
    T = xy.shape[0]
    centers = np.full((T, 2), np.nan, dtype=np.float32)

    for t in range(T):
        visible = conf[t] >= conf_thres

        torso_vis = [i for i in TORSO_IDXS if visible[i]]
        if len(torso_vis) >= 2:
            centers[t] = np.mean(xy[t, torso_vis], axis=0)
            continue

        pts = xy[t, visible]
        if len(pts) >= 4:
            centers[t] = np.mean(pts, axis=0)

    return centers


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


def in_box(center: np.ndarray, xmin: float, xmax: float, ymin: float, ymax: float) -> bool:
    if not np.isfinite(center[0]) or not np.isfinite(center[1]):
        return False
    return xmin <= center[0] <= xmax and ymin <= center[1] <= ymax


def fraction_in_box(
    centers: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> float:
    valid = np.isfinite(centers[:, 0])
    if valid.sum() == 0:
        return 0.0

    inside = 0
    total = 0
    for c in centers[valid]:
        total += 1
        if in_box(c, xmin, xmax, ymin, ymax):
            inside += 1

    return inside / total if total > 0 else 0.0


def analyze_npz(
    npz_path: Path,
    conf_thres: float,
    r1_xmin: float,
    r1_xmax: float,
    r1_ymin: float,
    r1_ymax: float,
    r2_xmin: float,
    r2_xmax: float,
    r2_ymin: float,
    r2_ymax: float,
    r3_xmin: float,
    r3_xmax: float,
    r3_ymin: float,
    r3_ymax: float,
    start_frames: int,
    min_valid_fraction: float,
    region1_fraction_threshold: float,
    region2_fraction_threshold: float,
    region3_fraction_threshold: float,
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
    valid = np.isfinite(centers[:, 0])
    valid_fraction = float(np.mean(valid)) if len(valid) else 0.0

    valid_centers = centers[valid]
    if len(valid_centers) > 0:
        mean_center = np.mean(valid_centers, axis=0)
        min_center = np.min(valid_centers, axis=0)
        max_center = np.max(valid_centers, axis=0)
    else:
        mean_center = np.array([np.nan, np.nan], dtype=np.float32)
        min_center = np.array([np.nan, np.nan], dtype=np.float32)
        max_center = np.array([np.nan, np.nan], dtype=np.float32)

    # Region 1: check whole clip
    region1_fraction = fraction_in_box(
        centers, r1_xmin, r1_xmax, r1_ymin, r1_ymax
    )
    region1_hit = region1_fraction >= region1_fraction_threshold

    # Region 2: check only the start
    n = min(start_frames, len(centers))
    start_centers = centers[:n]
    region2_fraction = fraction_in_box(
        start_centers, r2_xmin, r2_xmax, r2_ymin, r2_ymax
    )
    region2_hit = region2_fraction >= region2_fraction_threshold

    # Region 3: check whole clip
    region3_fraction = fraction_in_box(
        centers, r3_xmin, r3_xmax, r3_ymin, r3_ymax
    )
    region3_hit = region3_fraction >= region3_fraction_threshold

    suspicious = (
        valid_fraction >= min_valid_fraction
        and (region1_hit or region2_hit or region3_hit)
    )

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
        "frames": int(len(centers)),
        "valid_fraction": round(valid_fraction, 4),
        "mean_cx_px": round(float(mean_center[0]), 2),
        "mean_cy_px": round(float(mean_center[1]), 2),
        "min_cx_px": round(float(min_center[0]), 2) if np.isfinite(min_center[0]) else np.nan,
        "min_cy_px": round(float(min_center[1]), 2) if np.isfinite(min_center[1]) else np.nan,
        "max_cx_px": round(float(max_center[0]), 2) if np.isfinite(max_center[0]) else np.nan,
        "max_cy_px": round(float(max_center[1]), 2) if np.isfinite(max_center[1]) else np.nan,
        "region1_fraction": round(region1_fraction, 4),
        "region1_hit": bool(region1_hit),
        "region2_start_fraction": round(region2_fraction, 4),
        "region2_hit": bool(region2_hit),
        "region3_fraction": round(region3_fraction, 4),
        "region3_hit": bool(region3_hit),
        "suspicious": bool(suspicious),
    }


def find_npz_files(root: Path, activities: Sequence[str]) -> List[Path]:
    files = []
    activity_set = set(activities)

    for p in root.rglob("*.npz"):
        if not any("Camera2" in part for part in p.parts):
            continue

        if "all" in activity_set:
            files.append(p)
            continue

        parts = set(p.parts)
        if any(activity in parts for activity in activity_set):
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
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--conf-thres", type=float, default=0.3, help="Keypoint confidence threshold")
    parser.add_argument("--start-frames", type=int, default=100, help="Frames from start used for region 2 check")
    parser.add_argument("--min-valid-fraction", type=float, default=0.4)

    # Region 1: the back person around x≈588, y≈197
    parser.add_argument("--r1-xmin", type=float, default=540)
    parser.add_argument("--r1-xmax", type=float, default=660)
    parser.add_argument("--r1-ymin", type=float, default=160)
    parser.add_argument("--r1-ymax", type=float, default=230)
    parser.add_argument(
        "--region1-fraction-threshold",
        type=float,
        default=0.3,
        help="Fraction of valid frames that must fall in region 1",
    )

    # Region 2: the early-clip wrong person around x≈374, y≈134
    parser.add_argument("--r2-xmin", type=float, default=260)
    parser.add_argument("--r2-xmax", type=float, default=430)
    parser.add_argument("--r2-ymin", type=float, default=100)
    parser.add_argument("--r2-ymax", type=float, default=190)
    parser.add_argument(
        "--region2-fraction-threshold",
        type=float,
        default=0.3,
        help="Fraction of start frames that must fall in region 2",
    )

    # Region 3: another back person around x~=487, y~=137
    parser.add_argument("--r3-xmin", type=float, default=465)
    parser.add_argument("--r3-xmax", type=float, default=515)
    parser.add_argument("--r3-ymin", type=float, default=105)
    parser.add_argument("--r3-ymax", type=float, default=190)
    parser.add_argument(
        "--region3-fraction-threshold",
        type=float,
        default=0.3,
        help="Fraction of valid frames that must fall in region 3",
    )

    args = parser.parse_args()

    files = find_npz_files(args.root, args.activities)
    print(f"Found {len(files)} Camera2 .npz files to scan")

    rows: List[Dict[str, object]] = []
    failed: List[Tuple[str, str]] = []

    for i, fpath in enumerate(files, 1):
        try:
            row = analyze_npz(
                npz_path=fpath,
                conf_thres=args.conf_thres,
                r1_xmin=args.r1_xmin,
                r1_xmax=args.r1_xmax,
                r1_ymin=args.r1_ymin,
                r1_ymax=args.r1_ymax,
                r2_xmin=args.r2_xmin,
                r2_xmax=args.r2_xmax,
                r2_ymin=args.r2_ymin,
                r2_ymax=args.r2_ymax,
                r3_xmin=args.r3_xmin,
                r3_xmax=args.r3_xmax,
                r3_ymin=args.r3_ymin,
                r3_ymax=args.r3_ymax,
                start_frames=args.start_frames,
                min_valid_fraction=args.min_valid_fraction,
                region1_fraction_threshold=args.region1_fraction_threshold,
                region2_fraction_threshold=args.region2_fraction_threshold,
                region3_fraction_threshold=args.region3_fraction_threshold,
            )
            rows.append(row)

            if row["suspicious"]:
                print(
                    f"[{i}/{len(files)}] SUSPICIOUS | "
                    f"mean=({row['mean_cx_px']:.1f}, {row['mean_cy_px']:.1f}) | "
                    f"r1_frac={row['region1_fraction']:.2f} | "
                    f"r2_start_frac={row['region2_start_fraction']:.2f} | "
                    f"r3_frac={row['region3_fraction']:.2f} | "
                    f"{fpath}"
                )

        except Exception as e:
            failed.append((str(fpath), str(e)))
            print(f"[{i}/{len(files)}] ERROR | {fpath} | {e}")

    suspicious = [r for r in rows if r["suspicious"]]
    print(f"\nSuspicious files: {len(suspicious)} / {len(rows)}")

    if suspicious:
        print("\nFlagged files:")
        for r in suspicious[:100]:
            print(
                f"- {r['file']} | "
                f"mean=({r['mean_cx_px']}, {r['mean_cy_px']}) | "
                f"r1_hit={r['region1_hit']} | "
                f"r2_hit={r['region2_hit']} | "
                f"r3_hit={r['region3_hit']}"
            )

    if failed:
        print(f"\nFailed files: {len(failed)}")
        for path, err in failed[:20]:
            print(f"- {path}: {err}")

    if args.output is not None:
        if suspicious:
            write_csv(suspicious, args.output)
            print(f"\nWrote suspicious report: {args.output}")
        else:
            print("\nNo suspicious entries found; no CSV written.")


if __name__ == "__main__":
    main()
