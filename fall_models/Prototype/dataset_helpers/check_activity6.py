#!/usr/bin/env python3
"""
Scan UP-Fall keypoint .npz files under Activity6 and flag clips that do not
look like walking, or that appear to contain a long stationary intro.

Expected .npz structure (matching your sample):
- kpts_xy:       [T, P, 17, 2]
- kpts_conf:     [T, P, 17]
- person_conf:   [T, P]
- fps:           [1] or scalar-like

Example:
    python scan_activity6_walking.py \
        --root /home/people/21376026/scratch/keypoints/UPFall_keypoints \
        --output walking_scan.csv \
        --stationary-prefix-sec 2.0
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# YOLO/COCO 17-keypoint indices
NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

LEG_IDXS = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
TORSO_IDXS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]


def moving_average(x: np.ndarray, window: int = 5) -> np.ndarray:
    if len(x) == 0 or window <= 1:
        return x.copy()
    window = min(window, len(x))
    kernel = np.ones(window, dtype=np.float32) / window
    xpad = np.pad(x, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
    return np.vstack([
        np.convolve(xpad[:, 0], kernel, mode="valid"),
        np.convolve(xpad[:, 1], kernel, mode="valid"),
    ]).T


def safe_norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(v * v, axis=-1))


def choose_person_track(person_conf: np.ndarray) -> int:
    """
    Choose the most likely person track.
    person_conf shape: [T, P]
    """
    if person_conf.ndim != 2:
        raise ValueError(f"Expected person_conf shape [T,P], got {person_conf.shape}")
    mean_conf = np.nanmean(person_conf, axis=0)
    return int(np.nanargmax(mean_conf))


def estimate_body_scale(
    xy: np.ndarray, conf: np.ndarray, conf_thres: float = 0.3
) -> float:
    """
    Estimate a typical body size in pixels using shoulder/hip box when possible,
    otherwise bbox over all visible keypoints.
    xy:   [T, 17, 2]
    conf: [T, 17]
    """
    scales = []

    for t in range(xy.shape[0]):
        visible = conf[t] >= conf_thres
        pts = xy[t]

        torso_vis = [i for i in TORSO_IDXS if visible[i]]
        if len(torso_vis) >= 3:
            torso_pts = pts[torso_vis]
            w = torso_pts[:, 0].max() - torso_pts[:, 0].min()
            h = torso_pts[:, 1].max() - torso_pts[:, 1].min()
            s = max(w, h)
            if s > 1:
                scales.append(float(s))
                continue

        vis_pts = pts[visible]
        if len(vis_pts) >= 4:
            w = vis_pts[:, 0].max() - vis_pts[:, 0].min()
            h = vis_pts[:, 1].max() - vis_pts[:, 1].min()
            s = max(w, h)
            if s > 1:
                scales.append(float(s))

    if not scales:
        return 1.0
    return float(np.median(scales))


def compute_center_trajectory(
    xy: np.ndarray, conf: np.ndarray, conf_thres: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a robust body center per frame from torso if possible,
    otherwise from all visible keypoints.
    Returns:
        centers: [T, 2] with NaNs where unavailable
        valid:   [T] bool
    """
    T = xy.shape[0]
    centers = np.full((T, 2), np.nan, dtype=np.float32)

    for t in range(T):
        visible = conf[t] >= conf_thres
        torso_vis = [i for i in TORSO_IDXS if visible[i]]

        if len(torso_vis) >= 2:
            centers[t] = np.mean(xy[t, torso_vis], axis=0)
        else:
            vis_pts = xy[t, visible]
            if len(vis_pts) >= 4:
                centers[t] = np.mean(vis_pts, axis=0)

    valid = np.isfinite(centers[:, 0])
    return centers, valid


def compute_leg_motion(
    xy: np.ndarray, conf: np.ndarray, conf_thres: float = 0.3
) -> float:
    """
    Sum consecutive motion of leg keypoints. Higher usually means walking/running.
    Returns median leg motion per valid transition in pixels.
    """
    motions = []

    for kp in LEG_IDXS:
        vis = conf[:, kp] >= conf_thres
        pts = xy[:, kp, :]
        valid_idx = np.where(vis)[0]
        if len(valid_idx) < 2:
            continue
        diffs = pts[valid_idx[1:]] - pts[valid_idx[:-1]]
        d = safe_norm(diffs)
        if len(d):
            motions.extend(d.tolist())

    if not motions:
        return 0.0
    return float(np.median(motions))


def stationary_prefix_seconds(
    centers: np.ndarray,
    valid: np.ndarray,
    fps: float,
    body_scale: float,
    speed_ratio_thres: float = 0.01,
) -> float:
    """
    Measure how long the clip stays almost stationary at the start.
    Threshold is relative to body size.
    """
    idx = np.where(valid)[0]
    if len(idx) < 3:
        return math.inf

    c = centers[idx]
    c = moving_average(c, window=5)
    speeds = safe_norm(np.diff(c, axis=0))

    # stationary threshold in pixels/frame
    px_thres = max(0.5, body_scale * speed_ratio_thres)

    count = 0
    for s in speeds:
        if s <= px_thres:
            count += 1
        else:
            break

    return count / max(fps, 1e-6)


def analyze_npz(npz_path: Path, conf_thres: float = 0.3) -> Dict[str, float | str | bool]:
    data = np.load(npz_path, allow_pickle=True)

    required = {"kpts_xy", "kpts_conf", "person_conf"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Missing keys {missing} in {npz_path}")

    kpts_xy = data["kpts_xy"]      # [T,P,17,2]
    kpts_conf = data["kpts_conf"]  # [T,P,17]
    person_conf = data["person_conf"]  # [T,P]

    fps_arr = data["fps"] if "fps" in data.files else np.array([30], dtype=np.int32)
    fps = float(np.ravel(fps_arr)[0])

    if kpts_xy.ndim != 4 or kpts_conf.ndim != 3 or person_conf.ndim != 2:
        raise ValueError(
            f"Unexpected shapes in {npz_path}: "
            f"kpts_xy={kpts_xy.shape}, kpts_conf={kpts_conf.shape}, person_conf={person_conf.shape}"
        )

    person_idx = choose_person_track(person_conf)

    xy = kpts_xy[:, person_idx, :, :]      # [T,17,2]
    conf = kpts_conf[:, person_idx, :]     # [T,17]

    body_scale = estimate_body_scale(xy, conf, conf_thres=conf_thres)
    centers, valid = compute_center_trajectory(xy, conf, conf_thres=conf_thres)

    valid_fraction = float(np.mean(valid)) if len(valid) else 0.0

    if valid.sum() >= 2:
        c = centers[valid]
        c = moving_average(c, window=5)
        diffs = np.diff(c, axis=0)
        step_disp = safe_norm(diffs)

        total_path = float(np.sum(step_disp))
        net_disp = float(np.linalg.norm(c[-1] - c[0]))
        x_span = float(np.max(c[:, 0]) - np.min(c[:, 0]))
        y_span = float(np.max(c[:, 1]) - np.min(c[:, 1]))
    else:
        total_path = 0.0
        net_disp = 0.0
        x_span = 0.0
        y_span = 0.0

    leg_motion = compute_leg_motion(xy, conf, conf_thres=conf_thres)
    stationary_prefix = stationary_prefix_seconds(
        centers, valid, fps=fps, body_scale=body_scale
    )

    # Normalize by body scale to reduce camera-distance sensitivity
    body_scale = max(body_scale, 1.0)
    total_path_norm = total_path / body_scale
    net_disp_norm = net_disp / body_scale
    x_span_norm = x_span / body_scale
    leg_motion_norm = leg_motion / body_scale

    # Heuristic rules:
    # - walking should usually show horizontal translation and leg movement
    # - clips with long stationary prefix are suspicious for Activity6
    looks_like_walking = (
        valid_fraction >= 0.60
        and x_span_norm >= 0.60
        and leg_motion_norm >= 0.015
    )

    long_stationary_intro = stationary_prefix >= 2.0

    suspicious = (not looks_like_walking) or long_stationary_intro

    return {
        "file": str(npz_path),
        "fps": fps,
        "frames": int(xy.shape[0]),
        "valid_fraction": round(valid_fraction, 4),
        "body_scale_px": round(body_scale, 2),
        "total_path_px": round(total_path, 2),
        "net_disp_px": round(net_disp, 2),
        "x_span_px": round(x_span, 2),
        "y_span_px": round(y_span, 2),
        "leg_motion_px": round(leg_motion, 4),
        "total_path_norm": round(total_path_norm, 4),
        "net_disp_norm": round(net_disp_norm, 4),
        "x_span_norm": round(x_span_norm, 4),
        "leg_motion_norm": round(leg_motion_norm, 6),
        "stationary_prefix_sec": round(stationary_prefix, 3),
        "looks_like_walking": bool(looks_like_walking),
        "long_stationary_intro": bool(long_stationary_intro),
        "suspicious": bool(suspicious),
    }


def find_activity6_npz(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*.npz"):
        parts = set(p.parts)
        if "Activity6" in parts:
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
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory to scan, e.g. /home/people/21376026/scratch/keypoints/UPFall_keypoints",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("walking_scan.csv"),
        help="CSV report output path",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.3,
        help="Keypoint confidence threshold",
    )
    parser.add_argument(
        "--stationary-prefix-sec",
        type=float,
        default=2.0,
        help="Threshold used in reporting only; files above this are flagged",
    )
    args = parser.parse_args()

    files = find_activity6_npz(args.root)
    print(f"Found {len(files)} Activity6 .npz files")

    rows: List[Dict[str, object]] = []
    failed: List[Tuple[str, str]] = []

    for i, fpath in enumerate(files, 1):
        try:
            row = analyze_npz(fpath, conf_thres=args.conf_thres)
            row["long_stationary_intro"] = row["stationary_prefix_sec"] >= args.stationary_prefix_sec
            row["suspicious"] = (not row["looks_like_walking"]) or row["long_stationary_intro"]
            rows.append(row)

            status = "SUSPICIOUS" if row["suspicious"] else "OK"
            print(f"[{i}/{len(files)}] {status} - {fpath}")
        except Exception as e:
            failed.append((str(fpath), str(e)))
            print(f"[{i}/{len(files)}] ERROR - {fpath} - {e}")

    write_csv(rows, args.output)
    print(f"\nWrote report: {args.output}")

    suspicious = [r for r in rows if r["suspicious"]]
    print(f"Suspicious files: {len(suspicious)} / {len(rows)}")

    if suspicious:
        print("\nTop suspicious files:")
        suspicious_sorted = sorted(
            suspicious,
            key=lambda r: (
                not r["looks_like_walking"],
                r["stationary_prefix_sec"],
                -r["x_span_norm"],
            ),
            reverse=True,
        )
        for r in suspicious_sorted[:30]:
            print(
                f"- {r['file']} | walking={r['looks_like_walking']} "
                f"| stationary_prefix_sec={r['stationary_prefix_sec']} "
                f"| x_span_norm={r['x_span_norm']} "
                f"| leg_motion_norm={r['leg_motion_norm']}"
            )

    if failed:
        print(f"\nFailed files: {len(failed)}")
        for path, err in failed[:20]:
            print(f"- {path}: {err}")


if __name__ == "__main__":
    main()