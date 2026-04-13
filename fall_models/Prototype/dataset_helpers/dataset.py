from __future__ import annotations

from typing import Tuple, Optional, Iterable, Dict, Any, List
from pathlib import Path
import glob
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Label scheme: merge fall subclasses into a single "Fall" class (7 classes total)
#
# This repo supports two raw label conventions in the underlying NPZs:
#   Case A: 1–11  (falls are 1..5, ADLs are 6..11)
#   Case B: 0–10  (falls are 0..4, ADLs are 5..10)
#
# We remap to a unified 0-index scheme:
#   new_id 0: Fall  (merged)
#   new_id 1..6: remaining 6 ADL classes, kept distinct
#
# Why remap *before* majority vote?
#   If the raw fall frames are split across multiple fall subclasses, computing
#   a majority vote on raw IDs can incorrectly prefer an ADL label even when
#   falls dominate in aggregate. We therefore compute window labels on the
#   merged label space.
# =============================================================================

# Raw fall IDs (in NPZ frame_labels space) for each convention
FALL_MERGE_SET_1_11 = {1, 2, 3, 4, 5}
FALL_MERGE_SET_0_10 = {0, 1, 2, 3, 4}

# New label names (index = new class id)
# NOTE: These names reflect the UP-Fall merged 7-class mapping used by MotionBERT:
#   0: Fall
#   1: Walking
#   2: Standing
#   3: Sitting
#   4: Picking up an object
#   5: Jumping
#   6: Laying
#
# The raw conventions (1-11 vs 0-10) only affect how frame labels are remapped; the
# merged 0..6 names are the same for both.
NEW_LABEL_NAMES_1_11 = ["Fall", "Walking", "Standing", "Sitting", "Picking up an object", "Jumping", "Laying"]
NEW_LABEL_NAMES_0_10 = ["Fall", "Walking", "Standing", "Sitting", "Picking up an object", "Jumping", "Laying"]

# Filled at runtime after convention detection (for easy introspection/debugging)
FALL_MERGE_SET: set[int] = set()
NEW_LABEL_NAMES: list[str] = []

# When dropping ambiguous windows (drop_ambig_share), always keep windows that
# contain these merged-class IDs. This helps avoid further undersampling for
# rare/important classes (e.g., Fall=0, Picking up an object=4).
AMBIG_KEEP_CLASS_IDS_MERGED: set[int] = {0, 4}


def detect_label_convention(observed_labels: Iterable[int], hint: Optional[str] = None) -> str:
    """
    Returns:
      "1-11" or "0-10"

    Rules:
      - If any label 11 exists => "1-11"
      - Else if any label 0 exists and max label is 10 => "0-10"
      - If ambiguous:
          * prefer hint if it is valid
          * else infer from min/max when possible
          * else default to "1-11" and log a warning
    """
    obs = sorted({int(x) for x in observed_labels})
    if not obs:
        if hint in {"1-11", "0-10"}:
            return hint
        print("[labels] Warning: no labels observed, defaulting to convention 1-11.")
        return "1-11"

    if 11 in obs:
        return "1-11"

    mn, mx = obs[0], obs[-1]
    if 0 in obs and mx == 10:
        return "0-10"

    if hint in {"1-11", "0-10"}:
        print(f"[labels] Warning: ambiguous raw labels (min={mn}, max={mx}), using hint={hint}.")
        return hint

    # Heuristic fallback
    if mn >= 1 and mx <= 11:
        print(f"[labels] Warning: ambiguous raw labels (min={mn}, max={mx}), defaulting to 1-11.")
        return "1-11"
    if mn >= 0 and mx <= 10:
        print(f"[labels] Warning: ambiguous raw labels (min={mn}, max={mx}), defaulting to 0-10.")
        return "0-10"

    print(f"[labels] Warning: unexpected label range (min={mn}, max={mx}), defaulting to 1-11.")
    return "1-11"


def get_fall_merge_set(convention: str) -> set[int]:
    if convention == "1-11":
        return set(FALL_MERGE_SET_1_11)
    if convention == "0-10":
        return set(FALL_MERGE_SET_0_10)
    raise ValueError(f"Unknown convention: {convention}")


def get_new_label_names(convention: str) -> list[str]:
    if convention == "1-11":
        return list(NEW_LABEL_NAMES_1_11)
    if convention == "0-10":
        return list(NEW_LABEL_NAMES_0_10)
    raise ValueError(f"Unknown convention: {convention}")


def remap_label(original_label: int, convention: str) -> int:
    """
    Maps a single raw label to the merged 7-class id space (0..6).
    """
    x = int(original_label)
    if convention == "1-11":
        if x in FALL_MERGE_SET_1_11:
            return 0
        if 6 <= x <= 11:
            return x - 5
        raise ValueError(f"Label {x} out of expected range for convention 1-11.")
    if convention == "0-10":
        if x in FALL_MERGE_SET_0_10:
            return 0
        if 5 <= x <= 10:
            return x - 4
        raise ValueError(f"Label {x} out of expected range for convention 0-10.")
    raise ValueError(f"Unknown convention: {convention}")


def remap_labels(labels: np.ndarray, convention: str) -> np.ndarray:
    """
    Vectorised remap for a 1D array of raw labels.
    """
    labels = labels.astype(np.int64, copy=False)

    if convention == "1-11":
        bad = (labels < 1) | (labels > 11)
        if bool(np.any(bad)):
            bad_vals = np.unique(labels[bad]).tolist()
            raise ValueError(f"Found raw labels outside 1..11: {bad_vals}")
        out = np.where(labels <= 5, 0, labels - 5).astype(np.int64, copy=False)
        return out

    if convention == "0-10":
        bad = (labels < 0) | (labels > 10)
        if bool(np.any(bad)):
            bad_vals = np.unique(labels[bad]).tolist()
            raise ValueError(f"Found raw labels outside 0..10: {bad_vals}")
        out = np.where(labels <= 4, 0, labels - 4).astype(np.int64, copy=False)
        return out

    raise ValueError(f"Unknown convention: {convention}")


def detect_label_convention_from_npzs(npz_paths: Iterable[str], hint: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Scan NPZ frame_labels to decide convention once, then reuse that choice
    for window generation.

    Returns:
      convention, stats dict
    """
    seen0 = False
    seen11 = False
    mn = None
    mx = None

    for p in npz_paths:
        data = np.load(p, allow_pickle=True)
        labs = data["frame_labels"].astype(np.int64, copy=False)
        if labs.size == 0:
            continue
        lmin = int(labs.min())
        lmax = int(labs.max())
        mn = lmin if mn is None else min(mn, lmin)
        mx = lmax if mx is None else max(mx, lmax)

        if np.any(labs == 0):
            seen0 = True
        if np.any(labs == 11):
            seen11 = True

        if seen11:
            # Rule: if label 11 exists, it's convention 1-11
            break

    observed = []
    if mn is not None:
        observed.append(mn)
    if mx is not None and mx != mn:
        observed.append(mx)
    if seen0:
        observed.append(0)
    if seen11:
        observed.append(11)

    convention = detect_label_convention(observed_labels=observed, hint=hint)

    stats = {
        "min_label": mn,
        "max_label": mx,
        "seen0": bool(seen0),
        "seen11": bool(seen11),
        "hint": hint,
    }
    return convention, stats


# ----------------------------
# NPZ discovery
# ----------------------------

def find_keypoints_npzs_subjects(output_root: Path, camera: int = 1, subjects=range(1, 6)):
    """
    Matches:
      Subject{s}/Activity*/Trial*/Subject{s}Activity*Trial*Camera{camera}/keypoints.npz
    """
    npzs = []
    for s in subjects:
        subj_root = output_root / f"Subject{s}"
        if not subj_root.exists():
            continue

        pat = subj_root / "Activity*" / "Trial*" / f"Subject{s}Activity*Trial*Camera{camera}" / "keypoints.npz"
        npzs.extend(glob.glob(str(pat), recursive=True))

    return sorted(npzs)


_CAMERA_ID_RE = re.compile(r"camera[_-]?(\d+)", flags=re.IGNORECASE)


def infer_camera_id_from_npz_path(npz_path: str, default: int = 0) -> int:
    """
    Best-effort camera id parser from an NPZ path.

    Expected patterns include "...Camera1/..." or "...camera_2/...".
    Returns `default` when no camera token can be parsed.
    """
    s = str(npz_path)
    m = _CAMERA_ID_RE.search(s)
    if m is None:
        return int(default)
    try:
        return int(m.group(1))
    except Exception:
        return int(default)


# ----------------------------
# Preprocessing helpers (1–3)
# ----------------------------

COCO_K = 17
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def _interp_short_gaps_1d(x: np.ndarray, valid: np.ndarray, max_gap: int) -> np.ndarray:
    """
    x: (N,) float
    valid: (N,) bool
    Interpolates gaps where missing runs are <= max_gap and bounded by valid points.
    For longer gaps, performs nearest hold (ffill/bfill).
    """
    N = x.shape[0]
    out = x.copy()

    idx_valid = np.where(valid)[0]
    if idx_valid.size == 0:
        # nothing valid: just return zeros (but caller will keep conf=0)
        return np.zeros_like(out)

    # First, fill everything by nearest hold (ffill then bfill)
    # ffill
    last = idx_valid[0]
    for i in range(0, N):
        if valid[i]:
            last = i
        out[i] = out[last]
    # bfill
    last = idx_valid[-1]
    for i in range(N - 1, -1, -1):
        if valid[i]:
            last = i
        out[i] = out[last]

    # Then, for short gaps bounded by valid points, replace hold with linear interpolation
    for a, b in zip(idx_valid[:-1], idx_valid[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        if gap <= max_gap:
            ya, yb = out[a], out[b]
            for j in range(1, gap + 1):
                t = j / (gap + 1)
                out[a + j] = (1 - t) * ya + t * yb

    return out


def _fill_and_mask_kpts(kxy: np.ndarray, kconf: np.ndarray, conf_thres: float, max_interp_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    kxy: (N,K,2), kconf: (N,K)
    - Marks joint missing when conf < conf_thres or xy invalid.
    - Fills xy via short-gap interpolation and hold for long gaps.
    - Sets conf to 0 where missing (after thresholding).
    """
    N, K, _ = kxy.shape
    xy = kxy.astype(np.float32, copy=True)
    conf = kconf.astype(np.float32, copy=True)

    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

    missing = conf < conf_thres  # (N,K)

    for j in range(K):
        v = ~missing[:, j]
        xj = xy[:, j, 0]
        yj = xy[:, j, 1]

        xj_f = _interp_short_gaps_1d(xj, v, max_gap=max_interp_gap)
        yj_f = _interp_short_gaps_1d(yj, v, max_gap=max_interp_gap)

        xy[:, j, 0] = xj_f
        xy[:, j, 1] = yj_f

        conf[missing[:, j], j] = 0.0

    return xy, conf


def _build_missing_mask(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thres: float,
    missing_mode: str,
) -> np.ndarray:
    """
    xy: (N,K,2), conf: (N,K)
    Returns missing mask (N,K) based on the requested mode.

    Modes:
      - conf_thres    : missing if conf < conf_thres
      - zeros_only    : missing if (x==0 and y==0)
      - conf_or_zeros : missing if (conf < conf_thres) OR (x==0 and y==0)
    """
    mode = str(missing_mode).lower().strip()
    if mode not in {"conf_thres", "zeros_only", "conf_or_zeros"}:
        raise ValueError(f"Unknown missing_mode: {missing_mode}")

    zeros = (xy[..., 0] == 0.0) & (xy[..., 1] == 0.0)  # (N,K)
    conf_low = conf < float(conf_thres)

    if mode == "conf_thres":
        return conf_low
    if mode == "zeros_only":
        return zeros
    return conf_low | zeros


def _interp_paper_group_linear_1d(x: np.ndarray, valid: np.ndarray, group: int) -> np.ndarray:
    """
    Paper-style interpolation in contiguous groups of `group` frames.

    Rules within each group:
      - If at least 2 known points exist, linearly interpolate missing points between them.
      - Leading/trailing missing values are filled with nearest known value (ffill/bfill).
      - If the entire group is missing, leave it as zero.
    """
    N = int(x.shape[0])
    g = int(group)
    if g <= 0:
        raise ValueError(f"interp_group must be > 0, got {group}")

    out = x.astype(np.float32, copy=True)

    for s in range(0, N, g):
        e = min(N, s + g)
        vg = valid[s:e]
        if not bool(np.any(vg)):
            out[s:e] = 0.0
            continue

        idx_known = np.where(vg)[0]
        if idx_known.size == 1:
            out[s:e] = out[s + int(idx_known[0])]
            continue

        fp = out[s:e][idx_known].astype(np.float32, copy=False)
        out[s:e] = np.interp(
            np.arange(e - s, dtype=np.float32),
            idx_known.astype(np.float32, copy=False),
            fp,
        ).astype(np.float32, copy=False)

    return out


def _fill_and_mask_kpts_paper(
    kxy: np.ndarray,
    kconf: np.ndarray,
    *,
    conf_thres: float,
    missing_mode: str,
    interp_mode: str,
    max_interp_gap: int,
    interp_group: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extended fill/mask supporting paper-style missingness and interpolation.

    kxy: (N,K,2), kconf: (N,K)
    Returns:
      xy_filled: (N,K,2)
      conf_filled: (N,K)
    """
    interp_mode = str(interp_mode).lower().strip()
    if interp_mode not in {"short_gap_hold", "paper_group_linear"}:
        raise ValueError(f"Unknown interp_mode: {interp_mode}")

    xy = kxy.astype(np.float32, copy=True)
    conf = kconf.astype(np.float32, copy=True)
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

    missing = _build_missing_mask(xy=xy, conf=conf, conf_thres=float(conf_thres), missing_mode=missing_mode)

    N, K, _ = xy.shape
    xy_out = xy.copy()
    conf_out = conf.copy()
    conf_out[missing] = 0.0

    zeros_based = str(missing_mode).lower().strip() in {"zeros_only", "conf_or_zeros"}

    for j in range(K):
        valid = ~missing[:, j]
        xj = xy[:, j, 0]
        yj = xy[:, j, 1]

        if interp_mode == "short_gap_hold":
            xj_f = _interp_short_gaps_1d(xj, valid, max_gap=int(max_interp_gap))
            yj_f = _interp_short_gaps_1d(yj, valid, max_gap=int(max_interp_gap))
            xy_out[:, j, 0] = xj_f
            xy_out[:, j, 1] = yj_f

            if zeros_based and bool(np.any(valid)):
                conf_out[missing[:, j], j] = 1.0

        else:  # paper_group_linear
            xj_f = _interp_paper_group_linear_1d(xj, valid, group=int(interp_group))
            yj_f = _interp_paper_group_linear_1d(yj, valid, group=int(interp_group))
            xy_out[:, j, 0] = xj_f
            xy_out[:, j, 1] = yj_f

            if zeros_based:
                g = int(interp_group)
                if g <= 0:
                    raise ValueError(f"interp_group must be > 0, got {interp_group}")
                for s in range(0, N, g):
                    e = min(N, s + g)
                    if bool(np.any(valid[s:e])):
                        m = missing[s:e, j]
                        if bool(np.any(m)):
                            idx = np.where(m)[0] + s
                            conf_out[idx, j] = 1.0

    return xy_out, conf_out


def _frame_center_scale(xy_t: np.ndarray, conf_t: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes a robust center and scale for one frame.
    xy_t: (K,2), conf_t: (K,)
    """
    K = xy_t.shape[0]
    valid = conf_t > 0.0

    def safe_mid(a: int, b: int):
        if a < K and b < K and valid[a] and valid[b]:
            return 0.5 * (xy_t[a] + xy_t[b]), True
        return np.zeros((2,), dtype=np.float32), False

    # Center: mid-hip > mid-shoulder > mean(valid)
    center, ok = safe_mid(L_HIP, R_HIP)
    if not ok:
        center, ok = safe_mid(L_SHOULDER, R_SHOULDER)
    if not ok:
        if valid.any():
            center = xy_t[valid].mean(axis=0).astype(np.float32)
        else:
            center = np.zeros((2,), dtype=np.float32)

    # Scale: shoulder width > hip width > torso proxy > 1
    scale = 1.0
    if L_SHOULDER < K and R_SHOULDER < K and valid[L_SHOULDER] and valid[R_SHOULDER]:
        scale = float(np.linalg.norm(xy_t[L_SHOULDER] - xy_t[R_SHOULDER]))
    elif L_HIP < K and R_HIP < K and valid[L_HIP] and valid[R_HIP]:
        scale = float(np.linalg.norm(xy_t[L_HIP] - xy_t[R_HIP]))
    else:
        sh, ok_sh = safe_mid(L_SHOULDER, R_SHOULDER)
        hp, ok_hp = safe_mid(L_HIP, R_HIP)
        if ok_sh and ok_hp:
            scale = float(np.linalg.norm(sh - hp))

    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    return center, scale


def _frame_root(xy_t: np.ndarray, conf_t: np.ndarray) -> np.ndarray:
    """
    Root (translation anchor) for one frame.
    Preferred: mid-hip; fallback to single hip; then shoulder center/single shoulder;
    then mean(valid); finally zeros.
    """
    K = int(xy_t.shape[0])
    valid = conf_t > 0.0

    def _ok(idx: int) -> bool:
        return 0 <= int(idx) < K and bool(valid[int(idx)])

    if _ok(L_HIP) and _ok(R_HIP):
        return (0.5 * (xy_t[L_HIP] + xy_t[R_HIP])).astype(np.float32, copy=False)
    if _ok(L_HIP):
        return xy_t[L_HIP].astype(np.float32, copy=False)
    if _ok(R_HIP):
        return xy_t[R_HIP].astype(np.float32, copy=False)

    if _ok(L_SHOULDER) and _ok(R_SHOULDER):
        return (0.5 * (xy_t[L_SHOULDER] + xy_t[R_SHOULDER])).astype(np.float32, copy=False)
    if _ok(L_SHOULDER):
        return xy_t[L_SHOULDER].astype(np.float32, copy=False)
    if _ok(R_SHOULDER):
        return xy_t[R_SHOULDER].astype(np.float32, copy=False)

    if bool(np.any(valid)):
        return xy_t[valid].mean(axis=0).astype(np.float32, copy=False)
    return np.zeros((2,), dtype=np.float32)


def _frame_scale_root_relative(xy_t: np.ndarray, conf_t: np.ndarray, root: np.ndarray, eps: float = 1e-6) -> float:
    """
    Robust per-frame scale for root-relative normalization.
    Fallback order: shoulder width -> hip width -> torso length -> robust joint spread.
    """
    K = int(xy_t.shape[0])
    valid = conf_t > 0.0

    def _ok(idx: int) -> bool:
        return 0 <= int(idx) < K and bool(valid[int(idx)])

    scale = 0.0
    if _ok(L_SHOULDER) and _ok(R_SHOULDER):
        scale = float(np.linalg.norm(xy_t[L_SHOULDER] - xy_t[R_SHOULDER]))
    elif _ok(L_HIP) and _ok(R_HIP):
        scale = float(np.linalg.norm(xy_t[L_HIP] - xy_t[R_HIP]))
    else:
        sh_pts: List[np.ndarray] = []
        hp_pts: List[np.ndarray] = []
        if _ok(L_SHOULDER):
            sh_pts.append(xy_t[L_SHOULDER])
        if _ok(R_SHOULDER):
            sh_pts.append(xy_t[R_SHOULDER])
        if _ok(L_HIP):
            hp_pts.append(xy_t[L_HIP])
        if _ok(R_HIP):
            hp_pts.append(xy_t[R_HIP])
        if sh_pts and hp_pts:
            sh_mid = np.mean(np.stack(sh_pts, axis=0), axis=0)
            hp_mid = np.mean(np.stack(hp_pts, axis=0), axis=0)
            scale = float(np.linalg.norm(sh_mid - hp_mid))
        else:
            pts = xy_t[valid]
            if pts.shape[0] >= 2:
                d = np.linalg.norm(pts - root[None, :], axis=1)
                d = d[np.isfinite(d)]
                if d.size > 0:
                    scale = float(np.percentile(d, 75.0))

    if not np.isfinite(scale) or scale < float(eps):
        scale = 1.0
    return float(scale)


def _normalize_xy(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """
    Per-frame translation + scale normalisation.
    xy: (N,K,2), conf: (N,K)
    """
    N, K, _ = xy.shape
    out = np.empty_like(xy, dtype=np.float32)

    for t in range(N):
        center, scale = _frame_center_scale(xy[t], conf[t])
        out[t] = (xy[t] - center[None, :]) / float(scale)

    return out


def _normalize_xy_root_scale(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """
    Root-relative + scale-normalized coordinates.
    """
    N, _, _ = xy.shape
    out = np.empty_like(xy, dtype=np.float32)
    for t in range(N):
        root = _frame_root(xy[t], conf[t])
        scale = _frame_scale_root_relative(xy[t], conf[t], root=root, eps=1e-6)
        out[t] = (xy[t] - root[None, :]) / float(scale)
    return out


def _compute_image_center(
    xy: np.ndarray,
    rp_center_mode: str,
    rp_img_w: Optional[int],
    rp_img_h: Optional[int],
) -> np.ndarray:
    """
    Determine (cx, cy) for paper-style Relative Position (RP) normalisation.

    rp_center_mode:
      - normalized_01 : center=(0.5, 0.5)
      - pixel         : center=(W/2, H/2) (requires rp_img_w, rp_img_h)
      - auto          : if coords look like [0,1] => normalized_01 else pixel (requires dims)
    """
    mode = str(rp_center_mode).lower().strip()
    if mode not in {"auto", "normalized_01", "pixel"}:
        raise ValueError(f"Unknown rp_center_mode: {rp_center_mode}")

    if mode == "normalized_01":
        return np.array([0.5, 0.5], dtype=np.float32)

    if mode == "pixel":
        if rp_img_w is None or rp_img_h is None:
            raise ValueError(
                "paper_rp normalisation with rp_center_mode='pixel' requires rp_img_w and rp_img_h. "
                "Pass --rp-img-w/--rp-img-h (or regenerate NPZs with stored image dimensions)."
            )
        return np.array([float(rp_img_w) / 2.0, float(rp_img_h) / 2.0], dtype=np.float32)

    # auto
    finite = np.isfinite(xy)
    non_zero = finite & (xy != 0.0)
    if bool(np.any(non_zero)):
        max_val = float(np.max(np.abs(xy[non_zero])))
    else:
        max_val = float(np.nanmax(np.abs(xy))) if xy.size else 0.0

    if max_val <= 1.5:
        return np.array([0.5, 0.5], dtype=np.float32)

    if rp_img_w is None or rp_img_h is None:
        raise ValueError(
            "paper_rp normalisation with rp_center_mode='auto' detected pixel-like coordinates "
            f"(max|xy|={max_val:.3f}). Please pass --rp-img-w/--rp-img-h (or regenerate NPZs with stored image dimensions)."
        )
    return np.array([float(rp_img_w) / 2.0, float(rp_img_h) / 2.0], dtype=np.float32)


def _reference_hip(xy_t: np.ndarray, conf_t: np.ndarray) -> np.ndarray:
    """
    Reference "hip" for RP translation (COCO-17):
      preferred: mid-hip (L_HIP+R_HIP) when both valid
      fallback : whichever hip is valid
      fallback : mid-shoulder (or whichever shoulder is valid)
      fallback : mean of valid joints
      fallback : (0,0)
    """
    K = int(xy_t.shape[0])
    valid = conf_t > 0.0

    def is_valid(idx: int) -> bool:
        return 0 <= int(idx) < K and bool(valid[int(idx)])

    if is_valid(L_HIP) and is_valid(R_HIP):
        return (0.5 * (xy_t[L_HIP] + xy_t[R_HIP])).astype(np.float32, copy=False)
    if is_valid(L_HIP):
        return xy_t[L_HIP].astype(np.float32, copy=False)
    if is_valid(R_HIP):
        return xy_t[R_HIP].astype(np.float32, copy=False)

    if is_valid(L_SHOULDER) and is_valid(R_SHOULDER):
        return (0.5 * (xy_t[L_SHOULDER] + xy_t[R_SHOULDER])).astype(np.float32, copy=False)
    if is_valid(L_SHOULDER):
        return xy_t[L_SHOULDER].astype(np.float32, copy=False)
    if is_valid(R_SHOULDER):
        return xy_t[R_SHOULDER].astype(np.float32, copy=False)

    if bool(np.any(valid)):
        return xy_t[valid].mean(axis=0).astype(np.float32, copy=False)

    return np.zeros((2,), dtype=np.float32)


def _normalize_xy_paper_rp(xy: np.ndarray, conf: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Paper-style Relative Position (RP) normalisation:
      - translation only, no scale normalisation
      - displacement is (center - reference_hip) per-frame
    """
    N, K, _ = xy.shape
    out = xy.astype(np.float32, copy=True)
    c = np.asarray(center, dtype=np.float32).reshape(2)

    for t in range(N):
        hip = _reference_hip(out[t], conf[t])
        d = c - hip
        out[t] = out[t] + d[None, :]

    return out


def _normalize_xy_paper_rp_scale(xy: np.ndarray, conf: np.ndarray, center: np.ndarray) -> np.ndarray:
    """
    Paper-style RP with additional per-frame scale normalisation:
      - translation: hip shifted to image center (same as paper_rp)
      - scale: divide by shoulder/hip/torso width after translation
    Preserves pose orientation (standing vs. lying) while removing
    subject-distance variance.
    """
    N, K, _ = xy.shape
    out = xy.astype(np.float32, copy=True)
    c = np.asarray(center, dtype=np.float32).reshape(2)

    for t in range(N):
        hip = _reference_hip(out[t], conf[t])
        out[t] = out[t] + (c - hip)[None, :]
        _, scale = _frame_center_scale(out[t], conf[t])
        out[t] = out[t] / float(scale)

    return out


def _add_velocity_channels(xy_norm: np.ndarray) -> np.ndarray:
    """
    xy_norm: (N,K,2)
    Returns vel: (N,K,2) with vel[0]=0 and vel[t]=xy[t]-xy[t-1]
    """
    vel = np.zeros_like(xy_norm, dtype=np.float32)
    vel[1:] = xy_norm[1:] - xy_norm[:-1]
    return vel


def _add_acceleration_channels(vel: np.ndarray) -> np.ndarray:
    """
    vel: (N,K,2)
    acc[0]=0, acc[1]=0, acc[t]=vel[t]-vel[t-1]
    """
    acc = np.zeros_like(vel, dtype=np.float32)
    acc[2:] = vel[2:] - vel[1:-1]
    return acc


# ----------------------------
# Window building
# ----------------------------

def _load_windows_from_npzs_core(
    *,
    npz_paths,
    T: Optional[int] = None,
    use_conf: bool = True,
    normalize: bool = True,
    add_vel: bool = True,
    add_acc: bool = True,
    add_global: bool = True,
    conf_thres: float = 0.05,
    max_interp_gap: int = 5,
    stride: int = 16,
    label_mode: str = "majority",
    binary_any_fall: bool = False,
    fall_ids_0based: Optional[list[int]] = None,
    fall_pct: float = 0.25,
    min_valid_frac: float = 0.3,
    add_mask_channel: bool = True,
    drop_ambig_share: float = 0.0,
    drop_ambig_nonfall_only: bool = True,
    label_convention: Optional[str] = None,
    normalize_mode: str = "center_scale",
    missing_mode: str = "conf_thres",
    interp_mode: str = "short_gap_hold",
    interp_group: int = 100,
    rp_center_mode: str = "auto",
    rp_img_w: Optional[int] = None,
    rp_img_h: Optional[int] = None,
    feature_mode: str = "full",
    motion_xy_scale: float = 0.25,
    drop_empty_windows: bool = False,
    collect_source_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, Optional[Dict[str, Any]]]:
    """
    Core implementation shared by public NPZ loading helpers.
    """
    npz_list: List[Path] = [Path(p) for p in npz_paths]
    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    T_used = T

    window_camera_ids: List[np.ndarray] = []
    window_source_indices: List[np.ndarray] = []
    window_candidate_indices: List[np.ndarray] = []
    window_start_frames_sampled: List[np.ndarray] = []
    window_end_frames_sampled: List[np.ndarray] = []
    window_frame_counts_sampled: List[np.ndarray] = []
    window_is_padded: List[np.ndarray] = []
    source_paths: List[str] = [p.as_posix() for p in npz_list]
    source_camera_ids = np.array([infer_camera_id_from_npz_path(p.as_posix()) for p in npz_list], dtype=np.int64)

    conv = label_convention
    if conv is None:
        conv, stats = detect_label_convention_from_npzs(npz_list)
        print(f"[labels] Auto-detected convention={conv} from NPZs (min={stats['min_label']}, max={stats['max_label']}).")
    else:
        if conv not in {"1-11", "0-10"}:
            raise ValueError("label_convention must be '1-11' or '0-10'.")

    global FALL_MERGE_SET, NEW_LABEL_NAMES
    FALL_MERGE_SET = get_fall_merge_set(conv)
    NEW_LABEL_NAMES = get_new_label_names(conv)

    for i, p in enumerate(npz_list):
        if i == 0 and T_used is None:
            window_result = make_window_tensors(
                p.as_posix(),
                T=None,
                use_conf=use_conf,
                normalize=normalize,
                normalize_mode=normalize_mode,
                add_vel=add_vel,
                add_acc=add_acc,
                add_global=add_global,
                conf_thres=conf_thres,
                max_interp_gap=max_interp_gap,
                missing_mode=missing_mode,
                interp_mode=interp_mode,
                interp_group=interp_group,
                stride=stride,
                label_mode=label_mode,
                binary_any_fall=binary_any_fall,
                fall_ids_0based=fall_ids_0based,
                fall_pct=fall_pct,
                min_valid_frac=min_valid_frac,
                add_mask_channel=add_mask_channel,
                drop_ambig_share=drop_ambig_share,
                drop_ambig_nonfall_only=drop_ambig_nonfall_only,
                label_convention=conv,
                rp_center_mode=rp_center_mode,
                rp_img_w=rp_img_w,
                rp_img_h=rp_img_h,
                feature_mode=feature_mode,
                motion_xy_scale=motion_xy_scale,
                drop_empty_windows=drop_empty_windows,
                collect_window_meta=bool(collect_source_meta),
            )
        else:
            window_result = make_window_tensors(
                p.as_posix(),
                T=T_used,
                use_conf=use_conf,
                normalize=normalize,
                normalize_mode=normalize_mode,
                add_vel=add_vel,
                add_acc=add_acc,
                add_global=add_global,
                conf_thres=conf_thres,
                max_interp_gap=max_interp_gap,
                missing_mode=missing_mode,
                interp_mode=interp_mode,
                interp_group=interp_group,
                stride=stride,
                label_mode=label_mode,
                binary_any_fall=binary_any_fall,
                fall_ids_0based=fall_ids_0based,
                fall_pct=fall_pct,
                min_valid_frac=min_valid_frac,
                add_mask_channel=add_mask_channel,
                drop_ambig_share=drop_ambig_share,
                drop_ambig_nonfall_only=drop_ambig_nonfall_only,
                label_convention=conv,
                rp_center_mode=rp_center_mode,
                rp_img_w=rp_img_w,
                rp_img_h=rp_img_h,
                feature_mode=feature_mode,
                motion_xy_scale=motion_xy_scale,
                drop_empty_windows=drop_empty_windows,
                collect_window_meta=bool(collect_source_meta),
            )

        if bool(collect_source_meta):
            X, y, T_curr, window_meta = window_result
            if i == 0 and T_used is None:
                T_used = int(T_curr)
            if window_meta is None:
                raise RuntimeError("collect_source_meta=True but make_window_tensors did not return metadata.")
        else:
            X, y, T_curr = window_result
            if i == 0 and T_used is None:
                T_used = int(T_curr)

        X_all.append(X)
        y_all.append(y)

        if bool(collect_source_meta):
            cam_id = int(infer_camera_id_from_npz_path(p.as_posix()))
            window_camera_ids.append(np.full((int(y.shape[0]),), cam_id, dtype=np.int64))
            window_source_indices.append(np.full((int(y.shape[0]),), int(i), dtype=np.int64))
            window_candidate_indices.append(np.asarray(window_meta["window_candidate_indices"], dtype=np.int64))
            window_start_frames_sampled.append(np.asarray(window_meta["window_start_frames_sampled"], dtype=np.int64))
            window_end_frames_sampled.append(np.asarray(window_meta["window_end_frames_sampled"], dtype=np.int64))
            window_frame_counts_sampled.append(np.asarray(window_meta["window_frame_counts_sampled"], dtype=np.int64))
            window_is_padded.append(np.asarray(window_meta["window_is_padded"], dtype=bool))

    if not X_all:
        raise RuntimeError("No NPZs found / no windows loaded.")

    total_windows = sum(int(x.shape[0]) for x in X_all)
    if total_windows == 0:
        raise RuntimeError(
            "No windows loaded after filtering. Check subject splits and window filters "
            "(for example empty-window or ambiguity dropping)."
        )

    X_cat = np.concatenate(X_all, axis=0)
    y_cat = np.concatenate(y_all, axis=0)
    if not bool(collect_source_meta):
        return X_cat, y_cat, int(T_used), None

    meta: Dict[str, Any] = {
        "window_camera_ids": np.concatenate(window_camera_ids, axis=0) if window_camera_ids else np.zeros((0,), dtype=np.int64),
        "window_source_indices": np.concatenate(window_source_indices, axis=0) if window_source_indices else np.zeros((0,), dtype=np.int64),
        "window_candidate_indices": np.concatenate(window_candidate_indices, axis=0) if window_candidate_indices else np.zeros((0,), dtype=np.int64),
        "window_start_frames_sampled": np.concatenate(window_start_frames_sampled, axis=0) if window_start_frames_sampled else np.zeros((0,), dtype=np.int64),
        "window_end_frames_sampled": np.concatenate(window_end_frames_sampled, axis=0) if window_end_frames_sampled else np.zeros((0,), dtype=np.int64),
        "window_frame_counts_sampled": np.concatenate(window_frame_counts_sampled, axis=0) if window_frame_counts_sampled else np.zeros((0,), dtype=np.int64),
        "window_is_padded": np.concatenate(window_is_padded, axis=0) if window_is_padded else np.zeros((0,), dtype=bool),
        "source_npz_paths": source_paths,
        "source_camera_ids": source_camera_ids,
    }
    return X_cat, y_cat, int(T_used), meta


def load_windows_from_npzs(
    npz_paths,
    T: Optional[int] = None,
    use_conf: bool = True,
    normalize: bool = True,
    add_vel: bool = True,
    add_acc: bool = True,
    add_global: bool = True,
    conf_thres: float = 0.05,
    max_interp_gap: int = 5,
    stride: int = 16,
    label_mode: str = "majority",  # center, majority, hybrid_center_fallpct
    binary_any_fall: bool = False,
    # Backwards compatible arg name: fall_ids_0based (legacy). Prefer leave as None.
    fall_ids_0based: Optional[list[int]] = None,
    fall_pct: float = 0.25,
    min_valid_frac: float = 0.3,
    add_mask_channel: bool = True,
    drop_ambig_share: float = 0.0,
    drop_ambig_nonfall_only: bool = True,
    # NEW: label convention control (auto-detect if None)
    label_convention: Optional[str] = None,
    # NEW: preprocessing mode control (backwards compatible defaults)
    normalize_mode: str = "center_scale",
    missing_mode: str = "conf_thres",
    interp_mode: str = "short_gap_hold",
    interp_group: int = 100,
    # NEW: paper-style RP normalisation args (only used when normalize_mode='paper_rp')
    rp_center_mode: str = "auto",
    rp_img_w: Optional[int] = None,
    rp_img_h: Optional[int] = None,
    # NEW: feature composition
    feature_mode: str = "full",
    motion_xy_scale: float = 0.25,
    drop_empty_windows: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Loads multiple trial NPZs, converts each to (W, T, K, C) windows,
    then concatenates across trials. Ensures the same T is used for all files.

    IMPORTANT:
      Returned y is in the merged 7-class space (0..6), not the raw 11-class IDs.
    """
    X_cat, y_cat, T_used, _ = _load_windows_from_npzs_core(
        npz_paths=npz_paths,
        T=T,
        use_conf=use_conf,
        normalize=normalize,
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        conf_thres=conf_thres,
        max_interp_gap=max_interp_gap,
        stride=stride,
        label_mode=label_mode,
        binary_any_fall=binary_any_fall,
        fall_ids_0based=fall_ids_0based,
        fall_pct=fall_pct,
        min_valid_frac=min_valid_frac,
        add_mask_channel=add_mask_channel,
        drop_ambig_share=drop_ambig_share,
        drop_ambig_nonfall_only=drop_ambig_nonfall_only,
        label_convention=label_convention,
        normalize_mode=normalize_mode,
        missing_mode=missing_mode,
        interp_mode=interp_mode,
        interp_group=interp_group,
        rp_center_mode=rp_center_mode,
        rp_img_w=rp_img_w,
        rp_img_h=rp_img_h,
        feature_mode=feature_mode,
        motion_xy_scale=motion_xy_scale,
        drop_empty_windows=drop_empty_windows,
        collect_source_meta=False,
    )
    return X_cat, y_cat, int(T_used)


def load_windows_with_source_meta_from_npzs(
    npz_paths,
    T: Optional[int] = None,
    use_conf: bool = True,
    normalize: bool = True,
    add_vel: bool = True,
    add_acc: bool = True,
    add_global: bool = True,
    conf_thres: float = 0.05,
    max_interp_gap: int = 5,
    stride: int = 16,
    label_mode: str = "majority",
    binary_any_fall: bool = False,
    fall_ids_0based: Optional[list[int]] = None,
    fall_pct: float = 0.25,
    min_valid_frac: float = 0.3,
    add_mask_channel: bool = True,
    drop_ambig_share: float = 0.0,
    drop_ambig_nonfall_only: bool = True,
    label_convention: Optional[str] = None,
    normalize_mode: str = "center_scale",
    missing_mode: str = "conf_thres",
    interp_mode: str = "short_gap_hold",
    interp_group: int = 100,
    rp_center_mode: str = "auto",
    rp_img_w: Optional[int] = None,
    rp_img_h: Optional[int] = None,
    feature_mode: str = "full",
    motion_xy_scale: float = 0.25,
    drop_empty_windows: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, Dict[str, Any]]:
    """
    Same as load_windows_from_npzs, but also returns source metadata aligned
    with each window (camera id + source NPZ index).
    """
    X_cat, y_cat, T_used, meta = _load_windows_from_npzs_core(
        npz_paths=npz_paths,
        T=T,
        use_conf=use_conf,
        normalize=normalize,
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        conf_thres=conf_thres,
        max_interp_gap=max_interp_gap,
        stride=stride,
        label_mode=label_mode,
        binary_any_fall=binary_any_fall,
        fall_ids_0based=fall_ids_0based,
        fall_pct=fall_pct,
        min_valid_frac=min_valid_frac,
        add_mask_channel=add_mask_channel,
        drop_ambig_share=drop_ambig_share,
        drop_ambig_nonfall_only=drop_ambig_nonfall_only,
        label_convention=label_convention,
        normalize_mode=normalize_mode,
        missing_mode=missing_mode,
        interp_mode=interp_mode,
        interp_group=interp_group,
        rp_center_mode=rp_center_mode,
        rp_img_w=rp_img_w,
        rp_img_h=rp_img_h,
        feature_mode=feature_mode,
        motion_xy_scale=motion_xy_scale,
        drop_empty_windows=drop_empty_windows,
        collect_source_meta=True,
    )
    if meta is None:
        meta = {
            "window_camera_ids": np.zeros((0,), dtype=np.int64),
            "window_source_indices": np.zeros((0,), dtype=np.int64),
            "window_candidate_indices": np.zeros((0,), dtype=np.int64),
            "window_start_frames_sampled": np.zeros((0,), dtype=np.int64),
            "window_end_frames_sampled": np.zeros((0,), dtype=np.int64),
            "window_frame_counts_sampled": np.zeros((0,), dtype=np.int64),
            "window_is_padded": np.zeros((0,), dtype=bool),
            "source_npz_paths": [],
            "source_camera_ids": np.zeros((0,), dtype=np.int64),
        }
    return X_cat, y_cat, int(T_used), meta


def _make_sliding_windows(
    Xf: np.ndarray,          # (N, K, C)
    labels_raw: np.ndarray,  # (N,) raw labels in NPZ space (Case A or B)
    conf: np.ndarray,        # (N, K) (needed for validity)
    T: int,
    stride: int,
    conf_thres: float,
    min_valid_frac: float,
    label_mode: str,
    binary_any_fall: bool,
    fall_pct: float,
    add_mask_channel: bool,
    drop_ambig_share: float,
    drop_ambig_nonfall_only: bool,
    layout: dict,
    label_convention: str,
    drop_empty_windows: bool = False,
    collect_window_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int] | Tuple[np.ndarray, np.ndarray, int, Dict[str, np.ndarray]]:
    """
    Returns:
      X_windows: (W, T, K, C(+1 if mask))
      y_windows: (W,) in merged 7-class space (0..6)
      T_used: int
      window_meta: optional metadata aligned to kept windows
    """
    N, K, C = Xf.shape
    stride = max(1, int(stride))

    # Frame validity
    frac_valid = (conf > conf_thres).mean(axis=1)  # (N,)
    frame_valid = frac_valid >= float(min_valid_frac)  # (N,)

    # starts: include last partial window
    starts = list(range(0, max(1, N), stride))
    X_windows = []
    y_windows = []
    kept_window_candidate_indices: List[int] = []
    kept_window_start_frames: List[int] = []
    kept_window_end_frames: List[int] = []
    kept_window_frame_counts: List[int] = []
    kept_window_is_padded: List[bool] = []

    for win_idx, s in enumerate(starts):
        e = s + T
        seq = Xf[s:e]                      # (L,K,C)
        labs_raw = labels_raw[s:e]         # (L,)
        valid = frame_valid[s:e]           # (L,)

        L = seq.shape[0]
        frame_count_before_padding = int(L)
        if L < T:
            pad = np.repeat(seq[-1:, :, :], repeats=(T - L), axis=0) if L > 0 else np.zeros((T, K, C), np.float32)

            if layout.get("conf_idx") is not None:
                pad[:, :, layout["conf_idx"]] = 0.0
            if layout.get("vel_slice") is not None:
                pad[:, :, layout["vel_slice"]] = 0.0
            if layout.get("acc_slice") is not None:
                pad[:, :, layout["acc_slice"]] = 0.0

            seq = np.concatenate([seq, pad], axis=0)
            last_lab = int(labs_raw[-1]) if L > 0 else int(0)
            labs_raw = np.concatenate([labs_raw, np.full((T - L,), last_lab, dtype=labs_raw.dtype)], axis=0)
            valid = np.concatenate([valid, np.zeros((T - L,), dtype=bool)], axis=0)

        # Remap per-frame labels into merged space for all label logic
        labs = remap_labels(labs_raw.astype(np.int64, copy=False), convention=label_convention)  # (T,)

        # Build mask: 1 only where frame_valid is True (and non-padded)
        mask_t = valid.astype(np.float32)  # (T,)

        # Drop windows that would otherwise become all zeros after hard masking.
        if bool(drop_empty_windows) and not bool(np.any(valid)):
            continue

        # Center label (merged space) is used both for window labeling and for
        # safeguarding rare classes from ambiguity-based dropping.
        c = T // 2
        if valid[c]:
            center_y = int(labs[c])
        else:
            idxs = np.where(valid)[0]
            center_y = int(labs[idxs[len(idxs)//2]]) if idxs.size > 0 else int(labs[c])

        # Optional: drop ambiguous windows.
        # Ambiguity is measured on *valid* frames, in merged label space.
        if drop_ambig_share and drop_ambig_share > 0.0:
            labs_valid = labs[valid]
            if labs_valid.size > 0:
                _vals, counts_v = np.unique(labs_valid, return_counts=True)
                top_share = float(counts_v.max()) / float(labs_valid.size)
                if top_share < float(drop_ambig_share):
                    keep_ids = AMBIG_KEEP_CLASS_IDS_MERGED
                    # Never drop windows that contain (or are centered on) rare classes.
                    if (int(center_y) in keep_ids) or bool(np.any(np.isin(labs_valid, list(keep_ids)))):
                        pass
                    elif bool(drop_ambig_nonfall_only):
                        has_any_fall = bool(np.any(labs_valid == 0))
                        if not has_any_fall:
                            continue
                    else:
                        continue

        # HARD MASK: zero all features on invalid frames so padding/low-valid frames can't leak pose.
        seq[~valid] = 0.0

        # Label assignment (merged space)
        if binary_any_fall:
            y = 1 if bool(np.any(labs[valid] == 0)) else 0

        else:
            if label_mode == "center":
                y = center_y

            elif label_mode == "majority":
                labs_valid = labs[valid]
                if labs_valid.size == 0:
                    y = center_y
                else:
                    vals, counts = np.unique(labs_valid, return_counts=True)
                    y = int(vals[np.argmax(counts)])

            elif label_mode == "hybrid_center_fallpct":
                labs_valid = labs[valid]
                if labs_valid.size == 0:
                    y = center_y
                else:
                    fall_share = float(np.mean(labs_valid == 0))
                    if fall_share >= float(fall_pct) and bool(np.any(labs_valid == 0)):
                        y = 0
                    else:
                        y = center_y

            else:
                raise ValueError(f"Unknown label_mode: {label_mode}")

        if add_mask_channel:
            m = mask_t[:, None, None].repeat(K, axis=1)
            seq = np.concatenate([seq, m], axis=-1)  # (T,K,C+1)

        X_windows.append(seq)
        y_windows.append(y)
        if bool(collect_window_meta):
            kept_window_candidate_indices.append(int(win_idx))
            kept_window_start_frames.append(int(s))
            kept_window_end_frames.append(int(s + max(frame_count_before_padding - 1, 0)))
            kept_window_frame_counts.append(int(frame_count_before_padding))
            kept_window_is_padded.append(bool(frame_count_before_padding < T))

        if s + stride >= N and s >= N - 1:
            break

    out_channels = C + (1 if add_mask_channel else 0)
    if not X_windows:
        empty_X = np.zeros((0, int(T), K, out_channels), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        if bool(collect_window_meta):
            empty_meta = {
                "window_candidate_indices": np.zeros((0,), dtype=np.int64),
                "window_start_frames_sampled": np.zeros((0,), dtype=np.int64),
                "window_end_frames_sampled": np.zeros((0,), dtype=np.int64),
                "window_frame_counts_sampled": np.zeros((0,), dtype=np.int64),
                "window_is_padded": np.zeros((0,), dtype=bool),
            }
            return empty_X, empty_y, int(T), empty_meta
        return empty_X, empty_y, int(T)

    X_stack = np.stack(X_windows)
    y_stack = np.array(y_windows, dtype=np.int64)
    if not bool(collect_window_meta):
        return X_stack, y_stack, int(T)

    window_meta = {
        "window_candidate_indices": np.array(kept_window_candidate_indices, dtype=np.int64),
        "window_start_frames_sampled": np.array(kept_window_start_frames, dtype=np.int64),
        "window_end_frames_sampled": np.array(kept_window_end_frames, dtype=np.int64),
        "window_frame_counts_sampled": np.array(kept_window_frame_counts, dtype=np.int64),
        "window_is_padded": np.array(kept_window_is_padded, dtype=bool),
    }
    return X_stack, y_stack, int(T), window_meta


def _global_features(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """
    xy: (N,K,2), conf: (N,K)
    Returns g: (N,4):
      com_x, com_y, com_v, aspect
    """
    N, K, _ = xy.shape
    g = np.zeros((N, 4), dtype=np.float32)
    eps = 1e-6

    for t in range(N):
        valid = conf[t] > 0.0
        if not np.any(valid):
            continue
        pts = xy[t, valid]

        com = pts.mean(axis=0)
        g[t, 0:2] = com

        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        w = float(mx[0] - mn[0])
        h = float(mx[1] - mn[1])
        g[t, 3] = h / (w + eps)

    dcom = np.zeros((N, 2), dtype=np.float32)
    dcom[1:] = g[1:, 0:2] - g[:-1, 0:2]
    g[:, 2] = np.linalg.norm(dcom, axis=1)

    return g


def feature_channels_per_joint(
    *,
    use_conf: bool,
    add_vel: bool,
    add_acc: bool,
    add_global: bool,
    add_mask_channel: bool,
    feature_mode: str = "full",
    motion_xy_scale: float = 0.25,
) -> int:
    """
    Deterministic feature-channel count per joint after preprocessing/windowing.
    """
    mode = str(feature_mode).lower().strip()
    if mode not in {"full", "motion_primary"}:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    has_vel = bool(add_vel)
    has_acc = bool(add_acc)
    if has_acc and not has_vel:
        raise ValueError("add_acc=True requires add_vel=True")
    if mode == "motion_primary" and (not has_vel or not has_acc):
        raise ValueError("feature_mode='motion_primary' requires add_vel=True and add_acc=True")

    channels = 0
    if mode == "full":
        channels += 2  # xy
    else:
        if float(motion_xy_scale) > 0.0:
            channels += 2  # reduced xy

    if bool(use_conf):
        channels += 1
    if has_vel:
        channels += 2
    if has_acc:
        channels += 2
    if bool(add_global):
        channels += 4
    if bool(add_mask_channel):
        channels += 1
    return int(channels)


def _compose_window_features(
    *,
    xy_used: np.ndarray,
    conf_filled: np.ndarray,
    use_conf: bool,
    add_vel: bool,
    add_acc: bool,
    add_global: bool,
    feature_mode: str,
    motion_xy_scale: float,
) -> Tuple[np.ndarray, Dict[str, Any], bool, bool]:
    """
    Build per-frame joint features and expose channel layout for downstream padding logic.
    """
    mode = str(feature_mode).lower().strip()
    if mode not in {"full", "motion_primary"}:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    has_vel = bool(add_vel)
    has_acc = bool(add_acc)
    if has_acc and not has_vel:
        raise ValueError("add_acc=True requires add_vel=True")
    if mode == "motion_primary" and (not has_vel or not has_acc):
        raise ValueError("feature_mode='motion_primary' requires add_vel=True and add_acc=True")

    vel = None
    acc = None
    if has_vel:
        vel = _add_velocity_channels(xy_used)
    if has_acc:
        assert vel is not None
        acc = _add_acceleration_channels(vel)

    parts: List[np.ndarray] = []
    idx = 0

    conf_idx = None
    vel_slice = None
    acc_slice = None
    global_slice = None

    if mode == "full":
        parts.append(xy_used)
        idx += 2

        if bool(use_conf):
            conf_idx = idx
            parts.append(conf_filled[..., None])
            idx += 1

        if has_vel:
            vel_slice = slice(idx, idx + 2)
            parts.append(vel)
            idx += 2

        if has_acc:
            acc_slice = slice(idx, idx + 2)
            assert acc is not None
            parts.append(acc)
            idx += 2

    else:
        if has_vel:
            vel_slice = slice(idx, idx + 2)
            parts.append(vel)
            idx += 2

        if has_acc:
            acc_slice = slice(idx, idx + 2)
            assert acc is not None
            parts.append(acc)
            idx += 2

        if float(motion_xy_scale) > 0.0:
            parts.append((float(motion_xy_scale) * xy_used).astype(np.float32, copy=False))
            idx += 2

        if bool(use_conf):
            conf_idx = idx
            parts.append(conf_filled[..., None])
            idx += 1

    if bool(add_global):
        g = _global_features(xy_used, conf_filled)
        gk = np.repeat(g[:, None, :], repeats=xy_used.shape[1], axis=1)
        global_slice = slice(idx, idx + 4)
        parts.append(gk)
        idx += 4

    Xf = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

    layout: Dict[str, Any] = {
        "conf_idx": conf_idx,
        "vel_slice": vel_slice,
        "acc_slice": acc_slice,
        "global_slice": global_slice,
        "mask_idx": None,
    }
    return Xf, layout, has_vel, has_acc


def make_window_tensors(
    npz_path: str,
    T: Optional[int] = None,
    use_conf: bool = True,
    person_idx: int = 0,
    normalize: bool = True,
    add_vel: bool = True,
    add_acc: bool = True,
    add_global: bool = True,
    conf_thres: float = 0.05,
    max_interp_gap: int = 5,
    stride: int = 16,
    label_mode: str = "majority",
    binary_any_fall: bool = False,
    # Backwards compat: legacy fall_ids_0based, only used for older code paths.
    fall_ids_0based: Optional[list[int]] = None,
    fall_pct: float = 0.25,
    min_valid_frac: float = 0.3,
    add_mask_channel: bool = True,
    drop_ambig_share: float = 0.0,
    drop_ambig_nonfall_only: bool = True,
    # NEW: pass through detected convention so we do not re-detect per file.
    label_convention: Optional[str] = None,
    # NEW: preprocessing mode control (backwards compatible defaults)
    normalize_mode: str = "center_scale",
    missing_mode: str = "conf_thres",
    interp_mode: str = "short_gap_hold",
    interp_group: int = 100,
    # NEW: paper-style RP normalisation args (only used when normalize_mode='paper_rp')
    rp_center_mode: str = "auto",
    rp_img_w: Optional[int] = None,
    rp_img_h: Optional[int] = None,
    # NEW: feature composition controls
    feature_mode: str = "full",
    motion_xy_scale: float = 0.25,
    drop_empty_windows: bool = False,
    collect_window_meta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int] | Tuple[np.ndarray, np.ndarray, int, Dict[str, np.ndarray]]:
    """
    Converts frame-level pose data into window-level tensors.

    Returns:
        X: (W, T, K, C)
        y: (W,) merged 7-class labels (0..6)
        T: frames per window used
        window_meta: optional metadata aligned to the returned windows
    """
    data = np.load(npz_path, allow_pickle=True)

    kxy = data["kpts_xy"][:, person_idx]       # (N, K, 2)
    kconf = data["kpts_conf"][:, person_idx]   # (N, K)
    labels_raw = data["frame_labels"].astype(np.int64, copy=False)  # (N,)

    conv = label_convention
    if conv is None:
        conv = detect_label_convention(np.unique(labels_raw))
        print(f"[labels] Auto-detected convention={conv} for file: {Path(npz_path).as_posix()}")
    if conv not in {"1-11", "0-10"}:
        raise ValueError("label_convention must be '1-11' or '0-10'.")

    global FALL_MERGE_SET, NEW_LABEL_NAMES
    FALL_MERGE_SET = get_fall_merge_set(conv)
    NEW_LABEL_NAMES = get_new_label_names(conv)

    if str(missing_mode).lower().strip() == "conf_thres" and str(interp_mode).lower().strip() == "short_gap_hold":
        # Preserve exact legacy behaviour for the default code path.
        xy_filled, conf_filled = _fill_and_mask_kpts(kxy, kconf, conf_thres=conf_thres, max_interp_gap=max_interp_gap)
    else:
        xy_filled, conf_filled = _fill_and_mask_kpts_paper(
            kxy,
            kconf,
            conf_thres=float(conf_thres),
            missing_mode=str(missing_mode),
            interp_mode=str(interp_mode),
            max_interp_gap=int(max_interp_gap),
            interp_group=int(interp_group),
        )

    if not bool(normalize):
        xy_used = xy_filled.astype(np.float32, copy=False)
    else:
        nm = str(normalize_mode).lower().strip()
        if nm == "center_scale":
            xy_used = _normalize_xy(xy_filled, conf_filled)
        elif nm == "root_scale":
            xy_used = _normalize_xy_root_scale(xy_filled, conf_filled)
        elif nm == "paper_rp":
            center = _compute_image_center(
                xy=xy_filled,
                rp_center_mode=str(rp_center_mode),
                rp_img_w=rp_img_w,
                rp_img_h=rp_img_h,
            )
            xy_used = _normalize_xy_paper_rp(xy_filled, conf_filled, center=center)
        elif nm == "paper_rp_scale":
            center = _compute_image_center(
                xy=xy_filled,
                rp_center_mode=str(rp_center_mode),
                rp_img_w=rp_img_w,
                rp_img_h=rp_img_h,
            )
            xy_used = _normalize_xy_paper_rp_scale(xy_filled, conf_filled, center=center)
        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

    Xf, layout, has_vel, has_acc = _compose_window_features(
        xy_used=xy_used,
        conf_filled=conf_filled,
        use_conf=bool(use_conf),
        add_vel=bool(add_vel),
        add_acc=bool(add_acc),
        add_global=bool(add_global),
        feature_mode=str(feature_mode),
        motion_xy_scale=float(motion_xy_scale),
    )

    if bool(add_mask_channel):
        layout["mask_idx"] = int(Xf.shape[-1])
    else:
        layout["mask_idx"] = None

    if T is None:
        T = 64

    window_result = _make_sliding_windows(
        Xf=Xf,
        labels_raw=labels_raw.astype(np.int64, copy=False),
        conf=conf_filled,
        T=int(T),
        stride=int(stride),
        conf_thres=float(conf_thres),
        min_valid_frac=float(min_valid_frac),
        label_mode=str(label_mode),
        binary_any_fall=bool(binary_any_fall),
        fall_pct=float(fall_pct),
        add_mask_channel=bool(add_mask_channel),
        drop_ambig_share=float(drop_ambig_share),
        drop_ambig_nonfall_only=bool(drop_ambig_nonfall_only),
        layout=layout,
        label_convention=str(conv),
        drop_empty_windows=bool(drop_empty_windows),
        collect_window_meta=bool(collect_window_meta),
    )
    if not bool(collect_window_meta):
        X_windows, y_windows, T_used = window_result
        return X_windows, y_windows, T_used

    X_windows, y_windows, T_used, window_meta = window_result
    return X_windows, y_windows, T_used, window_meta


# ----------------------------
# Loader (legacy helper)
# ----------------------------

def make_loader(
    subjects: list[str],
    camera: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    use_conf: bool = True,
    normalize: bool = True,
    add_vel: bool = True,
    add_acc: bool = True,
    add_global: bool = True,
    conf_thres: float = 0.05,
    max_interp_gap: int = 5,
):
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints/outputs_npz")
    npz_paths = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=camera, subjects=subjects)
    if not npz_paths:
        raise RuntimeError("No NPZs found. Check OUTPUT_ROOT, camera, and subjects.")

    X, y, _ = load_windows_from_npzs(
        npz_paths,
        T=None,
        use_conf=use_conf,
        normalize=normalize,
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        conf_thres=conf_thres,
        max_interp_gap=max_interp_gap,
    )

    ds = WindowTensorDataset(X, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# ----------------------------
# Dataset
# ----------------------------

class WindowTensorDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, T, F) or (N, T, K, C)
        y: (N,)
        """
        if X.ndim == 4:
            N, T, K, C = X.shape
            X = X.reshape(N, T, K * C)
        assert X.ndim == 3, f"Expected X to be 3D (N,T,F). Got {X.shape}"

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
