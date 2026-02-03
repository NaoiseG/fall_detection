from __future__ import annotations

from typing import Tuple, Optional, Iterable, Dict, Any
from pathlib import Path
import glob

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
NEW_LABEL_NAMES_1_11 = ["Fall", "Class6", "Class7", "Class8", "Class9", "Class10", "Class11"]
NEW_LABEL_NAMES_0_10 = ["Fall", "Class5", "Class6", "Class7", "Class8", "Class9", "Class10"]

# Filled at runtime after convention detection (for easy introspection/debugging)
FALL_MERGE_SET: set[int] = set()
NEW_LABEL_NAMES: list[str] = []


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

def load_windows_from_npzs(
    npz_paths,
    T: Optional[int] = None,
    use_conf: bool = True,
    normalize: bool = True,
    add_vel: bool = True,
    add_acc: bool = True,
    add_global: bool = True,
    conf_thres: float = 0.2,
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
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Loads multiple trial NPZs, converts each to (W, T, K, C) windows,
    then concatenates across trials. Ensures the same T is used for all files.

    IMPORTANT:
      Returned y is in the merged 7-class space (0..6), not the raw 11-class IDs.
    """
    X_all, y_all = [], []
    T_used = T

    conv = label_convention
    if conv is None:
        conv, stats = detect_label_convention_from_npzs(npz_paths)
        print(f"[labels] Auto-detected convention={conv} from NPZs (min={stats['min_label']}, max={stats['max_label']}).")
    else:
        if conv not in {"1-11", "0-10"}:
            raise ValueError("label_convention must be '1-11' or '0-10'.")

    global FALL_MERGE_SET, NEW_LABEL_NAMES
    FALL_MERGE_SET = get_fall_merge_set(conv)
    NEW_LABEL_NAMES = get_new_label_names(conv)

    for i, p in enumerate(npz_paths):
        if i == 0 and T_used is None:
            X, y, T_used = make_window_tensors(
                p,
                T=None,
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
                label_convention=conv,
            )
        else:
            X, y, _ = make_window_tensors(
                p,
                T=T_used,
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
                label_convention=conv,
            )

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        raise RuntimeError("No NPZs found / no windows loaded.")

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0), int(T_used)


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
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns:
      X_windows: (W, T, K, C(+1 if mask))
      y_windows: (W,) in merged 7-class space (0..6)
      T_used: int
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

    for s in starts:
        e = s + T
        seq = Xf[s:e]                      # (L,K,C)
        labs_raw = labels_raw[s:e]         # (L,)
        valid = frame_valid[s:e]           # (L,)

        L = seq.shape[0]
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

        # Optional: drop ambiguous windows (train-time only).
        # Ambiguity is measured on *valid* frames, in merged label space.
        if drop_ambig_share and drop_ambig_share > 0.0:
            labs_valid = labs[valid]
            if labs_valid.size > 0:
                _vals, counts_v = np.unique(labs_valid, return_counts=True)
                top_share = float(counts_v.max()) / float(labs_valid.size)
                if top_share < float(drop_ambig_share):
                    if bool(drop_ambig_nonfall_only):
                        has_any_fall = bool(np.any(labs_valid == 0))
                        if not has_any_fall:
                            continue
                    else:
                        continue

        # HARD MASK: zero all features on invalid frames so padding/low-valid frames can't leak pose.
        seq[~valid] = 0.0

        # Label assignment (merged space)
        c = T // 2
        if valid[c]:
            center_y = int(labs[c])
        else:
            idxs = np.where(valid)[0]
            center_y = int(labs[idxs[len(idxs)//2]]) if idxs.size > 0 else int(labs[c])

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

        if s + stride >= N and s >= N - 1:
            break

    return np.stack(X_windows), np.array(y_windows, dtype=np.int64), int(T)


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


def make_window_tensors(
    npz_path: str,
    T: Optional[int] = None,
    use_conf: bool = True,
    person_idx: int = 0,
    normalize: bool = True,
    add_vel: bool = True,
    add_acc: bool = True,
    add_global: bool = True,
    conf_thres: float = 0.2,
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
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Converts frame-level pose data into window-level tensors.

    Returns:
        X: (W, T, K, C)
        y: (W,) merged 7-class labels (0..6)
        T: frames per window used
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

    xy_filled, conf_filled = _fill_and_mask_kpts(kxy, kconf, conf_thres=conf_thres, max_interp_gap=max_interp_gap)

    xy_used = _normalize_xy(xy_filled, conf_filled) if normalize else xy_filled.astype(np.float32, copy=False)

    has_vel = bool(add_vel)
    if has_vel:
        vel = _add_velocity_channels(xy_used)

    has_acc = bool(add_acc)
    if has_acc:
        if not has_vel:
            raise ValueError("add_acc=True requires add_vel=True")
        acc = _add_acceleration_channels(vel)

    parts = [xy_used]

    if use_conf:
        parts.append(conf_filled[..., None])

    if has_vel:
        parts.append(vel)

    if has_acc:
        parts.append(acc)

    if add_global:
        g = _global_features(xy_used, conf_filled)
        gk = np.repeat(g[:, None, :], repeats=xy_used.shape[1], axis=1)
        parts.append(gk)

    Xf = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)

    idx = 2
    conf_idx = None
    if use_conf:
        conf_idx = idx
        idx += 1

    vel_slice = None
    if has_vel:
        vel_slice = slice(idx, idx + 2)
        idx += 2

    acc_slice = None
    if has_acc:
        acc_slice = slice(idx, idx + 2)
        idx += 2

    global_slice = None
    if add_global:
        global_slice = slice(idx, idx + 4)
        idx += 4

    mask_idx = idx if add_mask_channel else None

    layout = {
        "conf_idx": conf_idx,
        "vel_slice": vel_slice,
        "acc_slice": acc_slice,
        "global_slice": global_slice,
        "mask_idx": mask_idx,
    }

    if T is None:
        T = 64

    X_windows, y_windows, T_used = _make_sliding_windows(
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
    )

    return X_windows, y_windows, T_used


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
    conf_thres: float = 0.2,
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
