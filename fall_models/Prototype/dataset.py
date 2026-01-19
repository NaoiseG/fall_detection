from __future__ import annotations

from typing import Tuple, Optional
from pathlib import Path
from collections import defaultdict
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
    # Identify missing runs between valid points
    for a, b in zip(idx_valid[:-1], idx_valid[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        if gap <= max_gap:
            # interpolate from out[a] to out[b]
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

    # Treat NaNs/Infs as missing
    xy = np.nan_to_num(xy, nan=0.0, posinf=0.0, neginf=0.0)
    conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

    missing = conf < conf_thres  # (N,K)

    # Fill per joint, per axis
    for j in range(K):
        v = ~missing[:, j]
        xj = xy[:, j, 0]
        yj = xy[:, j, 1]

        xj_f = _interp_short_gaps_1d(xj, v, max_gap=max_interp_gap)
        yj_f = _interp_short_gaps_1d(yj, v, max_gap=max_interp_gap)

        xy[:, j, 0] = xj_f
        xy[:, j, 1] = yj_f

        # Force conf to 0 where missing by threshold
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
        # torso proxy: distance between mid-shoulder and mid-hip if possible
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


def _pad_seq(seq: np.ndarray, T: int, use_conf: bool, has_vel: bool) -> np.ndarray:
    """
    Pads a (L,K,C) sequence to length T.
    Strategy:
      - repeat last frame for xy (stable)
      - set conf=0 on padded frames (so model knows it's padded)
      - set vel=0 on padded frames
    """
    L, K, C = seq.shape
    if L >= T:
        return seq[:T]

    pad_len = T - L
    pad = np.repeat(seq[-1:, :, :], repeats=pad_len, axis=0).astype(seq.dtype, copy=False)

    # conf is at channel 2 when use_conf=True and features are [x,y,conf,(vx,vy)]
    if use_conf:
        pad[:, :, 2] = 0.0
    if has_vel:
        # velocities are last two channels
        pad[:, :, -2:] = 0.0

    return np.concatenate([seq, pad], axis=0)


# ----------------------------
# Window building
# ----------------------------

def load_windows_from_npzs(
    npz_paths,
    T: Optional[int] = None,
    use_conf: bool = True,
    normalize: bool = True,
    add_vel: bool = True,
    conf_thres: float = 0.2,
    max_interp_gap: int = 5,
):
    """
    Loads multiple trial NPZs, converts each to (W, T, K, C) windows,
    then concatenates across trials. Ensures the same T is used for all files.
    """
    X_all, y_all = [], []
    T_used = T

    for i, p in enumerate(npz_paths):
        if i == 0 and T_used is None:
            X, y, T_used = make_window_tensors(
                p,
                T=None,
                use_conf=use_conf,
                normalize=normalize,
                add_vel=add_vel,
                conf_thres=conf_thres,
                max_interp_gap=max_interp_gap,
            )
        else:
            X, y, _ = make_window_tensors(
                p,
                T=T_used,
                use_conf=use_conf,
                normalize=normalize,
                add_vel=add_vel,
                conf_thres=conf_thres,
                max_interp_gap=max_interp_gap,
            )

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        raise RuntimeError("No NPZs found / no windows loaded.")

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0), int(T_used)


def make_window_tensors(
    npz_path: str,
    T: Optional[int] = None,
    use_conf: bool = True,
    person_idx: int = 0,
    normalize: bool = True,
    add_vel: bool = True,
    conf_thres: float = 0.2,
    max_interp_gap: int = 5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Converts frame-level pose data into window-level tensors.

    Returns:
        X: (W, T, K, C)
        y: (W,)
        T: frames per window used
    """
    data = np.load(npz_path, allow_pickle=True)

    kxy = data["kpts_xy"][:, person_idx]       # (N, K, 2)
    kconf = data["kpts_conf"][:, person_idx]   # (N, K)
    window_ids = data["window_ids"]            # (N,)
    labels = data["frame_labels"].astype(np.int64)  # (N,)

    # (1) Fill missing keypoints + keep a clean confidence mask
    xy_filled, conf_filled = _fill_and_mask_kpts(kxy, kconf, conf_thres=conf_thres, max_interp_gap=max_interp_gap)

    # (2) Normalise xy (recommended for generalisation)
    if normalize:
        xy_used = _normalize_xy(xy_filled, conf_filled)
    else:
        xy_used = xy_filled.astype(np.float32, copy=False)

    # (3) Add velocity channels (dynamics)
    has_vel = bool(add_vel)
    if has_vel:
        vel = _add_velocity_channels(xy_used)  # (N,K,2)

    # Build per-frame feature tensor Xf: (N,K,C)
    if use_conf and has_vel:
        # [x, y, conf, vx, vy]
        Xf = np.concatenate([xy_used, conf_filled[..., None], vel], axis=-1).astype(np.float32, copy=False)
    elif use_conf and (not has_vel):
        # [x, y, conf]
        Xf = np.concatenate([xy_used, conf_filled[..., None]], axis=-1).astype(np.float32, copy=False)
    elif (not use_conf) and has_vel:
        # [x, y, vx, vy]
        Xf = np.concatenate([xy_used, vel], axis=-1).astype(np.float32, copy=False)
    else:
        # [x, y]
        Xf = xy_used.astype(np.float32, copy=False)

    # Prevent NaNs (should be none, but keep safe)
    Xf = np.nan_to_num(Xf, nan=0.0, posinf=0.0, neginf=0.0)

    # group frames by window
    frames_by_window = defaultdict(list)
    for i, wid in enumerate(window_ids):
        if wid >= 0:
            frames_by_window[int(wid)].append(i)

    lengths = [len(v) for v in frames_by_window.values()]
    if not lengths:
        raise RuntimeError("No valid windows found")

    if T is None:
        T = int(np.median(lengths))
        T = max(1, T)

    X_windows = []
    y_windows = []

    for wid in sorted(frames_by_window.keys()):
        idxs = frames_by_window[wid]
        seq = Xf[idxs]  # (L,K,C)

        # majority vote label
        labs = labels[idxs]
        vals, counts = np.unique(labs, return_counts=True)
        y = vals[np.argmax(counts)]

        # pad or trim (padding uses last pose, conf=0, vel=0)
        seq = _pad_seq(seq, T=T, use_conf=use_conf, has_vel=has_vel)

        X_windows.append(seq)
        y_windows.append(y)

    return np.stack(X_windows), np.array(y_windows), int(T)


# ----------------------------
# Loader
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
    conf_thres: float = 0.2,
    max_interp_gap: int = 5,
):
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints/outputs_npz")
    npz_paths = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=camera, subjects=subjects)
    if not npz_paths:
        raise RuntimeError("No NPZs found. Check OUTPUT_ROOT, camera, and subjects.")

    X, y_tags, _ = load_windows_from_npzs(
        npz_paths,
        T=None,
        use_conf=use_conf,
        normalize=normalize,
        add_vel=add_vel,
        conf_thres=conf_thres,
        max_interp_gap=max_interp_gap,
    )

    y = y_tags

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
            # (N, T, K, C) -> (N, T, K*C)
            N, T, K, C = X.shape
            X = X.reshape(N, T, K * C)
        assert X.ndim == 3, f"Expected X to be 3D (N,T,F). Got {X.shape}"

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
