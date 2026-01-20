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
    label_mode: str = "majority", #center or majority
    binary_any_fall: bool = False,
    fall_ids_0based: Optional[list[int]] = None,
    min_valid_frac: float = 0.3,
    add_mask_channel: bool = True,
    drop_ambig_share: float = 0.0,
    drop_ambig_nonfall_only: bool = True,
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
                add_acc=add_acc,
                add_global=add_global,
                conf_thres=conf_thres,
                max_interp_gap=max_interp_gap,
                stride=stride,
                label_mode=label_mode,
                binary_any_fall=binary_any_fall,
                fall_ids_0based=fall_ids_0based,
                min_valid_frac=min_valid_frac,
                add_mask_channel=add_mask_channel,
                drop_ambig_share=drop_ambig_share,
                drop_ambig_nonfall_only=drop_ambig_nonfall_only,
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
                min_valid_frac=min_valid_frac,
                add_mask_channel=add_mask_channel,
                drop_ambig_share=drop_ambig_share,
                drop_ambig_nonfall_only=drop_ambig_nonfall_only,
            )

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        raise RuntimeError("No NPZs found / no windows loaded.")

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0), int(T_used)

#For sliding windows
def _make_sliding_windows(
    Xf: np.ndarray,          # (N, K, C)
    labels: np.ndarray,      # (N,)
    conf: np.ndarray,        # (N, K)  (needed for validity)
    T: int,
    stride: int,
    conf_thres: float,
    min_valid_frac: float,
    label_mode: str,         # "center" or "majority"
    binary_any_fall: bool,
    fall_ids_0based: Optional[set[int]],
    add_mask_channel: bool,
    drop_ambig_share: float,
    drop_ambig_nonfall_only: bool,
    layout: dict
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns:
      X_windows: (W, T, K, C(+1 if mask))
      y_windows: (W,)
      T_used: int (same T)
    """
    N, K, C = Xf.shape
    stride = max(1, int(stride))

    # Frame validity
    frac_valid = (conf > conf_thres).mean(axis=1)  # (N,)
    frame_valid = frac_valid >= float(min_valid_frac)  # (N,)

    # number of windows: include last partial window
    starts = list(range(0, max(1, N), stride))
    X_windows = []
    y_windows = []

    for s in starts:
        e = s + T
        seq = Xf[s:e]                      # (L,K,C)
        labs = labels[s:e]                 # (L,)
        valid = frame_valid[s:e]           # (L,)

        L = seq.shape[0]
        if L < T:
            # pad by repeating last frame (stable), but mask will mark padded steps
            pad = np.repeat(seq[-1:, :, :], repeats=(T - L), axis=0) if L > 0 else np.zeros((T, K, C), np.float32)

            if layout.get("conf_idx") is not None:
                pad[:, :, layout["conf_idx"]] = 0.0
            if layout.get("vel_slice") is not None:
                pad[:, :, layout["vel_slice"]] = 0.0
            if layout.get("acc_slice") is not None:
                pad[:, :, layout["acc_slice"]] = 0.0

            seq = np.concatenate([seq, pad], axis=0)
            labs = np.concatenate([labs, np.full((T - L,), labs[-1] if L > 0 else 0, dtype=labs.dtype)], axis=0)
            valid = np.concatenate([valid, np.zeros((T - L,), dtype=bool)], axis=0)
            L = T


        # Build mask: 1 only where frame_valid is True (and non-padded)
        mask_t = valid.astype(np.float32)  # (T,)

        # Optional: drop ambiguous windows (train-time only).
        # Ambiguity is measured on *valid* frames only. If drop_ambig_nonfall_only is True,
        # we only drop ambiguous windows that contain no fall frames (fall_ids_0based must be provided).
        if drop_ambig_share and drop_ambig_share > 0.0:
            labs_valid = labs[valid]
            if labs_valid.size > 0:
                vals_v, counts_v = np.unique(labs_valid, return_counts=True)
                top_share = float(counts_v.max()) / float(labs_valid.size)
                if top_share < float(drop_ambig_share):
                    if bool(drop_ambig_nonfall_only) and (fall_ids_0based is not None):
                        has_any_fall = any(int(v) in fall_ids_0based for v in labs_valid)
                        if not has_any_fall:
                            continue
                    else:
                        continue

        # HARD MASK: zero all features on invalid frames so padding/low-valid frames can't leak pose.
        seq[~valid] = 0.0

        # Label assignment
        if binary_any_fall:
            assert fall_ids_0based is not None
            # only consider valid frames
            any_fall = np.any([(int(l) in fall_ids_0based) for l, v in zip(labs, valid) if v])
            y = 1 if any_fall else 0
        else:
            if label_mode == "center":
                # pick center valid frame if possible, else nearest valid, else center raw
                c = T // 2
                if valid[c]:
                    y = int(labs[c])
                else:
                    idxs = np.where(valid)[0]
                    y = int(labs[idxs[len(idxs)//2]]) if idxs.size > 0 else int(labs[c])
            else:
                # majority vote over valid frames only
                labs_valid = labs[valid]
                if labs_valid.size == 0:
                    y = int(labs[T // 2])
                else:
                    vals, counts = np.unique(labs_valid, return_counts=True)
                    y = int(vals[np.argmax(counts)])

        if add_mask_channel:
            # broadcast mask to (T,K,1)
            m = mask_t[:, None, None].repeat(K, axis=1)
            seq = np.concatenate([seq, m], axis=-1)  # (T,K,C+1)

        X_windows.append(seq)
        y_windows.append(y)

        # Stop once we can no longer start a new meaningful window
        if s + stride >= N and s >= N - 1:
            break

    return np.stack(X_windows), np.array(y_windows, dtype=np.int64), int(T)

def _global_features(xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """
    xy: (N,K,2), conf: (N,K)
    Returns g: (N, G) where G=4:
      com_x, com_y, com_v, aspect
    com_v = ||com[t]-com[t-1]||
    aspect = bbox_h / (bbox_w + eps) over valid joints
    """
    N, K, _ = xy.shape
    g = np.zeros((N, 4), dtype=np.float32)
    eps = 1e-6

    for t in range(N):
        valid = conf[t] > 0.0
        if not np.any(valid):
            continue
        pts = xy[t, valid]  # (M,2)

        com = pts.mean(axis=0)  # (2,)
        g[t, 0:2] = com

        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        w = float(mx[0] - mn[0])
        h = float(mx[1] - mn[1])
        g[t, 3] = h / (w + eps)

    # COM speed
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
    fall_ids_0based: Optional[list[int]] = None,
    min_valid_frac: float = 0.3,
    add_mask_channel: bool = True,
    drop_ambig_share: float = 0.0,
    drop_ambig_nonfall_only: bool = True,
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
    labels = data["frame_labels"].astype(np.int64)  # (N,)

    # (1) Fill missing keypoints + keep a clean confidence mask
    xy_filled, conf_filled = _fill_and_mask_kpts(kxy, kconf, conf_thres=conf_thres, max_interp_gap=max_interp_gap)

    # (2) normalise
    xy_used = _normalize_xy(xy_filled, conf_filled) if normalize else xy_filled.astype(np.float32, copy=False)

    # (3) velocity
    has_vel = bool(add_vel)
    if has_vel:
        vel = _add_velocity_channels(xy_used)
    
    #Acceleration
    has_acc = bool(add_acc)
    if has_acc:
        if not has_vel:
            raise ValueError("add_acc=True requires add_vel=True")
        acc = _add_acceleration_channels(vel)

    # Build Xf
    parts = [xy_used]  # always (N,K,2)

    if use_conf:
        parts.append(conf_filled[..., None])  # (N,K,1)

    if has_vel:
        parts.append(vel)  # (N,K,2)

    if has_acc:
        parts.append(acc)  # (N,K,2)

    if add_global:
        g = _global_features(xy_used, conf_filled)  # (N,4)
        gk = np.repeat(g[:, None, :], repeats=xy_used.shape[1], axis=1)  # (N,K,4)
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
        global_slice = slice(idx, idx + 4)  # because _global_features returns 4
        idx += 4

    mask_idx = idx if add_mask_channel else None  # mask is appended later in sliding windows

    layout = {
        "conf_idx": conf_idx,
        "vel_slice": vel_slice,
        "acc_slice": acc_slice,
        "global_slice": global_slice,
        "mask_idx": mask_idx,
    }

    # Now build sliding windows (A/B/C)
    if T is None:
        T = 64  # choose default to match the CNN+LSTM-style pipeline

    X_windows, y_windows, T_used = _make_sliding_windows(
        Xf=Xf,
        labels=labels.astype(np.int64),
        conf=conf_filled,
        T=int(T),
        stride=int(stride),                # you need to add stride to function signature
        conf_thres=float(conf_thres),
        min_valid_frac=float(min_valid_frac),
        label_mode=str(label_mode),
        binary_any_fall=bool(binary_any_fall),
        fall_ids_0based=set(fall_ids_0based) if fall_ids_0based is not None else None,
        add_mask_channel=bool(add_mask_channel),
        drop_ambig_share=float(drop_ambig_share),
        drop_ambig_nonfall_only=bool(drop_ambig_nonfall_only),
        layout=layout
    )

    return X_windows, y_windows, T_used

    


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
    add_acc: bool = True,
    add_global: bool = True,
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
        add_acc=add_acc,
        add_global=add_global,
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
