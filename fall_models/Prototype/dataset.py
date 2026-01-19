from __future__ import annotations

from typing import Tuple, Optional
from pathlib import Path

from collections import defaultdict
import glob


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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

def load_windows_from_npzs(npz_paths, T=None, use_conf: bool = True):
    """
    Loads multiple trial NPZs, converts each to (W, T, K, C) windows,
    then concatenates across trials. Ensures the same T is used for all files.
    """
    X_all, y_all = [], []
    T_used = T

    for i, p in enumerate(npz_paths):
        if i == 0 and T_used is None:
            X, y, T_used = make_window_tensors(p, T=None, use_conf=use_conf)
        else:
            X, y, _ = make_window_tensors(p, T=T_used, use_conf=use_conf)

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        raise RuntimeError("No NPZs found / no windows loaded.")

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0), T_used

def make_window_tensors(
    npz_path: str,
    T: Optional[int] = None,
    use_conf: bool = True,
    person_idx: int = 0,
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
    window_ids = data["window_ids"]             # (N,)
    labels = data["frame_labels"]               # (N,)
    labels = labels.astype(np.int64)

    if use_conf:
        Xf = np.concatenate([kxy, kconf[..., None]], axis=-1)  # (N, K, 3)
    else:
        Xf = kxy  # (N, K, 2)
    
    # Prevent NaNs
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

    X_windows = []
    y_windows = []

    for wid in sorted(frames_by_window.keys()):
        idxs = frames_by_window[wid]
        seq = Xf[idxs]

        # majority vote label
        labs = labels[idxs]
        vals, counts = np.unique(labs, return_counts=True)
        y = vals[np.argmax(counts)]

        # pad or trim
        if len(seq) >= T:
            seq = seq[:T]
        else:
            pad = np.zeros((T - len(seq), seq.shape[1], seq.shape[2]), dtype=seq.dtype)
            seq = np.concatenate([seq, pad], axis=0)

        X_windows.append(seq)
        y_windows.append(y)

    return np.stack(X_windows), np.array(y_windows), T

def make_loader(
    subjects: list[str],
    camera: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    use_conf: bool = True,
):
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints/outputs_npz")
    npz_paths = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=camera, subjects=subjects)
    if not npz_paths:
        raise RuntimeError("No NPZs found. Check OUTPUT_ROOT, camera, and subjects.")

    X, y_tags, _ = load_windows_from_npzs(npz_paths, T=None, use_conf=use_conf)

    # If y_tags are already integers, keep as-is.
    # If they are strings, map them here.
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
    
    