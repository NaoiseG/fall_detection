# build_windows.py
import numpy as np
from collections import defaultdict
from typing import Tuple, Optional


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


if __name__ == "__main__":
    X, y, T = make_window_tensors(
        "../../Datasets/UPFall_keypoints/outputs_npz/Subject1/Activity8/Trial2/Subject1Activity8Trial2Camera1/keypoints.npz",
        T=None,
        use_conf=True
    )

    print("Window tensor shape:", X.shape)
    print("Frames per window:", T)
    print("Label distribution:", dict(zip(*np.unique(y, return_counts=True))))
    print("Label datatype = ", y.dtype)   
    print(np.unique(y))
