#!/usr/bin/env python3
"""
Prepare a MotionBERT-compatible action-recognition .pkl from UP-Fall keypoints .npz files.

Key idea:
- Each unique window_id inside each keypoints.npz becomes ONE training sample (no sliding windows here).

Output format matches MotionBERT's lib/data/dataset_action.py expectations:
{
  "split": {"xsub_train": [...], "xsub_val": [...]},
  "annotations": [
     {"frame_dir": str, "total_frames": int, "img_shape": (H,W),
      "keypoint": np.ndarray (M,T,17,2), "keypoint_score": np.ndarray (M,T,17), "label": int},
     ...
  ]
}
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None


# ----------------------------
# NPZ discovery (mirrors your dataset.py)
# ----------------------------

def find_keypoints_npzs_subjects(output_root: Path, camera: int = 1, subjects=range(1, 6)) -> List[str]:
    """
    Matches:
      Subject{s}/Activity*/Trial*/Subject{s}Activity*Trial*Camera{camera}/keypoints.npz
    """
    npzs: List[str] = []
    for s in subjects:
        subj_root = output_root / f"Subject{s}"
        if not subj_root.exists():
            continue

        pat = subj_root / "Activity*" / "Trial*" / f"Subject{s}Activity*Trial*Camera{camera}" / "keypoints.npz"
        npzs.extend(glob.glob(str(pat), recursive=True))

    return sorted(npzs)


# ----------------------------
# CLI parsing helpers
# ----------------------------

def parse_range_expr(expr: str) -> List[int]:
    """
    Supports:
      "1-12"
      "5"
      "1-4,7,9-10"  (extra-friendly)
    Mirrors the style of your train_models.py (range strings), but allows commas too.
    """
    expr = expr.strip()
    if not expr:
        return []
    out: List[int] = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if b < a:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# ----------------------------
# Path metadata
# ----------------------------

_META_RE = re.compile(
    r"Subject(?P<S>\d+)[/\\\\]Activity(?P<A>\d+)[/\\\\]Trial(?P<T>\d+)[/\\\\]"
    r"Subject(?P=S)Activity(?P=A)Trial(?P=T)Camera(?P<C>\d+)"
)

def extract_meta(npz_path: Union[str, Path]) -> Tuple[int, int, int, int]:
    """
    Returns (S, A, T, C) from the UP-Fall directory pattern.
    Falls back to best-effort regex searches if the full pattern doesn't match.
    """
    p = str(npz_path)
    m = _META_RE.search(p)
    if m:
        return int(m["S"]), int(m["A"]), int(m["T"]), int(m["C"])

    def find_first(pattern: str, default: int = -1) -> int:
        mm = re.search(pattern, p)
        return int(mm.group(1)) if mm else default

    S = find_first(r"Subject(\d+)")
    A = find_first(r"Activity(\d+)")
    T = find_first(r"Trial(\d+)")
    C = find_first(r"Camera(\d+)")
    return S, A, T, C


# ----------------------------
# Label mapping
# ----------------------------

_INT_RE = re.compile(r"^-?\d+$")

def _to_py(x):
    try:
        return x.item()
    except Exception:
        return x

def _is_int_like(x) -> bool:
    x = _to_py(x)
    if isinstance(x, (int, np.integer)):
        return True
    if isinstance(x, (float, np.floating)):
        return float(x).is_integer()
    if isinstance(x, str):
        return _INT_RE.match(x.strip()) is not None
    return False

def build_label_map(all_labels: Sequence) -> Tuple[Dict[str, int], Dict]:
    """
    Builds mapping -> 0..K-1, robust to string or int labels.

    If labels are int-like: decide 0-based vs 1-based by min value (per your spec).
    If labels are strings: stable mapping from sorted unique -> 0..K-1.
    """
    py_labels = [_to_py(x) for x in all_labels]
    if len(py_labels) == 0:
        raise ValueError("No labels found while building label map.")

    all_int_like = all(_is_int_like(x) for x in py_labels)

    if all_int_like:
        ints = [int(float(_to_py(x))) for x in py_labels]
        min_v = int(np.min(ints))
        offset = 0 if min_v == 0 else 1
        mapped = [i - offset for i in ints]
        uniq_raw = sorted(set(ints))
        mapping = {str(v): int(v - offset) for v in uniq_raw}
        meta = {
            "type": "int",
            "offset": offset,
            "mapping": mapping,
            "num_classes_observed": len(set(mapped)),
            "min_raw": int(min_v),
            "max_raw": int(np.max(ints)),
        }
        return mapping, meta

    strs = [str(_to_py(x)).strip() for x in py_labels]
    uniq = sorted(set(strs))
    mapping = {lab: i for i, lab in enumerate(uniq)}
    meta = {
        "type": "str",
        "mapping": mapping,
        "classes_sorted": uniq,
        "num_classes_observed": len(uniq),
    }
    return mapping, meta


def window_label_from_frames(
    frame_labels: Sequence,
    label_map: Dict[str, int],
    label_mode: str = "majority",
) -> int:
    if len(frame_labels) == 0:
        raise ValueError("Empty window labels.")

    if label_mode == "center":
        mid = len(frame_labels) // 2
        raw = _to_py(frame_labels[mid])
        key = str(raw).strip()
        return int(label_map[key])

    ids = [int(label_map[str(_to_py(x)).strip()]) for x in frame_labels]
    c = Counter(ids)
    top = max(c.values())
    winners = [k for k, v in c.items() if v == top]
    return int(min(winners))  # deterministic tie-break


# ----------------------------
# Image shape helper
# ----------------------------

def get_img_shape_from_frame(frame_path: Union[str, Path]) -> Tuple[int, int]:
    if Image is None:
        return (0, 0)
    p = Path(str(frame_path))
    if not p.exists():
        return (0, 0)
    try:
        with Image.open(p) as im:
            w, h = im.size
        return (int(h), int(w))
    except Exception:
        return (0, 0)

def infer_img_shape_from_kpts(kpts_xy: np.ndarray) -> Tuple[int, int]:
    """
    Fallback when frame images aren't available.
    Infers (H,W) from max keypoint coordinates. Assumes kpts_xy are pixel coords.
    """
    # kpts_xy: (T,1,17,2) or (T,17,2)
    arr = kpts_xy
    if arr.ndim == 4:
        arr = arr[:, 0]  # (T,17,2)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    max_x = float(np.max(arr[..., 0]))
    max_y = float(np.max(arr[..., 1]))
    W = max(1, int(np.ceil(max_x + 1)))
    H = max(1, int(np.ceil(max_y + 1)))
    return (H, W)



# ----------------------------
# Main conversion
# ----------------------------

def build_dataset(
    npz_paths_train: Sequence[str],
    npz_paths_val: Sequence[str],
    out_pkl: Path,
    out_label_map: Path,
    label_mode: str = "majority",
    bad_policy: str = "impute",
    drop_empty_windows: bool = True,
) -> None:
    # First pass: gather labels across selected NPZs to build stable label_map
    all_npzs = list(dict.fromkeys(list(npz_paths_train) + list(npz_paths_val)))  # preserve order, unique
    all_labels: List = []

    print(f"Found {len(npz_paths_train)} train videos, {len(npz_paths_val)} val videos ({len(all_npzs)} total).")
    for p in all_npzs:
        data = np.load(p, allow_pickle=True)
        if "frame_labels" not in data:
            raise KeyError(f"{p} missing 'frame_labels'")
        all_labels.extend([_to_py(x) for x in data["frame_labels"]])

    label_map, label_map_meta = build_label_map(all_labels)

    # Save label map JSON now (includes metadata)
    out_label_map.parent.mkdir(parents=True, exist_ok=True)
    label_map_json = {"label_mode": label_mode, "label_map": label_map, "meta": label_map_meta}
    with out_label_map.open("w", encoding="utf-8") as f:
        json.dump(label_map_json, f, indent=2)

    # Second pass: build annotations and split lists
    annotations: List[Dict] = []
    split = {"xsub_train": [], "xsub_val": []}

    label_dist = {"xsub_train": Counter(), "xsub_val": Counter()}
    stats = {"xsub_train": Counter(), "xsub_val": Counter()}
    win_counts_per_video = []

    # Cache per-video img_shape (all frames same resolution)
    img_shape_cache: Dict[str, Tuple[int, int]] = {}

    def process_npz(npz_path: str, split_key: str):
        nonlocal annotations

        data = np.load(npz_path, allow_pickle=True)
        kpts_xy = data["kpts_xy"]          # (T,1,17,2)
        kpts_conf = data["kpts_conf"]      # (T,1,17)
        frame_paths = data["frame_paths"]  # (T,)
        frame_labels = data["frame_labels"]# (T,)
        window_ids = data["window_ids"]    # (T,)

        if kpts_xy.ndim != 4 or kpts_xy.shape[2] != 17 or kpts_xy.shape[3] != 2:
            raise ValueError(f"{npz_path}: expected kpts_xy (T,1,17,2), got {kpts_xy.shape}")
        if kpts_conf.ndim != 3 or kpts_conf.shape[2] != 17:
            raise ValueError(f"{npz_path}: expected kpts_conf (T,1,17), got {kpts_conf.shape}")

        if npz_path not in img_shape_cache:
            shape = (0, 0)
            for fp in frame_paths[:10]:
                shape = get_img_shape_from_frame(fp)
                if shape != (0, 0):
                    break
            if shape == (0, 0):
                shape = infer_img_shape_from_kpts(kpts_xy)
            img_shape_cache[npz_path] = shape

        img_shape = img_shape_cache[npz_path]
        S, A, Ttrial, C = extract_meta(npz_path)

        uniq_wids = np.unique(window_ids)
        win_counts_per_video.append((npz_path, int(len(uniq_wids))))

        for wid in uniq_wids:
            idx = np.where(window_ids == wid)[0]
            if idx.size == 0:
                continue
            idx = np.sort(idx)

            frame_dir = f"Subject{S}_Activity{A}_Trial{Ttrial}_Cam{C}_win{int(wid)}"

            kxy = kpts_xy[idx, 0].astype(np.float32)      # (Tw,17,2)
            ksc = kpts_conf[idx, 0].astype(np.float32)    # (Tw,17)

            # --- Robust cleanup ---
            # Some pose extractors write NaN/Inf for missing joints or missing detections.
            nonfinite = (not np.isfinite(kxy).all()) or (not np.isfinite(ksc).all())
            if nonfinite:
                if bad_policy == "drop":
                    stats[split_key]["dropped_nonfinite"] += 1
                    continue
                stats[split_key]["imputed_nonfinite"] += 1
                kxy = np.nan_to_num(kxy, nan=0.0, posinf=0.0, neginf=0.0)
                ksc = np.nan_to_num(ksc, nan=0.0, posinf=0.0, neginf=0.0)

            # Confidence should be within [0,1]. Clip if necessary.
            if ((ksc < 0).any() or (ksc > 1).any()):
                stats[split_key]["clipped_score"] += 1
                ksc = np.clip(ksc, 0.0, 1.0)

            # When confidence is zero, coordinates are not meaningful. Force them to 0.
            kxy[ksc <= 0] = 0.0

            # Drop windows that are effectively empty (no confident joints) or degenerate (zero spatial extent).
            if drop_empty_windows:
                if np.all(ksc <= 0):
                    stats[split_key]["dropped_empty"] += 1
                    continue
                # ptp = max-min. Degenerate poses can cause divide-by-zero in MotionBERT's 2D normalization.
                if (np.ptp(kxy[..., 0]) < 1e-6) and (np.ptp(kxy[..., 1]) < 1e-6):
                    stats[split_key]["dropped_degenerate"] += 1
                    continue

            keypoint = kxy[None, ...]          # (1,Tw,17,2)
            keypoint_score = ksc[None, ...]    # (1,Tw,17)

            labs_win = frame_labels[idx]
            label_id = window_label_from_frames(labs_win, label_map, label_mode=label_mode)

            annotations.append({
                "frame_dir": frame_dir,
                "total_frames": int(idx.size),
                "img_shape": tuple(int(x) for x in img_shape),
                "keypoint": keypoint,
                "keypoint_score": keypoint_score,
                "label": int(label_id),
            })

            split[split_key].append(frame_dir)
            label_dist[split_key][int(label_id)] += 1

    for p in npz_paths_train:
        process_npz(p, "xsub_train")
    for p in npz_paths_val:
        process_npz(p, "xsub_val")

    dataset = {"split": split, "annotations": annotations}

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n--- Sanity checks ---")
    print(f"Total samples (windows): {len(annotations)}")
    print(f"Train windows: {len(split['xsub_train'])} | Val windows: {len(split['xsub_val'])}")
    print("Per-video window counts (first 10):")
    for p, n in win_counts_per_video[:10]:
        print(f"  {n:4d} windows | {p}")

    def fmt_dist(c: Counter) -> str:
        items = sorted(c.items())
        return ", ".join([f"{k}:{v}" for k, v in items])

    print("\nLabel distribution (0-based IDs):")
    print("  Train:", fmt_dist(label_dist["xsub_train"]))
    print("  Val:  ", fmt_dist(label_dist["xsub_val"]))

    print("\nBad-window handling stats:")
    for sk in ("xsub_train", "xsub_val"):
        if len(stats[sk]) == 0:
            print(f"  {sk}: none")
            continue
        items = ', '.join([f"{k}:{v}" for k, v in sorted(stats[sk].items())])
        print(f"  {sk}: {items}")

    if annotations:
        ex = annotations[0]
        print("\nExample shapes:")
        print("  keypoint:", tuple(ex["keypoint"].shape), "(M,T,17,2)")
        print("  keypoint_score:", tuple(ex["keypoint_score"].shape), "(M,T,17)")
        print("\nNOTE: MotionBERT will internally convert COCO-17 -> H36M-17 and append confidence as a 3rd channel.")

    # Final check: ensure no non-finite numbers made it into the saved dataset
    nonfinite_count = 0
    for ann in annotations:
        if (not np.isfinite(ann['keypoint']).all()) or (not np.isfinite(ann['keypoint_score']).all()):
            nonfinite_count += 1
            if nonfinite_count <= 5:
                print('Non-finite sample:', ann['frame_dir'])
    if nonfinite_count:
        print(f"WARNING: {nonfinite_count} samples still contain NaN/Inf. Training will likely produce NaNs.")
    else:
        print('All saved keypoints and scores are finite.')


def main():
    ap = argparse.ArgumentParser("prepare_motionbert_dataset.py")
    ap.add_argument("--data-root", type=str, required=True, help="Root folder of outputs_npz, e.g. ../../Datasets/UPFall_keypoints/outputs_npz/")
    ap.add_argument("--camera", type=int, required=True, help="Camera index, e.g. 1")
    ap.add_argument("--train-subjects", type=str, required=True, help="Train subject range like '1-12' (or '1-4,7,9-10')")
    ap.add_argument("--val-subjects", type=str, required=True, help="Val subject range like '13-17' (or '5-6')")
    ap.add_argument("--out-pkl", type=str, required=True, help="Output .pkl path")
    ap.add_argument("--out-label-map", type=str, required=True, help="Output label_map.json path")
    ap.add_argument("--label-mode", type=str, default="majority", choices=["majority", "center"], help="Window label rule")
    ap.add_argument("--bad-policy", type=str, default="impute", choices=["impute", "drop"],
                    help="What to do when a window contains NaN/Inf: impute->replace with 0, drop->skip window")
    ap.add_argument("--keep-empty-windows", action="store_true",
                    help="Do not drop windows with no confident joints or degenerate spatial extent (not recommended)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    train_subjects = parse_range_expr(args.train_subjects)
    val_subjects = parse_range_expr(args.val_subjects)

    if not train_subjects:
        raise SystemExit("No train subjects parsed.")
    if not val_subjects:
        raise SystemExit("No val subjects parsed.")

    npz_train = find_keypoints_npzs_subjects(data_root, camera=args.camera, subjects=train_subjects)
    npz_val = find_keypoints_npzs_subjects(data_root, camera=args.camera, subjects=val_subjects)

    build_dataset(
        npz_paths_train=npz_train,
        npz_paths_val=npz_val,
        out_pkl=Path(args.out_pkl),
        out_label_map=Path(args.out_label_map),
        label_mode=args.label_mode,
        bad_policy=args.bad_policy,
        drop_empty_windows=not args.keep_empty_windows,
    )


if __name__ == "__main__":
    main()
