#!/usr/bin/env python3
r"""
Prepare a MotionBERT-compatible action-recognition .pkl from UP-Fall keypoints .npz files.

Changes vs. previous version:
- Build true fixed-length temporal windows directly from the per-frame arrays in each .npz
  (default win_len=64) using a sliding window (default win_step=16, so overlap=48).
- Ignore window_ids entirely for sampling (use the full time axis).
- Handle missing joints by temporal interpolation instead of forcing coords to (0,0).

Output format matches MotionBERT's lib/data/dataset_action.py expectations:
{
  "split": {"xsub_train": [...], "xsub_val": [...]},
  "annotations": [
     {"frame_dir": str,
      "total_frames": int,
      "img_shape": (H,W),
      "keypoint": np.ndarray (M,T,17,2),
      "keypoint_score": np.ndarray (M,T,17),
      "label": int},
     ...
  ]
}

Notes:
- M is always 1 (single-person).
- T is always exactly win_len for every sample.
- label_mode ("majority" or "center") is applied on the frame_labels inside each window slice.

python .\dataset_helpers\prepare_motionbert_dataset.py --label-mode center --win-step 32 --train-subjects 1-12 \
    --val-subjects 13-15 --outputs-npz-root ..\..\Datasets\UPFall_keypoints\outputs_npz\ --camera 1 2 \
    --out-pkl .\models\MotionBERT\data\action\upfall.pkl \
    --out-label-map .\models\MotionBERT\data\action\upfall_label_map.json
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

def find_keypoints_npzs_subjects(
    output_root: Path,
    cameras: Sequence[int] = (1, 2),
    subjects=range(1, 6),
) -> List[str]:
    """
    Matches:
      Subject{s}/Activity*/Trial*/Subject{s}Activity*Trial*Camera{camera}/keypoints.npz
    for each requested camera.
    """
    if isinstance(cameras, (int, np.integer)):
        cameras = [int(cameras)]
    camera_ids = sorted(set(int(c) for c in cameras))

    npzs: List[str] = []
    for s in subjects:
        subj_root = output_root / f"Subject{s}"
        if not subj_root.exists():
            continue

        for camera in camera_ids:
            pat = subj_root / "Activity*" / "Trial*" / f"Subject{s}Activity*Trial*Camera{camera}" / "keypoints.npz"
            npzs.extend(glob.glob(str(pat), recursive=True))

    return sorted(set(npzs))


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

    If labels are int-like: decide 0-based vs 1-based by min value.
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
# Optional label remapping: merge fall classes
# ----------------------------

def build_merge_fall_remap(
    num_classes: int,
    fall_class_ids: Sequence[int],
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """
    Build an old_id -> new_id remap such that all fall_class_ids map to new 0 ("Fall"),
    and all remaining classes keep their relative order and become 1..K'.

    Returns:
      remap_old_to_new: dict[int,int]
      groups_new_to_old: dict[int, list[int]] (for audit/printing)
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")

    fall_ids = sorted(set(int(x) for x in fall_class_ids))
    if not fall_ids:
        raise ValueError("fall_class_ids is empty while merge_fall is enabled.")

    if fall_ids[0] < 0 or fall_ids[-1] >= num_classes:
        raise ValueError(f"fall_class_ids {fall_ids} out of range for num_classes={num_classes}")

    fall_set = set(fall_ids)
    remaining = [i for i in range(num_classes) if i not in fall_set]

    remap_old_to_new: Dict[int, int] = {}
    groups_new_to_old: Dict[int, List[int]] = {0: fall_ids}

    for old_id in range(num_classes):
        remap_old_to_new[old_id] = 0 if old_id in fall_set else -1

    for new_id, old_id in enumerate(remaining, start=1):
        remap_old_to_new[old_id] = int(new_id)
        groups_new_to_old[int(new_id)] = [int(old_id)]

    return remap_old_to_new, groups_new_to_old


def remap_label_id(old_id: int, remap_old_to_new: Dict[int, int]) -> int:
    old_id = int(old_id)
    if old_id not in remap_old_to_new:
        raise ValueError(f"Label id {old_id} not present in remap mapping.")
    new_id = int(remap_old_to_new[old_id])
    if new_id < 0:
        raise ValueError(f"Label id {old_id} maps to invalid new id {new_id}.")
    return new_id



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
# Missing-joint interpolation
# ----------------------------

def interpolate_missing_joints_inplace(
    kxy: np.ndarray,
    ksc: np.ndarray,
    missing_conf_thres: float = 0.0,
) -> None:
    """
    Interpolate missing joints over time within ONE window.

    Missing definition (per joint, per frame):
      - joint score <= missing_conf_thres OR non-finite coordinate(s)

    For each joint j and axis (x,y):
      - If >=2 valid points: fill invalid indices by linear interpolation along time
      - If exactly 1 valid point: fill all missing with that value
      - If 0 valid points: set coords and scores to zeros (leave for downstream "drop empty window" logic)

    IMPORTANT:
      - We keep keypoint_score unchanged for interpolated points (conservative).
        Only joints with 0 valid points get their score forced to 0.
    """
    # kxy: (T,17,2), ksc: (T,17)
    T = kxy.shape[0]
    V = kxy.shape[1]
    if V != 17 or kxy.shape[2] != 2:
        raise ValueError(f"Expected kxy (T,17,2), got {kxy.shape}")
    if ksc.shape != (T, 17):
        raise ValueError(f"Expected ksc (T,17), got {ksc.shape}")

    t_idx = np.arange(T, dtype=np.float64)

    # axis-wise interpolation, but also handle joint with 0 valid (both axes) in a clean way.
    for j in range(V):
        finite_joint = np.isfinite(kxy[:, j, 0]) & np.isfinite(kxy[:, j, 1])
        valid_joint = (ksc[:, j] > missing_conf_thres) & finite_joint
        n_valid_joint = int(np.sum(valid_joint))

        if n_valid_joint == 0:
            # No reliable evidence for this joint in this window
            kxy[:, j, :] = 0.0
            ksc[:, j] = 0.0
            continue

        for a in range(2):
            valid = (ksc[:, j] > missing_conf_thres) & np.isfinite(kxy[:, j, a])
            idx = np.where(valid)[0]
            if idx.size >= 2:
                vals = kxy[idx, j, a].astype(np.float64)
                interp_all = np.interp(t_idx, idx.astype(np.float64), vals)
                invalid = ~valid
                kxy[invalid, j, a] = interp_all[invalid].astype(np.float32)
            elif idx.size == 1:
                kxy[:, j, a] = float(kxy[idx[0], j, a])
            else:
                # This should be rare if valid_joint was non-zero, but keep safe.
                kxy[:, j, a] = 0.0


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
    win_len: int = 64,
    win_step: int = 16,
    missing_conf_thres: float = 0.0,
    pad_tail: bool = False,
    merge_fall: bool = True,
    fall_class_ids: Sequence[int] = (0, 1, 2, 3, 4),
) -> None:
    # First pass: gather labels across selected NPZs to build stable label_map
    all_npzs = list(dict.fromkeys(list(npz_paths_train) + list(npz_paths_val)))  # preserve order, unique
    all_labels: List = []

    print(f"Found {len(npz_paths_train)} train videos, {len(npz_paths_val)} val videos ({len(all_npzs)} total).")
    print(f"Windowing: win_len={win_len}, win_step={win_step} (overlap={max(0, win_len - win_step)})")
    print(f"Missing-joint interpolation: missing_conf_thres={missing_conf_thres}")
    if pad_tail:
        print("Tail policy: PAD (repeat last frame) to reach full length.")
    else:
        print("Tail policy: DROP (discard shorter tail windows).")

    for p in all_npzs:
        data = np.load(p, allow_pickle=True)
        if "frame_labels" not in data:
            raise KeyError(f"{p} missing 'frame_labels'")
        all_labels.extend([_to_py(x) for x in data["frame_labels"]])

    label_map, label_map_meta = build_label_map(all_labels)

    # Build human-friendly id->name mapping (0-based IDs) for the JSON output.
    # For UP-Fall this is typically just the raw activity id as a string, but we keep it generic.
    raw_to_id = label_map_meta.get("mapping", label_map)
    num_classes = int(label_map_meta.get("num_classes_observed", len(set(raw_to_id.values()))))
    old_id_to_name: Dict[int, str] = {}
    for raw_lab, idx in raw_to_id.items():
        try:
            old_id_to_name[int(idx)] = str(raw_lab).strip()
        except Exception:
            continue
    for i in range(num_classes):
        old_id_to_name.setdefault(i, str(i))

    # Optional: merge fall classes AFTER window label computation.
    # This preserves the existing window label logic (majority/center) and only changes the final class id.
    remap_old_to_new: Dict[int, int] = {}
    new_id_to_name: Dict[str, str] = {str(i): old_id_to_name[i] for i in range(num_classes)}
    new_num_classes = int(num_classes)

    if merge_fall:
        fall_ids = sorted(set(int(x) for x in fall_class_ids))
        remap_old_to_new, _groups = build_merge_fall_remap(num_classes, fall_ids)
        new_num_classes = int(max(remap_old_to_new.values())) + 1

        print(f"Merged fall classes: old {fall_ids} -> new 0; shifted others down by {max(0, len(fall_ids) - 1)}")

        # Build merged taxonomy names: 0->"Fall", 1.. reuse old names for old ids not in fall_ids.
        new_id_to_name = {"0": "Fall"}
        fall_set = set(fall_ids)
        remaining = [i for i in range(num_classes) if i not in fall_set]
        for new_id, old_id in enumerate(remaining, start=1):
            new_id_to_name[str(new_id)] = old_id_to_name.get(int(old_id), str(old_id))

        # Record mapping in meta for auditability.
        label_map_meta = dict(label_map_meta)
        label_map_meta["merge_fall"] = {
            "enabled": True,
            "fall_class_ids": fall_ids,
            "remap_old_to_new": {str(k): int(v) for k, v in remap_old_to_new.items()},
            "old_num_classes": int(num_classes),
            "new_num_classes": int(new_num_classes),
        }
    else:
        label_map_meta = dict(label_map_meta)
        label_map_meta["merge_fall"] = {"enabled": False, "old_num_classes": int(num_classes), "new_num_classes": int(new_num_classes)}

    # Save label map JSON now (keeps the same top-level structure, but label_map is 0-based id -> class name).
    out_label_map.parent.mkdir(parents=True, exist_ok=True)
    label_map_json = {"label_mode": label_mode, "label_map": new_id_to_name, "meta": label_map_meta}
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

    def compute_num_windows(T_total: int) -> int:
        if pad_tail:
            if T_total <= 0:
                return 0
            # Starts: 0, step, 2*step, ... last start < T_total
            return int((max(0, T_total - 1)) // win_step + 1)
        # drop tail
        if T_total < win_len:
            return 0
        return int((T_total - win_len) // win_step + 1)

    def process_npz(npz_path: str, split_key: str):
        nonlocal annotations

        data = np.load(npz_path, allow_pickle=True)
        kpts_xy = data["kpts_xy"]          # (T,1,17,2)
        kpts_conf = data["kpts_conf"]      # (T,1,17)
        frame_paths = data["frame_paths"]  # (T,)
        frame_labels = data["frame_labels"]# (T,)

        # window_ids may exist but are ignored by design
        T_total = int(kpts_xy.shape[0])

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

        n_wins = compute_num_windows(T_total)
        win_counts_per_video.append((npz_path, int(n_wins), int(T_total)))

        if n_wins == 0:
            stats[split_key]["videos_with_no_full_windows"] += 1
            return

        # ------------------------------------------------------------
        # NEW windowing logic:
        # Slide windows over the *full* frame axis:
        #   starts = 0, win_step, 2*win_step, ...
        # Each window is frames [start : start + win_len].
        # By default we DROP tail windows that are shorter than win_len.
        # ------------------------------------------------------------
        for start in range(0, T_total, win_step):
            end = start + win_len
            if end > T_total:
                if not pad_tail:
                    stats[split_key]["dropped_short_tail"] += 1
                    break
                # pad by repeating the last available frame
                pad_n = end - T_total
                if pad_n >= win_len:
                    # extreme edge case where T_total==0 handled earlier; keep safe.
                    stats[split_key]["dropped_short_video"] += 1
                    break

            # Build stable unique id per window (required by your spec)
            frame_dir = f"Subject{S}_Activity{A}_Trial{Ttrial}_Cam{C}_s{start}_len{win_len}"

            # Slice raw arrays
            raw_kxy = kpts_xy[start:min(end, T_total), 0].astype(np.float32)      # (Tw,17,2)
            raw_ksc = kpts_conf[start:min(end, T_total), 0].astype(np.float32)    # (Tw,17)
            raw_labs = frame_labels[start:min(end, T_total)]

            if pad_tail and end > T_total:
                # repeat last frame/score/label to reach full length
                last_xy = raw_kxy[-1:, :, :]
                last_sc = raw_ksc[-1:, :]
                last_lb = raw_labs[-1:]
                raw_kxy = np.concatenate([raw_kxy, np.repeat(last_xy, pad_n, axis=0)], axis=0)
                raw_ksc = np.concatenate([raw_ksc, np.repeat(last_sc, pad_n, axis=0)], axis=0)
                raw_labs = np.concatenate([raw_labs, np.repeat(last_lb, pad_n, axis=0)], axis=0)

            # Guard: ensure fixed length
            if raw_kxy.shape[0] != win_len or raw_ksc.shape[0] != win_len or len(raw_labs) != win_len:
                stats[split_key]["dropped_bad_len"] += 1
                continue

            kxy = raw_kxy.copy()
            ksc = raw_ksc.copy()

            # ------------------------------------------------------------
            # Robust cleanup (preserve existing bad_policy behavior):
            # If any NaN/Inf exists in coords or scores:
            #   - bad_policy=drop: drop this window
            #   - bad_policy=impute: replace non-finite with 0 (and force score=0 where coords were non-finite)
            # Interpolation happens AFTER this conversion to a consistent form.
            # ------------------------------------------------------------
            nonfinite_xy = ~np.isfinite(kxy)
            nonfinite_sc = ~np.isfinite(ksc)
            if nonfinite_xy.any() or nonfinite_sc.any():
                if bad_policy == "drop":
                    stats[split_key]["dropped_nonfinite"] += 1
                    continue
                stats[split_key]["imputed_nonfinite"] += 1
                kxy[nonfinite_xy] = 0.0
                ksc[nonfinite_sc] = 0.0
                # If either x or y was non-finite at (t,j), treat joint as missing by forcing score=0.
                nonfinite_joint = nonfinite_xy.any(axis=2) | nonfinite_sc
                ksc[nonfinite_joint] = 0.0

            # Confidence should be within [0,1]. Clip if necessary.
            if ((ksc < 0).any() or (ksc > 1).any()):
                stats[split_key]["clipped_score"] += 1
                ksc = np.clip(ksc, 0.0, 1.0)

            # ------------------------------------------------------------
            # NEW missing joint handling:
            # Instead of forcing kxy[ksc<=0]=0, we interpolate missing joints along time.
            # This prevents injecting fake skeletons at (0,0).
            # ------------------------------------------------------------
            interpolate_missing_joints_inplace(kxy, ksc, missing_conf_thres=missing_conf_thres)

            # Drop windows that are effectively empty (no confident joints) or degenerate (zero spatial extent).
            # Evaluated AFTER interpolation is applied (per your spec).
            if drop_empty_windows:
                if np.all(ksc <= missing_conf_thres):
                    stats[split_key]["dropped_empty"] += 1
                    continue
                # ptp = max-min. Degenerate poses can cause divide-by-zero in MotionBERT's 2D normalization.
                if (np.ptp(kxy[..., 0]) < 1e-6) and (np.ptp(kxy[..., 1]) < 1e-6):
                    stats[split_key]["dropped_degenerate"] += 1
                    continue

            keypoint = kxy[None, ...].astype(np.float32)          # (1,win_len,17,2)
            keypoint_score = ksc[None, ...].astype(np.float32)    # (1,win_len,17)

            label_id = window_label_from_frames(raw_labs, label_map, label_mode=label_mode)
            if merge_fall:
                label_id = remap_label_id(label_id, remap_old_to_new)

            annotations.append({
                "frame_dir": frame_dir,
                "total_frames": int(win_len),
                "img_shape": tuple(int(x) for x in img_shape),
                "keypoint": keypoint,
                "keypoint_score": keypoint_score,
                "label": int(label_id),
            })

            split[split_key].append(frame_dir)
            label_dist[split_key][int(label_id)] += 1

            # If we're padding tails, continue; if not padding and end==T_total we still naturally finish.

            if (not pad_tail) and (end == T_total):
                # exact fit to end, subsequent starts will exceed and trigger tail-drop break anyway
                pass

            if not pad_tail and end >= T_total:
                # avoid counting dropped_short_tail multiple times when T_total aligns awkwardly
                break

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
    for p, n, ttot in win_counts_per_video[:10]:
        print(f"  {n:4d} windows from {ttot:4d} frames | {p}")

    def fmt_dist(c: Counter) -> str:
        items = sorted(c.items())
        return ", ".join([f"{k}:{v}" for k, v in items])

    print("\nLabel distribution (0-based IDs):")
    print("  Train:", fmt_dist(label_dist["xsub_train"]))
    print("  Val:  ", fmt_dist(label_dist["xsub_val"]))

    # Sanity: ensure label IDs are within the expected range after any remapping.
    bad_labels = 0
    for ann in annotations:
        lb = int(ann["label"])
        if lb < 0 or lb >= int(new_num_classes):
            bad_labels += 1
            if bad_labels <= 5:
                print("Out-of-range label sample:", ann["frame_dir"], "label=", lb)
    if bad_labels:
        print(f"WARNING: {bad_labels} samples have label IDs outside [0..{int(new_num_classes) - 1}].")
    else:
        print(f"All labels are within [0..{int(new_num_classes) - 1}] ({int(new_num_classes)} classes).")

    print("\nBad-window handling stats:")
    for sk in ("xsub_train", "xsub_val"):
        if len(stats[sk]) == 0:
            print(f"  {sk}: none")
            continue
        items = ", ".join([f"{k}:{v}" for k, v in sorted(stats[sk].items())])
        print(f"  {sk}: {items}")

    if annotations:
        ex = annotations[0]
        print("\nExample shapes:")
        print("  keypoint:", tuple(ex["keypoint"].shape), "(M,T,17,2)")
        print("  keypoint_score:", tuple(ex["keypoint_score"].shape), "(M,T,17)")
        print("\nNOTE: MotionBERT will internally convert COCO-17 -> H36M-17 and append confidence as a 3rd channel.")

    # Sanity check: ensure all samples have T == win_len
    bad_T = 0
    for ann in annotations:
        if ann["keypoint"].shape[1] != win_len or ann["keypoint_score"].shape[1] != win_len:
            bad_T += 1
            if bad_T <= 5:
                print("Bad T sample:", ann["frame_dir"], ann["keypoint"].shape, ann["keypoint_score"].shape)
    if bad_T:
        print(f"WARNING: {bad_T} samples have incorrect T (expected {win_len}).")
    else:
        print(f"All samples have T == win_len == {win_len}.")

    # Final check: ensure no non-finite numbers made it into the saved dataset
    nonfinite_count = 0
    for ann in annotations:
        if (not np.isfinite(ann["keypoint"]).all()) or (not np.isfinite(ann["keypoint_score"]).all()):
            nonfinite_count += 1
            if nonfinite_count <= 5:
                print("Non-finite sample:", ann["frame_dir"])
    if nonfinite_count:
        print(f"WARNING: {nonfinite_count} samples still contain NaN/Inf. Training will likely produce NaNs.")
    else:
        print("All saved keypoints and scores are finite.")


def main():
    ap = argparse.ArgumentParser("prepare_motionbert_dataset.py")
    ap.add_argument("--outputs-npz-root", "--data-root", dest="data_root", type=str, required=True,
                    help="Root folder of outputs_npz, e.g. ../../Datasets/UPFall_keypoints/outputs_npz/ "
                         "(--data-root is a deprecated alias).")
    ap.add_argument(
        "--camera",
        nargs="+",
        type=int,
        default=[1, 2],
        help="One or more camera indices, e.g. --camera 1 or --camera 1 2 (default: 1 2)",
    )
    ap.add_argument("--train-subjects", type=str, required=True,
                    help="Train subject range like '1-12' (or '1-4,7,9-10')")
    ap.add_argument("--val-subjects", type=str, required=True,
                    help="Val subject range like '13-17' (or '5-6')")
    ap.add_argument("--out-pkl", type=str, required=True, help="Output .pkl path")
    ap.add_argument("--out-label-map", type=str, required=True, help="Output label_map.json path")
    ap.add_argument("--label-mode", type=str, default="majority", choices=["majority", "center"],
                    help="Window label rule applied within each fixed-length window slice")
    ap.add_argument("--bad-policy", type=str, default="impute", choices=["impute", "drop"],
                    help="What to do when a window contains NaN/Inf: impute->replace with 0, drop->skip window")
    ap.add_argument("--keep-empty-windows", action="store_true",
                    help="Do not drop windows with no confident joints or degenerate spatial extent (not recommended)")

    # NEW args
    ap.add_argument("--win-len", type=int, default=64,
                    help="Temporal window length T for MotionBERT samples (default: 64)")
    ap.add_argument("--win-step", type=int, default=16,
                    help="Window step in frames (default: 16). Overlap = win_len - win_step (default overlap: 48).")
    ap.add_argument("--missing-conf-thres", type=float, default=0.0,
                    help="Treat joints as missing when confidence <= this threshold (default: 0.0)")
    ap.add_argument("--pad-tail", action="store_true",
                    help="Pad the final short tail window by repeating the last frame (default: drop tail windows)")

    # Label remapping (default enabled): merge the first N fall classes into a single "Fall" class.
    # This remap is applied AFTER the window label is computed (majority/center).
    ap.add_argument("--merge-fall", action=argparse.BooleanOptionalAction, default=True,
                    help="Merge multiple fall classes into one (default: enabled). Use --no-merge-fall to disable.")
    ap.add_argument("--fall-class-ids", type=str, default="0-4",
                    help="0-based class IDs to merge into the single Fall class, e.g. '0-4' or '0,1,2,3,4'.")

    args = ap.parse_args()

    if args.win_len <= 0:
        raise SystemExit("--win-len must be > 0")
    if args.win_step <= 0:
        raise SystemExit("--win-step must be > 0")
    if args.win_step > args.win_len:
        print("WARNING: --win-step > --win-len will produce non-overlapping windows with gaps.")

    data_root = Path(args.data_root)
    train_subjects = parse_range_expr(args.train_subjects)
    val_subjects = parse_range_expr(args.val_subjects)
    camera_ids = sorted(set(int(c) for c in args.camera))

    if not camera_ids:
        raise SystemExit("--camera must contain at least one camera index.")
    if any(c <= 0 for c in camera_ids):
        raise SystemExit(f"--camera values must be positive integers. Got: {camera_ids}")

    print(f"Using camera(s): {camera_ids}")

    fall_class_ids = parse_range_expr(args.fall_class_ids)
    if args.merge_fall and not fall_class_ids:
        raise SystemExit("--merge-fall is enabled but --fall-class-ids parsed to an empty set.")

    if not train_subjects:
        raise SystemExit("No train subjects parsed.")
    if not val_subjects:
        raise SystemExit("No val subjects parsed.")

    npz_train = find_keypoints_npzs_subjects(data_root, cameras=camera_ids, subjects=train_subjects)
    npz_val = find_keypoints_npzs_subjects(data_root, cameras=camera_ids, subjects=val_subjects)

    build_dataset(
        npz_paths_train=npz_train,
        npz_paths_val=npz_val,
        out_pkl=Path(args.out_pkl),
        out_label_map=Path(args.out_label_map),
        label_mode=args.label_mode,
        bad_policy=args.bad_policy,
        drop_empty_windows=not args.keep_empty_windows,
        win_len=args.win_len,
        win_step=args.win_step,
        missing_conf_thres=args.missing_conf_thres,
        pad_tail=args.pad_tail,
        merge_fall=args.merge_fall,
        fall_class_ids=fall_class_ids,
    )


if __name__ == "__main__":
    main()
