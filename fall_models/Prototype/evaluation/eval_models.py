#!/usr/bin/env python3
"""eval_models.py

Evaluate one or more trained models on a chosen set of UP-Fall subjects, using the
same NPZ -> window loading pipeline as training (dataset.py).

Outputs (in --out-dir):
- metrics_summary.csv   : per-model/per-split summary including
    * binary sensitivity (recall) and precision macro-averaged over fall/no-fall
    * per-class multi-class precision/recall/F1/specificity/support columns keyed by class name
    * multi-class macro F1
- f1_per_class.csv      : per-model/per-split, per-class F1 (multi-class)
- misclassification_report.xlsx : per-video + per-window activity misclassifications
- report.html           : tables + plots
- plots/*.png           : quick comparison plots

Run (from project root):
python -m models.eval_models --models tcn lstm gru \
  --camera 1 2 \
  --test-subjects 1-1 \
  --fall-class-ids 9 10 11 \
  --ckpt-root models \
  --out-dir eval_outputs

Notes:
- Labels are mapped like training: original 1..N -> 0..N-1.
  So pass fall class ids in the ORIGINAL label space (1-based); this script shifts by -1.

Choosing model weights:
    - By default, the latest run folder under each model's checkpoint folder is used.
      If you pass nothing: each model uses its own latest timestamped folder

    If you pass --ckpt tcn=...: only tcn is pinned, others still use latest

    If you pass model=latest: explicitly forces latest for that model
"""

# -----------------------------------------------------------------------------
# Binary fall decision args (used for fall vs no-fall metrics)
#
# This eval script supports two ways to convert model outputs into a binary
# "fall" decision:
#
#   --binary-mode threshold
#     If the model has an explicit fall head (returns (activity_logits, fall_logit)),
#     then P(fall)=sigmoid(fall_logit) is used.
#     Otherwise, P(fall)=sum of softmax probabilities over the classes listed in
#     --fall-class-ids.
#     Predict fall if P(fall) >= --threshold.
#
#   --binary-mode argmax
#     Uses the model's argmax class and predicts fall if that class is in
#     --fall-class-ids. This matches the older, stricter behaviour but gives you
#     no operating point control.
#
# Automatic threshold tuning (recommended):
#   --tune-subjects <range/list>
#     If provided, the script will run a PR-curve sweep on the tuning split and
#     pick the threshold that maximises F_beta (see --beta). This chosen value is
#     written to tuned_threshold in metrics_summary.csv.
#
# The summary CSV includes a "p_fall_source" column which records whether
# P(fall) came from the model's fall head or from activity softmax mass.
# -----------------------------------------------------------------------------

from __future__ import annotations

from datetime import datetime
import json
import re
import math

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# Same dataset pipeline as training
from dataset_helpers.dataset import (
    find_keypoints_npzs_subjects,
    load_windows_with_source_meta_from_npzs,
    WindowTensorDataset,
    detect_label_convention_from_npzs,
    get_new_label_names,
    detect_label_convention as _detect_label_convention,
    remap_label as _remap_label,
    get_fall_merge_set as _get_fall_merge_set,
)

# Same model definitions as training
from models.tcn.simple_tcn import TCNBaseline
from models.lstm.simple_lstm import LSTMBaseline
from models.gru.simple_gru import GRUBaseline
from models.gcn.simple_gcn import GCNBaseline
from models.mlp.simple_mlp import MLPBaseline
from models.stgcn.simple_stgcn import STGCNBaseline
from models.complex_models import PaperLSTMClassifier, PaperSTGCNClassifier

# NEW: CNN + LSTM two-head model
from models.cnnlstm.cnn_lstm import CNNLSTMTwoHead
from models.rf.train_rf import windows_to_sklearn_features


import inspect
import pickle


# =============================================================================
# Label scheme: merge fall subclasses into a single "Fall" class (7 classes total)
#
# dataset.load_windows_from_npzs returns labels in merged 7-class space (0..6).
# In this scheme, Fall is always class id 0.
# =============================================================================
NUM_CLASSES_MERGED = 7
FALL_CLASS_ID = 0

# FALL_MERGE_SET and NEW_LABEL_NAMES are filled after we scan NPZ labels.
FALL_MERGE_SET: set[int] = set()
NEW_LABEL_NAMES: list[str] = []

def detect_label_convention(observed_labels) -> str:
    """Wrapper that calls the shared implementation in dataset.py."""
    return _detect_label_convention(observed_labels)

def remap_label(original_label: int, convention: str) -> int:
    """Wrapper that calls the shared implementation in dataset.py."""
    return _remap_label(original_label, convention)

def fall_merge_set(convention: str) -> set[int]:
    """Return the raw fall ID set for the detected convention."""
    return _get_fall_merge_set(convention)

def torch_load_safe(path: Path, map_location: str = "cpu"):
    """Load a torch checkpoint robustly across PyTorch versions.

    PyTorch 2.6+ defaults torch.load(weights_only=True), which can fail when the
    checkpoint dict contains NumPy scalar metadata (common in our saved ckpts).
    We first retry with a small allowlist under weights-only loading, then fall
    back to a full unpickle load if needed.
    """
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        # Retry with an allowlist for NumPy scalar metadata (PyTorch 2.6+).
        try:
            import numpy as _np
            try:
                _np_scalar = _np.core.multiarray.scalar
            except Exception:
                _np_scalar = _np._core.multiarray.scalar  # type: ignore[attr-defined]

            try:
                from torch.serialization import safe_globals  # PyTorch 2.6+
                with safe_globals([_np_scalar]):
                    return torch.load(path, map_location=map_location)
            except Exception:
                pass
        except Exception:
            pass

        # Final fallback: full unpickle (only safe for trusted checkpoints).
        try:
            sig = inspect.signature(torch.load)
            if "weights_only" in sig.parameters:
                return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            pass

        # Older torch versions (or if weights_only isn't a valid kwarg)
        return torch.load(path, map_location=map_location)


def slug_models(models: List[str], max_len: int = 80) -> str:
    # safe folder component: letters, numbers, underscore and dash only
    s = "-".join(models)
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
    return s[:max_len]


def slug_token(text: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text))
    s = s.strip("_")
    if not s:
        s = "na"
    return s[:max_len]


_ACTIVITY_TOKEN_RE = re.compile(r"^activity(\d+)$", re.IGNORECASE)
_SUBJECT_SEARCH_RE = re.compile(r"subject(\d+)", re.IGNORECASE)
_ACTIVITY_SEARCH_RE = re.compile(r"activity(\d+)", re.IGNORECASE)
_TRIAL_SEARCH_RE = re.compile(r"trial(\d+)", re.IGNORECASE)
_CAMERA_SEARCH_RE = re.compile(r"camera(\d+)", re.IGNORECASE)
_WINDOW_ALIGNED_META_KEYS = (
    "window_camera_ids",
    "window_source_indices",
    "window_candidate_indices",
    "window_start_frames_sampled",
    "window_end_frames_sampled",
    "window_frame_counts_sampled",
    "window_is_padded",
)
_EXCLUDED_EVAL_VIDEO_KEYS = frozenset({
    (16, 4, 3, 1),
    (16, 4, 3, 2),
    (16, 6, 1, 2),
})
_TS_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?$")


def pick_latest_run_dir(model_dir: Path) -> Path:
    run_dirs = [p for p in model_dir.iterdir() if p.is_dir() and _TS_DIR_RE.match(p.name)]
    if not run_dirs:
        raise FileNotFoundError(f"No timestamped run folders under: {model_dir.as_posix()}")
    return sorted(run_dirs, key=lambda p: p.name)[-1]


def parse_ckpt_overrides(items: Optional[List[str]]) -> Dict[str, str]:
    """
    Parses ['tcn=2026-...', 'lstm=latest'] -> {'tcn': '2026-...', 'lstm': 'latest'}
    """
    out: Dict[str, str] = {}
    if not items:
        return out
    for s in items:
        if "=" not in s:
            raise SystemExit(f"--ckpt entries must be like model=RUNFOLDER or model=latest, got: {s}")
        k, v = s.split("=", 1)
        out[k.lower().strip()] = v.strip()
    return out


def _activity_id_from_npz_path(npz_path: str | Path) -> Optional[int]:
    for part in Path(npz_path).parts:
        match = _ACTIVITY_TOKEN_RE.match(str(part))
        if match:
            return int(match.group(1))
    return None


def _extract_npz_subject_activity_trial_camera(npz_path: str | Path) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    subject = None
    activity = None
    trial = None
    camera = None

    for part in Path(npz_path).parts:
        text = str(part)
        if subject is None:
            match = _SUBJECT_SEARCH_RE.search(text)
            if match:
                subject = int(match.group(1))
        if activity is None:
            match = _ACTIVITY_SEARCH_RE.search(text)
            if match:
                activity = int(match.group(1))
        if trial is None:
            match = _TRIAL_SEARCH_RE.search(text)
            if match:
                trial = int(match.group(1))
        if camera is None:
            match = _CAMERA_SEARCH_RE.search(text)
            if match:
                camera = int(match.group(1))

    return subject, activity, trial, camera


def _is_excluded_eval_npz(npz_path: str | Path) -> bool:
    return _extract_npz_subject_activity_trial_camera(npz_path) in _EXCLUDED_EVAL_VIDEO_KEYS


def _filter_excluded_eval_npzs(npz_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    kept: List[Path] = []
    excluded: List[Path] = []
    for path in npz_paths:
        if _is_excluded_eval_npz(path):
            excluded.append(Path(path))
        else:
            kept.append(Path(path))
    return kept, excluded


def _window_count_from_meta(meta: Dict[str, Any]) -> int:
    return int(np.asarray(meta.get("window_source_indices", np.zeros((0,), dtype=np.int64))).shape[0])


def _load_frame_has_pose_flags(
    npz_path: str | Path,
    *,
    conf_thres: float,
    cache: Dict[Tuple[str, float], np.ndarray],
) -> np.ndarray:
    cache_key = (Path(npz_path).resolve().as_posix(), float(conf_thres))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    with np.load(npz_path, allow_pickle=True) as data:
        if "kpts_conf" not in data.files:
            raise KeyError(f"Expected 'kpts_conf' in {Path(npz_path).as_posix()}")
        kpts_conf = np.asarray(data["kpts_conf"])

    if kpts_conf.ndim == 2:
        conf = kpts_conf
    elif kpts_conf.ndim == 3:
        if int(kpts_conf.shape[1]) <= 0:
            raise ValueError(f"Unexpected empty person axis in kpts_conf for {Path(npz_path).as_posix()}")
        conf = kpts_conf[:, 0, :]
    else:
        raise ValueError(f"Unexpected kpts_conf shape {kpts_conf.shape} in {Path(npz_path).as_posix()}")

    conf = np.nan_to_num(conf.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    frame_has_pose = np.any(conf > float(conf_thres), axis=1).astype(bool, copy=False)
    cache[cache_key] = frame_has_pose
    return frame_has_pose


def _activity6_majority_missing_window_mask(
    *,
    meta: Dict[str, Any],
    conf_thres: float,
    frame_has_pose_cache: Dict[Tuple[str, float], np.ndarray],
) -> np.ndarray:
    source_indices = np.asarray(meta.get("window_source_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    start_frames = np.asarray(meta.get("window_start_frames_sampled", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    frame_counts = np.asarray(meta.get("window_frame_counts_sampled", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    source_paths = [str(p) for p in meta.get("source_npz_paths", [])]

    n_windows = int(source_indices.shape[0])
    if start_frames.shape[0] != n_windows or frame_counts.shape[0] != n_windows:
        raise RuntimeError(
            "Window metadata length mismatch while applying Activity 6 missing-pose filtering: "
            f"source_indices={n_windows}, start_frames={int(start_frames.shape[0])}, "
            f"frame_counts={int(frame_counts.shape[0])}"
        )

    skip_mask = np.zeros((n_windows,), dtype=bool)
    for idx in range(n_windows):
        src_idx = int(source_indices[idx])
        if not (0 <= src_idx < len(source_paths)):
            raise RuntimeError(
                "Window source index out of range while applying Activity 6 missing-pose filtering: "
                f"window={idx}, source_index={src_idx}, num_sources={len(source_paths)}"
            )

        src_path = source_paths[src_idx]
        if _activity_id_from_npz_path(src_path) != 6:
            continue

        frame_count = int(frame_counts[idx])
        if frame_count <= 0:
            continue

        start = int(start_frames[idx])
        frame_has_pose = _load_frame_has_pose_flags(
            src_path,
            conf_thres=float(conf_thres),
            cache=frame_has_pose_cache,
        )
        end = start + frame_count
        if start < 0 or end > int(frame_has_pose.shape[0]):
            raise RuntimeError(
                "Window frame span is out of range while applying Activity 6 missing-pose filtering: "
                f"path={src_path}, start={start}, frame_count={frame_count}, num_frames={int(frame_has_pose.shape[0])}"
            )

        window_has_pose = frame_has_pose[start:end]
        missing_count = int(window_has_pose.size - np.count_nonzero(window_has_pose))
        if missing_count > (window_has_pose.size / 2.0):
            skip_mask[idx] = True

    return skip_mask


def _filter_window_aligned_meta(meta: Dict[str, Any], keep_mask: np.ndarray) -> Dict[str, Any]:
    filtered = dict(meta)
    n_windows = int(keep_mask.shape[0])
    for key in _WINDOW_ALIGNED_META_KEYS:
        if key not in meta:
            continue
        arr = np.asarray(meta[key])
        if arr.shape[0] != n_windows:
            raise RuntimeError(
                f"Window metadata key '{key}' has length {int(arr.shape[0])}, expected {n_windows}."
            )
        filtered[key] = arr[keep_mask]
    return filtered


def _drop_activity6_majority_missing_windows(
    *,
    X_windows: np.ndarray,
    y_windows: np.ndarray,
    meta: Dict[str, Any],
    conf_thres: float,
    frame_has_pose_cache: Dict[Tuple[str, float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], int]:
    n_windows = _window_count_from_meta(meta)
    if int(X_windows.shape[0]) != n_windows or int(y_windows.shape[0]) != n_windows:
        raise RuntimeError(
            "Window array length mismatch while applying Activity 6 missing-pose filtering: "
            f"X={int(X_windows.shape[0])}, y={int(y_windows.shape[0])}, meta={n_windows}"
        )

    skip_mask = _activity6_majority_missing_window_mask(
        meta=meta,
        conf_thres=float(conf_thres),
        frame_has_pose_cache=frame_has_pose_cache,
    )
    skipped = int(np.count_nonzero(skip_mask))
    if skipped <= 0:
        return X_windows, y_windows, meta, 0

    keep_mask = ~skip_mask
    return (
        X_windows[keep_mask],
        y_windows[keep_mask],
        _filter_window_aligned_meta(meta, keep_mask),
        skipped,
    )


def parse_weights_path_overrides(
    items: Optional[List[str]],
    model_list: List[str],
) -> Dict[str, Path]:
    """
    Parse direct weight-file overrides.

    Accepted forms:
      - Single model only: --weights-path /abs/or/rel/path/to/file.pt
      - Multi-model (or explicit): --weights-path tcn=.../tcn_best.pt lstm=.../lstm_best.pt
    """
    out: Dict[str, Path] = {}
    if not items:
        return out

    models_norm = [str(m).lower().strip() for m in model_list]
    model_set = set(models_norm)

    if len(items) == 1 and "=" not in str(items[0]):
        if len(models_norm) != 1:
            raise SystemExit(
                "When --weights-path is provided as a single path, exactly one model must be selected.\n"
                "Use --weights-path model=PATH for multiple models."
            )
        out[models_norm[0]] = Path(str(items[0]).strip()).expanduser()
        return out

    for s in items:
        if "=" not in s:
            raise SystemExit(
                f"--weights-path entries must be like model=PATH (or a single PATH with one model). Got: {s}"
            )
        k, v = s.split("=", 1)
        mk = k.lower().strip()
        pv = v.strip()
        if not mk:
            raise SystemExit(f"--weights-path entry has empty model name: {s}")
        if not pv:
            raise SystemExit(f"--weights-path entry has empty path: {s}")
        out[mk] = Path(pv).expanduser()

    unknown = sorted(set(out.keys()) - model_set)
    if unknown:
        raise SystemExit(
            f"--weights-path contains model(s) not selected in --models/--all: {unknown}. "
            f"Selected models: {sorted(model_set)}"
        )

    return out


def resolve_run_dir(model_dir: Path, override: Optional[str]) -> Path:
    """
    override:
      - None -> latest
      - 'latest' -> latest
      - otherwise -> model_dir/override
    """
    if override is None or override.lower() == "latest":
        return pick_latest_run_dir(model_dir)

    run_dir = model_dir / override
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir.as_posix()}")
    return run_dir


def parse_range(r: str) -> List[int]:
    a, b = r.split("-")
    a, b = int(a), int(b)
    return list(range(a, b + 1))


def _ceil_div_pos(a: int, b: int) -> int:
    a_i = int(a)
    b_i = int(b)
    if b_i <= 0:
        raise ValueError(f"Expected positive divisor, got {b_i}")
    return max(1, (a_i + b_i - 1) // b_i)


def resolve_preprocess_config(
    *,
    ckpt: Optional[Dict[str, object]],
    is_rf: bool,
    use_conf_default: bool,
    normalize_default: bool,
    normalize_mode_default: str,
    add_vel_default: bool,
    add_acc_default: bool,
    add_global_default: bool,
    feature_mode_default: str,
    motion_xy_scale_default: float,
    conf_thres_default: float,
    max_interp_gap_default: int,
    missing_mode_default: str,
    interp_mode_default: str,
    interp_group_default: int,
    rp_center_mode_default: str,
    rp_img_w_default: Optional[int],
    rp_img_h_default: Optional[int],
    T_default: int,
    stride_default: int,
    fall_pct_default: float,
    label_mode_default: str,
    min_valid_frac_default: float,
    add_mask_channel_default: bool,
    drop_ambig_share_default: float,
    drop_ambig_nonfall_only_default: bool,
) -> Dict[str, object]:
    """
    Resolve preprocessing config from checkpoint metadata with CLI fallback.
    For RF checkpoints, "feature_mode" is reserved for sklearn feature extraction,
    so preprocessing feature mode is read from preprocess_feature_mode/window_feature_mode.
    """
    d = ckpt if isinstance(ckpt, dict) else {}

    if is_rf:
        preprocess_feature_mode = str(d.get("preprocess_feature_mode", d.get("window_feature_mode", feature_mode_default)))
    else:
        preprocess_feature_mode = str(d.get("feature_mode", d.get("preprocess_feature_mode", feature_mode_default)))

    mxy_raw = d.get("motion_xy_scale", motion_xy_scale_default)
    motion_xy_scale = float(motion_xy_scale_default if mxy_raw is None else mxy_raw)

    return {
        "use_conf": bool(d.get("use_conf", use_conf_default)),
        "normalize": bool(d.get("normalize", normalize_default)),
        "normalize_mode": str(d.get("normalize_mode", normalize_mode_default)),
        "add_vel": bool(d.get("add_vel", add_vel_default)),
        "add_acc": bool(d.get("add_acc", add_acc_default)),
        "add_global": bool(d.get("add_global", add_global_default)),
        "feature_mode": preprocess_feature_mode,
        "motion_xy_scale": motion_xy_scale,
        "conf_thres": float(d.get("conf_thres", conf_thres_default)),
        "max_interp_gap": int(d.get("max_interp_gap", max_interp_gap_default)),
        "missing_mode": str(d.get("missing_mode", missing_mode_default)),
        "interp_mode": str(d.get("interp_mode", interp_mode_default)),
        "interp_group": int(d.get("interp_group", interp_group_default)),
        "rp_center_mode": str(d.get("rp_center_mode", rp_center_mode_default)),
        "rp_img_w": d.get("rp_img_w", rp_img_w_default),
        "rp_img_h": d.get("rp_img_h", rp_img_h_default),
        "T_raw": int(d.get("T", d.get("T_used", T_default))),
        "stride_raw": int(d.get("stride", stride_default)),
        "fall_pct": float(d.get("fall_pct", fall_pct_default)),
        "label_mode": str(d.get("label_mode", label_mode_default)),
        "min_valid_frac": float(d.get("min_valid_frac", min_valid_frac_default)),
        "add_mask_channel": bool(d.get("add_mask_channel", add_mask_channel_default)),
        "drop_ambig_share": float(d.get("drop_ambig_share", drop_ambig_share_default)),
        "drop_ambig_nonfall_only": bool(d.get("drop_ambig_nonfall_only", drop_ambig_nonfall_only_default)),
    }


def _write_frame_step_npz(src_npz: Path, dst_npz: Path, frame_step: int) -> None:
    frame_step = int(frame_step)
    if frame_step <= 1:
        raise ValueError("frame_step must be >= 2 for subsampled NPZ export.")

    with np.load(src_npz, allow_pickle=True) as data:
        if "frame_labels" not in data:
            raise KeyError(f"Missing key 'frame_labels' in {src_npz.as_posix()}")
        n_frames = int(np.asarray(data["frame_labels"]).shape[0])
        if n_frames <= 0:
            raise ValueError(f"No frames in {src_npz.as_posix()}")

        out = {}
        for key in data.files:
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim >= 1 and int(arr.shape[0]) == n_frames:
                out[key] = arr[::frame_step]
            else:
                out[key] = arr

    for req in ("kpts_xy", "kpts_conf", "frame_labels"):
        if req not in out:
            raise KeyError(f"Missing required key '{req}' in subsampled NPZ derived from {src_npz.as_posix()}")

    dst_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_npz, **out)


def _materialize_frame_step_npzs(
    npz_paths: List[Path],
    frame_step: int,
    cache_dir: Path,
    cache: Dict[str, Path],
) -> List[Path]:
    if int(frame_step) <= 1:
        return [Path(p) for p in npz_paths]

    out_paths: List[Path] = []
    for p in npz_paths:
        src = Path(p).resolve()
        key = src.as_posix()
        if key not in cache:
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", src.stem)
            dst = cache_dir / f"{len(cache):06d}_{safe_stem}.npz"
            _write_frame_step_npz(src_npz=src, dst_npz=dst, frame_step=int(frame_step))
            cache[key] = dst
        out_paths.append(cache[key])
    return out_paths


def count_params_m(model: torch.nn.Module) -> float:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / 1e6


def unpack_model_output(out):
    """
    Supports:
      - single head: logits
      - two head: (activity_logits, fall_logit)
    """
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]
    return out, None


def get_model(
    model_name: str,
    in_features: int,
    num_classes: int,
    device: str,
    T_used: Optional[int] = None,
    node_features: Optional[int] = None,
):
    model_name = model_name.lower().strip()

    if model_name == "tcn":
        model = TCNBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            kernel_size=3,
            dropout=0.1,
        )

    elif model_name == "lstm":
        model = LSTMBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
            pool="last",
        )

    elif model_name == "paper_lstm":
        model = PaperLSTMClassifier(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=80,
            num_layers=10,
            dropout=0.2,
            pool="last",
        )

    elif model_name == "gru":
        model = GRUBaseline(
            in_features=in_features,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
            pool="last",
        )

    elif model_name == "gcn":
        if node_features is None:
            raise ValueError("node_features is required for GCN (load from ckpt).")
        model = GCNBaseline(
            num_nodes=17,
            node_features=node_features,
            num_classes=num_classes,
            hidden_size=64,
            dropout=0.1,
        )

    elif model_name == "mlp":
        if T_used is None:
            raise ValueError("T_used must be provided for MLP.")
        model = MLPBaseline(
            T=T_used,
            in_features=in_features,
            num_classes=num_classes,
            hidden_sizes=(256, 128),
            dropout=0.2,
        )

    elif model_name == "stgcn":
        if node_features is None:
            raise ValueError("node_features is required for STGCN (load from ckpt).")
        model = STGCNBaseline(
            num_nodes=17,
            node_features=node_features,
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            t_kernel=9,
            dropout=0.1,
        )

    elif model_name == "paper_stgcn":
        if node_features is None:
            raise ValueError("node_features is required for paper_stgcn (load from ckpt).")
        model = PaperSTGCNClassifier(
            num_nodes=17,
            node_features=node_features,
            num_classes=num_classes,
            channels=(64, 64, 64, 64, 128, 128, 128, 256, 256, 256),
            t_kernel=9,
            dropout=0.2,
        )

    elif model_name == "cnnlstm":
        # Uses keypoint-CNN path if in_features == 17 * node_features, else auto-falls back
        model = CNNLSTMTwoHead(
            in_features=in_features,
            num_classes=num_classes,
            embed_dim=128,
            hidden_size=128,
            lstm_layers=1,
            dropout=0.2,
            num_keypoints=17 if node_features is not None else None,
            kp_channels=node_features,
            pool="last",
        )

    else:
        raise ValueError(f"Unknown model '{model_name}'.")

    return model.to(device)


@torch.no_grad()
def predict_all(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_all, y_pred_all = [], []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        out = model(X)
        activity_logits, _fall_logit = unpack_model_output(out)

        preds = activity_logits.argmax(dim=1)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


@torch.no_grad()
def predict_probs(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      y_true:    (N,)
      probs:     (N,C) softmax probabilities over activity classes
      fall_prob: (N,) sigmoid(fall_logit) if model has fall head else None
    """
    model.eval()
    y_true_all, probs_all = [], []
    fall_prob_all: List[np.ndarray] = []

    has_fall_head = False

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        out = model(X)
        activity_logits, fall_logit = unpack_model_output(out)

        probs = torch.softmax(activity_logits, dim=1)

        y_true_all.append(y.detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())

        if fall_logit is not None:
            has_fall_head = True
            fp = torch.sigmoid(fall_logit).view(-1).detach().cpu().numpy()
            fall_prob_all.append(fp)

    y_true_np = np.concatenate(y_true_all)
    probs_np = np.concatenate(probs_all)
    if has_fall_head:
        fall_prob_np = np.concatenate(fall_prob_all)
        return y_true_np, probs_np, fall_prob_np
    return y_true_np, probs_np, None


def is_engine_weights_path(weights_path: Path) -> bool:
    return weights_path.suffix.lower() == ".engine"


def _strip_ultralytics_engine_header(raw: bytes) -> Optional[bytes]:
    """
    Ultralytics may prepend engine bytes with:
      [4-byte little-endian metadata length][JSON metadata][serialized TRT engine]
    Return stripped bytes if header looks valid, otherwise None.
    """
    if len(raw) <= 4:
        return None
    meta_len = int.from_bytes(raw[:4], byteorder="little", signed=False)
    max_reasonable = min(len(raw) - 4, 8 * 1024 * 1024)
    if meta_len <= 0 or meta_len > int(max_reasonable):
        return None
    meta_raw = raw[4 : 4 + meta_len]
    try:
        meta = json.loads(meta_raw.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    return raw[4 + meta_len :]


def _torch_dtype_from_numpy(np_dtype: np.dtype) -> torch.dtype:
    d = np.dtype(np_dtype)
    if d == np.float16:
        return torch.float16
    if d == np.float32:
        return torch.float32
    if d == np.float64:
        return torch.float64
    if d == np.int8:
        return torch.int8
    if d == np.int16:
        return torch.int16
    if d == np.int32:
        return torch.int32
    if d == np.int64:
        return torch.int64
    if d == np.uint8:
        return torch.uint8
    if d == np.bool_:
        return torch.bool
    raise TypeError(f"Unsupported numpy dtype for TensorRT tensor IO: {d}")


class TensorRTEngineRunner:
    """
    Minimal TensorRT runtime wrapper for batched inference on CUDA tensors.
    Supports both legacy (binding index) and name-based TensorRT Python APIs.
    """

    def __init__(self, engine_path: Path, device: str = "cuda"):
        self.engine_path = Path(engine_path)
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"TensorRT .engine requires CUDA, but CUDA is unavailable. "
                f"Cannot load: {self.engine_path.as_posix()}"
            )
        device_str = str(device).strip().lower()
        if not device_str.startswith("cuda"):
            raise RuntimeError(
                f"TensorRT .engine requires a CUDA device, but got --device={device!r} "
                f"for {self.engine_path.as_posix()}."
            )
        self.device = torch.device(device)

        try:
            import tensorrt as trt  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "TensorRT python package is required for .engine inference.\n"
                "Install the matching `tensorrt` wheel for your CUDA/TensorRT stack."
            ) from e

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        raw = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(raw)
        if self.engine is None:
            stripped = _strip_ultralytics_engine_header(raw)
            if stripped:
                self.engine = self.runtime.deserialize_cuda_engine(stripped)
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TensorRT engine: {self.engine_path.as_posix()}\n"
                "If this engine was exported with Ultralytics metadata, make sure it is valid for this TensorRT runtime."
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context for {self.engine_path.as_posix()}")

        self._name_api = hasattr(self.engine, "num_io_tensors") and hasattr(self.engine, "get_tensor_name")

        self.input_name: str
        self.input_index: Optional[int]
        self.input_shape_template: Tuple[int, ...]
        self.input_np_dtype: np.dtype
        self.output_names: List[str] = []
        self.output_indices: List[int] = []

        if self._name_api:
            n_io = int(self.engine.num_io_tensors)
            input_names: List[str] = []
            output_names: List[str] = []
            for i in range(n_io):
                name = str(self.engine.get_tensor_name(i))
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.INPUT:
                    input_names.append(name)
                elif mode == self.trt.TensorIOMode.OUTPUT:
                    output_names.append(name)

            if len(input_names) != 1:
                raise RuntimeError(
                    f"Expected exactly 1 TensorRT input tensor, found {len(input_names)} in {self.engine_path.as_posix()}"
                )
            if len(output_names) < 1:
                raise RuntimeError(f"No TensorRT output tensors found in {self.engine_path.as_posix()}")

            self.input_name = input_names[0]
            self.output_names = output_names
            self.input_index = None

            self.input_shape_template = tuple(int(v) for v in self.engine.get_tensor_shape(self.input_name))
            self.input_np_dtype = self._trt_dtype_to_numpy(self.engine.get_tensor_dtype(self.input_name))
        else:
            n_bindings = int(self.engine.num_bindings)
            input_indices = [i for i in range(n_bindings) if bool(self.engine.binding_is_input(i))]
            output_indices = [i for i in range(n_bindings) if not bool(self.engine.binding_is_input(i))]

            if len(input_indices) != 1:
                raise RuntimeError(
                    f"Expected exactly 1 TensorRT input binding, found {len(input_indices)} in {self.engine_path.as_posix()}"
                )
            if len(output_indices) < 1:
                raise RuntimeError(f"No TensorRT output bindings found in {self.engine_path.as_posix()}")

            self.input_index = int(input_indices[0])
            self.output_indices = [int(i) for i in output_indices]
            self.input_name = str(self.engine.get_binding_name(self.input_index))
            self.output_names = [str(self.engine.get_binding_name(i)) for i in self.output_indices]

            self.input_shape_template = tuple(int(v) for v in self.engine.get_binding_shape(self.input_index))
            self.input_np_dtype = self._trt_dtype_to_numpy(self.engine.get_binding_dtype(self.input_index))

        self.static_batch_size: Optional[int]
        if len(self.input_shape_template) >= 1 and int(self.input_shape_template[0]) > 0:
            self.static_batch_size = int(self.input_shape_template[0])
        else:
            self.static_batch_size = None

    def _trt_dtype_to_numpy(self, trt_dtype) -> np.dtype:
        try:
            return np.dtype(self.trt.nptype(trt_dtype))
        except Exception:
            s = str(trt_dtype).lower()
            if "float16" in s or "half" in s:
                return np.dtype(np.float16)
            if "float32" in s:
                return np.dtype(np.float32)
            if "int8" in s:
                return np.dtype(np.int8)
            if "int32" in s:
                return np.dtype(np.int32)
            if "bool" in s:
                return np.dtype(np.bool_)
            return np.dtype(np.float32)

    def _set_input_shape_if_needed(self, shape: Tuple[int, ...]) -> None:
        shape_t = tuple(int(v) for v in shape)
        if len(shape_t) != len(self.input_shape_template):
            raise ValueError(
                f"Input rank mismatch for TensorRT engine {self.engine_path.as_posix()}: "
                f"expected rank={len(self.input_shape_template)}, got rank={len(shape_t)} ({shape_t})"
            )

        for dim_i, (expect, got) in enumerate(zip(self.input_shape_template, shape_t)):
            if int(expect) >= 0 and int(expect) != int(got):
                raise ValueError(
                    f"Input shape mismatch at dim={dim_i} for {self.engine_path.as_posix()}: "
                    f"engine expects {self.input_shape_template}, got {shape_t}"
                )

        if any(int(v) < 0 for v in self.input_shape_template):
            if self._name_api and hasattr(self.context, "set_input_shape"):
                ok = self.context.set_input_shape(self.input_name, shape_t)
                if ok is False:
                    raise RuntimeError(
                        f"Failed to set TensorRT dynamic input shape {shape_t} on tensor '{self.input_name}'"
                    )
            elif self.input_index is not None and hasattr(self.context, "set_binding_shape"):
                ok = self.context.set_binding_shape(int(self.input_index), shape_t)
                if ok is False:
                    raise RuntimeError(
                        f"Failed to set TensorRT dynamic input shape {shape_t} on binding index {self.input_index}"
                    )
            else:
                raise RuntimeError(
                    "TensorRT engine appears to use dynamic shapes, but this runtime/context does not "
                    "expose set_input_shape or set_binding_shape."
                )

    def infer(self, x_batch: np.ndarray) -> List[np.ndarray]:
        x_np = np.asarray(x_batch)
        if x_np.ndim != len(self.input_shape_template):
            raise ValueError(
                f"TensorRT input rank mismatch for {self.engine_path.as_posix()}: "
                f"expected rank={len(self.input_shape_template)}, got shape={tuple(x_np.shape)}"
            )

        if x_np.dtype != self.input_np_dtype:
            x_np = x_np.astype(self.input_np_dtype, copy=False)
        if not x_np.flags.c_contiguous:
            x_np = np.ascontiguousarray(x_np)

        x_t = torch.as_tensor(
            x_np,
            device=self.device,
            dtype=_torch_dtype_from_numpy(self.input_np_dtype),
        )
        self._set_input_shape_if_needed(tuple(int(v) for v in x_t.shape))

        if self._name_api and hasattr(self.context, "set_tensor_address"):
            output_tensors: Dict[str, torch.Tensor] = {}
            self.context.set_tensor_address(self.input_name, int(x_t.data_ptr()))

            for out_name in self.output_names:
                out_shape = tuple(int(v) for v in self.context.get_tensor_shape(out_name))
                if any(int(v) < 0 for v in out_shape):
                    raise RuntimeError(
                        f"Unresolved dynamic output shape for tensor '{out_name}' in {self.engine_path.as_posix()}: {out_shape}"
                    )
                out_dtype = self._trt_dtype_to_numpy(self.engine.get_tensor_dtype(out_name))
                out_t = torch.empty(
                    out_shape,
                    device=self.device,
                    dtype=_torch_dtype_from_numpy(out_dtype),
                )
                self.context.set_tensor_address(out_name, int(out_t.data_ptr()))
                output_tensors[out_name] = out_t

            if hasattr(self.context, "execute_async_v3"):
                stream = torch.cuda.current_stream(device=self.device)
                ok = self.context.execute_async_v3(stream_handle=int(stream.cuda_stream))
            elif hasattr(self.context, "execute_v3"):
                ok = self.context.execute_v3()
            else:
                if not hasattr(self.engine, "num_bindings") or not hasattr(self.engine, "get_binding_index"):
                    raise RuntimeError("TensorRT context does not support execute_v3/execute_async_v3 or binding fallback.")
                bindings = [0] * int(self.engine.num_bindings)
                in_idx = int(self.engine.get_binding_index(self.input_name))
                bindings[in_idx] = int(x_t.data_ptr())
                for out_name, out_t in output_tensors.items():
                    out_idx = int(self.engine.get_binding_index(out_name))
                    bindings[out_idx] = int(out_t.data_ptr())
                ok = self.context.execute_v2(bindings)

            if not ok:
                raise RuntimeError(f"TensorRT execution failed for {self.engine_path.as_posix()}")
            torch.cuda.synchronize(self.device)
            return [output_tensors[name].detach().cpu().numpy() for name in self.output_names]

        if self.input_index is None or not hasattr(self.engine, "num_bindings"):
            raise RuntimeError(
                "TensorRT runtime does not expose required tensor-address or binding APIs for inference."
            )

        n_bindings = int(self.engine.num_bindings)
        bindings = [0] * n_bindings
        bindings[int(self.input_index)] = int(x_t.data_ptr())

        out_tensors: List[torch.Tensor] = []
        for out_idx in self.output_indices:
            out_shape = tuple(int(v) for v in self.context.get_binding_shape(int(out_idx)))
            if any(int(v) < 0 for v in out_shape):
                raise RuntimeError(
                    f"Unresolved dynamic output shape for binding index {out_idx} "
                    f"in {self.engine_path.as_posix()}: {out_shape}"
                )
            out_dtype = self._trt_dtype_to_numpy(self.engine.get_binding_dtype(int(out_idx)))
            out_t = torch.empty(
                out_shape,
                device=self.device,
                dtype=_torch_dtype_from_numpy(out_dtype),
            )
            bindings[int(out_idx)] = int(out_t.data_ptr())
            out_tensors.append(out_t)

        ok = self.context.execute_v2(bindings)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed for {self.engine_path.as_posix()}")
        torch.cuda.synchronize(self.device)
        return [t.detach().cpu().numpy() for t in out_tensors]


def _split_engine_outputs(outputs: List[np.ndarray], batch_size: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if not outputs:
        raise ValueError("TensorRT engine returned no outputs.")

    normalized: List[np.ndarray] = []
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        if arr.ndim == 1:
            if arr.shape[0] == int(batch_size):
                arr = arr.reshape(int(batch_size), 1)
            else:
                arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            if arr.shape[0] != int(batch_size):
                raise ValueError(
                    f"Unexpected TensorRT output shape {tuple(arr.shape)} for batch_size={batch_size}."
                )
            arr = arr.reshape(int(batch_size), -1)
        else:
            if arr.shape[0] != int(batch_size):
                raise ValueError(
                    f"Unexpected TensorRT output batch dimension: got {tuple(arr.shape)}, batch_size={batch_size}."
                )
        normalized.append(arr.astype(np.float32, copy=False))

    activity_i = max(range(len(normalized)), key=lambda i: int(normalized[i].shape[1]))
    activity_logits = normalized[activity_i]

    fall_logit = None
    if len(normalized) > 1:
        for i, arr in enumerate(normalized):
            if i == activity_i:
                continue
            if int(arr.shape[1]) == 1:
                fall_logit = arr.reshape(-1)
                break

    return activity_logits, fall_logit


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)


def predict_probs_engine(
    engine_runner: TensorRTEngineRunner,
    X_windows: np.ndarray,
    y_true: np.ndarray,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    X = np.asarray(X_windows)
    y = np.asarray(y_true).reshape(-1).astype(np.int64, copy=False)
    if int(X.shape[0]) != int(y.shape[0]):
        raise ValueError(f"X/y size mismatch for engine inference: X={int(X.shape[0])}, y={int(y.shape[0])}")

    if int(batch_size) <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    eff_bs = int(batch_size)
    static_bs = engine_runner.static_batch_size
    if static_bs is not None and int(static_bs) > 0:
        eff_bs = int(static_bs)

    y_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []
    fall_prob_all: List[np.ndarray] = []
    has_fall_head = False

    n = int(X.shape[0])
    for start in range(0, n, eff_bs):
        stop = min(start + eff_bs, n)
        xb = np.asarray(X[start:stop], dtype=np.float32)
        valid_n = int(xb.shape[0])

        if static_bs is not None and valid_n < eff_bs:
            pad_shape = (eff_bs - valid_n,) + tuple(int(v) for v in xb.shape[1:])
            xb = np.concatenate([xb, np.zeros(pad_shape, dtype=xb.dtype)], axis=0)

        out_list = engine_runner.infer(xb)
        activity_logits, fall_logit = _split_engine_outputs(out_list, batch_size=int(xb.shape[0]))

        if valid_n < int(xb.shape[0]):
            activity_logits = activity_logits[:valid_n]
            if fall_logit is not None:
                fall_logit = fall_logit[:valid_n]

        probs = _softmax_np(activity_logits.astype(np.float32, copy=False)).astype(np.float32, copy=False)

        y_all.append(y[start:stop])
        probs_all.append(probs)

        if fall_logit is not None:
            has_fall_head = True
            clipped = np.clip(fall_logit.astype(np.float32, copy=False), -50.0, 50.0)
            fp = (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32, copy=False)
            fall_prob_all.append(fp)

    y_np = np.concatenate(y_all)
    probs_np = np.concatenate(probs_all)
    if has_fall_head:
        return y_np, probs_np, np.concatenate(fall_prob_all)
    return y_np, probs_np, None


def load_engine_metadata(
    *,
    engine_path: Path,
    model_name: str,
    run_dir: Path,
    engine_meta_arg: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """
    Load optional metadata for TensorRT engines.

    `engine_meta_arg`:
      - "auto": try nearby .pt/.json metadata files
      - "none": disable metadata loading
      - otherwise: explicit path to .pt/.pth/.bin or .json
    """
    mode = str(engine_meta_arg).strip()
    mode_l = mode.lower()
    if mode_l == "none":
        return None, None

    candidate_paths: List[Path] = []
    if mode_l == "auto":
        seen: set[str] = set()

        def _add(p: Path):
            k = p.resolve().as_posix() if p.is_absolute() else p.as_posix()
            if k not in seen:
                seen.add(k)
                candidate_paths.append(p)

        stem = engine_path.stem
        _add(run_dir / f"{model_name}_best.pt")
        _add(run_dir / f"{stem}.pt")
        _add(run_dir / "best.pt")
        _add(engine_path.with_suffix(".json"))
        _add(run_dir / f"{stem}.json")
        _add(run_dir / f"{model_name}_engine_meta.json")
        _add(run_dir / "engine_meta.json")
    else:
        p_user = Path(mode).expanduser()
        if p_user.is_absolute():
            candidate_paths = [p_user]
        else:
            candidate_paths = [run_dir / p_user, p_user]

    for p in candidate_paths:
        if not p.exists() or not p.is_file():
            continue
        suf = p.suffix.lower()
        try:
            if suf in {".pt", ".pth", ".bin"}:
                obj = torch_load_safe(p, map_location="cpu")
                if isinstance(obj, dict):
                    return obj, p
            elif suf == ".json":
                with p.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    return obj, p
        except Exception as e:
            if mode_l != "auto":
                raise RuntimeError(f"Failed to load --engine-meta from {p.as_posix()}: {e}") from e

    if mode_l != "auto":
        raise FileNotFoundError(f"--engine-meta path not found or unsupported: {mode}")
    return None, None


def pickle_load_safe(path: Path):
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Failed to load pickle checkpoint. If this is an RF model, you likely need scikit-learn installed.\n"
            "Install it with: pip install scikit-learn\n"
            f"Import error: {e}"
        )


def rf_predict_probs(
    clf,
    X_windows: np.ndarray,
    feature_mode: str,
    num_classes: int,
    expected_feature_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Returns:
      probs: (N,C) aligned to class ids 0..C-1
    """
    X_feat = windows_to_sklearn_features(X_windows, mode=str(feature_mode))
    if expected_feature_dim is not None and int(expected_feature_dim) > 0 and int(X_feat.shape[1]) != int(expected_feature_dim):
        raise ValueError(
            f"RF feature_dim mismatch: extracted={int(X_feat.shape[1])}, ckpt feature_dim={int(expected_feature_dim)} "
            f"(mode={str(feature_mode)})"
        )

    if not hasattr(clf, "predict_proba"):
        raise TypeError("RF checkpoint model does not implement predict_proba().")

    raw = clf.predict_proba(X_feat)
    raw = np.asarray(raw)
    if raw.ndim != 2:
        raise ValueError(f"Expected predict_proba output (N,C). Got shape {getattr(raw, 'shape', None)}")

    num_classes = int(num_classes)
    out = np.zeros((int(raw.shape[0]), int(num_classes)), dtype=np.float32)

    classes = getattr(clf, "classes_", None)
    if classes is None:
        if int(raw.shape[1]) != int(num_classes):
            raise ValueError(f"RF predict_proba returned C={int(raw.shape[1])}, expected num_classes={int(num_classes)}")
        return raw.astype(np.float32, copy=False)

    classes_np = np.asarray(classes).astype(np.int64, copy=False).reshape(-1)
    for j, cls_id in enumerate(classes_np.tolist()):
        if 0 <= int(cls_id) < int(num_classes) and j < int(raw.shape[1]):
            out[:, int(cls_id)] = raw[:, int(j)]

    return out


def collapse_to_binary(y: np.ndarray, fall_class_ids_0based: List[int]) -> np.ndarray:
    fall = set(int(x) for x in fall_class_ids_0based)
    return np.array([1 if int(v) in fall else 0 for v in y], dtype=int)


def p_fall_from_probs(probs: np.ndarray, fall_class_ids_0based: List[int]) -> np.ndarray:
    idx = [int(i) for i in fall_class_ids_0based if 0 <= int(i) < probs.shape[1]]
    if len(idx) == 0:
        return np.zeros((probs.shape[0],), dtype=np.float32)
    return probs[:, idx].sum(axis=1)


def _ordered_unique_label_ids(labels: np.ndarray) -> Tuple[int, ...]:
    ordered: List[int] = []
    seen: set[int] = set()
    for value in np.asarray(labels, dtype=np.int64).tolist():
        lab = int(value)
        if lab not in seen:
            seen.add(lab)
            ordered.append(lab)
    return tuple(ordered)


def _load_frame_labels_merged(
    npz_path: str | Path,
    *,
    label_convention: str,
    cache: Dict[Tuple[str, str], np.ndarray],
) -> np.ndarray:
    cache_key = (Path(npz_path).resolve().as_posix(), str(label_convention))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    with np.load(npz_path, allow_pickle=True) as data:
        if "frame_labels" not in data.files:
            raise KeyError(f"Expected 'frame_labels' in {Path(npz_path).as_posix()}")
        raw_labels = np.asarray(data["frame_labels"]).astype(np.int64, copy=False).reshape(-1)

    merged = np.array(
        [remap_label(int(v), convention=str(label_convention)) for v in raw_labels.tolist()],
        dtype=np.int64,
    )
    cache[cache_key] = merged
    return merged


def _build_transition_eval_info(
    *,
    y_window_labels: np.ndarray,
    meta: Dict[str, Any],
    label_convention: str,
    fall_class_ids_0based: List[int],
    frame_label_cache: Dict[Tuple[str, str], np.ndarray],
) -> Dict[str, Any]:
    source_indices = np.asarray(meta.get("window_source_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    start_frames = np.asarray(meta.get("window_start_frames_sampled", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    frame_counts = np.asarray(meta.get("window_frame_counts_sampled", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    source_paths = [str(p) for p in meta.get("source_npz_paths", [])]

    y_default = np.asarray(y_window_labels, dtype=np.int64).reshape(-1).copy()
    n_windows = int(y_default.shape[0])
    if (
        int(source_indices.shape[0]) != n_windows
        or int(start_frames.shape[0]) != n_windows
        or int(frame_counts.shape[0]) != n_windows
    ):
        raise RuntimeError(
            "Window metadata length mismatch while building transitional evaluation info: "
            f"y={n_windows}, source_indices={int(source_indices.shape[0])}, "
            f"start_frames={int(start_frames.shape[0])}, frame_counts={int(frame_counts.shape[0])}"
        )

    fall_set = set(int(v) for v in fall_class_ids_0based)
    unique_label_counts = np.zeros((n_windows,), dtype=np.int64)
    center_labels = y_default.copy()
    acceptable_multi: List[Tuple[int, ...]] = []
    acceptable_binary: List[Tuple[int, ...]] = []

    for idx in range(n_windows):
        src_idx = int(source_indices[idx])
        if not (0 <= src_idx < len(source_paths)):
            raise RuntimeError(
                "Window source index out of range while building transitional evaluation info: "
                f"window={idx}, source_index={src_idx}, num_sources={len(source_paths)}"
            )

        src_path = source_paths[src_idx]
        labels_merged = _load_frame_labels_merged(
            src_path,
            label_convention=str(label_convention),
            cache=frame_label_cache,
        )

        start = int(start_frames[idx])
        frame_count = int(frame_counts[idx])
        end = start + frame_count
        if frame_count <= 0:
            raise RuntimeError(
                "Encountered a window with no sampled frames while building transitional evaluation info: "
                f"path={src_path}, start={start}, frame_count={frame_count}"
            )
        if start < 0 or end > int(labels_merged.shape[0]):
            raise RuntimeError(
                "Window label span is out of range while building transitional evaluation info: "
                f"path={src_path}, start={start}, frame_count={frame_count}, num_frames={int(labels_merged.shape[0])}"
            )

        window_labels = labels_merged[start:end]
        ordered_unique = _ordered_unique_label_ids(window_labels)
        if not ordered_unique:
            raise RuntimeError(
                "No labels found for sampled window while building transitional evaluation info: "
                f"path={src_path}, start={start}, frame_count={frame_count}"
            )

        unique_count = len(ordered_unique)
        unique_label_counts[idx] = int(unique_count)
        center_label = int(window_labels[int(window_labels.shape[0] // 2)])
        center_labels[idx] = center_label

        if unique_count == 2:
            acceptable = ordered_unique
            if int(y_default[idx]) not in acceptable:
                y_default[idx] = center_label
        elif unique_count >= 3:
            acceptable = (center_label,)
            y_default[idx] = center_label
        else:
            acceptable = (int(ordered_unique[0]),)
            y_default[idx] = int(ordered_unique[0])

        acceptable_multi.append(tuple(int(v) for v in acceptable))

        acceptable_bin_ordered: List[int] = []
        seen_bin: set[int] = set()
        for lab in acceptable:
            bin_lab = 1 if int(lab) in fall_set else 0
            if bin_lab not in seen_bin:
                seen_bin.add(bin_lab)
                acceptable_bin_ordered.append(int(bin_lab))
        acceptable_binary.append(tuple(acceptable_bin_ordered))

    return {
        "eval_default_true_multi": y_default,
        "eval_default_true_bin": collapse_to_binary(y_default, fall_class_ids_0based),
        "acceptable_multi_label_ids": acceptable_multi,
        "acceptable_binary_labels": acceptable_binary,
        "window_unique_label_count": unique_label_counts,
        "window_center_label": center_labels,
    }


def _singleton_allowed_label_sets(labels: np.ndarray) -> List[Tuple[int, ...]]:
    return [(int(v),) for v in np.asarray(labels, dtype=np.int64).reshape(-1).tolist()]


def _build_strict_eval_info(
    *,
    y_window_labels: np.ndarray,
    fall_class_ids_0based: List[int],
    reference_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    y_default = np.asarray(y_window_labels, dtype=np.int64).reshape(-1).copy()
    y_default_bin = collapse_to_binary(y_default, fall_class_ids_0based)

    if reference_info is not None:
        unique_counts = np.asarray(reference_info.get("window_unique_label_count", np.ones_like(y_default)), dtype=np.int64).copy()
        center_labels = np.asarray(reference_info.get("window_center_label", y_default), dtype=np.int64).copy()
    else:
        unique_counts = np.ones_like(y_default, dtype=np.int64)
        center_labels = y_default.copy()

    return {
        "eval_default_true_multi": y_default,
        "eval_default_true_bin": y_default_bin,
        "acceptable_multi_label_ids": _singleton_allowed_label_sets(y_default),
        "acceptable_binary_labels": _singleton_allowed_label_sets(y_default_bin),
        "window_unique_label_count": unique_counts,
        "window_center_label": center_labels,
    }


def _effective_true_labels_from_allowed(
    default_true: np.ndarray,
    y_pred: np.ndarray,
    acceptable_label_sets: List[Tuple[int, ...]],
) -> np.ndarray:
    default_arr = np.asarray(default_true, dtype=np.int64).reshape(-1)
    pred_arr = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if int(default_arr.shape[0]) != int(pred_arr.shape[0]) or int(pred_arr.shape[0]) != len(acceptable_label_sets):
        raise RuntimeError(
            "Length mismatch while applying transitional correctness rules: "
            f"default_true={int(default_arr.shape[0])}, y_pred={int(pred_arr.shape[0])}, "
            f"acceptable={len(acceptable_label_sets)}"
        )

    effective = default_arr.copy()
    for idx, pred in enumerate(pred_arr.tolist()):
        if int(pred) in acceptable_label_sets[idx]:
            effective[idx] = int(pred)
    return effective


def _label_ids_to_display(label_ids: Tuple[int, ...], class_names: List[str]) -> str:
    if len(label_ids) <= 0:
        return ""
    if len(label_ids) == 1:
        return _label_name(class_names, int(label_ids[0]))
    return " or ".join(_label_name(class_names, int(v)) for v in label_ids)


def _binary_metrics_for_allowed_predictions(
    *,
    default_true_bin: np.ndarray,
    acceptable_binary_labels: List[Tuple[int, ...]],
    y_pred_bin: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true_eval_bin = _effective_true_labels_from_allowed(
        default_true=np.asarray(default_true_bin, dtype=np.int64),
        y_pred=np.asarray(y_pred_bin, dtype=np.int64),
        acceptable_label_sets=acceptable_binary_labels,
    )
    pr, rc, f1b, _ = precision_recall_fscore_support(
        y_true_eval_bin,
        np.asarray(y_pred_bin, dtype=np.int64),
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    return y_true_eval_bin, pr, rc, f1b


def _append_eval_variant_results(
    *,
    variant_key: str,
    variant_cfg: Dict[str, Any],
    variant_info: Dict[str, Any],
    model_name: str,
    split_name: str,
    split_slug: str,
    split_camera_ids: List[int],
    y_model_true: np.ndarray,
    y_pred: np.ndarray,
    probs_test: np.ndarray,
    p_fall_test: np.ndarray,
    p_fall_source: str,
    num_classes_eval: int,
    test_meta: Dict[str, Any],
    y_window_labels: np.ndarray,
    output_root: Path,
    ckpt_path: Path,
    test_subjects: List[int],
    frame_step: int,
    T_ckpt_raw: int,
    stride_ckpt_raw: int,
    T_ckpt: int,
    stride_ckpt: int,
    normalize_mode_ckpt: str,
    missing_mode_ckpt: str,
    interp_mode_ckpt: str,
    interp_group_ckpt: int,
    rp_center_mode_ckpt: str,
    rp_img_w_ckpt: Optional[int],
    rp_img_h_ckpt: Optional[int],
    feature_mode_ckpt: str,
    motion_xy_scale_ckpt: float,
    model_params_m: float,
    binary_mode: str,
    threshold_arg: Optional[float],
    beta: float,
    tune_subjects_arg: Optional[str],
    fall_class_ids_0based: List[int],
    skipped_test_windows: int,
    skipped_tune_windows: int,
    tune_variant_info: Optional[Dict[str, Any]],
    p_fall_tune: Optional[np.ndarray],
) -> None:
    variant_out_dir = Path(variant_cfg["out_dir"])
    variant_plots_dir = Path(variant_cfg["plots_dir"])

    y_true_default_multi = np.asarray(variant_info["eval_default_true_multi"], dtype=np.int64)
    y_model_true = np.asarray(y_model_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if int(y_model_true.shape[0]) != int(y_true_default_multi.shape[0]):
        raise RuntimeError(
            f"[{model_name}][{split_name}][{variant_key}] Prediction/label length mismatch: "
            f"predictions={int(y_model_true.shape[0])}, labels={int(y_true_default_multi.shape[0])}"
        )

    acceptable_multi_label_ids = list(variant_info["acceptable_multi_label_ids"])
    y_true_eval = _effective_true_labels_from_allowed(
        default_true=y_true_default_multi,
        y_pred=y_pred,
        acceptable_label_sets=acceptable_multi_label_ids,
    )

    labels_all = list(range(int(num_classes_eval)))
    cm_counts = confusion_matrix(y_true_eval, y_pred, labels=labels_all).astype(np.float64)
    row_sums = cm_counts.sum(axis=1, keepdims=True) + 1e-9
    cm = cm_counts / row_sums

    if len(NEW_LABEL_NAMES) >= int(num_classes_eval):
        cm_names = NEW_LABEL_NAMES[:int(num_classes_eval)]
    else:
        cm_names = [str(i) for i in labels_all]

    cm_csv = variant_out_dir / f"confusion_matrix_{model_name}_{split_slug}.csv"
    pd.DataFrame(cm, index=cm_names, columns=cm_names).to_csv(cm_csv)
    make_cm_plot(
        cm,
        cm_names,
        variant_plots_dir / f"confusion_matrix_{model_name}_{split_slug}.png",
        title=f"Confusion Matrix: {model_name} [{split_name}]",
    )

    eps = 1e-12
    support = cm_counts.sum(axis=1)
    valid = support > 0
    tp = np.diag(cm_counts)
    pred_support = cm_counts.sum(axis=0)
    recall = tp / (support + eps)
    precision = tp / (pred_support + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    specificity = specificity_from_cm(cm_counts, eps=eps)

    total = float(np.sum(cm_counts))
    acc = float(np.sum(tp) / total) if total > 0 else 0.0
    macro_recall = float(np.mean(recall[valid])) if np.any(valid) else 0.0
    macro_precision = float(np.mean(precision[valid])) if np.any(valid) else 0.0
    macro_specificity = float(np.mean(specificity[valid])) if np.any(valid) else 0.0
    macro_f1 = float(np.mean(f1[valid])) if np.any(valid) else 0.0

    for lab, f1v in zip(labels_all, f1):
        name = cm_names[lab] if 0 <= lab < len(cm_names) else str(lab)
        variant_cfg["f1_rows"].append({
            "model": model_name,
            "eval_split": split_name,
            "class_id": int(lab),
            "class_name": name,
            "f1": float(f1v),
        })

    variant_cfg["overall_rows"].append({
        "model": model_name,
        "eval_split": split_name,
        "camera": ",".join(str(c) for c in split_camera_ids),
        "n_samples": int(len(y_true_eval)),
        "activity6_majority_missing_skipped_windows": int(skipped_test_windows),
        "accuracy": float(acc) * 100.0,
        "recall": float(macro_recall) * 100.0,
        "specificity": float(macro_specificity) * 100.0,
        "precision": float(macro_precision) * 100.0,
        "f1_score": float(macro_f1) * 100.0,
    })

    y_true_bin_default = np.asarray(variant_info["eval_default_true_bin"], dtype=np.int64)
    acceptable_binary_labels = list(variant_info["acceptable_binary_labels"])
    pred_conf = probs_test[np.arange(len(y_pred)), y_pred].astype(np.float32, copy=False)
    wrong_idx = np.flatnonzero(y_true_eval != y_pred)

    source_indices = np.asarray(test_meta.get("window_source_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    source_paths = [str(p) for p in test_meta.get("source_npz_paths", [])]
    window_candidate_indices = np.asarray(test_meta.get("window_candidate_indices", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    window_start_frames = np.asarray(test_meta.get("window_start_frames_sampled", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    window_end_frames = np.asarray(test_meta.get("window_end_frames_sampled", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    window_frame_counts = np.asarray(test_meta.get("window_frame_counts_sampled", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    window_is_padded = np.asarray(test_meta.get("window_is_padded", np.zeros((0,), dtype=bool)), dtype=bool)
    window_camera_ids = np.asarray(test_meta.get("window_camera_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    window_unique_label_counts = np.asarray(variant_info["window_unique_label_count"], dtype=np.int64)
    window_center_labels = np.asarray(variant_info["window_center_label"], dtype=np.int64)

    for idx in wrong_idx.tolist():
        src_idx = int(source_indices[idx]) if idx < len(source_indices) else -1
        src_path = source_paths[src_idx] if 0 <= src_idx < len(source_paths) else ""
        true_id = int(y_true_default_multi[idx])
        pred_id = int(y_pred[idx])
        start_frame = int(window_start_frames[idx]) if idx < len(window_start_frames) else -1
        end_frame = int(window_end_frames[idx]) if idx < len(window_end_frames) else -1
        acceptable_ids = tuple(int(v) for v in acceptable_multi_label_ids[idx])
        acceptable_label_display = _label_ids_to_display(acceptable_ids, cm_names)
        variant_cfg["misclassified_rows"].append({
            "model": model_name,
            "eval_split": split_name,
            "camera": int(window_camera_ids[idx]) if idx < len(window_camera_ids) else ",".join(str(c) for c in split_camera_ids),
            "video_id": _video_id_from_npz_path(src_path) if src_path else "",
            "source_npz_relpath": _relative_path_str(src_path, output_root) if src_path else "",
            "window_index_in_video": int(window_candidate_indices[idx]) if idx < len(window_candidate_indices) else idx,
            "window_span_sampled": f"{start_frame}-{end_frame}",
            "window_start_frame_sampled": start_frame,
            "window_end_frame_sampled": end_frame,
            "window_frame_count_sampled": int(window_frame_counts[idx]) if idx < len(window_frame_counts) else None,
            "window_is_padded": bool(window_is_padded[idx]) if idx < len(window_is_padded) else None,
            "window_unique_label_count": int(window_unique_label_counts[idx]) if idx < len(window_unique_label_counts) else None,
            "center_label": _label_name(cm_names, int(window_center_labels[idx])) if idx < len(window_center_labels) else "",
            "center_label_id": int(window_center_labels[idx]) if idx < len(window_center_labels) else None,
            "acceptable_true_labels": acceptable_label_display,
            "acceptable_true_label_ids": ", ".join(str(int(v)) for v in acceptable_ids),
            "original_true_label": _label_name(cm_names, int(y_window_labels[idx])),
            "original_true_label_id": int(y_window_labels[idx]),
            "true_label": acceptable_label_display,
            "predicted_label": _label_name(cm_names, pred_id),
            "true_label_id": true_id,
            "predicted_label_id": pred_id,
            "predicted_confidence": float(pred_conf[idx]),
            "fall_probability": float(p_fall_test[idx]),
        })

    tuned_thr = None
    tuned_prec = None
    tuned_rec = None
    tuned_fbeta = None

    if str(binary_mode).lower() == "argmax":
        y_pred_bin = collapse_to_binary(y_pred, fall_class_ids_0based)
        thr = None
    else:
        if threshold_arg is not None:
            thr = float(threshold_arg)
        elif tune_variant_info is not None and p_fall_tune is not None:
            y_tune_bin = np.asarray(tune_variant_info["eval_default_true_bin"], dtype=np.int64)
            acceptable_tune_binary_labels = list(tune_variant_info["acceptable_binary_labels"])
            thr, tuned_prec, tuned_rec, tuned_fbeta = pick_threshold_fbeta(
                y_tune_bin,
                p_fall_tune,
                beta=float(beta),
                acceptable_binary_labels=acceptable_tune_binary_labels,
            )
            tuned_thr = thr
        else:
            thr = 0.5

        y_pred_bin = (p_fall_test >= float(thr)).astype(int)

    chosen_thr = float(thr) if thr is not None else None
    _, pr, rc, f1b = _binary_metrics_for_allowed_predictions(
        default_true_bin=y_true_bin_default,
        acceptable_binary_labels=acceptable_binary_labels,
        y_pred_bin=y_pred_bin,
    )

    summary_row = {
        "model": model_name,
        "eval_split": split_name,
        "n_samples": int(len(y_true_eval)),
        "activity6_majority_missing_skipped_windows": int(skipped_test_windows),
        "activity6_majority_missing_skipped_tune_windows": int(skipped_tune_windows),
        "params_m": float(model_params_m),
        "macro_f1": float(macro_f1),
        "binary_mode": str(binary_mode).lower(),
        "p_fall_source": p_fall_source,
        "threshold": chosen_thr,
        "beta": float(beta) if str(binary_mode).lower() == "threshold" else None,
        "tune_subjects": str(tune_subjects_arg) if tune_subjects_arg is not None else None,
        "tuned_threshold": tuned_thr,
        "tuned_precision_fall": tuned_prec,
        "tuned_recall_fall": tuned_rec,
        "tuned_fbeta": tuned_fbeta,
        "binary_precision_avg": float(np.mean(pr)),
        "binary_sensitivity_avg": float(np.mean(rc)),
        "binary_precision_fall": float(pr[1]),
        "binary_sensitivity_fall": float(rc[1]),
        "binary_precision_no_fall": float(pr[0]),
        "binary_sensitivity_no_fall": float(rc[0]),
        "binary_f1_avg": float(np.mean(f1b)),
        "binary_f1_fall": float(f1b[1]),
        "binary_f1_no_fall": float(f1b[0]),
        "weights": ckpt_path.as_posix(),
        "camera": ",".join(str(c) for c in split_camera_ids),
        "subjects": ",".join(str(s) for s in test_subjects),
        "frame_step": int(frame_step),
        "window_T_raw": int(T_ckpt_raw),
        "window_stride_raw": int(stride_ckpt_raw),
        "window_T_sampled": int(T_ckpt),
        "window_stride_sampled": int(stride_ckpt),
        "normalize_mode": str(normalize_mode_ckpt),
        "missing_mode": str(missing_mode_ckpt),
        "interp_mode": str(interp_mode_ckpt),
        "interp_group": int(interp_group_ckpt),
        "rp_center_mode": str(rp_center_mode_ckpt),
        "rp_img_w": rp_img_w_ckpt,
        "rp_img_h": rp_img_h_ckpt,
        "feature_mode": str(feature_mode_ckpt),
        "motion_xy_scale": float(motion_xy_scale_ckpt),
    }

    used_metric_suffixes: set[str] = set()
    for lab in labels_all:
        suffix = _metric_label_suffix(cm_names, int(lab))
        if suffix in used_metric_suffixes:
            suffix = f"{suffix}_{int(lab)}"
        used_metric_suffixes.add(suffix)
        summary_row[f"precision_{suffix}"] = float(precision[lab])
        summary_row[f"recall_{suffix}"] = float(recall[lab])
        summary_row[f"specificity_{suffix}"] = float(specificity[lab])
        summary_row[f"f1_{suffix}"] = float(f1[lab])
        summary_row[f"support_{suffix}"] = int(support[lab])

    variant_cfg["summary_rows"].append(summary_row)


def pick_threshold_fbeta(
    y_true_bin: np.ndarray,
    p_fall: np.ndarray,
    beta: float = 2.0,
    acceptable_binary_labels: Optional[List[Tuple[int, ...]]] = None,
) -> Tuple[float, float, float, float]:
    """
    Returns: (best_threshold, precision_at_best, recall_at_best, fbeta_at_best)

    If acceptable_binary_labels is provided, a prediction is treated as correct
    when its binary class is allowed for that window.
    """
    y_true_bin = np.asarray(y_true_bin, dtype=np.int64).reshape(-1)
    p_fall = np.asarray(p_fall, dtype=np.float32).reshape(-1)
    if int(y_true_bin.shape[0]) != int(p_fall.shape[0]):
        raise ValueError(f"y_true_bin/p_fall size mismatch: {int(y_true_bin.shape[0])} vs {int(p_fall.shape[0])}")

    if acceptable_binary_labels is None:
        accepted = [(int(v),) for v in y_true_bin.tolist()]
    else:
        if len(acceptable_binary_labels) != int(y_true_bin.shape[0]):
            raise ValueError(
                "acceptable_binary_labels length mismatch: "
                f"{len(acceptable_binary_labels)} vs {int(y_true_bin.shape[0])}"
            )
        accepted = acceptable_binary_labels

    finite_scores = p_fall[np.isfinite(p_fall)]
    if finite_scores.size == 0:
        return 0.5, 0.0, 0.0, 0.0

    uniq = np.unique(np.clip(finite_scores, 0.0, 1.0))
    candidates: List[float] = [0.0, 0.5, 1.0]
    candidates.extend(float(v) for v in uniq.tolist())
    if uniq.size >= 2:
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        candidates.extend(float(v) for v in mids.tolist())
    candidates = sorted(set(float(v) for v in candidates))

    beta_sq = float(beta) * float(beta)
    best_thr = 0.5
    best_prec = 0.0
    best_rec = 0.0
    best_fbeta = -1.0

    for thr in candidates:
        y_pred_bin = (p_fall >= float(thr)).astype(np.int64)
        _, pr, rc, _ = _binary_metrics_for_allowed_predictions(
            default_true_bin=y_true_bin,
            acceptable_binary_labels=accepted,
            y_pred_bin=y_pred_bin,
        )
        prec_fall = float(pr[1])
        rec_fall = float(rc[1])
        denom = beta_sq * prec_fall + rec_fall + 1e-9
        fbeta = ((1.0 + beta_sq) * prec_fall * rec_fall / denom) if denom > 0.0 else 0.0

        if (
            fbeta > best_fbeta + 1e-12
            or (
                abs(fbeta - best_fbeta) <= 1e-12
                and (
                    rec_fall > best_rec + 1e-12
                    or (
                        abs(rec_fall - best_rec) <= 1e-12
                        and (
                            prec_fall > best_prec + 1e-12
                            or (abs(prec_fall - best_prec) <= 1e-12 and float(thr) < best_thr)
                        )
                    )
                )
            )
        ):
            best_thr = float(thr)
            best_prec = prec_fall
            best_rec = rec_fall
            best_fbeta = float(fbeta)

    return best_thr, best_prec, best_rec, max(best_fbeta, 0.0)


def specificity_from_cm(cm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    One-vs-rest specificity (true negative rate) per class from a multi-class confusion matrix.

    cm shape: (C, C) with rows=true labels, cols=predicted labels.
    """
    cm = cm.astype(np.float64)

    total = float(np.sum(cm))
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn

    return tn / (tn + fp + eps)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def make_cm_plot(cm: np.ndarray, class_names: List[str], out_path: Path, title: str):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    fmt = "d" if np.issubdtype(cm.dtype, np.integer) else ".2f"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            color = "white" if float(v) == 0.0 else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_plots(summary_df: pd.DataFrame, plots_dir: Path):
    import matplotlib.pyplot as plt
    ensure_dir(plots_dir)

    if "eval_split" in summary_df.columns:
        x_labels = [
            f"{str(m)} [{str(s)}]"
            for m, s in zip(summary_df["model"].tolist(), summary_df["eval_split"].tolist())
        ]
    else:
        x_labels = [str(m) for m in summary_df["model"].tolist()]

    plt.figure()
    x = np.arange(len(summary_df))
    plt.bar(x, summary_df["macro_f1"])
    plt.xticks(x, x_labels, rotation=30, ha="right")
    plt.ylabel("Macro F1 (multi-class)")
    plt.title("Macro F1 by model")
    plt.tight_layout()
    plt.savefig(plots_dir / "macro_f1.png", dpi=200)
    plt.close()

    plt.figure()
    x = np.arange(len(summary_df))
    w = 0.4
    plt.bar(x - w/2, summary_df["binary_sensitivity_avg"], width=w, label="Sensitivity (avg)")
    plt.bar(x + w/2, summary_df["binary_precision_avg"], width=w, label="Precision (avg)")
    plt.xticks(x, x_labels, rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Binary fall/no-fall metrics (macro-averaged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "binary_metrics.png", dpi=200)
    plt.close()


def _relative_path_str(path_value: str | Path, root: Optional[Path] = None) -> str:
    p = Path(path_value)
    if root is not None:
        try:
            return p.relative_to(root).as_posix()
        except Exception:
            pass
    return p.as_posix()


def _video_id_from_npz_path(npz_path: str | Path) -> str:
    p = Path(npz_path)
    if p.name.lower() == "keypoints.npz" and p.parent.name:
        return p.parent.name
    return p.stem


def _label_name(class_names: List[str], label_id: int) -> str:
    if 0 <= int(label_id) < len(class_names):
        return str(class_names[int(label_id)])
    return str(int(label_id))


def _metric_label_suffix(class_names: List[str], label_id: int) -> str:
    raw_name = _label_name(class_names, label_id).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", raw_name).strip("_")
    if not token or token.isdigit():
        return f"label_{int(label_id)}"
    return token


def build_misclassification_video_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "eval_split",
        "camera",
        "video_id",
        "source_npz_relpath",
        "num_incorrect_windows",
        "incorrect_windows",
        "incorrect_window_spans",
        "mistakes",
    ]
    if detail_df.empty:
        return pd.DataFrame(columns=columns)

    rows: List[Dict[str, object]] = []
    group_cols = ["model", "eval_split", "camera", "video_id", "source_npz_relpath"]
    grouped = detail_df.groupby(group_cols, sort=True, dropna=False)
    for group_key, group in grouped:
        group_sorted = group.sort_values(
            ["window_index_in_video", "window_start_frame_sampled", "predicted_confidence"],
            kind="stable",
        ).reset_index(drop=True)
        mistakes = "; ".join(
            (
                f"w{int(row.window_index_in_video)} ({row.window_span_sampled}): "
                f"{row.true_label} -> {row.predicted_label}"
            )
            for row in group_sorted.itertuples(index=False)
        )
        rows.append({
            "model": group_key[0],
            "eval_split": group_key[1],
            "camera": group_key[2],
            "video_id": group_key[3],
            "source_npz_relpath": group_key[4],
            "num_incorrect_windows": int(len(group_sorted)),
            "incorrect_windows": ", ".join(f"w{int(v)}" for v in group_sorted["window_index_in_video"].tolist()),
            "incorrect_window_spans": ", ".join(group_sorted["window_span_sampled"].astype(str).tolist()),
            "mistakes": mistakes,
        })

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["eval_split", "model", "video_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def write_misclassification_report(detail_df: pd.DataFrame, out_dir: Path) -> List[Path]:
    detail_columns = [
        "model",
        "eval_split",
        "camera",
        "video_id",
        "source_npz_relpath",
        "window_index_in_video",
        "window_span_sampled",
        "window_start_frame_sampled",
        "window_end_frame_sampled",
        "window_frame_count_sampled",
        "window_is_padded",
        "window_unique_label_count",
        "center_label",
        "center_label_id",
        "acceptable_true_labels",
        "acceptable_true_label_ids",
        "original_true_label",
        "original_true_label_id",
        "true_label",
        "predicted_label",
        "true_label_id",
        "predicted_label_id",
        "predicted_confidence",
        "fall_probability",
    ]
    detail_prepped = detail_df.copy()
    for col in detail_columns:
        if col not in detail_prepped.columns:
            detail_prepped[col] = pd.Series(dtype="object")
    detail_prepped = detail_prepped[detail_columns]

    numeric_cols = ["predicted_confidence", "fall_probability"]
    for col in numeric_cols:
        if col in detail_prepped.columns:
            detail_prepped[col] = pd.to_numeric(detail_prepped[col], errors="coerce").round(4)

    detail_prepped = detail_prepped.sort_values(
        ["eval_split", "model", "video_id", "window_index_in_video"],
        ascending=[True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    video_summary_df = build_misclassification_video_summary(detail_prepped)

    xlsx_path = out_dir / "misclassification_report.xlsx"
    try:
        from openpyxl.utils import get_column_letter

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            video_summary_df.to_excel(writer, sheet_name="videos_with_errors", index=False)
            detail_prepped.to_excel(writer, sheet_name="misclassified_windows", index=False)

            for sheet_name, df in (
                ("videos_with_errors", video_summary_df),
                ("misclassified_windows", detail_prepped),
            ):
                ws = writer.sheets[sheet_name]
                ws.freeze_panes = "A2"
                ws.auto_filter.ref = ws.dimensions

                for col_idx, column_name in enumerate(df.columns, start=1):
                    values = [
                        "" if pd.isna(v) else str(v)
                        for v in df[column_name].tolist()
                    ]
                    max_len = len(str(column_name))
                    if values:
                        max_len = max(max_len, max(len(v) for v in values))
                    ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 80)

        return [xlsx_path]
    except Exception as exc:
        detail_csv = out_dir / "misclassified_windows.csv"
        video_csv = out_dir / "misclassified_videos.csv"
        html_path = out_dir / "misclassification_report.html"

        detail_prepped.to_csv(detail_csv, index=False)
        video_summary_df.to_csv(video_csv, index=False)

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Misclassification Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    p {{ margin-top: 0; color: #555; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f5f5; position: sticky; top: 0; }}
    tr:nth-child(even) {{ background: #fafafa; }}
  </style>
</head>
<body>
  <h1>Misclassification Report</h1>
  <p>Generated by eval_models.py. The summary is grouped by video, and the detailed table lists one row per misclassified window.</p>
  <h2>Videos With Errors</h2>
  {video_summary_df.to_html(index=False)}
  <h2>Misclassified Windows</h2>
  {detail_prepped.to_html(index=False)}
</body>
</html>
"""
        html_path.write_text(html, encoding="utf-8")
        print(f"[report][WARN] Could not write Excel report ({exc}). Wrote CSV/HTML fallback instead.")
        return [html_path, video_csv, detail_csv]


def make_html_report(
    summary_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    f1_long: pd.DataFrame,
    plots_dir: Path,
    out_path: Path,
    model_list: List[str],
):
    def df_to_html(df: pd.DataFrame) -> str:
        return df.to_html(index=False, escape=False, classes="tbl")

    css = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#111;}
    h1,h2{margin:0.2em 0;}
    .tbl{border-collapse:collapse;width:100%;margin:12px 0;}
    .tbl th,.tbl td{border:1px solid #ddd;padding:8px;font-size:14px;}
    .tbl th{background:#f6f6f6;text-align:left;}
    img{max-width:100%;height:auto;border:1px solid #eee;border-radius:10px;padding:8px;background:#fff;}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
    code{background:#f6f6f6;padding:2px 6px;border-radius:6px;}
    """

    conf_parts = []
    if "eval_split" in summary_df.columns:
        unique_pairs = summary_df[["model", "eval_split"]].drop_duplicates()
        for _, row in unique_pairs.iterrows():
            m = str(row["model"])
            split_name = str(row["eval_split"])
            split_slug = slug_token(split_name)
            rel_png = f"confusion_matrix_{m}_{split_slug}.png"
            p = plots_dir / rel_png
            if p.exists():
                conf_parts.append(
                    f"<div><h3>{m} [{split_name}]</h3><img src='{plots_dir.name}/{rel_png}' alt='Confusion matrix {m} {split_name}'/></div>"
                )
    else:
        for m in model_list:
            p = plots_dir / f"confusion_matrix_{m}.png"
            if p.exists():
                conf_parts.append(f"<div><h3>{m}</h3><img src='{plots_dir.name}/confusion_matrix_{m}.png' alt='Confusion matrix {m}'/></div>")
    conf_imgs = "\n".join(conf_parts) if conf_parts else "<p>No confusion matrices found.</p>"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Model Evaluation Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>Model Evaluation Report</h1>
  <p>
    Binary fall/no-fall metrics are computed by collapsing multi-class labels using the provided fall class ids,
    then macro-averaging precision and sensitivity over the two binary classes.
  </p>

  <h2>Summary</h2>
  {df_to_html(summary_df)}

  <h2>Overall metrics</h2>
  <p style="margin:0 0 6px 0;color:#444;font-size:13px;">
    Values are percentages. Precision/recall/specificity/F1 are macro-averaged over classes present in this split.
  </p>
  {df_to_html(overall_df)}

  <div class="grid">
    <div>
      <h2>Macro F1</h2>
      <img src="{plots_dir.name}/macro_f1.png" alt="Macro F1"/>
    </div>
    <div>
      <h2>Binary Sensitivity and Precision</h2>
      <img src="{plots_dir.name}/binary_metrics.png" alt="Binary metrics"/>
    </div>
  </div>

  <h2>F1 per class (multi-class)</h2>
  {df_to_html(f1_long)}

  <h2>Confusion matrices</h2>
  <p>Rows are true labels, columns are predicted labels. Labels follow the merged 7-class scheme.</p>
  <div class="grid">
    {conf_imgs}
  </div>

  <p style="margin-top:24px;font-size:13px;color:#444;">
    Generated by <code>models.eval_models</code>.
  </p>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


def main():
    # NEW: added "cnnlstm"
    ALL_MODELS = ["tcn", "lstm", "paper_lstm", "gru", "gcn", "mlp", "stgcn", "paper_stgcn", "cnnlstm", "rf"]

    parser = argparse.ArgumentParser(description="Evaluate trained models on UP-Fall windowed pose tensors.")
    parser.add_argument("--models", nargs="+", default=None, help="Models to evaluate, e.g. --models tcn lstm")
    parser.add_argument("--all", action="store_true", help="Evaluate all models (overrides --models).")
    parser.add_argument(
        "--camera",
        nargs="+",
        type=int,
        default=[1, 2],
        help="One or more camera indices to evaluate on; if multiple are given, outputs each camera split and a combined split.",
    )
    parser.add_argument("--test-subjects", type=str, default="1-1", help="Test subject range like '1-5'")
    parser.add_argument(
        "--npz-root",
        type=str,
        default="../../Datasets/UPFall_keypoints/outputs_npz",
        help="Root directory containing keypoint NPZ outputs (default: ../../Datasets/UPFall_keypoints/outputs_npz).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-root", type=str, default="models", help="Checkpoint root (default: models)")
    parser.add_argument(
        "--ckpt",
        nargs="*",
        default=None,
        help="Optional per-model run folder overrides. Use: --ckpt tcn=2026-... lstm=latest. "
             "If omitted for a model, latest is used.",
    )
    parser.add_argument(
        "--weights-name",
        type=str,
        default=None,
        help="Override weights filename. If omitted uses '<model>_best.pt' (or '<model>_best.pkl' for rf). Supports .engine for TensorRT.",
    )
    parser.add_argument(
        "--weights-path",
        nargs="+",
        default=None,
        help=(
            "Optional direct weight file path override. "
            "Use either a single PATH when evaluating one model, or model=PATH entries for multiple models. "
            "When set for a model, this takes precedence over --ckpt-root/--ckpt/--weights-name."
        ),
    )
    parser.add_argument(
        "--engine-meta",
        type=str,
        default="auto",
        help="Metadata source when using .engine weights: 'auto', 'none', or a path to .pt/.pth/.bin/.json metadata.",
    )
    parser.add_argument("--out-dir", type=str, default="eval_outputs", help="Output directory")
    parser.add_argument("--use-conf", action="store_true", help="Use confidence channel (x,y,conf).")
    parser.add_argument("--no-conf", action="store_true", help="Disable confidence channel (use x,y only).")
        # With the merged 7-class scheme, Fall is always class id 0 so you no longer
    # need to pass fall class ids. This flag is kept only for backwards compatibility
    # with older runs.
    parser.add_argument(
        "--fall-class-ids",
        nargs="+",
        type=int,
        default=None,
        help="(Optional, legacy) Fall class ids in the ORIGINAL label space. Not needed for fall-merged 7-class models.",
    )

    # Binary decision options (deployment-style)
    parser.add_argument(
        "--binary-mode",
        type=str,
        default="threshold",
        choices=["threshold", "argmax"],
        help="How to form fall/no-fall decision. 'threshold' uses P(fall) score (fall head if present else sum fall-class probs).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fall threshold on P(fall). If omitted and --binary-mode threshold, uses --tune-subjects if provided else 0.5.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Fbeta to optimise when tuning threshold (beta>1 prioritises recall).",
    )
    parser.add_argument(
        "--tune-subjects",
        type=str,
        default=None,
        help="Optional subject range (e.g. 13-16) to tune threshold on. Uses same preprocessing as test.",
    )
    parser.add_argument(
        "--fall-pct",
        type=float,
        default=0.25,
        help="Used only when label_mode is hybrid_center_fallpct. Window is labeled fall if >= fall_pct of valid frames are fall.",
    )

    # Preprocessing options
    parser.add_argument("--normalize", type=int, default=1, help="Normalise pose per frame (0/1).")
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="center_scale",
        choices=["center_scale", "root_scale", "paper_rp"],
        help="Normalisation mode when --normalize 1. center_scale=legacy; root_scale=hip-root relative + robust scale; paper_rp=paper Relative Position (translation only).",
    )
    parser.add_argument("--add-vel", type=int, default=1, help="Add velocity channels vx, vy (0/1).")
    parser.add_argument("--add-acc", type=int, default=1, help="Add acceleration channels ax, ay (0/1).")
    parser.add_argument("--add-global", type=int, default=1, help="Add global features (0/1).")
    parser.add_argument(
        "--feature-mode",
        type=str,
        default="full",
        choices=["full", "motion_primary"],
        help="Feature composition fallback for checkpoints missing this metadata.",
    )
    parser.add_argument(
        "--motion-xy-scale",
        type=float,
        default=0.25,
        help="Only for --feature-mode motion_primary: fallback reduced-xy scale when checkpoint metadata is missing.",
    )
    parser.add_argument("--conf-thres", type=float, default=0.2, help="Conf threshold for missing joints.")
    parser.add_argument("--max-interp-gap", type=int, default=5, help="Max gap (frames) for interpolation.")
    parser.add_argument(
        "--missing-mode",
        type=str,
        default="conf_thres",
        choices=["conf_thres", "zeros_only", "conf_or_zeros"],
        help="Missing-keypoint definition. conf_thres=legacy; zeros_only=paper; conf_or_zeros=union.",
    )
    parser.add_argument(
        "--interp-mode",
        type=str,
        default="short_gap_hold",
        choices=["short_gap_hold", "paper_group_linear"],
        help="Interpolation strategy for missing keypoints. short_gap_hold=legacy; paper_group_linear=paper (requires --interp-group).",
    )
    parser.add_argument("--interp-group", type=int, default=100, help="Group size (frames) for paper_group_linear interpolation (default: 100).")
    parser.add_argument(
        "--rp-center-mode",
        type=str,
        default="auto",
        choices=["auto", "normalized_01", "pixel"],
        help="Image center definition for --normalize-mode paper_rp. pixel requires --rp-img-w/--rp-img-h; auto infers [0,1] vs pixel from coords.",
    )
    parser.add_argument("--rp-img-w", type=int, default=None, help="Image width W for paper_rp when using pixel coordinates.")
    parser.add_argument("--rp-img-h", type=int, default=None, help="Image height H for paper_rp when using pixel coordinates.")
    parser.add_argument("--T", type=int, default=64, help="Sliding window length T.")
    parser.add_argument("--stride", type=int, default=16, help="Sliding window stride.")
    parser.add_argument(
        "--frame-step", "--k",
        type=int,
        default=1,
        help="Subsample NPZ frames by k before windowing (k>=1). Window T/stride are interpreted in raw frames and scaled to sampled frames.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="center",
        choices=["center", "majority", "hybrid_center_fallpct"],
    )
    parser.add_argument("--min-valid-frac", type=float, default=0.3)
    parser.add_argument("--add-mask-channel", type=int, default=1)
    parser.add_argument(
        "--drop-ambig-share",
        type=float,
        default=0.0,
        help="Drop windows where top-label share < this value (measured on valid frames). 0 disables.",
    )
    parser.add_argument(
        "--drop-ambig-nonfall-only",
        type=int,
        default=1,
        help="If 1, only drop ambiguous windows that contain no fall frames (helps preserve fall transitions).",
    )
    args = parser.parse_args()
    frame_step = int(args.frame_step)
    if frame_step <= 0:
        raise SystemExit("--frame-step/--k must be >= 1.")

    normalize_cli = bool(args.normalize)
    add_vel_cli = bool(args.add_vel)
    add_acc_cli = bool(args.add_acc)
    add_global_cli = bool(args.add_global)
    add_mask_channel_cli = bool(args.add_mask_channel)
    feature_mode_cli = str(args.feature_mode).lower().strip()
    if feature_mode_cli not in {"full", "motion_primary"}:
        raise SystemExit(f"Unknown --feature-mode: {args.feature_mode}")
    if not math.isfinite(float(args.motion_xy_scale)) or float(args.motion_xy_scale) < 0.0:
        raise SystemExit("--motion-xy-scale must be a finite float >= 0.")

    if args.all:
        model_list = ALL_MODELS
    else:
        if args.models is None or len(args.models) == 0:
            raise SystemExit("You must pass --models <one or more> or use --all.")
        model_list = [m.lower().strip() for m in args.models]
    unknown = sorted(set(model_list) - set(ALL_MODELS))
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Valid: {ALL_MODELS}")

    use_conf = True
    if args.no_conf:
        use_conf = False
    if args.use_conf:
        use_conf = True

    test_subjects = parse_range(args.test_subjects)
    camera_ids = sorted(set(int(c) for c in args.camera))
    if not camera_ids:
        raise SystemExit("--camera must contain at least one camera index.")
    if any(c <= 0 for c in camera_ids):
        raise SystemExit(f"--camera values must be positive integers. Got: {camera_ids}")

    # Load test set using the SAME NPZ->windows pipeline
    OUTPUT_ROOT = Path(args.npz_root)
    camera_to_test_npzs: Dict[int, List[Path]] = {}
    excluded_eval_npzs: List[Path] = []
    for camera_id in camera_ids:
        cam_npzs_all = sorted(
            set(
                Path(p)
                for p in find_keypoints_npzs_subjects(
                    OUTPUT_ROOT,
                    camera=camera_id,
                    subjects=test_subjects,
                )
            ),
            key=lambda p: p.as_posix(),
        )
        cam_npzs, cam_excluded = _filter_excluded_eval_npzs(cam_npzs_all)
        excluded_eval_npzs.extend(cam_excluded)
        if not cam_npzs:
            raise RuntimeError(
                f"No test NPZs found for camera={camera_id}. Check OUTPUT_ROOT, test subjects={test_subjects}, "
                "and explicit evaluation exclusions."
            )
        camera_to_test_npzs[int(camera_id)] = cam_npzs

    excluded_eval_npzs = sorted(set(excluded_eval_npzs), key=lambda p: p.as_posix())
    if excluded_eval_npzs:
        print(f"[eval] Excluding {len(excluded_eval_npzs)} requested video(s) from evaluation:")
        for excluded_path in excluded_eval_npzs:
            print(f"  - {_relative_path_str(excluded_path, OUTPUT_ROOT)}")

    if len(camera_ids) > 1:
        combined_test_npzs = sorted(
            set(p for cam in camera_ids for p in camera_to_test_npzs[int(cam)]),
            key=lambda p: p.as_posix(),
        )
        eval_splits: List[Tuple[str, List[int], List[Path]]] = [
            (f"camera_{int(cam)}", [int(cam)], list(camera_to_test_npzs[int(cam)]))
            for cam in camera_ids
        ]
        eval_splits.append(("combined", list(camera_ids), combined_test_npzs))
    else:
        only_cam = int(camera_ids[0])
        combined_test_npzs = list(camera_to_test_npzs[only_cam])
        eval_splits = [("combined", [only_cam], combined_test_npzs)]

    if not combined_test_npzs:
        raise RuntimeError(f"No test NPZs found. Check OUTPUT_ROOT, camera(s)={camera_ids}, and test subjects.")
    print("Cameras:", camera_ids)
    for split_name, split_camera_ids, split_npzs in eval_splits:
        print(f"[split] {split_name}: cameras={split_camera_ids} npzs={len(split_npzs)}")

    # ---- Detect raw label convention once (1-11 vs 0-10), then keep it consistent ----
    label_convention, label_stats = detect_label_convention_from_npzs(combined_test_npzs)
    NEW_LABEL_NAMES = get_new_label_names(label_convention)
    labels_all = list(range(len(NEW_LABEL_NAMES))) #New ==========================================================
    FALL_MERGE_SET = fall_merge_set(label_convention)
    print(f"[labels] Using raw convention: {label_convention} | New labels: {NEW_LABEL_NAMES}")

    # In the merged 7-class scheme, Fall is always class id 0.
    # Keep legacy support: if user supplies --fall-class-ids, we can still interpret them
    # for older checkpoints, but for fall-merged models we simply use [0].
    fall_class_ids_0based = [FALL_CLASS_ID]

    # One unique output folder per eval run, includes timestamp + models list
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    models_tag = slug_models(model_list)
    base_out = Path(args.out_dir).resolve()

    out_dir = base_out / f"{ts}__models_{models_tag}"
    ensure_dir(out_dir)

    print("Eval output dir:", out_dir.as_posix())

    report_variants: Dict[str, Dict[str, Any]] = {
        "strict_label_mode": {
            "out_dir": out_dir / "strict_label_mode",
            "plots_dir": out_dir / "strict_label_mode" / "plots",
            "summary_rows": [],
            "overall_rows": [],
            "f1_rows": [],
            "misclassified_rows": [],
            "description": "Only the window label produced by --label-mode is treated as correct.",
        },
        "two_label_transition_relaxed": {
            "out_dir": out_dir / "two_label_transition_relaxed",
            "plots_dir": out_dir / "two_label_transition_relaxed" / "plots",
            "summary_rows": [],
            "overall_rows": [],
            "f1_rows": [],
            "misclassified_rows": [],
            "description": (
                "If a window contains exactly two label types, predicting either is treated as correct; "
                "for windows with three or more label types, only the center sampled-frame label is treated as correct."
            ),
        },
    }
    for variant_cfg in report_variants.values():
        ensure_dir(Path(variant_cfg["out_dir"]))
        ensure_dir(Path(variant_cfg["plots_dir"]))

    npz_subsample_cache: Dict[str, Path] = {}
    npz_subsample_ctx = None
    if frame_step > 1:
        npz_subsample_ctx = tempfile.TemporaryDirectory(prefix=f"eval_models_k{frame_step}_")
        npz_cache_dir = Path(npz_subsample_ctx.name)
        eval_splits = [
            (
                split_name,
                split_camera_ids,
                _materialize_frame_step_npzs(
                    npz_paths=split_npzs,
                    frame_step=frame_step,
                    cache_dir=npz_cache_dir,
                    cache=npz_subsample_cache,
                ),
            )
            for split_name, split_camera_ids, split_npzs in eval_splits
        ]
        print(f"[window] --frame-step={frame_step}: using subsampled NPZ cache at {npz_cache_dir.as_posix()}")

    frame_has_pose_cache: Dict[Tuple[str, float], np.ndarray] = {}
    frame_label_cache: Dict[Tuple[str, str], np.ndarray] = {}

    ckpt_root = Path(args.ckpt_root)
    ckpt_overrides = parse_ckpt_overrides(args.ckpt)
    weights_path_overrides = parse_weights_path_overrides(args.weights_path, model_list)

    for m in model_list:
        is_rf = str(m).lower().strip() == "rf"
        if args.weights_name:
            weights_name = str(args.weights_name)
        else:
            weights_name = f"{m}_best.pkl" if is_rf else f"{m}_best.pt"

        if m in weights_path_overrides:
            ckpt_path = weights_path_overrides[m]
            if not ckpt_path.is_absolute():
                ckpt_path = (Path.cwd() / ckpt_path).resolve()
            else:
                ckpt_path = ckpt_path.resolve()
            run_dir = ckpt_path.parent
            print(f"[{m}] Using direct weights path: {ckpt_path.as_posix()}")
        else:
            model_dir = ckpt_root / m
            run_dir = resolve_run_dir(model_dir, ckpt_overrides.get(m))
            ckpt_path = run_dir / weights_name
            print(f"[{m}] Using run folder: {run_dir.name}")

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Weights not found for {m}: {ckpt_path.as_posix()}")
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Weights path for {m} is not a file: {ckpt_path.as_posix()}")

        rf_model = None
        rf_feature_mode = "flatten"
        rf_feature_dim = None
        model_params_m = float("nan")
        is_engine = (not is_rf) and is_engine_weights_path(ckpt_path)
        engine_runner: Optional[TensorRTEngineRunner] = None
        has_state_dict = False
        state = None
        T_used = None
        in_features: Optional[int] = None
        num_classes: Optional[int] = None
        node_features_ckpt: Optional[int] = None

        if is_rf:
            ckpt = pickle_load_safe(ckpt_path)
        elif is_engine:
            ckpt, engine_meta_source = load_engine_metadata(
                engine_path=ckpt_path,
                model_name=str(m),
                run_dir=run_dir,
                engine_meta_arg=str(args.engine_meta),
            )
            if ckpt is None:
                ckpt = {}
                print(f"[{m}][engine] No metadata found (engine_meta={args.engine_meta!r}); using CLI preprocessing args.")
            else:
                print(f"[{m}][engine] Using metadata: {engine_meta_source.as_posix() if engine_meta_source else 'unknown'}")
            engine_runner = TensorRTEngineRunner(ckpt_path, device=args.device)
            if engine_runner.static_batch_size is not None and int(engine_runner.static_batch_size) != int(args.batch_size):
                print(
                    f"[{m}][engine][INFO] static batch size={int(engine_runner.static_batch_size)} "
                    f"overrides --batch-size={int(args.batch_size)} for TensorRT inference."
                )
        else:
            ckpt = torch_load_safe(ckpt_path, map_location="cpu")

        if is_rf:
            if not isinstance(ckpt, dict) or "model" not in ckpt:
                raise TypeError(f"[rf] Unsupported checkpoint format: expected dict with key 'model' at {ckpt_path.as_posix()}")

            rf_model = ckpt.get("model", None)
            rf_feature_mode = str(ckpt.get("feature_mode", ckpt.get("rf_feature_mode", "flatten")))
            rf_feature_dim = ckpt.get("feature_dim", None)

            num_classes = int(ckpt.get("num_classes", NUM_CLASSES_MERGED))
            pre_cfg = resolve_preprocess_config(
                ckpt=ckpt,
                is_rf=True,
                use_conf_default=True,
                normalize_default=normalize_cli,
                normalize_mode_default=str(args.normalize_mode),
                add_vel_default=add_vel_cli,
                add_acc_default=add_acc_cli,
                add_global_default=add_global_cli,
                feature_mode_default=feature_mode_cli,
                motion_xy_scale_default=float(args.motion_xy_scale),
                conf_thres_default=float(args.conf_thres),
                max_interp_gap_default=int(args.max_interp_gap),
                missing_mode_default=str(args.missing_mode),
                interp_mode_default=str(args.interp_mode),
                interp_group_default=int(args.interp_group),
                rp_center_mode_default=str(args.rp_center_mode),
                rp_img_w_default=args.rp_img_w,
                rp_img_h_default=args.rp_img_h,
                T_default=int(args.T),
                stride_default=int(args.stride),
                fall_pct_default=float(args.fall_pct),
                label_mode_default=str(args.label_mode),
                min_valid_frac_default=float(args.min_valid_frac),
                add_mask_channel_default=add_mask_channel_cli,
                drop_ambig_share_default=float(args.drop_ambig_share),
                drop_ambig_nonfall_only_default=bool(args.drop_ambig_nonfall_only),
            )
            use_conf_ckpt = bool(pre_cfg["use_conf"])
            normalize_ckpt = bool(pre_cfg["normalize"])
            normalize_mode_ckpt = str(pre_cfg["normalize_mode"])
            add_vel_ckpt = bool(pre_cfg["add_vel"])
            add_acc_ckpt = bool(pre_cfg["add_acc"])
            add_global_ckpt = bool(pre_cfg["add_global"])
            feature_mode_ckpt = str(pre_cfg["feature_mode"])
            motion_xy_scale_ckpt = float(pre_cfg["motion_xy_scale"])
            conf_thres_ckpt = float(pre_cfg["conf_thres"])
            max_interp_gap_ckpt = int(pre_cfg["max_interp_gap"])
            missing_mode_ckpt = str(pre_cfg["missing_mode"])
            interp_mode_ckpt = str(pre_cfg["interp_mode"])
            interp_group_ckpt = int(pre_cfg["interp_group"])
            rp_center_mode_ckpt = str(pre_cfg["rp_center_mode"])
            rp_img_w_ckpt = pre_cfg["rp_img_w"]
            rp_img_h_ckpt = pre_cfg["rp_img_h"]
            T_ckpt_raw = int(pre_cfg["T_raw"])
            stride_ckpt_raw = int(pre_cfg["stride_raw"])
            fall_pct_ckpt = float(pre_cfg["fall_pct"])
            label_mode_ckpt = str(pre_cfg["label_mode"])
            min_valid_frac_ckpt = float(pre_cfg["min_valid_frac"])
            add_mask_channel_ckpt = bool(pre_cfg["add_mask_channel"])
            drop_ambig_share_ckpt = float(pre_cfg["drop_ambig_share"])
            drop_ambig_nonfall_only_ckpt = bool(pre_cfg["drop_ambig_nonfall_only"])
            node_features_ckpt = None

        elif is_engine:
            ckpt_dict = ckpt if isinstance(ckpt, dict) else {}

            in_features_raw = ckpt_dict.get("in_features", None)
            if in_features_raw is not None:
                in_features = int(in_features_raw)

            num_classes_raw = ckpt_dict.get("num_classes", None)
            if num_classes_raw is not None:
                num_classes = int(num_classes_raw)

            pre_cfg = resolve_preprocess_config(
                ckpt=ckpt_dict if ckpt_dict else None,
                is_rf=False,
                use_conf_default=use_conf,
                normalize_default=normalize_cli,
                normalize_mode_default=str(args.normalize_mode),
                add_vel_default=add_vel_cli,
                add_acc_default=add_acc_cli,
                add_global_default=add_global_cli,
                feature_mode_default=feature_mode_cli,
                motion_xy_scale_default=float(args.motion_xy_scale),
                conf_thres_default=float(args.conf_thres),
                max_interp_gap_default=int(args.max_interp_gap),
                missing_mode_default=str(args.missing_mode),
                interp_mode_default=str(args.interp_mode),
                interp_group_default=int(args.interp_group),
                rp_center_mode_default=str(args.rp_center_mode),
                rp_img_w_default=args.rp_img_w,
                rp_img_h_default=args.rp_img_h,
                T_default=int(args.T),
                stride_default=int(args.stride),
                fall_pct_default=float(args.fall_pct),
                label_mode_default=str(args.label_mode),
                min_valid_frac_default=float(args.min_valid_frac),
                add_mask_channel_default=add_mask_channel_cli,
                drop_ambig_share_default=float(args.drop_ambig_share),
                drop_ambig_nonfall_only_default=bool(args.drop_ambig_nonfall_only),
            )
            use_conf_ckpt = bool(pre_cfg["use_conf"])
            normalize_ckpt = bool(pre_cfg["normalize"])
            normalize_mode_ckpt = str(pre_cfg["normalize_mode"])
            add_vel_ckpt = bool(pre_cfg["add_vel"])
            add_acc_ckpt = bool(pre_cfg["add_acc"])
            add_global_ckpt = bool(pre_cfg["add_global"])
            feature_mode_ckpt = str(pre_cfg["feature_mode"])
            motion_xy_scale_ckpt = float(pre_cfg["motion_xy_scale"])
            conf_thres_ckpt = float(pre_cfg["conf_thres"])
            max_interp_gap_ckpt = int(pre_cfg["max_interp_gap"])
            missing_mode_ckpt = str(pre_cfg["missing_mode"])
            interp_mode_ckpt = str(pre_cfg["interp_mode"])
            interp_group_ckpt = int(pre_cfg["interp_group"])
            rp_center_mode_ckpt = str(pre_cfg["rp_center_mode"])
            rp_img_w_ckpt = pre_cfg["rp_img_w"]
            rp_img_h_ckpt = pre_cfg["rp_img_h"]
            T_ckpt_raw = int(pre_cfg["T_raw"])
            stride_ckpt_raw = int(pre_cfg["stride_raw"])
            fall_pct_ckpt = float(pre_cfg["fall_pct"])
            label_mode_ckpt = str(pre_cfg["label_mode"])
            min_valid_frac_ckpt = float(pre_cfg["min_valid_frac"])
            add_mask_channel_ckpt = bool(pre_cfg["add_mask_channel"])
            drop_ambig_share_ckpt = float(pre_cfg["drop_ambig_share"])
            drop_ambig_nonfall_only_ckpt = bool(pre_cfg["drop_ambig_nonfall_only"])
            node_features_ckpt = ckpt_dict.get("node_features", None)
            if node_features_ckpt is None and in_features is not None and (int(in_features) % 17 == 0):
                node_features_ckpt = int(in_features) // 17
            if node_features_ckpt is not None:
                node_features_ckpt = int(node_features_ckpt)

        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            has_state_dict = True
            state = ckpt["state_dict"]
            T_used = int(ckpt["T_used"])
            in_features = int(ckpt["in_features"])
            num_classes = int(ckpt["num_classes"])
            pre_cfg = resolve_preprocess_config(
                ckpt=ckpt,
                is_rf=False,
                use_conf_default=True,
                normalize_default=normalize_cli,
                normalize_mode_default=str(args.normalize_mode),
                add_vel_default=add_vel_cli,
                add_acc_default=add_acc_cli,
                add_global_default=add_global_cli,
                feature_mode_default=feature_mode_cli,
                motion_xy_scale_default=float(args.motion_xy_scale),
                conf_thres_default=float(args.conf_thres),
                max_interp_gap_default=int(args.max_interp_gap),
                missing_mode_default=str(args.missing_mode),
                interp_mode_default=str(args.interp_mode),
                interp_group_default=int(args.interp_group),
                rp_center_mode_default=str(args.rp_center_mode),
                rp_img_w_default=args.rp_img_w,
                rp_img_h_default=args.rp_img_h,
                T_default=int(args.T),
                stride_default=int(args.stride),
                fall_pct_default=float(args.fall_pct),
                label_mode_default=str(args.label_mode),
                min_valid_frac_default=float(args.min_valid_frac),
                add_mask_channel_default=add_mask_channel_cli,
                drop_ambig_share_default=float(args.drop_ambig_share),
                drop_ambig_nonfall_only_default=bool(args.drop_ambig_nonfall_only),
            )
            use_conf_ckpt = bool(pre_cfg["use_conf"])
            normalize_ckpt = bool(pre_cfg["normalize"])
            normalize_mode_ckpt = str(pre_cfg["normalize_mode"])
            add_vel_ckpt = bool(pre_cfg["add_vel"])
            add_acc_ckpt = bool(pre_cfg["add_acc"])
            add_global_ckpt = bool(pre_cfg["add_global"])
            feature_mode_ckpt = str(pre_cfg["feature_mode"])
            motion_xy_scale_ckpt = float(pre_cfg["motion_xy_scale"])
            conf_thres_ckpt = float(pre_cfg["conf_thres"])
            max_interp_gap_ckpt = int(pre_cfg["max_interp_gap"])
            missing_mode_ckpt = str(pre_cfg["missing_mode"])
            interp_mode_ckpt = str(pre_cfg["interp_mode"])
            interp_group_ckpt = int(pre_cfg["interp_group"])
            rp_center_mode_ckpt = str(pre_cfg["rp_center_mode"])
            rp_img_w_ckpt = pre_cfg["rp_img_w"]
            rp_img_h_ckpt = pre_cfg["rp_img_h"]
            T_ckpt_raw = int(pre_cfg["T_raw"])
            stride_ckpt_raw = int(pre_cfg["stride_raw"])
            fall_pct_ckpt = float(pre_cfg["fall_pct"])
            label_mode_ckpt = str(pre_cfg["label_mode"])
            min_valid_frac_ckpt = float(pre_cfg["min_valid_frac"])
            add_mask_channel_ckpt = bool(pre_cfg["add_mask_channel"])
            drop_ambig_share_ckpt = float(pre_cfg["drop_ambig_share"])
            drop_ambig_nonfall_only_ckpt = bool(pre_cfg["drop_ambig_nonfall_only"])
            node_features_ckpt = ckpt.get("node_features", None)
            if node_features_ckpt is None and (in_features % 17 == 0):
                node_features_ckpt = in_features // 17
            if node_features_ckpt is not None:
                node_features_ckpt = int(node_features_ckpt)
        else:
            state = ckpt
            T_used = None
            pre_cfg = resolve_preprocess_config(
                ckpt=None,
                is_rf=False,
                use_conf_default=use_conf,
                normalize_default=normalize_cli,
                normalize_mode_default=str(args.normalize_mode),
                add_vel_default=add_vel_cli,
                add_acc_default=add_acc_cli,
                add_global_default=add_global_cli,
                feature_mode_default=feature_mode_cli,
                motion_xy_scale_default=float(args.motion_xy_scale),
                conf_thres_default=float(args.conf_thres),
                max_interp_gap_default=int(args.max_interp_gap),
                missing_mode_default=str(args.missing_mode),
                interp_mode_default=str(args.interp_mode),
                interp_group_default=int(args.interp_group),
                rp_center_mode_default=str(args.rp_center_mode),
                rp_img_w_default=args.rp_img_w,
                rp_img_h_default=args.rp_img_h,
                T_default=int(args.T),
                stride_default=int(args.stride),
                fall_pct_default=float(args.fall_pct),
                label_mode_default=str(args.label_mode),
                min_valid_frac_default=float(args.min_valid_frac),
                add_mask_channel_default=add_mask_channel_cli,
                drop_ambig_share_default=float(args.drop_ambig_share),
                drop_ambig_nonfall_only_default=bool(args.drop_ambig_nonfall_only),
            )
            use_conf_ckpt = bool(pre_cfg["use_conf"])
            normalize_ckpt = bool(pre_cfg["normalize"])
            normalize_mode_ckpt = str(pre_cfg["normalize_mode"])
            add_vel_ckpt = bool(pre_cfg["add_vel"])
            add_acc_ckpt = bool(pre_cfg["add_acc"])
            add_global_ckpt = bool(pre_cfg["add_global"])
            feature_mode_ckpt = str(pre_cfg["feature_mode"])
            motion_xy_scale_ckpt = float(pre_cfg["motion_xy_scale"])
            conf_thres_ckpt = float(pre_cfg["conf_thres"])
            max_interp_gap_ckpt = int(pre_cfg["max_interp_gap"])
            missing_mode_ckpt = str(pre_cfg["missing_mode"])
            interp_mode_ckpt = str(pre_cfg["interp_mode"])
            interp_group_ckpt = int(pre_cfg["interp_group"])
            rp_center_mode_ckpt = str(pre_cfg["rp_center_mode"])
            rp_img_w_ckpt = pre_cfg["rp_img_w"]
            rp_img_h_ckpt = pre_cfg["rp_img_h"]
            T_ckpt_raw = int(pre_cfg["T_raw"])
            stride_ckpt_raw = int(pre_cfg["stride_raw"])
            fall_pct_ckpt = float(pre_cfg["fall_pct"])
            label_mode_ckpt = str(pre_cfg["label_mode"])
            min_valid_frac_ckpt = float(pre_cfg["min_valid_frac"])
            add_mask_channel_ckpt = bool(pre_cfg["add_mask_channel"])
            drop_ambig_share_ckpt = float(pre_cfg["drop_ambig_share"])
            drop_ambig_nonfall_only_ckpt = bool(pre_cfg["drop_ambig_nonfall_only"])
            node_features_ckpt = None

        T_ckpt_raw = max(1, int(T_ckpt_raw))
        stride_ckpt_raw = max(1, int(stride_ckpt_raw))
        T_ckpt = int(T_ckpt_raw)
        stride_ckpt = int(stride_ckpt_raw)
        if frame_step > 1:
            T_ckpt = _ceil_div_pos(T_ckpt_raw, frame_step)
            stride_ckpt = _ceil_div_pos(stride_ckpt_raw, frame_step)
            if (T_ckpt_raw % frame_step) != 0 or (stride_ckpt_raw % frame_step) != 0:
                print(
                    f"[{m}][window][WARN] raw T/stride ({T_ckpt_raw}/{stride_ckpt_raw}) not divisible by k={frame_step}; "
                    "using ceil division."
                )
        print(f"[{m}][window] raw T/stride={T_ckpt_raw}/{stride_ckpt_raw} -> sampled T/stride={T_ckpt}/{stride_ckpt} (k={frame_step})")

        if str(m).lower().strip() == "mlp" and frame_step > 1:
            print("[mlp][WARN] --frame-step/--k > 1 changes T and is incompatible with fixed-size MLP input. Skipping model.")
            continue

        if str(feature_mode_ckpt).lower().strip() == "motion_primary" and (not bool(add_vel_ckpt) or not bool(add_acc_ckpt)):
            raise RuntimeError(
                f"[{m}] checkpoint requests feature_mode=motion_primary but add_vel/add_acc are not both enabled "
                f"(add_vel={bool(add_vel_ckpt)}, add_acc={bool(add_acc_ckpt)})."
            )
        if not math.isfinite(float(motion_xy_scale_ckpt)) or float(motion_xy_scale_ckpt) < 0.0:
            raise RuntimeError(f"[{m}] motion_xy_scale must be finite and >=0. Got {motion_xy_scale_ckpt}")

        # For hybrid window labelling
        extra = {}
        if label_mode_ckpt == "hybrid_center_fallpct":
            extra["fall_ids_0based"] = fall_class_ids_0based
            extra["fall_pct"] = fall_pct_ckpt

        for split_name, split_camera_ids, split_test_npzs in eval_splits:
            split_slug = slug_token(split_name)
            print(
                f"[{m}][{split_name}] evaluating cameras={split_camera_ids} "
                f"on {len(split_test_npzs)} NPZ files"
            )

            # Load windows using ckpt settings
            X_test, y_test_tags, _T_used, test_meta = load_windows_with_source_meta_from_npzs(
                split_test_npzs,
                T=T_ckpt,
                use_conf=use_conf_ckpt,
                normalize=normalize_ckpt,
                normalize_mode=normalize_mode_ckpt,
                add_vel=add_vel_ckpt,
                add_acc=add_acc_ckpt,
                add_global=add_global_ckpt,
                feature_mode=feature_mode_ckpt,
                motion_xy_scale=motion_xy_scale_ckpt,
                conf_thres=conf_thres_ckpt,
                max_interp_gap=max_interp_gap_ckpt,
                missing_mode=missing_mode_ckpt,
                interp_mode=interp_mode_ckpt,
                interp_group=interp_group_ckpt,
                stride=stride_ckpt,
                label_mode=label_mode_ckpt,
                min_valid_frac=min_valid_frac_ckpt,
                add_mask_channel=add_mask_channel_ckpt,
                drop_ambig_share=drop_ambig_share_ckpt,
                drop_ambig_nonfall_only=drop_ambig_nonfall_only_ckpt,
                rp_center_mode=rp_center_mode_ckpt,
                rp_img_w=rp_img_w_ckpt,
                rp_img_h=rp_img_h_ckpt,
                **extra,
                label_convention=label_convention,
            )
            T_used = int(_T_used)
            print(f"[{m}][{split_name}] Window length (T):", T_used)

            y_test = y_test_tags.astype(np.int64, copy=False)
            if int(y_test.shape[0]) != int(np.asarray(test_meta.get("window_source_indices", np.zeros((0,), dtype=np.int64))).shape[0]):
                raise RuntimeError(
                    f"[{m}][{split_name}] Source metadata length mismatch: "
                    f"windows={int(y_test.shape[0])}, meta={int(np.asarray(test_meta.get('window_source_indices', [])).shape[0])}"
                )

            X_test, y_test, test_meta, skipped_test_windows = _drop_activity6_majority_missing_windows(
                X_windows=X_test,
                y_windows=y_test,
                meta=test_meta,
                conf_thres=float(conf_thres_ckpt),
                frame_has_pose_cache=frame_has_pose_cache,
            )
            if skipped_test_windows > 0:
                print(
                    f"[{m}][{split_name}] skipped {skipped_test_windows} Activity6 windows "
                    "with no detected keypoints in the majority of sampled frames"
                )
            if int(y_test.shape[0]) == 0:
                print(
                    f"[{m}][{split_name}][WARN] no evaluation windows remain after "
                    "Activity6 majority-missing-keypoint filtering; skipping split."
                )
                continue

            test_eval_info = _build_transition_eval_info(
                y_window_labels=y_test,
                meta=test_meta,
                label_convention=label_convention,
                fall_class_ids_0based=fall_class_ids_0based,
                frame_label_cache=frame_label_cache,
            )

            if is_rf:
                num_classes_eval = int(NUM_CLASSES_MERGED)
                probs_test = rf_predict_probs(
                    rf_model,
                    X_windows=X_test,
                    feature_mode=rf_feature_mode,
                    num_classes=num_classes_eval,
                    expected_feature_dim=int(rf_feature_dim) if rf_feature_dim is not None else None,
                )
                y_true = y_test
                fall_prob_test = None
                y_pred = probs_test.argmax(axis=1).astype(int)
            else:
                if is_engine:
                    if engine_runner is None:
                        raise RuntimeError(f"[{m}] Internal error: TensorRT engine runner was not initialised.")
                    y_true, probs_test, fall_prob_test = predict_probs_engine(
                        engine_runner,
                        X_windows=X_test,
                        y_true=y_test,
                        batch_size=int(args.batch_size),
                    )
                    if num_classes is None or int(num_classes) <= 0:
                        num_classes = int(probs_test.shape[1])
                    y_pred = probs_test.argmax(axis=1).astype(int)
                else:
                    # Finalise dims for no-metadata checkpoints
                    if not has_state_dict:
                        num_classes = int(y_test.max() + 1)

                    test_ds = WindowTensorDataset(X_test, y_test)

                    sample_X0, _ = test_ds[0]
                    in_features_now = int(sample_X0.shape[-1])
                    if node_features_ckpt is None and (in_features_now % 17 == 0):
                        node_features_ckpt = in_features_now // 17

                    if has_state_dict and in_features is not None and in_features_now != int(in_features):
                        raise RuntimeError(f"[{m}] in_features mismatch: ckpt={in_features}, dataset={in_features_now}")

                    in_features_final = int(in_features) if (has_state_dict and in_features is not None) else in_features_now

                    test_loader = DataLoader(
                        test_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )

                    model = get_model(
                        m,
                        in_features=in_features_final,
                        num_classes=int(num_classes) if (has_state_dict and num_classes is not None) else int(y_test.max() + 1),
                        device=args.device,
                        T_used=T_used,
                        node_features=node_features_ckpt,
                    )

                    if state is None:
                        raise RuntimeError(f"[{m}] Missing model state while evaluating non-engine checkpoint.")
                    model.load_state_dict(state, strict=False)
                    model_params_m = float(count_params_m(model))

                    # Multi-class predictions + optional fall head probability
                    y_true, probs_test, fall_prob_test = predict_probs(model, test_loader, device=args.device)
                    y_pred = probs_test.argmax(axis=1).astype(int)

            if not is_rf:
                if has_state_dict and num_classes is not None:
                    num_classes_eval = int(num_classes)
                else:
                    num_classes_eval = int(probs_test.shape[1])

            test_variant_infos: Dict[str, Dict[str, Any]] = {
                "strict_label_mode": _build_strict_eval_info(
                    y_window_labels=y_test,
                    fall_class_ids_0based=fall_class_ids_0based,
                    reference_info=test_eval_info,
                ),
                "two_label_transition_relaxed": test_eval_info,
            }

            if fall_prob_test is not None:
                p_fall_test = fall_prob_test.astype(np.float32)
                p_fall_source = "fall_head"
            else:
                p_fall_test = p_fall_from_probs(probs_test, fall_class_ids_0based).astype(np.float32)
                p_fall_source = "rf_predict_proba" if is_rf else "activity_softmax"

            skipped_tune_windows = 0
            tune_variant_infos: Optional[Dict[str, Dict[str, Any]]] = None
            p_fall_tune: Optional[np.ndarray] = None
            if str(args.binary_mode).lower() == "threshold" and args.threshold is None and args.tune_subjects is not None:
                tune_subjects = parse_range(args.tune_subjects)
                tune_npzs: List[Path] = []
                for camera_id in split_camera_ids:
                    tune_npzs.extend(
                        Path(p)
                        for p in find_keypoints_npzs_subjects(
                            OUTPUT_ROOT,
                            camera=camera_id,
                            subjects=tune_subjects,
                        )
                    )
                tune_npzs = sorted(set(tune_npzs), key=lambda p: p.as_posix())
                if not tune_npzs:
                    raise RuntimeError(
                        f"No tune NPZs found for split={split_name}. "
                        f"Check OUTPUT_ROOT, camera(s)={split_camera_ids}, and tune subjects."
                    )
                if frame_step > 1:
                    tune_npzs = _materialize_frame_step_npzs(
                        npz_paths=tune_npzs,
                        frame_step=frame_step,
                        cache_dir=npz_cache_dir,
                        cache=npz_subsample_cache,
                    )

                X_tune, y_tune_tags, _, tune_meta = load_windows_with_source_meta_from_npzs(
                    tune_npzs,
                    T=T_ckpt,
                    use_conf=use_conf_ckpt,
                    normalize=normalize_ckpt,
                    normalize_mode=normalize_mode_ckpt,
                    add_vel=add_vel_ckpt,
                    add_acc=add_acc_ckpt,
                    add_global=add_global_ckpt,
                    feature_mode=feature_mode_ckpt,
                    motion_xy_scale=motion_xy_scale_ckpt,
                    conf_thres=conf_thres_ckpt,
                    max_interp_gap=max_interp_gap_ckpt,
                    missing_mode=missing_mode_ckpt,
                    interp_mode=interp_mode_ckpt,
                    interp_group=interp_group_ckpt,
                    stride=stride_ckpt,
                    label_mode=label_mode_ckpt,
                    min_valid_frac=min_valid_frac_ckpt,
                    add_mask_channel=add_mask_channel_ckpt,
                    drop_ambig_share=drop_ambig_share_ckpt,
                    drop_ambig_nonfall_only=drop_ambig_nonfall_only_ckpt,
                    rp_center_mode=rp_center_mode_ckpt,
                    rp_img_w=rp_img_w_ckpt,
                    rp_img_h=rp_img_h_ckpt,
                    **extra,
                    label_convention=label_convention,
                )

                y_tune = y_tune_tags.astype(np.int64, copy=False)
                X_tune, y_tune, tune_meta, skipped_tune_windows = _drop_activity6_majority_missing_windows(
                    X_windows=X_tune,
                    y_windows=y_tune,
                    meta=tune_meta,
                    conf_thres=float(conf_thres_ckpt),
                    frame_has_pose_cache=frame_has_pose_cache,
                )
                if skipped_tune_windows > 0:
                    print(
                        f"[{m}][{split_name}] skipped {skipped_tune_windows} Activity6 tuning windows "
                        "with no detected keypoints in the majority of sampled frames"
                    )
                if int(y_tune.shape[0]) == 0:
                    raise RuntimeError(
                        f"[{m}][{split_name}] No tuning windows remain after "
                        "Activity6 majority-missing-keypoint filtering."
                    )

                tune_transition_eval_info = _build_transition_eval_info(
                    y_window_labels=y_tune,
                    meta=tune_meta,
                    label_convention=label_convention,
                    fall_class_ids_0based=fall_class_ids_0based,
                    frame_label_cache=frame_label_cache,
                )
                tune_variant_infos = {
                    "strict_label_mode": _build_strict_eval_info(
                        y_window_labels=y_tune,
                        fall_class_ids_0based=fall_class_ids_0based,
                        reference_info=tune_transition_eval_info,
                    ),
                    "two_label_transition_relaxed": tune_transition_eval_info,
                }

                if is_rf:
                    probs_tune = rf_predict_probs(
                        rf_model,
                        X_windows=X_tune,
                        feature_mode=rf_feature_mode,
                        num_classes=num_classes_eval,
                        expected_feature_dim=int(rf_feature_dim) if rf_feature_dim is not None else None,
                    )
                    fall_prob_tune = None
                elif is_engine:
                    if engine_runner is None:
                        raise RuntimeError(f"[{m}] Internal error: TensorRT engine runner was not initialised.")
                    _y_tune_true_unused, probs_tune, fall_prob_tune = predict_probs_engine(
                        engine_runner,
                        X_windows=X_tune,
                        y_true=y_tune,
                        batch_size=int(args.batch_size),
                    )
                else:
                    tune_ds = WindowTensorDataset(X_tune, y_tune)
                    tune_loader = DataLoader(
                        tune_ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
                    _y_tune_true_unused, probs_tune, fall_prob_tune = predict_probs(model, tune_loader, device=args.device)

                if fall_prob_tune is not None:
                    p_fall_tune = fall_prob_tune.astype(np.float32)
                else:
                    p_fall_tune = p_fall_from_probs(probs_tune, fall_class_ids_0based).astype(np.float32)

            for variant_key, variant_info in test_variant_infos.items():
                _append_eval_variant_results(
                    variant_key=variant_key,
                    variant_cfg=report_variants[variant_key],
                    variant_info=variant_info,
                    model_name=m,
                    split_name=split_name,
                    split_slug=split_slug,
                    split_camera_ids=split_camera_ids,
                    y_model_true=y_true,
                    y_pred=y_pred,
                    probs_test=probs_test,
                    p_fall_test=p_fall_test,
                    p_fall_source=p_fall_source,
                    num_classes_eval=int(num_classes_eval),
                    test_meta=test_meta,
                    y_window_labels=y_test,
                    output_root=OUTPUT_ROOT,
                    ckpt_path=ckpt_path,
                    test_subjects=test_subjects,
                    frame_step=int(frame_step),
                    T_ckpt_raw=int(T_ckpt_raw),
                    stride_ckpt_raw=int(stride_ckpt_raw),
                    T_ckpt=int(T_ckpt),
                    stride_ckpt=int(stride_ckpt),
                    normalize_mode_ckpt=str(normalize_mode_ckpt),
                    missing_mode_ckpt=str(missing_mode_ckpt),
                    interp_mode_ckpt=str(interp_mode_ckpt),
                    interp_group_ckpt=int(interp_group_ckpt),
                    rp_center_mode_ckpt=str(rp_center_mode_ckpt),
                    rp_img_w_ckpt=rp_img_w_ckpt,
                    rp_img_h_ckpt=rp_img_h_ckpt,
                    feature_mode_ckpt=str(feature_mode_ckpt),
                    motion_xy_scale_ckpt=float(motion_xy_scale_ckpt),
                    model_params_m=float(model_params_m),
                    binary_mode=str(args.binary_mode),
                    threshold_arg=args.threshold,
                    beta=float(args.beta),
                    tune_subjects_arg=args.tune_subjects,
                    fall_class_ids_0based=fall_class_ids_0based,
                    skipped_test_windows=int(skipped_test_windows),
                    skipped_tune_windows=int(skipped_tune_windows),
                    tune_variant_info=tune_variant_infos.get(variant_key) if tune_variant_infos is not None else None,
                    p_fall_tune=p_fall_tune,
                )

    if not any(variant_cfg["summary_rows"] for variant_cfg in report_variants.values()):
        raise RuntimeError("No models were evaluated. Check --models/--all and --frame-step compatibility.")

    for variant_key, variant_cfg in report_variants.items():
        if not variant_cfg["summary_rows"]:
            continue

        variant_out_dir = Path(variant_cfg["out_dir"])
        variant_plots_dir = Path(variant_cfg["plots_dir"])
        summary_df = pd.DataFrame(variant_cfg["summary_rows"]).sort_values(
            ["eval_split", "macro_f1"],
            ascending=[True, False],
        ).reset_index(drop=True)
        f1_long = pd.DataFrame(variant_cfg["f1_rows"]).sort_values(["eval_split", "model", "class_id"]).reset_index(drop=True)
        overall_df = pd.DataFrame(variant_cfg["overall_rows"]).sort_values(["eval_split", "model"]).reset_index(drop=True).round(3)

        summary_csv = variant_out_dir / "metrics_summary.csv"
        f1_csv = variant_out_dir / "f1_per_class.csv"
        summary_df.to_csv(summary_csv, index=False)
        f1_long.to_csv(f1_csv, index=False)
        misclassification_paths = write_misclassification_report(pd.DataFrame(variant_cfg["misclassified_rows"]), variant_out_dir)

        make_plots(summary_df, variant_plots_dir)
        report_path = variant_out_dir / "report.html"
        make_html_report(summary_df, overall_df, f1_long, variant_plots_dir, report_path, model_list)

        print(f"\nOverall metrics (%) [{variant_key}]:")
        print(overall_df.to_string(index=False))
        print(f"[{variant_key}] {variant_cfg['description']}")
        print(f"Saved: {summary_csv}")
        print(f"Saved: {f1_csv}")
        for report_file in misclassification_paths:
            print(f"Saved: {report_file}")
        print(f"Saved: {report_path}")
        print(f"Plots in: {variant_plots_dir.as_posix()}")


if __name__ == "__main__":
    main()
