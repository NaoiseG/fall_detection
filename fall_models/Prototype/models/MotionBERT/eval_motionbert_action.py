#!/usr/bin/env python3
"""
eval_motionbert_action.py

Evaluate a trained MotionBERT ActionNet checkpoint on UP-Fall 2D keypoints,
mirroring the UX of models/eval_models.py (timestamped output folder, CSVs,
HTML report, plots), but compatible with MotionBERT's action training pipeline.

This script is intended to live at the MotionBERT repo root (same level as lib/).

Key points:
- Uses the same model construction as train_action.py (ActionNet + load_backbone)
- Uses the same dataset class as train_action.py (NTURGBD over data/action/*.pkl)
- Subject selection is applied by filtering the evaluation split *inside the pkl*
  based on the sample "frame_dir" field (UP-Fall entries contain strings like
  "Subject10_Activity10_Trial1_Cam1_win0"). This avoids changing the conversion
  pipeline or the sample format.

Outputs (under --out-dir/<timestamp>__motionbert_action__...):
- metrics_summary.csv
- f1_per_class.csv
- report.html
- plots/confusion_matrix.png
- plots/f1_per_class.png
(+ optional binary fall/no-fall plots if --fall-class-ids is provided)

Example (evaluate best checkpoint on subjects 1-5):
python eval_motionbert_action.py \
  --config configs/action/MB_ft_UPFall_xsub_LITE.yaml \
  --checkpoint checkpoint/action/FT_MB_lite_MB_ft_UPFall_xsub/best_epoch.bin \
  --subjects 1-5 \
  --out-dir eval_outputs \
  --batch-size 64 \
  --num-workers 0 \
  --device cuda

How to confirm success:
- report.html exists in the created output folder
- metrics_summary.csv exists and has top1/top5/loss
- plots/confusion_matrix.png exists
- stdout prints a final summary line including Loss, Acc@1, Acc@5

Notes for HPC:
- Default num_workers=0 (safe on shared filesystems)
- Deterministic seeds and CUDNN deterministic mode enabled
- Progress prints flush to stdout (Slurm friendly)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Optional dependencies (used for metrics + plots)
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
    accuracy_score,
)

import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

_TS_DIR_FMT = "%Y-%m-%d_%H-%M-%S_%f"


def _seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _slug(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
    return s[:max_len]


def _parse_subjects(spec: str) -> List[int]:
    """
    Parse subject list specs like:
      "1-5" or "1-3,7,9-10" (commas + ranges).
    Returns a sorted unique list of ints.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty --subjects spec.")

    out: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i, b_i = int(a), int(b)
            lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


_SUBJECT_RE = re.compile(r"(?:^|_)Subject(\d+)(?:_|$)", re.IGNORECASE)


def _subject_from_frame_dir(frame_dir: str) -> Optional[int]:
    """
    Extract subject id from frame_dir (UP-Fall uses 'Subject10_Activity...').
    """
    m = _SUBJECT_RE.search(frame_dir)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _torch_device(device: str) -> torch.device:
    # Allow "cuda", "cuda:0", "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device)


def _load_class_names_from_label_map_json(label_map_path: Path, num_classes: int) -> Optional[List[str]]:
    """
    Best-effort mapping of class_id -> human-friendly name for plots/CSVs.

    Supports both label-map shapes used in this repo:
      A) prepare_motionbert_dataset.py style:
         {"label_map": {"0":"Fall", "1":"Walking", ...}, "meta": {...}}
      B) legacy / raw->id style (e.g. {"label_map": {"1":0, ...}}) but with:
         meta.merge_fall.new_class_names: {"0":"Fall", "1":"Walking", ...}
    """
    if not label_map_path.exists():
        return None
    try:
        obj = json.loads(label_map_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    def _names_from_id_map(m: Dict[str, object]) -> Optional[List[str]]:
        out: List[str] = []
        for i in range(int(num_classes)):
            v = m.get(str(i), None)
            if not isinstance(v, str) or not v.strip():
                return None
            out.append(v.strip())
        return out

    lm = obj.get("label_map", None)
    if isinstance(lm, dict):
        names = _names_from_id_map(lm)
        if names is not None:
            return names

    meta = obj.get("meta", None)
    if isinstance(meta, dict):
        merge = meta.get("merge_fall", None)
        if isinstance(merge, dict):
            nc = merge.get("new_class_names", None)
            if isinstance(nc, dict):
                names = _names_from_id_map(nc)
                if names is not None:
                    return names

    return None


def _clean_state_dict_for_model(state: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Flexibly handle DataParallel 'module.' prefixes.
    """
    state_keys = list(state.keys())
    has_module_prefix = any(k.startswith("module.") for k in state_keys)
    model_is_dp = isinstance(model, nn.DataParallel)

    if has_module_prefix and not model_is_dp:
        # Strip "module."
        return {k.replace("module.", "", 1): v for k, v in state.items()}

    if (not has_module_prefix) and model_is_dp:
        # Add "module."
        return {("module." + k): v for k, v in state.items()}

    return state


# -----------------------------------------------------------------------------
# Pickle loading (numpy 2.0 -> 1.x compatibility)
# -----------------------------------------------------------------------------

def _install_numpy_pickle_compat() -> None:
    """
    Some MotionBERT action .pkl files can be created under NumPy 2.x and then fail
    to unpickle under NumPy 1.x due to module path changes (numpy._core).
    This creates aliases so unpickling works in both directions.
    """
    import numpy.core as npcore  # type: ignore

    sys.modules.setdefault("numpy._core", npcore)
    # These are commonly referenced in pickles
    sys.modules.setdefault("numpy._core.multiarray", npcore.multiarray)
    sys.modules.setdefault("numpy._core._multiarray_umath", npcore._multiarray_umath)


def _load_action_pkl(pkl_path: Path) -> Dict:
    _install_numpy_pickle_compat()
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "split" not in obj or "annotations" not in obj:
        raise ValueError(f"Unexpected pkl structure in {pkl_path.as_posix()}: expected dict with split+annotations.")
    return obj


def _write_action_pkl(obj: Dict, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    with out_path.open("wb") as f:
        pickle.dump(obj, f, protocol=4)


# -----------------------------------------------------------------------------
# Filtering helpers
# -----------------------------------------------------------------------------

@dataclass
class FilterResult:
    filtered_pkl_path: Path
    split_key: str
    n_split_before: int
    n_split_after: int
    subjects_found: List[int]


def _choose_eval_split_key(split_dict: Dict[str, Sequence[str]], base: str) -> str:
    """
    Choose which split key to evaluate.
    Prefers: <base>_val, then <base>_test, then <base>_train.
    """
    for suffix in ("val", "test", "train"):
        k = f"{base}_{suffix}"
        if k in split_dict:
            return k
    # If base itself is a key, use it.
    if base in split_dict:
        return base
    raise KeyError(f"Could not find an eval split for base='{base}'. Available keys: {sorted(split_dict.keys())}")


def _filter_pkl_by_subjects(
    pkl_path: Path,
    chosen_subjects: Set[int],
    base_split: str,
    out_path: Path,
) -> FilterResult:
    """
    Filter an action pkl so that its chosen split contains only samples from
    `chosen_subjects`. Subject is parsed from each sample's frame_dir.

    Returns metadata and writes a filtered pkl to out_path.
    """
    data = _load_action_pkl(pkl_path)
    split = data["split"]
    ann = data["annotations"]

    split_key = _choose_eval_split_key(split, base_split)
    frame_dirs: List[str] = list(split[split_key])
    n_before = len(frame_dirs)

    subjects_in_split: List[int] = []
    filtered_frame_dirs: List[str] = []
    for fd in frame_dirs:
        sid = _subject_from_frame_dir(str(fd))
        if sid is not None:
            subjects_in_split.append(sid)
        if sid is not None and sid in chosen_subjects:
            filtered_frame_dirs.append(fd)

    subjects_found = sorted(set(subjects_in_split))

    if len(filtered_frame_dirs) == 0:
        raise RuntimeError(
            f"No samples matched subjects={sorted(chosen_subjects)} in split '{split_key}'. "
            f"Subjects present in that split: {subjects_found[:50]}{'...' if len(subjects_found)>50 else ''}"
        )

    keep = set(filtered_frame_dirs)

    filtered_ann = [a for a in ann if a.get("frame_dir") in keep]
    if len(filtered_ann) != len(filtered_frame_dirs):
        # Not always a bug (split may be a subset), but usually indicates mismatch.
        missing = len(filtered_frame_dirs) - len(filtered_ann)
        print(
            f"WARNING: split list has {len(filtered_frame_dirs)} entries but only {len(filtered_ann)} annotations matched "
            f"by frame_dir (missing={missing}). Proceeding anyway.",
            flush=True,
        )

    new_obj = {
        "split": {split_key: filtered_frame_dirs},
        "annotations": filtered_ann,
    }
    _write_action_pkl(new_obj, out_path)

    return FilterResult(
        filtered_pkl_path=out_path,
        split_key=split_key,
        n_split_before=n_before,
        n_split_after=len(filtered_frame_dirs),
        subjects_found=subjects_found,
    )


# -----------------------------------------------------------------------------
# Evaluation + reporting
# -----------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    n_samples: int
    loss: float
    top1: float
    top5: float
    macro_f1: float
    weighted_f1: float


def _topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> List[float]:
    """
    Compute top-k accuracy in percent. Safe when num_classes < k.
    """
    with torch.no_grad():
        maxk = min(max(topk), logits.size(1))
        _, pred = logits.topk(maxk, 1, True, True)  # (N, maxk)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res: List[float] = []
        for k in topk:
            k2 = min(k, logits.size(1))
            correct_k = correct[:k2].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / logits.size(0)).item()))
        return res


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    print_freq: int = 100,
) -> Tuple[EvalMetrics, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    losses: List[float] = []
    top1_list: List[float] = []
    top5_list: List[float] = []

    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    end = time.time()
    with torch.no_grad():
        for idx, (batch_input, batch_gt) in enumerate(loader):
            batch_input = batch_input.to(device, non_blocking=True)
            batch_gt = batch_gt.to(device, non_blocking=True)

            logits = model(batch_input)  # (N, C)
            loss = criterion(logits, batch_gt)

            acc1, acc5 = _topk_accuracy(logits, batch_gt, topk=(1, 5))
            losses.append(float(loss.item()))
            top1_list.append(acc1)
            top5_list.append(acc5)

            preds = logits.argmax(dim=1)
            y_true_all.extend(batch_gt.detach().cpu().numpy().astype(int).tolist())
            y_pred_all.extend(preds.detach().cpu().numpy().astype(int).tolist())

            if (idx + 1) % print_freq == 0 or (idx + 1) == len(loader):
                elapsed = time.time() - end
                end = time.time()
                print(
                    f"[eval] batch {idx+1:05d}/{len(loader)} | "
                    f"loss {np.mean(losses):.4f} | "
                    f"acc@1 {np.mean(top1_list):.2f} | "
                    f"acc@5 {np.mean(top5_list):.2f} | "
                    f"{elapsed:.2f}s",
                    flush=True,
                )

    y_true_np = np.asarray(y_true_all, dtype=int)
    y_pred_np = np.asarray(y_pred_all, dtype=int)

    macro_f1 = float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0))

    metrics = EvalMetrics(
        n_samples=int(len(y_true_np)),
        loss=float(np.mean(losses) if losses else 0.0),
        top1=float(np.mean(top1_list) if top1_list else 0.0),
        top5=float(np.mean(top5_list) if top5_list else 0.0),
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
    )
    return metrics, y_true_np, y_pred_np


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path, normalize: bool = False) -> None:
    if normalize:
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-9
        cm = cm / row_sums

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Add text (lightweight; skip if too large)
    if len(class_names) <= 30:
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() * (0.6 if normalize else 0.5)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_f1_bar(per_class_df: pd.DataFrame, out_path: Path) -> None:
    # Expect columns: class_name, f1
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    ax.bar(per_class_df["class_name"].astype(str).tolist(), per_class_df["f1"].astype(float).tolist())
    ax.set_title("F1 per class")
    ax.set_ylabel("F1")
    ax.set_xticklabels(per_class_df["class_name"].astype(str).tolist(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _df_to_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, classes="tbl")


def _make_html_report(
    summary_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    out_dir: Path,
    plots_dir: Path,
    out_path: Path,
    extra_html: str = "",
) -> None:
    css = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#111;}
    h1,h2{margin:0.2em 0;}
    .tbl{border-collapse:collapse;width:100%;margin:12px 0;}
    .tbl th,.tbl td{border:1px solid #ddd;padding:8px;font-size:14px;}
    .tbl th{background:#f6f6f6;text-align:left;}
    img{max-width:100%;height:auto;border:1px solid #eee;border-radius:10px;padding:8px;background:#fff;}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}
    code{background:#f6f6f6;padding:2px 6px;border-radius:6px;}
    .meta{font-size:13px;color:#444;margin-top:16px;}
    """

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>MotionBERT Action Evaluation Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>MotionBERT Action Evaluation Report</h1>

  <h2>Summary</h2>
  {_df_to_html(summary_df)}

  <div class="grid">
    <div>
      <h2>Confusion matrix</h2>
      <img src="{plots_dir.name}/confusion_matrix.png" alt="Confusion matrix"/>
    </div>
    <div>
      <h2>F1 per class</h2>
      <img src="{plots_dir.name}/f1_per_class.png" alt="F1 per class"/>
    </div>
  </div>

  <h2>Per-class metrics</h2>
  {_df_to_html(per_class_df)}

  {extra_html}

  <div class="meta">
    Generated by <code>eval_motionbert_action.py</code>. Output folder: <code>{out_dir.name}</code>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, fall_ids: Set[int]) -> Dict[str, float]:
    """
    Binary metrics (fall vs no-fall) by collapsing multi-class labels.
    """
    y_true_bin = np.asarray([1 if int(y) in fall_ids else 0 for y in y_true], dtype=int)
    y_pred_bin = np.asarray([1 if int(y) in fall_ids else 0 for y in y_pred], dtype=int)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, labels=[0, 1], average=None, zero_division=0
    )
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    return {
        "bin_precision_avg": float(np.mean(pr)),
        "bin_recall_avg": float(np.mean(rc)),
        "bin_f1_avg": float(np.mean(f1)),
        "bin_precision_fall": float(pr[1]),
        "bin_recall_fall": float(rc[1]),
        "bin_f1_fall": float(f1[1]),
        "bin_precision_no_fall": float(pr[0]),
        "bin_recall_no_fall": float(rc[0]),
        "bin_f1_no_fall": float(f1[0]),
        "bin_cm_tn": float(cm[0, 0]),
        "bin_cm_fp": float(cm[0, 1]),
        "bin_cm_fn": float(cm[1, 0]),
        "bin_cm_tp": float(cm[1, 1]),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a MotionBERT action checkpoint on chosen UP-Fall subjects.")
    parser.add_argument("--config", type=str, required=True, help="MotionBERT YAML config used in training.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file (best_epoch.bin or latest_epoch.bin).")
    parser.add_argument("--subjects", type=str, required=True, help="Subject spec like '1-5' or '1-3,7,9-10'.")
    parser.add_argument("--out-dir", type=str, default="eval_outputs", help="Base output directory (timestamped subfolder created).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64).")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers (default: 0).")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device string (cuda, cuda:0, cpu).")
    parser.add_argument("--print-freq", type=int, default=100, help="Print progress every N batches (default: 100).")

    # Optional: binary fall vs no-fall metrics
    parser.add_argument(
        "--fall-class-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of class ids to treat as 'fall' for binary metrics (IDs must match the pkl label space).",
    )

    # Optional overrides
    parser.add_argument("--data-pkl", type=str, default=None, help="Override dataset pkl path (default: data/action/<dataset>.pkl).")
    parser.add_argument("--split-base", type=str, default=None, help="Override base split name (default: config data_split, e.g. xsub).")

    args_cli = parser.parse_args()

    # Make MotionBERT imports work when script is at repo root.
    repo_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo_root))

    from lib.utils.tools import get_config
    from lib.utils.learning import load_backbone  # noqa
    from lib.data.dataset_action import NTURGBD  # noqa
    from lib.model.model_action import ActionNet  # noqa

    _seed_everything(0)

    cfg = get_config(args_cli.config)
    chosen_subjects = set(_parse_subjects(args_cli.subjects))
    device = _torch_device(args_cli.device)

    # Extract config values used in train_action.py
    dataset_name = getattr(cfg, "dataset", None)
    if dataset_name is None:
        raise RuntimeError("Config missing 'dataset' (expected e.g. 'upfall').")

    base_split = args_cli.split_base or getattr(cfg, "data_split", None)
    if base_split is None:
        raise RuntimeError("Config missing 'data_split' (expected e.g. 'xsub').")

    clip_len = int(getattr(cfg, "clip_len", 0))
    if clip_len <= 0:
        raise RuntimeError("Config missing/invalid 'clip_len'.")

    action_classes = int(getattr(cfg, "action_classes", 0))
    if action_classes <= 1:
        raise RuntimeError("Config missing/invalid 'action_classes'.")

    scale_range_test = getattr(cfg, "scale_range_test", None)

    # Resolve original pkl path
    if args_cli.data_pkl is not None:
        src_pkl = Path(args_cli.data_pkl).expanduser().resolve()
    else:
        src_pkl = repo_root / "data" / "action" / f"{dataset_name}.pkl"
    if not src_pkl.exists():
        raise FileNotFoundError(f"Dataset pkl not found: {src_pkl.as_posix()}")

    # Optional: class names for plots/CSVs (if a label-map JSON exists).
    # Falls back to numeric strings if the file doesn't exist or doesn't match num_classes.
    label_map_json = repo_root / "data" / "action" / f"{dataset_name}_label_map.json"
    class_names = _load_class_names_from_label_map_json(label_map_json, action_classes)
    if class_names is None:
        class_names = [str(i) for i in range(action_classes)]

    # One unique output folder per eval run (like eval_models.py)
    ts = datetime.now().strftime(_TS_DIR_FMT)
    tag = _slug(f"motionbert_action__{dataset_name}__{base_split}__subjects_{args_cli.subjects}")
    out_dir = Path(args_cli.out_dir).expanduser().resolve() / f"{ts}__{tag}"
    plots_dir = out_dir / "plots"
    _ensure_dir(out_dir)
    _ensure_dir(plots_dir)

    print("Eval output dir:", out_dir.as_posix(), flush=True)

    # Create a filtered pkl (Approach A via split+frame_dir filtering; avoids altering sample format)
    filtered_pkl = out_dir / f"filtered_{dataset_name}_{base_split}_{_slug(args_cli.subjects)}.pkl"
    fr = _filter_pkl_by_subjects(
        pkl_path=src_pkl,
        chosen_subjects=chosen_subjects,
        base_split=base_split,
        out_path=filtered_pkl,
    )
    print(
        f"Split '{fr.split_key}': {fr.n_split_after}/{fr.n_split_before} samples kept "
        f"for subjects={sorted(chosen_subjects)}. Subjects present in split: {fr.subjects_found}",
        flush=True,
    )

    # Dataset + loader (match training evaluate settings)
    ds_kwargs = dict(
        data_path=str(fr.filtered_pkl_path),
        data_split=fr.split_key,
        n_frames=clip_len,
        random_move=False,
    )
    if scale_range_test is not None:
        ds_kwargs["scale_range"] = scale_range_test

    eval_ds = NTURGBD(**ds_kwargs)
    loader_kwargs = dict(
        batch_size=int(args_cli.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args_cli.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    if int(args_cli.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(eval_ds, **loader_kwargs)

    # Build model (same as train_action.py)
    model_backbone = load_backbone(cfg)
    model = ActionNet(
        backbone=model_backbone,
        dim_rep=getattr(cfg, "dim_rep", 512),
        num_classes=action_classes,
        dropout_ratio=getattr(cfg, "dropout_ratio", 0.0),
        version=getattr(cfg, "model_version", "v1"),
        hidden_dim=getattr(cfg, "hidden_dim", 1024),
        num_joints=getattr(cfg, "num_joints", 17),
    )

    # Wrap to match checkpoint format if needed
    use_dp = (device.type == "cuda" and torch.cuda.device_count() > 1)
    if use_dp:
        model = nn.DataParallel(model)
    model = model.to(device)

    ckpt_path = Path(args_cli.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.as_posix()}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = _clean_state_dict_for_model(state, model)
    model.load_state_dict(state, strict=True)

    # Log shape
    try:
        sample_x, sample_y = next(iter(loader))
        print(f"Sample batch_input shape: {tuple(sample_x.shape)} | batch_gt shape: {tuple(sample_y.shape)}", flush=True)
    except Exception as e:
        print(f"WARNING: could not read first batch for shape logging: {e}", flush=True)

    # Evaluate
    metrics, y_true, y_pred = _evaluate(model, loader, device=device, print_freq=int(args_cli.print_freq))

    # Compute per-class stats
    labels = list(range(action_classes))
    pr, rc, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    per_class_df = pd.DataFrame({
        "class_id": labels,
        "class_name": [class_names[int(i)] if 0 <= int(i) < len(class_names) else str(i) for i in labels],
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "support": support,
    })

    # Summary CSV
    summary = {
        "dataset": dataset_name,
        "split_key": fr.split_key,
        "subjects": args_cli.subjects,
        "n_samples": metrics.n_samples,
        "loss": metrics.loss,
        "acc_top1": metrics.top1,
        "acc_top5": metrics.top5,
        "macro_f1": metrics.macro_f1,
        "weighted_f1": metrics.weighted_f1,
        "checkpoint": ckpt_path.as_posix(),
        "config": Path(args_cli.config).expanduser().resolve().as_posix(),
        "device": str(device),
        "batch_size": int(args_cli.batch_size),
        "num_workers": int(args_cli.num_workers),
    }

    extra_html = ""
    if args_cli.fall_class_ids:
        fall_ids = set(int(x) for x in args_cli.fall_class_ids)
        summary.update(_binary_metrics(y_true, y_pred, fall_ids))
        extra_html = f"""
  <h2>Binary fall vs no-fall</h2>
  <p>Fall class ids: <code>{sorted(fall_ids)}</code></p>
"""

    summary_df = pd.DataFrame([summary])

    # Save CSVs
    summary_csv = out_dir / "metrics_summary.csv"
    per_class_csv = out_dir / "f1_per_class.csv"
    summary_df.to_csv(summary_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)

    # Plots
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    _plot_confusion_matrix(cm, class_names=[class_names[int(i)] if 0 <= int(i) < len(class_names) else str(i) for i in labels], out_path=plots_dir / "confusion_matrix.png", normalize=False)
    _plot_f1_bar(per_class_df, out_path=plots_dir / "f1_per_class.png")

    # Report
    report_path = out_dir / "report.html"
    _make_html_report(summary_df, per_class_df, out_dir=out_dir, plots_dir=plots_dir, out_path=report_path, extra_html=extra_html)

    # Print final summary line (matches MotionBERT validate output style)
    print(
        f"Loss {metrics.loss:.4f} \tAcc@1 {metrics.top1:.3f} \tAcc@5 {metrics.top5:.3f} \tMacroF1 {metrics.macro_f1:.3f}",
        flush=True,
    )
    print(f"Saved: {summary_csv.as_posix()}", flush=True)
    print(f"Saved: {per_class_csv.as_posix()}", flush=True)
    print(f"Saved: {report_path.as_posix()}", flush=True)
    print(f"Plots in: {plots_dir.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
