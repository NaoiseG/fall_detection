#!/usr/bin/env python3
"""eval_models.py

Evaluate one or more trained models on a chosen set of UP-Fall subjects, using the
same NPZ -> window loading pipeline as training (dataset.py).

Outputs (in --out-dir):
- metrics_summary.csv   : per-model summary including
    * binary sensitivity (recall) and precision macro-averaged over fall/no-fall
    * per-class (fall, no-fall) precision/recall
    * multi-class macro F1
- f1_per_class.csv      : per-model, per-class F1 (multi-class)
- report.html           : tables + plots
- plots/*.png           : quick comparison plots

Run (from project root):
python -m models.eval_models --models tcn lstm gru \
  --camera 1 \
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
import re

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support, f1_score, precision_recall_curve, confusion_matrix

# Same dataset pipeline as training
from dataset import (
    find_keypoints_npzs_subjects,
    load_windows_from_npzs,
    WindowTensorDataset,
    detect_label_convention_from_npzs,
    get_new_label_names,
    detect_label_convention as _detect_label_convention,
    remap_label as _remap_label,
    get_fall_merge_set as _get_fall_merge_set,
)

# Same model definitions as training
from .tcn.simple_tcn import TCNBaseline
from .lstm.simple_lstm import LSTMBaseline
from .gru.simple_gru import GRUBaseline
from .gcn.simple_gcn import GCNBaseline
from .mlp.simple_mlp import MLPBaseline
from .stgcn.simple_stgcn import STGCNBaseline

# NEW: CNN + LSTM two-head model
from .cnnlstm.cnn_lstm_two_head import CNNLSTMTwoHead


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


def collapse_to_binary(y: np.ndarray, fall_class_ids_0based: List[int]) -> np.ndarray:
    fall = set(int(x) for x in fall_class_ids_0based)
    return np.array([1 if int(v) in fall else 0 for v in y], dtype=int)


def p_fall_from_probs(probs: np.ndarray, fall_class_ids_0based: List[int]) -> np.ndarray:
    idx = [int(i) for i in fall_class_ids_0based if 0 <= int(i) < probs.shape[1]]
    if len(idx) == 0:
        return np.zeros((probs.shape[0],), dtype=np.float32)
    return probs[:, idx].sum(axis=1)


def pick_threshold_fbeta(y_true_bin: np.ndarray, p_fall: np.ndarray, beta: float = 2.0) -> Tuple[float, float, float, float]:
    """
    Returns: (best_threshold, precision_at_best, recall_at_best, fbeta_at_best)
    Uses sklearn precision_recall_curve.
    """
    prec, rec, th = precision_recall_curve(y_true_bin, p_fall)
    # th has len = len(prec) - 1
    denom = (beta * beta * prec + rec + 1e-9)
    fbeta = (1.0 + beta * beta) * (prec * rec) / denom
    if th.size == 0:
        return 0.5, float(prec[-1] if prec.size else 0.0), float(rec[-1] if rec.size else 0.0), 0.0
    best_i = int(np.nanargmax(fbeta[:-1]))
    return float(th[best_i]), float(prec[best_i]), float(rec[best_i]), float(fbeta[best_i])


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

    plt.figure()
    plt.bar(summary_df["model"], summary_df["macro_f1"])
    plt.xticks(rotation=30, ha="right")
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
    plt.xticks(x, summary_df["model"], rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Binary fall/no-fall metrics (macro-averaged)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "binary_metrics.png", dpi=200)
    plt.close()


def make_html_report(summary_df: pd.DataFrame, f1_long: pd.DataFrame, plots_dir: Path, out_path: Path, model_list: List[str]):
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
    ALL_MODELS = ["tcn", "lstm", "gru", "gcn", "mlp", "stgcn", "cnnlstm"]

    parser = argparse.ArgumentParser(description="Evaluate trained models on UP-Fall windowed pose tensors.")
    parser.add_argument("--models", nargs="+", default=None, help="Models to evaluate, e.g. --models tcn lstm")
    parser.add_argument("--all", action="store_true", help="Evaluate all models (overrides --models).")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument("--test-subjects", type=str, default="1-1", help="Test subject range like '1-5'")
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
        help="Override weights filename. If omitted uses '<model>_best.pt'.",
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
    parser.add_argument("--add-vel", type=int, default=1, help="Add velocity channels vx, vy (0/1).")
    parser.add_argument("--add-acc", type=int, default=1, help="Add acceleration channels ax, ay (0/1).")
    parser.add_argument("--add-global", type=int, default=1, help="Add global features (0/1).")
    parser.add_argument("--conf-thres", type=float, default=0.2, help="Conf threshold for missing joints.")
    parser.add_argument("--max-interp-gap", type=int, default=5, help="Max gap (frames) for interpolation.")
    parser.add_argument("--T", type=int, default=64, help="Sliding window length T.")
    parser.add_argument("--stride", type=int, default=16, help="Sliding window stride.")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="center",
        choices=["center", "majority", "hybrid_center_fallpct"],
    )
    parser.add_argument("--min-valid-frac", type=float, default=0.3)
    parser.add_argument("--add-mask-channel", type=int, default=1)
    args = parser.parse_args()

    normalize_cli = bool(args.normalize)
    add_vel_cli = bool(args.add_vel)
    add_acc_cli = bool(args.add_acc)
    add_global_cli = bool(args.add_global)
    add_mask_channel_cli = bool(args.add_mask_channel)

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

    # Load test set using the SAME NPZ->windows pipeline
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints/outputs_npz")
    test_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=args.camera, subjects=test_subjects)
    if not test_npzs:
        raise RuntimeError("No test NPZs found. Check OUTPUT_ROOT, camera, and test subjects.")

    # ---- Detect raw label convention once (1-11 vs 0-10), then keep it consistent ----
    label_convention, label_stats = detect_label_convention_from_npzs(test_npzs)
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
    plots_dir = out_dir / "plots"
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    print("Eval output dir:", out_dir.as_posix())

    summary_rows: List[Dict[str, object]] = []
    f1_rows: List[Dict[str, object]] = []

    ckpt_root = Path(args.ckpt_root)
    ckpt_overrides = parse_ckpt_overrides(args.ckpt)

    for m in model_list:
        weights_name = args.weights_name or f"{m}_best.pt"

        model_dir = ckpt_root / m
        run_dir = resolve_run_dir(model_dir, ckpt_overrides.get(m))
        ckpt_path = run_dir / weights_name

        print(f"[{m}] Using run folder: {run_dir.name}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Weights not found for {m}: {ckpt_path.as_posix()}")

        ckpt = torch_load_safe(ckpt_path, map_location="cpu")

        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
            T_used = int(ckpt["T_used"])
            in_features = int(ckpt["in_features"])
            num_classes = int(ckpt["num_classes"])
            use_conf_ckpt = bool(ckpt.get("use_conf", True))

            normalize_ckpt = bool(ckpt.get("normalize", normalize_cli))
            add_vel_ckpt = bool(ckpt.get("add_vel", add_vel_cli))
            add_acc_ckpt = bool(ckpt.get("add_acc", add_acc_cli))
            add_global_ckpt = bool(ckpt.get("add_global", add_global_cli))
            conf_thres_ckpt = float(ckpt.get("conf_thres", args.conf_thres))
            max_interp_gap_ckpt = int(ckpt.get("max_interp_gap", args.max_interp_gap))
            T_ckpt = int(ckpt.get("T", ckpt.get("T_used", args.T)))  # support both keys
            stride_ckpt = int(ckpt.get("stride", args.stride))

            fall_pct_ckpt = float(ckpt.get("fall_pct", args.fall_pct))
            label_mode_ckpt = str(ckpt.get("label_mode", args.label_mode))
            min_valid_frac_ckpt = float(ckpt.get("min_valid_frac", args.min_valid_frac))
            add_mask_channel_ckpt = bool(ckpt.get("add_mask_channel", add_mask_channel_cli))
            node_features_ckpt = ckpt.get("node_features", None)
            if node_features_ckpt is None and (in_features % 17 == 0):
                node_features_ckpt = in_features // 17
            if node_features_ckpt is not None:
                node_features_ckpt = int(node_features_ckpt)
        else:
            state = ckpt
            T_used = None
            use_conf_ckpt = use_conf
            normalize_ckpt = normalize_cli
            add_vel_ckpt = add_vel_cli
            add_acc_ckpt = add_acc_cli
            add_global_ckpt = add_global_cli
            conf_thres_ckpt = float(args.conf_thres)
            max_interp_gap_ckpt = int(args.max_interp_gap)
            T_ckpt = int(args.T)
            stride_ckpt = int(args.stride)

            fall_pct_ckpt = float(args.fall_pct)
            label_mode_ckpt = str(args.label_mode)
            min_valid_frac_ckpt = float(args.min_valid_frac)
            add_mask_channel_ckpt = add_mask_channel_cli
            node_features_ckpt = None

        # For hybrid window labelling
        extra = {}
        if label_mode_ckpt == "hybrid_center_fallpct":
            extra["fall_ids_0based"] = fall_class_ids_0based
            extra["fall_pct"] = fall_pct_ckpt

        # Load windows using ckpt settings
        if T_used is None:
            X_test, y_test_tags, _T_used = load_windows_from_npzs(
                test_npzs,
                T=T_ckpt,
                use_conf=use_conf_ckpt,
                normalize=normalize_ckpt,
                add_vel=add_vel_ckpt,
                add_acc=add_acc_ckpt,
                add_global=add_global_ckpt,
                conf_thres=conf_thres_ckpt,
                max_interp_gap=max_interp_gap_ckpt,
                stride=stride_ckpt,
                label_mode=label_mode_ckpt,
                min_valid_frac=min_valid_frac_ckpt,
                add_mask_channel=add_mask_channel_ckpt,
                **extra,
                label_convention=label_convention,
            )
            T_used = int(_T_used)
        else:
            X_test, y_test_tags, _T_used = load_windows_from_npzs(
                test_npzs,
                T=T_ckpt,
                use_conf=use_conf_ckpt,
                normalize=normalize_ckpt,
                add_vel=add_vel_ckpt,
                add_acc=add_acc_ckpt,
                add_global=add_global_ckpt,
                conf_thres=conf_thres_ckpt,
                max_interp_gap=max_interp_gap_ckpt,
                stride=stride_ckpt,
                label_mode=label_mode_ckpt,
                min_valid_frac=min_valid_frac_ckpt,
                add_mask_channel=add_mask_channel_ckpt,
                **extra,
                label_convention=label_convention,
            )
            T_used = int(_T_used)

        print("Window length (T):", T_used)

        y_test = y_test_tags.astype(np.int64, copy=False)

        # Finalise dims for no-metadata checkpoints
        if "state_dict" not in ckpt:
            num_classes = int(y_test.max() + 1)

        test_ds = WindowTensorDataset(X_test, y_test)

        sample_X0, _ = test_ds[0]
        in_features_now = int(sample_X0.shape[-1])
        if node_features_ckpt is None and (in_features_now % 17 == 0):
            node_features_ckpt = in_features_now // 17

        if "state_dict" in ckpt and in_features_now != in_features:
            raise RuntimeError(f"[{m}] in_features mismatch: ckpt={in_features}, dataset={in_features_now}")

        in_features_final = in_features if "state_dict" in ckpt else in_features_now

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
            num_classes=num_classes if "state_dict" in ckpt else int(y_test.max() + 1),
            device=args.device,
            T_used=T_used,
            node_features=node_features_ckpt,
        )

        model.load_state_dict(state, strict=False)

        # Multi-class predictions + optional fall head probability
        y_true, probs_test, fall_prob_test = predict_probs(model, test_loader, device=args.device)
        y_pred = probs_test.argmax(axis=1).astype(int)

        # Confusion matrix (merged 7-class scheme expects 7x7)
        cm = confusion_matrix(y_true, y_pred, labels=labels_all)
        cm_csv = out_dir / f"confusion_matrix_{m}.csv"
        pd.DataFrame(cm, index=NEW_LABEL_NAMES, columns=NEW_LABEL_NAMES).to_csv(cm_csv)
        make_cm_plot(cm, NEW_LABEL_NAMES, plots_dir / f"confusion_matrix_{m}.png", title=f"Confusion Matrix: {m}")

        num_classes_eval = int(num_classes if "state_dict" in ckpt else NUM_CLASSES_MERGED)
        labels_all = list(range(num_classes_eval))
        per_class_f1 = f1_score(y_true, y_pred, labels=labels_all, average=None, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, labels=labels_all, average="macro", zero_division=0)

        for lab, f1v in zip(labels_all, per_class_f1):
            name = NEW_LABEL_NAMES[lab] if ("NEW_LABEL_NAMES" in locals() and 0 <= lab < len(NEW_LABEL_NAMES)) else str(lab)
            f1_rows.append({"model": m, "class_id": int(lab), "class_name": name, "f1": float(f1v)})

        y_true_bin = collapse_to_binary(y_true, fall_class_ids_0based)

        # Binary fall score P(fall)
        if fall_prob_test is not None:
            p_fall_test = fall_prob_test.astype(np.float32)
            p_fall_source = "fall_head"
        else:
            p_fall_test = p_fall_from_probs(probs_test, fall_class_ids_0based).astype(np.float32)
            p_fall_source = "activity_softmax"

        tuned_thr = None
        tuned_prec = None
        tuned_rec = None
        tuned_fbeta = None

        if str(args.binary_mode).lower() == "argmax":
            y_pred_bin = collapse_to_binary(y_pred, fall_class_ids_0based)
            thr = None
        else:
            # Thresholded decision on P(fall)
            if args.threshold is not None:
                thr = float(args.threshold)
            elif args.tune_subjects is not None:
                tune_subjects = parse_range(args.tune_subjects)
                tune_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=args.camera, subjects=tune_subjects)
                if not tune_npzs:
                    raise RuntimeError("No tune NPZs found. Check OUTPUT_ROOT, camera, and tune subjects.")

                X_tune, y_tune_tags, _ = load_windows_from_npzs(
                    tune_npzs,
                    T=T_ckpt,
                    use_conf=use_conf_ckpt,
                    normalize=normalize_ckpt,
                    add_vel=add_vel_ckpt,
                    add_acc=add_acc_ckpt,
                    add_global=add_global_ckpt,
                    conf_thres=conf_thres_ckpt,
                    max_interp_gap=max_interp_gap_ckpt,
                    stride=stride_ckpt,
                    label_mode=label_mode_ckpt,
                    min_valid_frac=min_valid_frac_ckpt,
                    add_mask_channel=add_mask_channel_ckpt,
                    **extra,
                    label_convention=label_convention,
                )

                y_tune = y_tune_tags.astype(np.int64, copy=False)
                tune_ds = WindowTensorDataset(X_tune, y_tune)
                tune_loader = DataLoader(
                    tune_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )

                y_tune_true, probs_tune, fall_prob_tune = predict_probs(model, tune_loader, device=args.device)
                y_tune_bin = collapse_to_binary(y_tune_true, fall_class_ids_0based)

                if fall_prob_tune is not None:
                    p_fall_tune = fall_prob_tune.astype(np.float32)
                else:
                    p_fall_tune = p_fall_from_probs(probs_tune, fall_class_ids_0based).astype(np.float32)

                thr, tuned_prec, tuned_rec, tuned_fbeta = pick_threshold_fbeta(
                    y_tune_bin, p_fall_tune, beta=float(args.beta)
                )
                tuned_thr = thr
            else:
                thr = 0.5

            y_pred_bin = (p_fall_test >= float(thr)).astype(int)

        # Keep for reporting
        chosen_thr = float(thr) if thr is not None else None

        pr, rc, f1b, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, labels=[0, 1], average=None, zero_division=0
        )

        summary_rows.append({
            "model": m,
            "n_samples": int(len(y_true)),
            "params_m": float(count_params_m(model)),
            "macro_f1": float(macro_f1),
            "binary_mode": str(args.binary_mode).lower(),
            "p_fall_source": p_fall_source,  # NEW: records which score was used
            "threshold": chosen_thr,
            "beta": float(args.beta) if str(args.binary_mode).lower() == "threshold" else None,
            "tune_subjects": str(args.tune_subjects) if args.tune_subjects is not None else None,
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
            "camera": int(args.camera),
            "subjects": ",".join(str(s) for s in test_subjects),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    f1_long = pd.DataFrame(f1_rows).sort_values(["model", "class_id"]).reset_index(drop=True)

    summary_csv = out_dir / "metrics_summary.csv"
    f1_csv = out_dir / "f1_per_class.csv"
    summary_df.to_csv(summary_csv, index=False)
    f1_long.to_csv(f1_csv, index=False)

    make_plots(summary_df, plots_dir)
    report_path = out_dir / "report.html"
    make_html_report(summary_df, f1_long, plots_dir, report_path, model_list)

    print(f"Saved: {summary_csv}")
    print(f"Saved: {f1_csv}")
    print(f"Saved: {report_path}")
    print(f"Plots in: {plots_dir.as_posix()}")


if __name__ == "__main__":
    main()
