"""
Evaluate VideoMAE heatmap model on UP-Fall NPZ keypoints windows.

Run from project root, example:
  python -u -m models.eval_videomae_heatmaps --weights models/videomae_heatmaps/<run>/videomae_best.pt --camera 1 --test-subjects 1-5 --out-dir eval_outputs

Outputs:
  <out-dir>/<timestamp>__videomae_heatmaps/
    metrics_summary.csv
    f1_per_class.csv
    report.html
    plots/
      confusion_matrix.png
      f1_per_class.png
      binary_metrics.png

Notes:
- Uses PoseHeatmapWindowDataset + HeatmapSpec + WindowSpec to match training.
- Labels are 0-based in all downstream metrics because PoseHeatmapWindowDataset converts labels when labels_are_1_based=True.
- Binary fall metrics support:
  A) binary_any_fall=True checkpoints (num_labels=2): p_fall = softmax[:,1]
  B) multiclass checkpoints: p_fall = sum softmax over fall classes (provided via --fall-class-ids or checkpoint fall_class_ids_raw)
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.heatmaps.heatmap_dataset import (
    PoseHeatmapWindowDataset,
    WindowSpec,
    HeatmapSpec,
    build_npz_list,
)
from models.videomae.pose_videomae import build_videomae_for_heatmaps


# -------------------------
# Utils
# -------------------------

def _print(msg: str) -> None:
    print(msg, flush=True)


def parse_range(r: str) -> range:
    a, b = r.split("-")
    a, b = int(a), int(b)
    if b < a:
        raise ValueError(f"Bad range '{r}', expected a-b with b>=a")
    return range(a, b + 1)


def _maybe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Handle DDP checkpoints: keys start with "module."
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(path: str, map_location: str = "cpu") -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Robust loading:
      - If checkpoint is a dict with 'state_dict', use it and treat remaining keys as metadata.
      - If it's a raw state_dict, use it and metadata is {}.
    """
    ckpt = torch.load(path, map_location=map_location)
    meta: Dict[str, Any] = {}

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
        # everything else is meta (best-effort JSON-serialisable later)
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    elif isinstance(ckpt, dict):
        # Could be raw state_dict
        state = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")

    state = _strip_module_prefix(state)
    return state, meta


def resolve_default(cli_val: Any, meta: Dict[str, Any], meta_key: str, fallback: Any) -> Any:
    """
    Precedence:
      1) CLI (if not None)
      2) checkpoint metadata (if present)
      3) fallback
    """
    if cli_val is not None:
        return cli_val
    if meta_key in meta and meta[meta_key] is not None:
        return meta[meta_key]
    return fallback


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_csv_row(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)


def save_csv_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", newline="") as f:
            f.write("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def safe_jsonable(x: Any) -> Any:
    # Make checkpoint metadata more robust to dump
    try:
        json.dumps(x)
        return x
    except Exception:
        try:
            if isinstance(x, (np.ndarray,)):
                return x.tolist()
            if isinstance(x, (Path,)):
                return str(x)
            if isinstance(x, (torch.Tensor,)):
                return x.detach().cpu().tolist()
        except Exception:
            pass
    return str(x)


# -------------------------
# Metrics
# -------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def prf_from_cm(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (precision, recall, f1, support) per class from cm (rows=true, cols=pred).
    """
    num_classes = cm.shape[0]
    support = cm.sum(axis=1).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return precision, recall, f1, support


def accuracy_from_cm(cm: np.ndarray) -> float:
    total = float(cm.sum())
    if total <= 0:
        return 0.0
    return float(np.trace(cm) / total)


def macro_f1_from_f1(f1: np.ndarray, support: np.ndarray) -> float:
    # Mean over classes that appear in y_true (support > 0)
    mask = support > 0
    if not np.any(mask):
        return 0.0
    return float(f1[mask].mean())


def binary_metrics_from_counts(tp: float, fp: float, fn: float, tn: float) -> Dict[str, float]:
    # Positive class metrics (class=1 is "fall")
    prec_pos = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec_pos = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_pos = (2 * prec_pos * rec_pos / (prec_pos + rec_pos)) if (prec_pos + rec_pos) > 0 else 0.0

    # Negative class metrics (class=0 is "non-fall")
    prec_neg = (tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rec_neg = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_neg = (2 * prec_neg * rec_neg / (prec_neg + rec_neg)) if (prec_neg + rec_neg) > 0 else 0.0

    # Macro averages over both classes
    avg_precision = 0.5 * (prec_pos + prec_neg)
    avg_recall = 0.5 * (rec_pos + rec_neg)
    avg_f1 = 0.5 * (f1_pos + f1_neg)

    return {
        "precision_fall": float(prec_pos),
        "recall_fall": float(rec_pos),
        "f1_fall": float(f1_pos),
        "precision_nonfall": float(prec_neg),
        "recall_nonfall": float(rec_neg),
        "f1_nonfall": float(f1_neg),
        "avg_precision": float(avg_precision),
        "avg_recall": float(avg_recall),
        "avg_f1": float(avg_f1),
    }


def binary_counts(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Tuple[float, float, float, float]:
    # y_true_bin, y_pred_bin are 0/1
    y_true_bin = y_true_bin.astype(np.int64)
    y_pred_bin = y_pred_bin.astype(np.int64)
    tp = float(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    fp = float(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = float(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
    tn = float(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    return tp, fp, fn, tn


def fbeta_score(tp: float, fp: float, fn: float, beta: float) -> float:
    # For positive class
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    bb = beta * beta
    denom = (bb * prec + rec)
    return ((1 + bb) * prec * rec / denom) if denom > 0 else 0.0


def tune_threshold(p_fall: np.ndarray, y_true_bin: np.ndarray, beta: float = 2.0) -> Tuple[float, float]:
    # Simple grid search, stable and predictable
    best_thr = 0.5
    best = -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        y_pred = (p_fall >= thr).astype(np.int64)
        tp, fp, fn, tn = binary_counts(y_true_bin, y_pred)
        score = fbeta_score(tp, fp, fn, beta=beta)
        if score > best:
            best = score
            best_thr = float(thr)
    return best_thr, float(best)


# -------------------------
# Inference
# -------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    *,
    log_every: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: List[int] = []
    logits_all: List[np.ndarray] = []

    t0 = time.time()
    seen = 0

    for i, (pixel_values, y) in enumerate(loader):
        pixel_values = pixel_values.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        out = model(pixel_values=pixel_values)
        logits = out.logits

        ys.extend(y.detach().cpu().numpy().astype(np.int64).tolist())
        logits_all.append(logits.detach().cpu().numpy().astype(np.float32))

        seen += int(y.shape[0])
        if (i % log_every) == 0:
            dt = time.time() - t0
            ips = (seen / dt) if dt > 0 else 0.0
            _print(f"[eval] batch {i:05d} | samples {seen} | {ips:.1f} samples/s")

    if not logits_all:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 0), dtype=np.float32)

    logits_np = np.concatenate(logits_all, axis=0)
    y_np = np.asarray(ys, dtype=np.int64)
    return y_np, logits_np


def probs_from_logits(logits: np.ndarray) -> np.ndarray:
    # logits: (N,C)
    t = torch.from_numpy(logits)
    p = F.softmax(t, dim=1).cpu().numpy().astype(np.float32)
    return p


def compute_p_fall(
    probs: np.ndarray,
    *,
    binary_any_fall: bool,
    fall_ids_0based: Optional[List[int]],
) -> np.ndarray:
    """
    Returns p_fall for each sample.

    A) binary_any_fall=True: class 1 is fall => p_fall = probs[:,1]
    B) multiclass: p_fall = sum probs over fall_ids_0based (must be provided)
    """
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError(f"Bad probs shape {probs.shape}")

    if binary_any_fall:
        if probs.shape[1] != 2:
            _print(f"[warn] binary_any_fall=True but num_labels={probs.shape[1]}, using class 1 as fall anyway")
        return probs[:, 1].copy()

    if not fall_ids_0based:
        raise ValueError("Multiclass checkpoint needs --fall-class-ids (or checkpoint fall_class_ids_raw) for binary fall metrics")

    fall_ids = [int(i) for i in fall_ids_0based if 0 <= int(i) < probs.shape[1]]
    if not fall_ids:
        raise ValueError(f"After bounds filtering, no fall class ids remain (num_labels={probs.shape[1]})")

    return probs[:, fall_ids].sum(axis=1).astype(np.float32)


def y_true_to_binary(
    y_true_mc: np.ndarray,
    *,
    binary_any_fall: bool,
    fall_ids_0based: Optional[List[int]],
) -> np.ndarray:
    if binary_any_fall:
        return y_true_mc.astype(np.int64)

    if not fall_ids_0based:
        raise ValueError("Need fall_ids_0based for multiclass -> binary mapping")
    fall_set = set(int(i) for i in fall_ids_0based)
    return np.asarray([1 if int(y) in fall_set else 0 for y in y_true_mc.tolist()], dtype=np.int64)


def predict_binary(
    *,
    binary_mode: str,
    probs: np.ndarray,
    y_pred_mc: np.ndarray,
    binary_any_fall: bool,
    fall_ids_0based: Optional[List[int]],
    threshold: float,
) -> np.ndarray:
    """
    binary_mode:
      - "threshold": predict fall if p_fall >= threshold
      - "argmax":
          A) binary_any_fall=True: argmax over 2 classes
          B) multiclass: predict fall if predicted multiclass label is in fall_ids_0based
    """
    binary_mode = str(binary_mode)
    if binary_mode not in ("threshold", "argmax"):
        raise ValueError(f"Unknown binary_mode '{binary_mode}'")

    if binary_mode == "threshold":
        p_fall = compute_p_fall(probs, binary_any_fall=binary_any_fall, fall_ids_0based=fall_ids_0based)
        return (p_fall >= float(threshold)).astype(np.int64)

    # argmax mode
    if binary_any_fall:
        return y_pred_mc.astype(np.int64)

    if not fall_ids_0based:
        raise ValueError("argmax binary mode for multiclass needs fall_ids_0based")
    fall_set = set(int(i) for i in fall_ids_0based)
    return np.asarray([1 if int(y) in fall_set else 0 for y in y_pred_mc.tolist()], dtype=np.int64)


# -------------------------
# Plotting + HTML
# -------------------------

def _mpl_import():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    plt = _mpl_import()
    ensure_dir(out_path.parent)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Tick labels
    n = cm.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([str(i) for i in range(n)], rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in range(n)])

    # Annotate (avoid clutter for large n)
    if n <= 20:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=160)
    plt.close(fig)


def plot_f1_per_class(f1: np.ndarray, support: np.ndarray, out_path: Path, title: str) -> None:
    plt = _mpl_import()
    ensure_dir(out_path.parent)

    idx = np.arange(len(f1))
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.bar(idx, f1)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("Class id (0-based)")
    ax.set_ylabel("F1")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(i) for i in idx], rotation=0)

    # show supports lightly as text
    if len(f1) <= 30:
        for i in range(len(f1)):
            ax.text(i, min(1.0, float(f1[i]) + 0.02), str(int(support[i])), ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=160)
    plt.close(fig)


def plot_binary_metrics(metrics: Dict[str, float], out_path: Path, title: str) -> None:
    plt = _mpl_import()
    ensure_dir(out_path.parent)

    keys = ["precision_fall", "recall_fall", "f1_fall", "avg_precision", "avg_recall", "avg_f1"]
    vals = [float(metrics.get(k, 0.0)) for k in keys]

    fig = plt.figure(figsize=(10, 3.2))
    ax = fig.add_subplot(111)
    ax.bar(range(len(keys)), vals)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=25, ha="right")
    ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(out_path.as_posix(), dpi=160)
    plt.close(fig)


def html_escape(s: Any) -> str:
    s = str(s)
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#039;")
    )


def dict_to_html_table(d: Dict[str, Any]) -> str:
    rows = []
    for k, v in d.items():
        rows.append(f"<tr><td class='k'>{html_escape(k)}</td><td class='v'>{html_escape(v)}</td></tr>")
    return "<table class='kv'>" + "".join(rows) + "</table>"


def rows_to_html_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "<p>(empty)</p>"
    cols = list(rows[0].keys())
    head = "".join([f"<th>{html_escape(c)}</th>" for c in cols])
    body_rows = []
    for r in rows:
        tds = "".join([f"<td>{html_escape(r.get(c,''))}</td>" for c in cols])
        body_rows.append(f"<tr>{tds}</tr>")
    return f"<table class='tbl'><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def read_csv_as_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def write_html_report(
    out_path: Path,
    *,
    summary_row: Dict[str, Any],
    per_class_rows: List[Dict[str, Any]],
    plot_paths: Dict[str, str],
    extra_meta: Dict[str, Any],
) -> None:
    ensure_dir(out_path.parent)

    css = """
    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #111; }
    h1, h2 { margin: 0.2em 0; }
    .sub { color: #444; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 14px; background: #fff; }
    table.tbl { border-collapse: collapse; width: 100%; font-size: 13px; }
    table.tbl th, table.tbl td { border: 1px solid #ddd; padding: 6px 8px; }
    table.tbl th { background: #f6f6f6; text-align: left; }
    table.kv { border-collapse: collapse; width: 100%; font-size: 13px; }
    table.kv td { border: 1px solid #eee; padding: 6px 8px; vertical-align: top; }
    table.kv td.k { width: 38%; color: #333; background: #fafafa; }
    img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; }
    .small { font-size: 12px; color: #555; }
    code { background: #f6f6f6; padding: 1px 4px; border-radius: 4px; }
    """

    summary_table = dict_to_html_table(summary_row)
    meta_table = dict_to_html_table(extra_meta)
    per_class_table = rows_to_html_table(per_class_rows)

    plots_html = ""
    for name, rel in plot_paths.items():
        plots_html += f"<div class='card'><h2>{html_escape(name)}</h2><img src='{html_escape(rel)}' /></div>"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>VideoMAE Heatmaps Evaluation Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>VideoMAE Heatmaps Evaluation</h1>
  <div class="sub small">
    Generated {html_escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}
  </div>

  <div class="grid">
    <div class="card">
      <h2>Summary</h2>
      {summary_table}
    </div>
    <div class="card">
      <h2>Run config</h2>
      {meta_table}
    </div>
  </div>

  <h2 style="margin-top: 18px;">Plots</h2>
  <div class="grid">
    {plots_html}
  </div>

  <h2 style="margin-top: 18px;">Per-class metrics</h2>
  <div class="card">
    {per_class_table}
  </div>

  <div class="small" style="margin-top:18px;">
    Label ids are 0-based (dataset converts from NPZ label space when <code>labels_are_1_based=True</code>).
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VideoMAE heatmap checkpoint (window-based).")

    # Required
    parser.add_argument("--weights", type=str, required=True)

    # Data discovery
    parser.add_argument("--output-root", type=str, default="../../Datasets/UPFall_keypoints/outputs_npz")
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--test-subjects", type=str, required=True)

    # Optional: threshold tuning set
    parser.add_argument("--tune-subjects", type=str, default=None,
                        help="Optional subject range a-b used to tune threshold (same pipeline).")

    # Windowing overrides
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--label-mode", type=str, default=None,
                        choices=["center", "majority", "hybrid_center_fallpct"])
    parser.add_argument("--fall-pct", type=float, default=None)
    parser.add_argument("--min-valid-frac", type=float, default=None)
    parser.add_argument("--drop-ambig-share", type=float, default=None)
    parser.add_argument("--drop-ambig-nonfall-only", type=int, default=None)

    # Heatmap overrides
    parser.add_argument("--hm-size", type=int, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--norm-range", type=float, default=None)

    # Cleaning overrides
    parser.add_argument("--conf-thres", type=float, default=None)
    parser.add_argument("--max-interp-gap", type=int, default=None)
    parser.add_argument("--normalize-xy", type=int, default=None)

    # Task / binary fall metrics
    parser.add_argument("--fall-class-ids", nargs="+", type=int, default=None,
                        help="Fall class IDs in NPZ label space (usually 1-based), eg 1 2 3 4 5")
    parser.add_argument("--binary-mode", type=str, default="threshold",
                        choices=["threshold", "argmax"])
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold on p_fall if binary-mode=threshold. Default 0.5 (or tuned if tune-subjects used).")
    parser.add_argument("--beta", type=float, default=2.0,
                        help="F-beta used for threshold tuning if tune-subjects is set.")

    # Model build (kept for parity with training)
    parser.add_argument("--pretrained-name", type=str, default="MCG-NJU/videomae-base")
    parser.add_argument("--init-from-rgb", type=str, default="mean", choices=["mean", "random"])

    # Performance
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))

    # Output
    parser.add_argument("--out-dir", type=str, default="eval_outputs")

    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_root = Path(args.out_dir) / f"{run_id}__videomae_heatmaps"
    plots_dir = run_root / "plots"
    ensure_dir(plots_dir)

    _print(f"Run ID: {run_id}")
    _print(f"Output dir: {run_root.as_posix()}")

    # Load checkpoint
    _print(f"Loading checkpoint: {args.weights}")
    state_dict, meta = load_checkpoint(args.weights, map_location="cpu")

    # Resolve defaults from checkpoint metadata (if present), then CLI overrides
    # Training saved: num_labels, in_channels, T, stride, hm_size, sigma, normalize_xy, conf_thres, max_interp_gap, label_mode, binary_any_fall, fall_class_ids_raw
    num_labels = int(resolve_default(None, meta, "num_labels", fallback=2))
    in_channels = int(resolve_default(None, meta, "in_channels", fallback=17))

    T = int(resolve_default(args.T, meta, "T", fallback=WindowSpec.T))
    stride = int(resolve_default(args.stride, meta, "stride", fallback=WindowSpec.stride))
    label_mode = str(resolve_default(args.label_mode, meta, "label_mode", fallback=WindowSpec.label_mode))

    # The following are not always in the checkpoint, so we fall back to WindowSpec defaults
    fall_pct = float(resolve_default(args.fall_pct, meta, "fall_pct", fallback=WindowSpec.fall_pct))
    min_valid_frac = float(resolve_default(args.min_valid_frac, meta, "min_valid_frac", fallback=WindowSpec.min_valid_frac))
    drop_ambig_share = float(resolve_default(args.drop_ambig_share, meta, "drop_ambig_share", fallback=WindowSpec.drop_ambig_share))
    drop_ambig_nonfall_only = bool(resolve_default(args.drop_ambig_nonfall_only, meta, "drop_ambig_nonfall_only", fallback=WindowSpec.drop_ambig_nonfall_only))

    hm_size = int(resolve_default(args.hm_size, meta, "hm_size", fallback=HeatmapSpec.out_h))
    sigma = float(resolve_default(args.sigma, meta, "sigma", fallback=HeatmapSpec.sigma))
    norm_range = float(resolve_default(args.norm_range, meta, "norm_range", fallback=HeatmapSpec.norm_range))

    normalize_xy = bool(resolve_default(args.normalize_xy, meta, "normalize_xy", fallback=True))
    conf_thres = float(resolve_default(args.conf_thres, meta, "conf_thres", fallback=HeatmapSpec.conf_thr))
    max_interp_gap = int(resolve_default(args.max_interp_gap, meta, "max_interp_gap", fallback=5))

    binary_any_fall = bool(resolve_default(None, meta, "binary_any_fall", fallback=False))

    # Fall ids:
    fall_class_ids_raw = None
    if args.fall_class_ids is not None and len(args.fall_class_ids) > 0:
        fall_class_ids_raw = [int(x) for x in args.fall_class_ids]
    elif "fall_class_ids_raw" in meta and meta["fall_class_ids_raw"]:
        try:
            fall_class_ids_raw = [int(x) for x in meta["fall_class_ids_raw"]]
        except Exception:
            fall_class_ids_raw = None

    fall_ids_0based: Optional[List[int]] = None
    if fall_class_ids_raw is not None and len(fall_class_ids_raw) > 0:
        fall_ids_0based = [int(x) - 1 for x in fall_class_ids_raw]

    # Sanity
    if T % 2 != 0:
        _print("[warn] --T is odd. Training script recommends even T for VideoMAE (tubelet temporal stride is typically 2).")

    if in_channels != 17:
        _print(f"[warn] checkpoint in_channels={in_channels}, expected 17 for joints. Proceeding anyway.")

    if binary_any_fall:
        # num_labels should be 2 for binary mode
        if num_labels != 2:
            _print(f"[warn] binary_any_fall=True but checkpoint num_labels={num_labels}, proceeding with {num_labels}")
        # Dataset requires fall ids for binary_any_fall
        if not fall_ids_0based:
            raise SystemExit("binary_any_fall=True checkpoint requires fall ids. Provide --fall-class-ids or ensure checkpoint has fall_class_ids_raw.")

    if label_mode == "hybrid_center_fallpct" and not fall_ids_0based:
        raise SystemExit("label_mode=hybrid_center_fallpct requires fall ids. Provide --fall-class-ids (NPZ label ids, usually 1-based).")

    # Build dataset
    output_root = Path(args.output_root)
    test_subjects = parse_range(args.test_subjects)
    test_npzs = build_npz_list(output_root, camera=int(args.camera), subjects=test_subjects)

    if not test_npzs:
        raise SystemExit("No test NPZs found. Check --output-root, --camera, --test-subjects.")

    _print(f"Test sequences: {len(test_npzs)}")

    window_spec = WindowSpec(
        T=int(T),
        stride=int(stride),
        label_mode=str(label_mode),
        fall_pct=float(fall_pct),
        min_valid_frac=float(min_valid_frac),
        drop_ambig_share=float(drop_ambig_share),
        drop_ambig_nonfall_only=bool(drop_ambig_nonfall_only),
    )
    heatmap_spec = HeatmapSpec(
        out_h=int(hm_size),
        out_w=int(hm_size),
        sigma=float(sigma),
        conf_thr=float(conf_thres),
        norm_range=float(norm_range),
    )

    try:
        test_ds = PoseHeatmapWindowDataset(
            test_npzs,
            window=window_spec,
            heatmap=heatmap_spec,
            normalize_xy=bool(normalize_xy),
            conf_thres=float(conf_thres),
            max_interp_gap=int(max_interp_gap),
            labels_are_1_based=True,
            binary_any_fall=bool(binary_any_fall),
            fall_ids_0based=fall_ids_0based,
        )
    except RuntimeError as e:
        _print(f"[error] Failed to build dataset: {e}")
        _print("No windows produced. Check T/stride/conf thresholds, subject range, or NPZ contents.")
        raise SystemExit(2)

    _print(f"Windows: {len(test_ds)} test")

    # Build model arch and load checkpoint weights
    _print("Building model...")
    model = build_videomae_for_heatmaps(
        pretrained_name=str(args.pretrained_name),
        num_labels=int(num_labels),
        in_channels=int(in_channels),
        init_from_rgb=str(args.init_from_rgb),
    )

    missing, unexpected = [], []
    try:
        msg = model.load_state_dict(state_dict, strict=True)
        # torch returns None or IncompatibleKeys depending on version
        if hasattr(msg, "missing_keys"):
            missing = list(msg.missing_keys)
            unexpected = list(msg.unexpected_keys)
    except RuntimeError as e:
        _print(f"[warn] strict load_state_dict failed: {e}")
        _print("[warn] retrying with strict=False")
        msg = model.load_state_dict(state_dict, strict=False)
        if hasattr(msg, "missing_keys"):
            missing = list(msg.missing_keys)
            unexpected = list(msg.unexpected_keys)

    if missing:
        _print(f"[warn] Missing keys ({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
    if unexpected:
        _print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")

    device = str(args.device)
    model = model.to(device)

    # Loader
    pin_memory = device.startswith("cuda")
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
    )

    # Inference
    _print("Running inference...")
    y_true, logits = run_inference(model, test_loader, device, log_every=50)

    if y_true.size == 0:
        _print("[error] No predictions produced. Exiting.")
        raise SystemExit(2)

    probs = probs_from_logits(logits)
    y_pred = probs.argmax(axis=1).astype(np.int64)

    # Multiclass metrics
    n_classes = int(probs.shape[1])
    cm = confusion_matrix(y_true, y_pred, num_classes=n_classes)
    prec_c, rec_c, f1_c, sup_c = prf_from_cm(cm)
    acc = accuracy_from_cm(cm)
    macro_f1 = macro_f1_from_f1(f1_c, sup_c)

    # Per-class CSV
    per_class_rows: List[Dict[str, Any]] = []
    for i in range(n_classes):
        per_class_rows.append({
            "class_id_0based": int(i),
            "support": int(sup_c[i]),
            "precision": float(prec_c[i]),
            "recall": float(rec_c[i]),
            "f1": float(f1_c[i]),
        })
    f1_csv_path = run_root / "f1_per_class.csv"
    save_csv_table(f1_csv_path, per_class_rows)

    # Binary fall metrics
    binary_mode = str(args.binary_mode)
    beta = float(args.beta)

    threshold = float(args.threshold) if args.threshold is not None else 0.5
    tuned_thr = None
    tuned_fbeta = None

    if args.tune_subjects is not None:
        tune_subjects = parse_range(args.tune_subjects)
        tune_npzs = build_npz_list(output_root, camera=int(args.camera), subjects=tune_subjects)
        if not tune_npzs:
            _print("[warn] tune-subjects specified but no NPZs found, skipping tuning")
        else:
            _print(f"Tune sequences: {len(tune_npzs)}")
            try:
                tune_ds = PoseHeatmapWindowDataset(
                    tune_npzs,
                    window=window_spec,
                    heatmap=heatmap_spec,
                    normalize_xy=bool(normalize_xy),
                    conf_thres=float(conf_thres),
                    max_interp_gap=int(max_interp_gap),
                    labels_are_1_based=True,
                    binary_any_fall=bool(binary_any_fall),
                    fall_ids_0based=fall_ids_0based,
                )
                tune_loader = DataLoader(
                    tune_ds,
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    num_workers=int(args.num_workers),
                    pin_memory=pin_memory,
                )
                _print("Running inference on tune set...")
                y_tune, logits_tune = run_inference(model, tune_loader, device, log_every=50)
                if y_tune.size > 0:
                    probs_tune = probs_from_logits(logits_tune)
                    p_fall_tune = compute_p_fall(probs_tune, binary_any_fall=binary_any_fall, fall_ids_0based=fall_ids_0based)
                    y_tune_bin = y_true_to_binary(y_tune, binary_any_fall=binary_any_fall, fall_ids_0based=fall_ids_0based)
                    best_thr, best_score = tune_threshold(p_fall_tune, y_tune_bin, beta=beta)
                    threshold = best_thr
                    tuned_thr = best_thr
                    tuned_fbeta = best_score
                    _print(f"Tuned threshold: {best_thr:.2f} (best F{beta:.1f}={best_score:.4f})")
            except RuntimeError as e:
                _print(f"[warn] tuning dataset produced no windows: {e}")

    # Compute binary predictions and metrics
    y_true_bin = y_true_to_binary(y_true, binary_any_fall=binary_any_fall, fall_ids_0based=fall_ids_0based)
    y_pred_bin = predict_binary(
        binary_mode=binary_mode,
        probs=probs,
        y_pred_mc=y_pred,
        binary_any_fall=binary_any_fall,
        fall_ids_0based=fall_ids_0based,
        threshold=threshold,
    )
    tp, fp, fn, tn = binary_counts(y_true_bin, y_pred_bin)
    bin_metrics = binary_metrics_from_counts(tp, fp, fn, tn)

    # Plots
    plot_confusion_matrix(cm, plots_dir / "confusion_matrix.png", title="Confusion matrix (0-based labels)")
    plot_f1_per_class(f1_c, sup_c, plots_dir / "f1_per_class.png", title=f"F1 per class (macro_f1={macro_f1:.3f})")
    plot_binary_metrics(
        bin_metrics,
        plots_dir / "binary_metrics.png",
        title=f"Binary fall metrics ({binary_mode}, thr={threshold:.2f})",
    )

    # Summary CSV (single row)
    summary_row: Dict[str, Any] = {
        "run_id": run_id,
        "weights": str(args.weights),
        "camera": int(args.camera),
        "test_subjects": str(args.test_subjects),
        "num_sequences": int(len(test_npzs)),
        "num_windows": int(len(test_ds)),
        "device": device,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "num_labels": int(n_classes),
        "in_channels": int(in_channels),

        "T": int(T),
        "stride": int(stride),
        "label_mode": str(label_mode),
        "fall_pct": float(fall_pct),
        "min_valid_frac": float(min_valid_frac),
        "drop_ambig_share": float(drop_ambig_share),
        "drop_ambig_nonfall_only": int(bool(drop_ambig_nonfall_only)),

        "hm_size": int(hm_size),
        "sigma": float(sigma),
        "norm_range": float(norm_range),
        "normalize_xy": int(bool(normalize_xy)),
        "conf_thres": float(conf_thres),
        "max_interp_gap": int(max_interp_gap),

        "binary_any_fall": int(bool(binary_any_fall)),
        "fall_class_ids_raw": json.dumps(fall_class_ids_raw) if fall_class_ids_raw is not None else "",
        "binary_mode": str(binary_mode),
        "threshold": float(threshold),
        "tune_subjects": str(args.tune_subjects) if args.tune_subjects is not None else "",
        "tuned_threshold": float(tuned_thr) if tuned_thr is not None else "",
        "tuned_fbeta": float(tuned_fbeta) if tuned_fbeta is not None else "",
        "beta": float(beta),

        # Core metrics
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),

        # Binary fall metrics (positive class is "fall")
        "precision_fall": float(bin_metrics["precision_fall"]),
        "recall_fall": float(bin_metrics["recall_fall"]),
        "f1_fall": float(bin_metrics["f1_fall"]),
        "avg_precision": float(bin_metrics["avg_precision"]),
        "avg_recall": float(bin_metrics["avg_recall"]),
        "avg_f1": float(bin_metrics["avg_f1"]),
    }
    summary_csv_path = run_root / "metrics_summary.csv"
    save_csv_row(summary_csv_path, summary_row)

    # HTML report
    plot_paths = {
        "Confusion matrix": "plots/confusion_matrix.png",
        "F1 per class": "plots/f1_per_class.png",
        "Binary fall metrics": "plots/binary_metrics.png",
    }

    # extra meta for report (checkpoint meta + resolved args)
    extra_meta = {
        "pretrained_name": str(args.pretrained_name),
        "init_from_rgb": str(args.init_from_rgb),
        "checkpoint_meta": json.dumps({k: safe_jsonable(v) for k, v in meta.items()}, indent=2),
    }

    # Keep summary_row readable in report (avoid huge JSON strings)
    summary_for_report = dict(summary_row)
    if len(str(summary_for_report.get("weights", ""))) > 120:
        summary_for_report["weights"] = str(Path(str(summary_for_report["weights"])))
    if summary_for_report.get("fall_class_ids_raw", ""):
        summary_for_report["fall_class_ids_raw"] = summary_for_report["fall_class_ids_raw"]

    report_path = run_root / "report.html"
    write_html_report(
        report_path,
        summary_row=summary_for_report,
        per_class_rows=per_class_rows,
        plot_paths=plot_paths,
        extra_meta=extra_meta,
    )

    _print("Done.")
    _print(f"Wrote: {summary_csv_path.as_posix()}")
    _print(f"Wrote: {f1_csv_path.as_posix()}")
    _print(f"Wrote: {report_path.as_posix()}")


if __name__ == "__main__":
    main()
