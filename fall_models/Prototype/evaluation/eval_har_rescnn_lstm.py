#!/usr/bin/env python3
"""
eval_har_rescnn_lstm.py

Evaluate a trained ResCNNLSTM checkpoint on UP-Fall-style NPZs or a pre-windowed NPZ.
Outputs mirror models/eval_models.py and models/MotionBERT/eval_motionbert_action.py:
- metrics_summary.csv
- f1_per_class.csv
- report.html
- plots/confusion_matrix.png
- plots/f1_per_class.png

Examples:
python -m models.har_cnn_lstm.eval_har_rescnn_lstm \
  --checkpoint checkpoints_rescnn_lstm/2026-02-08_12-00-00/best.pt \
  --output-root ../../Datasets/UPFall_keypoints/outputs_npz \
  --camera 1 --subjects 16-20 \
  --out-dir eval_outputs

python -m models.har_cnn_lstm.eval_har_rescnn_lstm \
  --checkpoint checkpoints_rescnn_lstm/2026-02-08_12-00-00/best.pt \
  --windows-npz path/to/windows.npz \
  --out-dir eval_outputs
"""
from __future__ import annotations

import argparse
from datetime import datetime
import inspect
import pickle
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support

from dataset_helpers.dataset import (
    load_windows_from_npzs,
    find_keypoints_npzs_subjects,
    WindowTensorDataset,
    detect_label_convention_from_npzs,
    get_new_label_names,
)
from models.har_cnn_lstm.model_har_rescnn_lstm import ResCNNLSTM, ResCNNLSTMConfig


_TS_DIR_FMT = "%Y-%m-%d_%H-%M-%S_%f"


def torch_load_safe(path: Path, map_location: str = "cpu"):
    """Load a torch checkpoint robustly across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError:
        try:
            import numpy as _np
            try:
                _np_scalar = _np.core.multiarray.scalar
            except Exception:
                _np_scalar = _np._core.multiarray.scalar  # type: ignore[attr-defined]

            try:
                from torch.serialization import safe_globals
                with safe_globals([_np_scalar]):
                    return torch.load(path, map_location=map_location)
            except Exception:
                pass
        except Exception:
            pass

        try:
            sig = inspect.signature(torch.load)
            if "weights_only" in sig.parameters:
                return torch.load(path, map_location=map_location, weights_only=False)
        except Exception:
            pass

        return torch.load(path, map_location=map_location)


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
    spec = str(spec).strip()
    if not spec:
        raise ValueError("Empty subjects spec.")

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


def _torch_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available, falling back to CPU.", flush=True)
        return torch.device("cpu")
    return torch.device(device)


def _count_params_m(model: torch.nn.Module) -> float:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / 1e6


def _normalize_class_names(names: Optional[Sequence[str]], num_classes: int) -> List[str]:
    if names is not None and len(names) >= int(num_classes):
        return [str(x) for x in list(names)[: int(num_classes)]]
    return [str(i) for i in range(int(num_classes))]


def _infer_fall_ids(names: Sequence[str]) -> List[int]:
    out: List[int] = []
    for i, n in enumerate(names):
        if "fall" in str(n).lower():
            out.append(int(i))
    return out


def _load_windows_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise ValueError("windows NPZ must contain arrays named 'X' and 'y'")
    return data["X"], data["y"].astype(np.int64, copy=False)


def _predict_probs(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true_all: List[np.ndarray] = []
    y_pred_all: List[np.ndarray] = []
    probs_all: List[np.ndarray] = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(preds.detach().cpu().numpy())
            probs_all.append(probs.detach().cpu().numpy())

    return (
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        np.concatenate(probs_all),
    )


def _specificity_from_cm(cm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    cm = cm.astype(np.float64)
    total = float(np.sum(cm))
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn
    return tn / (tn + fp + eps)


def _fall_fbeta_from_cm(cm: np.ndarray, class_idx: int = 0, beta: float = 2.0, eps: float = 1e-12) -> float:
    """
    One-vs-rest F_beta for a target class computed from a multiclass confusion matrix.
    cm shape: (C, C) with rows=gt, cols=pred.
    """
    cm = cm.astype(np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square [C,C], got {tuple(cm.shape)}")

    num_classes = int(cm.shape[0])
    class_idx = int(class_idx)
    if class_idx < 0 or class_idx >= num_classes:
        return 0.0

    beta = float(beta)
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    beta2 = beta * beta

    tp = cm[class_idx, class_idx]
    fn = cm[class_idx, :].sum() - tp
    fp = cm[:, class_idx].sum() - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    denom = beta2 * precision + recall + eps
    fbeta = (1.0 + beta2) * precision * recall / denom
    return float(fbeta)


def _collapse_to_binary(y: np.ndarray, fall_ids: Sequence[int]) -> np.ndarray:
    fall = set(int(x) for x in fall_ids)
    return np.array([1 if int(v) in fall else 0 for v in y], dtype=int)


def _p_fall_from_probs(probs: np.ndarray, fall_ids: Sequence[int]) -> np.ndarray:
    idx = [int(i) for i in fall_ids if 0 <= int(i) < probs.shape[1]]
    if len(idx) == 0:
        return np.zeros((probs.shape[0],), dtype=np.float32)
    return probs[:, idx].sum(axis=1)


def _pick_threshold_fbeta(
    y_true_bin: np.ndarray, p_fall: np.ndarray, beta: float = 2.0
) -> Tuple[float, float, float, float]:
    prec, rec, th = precision_recall_curve(y_true_bin, p_fall)
    denom = (beta * beta * prec + rec + 1e-9)
    fbeta = (1.0 + beta * beta) * (prec * rec) / denom
    if th.size == 0:
        return 0.5, float(prec[-1] if prec.size else 0.0), float(rec[-1] if rec.size else 0.0), 0.0
    best_i = int(np.nanargmax(fbeta[:-1]))
    return float(th[best_i]), float(prec[best_i]), float(rec[best_i]), float(fbeta[best_i])


def _binary_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "binary_precision_avg": float(np.mean(pr)),
        "binary_sensitivity_avg": float(np.mean(rc)),
        "binary_f1_avg": float(np.mean(f1)),
        "binary_precision_fall": float(pr[1]),
        "binary_sensitivity_fall": float(rc[1]),
        "binary_f1_fall": float(f1[1]),
        "binary_precision_no_fall": float(pr[0]),
        "binary_sensitivity_no_fall": float(rc[0]),
        "binary_f1_no_fall": float(f1[0]),
    }


def _make_cm_plot(cm: np.ndarray, class_names: List[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")

    fmt = ".2f"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            color = "white" if float(v) == 0.0 else "black"
            ax.text(j, i, format(v, fmt), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_f1_bar(per_class_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

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
    overall_df: pd.DataFrame,
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
  <title>ResCNNLSTM Evaluation Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>ResCNNLSTM Evaluation Report</h1>

  <h2>Summary</h2>
  {_df_to_html(summary_df)}

  <h2>Overall metrics</h2>
  <p style="margin:0 0 6px 0;color:#444;font-size:13px;">
    Values are percentages. Precision/recall/specificity/F1 are macro-averaged over classes present in this split.
  </p>
  {_df_to_html(overall_df)}

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
    Generated by <code>eval_har_rescnn_lstm.py</code>. Output folder: <code>{out_dir.name}</code>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def _load_upfall_windows(
    output_root: Path,
    subjects: Sequence[int],
    args: argparse.Namespace,
    fall_ids_for_labeling: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, int, List[str]]:
    npzs = find_keypoints_npzs_subjects(output_root, camera=int(args.camera), subjects=subjects)
    if not npzs:
        raise RuntimeError("No NPZs found. Check --output-root/--camera/--subjects.")

    conv, _ = detect_label_convention_from_npzs(npzs)
    label_names = get_new_label_names(conv)

    extra: Dict[str, object] = {}
    if str(args.label_mode).lower() == "hybrid_center_fallpct":
        extra["fall_ids_0based"] = [int(x) for x in fall_ids_for_labeling]
        extra["fall_pct"] = float(args.fall_pct)

    X, y, T_used = load_windows_from_npzs(
        npzs,
        T=int(args.T),
        use_conf=bool(args.use_conf),
        normalize=bool(args.normalize),
        normalize_mode=str(args.normalize_mode),
        add_vel=bool(args.add_vel),
        add_acc=bool(args.add_acc),
        add_global=bool(args.add_global),
        conf_thres=float(args.conf_thres),
        max_interp_gap=int(args.max_interp_gap),
        missing_mode=str(args.missing_mode),
        interp_mode=str(args.interp_mode),
        interp_group=int(args.interp_group),
        stride=int(args.stride),
        label_mode=str(args.label_mode),
        min_valid_frac=float(args.min_valid_frac),
        add_mask_channel=bool(args.add_mask_channel),
        drop_ambig_share=float(args.drop_ambig_share),
        drop_ambig_nonfall_only=bool(args.drop_ambig_nonfall_only),
        label_convention=conv,
        rp_center_mode=str(args.rp_center_mode),
        rp_img_w=args.rp_img_w,
        rp_img_h=args.rp_img_h,
        **extra,
    )
    return X, y, int(T_used), label_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ResCNNLSTM (residual CNN-LSTM) for 7-class HAR.")

    # Checkpoint
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (best.pt).")

    # Data (choose one)
    parser.add_argument("--windows-npz", type=str, default=None, help="NPZ with arrays X and y (already windowed).")
    parser.add_argument("--output-root", type=str, default=None, help="Root of UP-Fall-style outputs_npz directory.")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (only used with --output-root).")
    parser.add_argument(
        "--subjects",
        "--test-subjects",
        dest="subjects",
        type=str,
        default="1-1",
        help="Subject spec like '1-5' or '1-3,7,9-10'.",
    )

    # Windowing/preproc (only used with --output-root)
    parser.add_argument("--T", type=int, default=64, help="Window length.")
    parser.add_argument("--stride", type=int, default=16, help="Window stride.")
    parser.add_argument("--label-mode", type=str, default="center", choices=["center", "majority", "hybrid_center_fallpct"])
    parser.add_argument("--use-conf", type=int, default=1)
    parser.add_argument("--normalize", type=int, default=1)
    parser.add_argument("--normalize-mode", type=str, default="center_scale", choices=["center_scale", "paper_rp"])
    parser.add_argument("--add-vel", type=int, default=1)
    parser.add_argument("--add-acc", type=int, default=1)
    parser.add_argument("--add-global", type=int, default=1)
    parser.add_argument("--add-mask-channel", type=int, default=1)
    parser.add_argument("--conf-thres", type=float, default=0.2)
    parser.add_argument("--max-interp-gap", type=int, default=5)
    parser.add_argument("--missing-mode", type=str, default="conf_thres", choices=["conf_thres", "zeros_only", "conf_or_zeros"])
    parser.add_argument("--interp-mode", type=str, default="short_gap_hold", choices=["short_gap_hold", "paper_group_linear"])
    parser.add_argument("--interp-group", type=int, default=100)
    parser.add_argument("--min-valid-frac", type=float, default=0.3)
    parser.add_argument("--drop-ambig-share", type=float, default=0.0)
    parser.add_argument("--drop-ambig-nonfall-only", type=int, default=1)
    parser.add_argument("--fall-pct", type=float, default=0.25)
    parser.add_argument("--rp-center-mode", type=str, default="auto", choices=["auto", "normalized_01", "pixel"])
    parser.add_argument("--rp-img-w", type=int, default=None)
    parser.add_argument("--rp-img-h", type=int, default=None)

    # Eval
    parser.add_argument("--num-classes", type=int, default=None, help="Override num_classes (defaults to checkpoint).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="eval_outputs", help="Base output directory.")

    # Optional: binary fall vs no-fall metrics
    parser.add_argument(
        "--fall-class-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional list of 0-based class ids to treat as 'fall' for binary metrics.",
    )
    parser.add_argument(
        "--binary-mode",
        type=str,
        default="threshold",
        choices=["threshold", "argmax"],
        help="How to form fall/no-fall decision. 'threshold' uses P(fall)=sum of fall-class softmax probs.",
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
        "--fall-class-idx",
        type=int,
        default=None,
        help="Single class index for fall F_beta (one-vs-rest). Overrides checkpoint if provided.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default=None,
        choices=["macro_f1", "fall_fbeta", "composite_fall_fbeta_macro_f1"],
        help="Override checkpoint selection metric for reporting.",
    )
    parser.add_argument(
        "--selection-w",
        type=float,
        default=None,
        help="Override selection weight when selection metric is composite.",
    )
    parser.add_argument(
        "--selection-beta",
        type=float,
        default=None,
        help="Override beta for fall F_beta used in selection metric reporting.",
    )
    parser.add_argument(
        "--tune-subjects",
        type=str,
        default=None,
        help="Optional subject spec (e.g. 13-16) to tune threshold on. Only valid with --output-root.",
    )

    args = parser.parse_args()

    device = _torch_device(str(args.device))

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.as_posix()}")

    ckpt = torch_load_safe(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        cfg_dict = ckpt.get("model_cfg", None)
        in_features_ckpt = ckpt.get("in_features", None)
        num_classes_ckpt = ckpt.get("num_classes", None)
        label_info = ckpt.get("label_info", None)
        T_used_ckpt = ckpt.get("T_used", None)
        selection_metric_ckpt = ckpt.get("selection_metric", None)
        selection_w_ckpt = ckpt.get("selection_w", None)
        selection_beta_ckpt = ckpt.get("selection_beta", None)
        fall_class_idx_ckpt = ckpt.get("fall_class_idx", None)
        ckpt_best_val_score = ckpt.get("best_val_score", None)
        ckpt_best_val_macro_f1 = ckpt.get("best_val_macro_f1", None)
        ckpt_best_val_fall_fbeta = ckpt.get("best_val_fall_fbeta", None)
    else:
        state = ckpt
        cfg_dict = None
        in_features_ckpt = None
        num_classes_ckpt = None
        label_info = None
        T_used_ckpt = None
        selection_metric_ckpt = None
        selection_w_ckpt = None
        selection_beta_ckpt = None
        fall_class_idx_ckpt = None
        ckpt_best_val_score = None
        ckpt_best_val_macro_f1 = None
        ckpt_best_val_fall_fbeta = None

    fall_ids_for_labeling = [int(x) for x in args.fall_class_ids] if args.fall_class_ids else [0]

    data_source = "windows_npz" if args.windows_npz else "upfall_npzs"

    if args.windows_npz:
        if args.output_root:
            print("WARNING: --output-root ignored because --windows-npz was provided.", flush=True)
        X, y = _load_windows_npz(args.windows_npz)
        T_used = int(X.shape[1])
        label_names_data: Optional[List[str]] = None
        subjects_used: Optional[str] = None
    else:
        if not args.output_root:
            raise SystemExit("Provide either --windows-npz or --output-root.")
        subjects_used = str(args.subjects)
        subject_list = _parse_subjects(args.subjects)
        X, y, T_used, label_names_data = _load_upfall_windows(
            output_root=Path(args.output_root),
            subjects=subject_list,
            args=args,
            fall_ids_for_labeling=fall_ids_for_labeling,
        )

    y = y.astype(np.int64, copy=False)
    if X.ndim == 3:
        in_features_data = int(X.shape[-1])
    elif X.ndim == 4:
        in_features_data = int(X.shape[-2] * X.shape[-1])
    else:
        raise RuntimeError(f"Unexpected X shape {tuple(X.shape)} (expected 3D or 4D).")

    if cfg_dict is not None:
        cfg = ResCNNLSTMConfig(**cfg_dict)
        if int(cfg.in_features) != int(in_features_data):
            raise RuntimeError(
                f"in_features mismatch: ckpt={int(cfg.in_features)} vs data={int(in_features_data)}"
            )
        num_classes = int(cfg.num_classes)
        if args.num_classes is not None and int(args.num_classes) != num_classes:
            raise RuntimeError(
                f"num_classes mismatch: ckpt={num_classes} vs --num-classes={int(args.num_classes)}"
            )
    else:
        num_classes = int(args.num_classes) if args.num_classes is not None else int(num_classes_ckpt or (y.max() + 1))
        if in_features_ckpt is not None and int(in_features_ckpt) != int(in_features_data):
            raise RuntimeError(
                f"in_features mismatch: ckpt={int(in_features_ckpt)} vs data={int(in_features_data)}"
            )
        cfg = ResCNNLSTMConfig(in_features=in_features_data, num_classes=num_classes)

    if int(y.max()) >= int(num_classes) or int(y.min()) < 0:
        raise SystemExit(
            f"Labels must be in [0,{num_classes-1}]. "
            f"Got min={int(y.min())}, max={int(y.max())}"
        )

    model = ResCNNLSTM(cfg).to(device)
    model.load_state_dict(state, strict=True)
    params_m = float(_count_params_m(model))

    ds = WindowTensorDataset(X, y)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )

    y_true, y_pred, probs = _predict_probs(model, loader, device=device)

    labels_all = list(range(int(num_classes)))
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels_all).astype(np.float64)
    row_sums = cm_counts.sum(axis=1, keepdims=True) + 1e-9
    cm_norm = cm_counts / row_sums

    label_names_ckpt = None
    if isinstance(label_info, dict):
        maybe = label_info.get("id_to_name", None)
        if isinstance(maybe, (list, tuple)):
            label_names_ckpt = list(maybe)
    class_names = _normalize_class_names(label_names_data or label_names_ckpt, num_classes)

    support = cm_counts.sum(axis=1)
    valid = support > 0
    tp = np.diag(cm_counts)
    pred_support = cm_counts.sum(axis=0)
    recall = tp / (support + 1e-12)
    precision = tp / (pred_support + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    specificity = _specificity_from_cm(cm_counts)

    total = float(np.sum(cm_counts))
    acc = float(np.sum(tp) / total) if total > 0 else 0.0
    macro_recall = float(np.mean(recall[valid])) if np.any(valid) else 0.0
    macro_precision = float(np.mean(precision[valid])) if np.any(valid) else 0.0
    macro_specificity = float(np.mean(specificity[valid])) if np.any(valid) else 0.0
    macro_f1 = float(np.mean(f1[valid])) if np.any(valid) else 0.0

    # One-vs-rest fall F_beta (for compatibility with training selection metrics)
    if args.fall_class_idx is not None:
        fall_class_idx = int(args.fall_class_idx)
    elif fall_class_idx_ckpt is not None:
        fall_class_idx = int(fall_class_idx_ckpt)
    elif args.fall_class_ids is not None and len(args.fall_class_ids) == 1:
        fall_class_idx = int(args.fall_class_ids[0])
    else:
        fall_class_idx = 0

    if args.selection_beta is not None:
        selection_beta = float(args.selection_beta)
    elif selection_beta_ckpt is not None:
        selection_beta = float(selection_beta_ckpt)
    else:
        selection_beta = 2.0

    fall_fbeta = _fall_fbeta_from_cm(cm_counts, class_idx=fall_class_idx, beta=selection_beta)

    if args.selection_metric is not None:
        selection_metric = str(args.selection_metric)
    elif selection_metric_ckpt is not None:
        selection_metric = str(selection_metric_ckpt)
    else:
        selection_metric = "macro_f1"

    if args.selection_w is not None:
        selection_w = float(args.selection_w)
    elif selection_w_ckpt is not None:
        selection_w = float(selection_w_ckpt)
    else:
        selection_w = 0.7

    selection_w = max(0.0, min(1.0, float(selection_w)))
    if selection_metric == "fall_fbeta":
        selection_score = float(fall_fbeta)
    elif selection_metric == "composite_fall_fbeta_macro_f1":
        selection_score = float(selection_w) * float(fall_fbeta) + (1.0 - float(selection_w)) * float(macro_f1)
    else:
        selection_score = float(macro_f1)

    per_class_df = pd.DataFrame({
        "class_id": labels_all,
        "class_name": [class_names[int(i)] if 0 <= int(i) < len(class_names) else str(i) for i in labels_all],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(np.int64, copy=False),
    })

    overall_df = (
        pd.DataFrame([{
            "accuracy": float(acc) * 100.0,
            "recall": float(macro_recall) * 100.0,
            "specificity": float(macro_specificity) * 100.0,
            "precision": float(macro_precision) * 100.0,
            "f1_score": float(macro_f1) * 100.0,
        }])
        .round(3)
    )

    # Binary fall/no-fall metrics (optional)
    fall_ids: List[int] = [int(x) for x in args.fall_class_ids] if args.fall_class_ids else _infer_fall_ids(class_names)
    extra_html = ""
    binary_summary: Dict[str, object] = {}
    if fall_ids:
        y_true_bin = _collapse_to_binary(y_true, fall_ids)
        tuned_thr = None
        tuned_prec = None
        tuned_rec = None
        tuned_fbeta = None

        if str(args.binary_mode).lower() == "argmax":
            y_pred_bin = _collapse_to_binary(y_pred, fall_ids)
            thr = None
        else:
            p_fall = _p_fall_from_probs(probs, fall_ids)
            if args.threshold is not None:
                thr = float(args.threshold)
            elif args.tune_subjects is not None:
                if args.windows_npz:
                    raise SystemExit("--tune-subjects is only valid with --output-root.")
                tune_subjects = _parse_subjects(args.tune_subjects)
                X_tune, y_tune, _T_tune, _ = _load_upfall_windows(
                    output_root=Path(args.output_root),
                    subjects=tune_subjects,
                    args=args,
                    fall_ids_for_labeling=fall_ids_for_labeling,
                )
                tune_ds = WindowTensorDataset(X_tune, y_tune.astype(np.int64, copy=False))
                tune_loader = DataLoader(
                    tune_ds,
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    drop_last=False,
                    num_workers=int(args.num_workers),
                    pin_memory=(device.type == "cuda"),
                )
                y_tune_true, _y_tune_pred, probs_tune = _predict_probs(model, tune_loader, device=device)
                y_tune_bin = _collapse_to_binary(y_tune_true, fall_ids)
                p_fall_tune = _p_fall_from_probs(probs_tune, fall_ids)

                thr, tuned_prec, tuned_rec, tuned_fbeta = _pick_threshold_fbeta(
                    y_tune_bin, p_fall_tune, beta=float(args.beta)
                )
                tuned_thr = thr
            else:
                thr = 0.5

            y_pred_bin = (p_fall >= float(thr)).astype(int)

        binary_summary.update({
            "binary_mode": str(args.binary_mode).lower(),
            "p_fall_source": "activity_softmax",
            "threshold": float(thr) if thr is not None else None,
            "beta": float(args.beta) if str(args.binary_mode).lower() == "threshold" else None,
            "tune_subjects": str(args.tune_subjects) if args.tune_subjects is not None else None,
            "tuned_threshold": tuned_thr,
            "tuned_precision_fall": tuned_prec,
            "tuned_recall_fall": tuned_rec,
            "tuned_fbeta": tuned_fbeta,
        })
        binary_summary.update(_binary_metrics(y_true_bin, y_pred_bin))

        extra_html = f"""
  <h2>Binary fall vs no-fall</h2>
  <p>Fall class ids: <code>{sorted(fall_ids)}</code></p>
"""

    # Output directory
    ts = datetime.now().strftime(_TS_DIR_FMT)
    tag_parts = ["har_rescnn_lstm"]
    if args.windows_npz:
        tag_parts.append(Path(args.windows_npz).stem)
    else:
        tag_parts.append(f"cam{int(args.camera)}__subjects_{_slug(str(args.subjects))}")
    tag = _slug("__".join(tag_parts))
    out_dir = Path(args.out_dir).expanduser().resolve() / f"{ts}__{tag}"
    plots_dir = out_dir / "plots"
    _ensure_dir(out_dir)
    _ensure_dir(plots_dir)

    print("Eval output dir:", out_dir.as_posix(), flush=True)

    summary = {
        "model": "har_rescnn_lstm",
        "n_samples": int(len(y_true)),
        "params_m": float(params_m),
        "acc_top1": float(acc) * 100.0,
        "balanced_acc": float(macro_recall),
        "macro_f1": float(macro_f1),
        "fall_fbeta": float(fall_fbeta),
        "fall_fbeta_beta": float(selection_beta),
        "fall_class_idx": int(fall_class_idx),
        "selection_metric": str(selection_metric),
        "selection_w": float(selection_w) if selection_metric == "composite_fall_fbeta_macro_f1" else None,
        "selection_score": float(selection_score),
        "ckpt_best_val_score": ckpt_best_val_score,
        "ckpt_best_val_macro_f1": ckpt_best_val_macro_f1,
        "ckpt_best_val_fall_fbeta": ckpt_best_val_fall_fbeta,
        "ckpt_selection_metric": selection_metric_ckpt,
        "ckpt_selection_w": selection_w_ckpt,
        "ckpt_selection_beta": selection_beta_ckpt,
        "ckpt_fall_class_idx": fall_class_idx_ckpt,
        "checkpoint": ckpt_path.as_posix(),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "data_source": data_source,
        "camera": int(args.camera) if not args.windows_npz else None,
        "subjects": subjects_used,
        "windows_npz": str(args.windows_npz) if args.windows_npz else None,
        "T_used": int(T_used),
        "T_used_ckpt": int(T_used_ckpt) if T_used_ckpt is not None else None,
    }
    summary.update(binary_summary)
    summary_df = pd.DataFrame([summary])

    # Save CSVs
    summary_csv = out_dir / "metrics_summary.csv"
    per_class_csv = out_dir / "f1_per_class.csv"
    summary_df.to_csv(summary_csv, index=False)
    per_class_df.to_csv(per_class_csv, index=False)

    # Confusion matrix CSV + plots
    cm_csv = out_dir / "confusion_matrix.csv"
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(cm_csv)
    _make_cm_plot(cm_norm, class_names, plots_dir / "confusion_matrix.png", title="Confusion Matrix (normalized)")
    _plot_f1_bar(per_class_df, plots_dir / "f1_per_class.png")

    # Report
    report_path = out_dir / "report.html"
    _make_html_report(
        summary_df,
        overall_df,
        per_class_df,
        out_dir=out_dir,
        plots_dir=plots_dir,
        out_path=report_path,
        extra_html=extra_html,
    )

    print("\nOverall metrics (%):", flush=True)
    print(overall_df.to_string(index=False), flush=True)
    print(f"Saved: {summary_csv.as_posix()}", flush=True)
    print(f"Saved: {per_class_csv.as_posix()}", flush=True)
    print(f"Saved: {report_path.as_posix()}", flush=True)
    print(f"Plots in: {plots_dir.as_posix()}", flush=True)


if __name__ == "__main__":
    main()
