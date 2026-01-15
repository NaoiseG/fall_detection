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
"""  # noqa: E501

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import precision_recall_fscore_support, f1_score

# Same dataset pipeline as training
from dataset import find_keypoints_npzs_subjects, load_windows_from_npzs, WindowTensorDataset

# Same model definitions as training
from .tcn.simple_tcn import TCNBaseline
from .lstm.simple_lstm import LSTMBaseline
from .gru.simple_gru import GRUBaseline
from .gcn.simple_gcn import GCNBaseline
from .mlp.simple_mlp import MLPBaseline
from .stgcn.simple_stgcn import STGCNBaseline


def parse_range(r: str) -> List[int]:
    a, b = r.split("-")
    a, b = int(a), int(b)
    return list(range(a, b + 1))


def count_params_m(model: torch.nn.Module) -> float:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / 1e6


def get_model(
    model_name: str,
    in_features: int,
    num_classes: int,
    device: str,
    T_used: Optional[int] = None,
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
        model = GCNBaseline(
            num_nodes=17,
            node_features=3,
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
        model = STGCNBaseline(
            num_nodes=17,
            node_features=3,
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            t_kernel=9,
            dropout=0.1,
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
        logits = model(X)
        preds = logits.argmax(dim=1)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(preds.detach().cpu().numpy())
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


def collapse_to_binary(y: np.ndarray, fall_class_ids_0based: List[int]) -> np.ndarray:
    fall = set(int(x) for x in fall_class_ids_0based)
    return np.array([1 if int(v) in fall else 0 for v in y], dtype=int)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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


def make_html_report(summary_df: pd.DataFrame, f1_long: pd.DataFrame, plots_dir: Path, out_path: Path):
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

    html = f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\"/>
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

  <div class=\"grid\">
    <div>
      <h2>Macro F1</h2>
      <img src=\"{plots_dir.name}/macro_f1.png\" alt=\"Macro F1\"/>
    </div>
    <div>
      <h2>Binary Sensitivity and Precision</h2>
      <img src=\"{plots_dir.name}/binary_metrics.png\" alt=\"Binary metrics\"/>
    </div>
  </div>

  <h2>F1 per class (multi-class)</h2>
  {df_to_html(f1_long)}

  <p style=\"margin-top:24px;font-size:13px;color:#444;\">
    Generated by <code>models.eval_models</code>.
  </p>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


def main():
    ALL_MODELS = ["tcn", "lstm", "gru", "gcn", "mlp", "stgcn"]

    parser = argparse.ArgumentParser(description="Evaluate trained models on UP-Fall windowed pose tensors.")
    parser.add_argument("--models", nargs="+", required=True, help="Models to evaluate, e.g. --models tcn lstm")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument("--test-subjects", type=str, default="1-1", help="Test subject range like '1-5'")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ckpt-root", type=str, default="models", help="Checkpoint root (default: models)")
    parser.add_argument("--ckpt-subdir", type=str, default="test_run", help="Subdir under each model folder (default: test_run)")
    parser.add_argument("--weights-name", type=str, default=None,
                        help="Override weights filename. If omitted uses '<model>_best.pt'.")
    parser.add_argument("--out-dir", type=str, default="eval_outputs", help="Output directory")
    parser.add_argument("--use-conf", action="store_true", help="Use confidence channel (x,y,conf).")
    parser.add_argument("--no-conf", action="store_true", help="Disable confidence channel (use x,y only).")
    parser.add_argument(
        "--fall-class-ids",
        nargs="+",
        type=int,
        required=True,
        help="Fall class ids in ORIGINAL label space (1-based). Example: --fall-class-ids 9 10 11",
    )
    args = parser.parse_args()

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

    X_test, y_test_tags, T_used = load_windows_from_npzs(test_npzs, T=None, use_conf=use_conf)

    # Same label mapping as training: 1..N -> 0..N-1
    y_test = (y_test_tags.astype(int) - 1).astype(np.int64)

    test_ds = WindowTensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sample_X, _ = test_ds[0]
    in_features = int(sample_X.shape[-1])
    num_classes = int(y_test.max() + 1)

    fall_class_ids_0based = [int(x) - 1 for x in args.fall_class_ids]

    out_dir = Path(args.out_dir).resolve()
    plots_dir = out_dir / "plots"
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    summary_rows: List[Dict[str, object]] = []
    f1_rows: List[Dict[str, object]] = []

    ckpt_root = Path(args.ckpt_root)

    for m in model_list:
        weights_name = args.weights_name or f"{m}_best.pt"
        ckpt_path = ckpt_root / m / args.ckpt_subdir / weights_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Weights not found for {m}: {ckpt_path.as_posix()}")

        model = get_model(m, in_features=in_features, num_classes=num_classes, device=args.device, T_used=T_used)
        state = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected checkpoint format for {m} (expected state_dict dict).")

        model.load_state_dict(state, strict=False)
        y_true, y_pred = predict_all(model, test_loader, device=args.device)

        labels_present = np.unique(y_true).astype(int).tolist()
        per_class_f1 = f1_score(y_true, y_pred, labels=labels_present, average=None, zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, labels=labels_present, average="macro", zero_division=0)

        for lab, f1v in zip(labels_present, per_class_f1):
            f1_rows.append({"model": m, "class_id": int(lab), "f1": float(f1v)})

        y_true_bin = collapse_to_binary(y_true, fall_class_ids_0based)
        y_pred_bin = collapse_to_binary(y_pred, fall_class_ids_0based)

        pr, rc, _, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, labels=[0, 1], average=None, zero_division=0
        )

        summary_rows.append({
            "model": m,
            "n_samples": int(len(y_true)),
            "params_m": float(count_params_m(model)),
            "macro_f1": float(macro_f1),
            "binary_precision_avg": float(np.mean(pr)),
            "binary_sensitivity_avg": float(np.mean(rc)),
            "binary_precision_fall": float(pr[1]),
            "binary_sensitivity_fall": float(rc[1]),
            "binary_precision_no_fall": float(pr[0]),
            "binary_sensitivity_no_fall": float(rc[0]),
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
    make_html_report(summary_df, f1_long, plots_dir, report_path)

    print(f"Saved: {summary_csv}")
    print(f"Saved: {f1_csv}")
    print(f"Saved: {report_path}")
    print(f"Plots in: {plots_dir.as_posix()}") 


if __name__ == "__main__":
    main()
