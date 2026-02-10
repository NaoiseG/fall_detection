#!/usr/bin/env python3
"""eval_har_keras.py

Evaluate Keras HAR checkpoints produced by training/train_har_keras.py, using
the same UP-Fall NPZ -> window pipeline as eval_models.py.

Outputs (in --out-dir/<timestamp>__models_<tag>/):
- metrics_summary.csv
- f1_per_class.csv
- report.html
- plots/*.png
- confusion_matrix_<model>.csv

Example:
python evaluation/eval_har_keras.py --models my_har_model --camera 1 --test-subjects 1-1 --out-dir eval_outputs
"""
from __future__ import annotations

import argparse
from datetime import datetime
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_recall_fscore_support


def _find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(6):
        if (cur / "dataset_helpers").exists() and (cur / "pose_models").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve().parent


PROJECT_ROOT = _find_project_root(Path(__file__))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from dataset_helpers.dataset import (  # noqa: E402
    load_windows_from_npzs,
    find_keypoints_npzs_subjects,
    detect_label_convention_from_npzs,
    get_new_label_names,
)


NUM_CLASSES_MERGED = 7
FALL_CLASS_ID = 0


def parse_range(r: str) -> List[int]:
    a, b = r.split("-")
    a, b = int(a), int(b)
    return list(range(a, b + 1))


def slug_models(models: List[str], max_len: int = 80) -> str:
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
    if override is None or override.lower() == "latest":
        return pick_latest_run_dir(model_dir)
    run_dir = model_dir / override
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir.as_posix()}")
    return run_dir


def merged_id_to_raw(merged_id: int, convention: str) -> int:
    if convention == "1-11":
        if merged_id == 0:
            return 1
        return int(merged_id) + 5
    if convention == "0-10":
        if merged_id == 0:
            return 0
        return int(merged_id) + 4
    raise ValueError(f"Unknown convention: {convention}")


def _parse_int_like(x):
    try:
        return int(x)
    except Exception:
        try:
            xf = float(x)
            if float(xf).is_integer():
                return int(xf)
        except Exception:
            return None
    return None


def coerce_labels_to_int(labels: np.ndarray, label_names, convention_default: str = "1-11") -> np.ndarray:
    try:
        return labels.astype(np.int64, copy=False)
    except Exception:
        name_to_id = {str(n).strip().lower(): i for i, n in enumerate(label_names)}
        out = []
        for item in labels:
            val = _parse_int_like(item)
            if val is not None:
                out.append(int(val))
                continue
            key = str(item).strip().lower()
            if key in name_to_id:
                merged_id = int(name_to_id[key])
                raw_id = merged_id_to_raw(merged_id, convention_default)
                out.append(int(raw_id))
            else:
                raise ValueError(
                    f"Unknown non-numeric label '{item}'. Expected one of: {sorted(name_to_id.keys())}"
                )
        return np.array(out, dtype=np.int64)


def sanitize_npz_labels(npz_paths, out_dir: Path, label_names, convention_default: str = "1-11"):
    out_dir.mkdir(parents=True, exist_ok=True)
    sanitized_paths = []
    for p in npz_paths:
        p = Path(p)
        with np.load(p, allow_pickle=True) as data:
            labels = data["frame_labels"]
            try:
                labels.astype(np.int64, copy=False)
                sanitized_paths.append(str(p))
                continue
            except Exception:
                labels_int = coerce_labels_to_int(labels, label_names, convention_default=convention_default)
                arrays = {k: data[k] for k in data.files}
                arrays["frame_labels"] = labels_int
                new_path = out_dir / f"{p.stem}_labels_int.npz"
                np.savez(new_path, **arrays)
                sanitized_paths.append(str(new_path))
    return sanitized_paths


def get_model_input_spec(model: tf.keras.Model):
    input_shape = model.input_shape
    if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
        if len(input_shape) != 1:
            raise ValueError("Only single-input models are supported.")
        input_shape = input_shape[0]

    if input_shape is None:
        raise ValueError("Model input shape is None.")

    if len(input_shape) == 2:
        return {"rank": 2, "T": None, "F": input_shape[-1]}
    if len(input_shape) == 3:
        return {"rank": 3, "T": input_shape[-2], "F": input_shape[-1]}
    raise ValueError(f"Unsupported model input rank: {len(input_shape)} (shape={input_shape})")


def get_model_output_dim(model: tf.keras.Model):
    output_shape = model.output_shape
    if isinstance(output_shape, (list, tuple)) and len(output_shape) > 0 and isinstance(output_shape[0], (list, tuple)):
        if len(output_shape) != 1:
            raise ValueError("Only single-output models are supported.")
        output_shape = output_shape[0]
    if output_shape is None:
        return None
    return output_shape[-1]


def adjust_sequence_length(X: np.ndarray, target_T: Optional[int]):
    if target_T is None:
        return X, "keep"
    T = int(X.shape[1])
    if T == int(target_T):
        return X, "match"
    if T > int(target_T):
        start = (T - int(target_T)) // 2
        return X[:, start : start + int(target_T), :], f"trim({T}->{target_T})"
    pad = int(target_T) - T
    X_pad = np.pad(X, ((0, 0), (0, pad), (0, 0)), mode="constant")
    return X_pad, f"pad({T}->{target_T})"


def count_params_m_keras(model: tf.keras.Model) -> float:
    n = 0
    for w in model.trainable_weights:
        n += int(np.prod(w.shape))
    return float(n) / 1e6


def collapse_to_binary(y: np.ndarray, fall_class_ids_0based: List[int]) -> np.ndarray:
    fall = set(int(x) for x in fall_class_ids_0based)
    return np.array([1 if int(v) in fall else 0 for v in y], dtype=int)


def p_fall_from_probs(probs: np.ndarray, fall_class_ids_0based: List[int]) -> np.ndarray:
    idx = [int(i) for i in fall_class_ids_0based if 0 <= int(i) < probs.shape[1]]
    if len(idx) == 0:
        return np.zeros((probs.shape[0],), dtype=np.float32)
    return probs[:, idx].sum(axis=1)


def pick_threshold_fbeta(y_true_bin: np.ndarray, p_fall: np.ndarray, beta: float = 2.0) -> Tuple[float, float, float, float]:
    prec, rec, th = precision_recall_curve(y_true_bin, p_fall)
    denom = (beta * beta * prec + rec + 1e-9)
    fbeta = (1.0 + beta * beta) * (prec * rec) / denom
    if th.size == 0:
        return 0.5, float(prec[-1] if prec.size else 0.0), float(rec[-1] if rec.size else 0.0), 0.0
    best_i = int(np.nanargmax(fbeta[:-1]))
    return float(th[best_i]), float(prec[best_i]), float(rec[best_i]), float(fbeta[best_i])


def specificity_from_cm(cm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
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
    for m in model_list:
        p = plots_dir / f"confusion_matrix_{m}.png"
        if p.exists():
            conf_parts.append(
                f"<div><h3>{m}</h3><img src='{plots_dir.name}/confusion_matrix_{m}.png' "
                f"alt='Confusion matrix {m}'/></div>"
            )
    conf_imgs = "\n".join(conf_parts) if conf_parts else "<p>No confusion matrices found.</p>"

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

  <h2>Overall metrics</h2>
  <p style=\"margin:0 0 6px 0;color:#444;font-size:13px;\">
    Values are percentages. Precision/recall/specificity/F1 are macro-averaged over classes present in this split.
  </p>
  {df_to_html(overall_df)}

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

  <h2>Confusion matrices</h2>
  <p>Rows are true labels, columns are predicted labels. Labels follow the merged 7-class scheme.</p>
  <div class=\"grid\">
    {conf_imgs}
  </div>

  <p style=\"margin-top:24px;font-size:13px;color:#444;\">
    Generated by <code>evaluation.eval_har_keras</code>.
  </p>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")


def _looks_like_probs(arr: np.ndarray) -> bool:
    if arr.ndim != 2:
        return False
    if not np.isfinite(arr).all():
        return False
    if arr.min() < -1e-6 or arr.max() > 1.0 + 1e-6:
        return False
    row_sums = arr.sum(axis=1)
    return bool(np.allclose(row_sums, 1.0, atol=1e-3))


def _to_probs(logits_or_probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits_or_probs)
    if _looks_like_probs(arr):
        return arr.astype(np.float32, copy=False)
    probs = tf.nn.softmax(arr, axis=1).numpy()
    return probs.astype(np.float32, copy=False)


def _prepare_inputs_for_model(X_base: np.ndarray, spec: dict) -> Tuple[np.ndarray, str]:
    if spec["rank"] == 3:
        X_adj, action = adjust_sequence_length(X_base, spec["T"])
        if spec["F"] is not None and int(X_adj.shape[-1]) != int(spec["F"]):
            raise RuntimeError(
                f"Feature dim mismatch: model expects F={int(spec['F'])}, data has F={int(X_adj.shape[-1])}."
            )
        return X_adj, action

    if spec["rank"] == 2:
        X_flat = X_base.reshape(X_base.shape[0], -1)
        if spec["F"] is not None and int(X_flat.shape[-1]) != int(spec["F"]):
            raise RuntimeError(
                f"Feature dim mismatch: model expects F={int(spec['F'])}, data has F={int(X_flat.shape[-1])}."
            )
        return X_flat, "flatten"

    raise ValueError(f"Unsupported model input rank: {spec['rank']}")


def _find_all_keras_models(ckpt_root: Path, weights_name_override: Optional[str]) -> List[str]:
    names: List[str] = []
    if not ckpt_root.exists():
        return names
    for p in ckpt_root.iterdir():
        if not p.is_dir():
            continue
        try:
            run_dir = pick_latest_run_dir(p)
        except FileNotFoundError:
            continue
        weights_name = weights_name_override or f"{p.name}_best.keras"
        if (run_dir / weights_name).exists():
            names.append(p.name)
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Keras HAR models on UP-Fall windowed pose tensors.")
    parser.add_argument("--models", nargs="+", default=None, help="Model names to evaluate (folder under --ckpt-root).")
    parser.add_argument("--all", action="store_true", help="Evaluate all Keras models found under --ckpt-root.")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 1)")
    parser.add_argument("--test-subjects", type=str, default="1-1", help="Test subject range like '1-5'")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--ckpt-root", type=str, default="models", help="Checkpoint root (default: models)")
    parser.add_argument(
        "--ckpt",
        nargs="*",
        default=None,
        help="Optional per-model run folder overrides. Use: --ckpt model=2026-... model=latest.",
    )
    parser.add_argument(
        "--weights-name",
        type=str,
        default=None,
        help="Override weights filename. If omitted uses '<model>_best.keras'.",
    )
    parser.add_argument("--out-dir", type=str, default="eval_outputs", help="Output directory")
    parser.add_argument("--use-conf", action="store_true", help="Use confidence channel (x,y,conf).")
    parser.add_argument("--no-conf", action="store_true", help="Disable confidence channel (use x,y only).")

    # Binary decision options
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
        choices=["center_scale", "paper_rp"],
        help="Normalisation mode when --normalize 1. center_scale=legacy translation+scale; paper_rp=paper Relative Position (translation only).",
    )
    parser.add_argument("--add-vel", type=int, default=1, help="Add velocity channels vx, vy (0/1).")
    parser.add_argument("--add-acc", type=int, default=1, help="Add acceleration channels ax, ay (0/1).")
    parser.add_argument("--add-global", type=int, default=1, help="Add global features (0/1).")
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

    ckpt_root = Path(args.ckpt_root)
    ckpt_overrides = parse_ckpt_overrides(args.ckpt)

    if args.all:
        model_list = _find_all_keras_models(ckpt_root, args.weights_name)
        if not model_list:
            raise SystemExit(f"No Keras models found under {ckpt_root.as_posix()}")
    else:
        if args.models is None or len(args.models) == 0:
            raise SystemExit("You must pass --models <one or more> or use --all.")
        model_list = [m.strip() for m in args.models]

    use_conf = True
    if args.no_conf:
        use_conf = False
    if args.use_conf:
        use_conf = True

    test_subjects = parse_range(args.test_subjects)

    # Output folder
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    models_tag = slug_models(model_list)
    base_out = Path(args.out_dir).resolve()
    out_dir = base_out / f"{ts}__models_{models_tag}"
    plots_dir = out_dir / "plots"
    ensure_dir(out_dir)
    ensure_dir(plots_dir)

    datasets_root = PROJECT_ROOT.parent.parent / "Datasets"
    OUTPUT_ROOT = datasets_root / "UPFall_keypoints" / "outputs_npz"
    test_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=args.camera, subjects=test_subjects)
    if not test_npzs:
        raise RuntimeError("No test NPZs found. Check OUTPUT_ROOT, camera, and test subjects.")

    label_names_default = get_new_label_names("1-11")
    test_npzs = sanitize_npz_labels(test_npzs, out_dir / "sanitized_npzs", label_names_default, "1-11")

    label_convention, _label_stats = detect_label_convention_from_npzs(test_npzs)
    new_label_names = get_new_label_names(label_convention)
    print(f"[labels] Using raw convention: {label_convention} | New labels: {new_label_names}")

    fall_class_ids_0based = [FALL_CLASS_ID]

    extra = {}
    if str(args.label_mode).lower() == "hybrid_center_fallpct":
        extra["fall_ids_0based"] = fall_class_ids_0based
        extra["fall_pct"] = float(args.fall_pct)

    X_test, y_test_tags, _T_used = load_windows_from_npzs(
        test_npzs,
        T=int(args.T),
        use_conf=use_conf,
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
        label_convention=label_convention,
        rp_center_mode=str(args.rp_center_mode),
        rp_img_w=args.rp_img_w,
        rp_img_h=args.rp_img_h,
        **extra,
    )

    y_test = y_test_tags.astype(np.int64, copy=False)
    if int(y_test.max()) >= int(NUM_CLASSES_MERGED) or int(y_test.min()) < 0:
        raise RuntimeError(f"Unexpected label id outside [0,{NUM_CLASSES_MERGED-1}]. Check label remap.")

    X_test = X_test.astype(np.float32, copy=False).reshape(X_test.shape[0], X_test.shape[1], -1)

    # Pre-load tune data if requested (base version, per-model adjustment happens later)
    X_tune_base = None
    y_tune_base = None
    if args.tune_subjects is not None and str(args.binary_mode).lower() == "threshold" and args.threshold is None:
        tune_subjects = parse_range(args.tune_subjects)
        tune_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=args.camera, subjects=tune_subjects)
        if not tune_npzs:
            raise RuntimeError("No tune NPZs found. Check OUTPUT_ROOT, camera, and tune subjects.")
        tune_npzs = sanitize_npz_labels(tune_npzs, out_dir / "sanitized_npzs_tune", label_names_default, "1-11")
        X_tune, y_tune_tags, _ = load_windows_from_npzs(
            tune_npzs,
            T=int(args.T),
            use_conf=use_conf,
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
            label_convention=label_convention,
            rp_center_mode=str(args.rp_center_mode),
            rp_img_w=args.rp_img_w,
            rp_img_h=args.rp_img_h,
            **extra,
        )
        X_tune_base = X_tune.astype(np.float32, copy=False).reshape(X_tune.shape[0], X_tune.shape[1], -1)
        y_tune_base = y_tune_tags.astype(np.int64, copy=False)

    summary_rows: List[Dict[str, object]] = []
    overall_rows: List[Dict[str, object]] = []
    f1_rows: List[Dict[str, object]] = []

    for m in model_list:
        if args.weights_name:
            weights_name = str(args.weights_name)
        else:
            weights_name = f"{m}_best.keras"

        model_dir = ckpt_root / m
        run_dir = resolve_run_dir(model_dir, ckpt_overrides.get(m.lower()))
        ckpt_path = run_dir / weights_name

        print(f"[{m}] Using run folder: {run_dir.name}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Weights not found for {m}: {ckpt_path.as_posix()}")

        model = tf.keras.models.load_model(str(ckpt_path), compile=False)
        spec = get_model_input_spec(model)
        out_dim = get_model_output_dim(model)

        X_model, action = _prepare_inputs_for_model(X_test, spec)
        if action not in {"keep", "match"}:
            print(f"[{m}] Input sequence adjusted: {action}")

        if out_dim is None:
            num_classes_eval = int(y_test.max() + 1)
        else:
            num_classes_eval = int(out_dim)
        if int(y_test.max()) >= num_classes_eval:
            raise RuntimeError(f"[{m}] Label id >= num_classes (labels max={int(y_test.max())}, num_classes={num_classes_eval}).")

        preds_out = model.predict(X_model, batch_size=int(args.batch_size), verbose=0)
        if isinstance(preds_out, (list, tuple)):
            if len(preds_out) != 1:
                raise ValueError(f"[{m}] Only single-output models are supported.")
            preds_out = preds_out[0]

        probs_test = _to_probs(np.asarray(preds_out))
        if probs_test.ndim != 2:
            raise ValueError(f"[{m}] Model output must be rank-2 (batch, num_classes). Got shape {probs_test.shape}")
        if int(probs_test.shape[1]) != int(num_classes_eval):
            raise RuntimeError(
                f"[{m}] Output dim mismatch: model outputs {int(probs_test.shape[1])} classes, "
                f"expected {int(num_classes_eval)}."
            )

        y_true = y_test
        y_pred = probs_test.argmax(axis=1).astype(int)

        labels_all = list(range(num_classes_eval))
        cm_counts = confusion_matrix(y_true, y_pred, labels=labels_all).astype(np.float64)
        row_sums = cm_counts.sum(axis=1, keepdims=True) + 1e-9
        cm = cm_counts / row_sums

        if len(new_label_names) >= num_classes_eval:
            cm_names = new_label_names[:num_classes_eval]
        else:
            cm_names = [str(i) for i in labels_all]

        cm_csv = out_dir / f"confusion_matrix_{m}.csv"
        pd.DataFrame(cm, index=cm_names, columns=cm_names).to_csv(cm_csv)
        make_cm_plot(cm, cm_names, plots_dir / f"confusion_matrix_{m}.png", title=f"Confusion Matrix: {m}")

        support = cm_counts.sum(axis=1)
        valid = support > 0
        tp = np.diag(cm_counts)
        pred_support = cm_counts.sum(axis=0)
        recall = tp / (support + 1e-12)
        precision = tp / (pred_support + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        specificity = specificity_from_cm(cm_counts)

        total = float(np.sum(cm_counts))
        acc = float(np.sum(tp) / total) if total > 0 else 0.0
        macro_recall = float(np.mean(recall[valid])) if np.any(valid) else 0.0
        macro_precision = float(np.mean(precision[valid])) if np.any(valid) else 0.0
        macro_specificity = float(np.mean(specificity[valid])) if np.any(valid) else 0.0
        macro_f1 = float(np.mean(f1[valid])) if np.any(valid) else 0.0

        for lab, f1v in zip(labels_all, f1):
            name = cm_names[lab] if 0 <= lab < len(cm_names) else str(lab)
            f1_rows.append({"model": m, "class_id": int(lab), "class_name": name, "f1": float(f1v)})

        overall_rows.append({
            "model": m,
            "accuracy": float(acc) * 100.0,
            "recall": float(macro_recall) * 100.0,
            "specificity": float(macro_specificity) * 100.0,
            "precision": float(macro_precision) * 100.0,
            "f1_score": float(macro_f1) * 100.0,
        })

        y_true_bin = collapse_to_binary(y_true, fall_class_ids_0based)
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
            if args.threshold is not None:
                thr = float(args.threshold)
            elif X_tune_base is not None and y_tune_base is not None:
                X_tune_model, _action_t = _prepare_inputs_for_model(X_tune_base, spec)
                preds_tune = model.predict(X_tune_model, batch_size=int(args.batch_size), verbose=0)
                if isinstance(preds_tune, (list, tuple)):
                    if len(preds_tune) != 1:
                        raise ValueError(f"[{m}] Only single-output models are supported.")
                    preds_tune = preds_tune[0]
                probs_tune = _to_probs(np.asarray(preds_tune))
                y_tune_true = y_tune_base
                y_tune_bin = collapse_to_binary(y_tune_true, fall_class_ids_0based)
                p_fall_tune = p_fall_from_probs(probs_tune, fall_class_ids_0based).astype(np.float32)

                thr, tuned_prec, tuned_rec, tuned_fbeta = pick_threshold_fbeta(
                    y_tune_bin, p_fall_tune, beta=float(args.beta)
                )
                tuned_thr = thr
            else:
                thr = 0.5

            y_pred_bin = (p_fall_test >= float(thr)).astype(int)

        pr, rc, f1b, _ = precision_recall_fscore_support(
            y_true_bin, y_pred_bin, labels=[0, 1], average=None, zero_division=0
        )

        summary_rows.append({
            "model": m,
            "n_samples": int(len(y_true)),
            "params_m": float(count_params_m_keras(model)),
            "macro_f1": float(macro_f1),
            "binary_mode": str(args.binary_mode).lower(),
            "p_fall_source": p_fall_source,
            "threshold": float(thr) if thr is not None else None,
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

        # Avoid TF graph/resource buildup across multiple models.
        del model
        tf.keras.backend.clear_session()

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    f1_long = pd.DataFrame(f1_rows).sort_values(["model", "class_id"]).reset_index(drop=True)
    overall_df = summary_df[["model"]].merge(pd.DataFrame(overall_rows), on="model", how="left").round(3)

    summary_csv = out_dir / "metrics_summary.csv"
    f1_csv = out_dir / "f1_per_class.csv"
    summary_df.to_csv(summary_csv, index=False)
    f1_long.to_csv(f1_csv, index=False)

    make_plots(summary_df, plots_dir)
    report_path = out_dir / "report.html"
    make_html_report(summary_df, overall_df, f1_long, plots_dir, report_path, model_list)

    print("\nOverall metrics (%):")
    print(overall_df.to_string(index=False))
    print(f"Saved: {summary_csv}")
    print(f"Saved: {f1_csv}")
    print(f"Saved: {report_path}")
    print(f"Plots in: {plots_dir.as_posix()}")


if __name__ == "__main__":
    main()
