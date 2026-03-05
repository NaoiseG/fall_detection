#!/usr/bin/env python
import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import tensorflow as tf


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
RARE_CLASS_IDS_MERGED = [0, 4]


def parse_range(r: str):
    a, b = r.split("-")
    a, b = int(a), int(b)
    return range(a, b + 1)


def compute_class_weights(
    y: np.ndarray,
    num_classes: int,
    mode: str = "inv_sqrt",
    eps: float = 1e-6,
    rare_boost: float = 1.0,
    rare_class_ids=None,
) -> np.ndarray:
    mode = str(mode).lower().strip()
    if mode == "none":
        return np.ones((int(num_classes),), dtype=np.float32)

    counts = np.bincount(y.astype(np.int64, copy=False), minlength=int(num_classes)).astype(np.float64)
    safe = np.maximum(counts, 1.0)

    if mode == "inv":
        w = 1.0 / (safe + float(eps))
    elif mode == "inv_sqrt":
        w = 1.0 / (np.sqrt(safe) + float(eps))
    else:
        raise ValueError(f"Unknown class weight mode: {mode}")

    if rare_class_ids and float(rare_boost) != 1.0:
        for cid in rare_class_ids:
            if 0 <= int(cid) < int(num_classes):
                w[int(cid)] *= float(rare_boost)

    w = w / (np.mean(w) + float(eps))
    return w.astype(np.float32, copy=False)


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

def replace_head_if_needed(orig_model: tf.keras.Model, num_classes: int):
    out_dim = get_model_output_dim(orig_model)
    if out_dim == num_classes:
        return orig_model, False

    if len(orig_model.layers) >= 2:
        feats = orig_model.layers[-2].output
    else:
        feats = orig_model.output
    logits = tf.keras.layers.Dense(num_classes, name="classifier")(feats)
    new_model = tf.keras.Model(inputs=orig_model.inputs, outputs=logits, name=f"{orig_model.name}_head7")
    return new_model, True


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


def update_confusion_matrix(cm: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    y_true = y_true.reshape(-1).astype(np.int64, copy=False)
    y_pred = y_pred.reshape(-1).astype(np.int64, copy=False)
    idx = y_true * int(num_classes) + y_pred
    cm += np.bincount(idx, minlength=int(num_classes) * int(num_classes)).reshape(int(num_classes), int(num_classes))


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    support = cm.sum(axis=1)
    pred_support = cm.sum(axis=0)
    tp = np.diag(cm)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=np.float32), where=support > 0)
    precision = np.divide(tp, pred_support, out=np.zeros_like(tp, dtype=np.float32), where=pred_support > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(tp, dtype=np.float32), where=denom > 0)
    mask = support > 0
    if not np.any(mask):
        return 0.0
    return float(f1[mask].mean())


def macro_recall_from_confusion(cm: np.ndarray) -> float:
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=np.float32), where=support > 0)
    mask = support > 0
    if not np.any(mask):
        return 0.0
    return float(recall[mask].mean())


def fall_fbeta_from_confusion(cm: np.ndarray, fall_ids=None, beta: float = 2.0, fall_class_id: int = FALL_CLASS_ID) -> float:
    beta = float(beta)
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError(f"beta must be finite and > 0, got {beta}")

    num_classes = int(cm.shape[0])
    fall_ids = fall_ids if fall_ids else [int(fall_class_id)]
    fall_ids = sorted({int(i) for i in fall_ids if 0 <= int(i) < num_classes})
    if not fall_ids:
        return 0.0

    fall_idx = np.array(fall_ids, dtype=np.int64)
    tp = cm[np.ix_(fall_idx, fall_idx)].sum()
    fn = cm[fall_idx, :].sum() - tp

    nonfall_idx = np.array([i for i in range(num_classes) if i not in fall_ids], dtype=np.int64)
    if nonfall_idx.size > 0:
        fp = cm[np.ix_(nonfall_idx, fall_idx)].sum()
    else:
        fp = 0

    denom_p = tp + fp
    denom_r = tp + fn
    precision = (tp / denom_p) if denom_p > 0 else 0.0
    recall = (tp / denom_r) if denom_r > 0 else 0.0

    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom <= 0.0:
        return 0.0
    return float((1.0 + beta2) * precision * recall / denom)


def selection_score_from_confusion(
    cm: np.ndarray,
    selection_metric: str,
    metric_weights: Optional[np.ndarray] = None,
    fall_ids=None,
    selection_w: float = 0.7,
    selection_beta: float = 2.0,
):
    selection_metric = str(selection_metric)
    acc = float(np.trace(cm) / max(cm.sum(), 1))
    macro_f1 = macro_f1_from_confusion(cm)
    macro_recall = macro_recall_from_confusion(cm)
    fall_fbeta = fall_fbeta_from_confusion(cm, fall_ids=fall_ids, beta=float(selection_beta))
    w = max(0.0, min(1.0, float(selection_w)))
    composite = w * fall_fbeta + (1.0 - w) * macro_f1

    if selection_metric == "composite_fall_fbeta_macro_f1":
        score = composite
    elif selection_metric == "macro_f1":
        score = macro_f1
    elif selection_metric == "macro_recall":
        score = macro_recall
    elif selection_metric == "inv_freq_recall":
        if metric_weights is None:
            score = macro_recall
        else:
            support = cm.sum(axis=1)
            tp = np.diag(cm)
            recall = np.divide(tp, support, out=np.zeros_like(tp, dtype=np.float32), where=support > 0)
            mask = support > 0
            wts = metric_weights.astype(np.float32, copy=False).reshape(-1)
            if mask.any() and wts.sum() > 0:
                wts = wts / float(wts.sum())
                score = float((recall * wts)[mask].sum())
            else:
                score = float(recall[mask].mean()) if mask.any() else 0.0
    elif selection_metric == "acc":
        score = acc
    else:
        raise ValueError(f"Unknown selection_metric: {selection_metric}")

    details = {
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "fall_fbeta": fall_fbeta,
        "composite": composite,
        "selection_w": w,
        "selection_beta": float(selection_beta),
    }
    return float(score), details

def make_train_dataset(X, y, batch_size: int, balanced_sampling: bool, seed: Optional[int] = None):
    if not balanced_sampling:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.shuffle(min(len(y), 10000), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, None

    class_ids = np.unique(y)
    datasets = []
    for cid in class_ids:
        idx = np.where(y == cid)[0]
        if idx.size == 0:
            continue
        ds_c = tf.data.Dataset.from_tensor_slices((X[idx], y[idx]))
        ds_c = ds_c.shuffle(len(idx), seed=seed, reshuffle_each_iteration=True).repeat()
        datasets.append(ds_c)

    if not datasets:
        raise RuntimeError("No samples available for balanced sampling.")

    sample_fn = getattr(tf.data.Dataset, "sample_from_datasets", None)
    if sample_fn is None:
        sample_fn = tf.data.experimental.sample_from_datasets

    ds = sample_fn(datasets, weights=None, seed=seed)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    steps = math.ceil(len(y) / float(batch_size))
    return ds, steps


def make_eval_dataset(X, y, batch_size: int):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def compute_l2_loss(model: tf.keras.Model, weight_decay: float) -> float:
    if float(weight_decay) <= 0.0:
        return 0.0
    weights = [w for w in model.trainable_weights if len(w.shape) > 1]
    if not weights:
        return 0.0
    l2 = tf.add_n([tf.nn.l2_loss(w) for w in weights])
    return float(weight_decay) * float(l2.numpy())


class EvalAndCheckpoint(tf.keras.callbacks.Callback):
    def __init__(
        self,
        val_ds,
        num_classes: int,
        selection_metric: str,
        selection_w: float,
        selection_beta: float,
        metric_weights: Optional[np.ndarray],
        class_weights: Optional[np.ndarray],
        ckpt_path: Path,
        use_l2: bool,
        weight_decay: float,
        base_loss_fn,
        fall_ids=None,
    ):
        super().__init__()
        self.val_ds = val_ds
        self.num_classes = int(num_classes)
        self.selection_metric = selection_metric
        self.selection_w = selection_w
        self.selection_beta = selection_beta
        self.metric_weights = metric_weights
        self.class_weights = class_weights
        self.ckpt_path = ckpt_path
        self.use_l2 = use_l2
        self.weight_decay = weight_decay
        self.base_loss_fn = base_loss_fn
        self.fall_ids = fall_ids if fall_ids else [FALL_CLASS_ID]

        self.best_score = -1.0
        self.best_epoch = -1
        self.best_val_loss = float("inf")
        self.best_val_acc = -1.0
    def on_epoch_end(self, epoch, logs=None):
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        total_loss = 0.0
        total_count = 0

        class_w = None
        if self.class_weights is not None:
            class_w = tf.constant(self.class_weights, dtype=tf.float32)

        for batch_x, batch_y in self.val_ds:
            logits = self.model(batch_x, training=False)
            if len(logits.shape) != 2:
                raise ValueError(f"Model output must be rank-2 (batch, num_classes); got shape {logits.shape}")

            losses = self.base_loss_fn(batch_y, logits)
            losses = tf.reshape(losses, (-1,))
            if class_w is not None:
                w = tf.gather(class_w, tf.cast(batch_y, tf.int32))
                losses = losses * tf.cast(w, losses.dtype)
            batch_loss = tf.reduce_sum(losses)
            total_loss += float(batch_loss.numpy())
            total_count += int(batch_y.shape[0])

            preds = tf.argmax(logits, axis=-1)
            update_confusion_matrix(cm, batch_y.numpy(), preds.numpy(), self.num_classes)

        val_loss = total_loss / max(total_count, 1)
        if self.use_l2 and float(self.weight_decay) > 0.0:
            val_loss += compute_l2_loss(self.model, self.weight_decay)

        val_acc = float(np.trace(cm) / max(cm.sum(), 1))
        val_score, details = selection_score_from_confusion(
            cm=cm,
            selection_metric=self.selection_metric,
            metric_weights=self.metric_weights,
            fall_ids=self.fall_ids,
            selection_w=self.selection_w,
            selection_beta=self.selection_beta,
        )

        if val_score > self.best_score:
            self.best_score = float(val_score)
            self.best_epoch = int(epoch + 1)
            self.best_val_loss = float(val_loss)
            self.best_val_acc = float(val_acc)
            self.model.save(self.ckpt_path)

        tr_loss = float(logs.get("loss", 0.0)) if logs else 0.0
        tr_acc = float(logs.get("acc", 0.0)) if logs else 0.0

        if self.selection_metric == "composite_fall_fbeta_macro_f1":
            mf1 = float(details.get("macro_f1", 0.0))
            fbeta = float(details.get("fall_fbeta", 0.0))
            w_used = float(details.get("selection_w", self.selection_w))
            beta_used = float(details.get("selection_beta", self.selection_beta))
            print(
                f"Epoch {epoch + 1:02d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
                f"macro_f1 {mf1:.3f} fall_fbeta(beta={beta_used:.2f}) {fbeta:.3f} | "
                f"composite(w={w_used:.2f}) {val_score:.3f}"
            )
        else:
            print(
                f"Epoch {epoch + 1:02d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.3f} | "
                f"score {val_score:.3f} ({self.selection_metric})"
            )


class UnfreezeBackboneCallback(tf.keras.callbacks.Callback):
    def __init__(self, base_model: tf.keras.Model, unfreeze_epoch: int, compile_kwargs: dict):
        super().__init__()
        self.base_model = base_model
        self.unfreeze_epoch = int(unfreeze_epoch)
        self.compile_kwargs = compile_kwargs
        self.has_unfroze = False

    def on_epoch_begin(self, epoch, logs=None):
        if self.has_unfroze:
            return
        if int(epoch + 1) == self.unfreeze_epoch:
            for layer in self.base_model.layers:
                layer.trainable = True
            self.model.compile(**self.compile_kwargs)
            self.has_unfroze = True
            print(f"[freeze] Unfroze backbone at epoch {self.unfreeze_epoch}.")

def main():
    parser = argparse.ArgumentParser(description="Train a Keras fall/HAR classifier on keypoint windows.")
    parser.add_argument("--base-model", type=str, required=True, help="Path to .keras base model.")
    parser.add_argument(
        "--reinit-weights",
        action="store_true",
        help="Reinitialize weights (train same architecture from scratch).",
    )
    parser.add_argument("--camera", type=int, default=1, help="Camera index to train on (default: 1)")
    parser.add_argument("--train-subjects", type=str, default="16-17", help="Train subject range like '1-12'")
    parser.add_argument("--val-subjects", type=str, default="1-1", help="Val subject range like '13-16'")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze all loaded layers except new projection/head.")
    parser.add_argument("--unfreeze-at-epoch", type=int, default=-1, help="Unfreeze backbone at epoch N (default: -1)")
    parser.add_argument("--class-weight-mode", type=str, default="inv_sqrt", choices=["none", "inv", "inv_sqrt"])
    parser.add_argument("--rare-class-boost", type=float, default=1.0, help="Boost rare classes [0,4].")
    parser.add_argument("--balanced-sampling", type=int, default=0, help="Use balanced sampling (0/1).")
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="composite_fall_fbeta_macro_f1",
        choices=[
            "composite_fall_fbeta_macro_f1",
            "macro_f1",
            "macro_recall",
            "inv_freq_recall",
            "acc",
        ],
    )
    parser.add_argument("--selection-w", type=float, default=0.7, help="Weight for composite metric (default: 0.7)")
    parser.add_argument("--selection-beta", type=float, default=2.0, help="Beta for fall Fbeta (default: 2.0)")

    # Data preprocessing options
    parser.add_argument("--use-conf", type=int, default=1, help="Include keypoint confidence channel (0/1).")
    parser.add_argument("--normalize", type=int, default=1, help="Normalise pose per frame (0/1).")
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="center_scale",
        choices=["center_scale", "paper_rp"],
        help="Normalisation mode when --normalize 1.",
    )
    parser.add_argument("--add-vel", type=int, default=1, help="Add velocity channels vx, vy (0/1).")
    parser.add_argument("--add-acc", type=int, default=1, help="Add acceleration channels ax, ay (0/1).")
    parser.add_argument("--add-global", type=int, default=1, help="Add global features (0/1).")
    parser.add_argument("--conf-thres", type=float, default=0.2, help="Conf threshold for missing joints.")
    parser.add_argument("--max-interp-gap", type=int, default=5, help="Max gap for interpolation.")
    parser.add_argument(
        "--missing-mode",
        type=str,
        default="conf_thres",
        choices=["conf_thres", "zeros_only", "conf_or_zeros"],
    )
    parser.add_argument(
        "--interp-mode",
        type=str,
        default="short_gap_hold",
        choices=["short_gap_hold", "paper_group_linear"],
    )
    parser.add_argument("--interp-group", type=int, default=100, help="Group size for paper_group_linear.")
    parser.add_argument(
        "--rp-center-mode",
        type=str,
        default="auto",
        choices=["auto", "normalized_01", "pixel"],
    )
    parser.add_argument("--rp-img-w", type=int, default=None, help="Image width for paper_rp.")
    parser.add_argument("--rp-img-h", type=int, default=None, help="Image height for paper_rp.")
    parser.add_argument("--T", type=int, default=64, help="Sliding window length T.")
    parser.add_argument("--stride", type=int, default=16, help="Sliding window stride.")
    parser.add_argument(
        "--label-mode",
        type=str,
        default="center",
        choices=["center", "majority", "hybrid_center_fallpct"],
    )
    parser.add_argument("--fall-pct", type=float, default=0.25, help="Used when label-mode=hybrid_center_fallpct.")
    parser.add_argument("--min-valid-frac", type=float, default=0.3, help="Min fraction of valid joints per frame.")
    parser.add_argument("--add-mask-channel", type=int, default=1, help="Append mask channel (0/1).")
    parser.add_argument("--drop-ambig-share", type=float, default=0.6, help="Drop ambiguous windows below this share.")
    parser.add_argument("--drop-ambig-nonfall-only", type=int, default=1, help="Drop ambiguous windows only if non-fall.")

    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    print("Run ID:", run_id)

    datasets_root = PROJECT_ROOT.parent.parent / "Datasets"
    OUTPUT_ROOT = datasets_root / "UPFall_keypoints" / "outputs_npz"
    ckpt_root = PROJECT_ROOT / "models"

    train_subjects = parse_range(args.train_subjects)
    val_subjects = parse_range(args.val_subjects)

    train_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=args.camera, subjects=train_subjects)
    val_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=args.camera, subjects=val_subjects)

    if not train_npzs:
        raise RuntimeError("No training NPZs found. Check OUTPUT_ROOT, camera, and train subjects.")
    if not val_npzs:
        raise RuntimeError("No validation NPZs found. Check OUTPUT_ROOT, camera, and val subjects.")

    print("Train sequences:", len(train_npzs))
    print("Val sequences:", len(val_npzs))

    base_model_path = Path(args.base_model)
    if not base_model_path.exists():
        candidate = PROJECT_ROOT / base_model_path
        if candidate.exists():
            base_model_path = candidate
        else:
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
    if base_model_path.suffix.lower() == ".keras" and base_model_path.suffix != ".keras":
        lower_path = base_model_path.with_suffix(".keras")
        if not lower_path.exists():
            shutil.copy2(base_model_path, lower_path)
        base_model_path = lower_path

    model_name = base_model_path.stem
    run_dir = ckpt_root / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f"{model_name}_best.keras"

    label_names_default = get_new_label_names("1-11")
    train_npzs = sanitize_npz_labels(train_npzs, run_dir / "sanitized_npzs", label_names_default, "1-11")
    val_npzs = sanitize_npz_labels(val_npzs, run_dir / "sanitized_npzs", label_names_default, "1-11")

    label_convention, label_stats = detect_label_convention_from_npzs(train_npzs + val_npzs)
    new_label_names = get_new_label_names(label_convention)
    print(f"[labels] Using raw convention: {label_convention} | New labels: {new_label_names}")

    # Save label map next to checkpoint
    try:
        label_map_path = run_dir / "label_map_fallmerged.json"
        if not label_map_path.exists():
            label_map_path.write_text(
                json.dumps(
                    {"label_scheme": "fall_merged_7c", "label_convention": label_convention, "id_to_name": new_label_names},
                    indent=2,
                ),
                encoding="utf-8",
            )
    except Exception as e:
        print(f"Warning: failed to write label map JSON: {e}")

    use_conf = bool(args.use_conf)
    normalize = bool(args.normalize)
    add_vel = bool(args.add_vel)
    add_acc = bool(args.add_acc)
    add_global = bool(args.add_global)
    add_mask_channel = bool(args.add_mask_channel)

    fall_ids_0based = [FALL_CLASS_ID]
    X_train, y_train_tags, T_used = load_windows_from_npzs(
        train_npzs,
        T=int(args.T),
        use_conf=use_conf,
        normalize=normalize,
        normalize_mode=str(args.normalize_mode),
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        conf_thres=float(args.conf_thres),
        max_interp_gap=int(args.max_interp_gap),
        missing_mode=str(args.missing_mode),
        interp_mode=str(args.interp_mode),
        interp_group=int(args.interp_group),
        stride=int(args.stride),
        label_mode=str(args.label_mode),
        min_valid_frac=float(args.min_valid_frac),
        add_mask_channel=add_mask_channel,
        fall_ids_0based=fall_ids_0based,
        fall_pct=float(args.fall_pct),
        drop_ambig_share=float(args.drop_ambig_share),
        drop_ambig_nonfall_only=bool(args.drop_ambig_nonfall_only),
        label_convention=label_convention,
        rp_center_mode=str(args.rp_center_mode),
        rp_img_w=args.rp_img_w,
        rp_img_h=args.rp_img_h,
    )

    X_val, y_val_tags, _ = load_windows_from_npzs(
        val_npzs,
        T=int(T_used),
        use_conf=use_conf,
        normalize=normalize,
        normalize_mode=str(args.normalize_mode),
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        conf_thres=float(args.conf_thres),
        max_interp_gap=int(args.max_interp_gap),
        missing_mode=str(args.missing_mode),
        interp_mode=str(args.interp_mode),
        interp_group=int(args.interp_group),
        stride=int(args.stride),
        label_mode=str(args.label_mode),
        min_valid_frac=float(args.min_valid_frac),
        add_mask_channel=add_mask_channel,
        fall_ids_0based=fall_ids_0based,
        fall_pct=float(args.fall_pct),
        drop_ambig_share=float(args.drop_ambig_share),
        drop_ambig_nonfall_only=bool(args.drop_ambig_nonfall_only),
        label_convention=label_convention,
        rp_center_mode=str(args.rp_center_mode),
        rp_img_w=args.rp_img_w,
        rp_img_h=args.rp_img_h,
    )

    y_train = y_train_tags.astype(np.int64, copy=False)
    y_val = y_val_tags.astype(np.int64, copy=False)

    num_classes = int(NUM_CLASSES_MERGED)
    if int(y_train.max()) >= num_classes or int(y_val.max()) >= num_classes:
        raise RuntimeError(f"Unexpected label id >= {num_classes}. Check label remap.")

    X_train = X_train.astype(np.float32, copy=False).reshape(X_train.shape[0], X_train.shape[1], -1)
    X_val = X_val.astype(np.float32, copy=False).reshape(X_val.shape[0], X_val.shape[1], -1)

    print("num_classes:", num_classes, "| T_used:", int(T_used))
    print("window:", int(T_used), "frames | stride:", int(args.stride))

    orig_model = tf.keras.models.load_model(str(base_model_path))
    spec = get_model_input_spec(orig_model)
    print("Base model input shape:", orig_model.input_shape, "| output shape:", orig_model.output_shape)

    if spec["rank"] == 3:
        if not args.reinit_weights:
            target_T = spec["T"]
            if target_T is not None and int(X_train.shape[1]) != int(target_T):
                X_train, action_tr = adjust_sequence_length(X_train, int(target_T))
                X_val, action_va = adjust_sequence_length(X_val, int(target_T))
                print(f"[seq] Adjusted sequence length: train {action_tr}, val {action_va}")
        input_shape_for_rebuild = (int(X_train.shape[1]), int(X_train.shape[2]))
    elif spec["rank"] == 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_val = X_val.reshape(X_val.shape[0], -1)
        print("[seq] Flattened windows to rank-2 inputs.")
        input_shape_for_rebuild = (int(X_train.shape[1]),)
    else:
        raise ValueError(f"Unsupported model input rank: {spec['rank']}")

    if args.reinit_weights:
        input_name = None
        if hasattr(orig_model, "input_names"):
            try:
                names = orig_model.input_names
                if isinstance(names, (list, tuple)) and names:
                    input_name = str(names[0])
            except Exception:
                input_name = None
        if input_name is None:
            try:
                input_name = str(orig_model.inputs[0].name).split(":")[0]
            except Exception:
                input_name = "input_sequence"

        new_inputs = tf.keras.Input(shape=input_shape_for_rebuild, name=input_name)
        input_tensors = new_inputs
        if isinstance(orig_model.input, (list, tuple)):
            if len(orig_model.input) != 1:
                raise ValueError("Only single-input models are supported for reinit-weights.")
            input_tensors = [new_inputs]
        orig_model = tf.keras.models.clone_model(orig_model, input_tensors=input_tensors)
        orig_model.build(new_inputs.shape)
        print(f"[model] Reinitialized base model weights with input shape {input_shape_for_rebuild}.")
        spec = get_model_input_spec(orig_model)

    F_new = int(X_train.shape[-1])
    F_old = spec["F"]

    base_model, head_replaced = replace_head_if_needed(orig_model, num_classes=num_classes)
    if head_replaced:
        print("[model] Replaced classification head with Dense(7) logits.")
    else:
        print("[model] Using existing 7-logit head.")

    if F_old is not None and int(F_old) != int(F_new):
        if spec["rank"] == 3:
            input_T = spec["T"] if spec["T"] is not None else None
            inputs = tf.keras.Input(shape=(input_T, F_new), name="input_sequence")
            x = tf.keras.layers.Dense(int(F_old), name="input_projection")(inputs)
            outputs = base_model(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{base_model.name}_proj")
        else:
            inputs = tf.keras.Input(shape=(F_new,), name="input_flat")
            x = tf.keras.layers.Dense(int(F_old), name="input_projection")(inputs)
            outputs = base_model(x)
            model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{base_model.name}_proj")
        print(f"[model] Added projection: F_new={F_new} -> F_old={F_old}")
    else:
        model = base_model
        if F_old is None:
            print("[model] Base model has flexible feature dim; no projection added.")
        else:
            print("[model] Feature dims match; no projection added.")

    if args.freeze_backbone:
        for layer in orig_model.layers:
            layer.trainable = False
        print("[freeze] Backbone frozen. New head/projection remain trainable.")

    base_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    train_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    )

    has_adamw = hasattr(tf.keras.optimizers, "AdamW")
    if has_adamw:
        optimizer = tf.keras.optimizers.AdamW(learning_rate=float(args.lr), weight_decay=float(args.weight_decay))
        use_l2 = False
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(args.lr))
        use_l2 = float(args.weight_decay) > 0.0
        if use_l2:
            def l2_reg():
                weights = [w for w in model.trainable_weights if len(w.shape) > 1]
                if not weights:
                    return 0.0
                return float(args.weight_decay) * tf.add_n([tf.nn.l2_loss(w) for w in weights])
            model.add_loss(l2_reg)

    model.compile(
        optimizer=optimizer,
        loss=train_loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    metric_weights_np = None
    if args.selection_metric == "inv_freq_recall":
        counts = np.bincount(y_train, minlength=int(num_classes)).astype(np.float32)
        w = np.zeros((num_classes,), dtype=np.float32)
        nz = counts > 0
        w[nz] = 1.0 / counts[nz]
        if float(w.sum()) > 0.0:
            w = w / float(w.sum())
            metric_weights_np = w
        print("Selection metric:", args.selection_metric, "(minority-upweighted)")
    elif args.selection_metric == "macro_f1":
        print("Selection metric:", args.selection_metric, "(equal weight per class)")
    elif args.selection_metric == "macro_recall":
        print("Selection metric:", args.selection_metric, "(equal weight per class)")
    else:
        print("Selection metric:", args.selection_metric, "(overall accuracy/composite)")

    class_weights_np = None
    class_weight_dict = None
    if str(args.class_weight_mode).lower().strip() != "none":
        class_weights_np = compute_class_weights(
            y_train,
            num_classes=int(num_classes),
            mode=str(args.class_weight_mode),
            rare_boost=float(args.rare_class_boost),
            rare_class_ids=RARE_CLASS_IDS_MERGED,
        )
        class_weight_dict = {int(i): float(w) for i, w in enumerate(class_weights_np.tolist())}
        counts_dbg = np.bincount(y_train, minlength=int(num_classes)).astype(np.int64)
        counts_dbg_d = {int(i): int(c) for i, c in enumerate(counts_dbg.tolist()) if int(c) > 0}
        print("Train window counts:", counts_dbg_d)
        print("CE class weights:", [float(x) for x in class_weights_np.tolist()])

    balanced_sampling = bool(args.balanced_sampling)
    train_ds, steps_per_epoch = make_train_dataset(
        X_train, y_train, batch_size=int(args.batch_size), balanced_sampling=balanced_sampling
    )
    val_ds = make_eval_dataset(X_val, y_val, batch_size=int(args.batch_size))

    eval_cb = EvalAndCheckpoint(
        val_ds=val_ds,
        num_classes=num_classes,
        selection_metric=str(args.selection_metric),
        selection_w=float(args.selection_w),
        selection_beta=float(args.selection_beta),
        metric_weights=metric_weights_np,
        class_weights=class_weights_np,
        ckpt_path=ckpt_path,
        use_l2=use_l2,
        weight_decay=float(args.weight_decay),
        base_loss_fn=base_loss_fn,
        fall_ids=[FALL_CLASS_ID],
    )

    callbacks = [eval_cb]

    if args.freeze_backbone and int(args.unfreeze_at_epoch) > 0:
        compile_kwargs = {
            "optimizer": optimizer,
            "loss": train_loss_fn,
            "metrics": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        }
        callbacks.append(UnfreezeBackboneCallback(orig_model, int(args.unfreeze_at_epoch), compile_kwargs))

    fit_kwargs = {
        "epochs": int(args.epochs),
        "callbacks": callbacks,
        "verbose": 0,
        "class_weight": class_weight_dict,
    }
    if steps_per_epoch is not None:
        fit_kwargs["steps_per_epoch"] = int(steps_per_epoch)

    model.fit(train_ds, **fit_kwargs)

    print(
        f"Best checkpoint: {ckpt_path.as_posix()} | "
        f"epoch {eval_cb.best_epoch} | score {eval_cb.best_score:.4f} | "
        f"val acc {eval_cb.best_val_acc:.4f} | val loss {eval_cb.best_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()
