from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import json
import math
import pickle
import time

import numpy as np


FALL_CLASS_ID = 0


@dataclass
class RunResult:
    model: str
    best_val_score: float
    best_val_acc: float
    best_val_loss: float
    best_epoch: int
    final_val_score: float
    final_val_acc: float
    final_val_loss: float
    params_m: float
    train_seconds: float
    ckpt_path: str


def windows_to_sklearn_features(
    X: np.ndarray,
    mode: str = "flatten",
    *,
    paper_num_skeletons: int = 3,
    paper_num_features: int = 51,
    keypoints: int = 17,
) -> np.ndarray:
    """
    Convert window tensors to a 2D feature matrix for sklearn models.

    Accepts:
      - (N,T,K,C) or (N,T,F)

    Modes:
      - flatten : flatten entire window (T*F)
      - center  : use the center frame only (F)
      - stats   : per-feature [mean, std, min, max] over time (4*F)
      - paper_windowing :
          Implements the feature-vector construction described in
          Sensors 2022 (AlphaPose + RF/SVM/MLP on UP-Fall) "windowed RF":

          * Select S skeletons (frames) evenly spaced within the window
            (first, middle, last when S=3).
          * Keep only the 51 pose features per skeleton (17 joints × [x,y,score]).
          * Output a flattened vector of size (S * 51).

          This matches the paper's best candidate for UP-Fall: W=2s => T=36 (18 fps),
          S=3 => 153 features per window.
    """
    mode = str(mode).lower().strip()

    if X.ndim == 4:
        # (N,T,K,C)
        N, T, K, C = X.shape
        if int(K) != int(keypoints) and mode == "paper_windowing":
            raise ValueError(f"paper_windowing expects K={keypoints} keypoints. Got K={K}")

        if mode == "paper_windowing":
            if int(C) < 3:
                raise ValueError(f"paper_windowing requires >=3 channels per keypoint (x,y,conf). Got C={C}")
            X3 = X[:, :, :, :3].reshape(int(N), int(T), int(keypoints) * 3)  # (N,T,51)
        else:
            X3 = X.reshape(int(N), int(T), int(K) * int(C))

    elif X.ndim == 3:
        # (N,T,F)
        N, T, F = X.shape
        if mode == "paper_windowing":
            if int(F) == int(paper_num_features):
                X3 = X  # already (N,T,51)
            else:
                if int(F) % int(keypoints) != 0:
                    raise ValueError(
                        f"paper_windowing expects features to be either 51 or (17*C). Got F={F} not divisible by {keypoints}."
                    )
                C = int(F) // int(keypoints)
                if int(C) < 3:
                    raise ValueError(f"paper_windowing requires >=3 channels per keypoint. Got inferred C={C}")
                X4 = X.reshape(int(N), int(T), int(keypoints), int(C))
                X3 = X4[:, :, :, :3].reshape(int(N), int(T), int(keypoints) * 3)  # (N,T,51)
        else:
            X3 = X
    else:
        raise ValueError(f"Expected X with ndim 3 or 4. Got shape {getattr(X, 'shape', None)}")

    N, T, F = X3.shape

    if mode == "flatten":
        return X3.reshape(int(N), int(T) * int(F))

    if mode == "center":
        return X3[:, int(T) // 2, :]

    if mode == "stats":
        mu = X3.mean(axis=1)
        sd = X3.std(axis=1)
        mn = X3.min(axis=1)
        mx = X3.max(axis=1)
        return np.concatenate([mu, sd, mn, mx], axis=1)

    if mode == "paper_windowing":
        S = int(paper_num_skeletons)
        if S <= 0:
            raise ValueError("paper_num_skeletons must be >= 1")
        if S > int(T):
            raise ValueError(f"paper_num_skeletons={S} cannot exceed window length T={T}")

        idx = np.linspace(0, int(T) - 1, num=int(S))
        idx = np.round(idx).astype(np.int64)
        idx = np.clip(idx, 0, int(T) - 1)

        uniq = np.unique(idx)
        if uniq.size < idx.size:
            idx = uniq
            if idx.size < S:
                pad = np.full((S - idx.size,), int(idx[-1]), dtype=np.int64)
                idx = np.concatenate([idx, pad], axis=0)

        Xs = X3[:, idx, :]  # (N,S,51)
        if int(paper_num_features) != 51:
            Xs = Xs[:, :, : int(paper_num_features)]
        return Xs.reshape(int(N), int(S) * int(Xs.shape[-1]))

    raise ValueError(f"Unknown sklearn feature mode: {mode}")


def _confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    y_true = y_true.astype(np.int64, copy=False).reshape(-1)
    y_pred = y_pred.astype(np.int64, copy=False).reshape(-1)
    num_classes = int(num_classes)
    if y_true.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    idx = y_true * num_classes + y_pred
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes).astype(np.int64, copy=False)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Macro F1 across classes with support>0."""
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square [C,C], got {tuple(cm.shape)}")

    support = cm.sum(axis=1).astype(np.float64, copy=False)  # (C,)
    pred_support = cm.sum(axis=0).astype(np.float64, copy=False)  # (C,)
    tp = np.diag(cm).astype(np.float64, copy=False)

    fp = pred_support - tp
    fn = support - tp

    denom = (2.0 * tp + fp + fn)
    f1 = np.zeros_like(tp, dtype=np.float64)
    nz = denom > 0
    f1[nz] = (2.0 * tp[nz]) / denom[nz]

    valid = support > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(f1[valid]))


def macro_recall_from_confusion(cm: np.ndarray) -> float:
    support = cm.sum(axis=1).astype(np.float64, copy=False)
    tp = np.diag(cm).astype(np.float64, copy=False)
    denom = support
    rec = np.zeros_like(tp, dtype=np.float64)
    nz = denom > 0
    rec[nz] = tp[nz] / denom[nz]
    valid = support > 0
    if not np.any(valid):
        return 0.0
    return float(np.mean(rec[valid]))


def inv_freq_recall_from_confusion(cm: np.ndarray, weights: Optional[np.ndarray]) -> float:
    # weights should sum to 1 across classes with support>0
    if weights is None:
        return macro_recall_from_confusion(cm)
    support = cm.sum(axis=1).astype(np.float64, copy=False)
    tp = np.diag(cm).astype(np.float64, copy=False)
    denom = support
    rec = np.zeros_like(tp, dtype=np.float64)
    nz = denom > 0
    rec[nz] = tp[nz] / denom[nz]
    valid = support > 0
    if not np.any(valid):
        return 0.0
    w = weights.astype(np.float64, copy=False)
    return float(np.sum(w[valid] * rec[valid]))


def binary_fbeta_from_confusion(cm: np.ndarray, pos_ids: List[int], beta: float = 1.0) -> float:
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be square")
    C = cm.shape[0]
    pos = np.array(sorted(set(int(i) for i in pos_ids)), dtype=np.int64)
    pos = pos[(pos >= 0) & (pos < C)]
    if pos.size == 0:
        return 0.0

    # Aggregate into binary confusion: positive = any pos class, negative = all others.
    # TP: true pos predicted pos, etc.
    tp = cm[np.ix_(pos, pos)].sum()
    fn = cm[np.ix_(pos, np.setdiff1d(np.arange(C), pos))].sum()
    fp = cm[np.ix_(np.setdiff1d(np.arange(C), pos), pos)].sum()

    tp = float(tp)
    fn = float(fn)
    fp = float(fp)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = float(beta) * float(beta)
    denom = (b2 * prec + rec)
    if denom <= 0:
        return 0.0
    return float((1 + b2) * prec * rec / denom)


def composite_fall_fbeta_macro_f1_from_confusion(
    cm: np.ndarray,
    *,
    fall_ids_0based: Optional[List[int]],
    w: float = 0.7,
    beta: float = 2.0,
) -> Tuple[float, float, float]:
    macro_f1 = macro_f1_from_confusion(cm)
    fall_fbeta = 0.0
    if fall_ids_0based is not None and len(fall_ids_0based) > 0:
        fall_fbeta = binary_fbeta_from_confusion(cm, pos_ids=fall_ids_0based, beta=float(beta))
    w = float(max(0.0, min(1.0, w)))
    composite = w * fall_fbeta + (1.0 - w) * macro_f1
    return float(composite), float(fall_fbeta), float(macro_f1)


def selection_score_from_confusion(
    cm: np.ndarray,
    *,
    selection_metric: str,
    metric_weights: Optional[np.ndarray],
    fall_ids_0based: Optional[List[int]],
    selection_w: float = 0.7,
    selection_beta: float = 2.0,
) -> float:
    selection_metric = str(selection_metric).lower().strip()
    if selection_metric == "macro_f1":
        return macro_f1_from_confusion(cm)
    if selection_metric == "macro_recall":
        return macro_recall_from_confusion(cm)
    if selection_metric == "inv_freq_recall":
        return inv_freq_recall_from_confusion(cm, weights=metric_weights)
    if selection_metric == "composite_fall_fbeta_macro_f1":
        composite, _, _ = composite_fall_fbeta_macro_f1_from_confusion(
            cm, fall_ids_0based=fall_ids_0based, w=float(selection_w), beta=float(selection_beta)
        )
        return float(composite)
    if selection_metric == "acc":
        total = float(cm.sum())
        if total <= 0:
            return 0.0
        return float(np.trace(cm) / total)
    raise ValueError(f"Unknown selection_metric: {selection_metric}")


def train_random_forest_once(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    ckpt_root: Path,
    run_id: str,
    label_convention: str,
    new_label_names: List[str],
    use_conf: bool,
    normalize: bool,
    add_vel: bool,
    add_acc: bool,
    add_global: bool,
    T_used: int,
    conf_thres: float,
    max_interp_gap: int,
    stride: int,
    label_mode: str,
    min_valid_frac: float,
    add_mask_channel: bool,
    drop_ambig_share: float,
    drop_ambig_nonfall_only: bool,
    fall_class_ids_raw: Optional[List[int]] = None,
    fall_ids_0based: Optional[List[int]] = None,
    selection_metric: str = "composite_fall_fbeta_macro_f1",
    selection_w: float = 0.7,
    selection_beta: float = 2.0,
    metric_weights_np: Optional[np.ndarray] = None,
    rf_feature_mode: str = "flatten",
    rf_paper_num_skeletons: int = 3,
    rf_paper_num_features: int = 51,
    rf_n_estimators: int = 300,
    rf_max_depth: Optional[int] = None,
    rf_max_features: str = "sqrt",
    rf_min_samples_split: int = 2,
    rf_min_samples_leaf: int = 1,
    rf_bootstrap: bool = True,
    rf_criterion: str = "gini",
    rf_class_weight: Optional[str] = None,
    rf_n_jobs: int = -1,
    rf_random_state: int = 42,
    rf_use_sample_weights: bool = False,
    class_weights_np: Optional[np.ndarray] = None,
) -> RunResult:
    """
    RandomForestClassifier baseline trained on the same window tensors as the deep models.

    Note: this is a single "fit" (no epochs), but we still report metrics using the
    same selection metric logic as the torch models for comparability.
    """
    model_name = "rf"

    # Lazy import so non-RF training does not require sklearn.
    try:
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
    except Exception as e:
        raise SystemExit(
            "scikit-learn is required for --model rf.\n"
            "Install it with: pip install scikit-learn\n"
            f"Import error: {e}"
        )

    run_dir = ckpt_root / model_name / run_id
    ckpt_path = run_dir / f"{model_name}_best.pkl"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Save a human-readable label map alongside checkpoints
    try:
        label_map_path = run_dir / "label_map_fallmerged.json"
        if not label_map_path.exists():
            label_map_path.write_text(
                json.dumps({"label_scheme": "fall_merged_7c", "label_convention": label_convention, "id_to_name": new_label_names}, indent=2),
                encoding="utf-8",
            )
    except Exception as e:
        print(f"Warning: failed to write label map JSON: {e}")

    Xtr = windows_to_sklearn_features(
        X_train,
        mode=rf_feature_mode,
        paper_num_skeletons=int(rf_paper_num_skeletons),
        paper_num_features=int(rf_paper_num_features),
    )
    Xva = windows_to_sklearn_features(
        X_val,
        mode=rf_feature_mode,
        paper_num_skeletons=int(rf_paper_num_skeletons),
        paper_num_features=int(rf_paper_num_features),
    )

    # sklearn expects 1D int labels
    ytr = y_train.astype(np.int64, copy=False).reshape(-1)
    yva = y_val.astype(np.int64, copy=False).reshape(-1)

    if rf_class_weight is not None:
        rf_class_weight = str(rf_class_weight).strip()
        if rf_class_weight.lower() in {"none", ""}:
            rf_class_weight = None

    clf = RandomForestClassifier(
        n_estimators=int(rf_n_estimators),
        max_depth=None if rf_max_depth is None else int(rf_max_depth),
        max_features=str(rf_max_features),
        min_samples_split=int(rf_min_samples_split),
        min_samples_leaf=int(rf_min_samples_leaf),
        bootstrap=bool(rf_bootstrap),
        criterion=str(rf_criterion),
        class_weight=rf_class_weight,
        n_jobs=int(rf_n_jobs),
        random_state=int(rf_random_state),
    )

    sample_weight = None
    if bool(rf_use_sample_weights):
        if class_weights_np is None:
            print("Warning: --rf-use-sample-weights set but class_weights_np is None, ignoring sample weights.")
        else:
            sample_weight = class_weights_np[ytr].astype(np.float64, copy=False)

    t0 = time.time()
    clf.fit(Xtr, ytr, sample_weight=sample_weight)
    dt = time.time() - t0

    pred = clf.predict(Xva).astype(np.int64, copy=False)
    val_acc = float(np.mean(pred == yva)) if yva.size > 0 else 0.0

    cm = _confusion_matrix_np(yva, pred, num_classes=int(num_classes))

    if selection_metric == "acc":
        val_score = float(val_acc)
    elif selection_metric == "composite_fall_fbeta_macro_f1":
        composite, fall_fbeta, macro_f1 = composite_fall_fbeta_macro_f1_from_confusion(
            cm,
            fall_ids_0based=fall_ids_0based,
            w=float(selection_w),
            beta=float(selection_beta),
        )
        val_score = float(composite)
        print(
            f"RF | fit {dt:.1f}s | "
            f"val acc {val_acc:.3f} | macro_f1 {macro_f1:.3f} fall_fbeta(beta={float(selection_beta):.2f}) {fall_fbeta:.3f} | "
            f"composite(w={float(selection_w):.2f}) {val_score:.3f}"
        )
    else:
        val_score = selection_score_from_confusion(
            cm,
            selection_metric=str(selection_metric),
            metric_weights=metric_weights_np,
            fall_ids_0based=fall_ids_0based,
            selection_w=float(selection_w),
            selection_beta=float(selection_beta),
        )
        print(f"RF | fit {dt:.1f}s | val acc {val_acc:.3f} | {selection_metric} {val_score:.3f}")

    # Save checkpoint pickle (clf + minimal metadata)
    payload = {
        "model": clf,
        "rf_feature_mode": str(rf_feature_mode),
        "rf_paper_num_skeletons": int(rf_paper_num_skeletons),
        "rf_paper_num_features": int(rf_paper_num_features),
        "label_convention": str(label_convention),
        "id_to_name": list(new_label_names),
        "num_classes": int(num_classes),
    }
    with open(ckpt_path.as_posix(), "wb") as f:
        pickle.dump(payload, f)

    return RunResult(
        model=model_name,
        best_val_score=float(val_score),
        best_val_acc=float(val_acc),
        best_val_loss=float("nan"),
        best_epoch=1,
        final_val_score=float(val_score),
        final_val_acc=float(val_acc),
        final_val_loss=float("nan"),
        params_m=float("nan"),
        train_seconds=float(dt),
        ckpt_path=str(ckpt_path.as_posix()),
    )
