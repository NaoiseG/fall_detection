"""
Run from project root:

Single model:
  python -m models.train_models --model tcn

Multiple models:
  python -m models.train_models --model tcn lstm gru

All models:
  python -m models.train_models --model tcn lstm gru

All models:
  python -m models.train_models --all

Save results table:
  python -m models.train_models --all --save-results results.csv

  python -m training.train_models --all --train-subjects 1-12 --val-subjects 13-15 --label-mode center \
    --drop-ambig-share 0 --T 64 --stride 18 --epochs 100 --selection-metric macro_recall \
    --normalize 1 --normalize-mode paper_rp --rp-center-mode pixel --rp-img-w 640 --rp-img-h 480 \
    --missing-mode zeros_only --interp-mode paper_group_linear --interp-group 100
"""

from dataclasses import dataclass, asdict
from typing import Tuple, List, Optional
import argparse
import math
from pathlib import Path
import time
import csv
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset_helpers.dataset import (
    load_windows_from_npzs,
    find_keypoints_npzs_subjects,
    WindowTensorDataset,
    detect_label_convention_from_npzs,
    get_new_label_names,
    detect_label_convention as _detect_label_convention,
    remap_label as _remap_label,
    get_fall_merge_set as _get_fall_merge_set,
)

from models.tcn.simple_tcn import TCNBaseline
from models.lstm.simple_lstm import LSTMBaseline
from models.gru.simple_gru import GRUBaseline
from models.gcn.simple_gcn import GCNBaseline
from models.mlp.simple_mlp import MLPBaseline
from models.stgcn.simple_stgcn import STGCNBaseline
from models.cnnlstm.cnn_lstm import CNNLSTMTwoHead
from models.rf.train_rf import train_random_forest_once


# =============================================================================
# Label scheme: merge fall subclasses into a single "Fall" class (7 classes total)
#
# The window labels produced by dataset.load_windows_from_npzs are now *already*
# remapped into the merged 7-class space (0..6). Training and evaluation must
# therefore NOT shift labels by -1.
#
# Auto-detection of raw NPZ label convention:
#   - If any label 11 exists => raw convention is 1–11
#   - Else if label 0 exists and max label is 10 => raw convention is 0–10
# =============================================================================
NUM_CLASSES_MERGED = 7
FALL_CLASS_ID = 0
RARE_CLASS_IDS_MERGED = [0, 4]

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

# ----------------------------
# Config
# ----------------------------

@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Results
# ----------------------------

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


# ----------------------------
# Metrics
# ----------------------------

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def compute_class_weights(
    y: np.ndarray,
    num_classes: int,
    mode: str = "inv_sqrt",
    eps: float = 1e-6,
    rare_boost: float = 1.0,
    rare_class_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Compute per-class weights for CrossEntropyLoss.

    mode:
      - 'none'     : uniform weights
      - 'inv'      : weight ~ 1 / count
      - 'inv_sqrt' : weight ~ 1 / sqrt(count)   (usually more stable)

    Weights are normalised to mean=1 to keep the loss scale roughly comparable.
    """
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


@torch.no_grad()
def _update_confusion_matrix(
    cm: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> None:
    """Accumulates into cm (shape [C,C]) on CPU."""
    y_true = y_true.view(-1).to(torch.int64).cpu()
    y_pred = y_pred.view(-1).to(torch.int64).cpu()
    idx = y_true * int(num_classes) + y_pred
    cm += torch.bincount(
        idx,
        minlength=int(num_classes) * int(num_classes),
    ).view(int(num_classes), int(num_classes))


@torch.no_grad()
def fall_fbeta_from_confusion(
    cm: torch.Tensor,
    fall_ids_0based: Optional[List[int]],
    beta: float = 2.0,
    fall_class_id: int = FALL_CLASS_ID,
) -> float:
    """Fβ for fall-vs-nonfall computed from a multi-class confusion matrix."""
    beta = float(beta)
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError(f"beta must be finite and > 0, got {beta}")

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be square [C,C], got {tuple(cm.shape)}")

    num_classes = int(cm.shape[0])
    if num_classes <= 0:
        return 0.0

    fall_ids = fall_ids_0based if fall_ids_0based else [int(fall_class_id)]
    fall_ids = sorted({int(i) for i in fall_ids if 0 <= int(i) < num_classes})
    if not fall_ids and 0 <= int(fall_class_id) < num_classes:
        fall_ids = [int(fall_class_id)]
    if not fall_ids:
        return 0.0

    fall_idx = torch.tensor(fall_ids, dtype=torch.long, device=cm.device)
    is_fall = torch.zeros((num_classes,), dtype=torch.bool, device=cm.device)
    is_fall[fall_idx] = True

    # TP = true in fall set AND predicted in fall set
    cm_fall_rows = cm.index_select(0, fall_idx)
    tp = cm_fall_rows.index_select(1, fall_idx).sum().float()

    # FN = true in fall set AND predicted not in fall set
    fn = cm_fall_rows.sum().float() - tp

    # FP = true not in fall set AND predicted in fall set
    cm_nonfall_rows = cm[~is_fall]
    fp = cm_nonfall_rows.index_select(1, fall_idx).sum().float()

    denom_p = tp + fp
    denom_r = tp + fn
    precision = (tp / denom_p) if float(denom_p.item()) > 0.0 else tp.new_tensor(0.0)
    recall = (tp / denom_r) if float(denom_r.item()) > 0.0 else tp.new_tensor(0.0)

    beta2 = beta * beta
    denom = beta2 * precision + recall
    if float(denom.item()) <= 0.0:
        return 0.0
    fbeta = (1.0 + beta2) * precision * recall / denom
    return float(fbeta.item())


@torch.no_grad()
def macro_f1_from_confusion(cm: torch.Tensor) -> float:
    """Macro F1 across classes with support>0."""
    # cm: rows = true, cols = pred
    support = cm.sum(dim=1)  # (C,)
    pred_support = cm.sum(dim=0)  # (C,)
    tp = cm.diag()
    recall = tp.float() / support.clamp(min=1).float()
    precision = tp.float() / pred_support.clamp(min=1).float()
    f1 = (2.0 * precision * recall) / (precision + recall).clamp(min=1e-12)

    mask = support > 0
    if not bool(mask.any()):
        return 0.0
    return float(f1[mask].mean().item())


@torch.no_grad()
def composite_fall_fbeta_macro_f1_from_confusion(
    cm: torch.Tensor,
    fall_ids_0based: Optional[List[int]],
    w: float = 0.7,
    beta: float = 2.0,
) -> Tuple[float, float, float]:
    """Composite = w*Fβ(fall)+(1-w)*MacroF1, and returns (composite, fall_fbeta, macro_f1)."""
    if not math.isfinite(float(w)):
        raise ValueError(f"w must be finite, got {w}")
    w = float(w)
    w = max(0.0, min(1.0, w))

    macro_f1 = macro_f1_from_confusion(cm)
    fall_fbeta = fall_fbeta_from_confusion(cm, fall_ids_0based=fall_ids_0based, beta=float(beta))
    composite = w * fall_fbeta + (1.0 - w) * macro_f1
    return float(composite), float(fall_fbeta), float(macro_f1)


@torch.no_grad()
def selection_score_from_confusion(
    cm: torch.Tensor,
    selection_metric: str,
    metric_weights: Optional[torch.Tensor] = None,
    fall_ids_0based: Optional[List[int]] = None,
    selection_w: float = 0.7,
    selection_beta: float = 2.0,
) -> float:
    """Returns a scalar score where larger is better."""
    if selection_metric == "composite_fall_fbeta_macro_f1":
        score, _, _ = composite_fall_fbeta_macro_f1_from_confusion(
            cm,
            fall_ids_0based=fall_ids_0based,
            w=float(selection_w),
            beta=float(selection_beta),
        )
        return float(score)

    # cm: rows = true, cols = pred
    support = cm.sum(dim=1)  # (C,)
    pred_support = cm.sum(dim=0)  # (C,)
    tp = cm.diag()
    recall = tp.float() / support.clamp(min=1).float()
    precision = tp.float() / pred_support.clamp(min=1).float()
    f1 = (2.0 * precision * recall) / (precision + recall).clamp(min=1e-12)

    mask = support > 0
    if not bool(mask.any()):
        return 0.0

    recall = recall[mask]
    f1 = f1[mask]

    if selection_metric == "macro_recall":
        return float(recall.mean().item())

    if selection_metric == "inv_freq_recall":
        if metric_weights is None:
            return float(recall.mean().item())
        w = metric_weights.view(-1).float().cpu()[mask]
        if float(w.sum().item()) <= 0.0:
            return float(recall.mean().item())
        w = w / w.sum()
        return float((recall * w).sum().item())

    if selection_metric == "macro_f1":
        return float(f1.mean().item())

    raise ValueError(f"Unknown selection_metric: {selection_metric}")


# ----------------------------
# Model factory
# ----------------------------

def get_model(
    model_name: str,
    in_features: int,
    num_classes: int,
    device: str,
    T_used: Optional[int] = None,
    node_features: Optional[int] = None,
):
    # strip() avoids weird CLI whitespace issues
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
            raise ValueError("node_features must be provided for GCN.")
        model = GCNBaseline(
            num_nodes=17,
            node_features=node_features,
            num_classes=num_classes,
            hidden_size=64,
            dropout=0.1,
        )

    elif model_name == "mlp":
        if T_used is None:
            raise ValueError("T_used must be provided for MLP (needed to flatten T*F).")
        model = MLPBaseline(
            T=T_used,
            in_features=in_features,
            num_classes=num_classes,
            hidden_sizes=(256, 128),
            dropout=0.2,
        )

    elif model_name == "stgcn":
        if node_features is None:
            raise ValueError("node_features must be provided for STGCN.")
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
        # If node_features is available and in_features == 17*node_features,
        # the model can use the keypoint-CNN path. Otherwise it auto-falls back.
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

    print(f"\nUsing model: {model.__class__.__name__} (arg --model {model_name})")
    return model.to(device)


def count_params_m(model: torch.nn.Module) -> float:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n / 1e6


# ----------------------------
# Train / Eval
# ----------------------------

def unpack_model_output(out):
    # Supports both:
    #  - single head: logits
    #  - two head: (activity_logits, fall_logit)
    if isinstance(out, (tuple, list)) and len(out) == 2:
        return out[0], out[1]
    return out, None


def train_one_epoch(
    model,
    loader,
    opt,
    device,
    fall_ids_0based: Optional[List[int]] = None,
    lambda_bin: float = 0.5,
    pos_weight: Optional[float] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    if pos_weight is not None:
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        bce = torch.nn.BCEWithLogitsLoss()

    fall_ids_t = None
    if fall_ids_0based is not None and len(fall_ids_0based) > 0:
        fall_ids_t = torch.tensor(fall_ids_0based, device=device, dtype=torch.long)

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)

        out = model(X)
        activity_logits, fall_logit = unpack_model_output(out)

        loss = F.cross_entropy(activity_logits, y, weight=class_weights)

        # If the model supports fall head AND we have fall class ids, train it
        if fall_logit is not None and fall_ids_t is not None:
            fall_logit = fall_logit.view(-1, 1)  # (B,1)
            y_fall = torch.isin(y, fall_ids_t).float().view(-1, 1)  # (B,1)
            loss = loss + lambda_bin * bce(fall_logit, y_fall)

        loss.backward()
        opt.step()

        b = X.size(0)
        total_loss += loss.item() * b
        total_acc += accuracy(activity_logits, y) * b
        n += b

    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    device,
    fall_ids_0based: Optional[List[int]] = None,
    lambda_bin: float = 0.5,
    pos_weight: Optional[float] = None,
    class_weights: Optional[torch.Tensor] = None,
    num_classes: Optional[int] = None,
    selection_metric: str = "composite_fall_fbeta_macro_f1",
    metric_weights: Optional[torch.Tensor] = None,
    selection_w: float = 0.7,
    selection_beta: float = 2.0,
) -> Tuple[float, float, float, dict]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    if pos_weight is not None:
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        bce = torch.nn.BCEWithLogitsLoss()

    fall_ids_t = None
    if fall_ids_0based is not None and len(fall_ids_0based) > 0:
        fall_ids_t = torch.tensor(fall_ids_0based, device=device, dtype=torch.long)

    cm = None
    if selection_metric in {"macro_recall", "inv_freq_recall", "macro_f1", "composite_fall_fbeta_macro_f1"}:
        if num_classes is None:
            raise ValueError("num_classes must be provided when using confusion-matrix-based selection metrics.")
        cm = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.long)

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        out = model(X)
        activity_logits, fall_logit = unpack_model_output(out)

        loss = F.cross_entropy(activity_logits, y, weight=class_weights)

        if fall_logit is not None and fall_ids_t is not None:
            fall_logit = fall_logit.view(-1, 1)
            y_fall = torch.isin(y, fall_ids_t).float().view(-1, 1)
            loss = loss + lambda_bin * bce(fall_logit, y_fall)

        b = X.size(0)
        total_loss += loss.item() * b
        total_acc += accuracy(activity_logits, y) * b
        n += b

        if cm is not None:
            preds = activity_logits.argmax(dim=1)
            _update_confusion_matrix(cm, y_true=y, y_pred=preds, num_classes=int(num_classes))

    val_loss = total_loss / max(1, n)
    val_acc = total_acc / max(1, n)

    details: dict = {}
    if selection_metric == "acc":
        val_score = float(val_acc)
    elif selection_metric == "composite_fall_fbeta_macro_f1":
        assert cm is not None
        composite, fall_fbeta, macro_f1 = composite_fall_fbeta_macro_f1_from_confusion(
            cm,
            fall_ids_0based=fall_ids_0based,
            w=float(selection_w),
            beta=float(selection_beta),
        )
        val_score = float(composite)
        details = {
            "macro_f1": float(macro_f1),
            "fall_fbeta": float(fall_fbeta),
            "selection_w": float(max(0.0, min(1.0, float(selection_w)))),
            "selection_beta": float(selection_beta),
            "composite": float(composite),
        }
    else:
        assert cm is not None
        val_score = selection_score_from_confusion(
            cm,
            selection_metric=selection_metric,
            metric_weights=metric_weights,
            fall_ids_0based=fall_ids_0based,
            selection_w=float(selection_w),
            selection_beta=float(selection_beta),
        )

    return val_loss, val_acc, float(val_score), details



def train_model_once(
    model_name: str,
    cfg: TrainConfig,
    in_features: int,
    num_classes: int,
    label_convention: str,
    new_label_names: List[str],
    use_conf: bool,
    normalize: bool,
    add_vel: bool,
    add_acc: bool,
    add_global: bool,
    T_used: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    ckpt_root: Path,
    run_id: str,
    conf_thres: float,
    max_interp_gap: int,
    stride: int,
    label_mode: str,
    min_valid_frac: float,
    add_mask_channel: bool,
    drop_ambig_share: float,
    drop_ambig_nonfall_only: bool,
    fall_class_ids_raw: Optional[List[int]] = None,
    node_features: Optional[int] = None,
    fall_ids_0based: Optional[List[int]] = None,
    pos_weight: Optional[float] = None,
    lambda_bin: float = 0.5,
    selection_metric: str = "composite_fall_fbeta_macro_f1",
    selection_w: float = 0.7,
    selection_beta: float = 2.0,
    metric_weights_np: Optional[np.ndarray] = None,
    class_weights_np: Optional[np.ndarray] = None,
) -> RunResult:
    model_name = model_name.lower().strip()

    run_dir = ckpt_root / model_name / run_id
    ckpt_path = run_dir / f"{model_name}_best.pt"
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

    model = get_model(
        model_name=model_name,
        in_features=in_features,
        num_classes=num_classes,
        device=cfg.device,
        T_used=T_used,
        node_features=node_features,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    metric_weights_t = None
    if metric_weights_np is not None:
        metric_weights_t = torch.tensor(metric_weights_np, device=cfg.device, dtype=torch.float32)

    class_weights_t = None
    if class_weights_np is not None:
        class_weights_t = torch.tensor(class_weights_np, device=cfg.device, dtype=torch.float32)

    # Best checkpoint is chosen by selection_metric (default: composite_fall_fbeta_macro_f1),
    # not by overall accuracy.
    best_vs = -1.0
    best_va = -1.0
    best_vl = float("inf")
    best_epoch = -1

    final_vs = -1.0
    final_va = -1.0
    final_vl = float("inf")

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, opt, cfg.device,
            fall_ids_0based=fall_ids_0based,
            lambda_bin=lambda_bin,
            pos_weight=pos_weight,
            class_weights=class_weights_t,
        )
        va_loss, va_acc, va_score, va_details = eval_one_epoch(
            model, val_loader, cfg.device,
            fall_ids_0based=fall_ids_0based,
            lambda_bin=lambda_bin,
            pos_weight=pos_weight,
            class_weights=class_weights_t,
            num_classes=num_classes,
            selection_metric=selection_metric,
            metric_weights=metric_weights_t,
            selection_w=float(selection_w),
            selection_beta=float(selection_beta),
        )

        final_vs, final_va, final_vl = va_score, va_acc, va_loss

        if va_score > best_vs:
            best_vs = float(va_score)
            best_va = float(va_acc)
            best_vl = float(va_loss)
            best_epoch = int(epoch)
            torch.save({
                "state_dict": model.state_dict(),
                "in_features": in_features,
                "num_classes": num_classes,
                "label_scheme": "fall_merged_7c",
                "label_convention": str(label_convention),
                "new_label_names": list(new_label_names),
                "fall_class_id": int(FALL_CLASS_ID),
                "use_conf": bool(use_conf),
                "normalize": bool(normalize),
                "add_vel": bool(add_vel),
                "add_acc": bool(add_acc),
                "add_global": bool(add_global),
                "conf_thres": float(conf_thres),
                "max_interp_gap": int(max_interp_gap),
                "T_used": int(T_used),
                "stride": int(stride),
                "label_mode": str(label_mode),
                "min_valid_frac": float(min_valid_frac),
                "add_mask_channel": bool(add_mask_channel),
                "drop_ambig_share": float(drop_ambig_share),
                "drop_ambig_nonfall_only": bool(drop_ambig_nonfall_only),
                "fall_class_ids_raw": list(fall_class_ids_raw) if fall_class_ids_raw is not None else None,
                "fall_ids_0based": list(fall_ids_0based) if fall_ids_0based is not None else None,
                "pos_weight": float(pos_weight) if pos_weight is not None else None,
                "lambda_bin": float(lambda_bin),
                "selection_metric": str(selection_metric),
                "selection_w": float(max(0.0, min(1.0, float(selection_w)))),
                "selection_beta": float(selection_beta),
                "metric_weights": metric_weights_np.tolist() if metric_weights_np is not None else None,
                "class_weights": class_weights_np.tolist() if class_weights_np is not None else None,
                "node_features": int(node_features) if node_features is not None else None,
            }, ckpt_path)

        if selection_metric == "composite_fall_fbeta_macro_f1":
            mf1 = float(va_details.get("macro_f1", 0.0))
            fbeta = float(va_details.get("fall_fbeta", 0.0))
            w_used = float(va_details.get("selection_w", max(0.0, min(1.0, float(selection_w)))))
            beta_used = float(va_details.get("selection_beta", float(selection_beta)))
            print(
                f"{model_name.upper()} | Epoch {epoch:02d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
                f"macro_f1 {mf1:.3f} fall_fbeta(β={beta_used:.2f}) {fbeta:.3f} | "
                f"composite(w={w_used:.2f}) {va_score:.3f}"
            )
        else:
            print(
                f"{model_name.upper()} | Epoch {epoch:02d} | "
                f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                f"val loss {va_loss:.4f} acc {va_acc:.3f} score {va_score:.3f} ({selection_metric})"
            )

    dt = time.time() - t0
    res = RunResult(
        model=model_name,
        best_val_score=float(best_vs),
        best_val_acc=float(best_va),
        best_val_loss=float(best_vl),
        best_epoch=int(best_epoch),
        final_val_score=float(final_vs),
        final_val_acc=float(final_va),
        final_val_loss=float(final_vl),
        params_m=float(count_params_m(model)),
        train_seconds=float(dt),
        ckpt_path=str(ckpt_path.as_posix()),
    )
    return res


# ----------------------------
# Results table
# ----------------------------

def print_results_table(results: List[RunResult]) -> None:
    # Sort by best val score desc
    results = sorted(results, key=lambda r: r.best_val_score, reverse=True)

    headers = [
        "model",
        "best_val_score",
        "best_val_acc",
        "best_val_loss",
        "best_epoch",
        "final_val_score",
        "final_val_acc",
        "final_val_loss",
        "params(M)",
        "train_s",
        "ckpt_path",
    ]

    rows = []
    for r in results:
        rows.append([
            r.model,
            f"{100.0 * r.best_val_score:.2f}%",
            f"{100.0 * r.best_val_acc:.2f}%",
            f"{r.best_val_loss:.4f}",
            str(r.best_epoch),
            f"{100.0 * r.final_val_score:.2f}%",
            f"{100.0 * r.final_val_acc:.2f}%",
            f"{r.final_val_loss:.4f}",
            f"{r.params_m:.3f}",
            f"{r.train_seconds:.1f}",
            r.ckpt_path,
        ])

    # Markdown-style table (reads nicely in terminal too)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    def fmt_row(row_vals):
        return "| " + " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row_vals)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"

    print("\nResults:")
    print(fmt_row(headers))
    print(sep)
    for row in rows:
        print(fmt_row(row))


def save_results_csv(results: List[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        fieldnames = list(asdict(results[0]).keys())
    else:
        fieldnames = list(asdict(RunResult(
            model="", best_val_acc=0, best_val_loss=0, best_epoch=0,
            final_val_acc=0, final_val_loss=0, params_m=0, train_seconds=0, ckpt_path=""
        )).keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")  # e.g. 2026-01-19_14-03-22_123456
    print("Run ID:", run_id)

    ALL_MODELS = ["tcn", "lstm", "gru", "gcn", "mlp", "stgcn", "cnnlstm", "rf"]

    parser = argparse.ArgumentParser(description="Train one or more models on UP-Fall windowed pose tensors.")
    parser.add_argument(
        "--model",
        nargs="+",
        type=str,
        default=None,
        help="One or more models to train, e.g. --model tcn lstm gru",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models (overrides --model).",
    )
    parser.add_argument("--camera", type=int, default=1, help="Camera index to train on (default: 1)")
    parser.add_argument("--train-subjects", type=str, default="16-17", help="Train subject range like '1-12' or '16-17'")
    parser.add_argument("--val-subjects", type=str, default="1-1", help="Val subject range like '13-16' or '1-1'")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per model (default: 20)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Path to save results as CSV, e.g. --save-results results/summary.csv",
    )
    # Data preprocessing options
    parser.add_argument("--use-conf", type=int, default=1, help="Include keypoint confidence channel (0/1).")
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
    parser.add_argument("--conf-thres", type=float, default=0.2, help="Conf threshold below which joints are treated as missing.")
    parser.add_argument("--max-interp-gap", type=int, default=5, help="Max gap (frames) for linear interpolation of missing joints.")
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
        help="Interpolation/fill mode for missing joints.",
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
        help="Window label rule. hybrid_center_fallpct: label window as fall if fall frames >= fall_pct else center."
    )
    parser.add_argument("--min-valid-frac", type=float, default=0.3, help="Min fraction of joints above conf_thres for a frame to be valid.")
    parser.add_argument("--add-mask-channel", type=int, default=1, help="Append mask channel (0/1).")
    parser.add_argument("--drop-ambig-share", type=float, default=0.6,
                        help="Drop windows where top-label share < this value (measured on valid frames). 0 disables.")
    parser.add_argument("--drop-ambig-nonfall-only", type=int, default=1,
                         help="If 1, only drop ambiguous windows that contain no fall frames (helps preserve fall transitions).")
    parser.add_argument(
        "--fall-class-ids",
        nargs="+",
        type=int,
        default=None,
        help="(Optional) Raw fall class IDs in the NPZ label space (either 1..5 or 0..4). With the merged 7-class scheme this is usually unnecessary, but it is kept for backward compatibility."
    )
    parser.add_argument(
        "--fall-pct",
        type=float,
        default=0.25,
        help="Used only when --label-mode hybrid_center_fallpct. Window is labeled fall if >= fall_pct of valid frames are fall. Try 0.20-0.30."
    )
    parser.add_argument("--lambda-bin", type=float, default=0.5, help="Weight for binary fall head loss (default: 0.5)")
    parser.add_argument(
        "--class-weight-mode",
        type=str,
        default="inv_sqrt",
        choices=["none", "inv", "inv_sqrt"],
        help="Cross-entropy class weighting to mitigate imbalance (default: inv_sqrt).",
    )
    parser.add_argument(
        "--rare-class-boost",
        type=float,
        default=1.0,
        help="Optional multiplicative boost applied to rare classes [0,4] in CE weights (default: 1.0).",
    )
    parser.add_argument(
        "--weighted-sampler",
        type=int,
        default=0,
        help="If 1, use WeightedRandomSampler for the training loader (0/1).",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="composite_fall_fbeta_macro_f1",
        choices=[
            "composite_fall_fbeta_macro_f1",
            "macro_f1",
            "inv_freq_recall",
            "macro_recall",
            "acc",
        ],
        help=(
            "Metric used to choose the best checkpoint (default: composite_fall_fbeta_macro_f1). "
            "composite_fall_fbeta_macro_f1 = w*Fβ(fall)+(1-w)*MacroF1 (best practical choice). "
            "inv_freq_recall upweights minority classes using inverse train frequency."
        ),
    )
    parser.add_argument(
        "--selection-w",
        type=float,
        default=0.7,
        help="Weight w in the composite selection metric w*Fβ(fall)+(1-w)*MacroF1. Clamped to [0,1] (default: 0.7).",
    )
    parser.add_argument(
        "--selection-beta",
        type=float,
        default=2.0,
        help="β for Fβ(fall) in the composite selection metric (β>1 emphasizes recall). Must be >0 (default: 2.0).",
    )
    # Random Forest baseline (sklearn)
        # Random Forest baseline (sklearn)
    parser.add_argument(
        "--rf-feature-mode",
        type=str,
        default="flatten",
        choices=["flatten", "center", "stats", "paper_windowing"],
        help=(
            "Feature extraction for RF: "
            "flatten (T*F), center (F), stats (mean/std/min/max over time), "
            "paper_windowing (sample S evenly spaced skeletons, keep 51 pose feats each => S*51)."
        ),
    )
    parser.add_argument(
        "--rf-paper-num-skeletons",
        type=int,
        default=3,
        help="Only for --rf-feature-mode paper_windowing: number of skeletons (frames) sampled evenly per window (default: 3 => first/middle/last).",
    )
    parser.add_argument(
        "--rf-paper-num-features",
        type=int,
        default=51,
        help="Only for --rf-feature-mode paper_windowing: number of pose features per skeleton (default: 51 = 17*(x,y,conf)).",
    )
    parser.add_argument(
        "--rf-paper-defaults",
        type=int,
        default=0,
        help=(
            "If 1, apply the Sensors 2022 UP-Fall windowed-RF best-candidate defaults: "
            "T=36, stride=1, label-mode=majority, use-conf=1, normalize=0, add-vel/acc/global/mask=0, "
            "drop-ambig-share=0, rf-feature-mode=paper_windowing, rf-paper-num-skeletons=3, rf-paper-num-features=51."
        ),
    )
    parser.add_argument("--rf-n-estimators", type=int, default=300, help="Number of trees for RandomForestClassifier.")
    parser.add_argument("--rf-max-depth", type=int, default=0, help="Max tree depth (0 = unlimited).")
    parser.add_argument(
        "--rf-max-features",
        type=str,
        default="sqrt",
        choices=["sqrt", "log2"],
        help="max_features for RandomForestClassifier (default: sqrt).",
    )
    parser.add_argument("--rf-min-samples-split", type=int, default=2, help="min_samples_split for RandomForestClassifier.")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1, help="min_samples_leaf for RandomForestClassifier.")
    parser.add_argument("--rf-bootstrap", type=int, default=1, help="Use bootstrap samples for each tree (0/1).")
    parser.add_argument(
        "--rf-criterion",
        type=str,
        default="gini",
        choices=["gini", "entropy", "log_loss"],
        help="Split criterion for RandomForestClassifier.",
    )
    parser.add_argument(
        "--rf-class-weight",
        type=str,
        default="balanced_subsample",
        choices=["none", "balanced", "balanced_subsample"],
        help="Class weighting for RandomForestClassifier (default: balanced_subsample).",
    )
    parser.add_argument("--rf-n-jobs", type=int, default=-1, help="Number of parallel jobs for RF (-1 = all cores).")
    parser.add_argument("--rf-random-state", type=int, default=42, help="Random seed for RF.")
    parser.add_argument(
        "--rf-use-sample-weights",
        type=int,
        default=0,
        help="If 1, pass the same per-class weights used for CE as sample_weight to RF fit() (0/1).",
    )
    args = parser.parse_args()

    if int(getattr(args, "rf_paper_defaults", 0)) == 1:
        args.T = 36
        args.stride = 1
        args.label_mode = "majority"
        args.use_conf = 1
        args.normalize = 0
        args.add_vel = 0
        args.add_acc = 0
        args.add_global = 0
        args.add_mask_channel = 0
        args.drop_ambig_share = 0.0
        args.rf_feature_mode = "paper_windowing"
        args.rf_paper_num_skeletons = 3
        args.rf_paper_num_features = 51

    if not math.isfinite(float(args.selection_w)):
        raise SystemExit("--selection-w must be a finite float.")
    selection_w = float(args.selection_w)
    selection_w_clamped = max(0.0, min(1.0, selection_w))
    if selection_w != selection_w_clamped:
        print(f"[selection] Clamped --selection-w from {selection_w} to {selection_w_clamped}")
    args.selection_w = selection_w_clamped

    if not math.isfinite(float(args.selection_beta)) or float(args.selection_beta) <= 0.0:
        raise SystemExit("--selection-beta must be a finite float > 0.")
    args.selection_beta = float(args.selection_beta)

    use_conf = bool(args.use_conf)
    normalize = bool(args.normalize)
    add_vel = bool(args.add_vel)
    add_acc = bool(args.add_acc)
    if add_acc and not add_vel:
        raise SystemExit("--add-acc 1 requires --add-vel 1 (acc is computed from vel).")
    add_global = bool(args.add_global)
    add_mask_channel = bool(args.add_mask_channel)
    fall_class_ids_raw = None
    if args.fall_class_ids is not None and len(args.fall_class_ids) > 0:
        fall_class_ids_raw = [int(x) for x in args.fall_class_ids]

    # NOTE: hybrid_center_fallpct no longer requires --fall-class-ids because
    # the dataset loader auto-detects the raw label convention and knows which
    # raw IDs correspond to fall frames.

    if args.all:
        model_list = ALL_MODELS
    elif args.model is not None:
        model_list = [m.lower().strip() for m in args.model]
    else:
        model_list = ["tcn"]


    # Validate model names early
    unknown = sorted(set(model_list) - set(ALL_MODELS))
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}. Valid: {ALL_MODELS}")

    # Parse subject ranges
    def parse_range(r: str):
        a, b = r.split("-")
        a, b = int(a), int(b)
        return range(a, b + 1)

    train_subjects = parse_range(args.train_subjects)
    val_subjects = parse_range(args.val_subjects)

    # Paths
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints/outputs_npz")
    ckpt_root = Path("models")  # keeps your existing layout

    # ---- Compute T_used + num_classes exactly as before (so results stay comparable) ----
    train_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=args.camera, subjects=train_subjects)
    val_npzs   = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=args.camera, subjects=val_subjects)

    if not train_npzs:
        raise RuntimeError("No training NPZs found. Check OUTPUT_ROOT, camera, and train subjects.")
    if not val_npzs:
        raise RuntimeError("No validation NPZs found. Check OUTPUT_ROOT, camera, and val subjects.")

    print("Train sequences:", len(train_npzs))
    print("Val sequences:", len(val_npzs))
    print("Models to train:", model_list)

    
    # ---- Detect raw label convention once (1-11 vs 0-10), then keep it consistent ----
    label_convention, label_stats = detect_label_convention_from_npzs(train_npzs + val_npzs)
    NEW_LABEL_NAMES = get_new_label_names(label_convention)
    FALL_MERGE_SET = fall_merge_set(label_convention)
    print(f"[labels] Using raw convention: {label_convention} | New labels: {NEW_LABEL_NAMES}")
    fall_ids_0based = [int(FALL_CLASS_ID)]
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

    # Labels are already remapped to the merged 7-class space (0..6)
    y_train = y_train_tags.astype(np.int64, copy=False)
    y_val   = y_val_tags.astype(np.int64, copy=False)

    num_classes = int(NUM_CLASSES_MERGED)
    if int(y_train.max()) >= num_classes or int(y_val.max()) >= num_classes:
        raise RuntimeError(f"Unexpected label id >= {num_classes}. Check label remap.")
    print("num_classes:", num_classes, "| T_used:", T_used)

    print("window:", int(T_used), "frames | stride:", int(args.stride))

    # For checkpoint selection, we can upweight minority classes using inverse train frequency.
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
        print("Selection metric:", args.selection_metric, "(overall accuracy)")

    # Cross-entropy class weights to mitigate multi-class imbalance.
    class_weights_np = None
    if str(args.class_weight_mode).lower().strip() != "none":
        class_weights_np = compute_class_weights(
            y_train,
            num_classes=int(num_classes),
            mode=str(args.class_weight_mode),
            rare_boost=float(args.rare_class_boost),
            rare_class_ids=RARE_CLASS_IDS_MERGED,
        )
        counts_dbg = np.bincount(y_train, minlength=int(num_classes)).astype(np.int64)
        counts_dbg_d = {int(i): int(c) for i, c in enumerate(counts_dbg.tolist()) if int(c) > 0}
        print("Train window counts:", counts_dbg_d)
        print("CE class weights:", [float(x) for x in class_weights_np.tolist()])
    else:
        print("CE class weights: disabled (--class-weight-mode none)")

    # pos_weight for BCE (neg/pos) computed from y_train (0-based)
    pos_weight = None
    if fall_ids_0based is not None and len(fall_ids_0based) > 0:
        pos = int((y_train == int(FALL_CLASS_ID)).sum())
        neg = int(len(y_train) - pos)
        if pos > 0:
            pos_weight = neg / pos
            print(f"Binary fall head pos_weight: {pos_weight:.3f} (neg={neg}, pos={pos})")
        else:
            print("Warning: no positive fall windows in y_train. pos_weight not set.")

    # Build datasets to infer in_features and keep mapping consistent
    train_ds = WindowTensorDataset(X_train, y_train)
    val_ds   = WindowTensorDataset(X_val, y_val)

    sample_X, _ = train_ds[0]
    in_features = int(sample_X.shape[-1])

    # node_features is used by GCN/STGCN and also to enable keypoint CNN path in CNNLSTM
    K = 17
    node_features = int(in_features // K)
    if node_features * K != in_features:
        node_features = None

    cfg = TrainConfig(
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if bool(args.weighted_sampler):
        sampler_weights_np = class_weights_np
        if sampler_weights_np is None:
            # If CE weights are disabled, still build sampler weights from inverse-sqrt counts.
            sampler_weights_np = compute_class_weights(
                y_train,
                num_classes=int(num_classes),
                mode="inv_sqrt",
                rare_boost=float(args.rare_class_boost),
                rare_class_ids=RARE_CLASS_IDS_MERGED,
            )
        sample_w = torch.from_numpy(sampler_weights_np[y_train]).double()
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=int(len(sample_w)), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=False, drop_last=False, num_workers=0)
        print("Train loader: WeightedRandomSampler enabled.")
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)

    results: List[RunResult] = []
    for m in model_list:
        if m == "rf":
            rf_max_depth = None if int(args.rf_max_depth) <= 0 else int(args.rf_max_depth)
            rf_class_weight = str(args.rf_class_weight).strip()
            if rf_class_weight.lower() == "none":
                rf_class_weight = None
            res = train_random_forest_once(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                num_classes=num_classes,
                ckpt_root=ckpt_root,
                run_id=run_id,
                label_convention=label_convention,
                new_label_names=NEW_LABEL_NAMES,
                use_conf=use_conf,
                normalize=normalize,
                add_vel=add_vel,
                add_acc=add_acc,
                add_global=add_global,
                T_used=T_used,
                conf_thres=float(args.conf_thres),
                max_interp_gap=int(args.max_interp_gap),
                stride=int(args.stride),
                label_mode=str(args.label_mode),
                min_valid_frac=float(args.min_valid_frac),
                add_mask_channel=add_mask_channel,
                drop_ambig_share=float(args.drop_ambig_share),
                drop_ambig_nonfall_only=bool(args.drop_ambig_nonfall_only),
                fall_class_ids_raw=fall_class_ids_raw,
                fall_ids_0based=fall_ids_0based,
                selection_metric=str(args.selection_metric),
                selection_w=float(args.selection_w),
                selection_beta=float(args.selection_beta),
                metric_weights_np=metric_weights_np,
                rf_feature_mode=str(args.rf_feature_mode),
                rf_paper_num_skeletons=int(args.rf_paper_num_skeletons),
                rf_paper_num_features=int(args.rf_paper_num_features),
                rf_n_estimators=int(args.rf_n_estimators),
                rf_max_depth=rf_max_depth,
                rf_max_features=str(args.rf_max_features),
                rf_min_samples_split=int(args.rf_min_samples_split),
                rf_min_samples_leaf=int(args.rf_min_samples_leaf),
                rf_bootstrap=bool(args.rf_bootstrap),
                rf_criterion=str(args.rf_criterion),
                rf_class_weight=rf_class_weight,
                rf_n_jobs=int(args.rf_n_jobs),
                rf_random_state=int(args.rf_random_state),
                rf_use_sample_weights=bool(args.rf_use_sample_weights),
                class_weights_np=class_weights_np,
            )
        else:
            res = train_model_once(
                model_name=m,
                cfg=cfg,
                in_features=in_features,
                num_classes=num_classes,
                label_convention=label_convention,
                new_label_names=NEW_LABEL_NAMES,
                T_used=T_used,
                train_loader=train_loader,
                val_loader=val_loader,
                ckpt_root=ckpt_root,
                run_id=run_id,
                node_features=node_features,
                use_conf=use_conf,
                normalize=normalize,
                add_vel=add_vel,
                add_acc=add_acc,
                add_global=add_global,
                conf_thres=float(args.conf_thres),
                max_interp_gap=int(args.max_interp_gap),
                stride=int(args.stride),
                label_mode=str(args.label_mode),
                min_valid_frac=float(args.min_valid_frac),
                add_mask_channel=add_mask_channel,
                drop_ambig_share=float(args.drop_ambig_share),
                drop_ambig_nonfall_only=bool(args.drop_ambig_nonfall_only),
                fall_class_ids_raw=fall_class_ids_raw,
                fall_ids_0based=fall_ids_0based,
                pos_weight=pos_weight,
                lambda_bin=float(args.lambda_bin),
                selection_metric=str(args.selection_metric),
                selection_w=float(args.selection_w),
                selection_beta=float(args.selection_beta),
                metric_weights_np=metric_weights_np,
                class_weights_np=class_weights_np,
            )
        results.append(res)

    print_results_table(results)

    if args.save_results:
        out_path = Path(args.save_results)
        save_results_csv(results, out_path)
        print(f"\nSaved results CSV to: {out_path.as_posix()}")
