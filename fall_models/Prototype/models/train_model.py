"""
Run from project root:
  python -m models.train_model --model <MODEL>
"""

from dataclasses import dataclass
from typing import Tuple
import argparse
from pathlib import Path
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WindowTensorDataset
from build_windows import make_window_tensors

from .tcn.simple_tcn import TCNBaseline
from .lstm.simple_lstm import LSTMBaseline
from .gru.simple_gru import GRUBaseline
from .gcn.simple_gcn import GCNBaseline
from .mlp.simple_mlp import MLPBaseline
from .stgcn.simple_stgcn import STGCNBaseline


# ----------------------------
# Config
# ----------------------------

@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Metrics
# ----------------------------

@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


# ----------------------------
# Data discovery / loading
# ----------------------------

def find_keypoints_npzs_subjects(output_root: Path, camera: int = 1, subjects=range(1, 6)):
    """
    Matches:
      Subject{s}/Activity*/Trial*/Subject{s}Activity*Trial*Camera{camera}/keypoints.npz
    """
    npzs = []
    for s in subjects:
        subj_root = output_root / f"Subject{s}"
        if not subj_root.exists():
            continue

        pat = subj_root / "Activity*" / "Trial*" / f"Subject{s}Activity*Trial*Camera{camera}" / "keypoints.npz"
        npzs.extend(glob.glob(str(pat), recursive=True))

    return sorted(npzs)


def load_windows_from_npzs(npz_paths, T=None, use_conf: bool = True):
    """
    Loads multiple trial NPZs, converts each to (W, T, K, C) windows,
    then concatenates across trials. Ensures the same T is used for all files.
    """
    X_all, y_all = [], []
    T_used = T

    for i, p in enumerate(npz_paths):
        if i == 0 and T_used is None:
            X, y, T_used = make_window_tensors(p, T=None, use_conf=use_conf)
        else:
            X, y, _ = make_window_tensors(p, T=T_used, use_conf=use_conf)

        X_all.append(X)
        y_all.append(y)

    if not X_all:
        raise RuntimeError("No NPZs found / no windows loaded.")

    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0), T_used


# ----------------------------
# Model factory
# ----------------------------

def get_model(model_name: str, in_features: int, num_classes: int, device: str):
    model_name = model_name.lower()

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
            node_features=3,     # x,y,conf
            num_classes=num_classes,
            hidden_size=64,
            dropout=0.1,
        )
    elif model_name == "mlp":
        model = MLPBaseline(
            T=T_used,                 # window length you already compute
            in_features=in_features,  # feature dim per frame
            num_classes=num_classes,
            hidden_sizes=(256, 128),
            dropout=0.2,
        )
    elif model_name == "stgcn":
        model = STGCNBaseline(
            num_nodes=17,
            node_features=3,      # x, y, conf
            num_classes=num_classes,
            hidden_channels=128,
            num_blocks=4,
            t_kernel=9,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'. Use 'tcn', 'gru', 'gcn', 'mlp', 'stgcn' or 'lstm'.")

    print(f"\nUsing model: {model.__class__.__name__} (arg --model {model_name})")
    return model.to(device)


# ----------------------------
# Train / Eval
# ----------------------------

def train_one_epoch(model, loader, opt, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        # Debug checks (safe to remove once stable)
        if torch.isnan(X).any() or torch.isinf(X).any():
            print("BAD X:", torch.isnan(X).sum().item(), "NaNs,", torch.isinf(X).sum().item(), "Infs")
            print("X nanmin/nanmax:", torch.nanmin(X).item(), torch.nanmax(X).item())
            raise SystemExit

        opt.zero_grad(set_to_none=True)
        logits = model(X)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("BAD logits")
            raise SystemExit

        loss = F.cross_entropy(logits, y)

        if torch.isnan(loss) or torch.isinf(loss):
            print("BAD loss")
            raise SystemExit

        loss.backward()
        opt.step()

        b = X.size(0)
        total_loss += loss.item() * b
        total_acc += accuracy(logits, y) * b
        n += b

    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_one_epoch(model, loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        logits = model(X)
        loss = F.cross_entropy(logits, y)

        b = X.size(0)
        total_loss += loss.item() * b
        total_acc += accuracy(logits, y) * b
        n += b

    return total_loss / n, total_acc / n


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train temporal model for UP-Fall windowed pose tensors.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["tcn", "lstm", "gru", "gcn", "mlp", "stgcn"],
        help="Model type to train: 'tcn', 'gru', 'gcn', 'mlp', 'stgcn' or 'lstm'",
    )
    parser.add_argument("--camera", type=int, default=1, help="Camera index to train on (default: 1)")
    parser.add_argument("--train-subjects", type=str, default="16-17", help="Train subject range like '1-12' or '16-17'")
    parser.add_argument("--val-subjects", type=str, default="1-1", help="Val subject range like '13-16' or '1-1'")
    args = parser.parse_args()

    MODEL_NAME = args.model
    camera = args.camera

    def parse_range(r: str):
        # accepts "16-17" or "1-12"
        a, b = r.split("-")
        a, b = int(a), int(b)
        return range(a, b + 1)

    train_subjects = parse_range(args.train_subjects)
    val_subjects = parse_range(args.val_subjects)

    # Paths
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints/outputs_npz")

    # Checkpoint path (model-specific)
    ckpt_path = Path(f"models/{MODEL_NAME}/test_run/{MODEL_NAME}_best.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover sequences
    train_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=camera, subjects=train_subjects)
    val_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=camera, subjects=val_subjects)

    if not train_npzs:
        raise RuntimeError("No training NPZs found. Check OUTPUT_ROOT, camera, and train subjects.")
    if not val_npzs:
        raise RuntimeError("No validation NPZs found. Check OUTPUT_ROOT, camera, and val subjects.")

    # Load windows
    X_train, y_train_tags, T_used = load_windows_from_npzs(train_npzs, T=None, use_conf=True)
    X_val, y_val_tags, _ = load_windows_from_npzs(val_npzs, T=T_used, use_conf=True)

    # Multiclass: 1..11 -> 0..10
    y_train = (y_train_tags.astype(int) - 1).astype(np.int64)
    y_val = (y_val_tags.astype(int) - 1).astype(np.int64)

    num_classes = int(max(y_train.max(), y_val.max()) + 1)

    # Datasets / loaders
    train_ds = WindowTensorDataset(X_train, y_train)
    val_ds = WindowTensorDataset(X_val, y_val)

    cfg = TrainConfig()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Feature size
    sample_X, _ = train_ds[0]
    in_features = sample_X.shape[-1]

    # Model
    model = get_model(
        model_name=MODEL_NAME,
        in_features=in_features,
        num_classes=num_classes,
        device=cfg.device,
    )

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Training loop
    best_va = -1.0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, cfg.device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, cfg.device)

        if va_acc > best_va:
            best_va = va_acc
            torch.save(model.state_dict(), ckpt_path)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

    print(f"\nBest val acc: {best_va:.3f}")
    print(f"Saved best weights to: {ckpt_path}")
