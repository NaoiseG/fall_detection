"""
python -m models.tcn.train_tcn
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WindowTensorDataset    
from .simple_tcn import TCNBaseline

from pathlib import Path
import glob
import os

from build_windows import make_window_tensors

@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def find_keypoints_npzs_subjects(output_root: Path, camera=1, subjects=range(1, 6)):
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

def load_windows_from_npzs(npz_paths, T=None, use_conf=True):
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

def train_one_epoch(model, loader, opt, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        ##Checks =====================
        if torch.isnan(X).any() or torch.isinf(X).any():
            print("BAD X:", torch.isnan(X).sum().item(), "NaNs,", torch.isinf(X).sum().item(), "Infs")
            print("X min/max:", X.min().item(), X.max().item())
            raise SystemExit


        opt.zero_grad(set_to_none=True)
        logits = model(X)

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("BAD logits")
            raise SystemExit

        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        b = X.size(0)
        total_loss += loss.item() * b

        if torch.isnan(loss) or torch.isinf(loss):
            print("BAD loss")
            raise SystemExit

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
# Example: plug in your data
# ----------------------------

if __name__ == "__main__":

    # 1) Load real window tensors from your exported NPZ
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints/outputs_npz")
    camera = 1

    ckpt_path = Path("models/tcn/test_run/tcn_best.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train_subjects = range(16, 18)
    val_subjects   = range(1, 2)

    train_npzs = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=camera, subjects=train_subjects)
    val_npzs   = find_keypoints_npzs_subjects(OUTPUT_ROOT, camera=camera, subjects=val_subjects)

    print("Train sequences:", len(train_npzs))
    print("Val sequences:", len(val_npzs))

    X_train, y_train_tags, T_used = load_windows_from_npzs(train_npzs, T=None, use_conf=True)
    X_val,   y_val_tags,   _      = load_windows_from_npzs(val_npzs,   T=T_used, use_conf=True)

    # Multiclass: 1..11 -> 0..10
    y_train = (y_train_tags.astype(int) - 1).astype(np.int64)
    y_val   = (y_val_tags.astype(int) - 1).astype(np.int64)

    print("Train label ids:", np.unique(y_train))
    print("Val label ids:", np.unique(y_val))

    num_classes = int(max(y_train.max(), y_val.max()) + 1)
    print("num_classes:", num_classes)

    train_ds = WindowTensorDataset(X_train, y_train)
    val_ds = WindowTensorDataset(X_val, y_val)

    cfg = TrainConfig()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Infer feature size after reshape in the Dataset
    sample_X, _ = train_ds[0]
    in_features = sample_X.shape[-1]

    model = TCNBaseline(
        in_features=in_features,
        num_classes=num_classes,
        hidden_channels=128,
        num_blocks=4,
        kernel_size=3,
        dropout=0.1,
    ).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

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
