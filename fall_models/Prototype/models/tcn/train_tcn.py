from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import WindowTensorDataset    
from simple_tcn import TCNBaseline


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


def train_one_epoch(model, loader, opt, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(X)
        loss = F.cross_entropy(logits, y)
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
# Example: plug in your data
# ----------------------------

if __name__ == "__main__":

    # 1) Load real window tensors from your exported NPZ
    npz_path = "../../outputs/frames_run/keypoints.npz"
    X_windows, y_tags, T_used = make_window_tensors(npz_path, T=None, use_conf=True)

    # X_windows is (N, T, K, C) or (N, T, F)
    N = X_windows.shape[0]

    # 2) UP-Fall binary mapping:
    # Falls: 1-5 -> 1, ADL: 6-11 -> 0
    y_int = y_tags.astype(int)
    y_windows = (y_int <= 5).astype(np.int64)  # 1 = fall, 0 = ADL

    # Optional sanity print
    print("Loaded windows:", N, "T_used:", T_used)
    print("Activity IDs present:", np.unique(y_int))
    print("Binary counts (ADL=0, Fall=1):", np.bincount(y_windows))

    # 3) Shuffle split (recommended baseline split)
    rng = np.random.default_rng(42)
    perm = rng.permutation(N)
    n_train = int(0.8 * N)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    X_train, X_val = X_windows[train_idx], X_windows[val_idx]
    y_train, y_val = y_windows[train_idx], y_windows[val_idx]

    train_ds = WindowTensorDataset(X_train, y_train)
    val_ds = WindowTensorDataset(X_val, y_val)

    cfg = TrainConfig()
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    # Infer feature size after reshape in the Dataset
    sample_X, _ = train_ds[0]
    in_features = sample_X.shape[-1]
    num_classes = 2  # binary

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
            torch.save(model.state_dict(), "tcn_best.pt")

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )
