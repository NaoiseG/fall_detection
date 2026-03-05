"""
Run from project root:

Binary fall vs non-fall:
  python -m models.train_videomae_heatmaps --binary-any-fall 1 --fall-class-ids 1 2 3 4 5

Multiclass (activities):
  python -m models.train_videomae_heatmaps

Notes:
- VideoMAE tubelet temporal stride is typically 2, so T should be even.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import argparse
import time
import math

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


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 3e-5
    weight_decay: float = 0.05
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


def train_one_epoch(model, loader, opt, device, scaler: Optional[torch.cuda.amp.GradScaler], amp: bool) -> tuple[float, float]:
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for i, (pixel_values, y) in enumerate(loader):
        if i == 0:
            print("First batch shapes:", pixel_values.shape, y.shape)
        if i % 20 == 0:
            print("batch", i)

        pixel_values = pixel_values.to(device)  # (B,T,C,H,W)
        y = y.to(device)

        opt.zero_grad(set_to_none=True)

        if amp and scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(pixel_values=pixel_values, labels=y)
                loss = out.loss
                logits = out.logits
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = model(pixel_values=pixel_values, labels=y)
            loss = out.loss
            logits = out.logits
            loss.backward()
            opt.step()

        b = int(y.size(0))
        total_loss += float(loss.item()) * b
        total_acc += accuracy(logits, y) * b
        n += b

    return total_loss / n, total_acc / n


@torch.no_grad()
def eval_one_epoch(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for pixel_values, y in loader:
        pixel_values = pixel_values.to(device)
        y = y.to(device)

        out = model(pixel_values=pixel_values, labels=y)
        loss = out.loss
        logits = out.logits

        b = int(y.size(0))
        total_loss += float(loss.item()) * b
        total_acc += accuracy(logits, y) * b
        n += b

    return total_loss / n, total_acc / n


def parse_range(r: str):
    a, b = r.split("-")
    a, b = int(a), int(b)
    return range(a, b + 1)


def _is_improvement(current: float, best: float, *, mode: str, min_delta: float) -> bool:
    """Return True if `current` improved vs `best`.

    mode:
      - "max": larger is better (e.g. accuracy)
      - "min": smaller is better (e.g. loss)
    """
    if mode == "max":
        return current > (best + min_delta)
    if mode == "min":
        return current < (best - min_delta)
    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    print("Run ID:", run_id)

    parser = argparse.ArgumentParser(description="Train VideoMAE on keypoint heatmap videos.")
    parser.add_argument("--output-root", type=str, default="../../Datasets/UPFall_keypoints/outputs_npz")
    parser.add_argument("--camera", type=int, default=1)

    parser.add_argument("--train-subjects", type=str, default="16-17")
    parser.add_argument("--val-subjects", type=str, default="1-1")

    # windowing (matches your existing style)
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--label-mode", type=str, default="center",
                        choices=["center", "majority", "hybrid_center_fallpct"])
    parser.add_argument("--fall-pct", type=float, default=0.25)
    parser.add_argument("--min-valid-frac", type=float, default=0.3)
    parser.add_argument("--drop-ambig-share", type=float, default=0.0)
    parser.add_argument("--drop-ambig-nonfall-only", type=int, default=1)

    # keypoint cleaning
    parser.add_argument("--conf-thres", type=float, default=0.2)
    parser.add_argument("--max-interp-gap", type=int, default=5)
    parser.add_argument("--normalize-xy", type=int, default=1)

    # heatmap settings
    parser.add_argument("--hm-size", type=int, default=224)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--norm-range", type=float, default=2.5)

    # task options
    parser.add_argument("--binary-any-fall", type=int, default=0)
    parser.add_argument("--fall-class-ids", nargs="+", type=int, default=None,
                        help="Fall class IDs in the NPZ label space (usually 1-based), eg 1 2 3 4 5")

    # training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--ckpt-root", type=str, default="models/videomae_heatmaps")

    # early stopping / checkpointing
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
        help="Stop if monitored metric doesn't improve for this many epochs. Set 0 to disable.",
    )
    parser.add_argument(
        "--early-stop-metric",
        type=str,
        default="val_acc",
        choices=["val_acc", "val_loss"],
        help="Metric to monitor for checkpointing / early stopping.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum change required to count as an improvement.",
    )

    # model
    parser.add_argument("--pretrained-name", type=str, default="MCG-NJU/videomae-base")
    parser.add_argument("--init-from-rgb", type=str, default="mean", choices=["mean", "random"])

    args = parser.parse_args()

    if args.T % 2 != 0:
        raise SystemExit("--T should be even for VideoMAE (tubelet temporal stride is typically 2).")

    fall_ids_0based = None
    if args.fall_class_ids is not None and len(args.fall_class_ids) > 0:
        fall_ids_0based = [int(x) - 1 for x in args.fall_class_ids]

    binary_any_fall = bool(args.binary_any_fall)
    normalize_xy = bool(args.normalize_xy)

    # Build NPZ lists
    output_root = Path(args.output_root)
    train_npzs = build_npz_list(output_root, camera=args.camera, subjects=parse_range(args.train_subjects))
    val_npzs = build_npz_list(output_root, camera=args.camera, subjects=parse_range(args.val_subjects))

    if not train_npzs:
        raise RuntimeError("No training NPZs found. Check --output-root, --camera, --train-subjects.")
    if not val_npzs:
        raise RuntimeError("No validation NPZs found. Check --output-root, --camera, --val-subjects.")

    print("Train sequences:", len(train_npzs))
    print("Val sequences:", len(val_npzs))

    window_spec = WindowSpec(
        T=int(args.T),
        stride=int(args.stride),
        label_mode=str(args.label_mode),
        fall_pct=float(args.fall_pct),
        min_valid_frac=float(args.min_valid_frac),
        drop_ambig_share=float(args.drop_ambig_share),
        drop_ambig_nonfall_only=bool(args.drop_ambig_nonfall_only),
    )
    heatmap_spec = HeatmapSpec(
        out_h=int(args.hm_size),
        out_w=int(args.hm_size),
        sigma=float(args.sigma),
        conf_thr=float(args.conf_thres),
        norm_range=float(args.norm_range),
    )

    # Datasets
    train_ds = PoseHeatmapWindowDataset(
        train_npzs,
        window=window_spec,
        heatmap=heatmap_spec,
        normalize_xy=normalize_xy,
        conf_thres=float(args.conf_thres),
        max_interp_gap=int(args.max_interp_gap),
        labels_are_1_based=True,
        binary_any_fall=binary_any_fall,
        fall_ids_0based=fall_ids_0based,
    )
    val_ds = PoseHeatmapWindowDataset(
        val_npzs,
        window=window_spec,
        heatmap=heatmap_spec,
        normalize_xy=normalize_xy,
        conf_thres=float(args.conf_thres),
        max_interp_gap=int(args.max_interp_gap),
        labels_are_1_based=True,
        binary_any_fall=binary_any_fall,
        fall_ids_0based=fall_ids_0based,
    )

    # Decide num_labels
    if binary_any_fall:
        num_labels = 2
    else:
        # infer from dataset labels by scanning a small sample
        ys = []
        for i in range(min(512, len(train_ds))):
            _, y = train_ds[i]
            ys.append(int(y))
        num_labels = int(max(ys) + 1)

    print("Windows:", len(train_ds), "train |", len(val_ds), "val")
    print("num_labels:", num_labels)

    # Load model
    model = build_videomae_for_heatmaps(
        pretrained_name=str(args.pretrained_name),
        num_labels=int(num_labels),
        in_channels=17,
        init_from_rgb=str(args.init_from_rgb),
    )

    # Test =============================
    print("conv in_channels:", model.videomae.embeddings.patch_embeddings.projection.in_channels)
    print("patch num_channels:", getattr(model.videomae.embeddings.patch_embeddings, "num_channels", None))
    print("config num_channels:", model.config.num_channels)

    cfg = TrainConfig(
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        device="cuda" if torch.cuda.is_available() else "cpu",
        amp=not bool(args.no_amp),
    )

    model = model.to(cfg.device)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and cfg.device.startswith("cuda")) else None

    ckpt_root = Path(args.ckpt_root) / run_id
    ckpt_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_root / "videomae_best.pt"

    # Early stopping / checkpointing is driven by the monitored validation metric.
    monitor = str(args.early_stop_metric)
    patience = int(args.early_stop_patience)
    min_delta = float(args.early_stop_min_delta)

    if monitor == "val_acc":
        mode = "max"
        best_monitor = -math.inf
    else:
        mode = "min"
        best_monitor = math.inf

    best_ep = -1
    best_va_at_best, best_vl_at_best = -1.0, 1e9
    epochs_no_improve = 0

    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, cfg.device, scaler, cfg.amp)
        va_loss, va_acc = eval_one_epoch(model, val_loader, cfg.device)

        print(
            f"VIDEOMAE | Epoch {ep:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

        current = float(va_acc) if monitor == "val_acc" else float(va_loss)
        if _is_improvement(current, best_monitor, mode=mode, min_delta=min_delta):
            best_monitor = current
            best_ep = int(ep)
            best_va_at_best, best_vl_at_best = float(va_acc), float(va_loss)
            epochs_no_improve = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_labels": int(num_labels),
                    "in_channels": 17,
                    "T": int(args.T),
                    "stride": int(args.stride),
                    "hm_size": int(args.hm_size),
                    "sigma": float(args.sigma),
                    "normalize_xy": bool(normalize_xy),
                    "conf_thres": float(args.conf_thres),
                    "max_interp_gap": int(args.max_interp_gap),
                    "label_mode": str(args.label_mode),
                    "binary_any_fall": bool(binary_any_fall),
                    "fall_class_ids_raw": list(args.fall_class_ids) if args.fall_class_ids else None,
                    "early_stop_metric": monitor,
                    "early_stop_patience": int(patience),
                    "early_stop_min_delta": float(min_delta),
                },
                ckpt_path,
            )

            print(f"  ✓ New best {monitor}: {best_monitor:.4f} (saved checkpoint)")
        else:
            if patience > 0:
                epochs_no_improve += 1
                print(f"  no improvement in {monitor} for {epochs_no_improve}/{patience} epoch(s)")

        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping: {monitor} did not improve for {patience} epochs. Stopping at epoch {ep}.")
            break

    dt = time.time() - t0
    if monitor == "val_acc":
        print(f"\nBest val acc: {best_va_at_best:.3f} at epoch {best_ep}, val loss {best_vl_at_best:.4f}")
    else:
        print(f"\nBest val loss: {best_vl_at_best:.4f} at epoch {best_ep}, val acc {best_va_at_best:.3f}")
    print(f"Saved best checkpoint to: {ckpt_path.as_posix()}")
    print(f"Train seconds: {dt:.1f}")
