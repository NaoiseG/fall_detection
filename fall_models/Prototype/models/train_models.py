"""
Run from project root:

Single model:
  python -m models.train_models --model tcn

Multiple models:
  python -m models.train_models --model tcn lstm gru

All models:
  python -m models.train_models --all

Save results table:
  python -m models.train_models --all --save-results results.csv
"""

from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Optional
import argparse
from pathlib import Path
import glob
import time
import csv
from datetime import datetime


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import load_windows_from_npzs, find_keypoints_npzs_subjects, WindowTensorDataset



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
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Results
# ----------------------------

@dataclass
class RunResult:
    model: str
    best_val_acc: float
    best_val_loss: float
    best_epoch: int
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


def train_model_once(
    model_name: str,
    cfg: TrainConfig,
    in_features: int,
    num_classes: int,
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
) -> RunResult:
    model_name = model_name.lower().strip()

    run_dir = ckpt_root / model_name / run_id
    ckpt_path = run_dir / f"{model_name}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    model = get_model(
        model_name=model_name,
        in_features=in_features,
        num_classes=num_classes,
        device=cfg.device,
        T_used=T_used,
        node_features=node_features,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_va = -1.0
    best_vl = float("inf")
    best_epoch = -1

    final_va = -1.0
    final_vl = float("inf")

    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, cfg.device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, cfg.device)

        final_va, final_vl = va_acc, va_loss

        if va_acc > best_va:
            best_va = va_acc
            best_vl = va_loss
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "in_features": in_features,
                "num_classes": num_classes,
                "use_conf": bool(use_conf),
                "normalize": bool(normalize),
                "add_vel": bool(add_vel),
                "add_acc": bool(add_acc),
                "add_global": bool(add_global),
                "conf_thres": conf_thres,
                "max_interp_gap": max_interp_gap,
                "T_used": T_used,
                "stride": stride,
                "label_mode": label_mode,
                "min_valid_frac": min_valid_frac,
                "add_mask_channel": bool(add_mask_channel),
                "drop_ambig_share": float(drop_ambig_share),
                "drop_ambig_nonfall_only": bool(drop_ambig_nonfall_only),
                "fall_class_ids_raw": list(fall_class_ids_raw) if fall_class_ids_raw is not None else None,
                "node_features": int(node_features) if node_features is not None else None,
            }, ckpt_path)

        print(
            f"{model_name.upper()} | Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

    dt = time.time() - t0
    res = RunResult(
        model=model_name,
        best_val_acc=float(best_va),
        best_val_loss=float(best_vl),
        best_epoch=int(best_epoch),
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
    # Sort by best val acc desc
    results = sorted(results, key=lambda r: r.best_val_acc, reverse=True)

    headers = [
        "model",
        "best_val_acc",
        "best_val_loss",
        "best_epoch",
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
            f"{100.0 * r.best_val_acc:.2f}%",
            f"{r.best_val_loss:.4f}",
            str(r.best_epoch),
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
    fieldnames = list(asdict(results[0]).keys()) if results else list(asdict(RunResult(
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

    ALL_MODELS = ["tcn", "lstm", "gru", "gcn", "mlp", "stgcn"]

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
    #Data preprocessing options
    parser.add_argument("--use-conf", type=int, default=1, help="Include keypoint confidence channel (0/1).")
    parser.add_argument("--normalize", type=int, default=1, help="Normalise pose per frame (0/1).")
    parser.add_argument("--add-vel", type=int, default=1, help="Add velocity channels vx, vy (0/1).")
    parser.add_argument("--add-acc", type=int, default=1, help="Add acceleration channels ax, ay (0/1).")
    parser.add_argument("--add-global", type=int, default=1, help="Add global features (0/1).")
    parser.add_argument("--conf-thres", type=float, default=0.2, help="Conf threshold below which joints are treated as missing.")
    parser.add_argument("--max-interp-gap", type=int, default=5, help="Max gap (frames) for linear interpolation of missing joints.")
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
                        help="Train-only: drop windows where top-label share < this value. 0 disables.")
    parser.add_argument("--drop-ambig-nonfall-only", type=int, default=1,
                        help="Train-only: if 1, only drop ambiguous windows that contain no fall frames (requires --fall-class-ids).")
    parser.add_argument(
        "--fall-class-ids",
        nargs="+",
        type=int,
        default=None,
        help="Fall class IDs in the SAME space as frame_labels in the NPZ (typically 1-based), e.g. --fall-class-ids 1 2 3 4 5"
    )
    parser.add_argument(
        "--fall-pct",
        type=float,
        default=0.25,
        help="Used only when --label-mode hybrid_center_fallpct. Window is labeled fall if >= fall_pct of valid frames are fall. Try 0.20-0.30."
    )
    args = parser.parse_args()

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
    
    if args.label_mode == "hybrid_center_fallpct" and (args.fall_class_ids is None or len(args.fall_class_ids) == 0):
        raise SystemExit("--label-mode hybrid_center_fallpct requires --fall-class-ids (e.g. 1 2 3 4 5).")


    # Decide which models to run
    if args.all:
        model_list = ALL_MODELS
    else:
        if args.model is None or len(args.model) == 0:
            raise SystemExit("You must pass --model <one or more> or use --all.")
        model_list = [m.lower().strip() for m in args.model]

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

    # Paths
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints/outputs_npz")
    ckpt_root = Path("models")

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

    X_train, y_train_tags, T_used = load_windows_from_npzs(
        train_npzs,
        T=int(args.T),
        use_conf=use_conf,
        normalize=normalize,
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        conf_thres=float(args.conf_thres),
        max_interp_gap=int(args.max_interp_gap),
        stride=args.stride,
        label_mode=args.label_mode,
        min_valid_frac=args.min_valid_frac,
        add_mask_channel=add_mask_channel,
        fall_ids_0based=fall_class_ids_raw,
        fall_pct=args.fall_pct,
        drop_ambig_share=float(args.drop_ambig_share),
        drop_ambig_nonfall_only=bool(args.drop_ambig_nonfall_only),
    )

    X_val, y_val_tags, _ = load_windows_from_npzs(
        val_npzs,
        T=int(T_used),
        use_conf=use_conf,
        normalize=normalize,
        add_vel=add_vel,
        add_acc=add_acc,
        add_global=add_global,
        conf_thres=float(args.conf_thres),
        max_interp_gap=int(args.max_interp_gap),
        stride=args.stride,
        label_mode=args.label_mode,
        min_valid_frac=args.min_valid_frac,
        add_mask_channel=add_mask_channel,
        fall_ids_0based=fall_class_ids_raw,
        fall_pct=args.fall_pct,
        drop_ambig_share=0.0,
        drop_ambig_nonfall_only=bool(args.drop_ambig_nonfall_only),
    )

    y_train = (y_train_tags.astype(int) - 1).astype(np.int64)
    y_val   = (y_val_tags.astype(int) - 1).astype(np.int64)

    num_classes = int(max(y_train.max(), y_val.max()) + 1)
    print("num_classes:", num_classes, "| T_used:", T_used)

    # Build datasets just to infer in_features and to avoid duplicating mapping logic
    train_ds = WindowTensorDataset(X_train, y_train)
    val_ds   = WindowTensorDataset(X_val, y_val)

    sample_X, _ = train_ds[0]
    in_features = int(sample_X.shape[-1])

    K = 17
    node_features = int(in_features // K)
    if node_features * K != in_features:
        node_features = None

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ---- Now actually use your loader function ----
    # NOTE: make_loader currently returns y as raw labels (1..11). We need 0..10.
    # So we’ll wrap it by creating loaders from the datasets we already built:
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=False, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Train selected models and collect results
    results: List[RunResult] = []
    for m in model_list:
        res = train_model_once(
            model_name=m,
            cfg=cfg,
            in_features=in_features,
            num_classes=num_classes,
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
        )
        results.append(res)
        

    # Print table
    print_results_table(results)

    # Save table
    if args.save_results:
        out_path = Path(args.save_results)
        save_results_csv(results, out_path)
        print(f"\nSaved results CSV to: {out_path.as_posix()}")
