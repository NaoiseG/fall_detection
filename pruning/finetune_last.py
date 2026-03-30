#!/usr/bin/env python3
"""
Fine-tune a completed pruning run's exported ``last.pt`` checkpoint.

This script is meant for the common follow-up case after ``prune_pose.py`` has
finished and written a normal Ultralytics checkpoint at ``weights/last.pt``.
It accepts either:
  - a direct path to a ``.pt`` checkpoint, or
  - a pruning run directory containing ``weights/last.pt``.

Examples:
  python pruning/finetune_last.py \
      --weights quantisation/pruned_models/pruned_pose_yolo11n_pose_flops90p_trainfrac25p

  python pruning/finetune_last.py \
      --weights quantisation/pruned_models/pruned_pose_yolo11n_pose_flops90p_trainfrac25p/weights/last.pt \
      --epochs 40 \
      --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from ultralytics import YOLO
from ultralytics.nn.modules import Detect, Pose


ROOT = Path(__file__).resolve().parent


def resolve_weights_path(weights_arg: str) -> Path:
    """Resolve either a direct checkpoint path or a pruning run directory."""
    path = Path(weights_arg).expanduser().resolve()

    if path.is_file():
        if path.suffix.lower() != ".pt":
            raise ValueError(f"Expected a .pt checkpoint, got: {path}")
        return path

    if path.is_dir():
        candidates = [
            path / "weights" / "last.pt",
            path / "last.pt",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(
            "Could not find last.pt under the provided directory. "
            f"Checked: {', '.join(str(candidate) for candidate in candidates)}"
        )

    raise FileNotFoundError(f"Checkpoint path does not exist: {path}")


def infer_source_run_dir(weights_path: Path) -> Path:
    """Infer the originating pruning run directory from the checkpoint path."""
    if weights_path.parent.name == "weights":
        return weights_path.parent.parent
    return weights_path.parent


def detect_task(model: YOLO) -> str:
    """Detect whether the loaded checkpoint is a pose or detect model."""
    task = getattr(model, "task", None)
    if task in {"pose", "detect"}:
        return str(task)

    head = model.model.model[-1]
    if isinstance(head, Pose):
        return "pose"
    if isinstance(head, Detect):
        return "detect"
    raise RuntimeError(f"Unsupported terminal head type: {type(head).__name__}")


def resolve_task(user_task: str, detected_task: str) -> str:
    if user_task == "auto":
        return detected_task
    if user_task != detected_task:
        raise ValueError(f"--task={user_task} does not match loaded model task '{detected_task}'")
    return user_task


def resolve_data_path(task: str, data_arg: Optional[str]) -> Path:
    if data_arg:
        return Path(data_arg).expanduser().resolve()

    if task == "pose":
        return (ROOT / "coco-pose.yaml").resolve()

    raise ValueError("Please provide --data for non-pose fine-tuning runs.")


def resolve_output_location(
    weights_path: Path,
    project_arg: Optional[str],
    name_arg: Optional[str],
) -> Tuple[Path, str]:
    source_run_dir = infer_source_run_dir(weights_path)

    if project_arg:
        project_dir = Path(project_arg).expanduser().resolve()
    else:
        project_dir = source_run_dir / "finetune_runs"

    if name_arg:
        run_name = name_arg
    else:
        run_name = f"{source_run_dir.name}_finetune"

    return project_dir, run_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pruning run's exported last.pt checkpoint."
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to a .pt checkpoint or a pruning run directory containing weights/last.pt.",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "detect", "pose"],
        help="Override the detected task if needed.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Dataset YAML path. Defaults to pruning/coco-pose.yaml for pose checkpoints.",
    )
    parser.add_argument("--project", default=None, help="Ultralytics output project directory.")
    parser.add_argument("--name", default=None, help="Ultralytics run name.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of fine-tuning epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--optimizer", default="SGD", help="Optimizer name.")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0 or cuda:0.")
    parser.add_argument("--save_period", type=int, default=-1, help="Checkpoint save period.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset caching.")
    parser.add_argument("--plots", action="store_true", help="Enable Ultralytics training plots.")
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine learning-rate schedule.")
    parser.add_argument("--exist_ok", action="store_true", help="Allow reusing an existing run name.")
    parser.add_argument("--run_val", action="store_true", help="Run validation after training.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    weights_path = resolve_weights_path(args.weights)
    project_dir, run_name = resolve_output_location(weights_path, args.project, args.name)

    model = YOLO(str(weights_path))
    detected_task = detect_task(model)
    task = resolve_task(args.task, detected_task)
    data_path = resolve_data_path(task, args.data)

    print(f"source checkpoint: {weights_path}")
    print(f"detected task: {detected_task}")
    print(f"fine-tune task: {task}")
    print(f"dataset: {data_path}")
    print(f"project: {project_dir}")
    print(f"run name: {run_name}")

    train_kwargs = {
        "data": str(data_path),
        "task": task,
        "project": str(project_dir),
        "name": run_name,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "workers": args.workers,
        "save_period": args.save_period,
        "cache": args.cache,
        "plots": args.plots,
        "cos_lr": args.cos_lr,
        "exist_ok": args.exist_ok,
        "resume": False,
    }
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)

    if args.run_val:
        model.val(
            data=str(data_path),
            task=task,
            imgsz=args.imgsz,
        )


if __name__ == "__main__":
    main()
