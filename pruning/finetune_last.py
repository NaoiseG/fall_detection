#!/usr/bin/env python3
"""
Fine-tune a completed pruning run's exported checkpoint without rebuilding the
pruned architecture from YAML.

This script is meant for the common follow-up case after ``prune_pose.py`` has
finished and written a normal Ultralytics checkpoint at ``weights/last.pt``.
It accepts either:
  - a direct path to a ``.pt`` checkpoint, or
  - a pruning run directory containing ``weights/last.pt``.

Compared with a plain ``YOLO(...).train(...)`` call, this helper keeps the
checkpoint's serialized pruned module intact and tracks the best checkpoint
using pose ``map50-95`` by default.

Examples:
  python pruning/finetune_last.py \
      --weights quantisation/pruned_models/pruned_pose_yolo11n_pose_flops90p_trainfrac25p

  python pruning/finetune_last.py \
      --weights quantisation/pruned_models/pruned_pose_yolo11n_pose_flops90p_trainfrac25p/weights/last.pt \
      --epochs 40 \
      --device 0 \
      --run_val
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
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


def clear_instance_override(model: torch.nn.Module, attr_name: str) -> None:
    if attr_name in getattr(model, "__dict__", {}):
        delattr(model, attr_name)


def clone_serializable_model(
    model: torch.nn.Module,
    *,
    half: bool = False,
    freeze: bool = False,
) -> torch.nn.Module:
    """Copy a checkpoint model into a clean module that can be trained or saved."""
    model_obj = deepcopy(getattr(model, "module", model))
    clear_instance_override(model_obj, "is_fused")
    if hasattr(model_obj, "criterion"):
        model_obj.criterion = None
    model_obj = model_obj.half() if half else model_obj.float()
    for param in model_obj.parameters():
        param.requires_grad = not freeze
    return model_obj


def checkpoint_model_object(ckpt: Any, *, prefer_ema: bool = False) -> Optional[torch.nn.Module]:
    if isinstance(ckpt, dict):
        keys = ("ema", "model") if prefer_ema else ("model", "ema")
        for key in keys:
            model_obj = ckpt.get(key, None)
            if isinstance(model_obj, torch.nn.Module):
                return model_obj
    if isinstance(ckpt, torch.nn.Module):
        return ckpt
    return None


def load_checkpoint_model_direct(pt_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load the pruned model module directly from a checkpoint when possible.

    This avoids relying on the default Ultralytics rebuild path for the initial
    fine-tune model construction.
    """
    from ultralytics.nn.tasks import guess_model_task
    from ultralytics.utils import DEFAULT_CFG_DICT

    try:
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except Exception:
        ckpt = None
    if isinstance(ckpt, dict):
        model_obj = checkpoint_model_object(ckpt, prefer_ema=True)
        if model_obj is not None:
            model_obj = clone_serializable_model(model_obj, half=False, freeze=False)
            model_obj.args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}) or {})}
            model_obj.pt_path = str(pt_path)
            model_obj.task = getattr(model_obj, "task", guess_model_task(model_obj))
            if not hasattr(model_obj, "stride"):
                model_obj.stride = torch.tensor([32.0])
            return model_obj, ckpt

    y = YOLO(str(pt_path))
    model_obj = clone_serializable_model(y.model, half=False, freeze=False)
    model_obj.args = getattr(y.model, "args", {}) or {}
    model_obj.pt_path = str(pt_path)
    model_obj.task = getattr(y.model, "task", guess_model_task(model_obj))
    if not hasattr(model_obj, "stride"):
        model_obj.stride = torch.tensor([32.0])
    return model_obj, ckpt if isinstance(ckpt, dict) else {}


def _first_present(results_dict: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in results_dict and results_dict[key] is not None:
            try:
                return float(results_dict[key])
            except Exception:
                continue
    return None


def pose_metric_dict_keys(metric: str) -> List[str]:
    metric = metric.lower().strip()
    key_map = {
        "map5095": ["metrics/mAP50-95(P)", "metrics/mAP50-95"],
        "map50": ["metrics/mAP50(P)", "metrics/mAP50"],
        "precision": ["metrics/precision(P)", "metrics/precision"],
        "recall": ["metrics/recall(P)", "metrics/recall"],
    }
    return key_map.get(metric, [])


def box_metric_dict_keys(metric: str) -> List[str]:
    metric = metric.lower().strip()
    key_map = {
        "map5095": ["metrics/mAP50-95(B)", "metrics/mAP50-95"],
        "map50": ["metrics/mAP50(B)", "metrics/mAP50"],
        "precision": ["metrics/precision(B)", "metrics/precision"],
        "recall": ["metrics/recall(B)", "metrics/recall"],
    }
    return key_map.get(metric, [])


def csv_metric_keys(task: str, metric: str) -> List[str]:
    task = task.lower().strip()
    if task == "pose":
        return pose_metric_dict_keys(metric)
    return box_metric_dict_keys(metric)


def score_from_results_csv(save_dir: Path, metric_keys: Iterable[str], epoch_idx: int) -> Optional[float]:
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.exists():
        return None

    try:
        with open(csv_path, "r", newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))
        if not rows or epoch_idx < 0 or epoch_idx >= len(rows):
            return None
        return _first_present(rows[epoch_idx], metric_keys)
    except Exception:
        return None


def parse_validation_output(out: Any, trainer: Optional[Any] = None) -> Tuple[Any, Dict[str, Any], Optional[float]]:
    """Normalize Ultralytics validation outputs into metrics object, dict, and fitness."""
    metrics_obj = out
    results_dict: Dict[str, Any] = {}
    fitness: Optional[float] = None

    if isinstance(out, tuple):
        if len(out) >= 1:
            metrics_obj = out[0]
        for item in out[1:]:
            if isinstance(item, dict) and not results_dict:
                results_dict = item
            else:
                try:
                    fitness = float(item)
                except Exception:
                    pass

    if isinstance(metrics_obj, dict) and not results_dict:
        results_dict = metrics_obj

    maybe_results = getattr(metrics_obj, "results_dict", None)
    if isinstance(maybe_results, dict) and not results_dict:
        results_dict = maybe_results

    maybe_fitness = getattr(metrics_obj, "fitness", None)
    if fitness is None and maybe_fitness is not None and not callable(maybe_fitness):
        try:
            fitness = float(maybe_fitness)
        except Exception:
            pass

    if trainer is not None:
        trainer_metrics = getattr(trainer, "metrics", None)
        if isinstance(trainer_metrics, dict) and not results_dict:
            results_dict = trainer_metrics
        elif trainer_metrics is not None and not results_dict:
            maybe_results = getattr(trainer_metrics, "results_dict", None)
            if isinstance(maybe_results, dict):
                results_dict = maybe_results

        validator = getattr(trainer, "validator", None)
        validator_metrics = getattr(validator, "metrics", None) if validator is not None else None
        if isinstance(validator_metrics, dict) and not results_dict:
            results_dict = validator_metrics
        elif validator_metrics is not None and not results_dict:
            maybe_results = getattr(validator_metrics, "results_dict", None)
            if isinstance(maybe_results, dict):
                results_dict = maybe_results

        if fitness is None and isinstance(results_dict, dict):
            raw_fitness = results_dict.get("fitness", None)
            if raw_fitness is not None:
                try:
                    fitness = float(raw_fitness)
                except Exception:
                    pass

    return metrics_obj, results_dict, fitness


def extract_metric_score(metrics_obj: Any, results_dict: Dict[str, Any], metric: str, task: str) -> float:
    """Extract the project metric, preferring pose metrics for pose models."""
    metric = metric.lower().strip()
    task = task.lower().strip()

    attr_lookup = {
        "map5095": "map",
        "map50": "map50",
        "precision": "p",
        "recall": "r",
    }
    attr_name = attr_lookup[metric]

    if task == "pose":
        pose_metrics = getattr(metrics_obj, "pose", None)
        if pose_metrics is not None and hasattr(pose_metrics, attr_name):
            value = getattr(pose_metrics, attr_name)
            if value is not None:
                return float(value)
    else:
        box_metrics = getattr(metrics_obj, "box", None)
        if box_metrics is not None and hasattr(box_metrics, attr_name):
            value = getattr(box_metrics, attr_name)
            if value is not None:
                return float(value)

    keys = pose_metric_dict_keys(metric) if task == "pose" else box_metric_dict_keys(metric)
    score = _first_present(results_dict, keys)
    if score is not None:
        return score

    raise RuntimeError(
        f"Could not extract {task} metric '{metric}'. "
        f"Available results_dict keys: {sorted(list(results_dict.keys()))}"
    )


def get_ema_state_dict(trainer: Any) -> Dict[str, torch.Tensor]:
    """Prefer EMA weights when capturing the current best checkpoint."""
    ema = getattr(trainer, "ema", None)
    if ema is not None and getattr(ema, "ema", None) is not None:
        state_dict = ema.ema.state_dict()
    else:
        state_dict = trainer.model.state_dict()
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def save_yolo_checkpoint(model_wrapper: YOLO, out_pt: Path) -> None:
    try:
        model_wrapper.save(str(out_pt), use_dill=False)
    except TypeError as exc:
        if "use_dill" not in str(exc):
            raise
        model_wrapper.save(str(out_pt))


def repack_to_ultralytics_pt(base_pt: Path, state_dict: Dict[str, torch.Tensor], out_pt: Path) -> None:
    """Write a standard Ultralytics checkpoint with the exact pruned architecture."""
    y = YOLO(str(base_pt))
    y.model = clone_serializable_model(y.model, half=False, freeze=False)
    y.model.load_state_dict(state_dict, strict=True)
    clear_instance_override(y.model, "is_fused")
    save_yolo_checkpoint(y, out_pt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pruning run's exported checkpoint without rebuilding the pruned model."
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
    parser.add_argument("--metric", choices=["map5095", "map50", "precision", "recall"], default="map5095")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset caching.")
    parser.add_argument("--plots", action="store_true", help="Enable Ultralytics training plots.")
    parser.add_argument("--cos_lr", action="store_true", help="Use cosine learning-rate schedule.")
    parser.add_argument("--multi_scale", action="store_true", help="Enable multi-scale training.")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Ultralytics label smoothing value.")
    parser.add_argument("--exist_ok", action="store_true", help="Allow reusing an existing run name.")
    parser.add_argument("--run_val", action="store_true", help="Run validation on the pose-best checkpoint.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    weights_path = resolve_weights_path(args.weights)
    project_dir, run_name = resolve_output_location(weights_path, args.project, args.name)

    seed_model, _ = load_checkpoint_model_direct(weights_path)
    model = YOLO(str(weights_path))
    detected_task = detect_task(model)
    task = resolve_task(args.task, detected_task)
    data_path = resolve_data_path(task, args.data)

    model.model = clone_serializable_model(seed_model, half=False, freeze=False)
    model.task = task

    print(f"source checkpoint: {weights_path}")
    print(f"detected task: {detected_task}")
    print(f"fine-tune task: {task}")
    print(f"dataset: {data_path}")
    print(f"project: {project_dir}")
    print(f"run name: {run_name}")
    print(f"best metric tracking: {args.metric} ({task}-first)")

    base_task = task or model.task
    BaseTrainer = model.task_map[base_task]["trainer"]

    class SafeFinetuneTrainer(BaseTrainer):
        """Trainer that keeps the pruned architecture intact and tracks pose-first best weights."""

        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self._best_score_custom: float = float("-inf")
            self._best_sd_custom: Optional[Dict[str, torch.Tensor]] = None
            self._best_epoch_custom: Optional[int] = None
            self._last_val_epoch: int = -1

        def get_model(self, cfg=None, weights=None, verbose=True, **kwargs):
            return clone_serializable_model(seed_model, half=False, freeze=False)

        def validate(self):
            out = super().validate()
            epoch_i = int(getattr(self, "epoch", -1))
            self._last_val_epoch = epoch_i

            metrics_obj, results_dict, fitness = parse_validation_output(out, trainer=self)

            try:
                score = extract_metric_score(metrics_obj, results_dict, args.metric, task)
            except Exception:
                score = score_from_results_csv(
                    Path(self.save_dir),
                    csv_metric_keys(task, args.metric),
                    epoch_i,
                )
                if score is None and fitness is not None:
                    score = float(fitness)
                if score is None:
                    return out

            if score > self._best_score_custom:
                previous = self._best_score_custom
                self._best_score_custom = score
                self._best_sd_custom = get_ema_state_dict(self)
                self._best_epoch_custom = epoch_i

                Path(self.save_dir).mkdir(parents=True, exist_ok=True)
                (Path(self.save_dir) / "best_epoch.txt").write_text(
                    f"epoch={epoch_i}\n"
                    f"metric={args.metric}\n"
                    f"score={score:.8f}\n"
                    f"previous_best={previous:.8f}\n"
                )

            return out

        def on_train_epoch_end(self):
            super().on_train_epoch_end()
            epoch_i = int(getattr(self, "epoch", -1))
            if self._last_val_epoch != epoch_i:
                self.validate()

    train_kwargs: Dict[str, Any] = {
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
    if args.multi_scale:
        train_kwargs["multi_scale"] = True
    if args.label_smoothing > 0:
        train_kwargs["label_smoothing"] = args.label_smoothing

    model.train(trainer=SafeFinetuneTrainer, **train_kwargs)

    trainer = getattr(model, "trainer", None)
    if trainer is None:
        raise RuntimeError("Ultralytics trainer was not available after fine-tuning.")

    save_dir = Path(trainer.save_dir)
    weights_dir = save_dir / "weights"
    best_path = weights_dir / "best.pt"

    if getattr(trainer, "_best_sd_custom", None) is not None:
        repack_to_ultralytics_pt(weights_path, trainer._best_sd_custom, best_path)
        if getattr(trainer, "_best_score_custom", None) is not None:
            (weights_dir / "best_score.txt").write_text(f"{trainer._best_score_custom:.8f}\n")
        print(
            "[INFO] Overwrote best.pt using "
            f"{task} {args.metric} best epoch={trainer._best_epoch_custom} "
            f"score={trainer._best_score_custom:.6f}"
        )
    else:
        print("[WARN] Could not derive a custom pose-first best checkpoint; leaving Ultralytics best.pt unchanged.")

    if args.run_val:
        val_model_path = best_path if best_path.exists() else (weights_dir / "last.pt")
        val_model = YOLO(str(val_model_path))
        val_kwargs: Dict[str, Any] = {
            "data": str(data_path),
            "task": task,
            "imgsz": args.imgsz,
        }
        if args.device:
            val_kwargs["device"] = args.device
        val_model.val(**val_kwargs)


if __name__ == "__main__":
    main()
