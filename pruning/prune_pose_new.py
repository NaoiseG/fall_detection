#!/usr/bin/env python3
"""
Project-specific structured pruning + fine-tuning for Ultralytics YOLO11 pose models
using NVIDIA ModelOpt FastNAS.

This script is adapted from the attached pruning workflow, but made explicitly pose-first:
  - pruning task defaults to pose
  - best model selection is STRICTLY pose mAP50-95
  - FastNAS search scoring uses pose mAP50-95 when available
  - validation fallback logic prefers pose metrics, never box metrics, for task=pose

Final loadable artifacts per completed run:
  - weights/last.pt
  - weights/best.pt
  - weights/best_score.txt
  - weights/artifacts_manifest.txt
  - best_epoch.txt

The script keeps the attached script's general pattern:
  - custom trainer
  - prune inside _setup_train()
  - custom validation tracking
  - best weights kept in RAM
  - Ultralytics checkpoint saving disabled during training to avoid ModelOpt pickling issues
  - final weights re-packed into real Ultralytics .pt files
  - optional subset fine-tuning/search via --train_fraction

Example:
  python prune_pose.py \
    --models /home/people/21376026/scratch/prune_models/yolo11n-pose/yolo11n-pose.pt \
             /home/people/21376026/scratch/prune_models/yolo11s-pose/yolo11s-pose.pt \
    --data /home/people/21376026/fall_detection/pruning/coco-pose.yaml \
    --task pose \
    --imgsz 640 \
    --batch 16 \
    --patience 20 \
    --workers 8 \
    --flops 90% 80% 70% \
    --name_prefix pruned_pose \
    --project /home/people/21376026/scratch/pruned_pose_runs \
    --device 0 \
    --cache \
    --train_fraction 0.2
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from ultralytics import YOLO

try:
    import torchprofile.handlers as tp_handlers
    import torchprofile.profile as tp_profile
except Exception:
    tp_handlers = None
    tp_profile = None


def _normalize_torchprofile_handlers() -> None:
    if tp_handlers is None or tp_profile is None:
        return

    def _coerce_handler_pairs(raw_handlers: Any) -> Optional[List[Tuple[Any, Any]]]:
        if raw_handlers is None:
            return None
        if isinstance(raw_handlers, dict):
            pairs = list(raw_handlers.items())
        else:
            try:
                pairs = list(raw_handlers)
            except TypeError:
                return None

        if not pairs:
            return pairs
        if all(isinstance(item, tuple) and len(item) == 2 for item in pairs):
            return pairs
        return None

    # ModelOpt iterates torchprofile.profile.handlers as (op_names, op) pairs.
    normalized_handlers = None
    for attr_name in ("handlers", "HANDLER_MAP"):
        candidate_handlers = _coerce_handler_pairs(getattr(tp_handlers, attr_name, None))
        if candidate_handlers is None:
            continue
        normalized_handlers = candidate_handlers
        if candidate_handlers:
            break
    if normalized_handlers is not None:
        tp_profile.handlers = normalized_handlers


_normalize_torchprofile_handlers()

import modelopt.torch.nas as mtn
import modelopt.torch.opt as mto
import modelopt.torch.prune as mtp


# ---------- pickling-safe / top-level helpers ----------


def always_true() -> bool:
    """Small top-level helper used to mimic the attached script's fused-model workaround."""
    return True


def clear_instance_override(model: torch.nn.Module, attr_name: str) -> None:
    """Drop instance-level monkey patches so checkpoints only rely on class-defined behavior."""
    if attr_name in getattr(model, "__dict__", {}):
        delattr(model, attr_name)


def get_last_modelopt_mode_name(model: torch.nn.Module) -> Optional[str]:
    """Return the raw last mode name from saved ModelOpt state without requiring registry lookup."""
    state = getattr(model, "_modelopt_state", None)
    if not state:
        return None
    try:
        return str(state[-1][0])
    except Exception:
        return None


def finalize_pruned_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Export a FastNAS-pruned model back to a regular module and strip ModelOpt metadata.

    This keeps the final checkpoints loadable in plain Ultralytics environments.
    """
    model = getattr(model, "module", model)
    clear_instance_override(model, "is_fused")

    last_mode_name = get_last_modelopt_mode_name(model)
    already_exported = last_mode_name in {"export", "export_nas"}

    if mto.ModeloptStateManager.is_converted(model) and not already_exported:
        model = mtn.export(model)

    if hasattr(model, "_modelopt_state"):
        mto.ModeloptStateManager.remove_state(model)
    if hasattr(model, "_modelopt_state_version"):
        delattr(model, "_modelopt_state_version")

    clear_instance_override(model, "is_fused")
    if hasattr(model, "criterion"):
        model.criterion = None
    return model


def model_needs_checkpoint_normalization(model: Optional[torch.nn.Module]) -> bool:
    if model is None:
        return False
    model = getattr(model, "module", model)
    return any(
        [
            "is_fused" in getattr(model, "__dict__", {}),
            hasattr(model, "_modelopt_state"),
            hasattr(model, "_modelopt_state_version"),
            hasattr(model, "criterion"),
        ]
    )


def clone_serializable_model(
    model: torch.nn.Module,
    *,
    half: bool = True,
    freeze: bool = True,
) -> torch.nn.Module:
    """Create a checkpoint-safe copy of a model with pruning metadata stripped."""
    from copy import deepcopy

    model_obj = deepcopy(getattr(model, "module", model))
    model_obj = finalize_pruned_model(model_obj)
    if half:
        model_obj = model_obj.half()
    else:
        model_obj = model_obj.float()
    if hasattr(model_obj, "criterion"):
        model_obj.criterion = None
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


def is_resume_checkpoint_path(path_like: Any) -> bool:
    try:
        name = Path(path_like).name
    except Exception:
        return False
    return name in {RESUME_LAST_CKPT_NAME, RESUME_BEST_CKPT_NAME}


def load_checkpoint_model_direct(pt_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load the pruned model module directly from a resumable checkpoint.

    Ultralytics' default trainer path rebuilds a model from weights.yaml, which loses the pruned channel
    structure. For resume checkpoints we instead take the serialized module object itself.
    """
    from ultralytics.nn.tasks import guess_model_task
    from ultralytics.utils import DEFAULT_CFG_DICT

    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected dict checkpoint at {pt_path}, got {type(ckpt)}")

    model_obj = checkpoint_model_object(ckpt, prefer_ema=True)
    if model_obj is None:
        raise RuntimeError(f"Checkpoint {pt_path} does not contain a loadable model or ema module.")

    model_obj = clone_serializable_model(model_obj, half=False, freeze=False)
    model_obj.args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}) or {})}
    model_obj.pt_path = str(pt_path)
    model_obj.task = getattr(model_obj, "task", guess_model_task(model_obj))
    if not hasattr(model_obj, "stride"):
        model_obj.stride = torch.tensor([32.0])
    return model_obj, ckpt


def ensure_resume_checkpoint_compat(path: Path) -> bool:
    """
    Upgrade legacy resumable checkpoints in place so Ultralytics resume sees a real pruned model.

    Older checkpoints stored model=None and only kept the EMA module, which is enough for this script to
    serialize but can cause architecture mismatches on resume/final repack.
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return False

    changed = False
    source_model = checkpoint_model_object(ckpt)
    if source_model is None:
        return False

    if ckpt.get("model", None) is None or model_needs_checkpoint_normalization(ckpt.get("model", None)):
        ckpt["model"] = clone_serializable_model(ckpt.get("model", None) or source_model)
        changed = True

    ema_model = ckpt.get("ema", None)
    if ema_model is None:
        ckpt["ema"] = clone_serializable_model(source_model)
        changed = True
    elif model_needs_checkpoint_normalization(ema_model):
        ckpt["ema"] = clone_serializable_model(ema_model)
        changed = True

    if changed:
        torch.save(ckpt, str(path))
    return changed


def safe_tag(text: str) -> str:
    return text.replace("%", "p").replace("/", "_").replace("-", "_").replace(" ", "_")


RESUME_LAST_CKPT_NAME = "resume_last.pt"
RESUME_BEST_CKPT_NAME = "resume_best.pt"


def resolve_search_checkpoint_path(run_dir: Path, search_ckpt: str) -> Path:
    path = Path(search_ckpt)
    return path if path.is_absolute() else run_dir / path.name


def resume_checkpoint_paths(run_dir: Path) -> Tuple[Path, Path]:
    weights_dir = run_dir / "weights"
    return weights_dir / RESUME_LAST_CKPT_NAME, weights_dir / RESUME_BEST_CKPT_NAME


def fraction_suffix(train_fraction: float) -> str:
    """
    Stable run-name suffix so subset runs do not overwrite full-data runs.
    Examples:
      1.0 -> ""
      0.2 -> "_trainfrac20p"
      0.125 -> "_trainfrac12_5p"
    """
    try:
        frac = float(train_fraction)
    except Exception:
        return ""

    if frac >= 0.999999:
        return ""

    pct = frac * 100.0
    pct_text = f"{pct:.4f}".rstrip("0").rstrip(".")
    return f"_trainfrac{safe_tag(pct_text)}p"


def model_run_tag(model_path: str) -> str:
    return safe_tag(Path(model_path).stem)


def parse_target_percent(flops_target: str) -> Optional[int]:
    digits = "".join(ch for ch in str(flops_target) if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def resolve_epochs_for_target(flops_target: str, args: argparse.Namespace) -> int:
    """
    Per-target schedule when --epochs is omitted:
      90% -> 60
      80% -> 80
      70% -> 80
    Any other target falls back to 60.
    """
    if args.epochs is not None:
        return int(args.epochs)

    pct = parse_target_percent(flops_target)
    if pct == 90:
        return 60
    if pct == 80:
        return 80
    if pct == 70:
        return 80
    return 60


def expected_run_dir(args: argparse.Namespace, run_name: str) -> Path:
    if args.project:
        return Path(args.project) / run_name
    return Path("runs") / args.task / run_name


def run_is_complete(run_dir: Path) -> bool:
    weights_dir = run_dir / "weights"
    return all(
        p.exists()
        for p in [
            weights_dir / "last.pt",
            weights_dir / "best.pt",
            weights_dir / "best_score.txt",
            weights_dir / "artifacts_manifest.txt",
            run_dir / "best_epoch.txt",
        ]
    )


def cleanup_stale_run_outputs(
    run_dir: Path,
    search_ckpt_path: Optional[Path] = None,
    *,
    preserve_search_ckpt: bool = False,
) -> None:
    """
    Keep the run directory stable for restart-friendliness, but remove stale files that can
    confuse later validation/manifest generation on rerun.
    """
    weights_dir = run_dir / "weights"
    if weights_dir.exists():
        shutil.rmtree(weights_dir, ignore_errors=True)

    stale_paths = [
        run_dir / "results.csv",
        run_dir / "best_epoch.txt",
    ]
    if search_ckpt_path is not None and not preserve_search_ckpt:
        stale_paths.append(search_ckpt_path)

    for path in stale_paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def write_manifest(
    manifest_path: Path,
    *,
    status: str,
    run_name: str,
    model_path: str,
    task: str,
    flops_target: str,
    metric: str,
    epochs: int,
    train_fraction: float,
    reason: Optional[str] = None,
    best_score: Optional[float] = None,
    best_epoch_text: Optional[str] = None,
    last_epoch: Optional[int] = None,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"status={status}",
        f"run_name={run_name}",
        f"model_path={model_path}",
        f"task={task}",
        f"flops_target={flops_target}",
        f"metric={metric}",
        f"epochs={epochs}",
        f"train_fraction={train_fraction}",
    ]
    if last_epoch is not None:
        lines.append(f"last_epoch={last_epoch}")
    if best_score is not None:
        lines.append(f"best_score={best_score:.8f}")
    if reason:
        lines.append(f"reason={reason}")
    if best_epoch_text:
        lines.append(best_epoch_text.strip())

    manifest_path.write_text("\n".join(lines) + "\n")


def get_ema_state_dict(trainer) -> Dict[str, torch.Tensor]:
    """
    Prefer EMA weights to match Ultralytics best.pt behavior as closely as possible.
    Store on CPU to keep the tracked best state RAM-safe and pickling-safe.
    """
    ema = getattr(trainer, "ema", None)
    if ema is not None and getattr(ema, "ema", None) is not None:
        state_dict = ema.ema.state_dict()
    else:
        state_dict = trainer.model.state_dict()
    return {k: v.detach().cpu() for k, v in state_dict.items()}


def save_yolo_checkpoint(y: YOLO, out_pt: Path) -> None:
    try:
        y.save(str(out_pt), use_dill=False)
    except TypeError as exc:
        if "use_dill" not in str(exc):
            raise
        y.save(str(out_pt))


def repack_to_ultralytics_pt(base_pt: Path, state_dict: Dict[str, torch.Tensor], out_pt: Path) -> None:
    """
    Rebuild a normal Ultralytics .pt using a saved pruned-architecture base checkpoint.
    This preserves a loadable Ultralytics artifact instead of a raw state_dict-only file.
    """
    y = YOLO(str(base_pt))
    y.model = finalize_pruned_model(y.model)
    y.model.load_state_dict(state_dict, strict=True)
    clear_instance_override(y.model, "is_fused")
    save_yolo_checkpoint(y, out_pt)


def load_state_dict_from_ultralytics_pt(pt_path: Path) -> Dict[str, torch.Tensor]:
    try:
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except Exception:
        ckpt = None

    model_obj = checkpoint_model_object(ckpt, prefer_ema=True)
    if model_obj is not None:
        model_obj = finalize_pruned_model(model_obj)
        return {k: v.detach().cpu() for k, v in model_obj.state_dict().items()}

    y = YOLO(str(pt_path))
    y.model = finalize_pruned_model(y.model)
    return {k: v.detach().cpu() for k, v in y.model.state_dict().items()}


def load_checkpoint_dict(pt_path: Path) -> Optional[Dict[str, Any]]:
    try:
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    except Exception:
        return None
    return ckpt if isinstance(ckpt, dict) else None


def checkpoint_epoch(pt_path: Path) -> Optional[int]:
    ckpt = load_checkpoint_dict(pt_path)
    if ckpt is None:
        return None
    try:
        return int(ckpt.get("epoch", -1))
    except Exception:
        return None


def checkpoint_best_score(pt_path: Path) -> Optional[float]:
    ckpt = load_checkpoint_dict(pt_path)
    if ckpt is None:
        return None
    for key in ("custom_best_score", "best_fitness"):
        value = ckpt.get(key, None)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def finalize_run_artifacts(
    *,
    run_name: str,
    model_path: str,
    task: str,
    flops_target: str,
    metric: str,
    epochs: int,
    train_fraction: float,
    out_dir: Path,
    base_pt: Path,
    last_sd: Dict[str, torch.Tensor],
    best_sd: Optional[Dict[str, torch.Tensor]],
    resume_last_path: Path,
    resume_best_path: Path,
    search_ckpt_path: Optional[Path] = None,
    best_score: Optional[float] = None,
    last_epoch: Optional[int] = None,
) -> None:
    out_dir = Path(out_dir)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    meta_paths = [resume_best_path, resume_last_path, base_pt]
    if best_score is None:
        for meta_path in meta_paths:
            if meta_path.exists():
                best_score = checkpoint_best_score(meta_path)
                if best_score is not None:
                    break
    if last_epoch is None:
        for meta_path in [resume_last_path, base_pt]:
            if meta_path.exists():
                last_epoch = checkpoint_epoch(meta_path)
                if last_epoch is not None:
                    break

    best_sd = best_sd or last_sd
    repack_to_ultralytics_pt(base_pt, last_sd, weights_dir / "last.pt")
    repack_to_ultralytics_pt(base_pt, best_sd, weights_dir / "best.pt")

    if search_ckpt_path is not None:
        try:
            Path(search_ckpt_path).unlink()
        except Exception:
            pass

    for resume_ckpt in [resume_last_path, resume_best_path]:
        try:
            if resume_ckpt.exists():
                resume_ckpt.unlink()
        except Exception:
            pass

    best_epoch_path = out_dir / "best_epoch.txt"
    best_epoch_text = best_epoch_path.read_text() if best_epoch_path.exists() else "epoch=unknown\n"

    if best_score is not None:
        (weights_dir / "best_score.txt").write_text(f"{best_score:.8f}\n")
    else:
        (weights_dir / "best_score.txt").write_text("nan\n")

    write_manifest(
        weights_dir / "artifacts_manifest.txt",
        status="completed",
        run_name=run_name,
        model_path=model_path,
        task=task,
        flops_target=flops_target,
        metric=metric,
        epochs=epochs,
        train_fraction=train_fraction,
        best_score=best_score,
        best_epoch_text=best_epoch_text,
        last_epoch=last_epoch,
    )


def normalize_modelopt_prune_result(prune_result: Any) -> Tuple[Any, Any]:
    """
    Handle small ModelOpt return-shape differences without changing the pruning flow.
    """
    if isinstance(prune_result, tuple):
        if len(prune_result) >= 2:
            return prune_result[0], prune_result[1]
        if len(prune_result) == 1:
            return prune_result[0], {}
    return prune_result, {}


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


def score_from_results_csv(save_dir: Path, metric_keys: Sequence[str], epoch_idx: int) -> Optional[float]:
    """
    Read a metric value from Ultralytics results.csv for the given 0-based epoch index.
    For pose, prefer pose columns first and only then generic columns.
    """
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.exists():
        return None

    try:
        with open(csv_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        if epoch_idx < 0 or epoch_idx >= len(rows):
            return None
        row = rows[epoch_idx]
        return _first_present(row, metric_keys)
    except Exception:
        return None


def parse_validation_output(out: Any, trainer: Optional[Any] = None) -> Tuple[Any, Dict[str, Any], Optional[float]]:
    """
    Normalize several Ultralytics validate() / validator() return styles into:
      (metrics_obj, results_dict, fitness)
    """
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
    """
    Pose-first metric extraction.

    For task=pose, the order is:
      1) attribute-based pose metrics_obj.pose.*
      2) dict-based pose keys, preferring (P)
      3) generic dict key without modality suffix

    Importantly, box metrics are NOT preferred or used for task=pose.
    """
    metric = metric.lower().strip()
    task = task.lower().strip()

    attr_lookup = {
        "map5095": "map",
        "map50": "map50",
        "precision": "p",
        "recall": "r",
    }
    attr_name = attr_lookup[metric]

    # Attribute-based extraction first.
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

    # Dict fallback.
    keys = pose_metric_dict_keys(metric) if task == "pose" else box_metric_dict_keys(metric)
    score = _first_present(results_dict, keys)
    if score is not None:
        return score

    raise RuntimeError(
        f"Could not extract {task} metric '{metric}'. "
        f"Available results_dict keys: {sorted(list(results_dict.keys()))}"
    )


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="One or more YOLO .pt checkpoints to prune")
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--task", choices=["detect", "pose", "segment", "classify"], default="pose")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Fine-tune epochs for every run. If omitted, use per-target defaults: 90%%->60, 80%%->80, 70%%->80.",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of the training split to use for both FastNAS search and fine-tuning. "
            "For example, --train_fraction 0.2 uses 20%% of the train split per epoch. "
            "Validation remains on the full val split."
        ),
    )

    parser.add_argument("--flops", nargs="+", default=["90%", "80%", "70%"], help="FastNAS FLOPs targets")
    parser.add_argument("--max_iter_data_loader", type=int, default=20, help="FastNAS search budget")
    parser.add_argument(
        "--search_ckpt",
        default="modelopt_fastnas_search_checkpoint.pth",
        help="FastNAS search checkpoint filename. Relative paths are resolved inside each run directory.",
    )

    parser.add_argument("--name_prefix", default="pruned_pose", help="Readable run name prefix")
    parser.add_argument("--project", default=None, help="Ultralytics project dir (optional)")
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0 or cuda:0")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset caching")
    parser.add_argument("--plots", action="store_true", help="Enable Ultralytics plots")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics logging")
    parser.add_argument("--force", action="store_true", help="Rerun even if final artifacts already exist")
    parser.add_argument(
        "--resume_incomplete",
        action="store_true",
        help="Resume incomplete runs from saved FastNAS search or fine-tuning checkpoints when available.",
    )
    parser.add_argument(
        "--resume_save_period",
        type=int,
        default=1,
        help="Save a resumable fine-tuning checkpoint every N epochs when --resume_incomplete is enabled.",
    )

    parser.add_argument(
        "--metric",
        choices=["map5095", "map50", "precision", "recall"],
        default="map5095",
        help="Primary validation metric. For this project, keep this at pose mAP50-95.",
    )

    args = parser.parse_args()
    if not (0.0 < float(args.train_fraction) <= 1.0):
        parser.error("--train_fraction must be in the range (0, 1].")
    if int(args.resume_save_period) < 1:
        parser.error("--resume_save_period must be >= 1.")
    return args


# ---------- main pruning logic ----------


def run_one_model(model_path: str, args: argparse.Namespace, flops_targets: Sequence[str]) -> None:
    model_path = str(Path(model_path))
    src_model_tag = model_run_tag(model_path)

    for flops_target in flops_targets:
        run_name = f"{args.name_prefix}_{src_model_tag}_flops{safe_tag(flops_target)}{fraction_suffix(args.train_fraction)}"
        run_dir = expected_run_dir(args, run_name)
        weights_dir = run_dir / "weights"
        epochs_this_run = resolve_epochs_for_target(flops_target, args)
        search_ckpt_path = resolve_search_checkpoint_path(run_dir, args.search_ckpt)
        resume_last_path, resume_best_path = resume_checkpoint_paths(run_dir)
        load_model_path = model_path
        resume_training = False
        finalize_only = False
        finalize_base_pt: Optional[Path] = None

        if run_is_complete(run_dir) and not args.force:
            print(f"[INFO] Skipping completed run: {run_name}")
            continue

        if run_dir.exists():
            if args.force:
                cleanup_stale_run_outputs(run_dir, search_ckpt_path)
            elif args.resume_incomplete and resume_last_path.exists():
                upgraded_resume_paths = []
                for resume_ckpt in [resume_last_path, resume_best_path]:
                    if resume_ckpt.exists() and ensure_resume_checkpoint_compat(resume_ckpt):
                        upgraded_resume_paths.append(resume_ckpt.name)
                if upgraded_resume_paths:
                    print(
                        "[INFO] Upgraded legacy resume checkpoint(s) for "
                        f"{run_name}: {', '.join(upgraded_resume_paths)}"
                    )
                resume_epoch = checkpoint_epoch(resume_last_path)
                if resume_epoch is not None and (resume_epoch + 1) >= int(epochs_this_run):
                    finalize_base_pt = weights_dir / "last.pt" if (weights_dir / "last.pt").exists() else resume_last_path
                    finalize_only = True
                    print(
                        f"[INFO] Training already reached {resume_epoch + 1} epoch(s) for {run_name}; "
                        "finalizing artifacts without another resume."
                    )
                else:
                    print(f"[INFO] Resuming fine-tuning from checkpoint: {resume_last_path}")
                    load_model_path = str(resume_last_path)
                    resume_training = True
            elif args.resume_incomplete and search_ckpt_path.exists():
                print(f"[INFO] Resuming FastNAS search from checkpoint: {search_ckpt_path}")
                cleanup_stale_run_outputs(run_dir, search_ckpt_path, preserve_search_ckpt=True)
            else:
                cleanup_stale_run_outputs(run_dir, search_ckpt_path)

        if finalize_only:
            try:
                base_pt = finalize_base_pt if finalize_base_pt is not None else resume_last_path
                last_sd = load_state_dict_from_ultralytics_pt(base_pt)
                best_source = resume_best_path if resume_best_path.exists() else base_pt
                best_sd = load_state_dict_from_ultralytics_pt(best_source)

                finalize_run_artifacts(
                    run_name=run_name,
                    model_path=model_path,
                    task=args.task,
                    flops_target=flops_target,
                    metric=args.metric,
                    epochs=epochs_this_run,
                    train_fraction=args.train_fraction,
                    out_dir=run_dir,
                    base_pt=base_pt,
                    last_sd=last_sd,
                    best_sd=best_sd,
                    resume_last_path=resume_last_path,
                    resume_best_path=resume_best_path,
                    search_ckpt_path=search_ckpt_path,
                )
                print(f"[INFO] Completed run: {run_name}")
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                status = "skipped" if "Unachievable FLOPs target" in reason else "failed"

                print(f"[WARN] {status.upper()} run {run_name}: {reason}")
                print(traceback.format_exc())

                if status == "skipped" or not args.resume_incomplete:
                    try:
                        if search_ckpt_path.exists():
                            search_ckpt_path.unlink()
                    except Exception:
                        pass

                write_manifest(
                    (run_dir / "weights" / "artifacts_manifest.txt"),
                    status=status,
                    run_name=run_name,
                    model_path=model_path,
                    task=args.task,
                    flops_target=flops_target,
                    metric=args.metric,
                    epochs=epochs_this_run,
                    train_fraction=args.train_fraction,
                    reason=reason,
                )
            continue

        y = YOLO(load_model_path)
        base_task = args.task or y.task
        BaseTrainer = y.task_map[base_task]["trainer"]

        class PrunedPoseTrainer(BaseTrainer):
            """
            Custom trainer closely following the attached workflow:
              - prune in _setup_train()
              - keep best weights in RAM
              - avoid Ultralytics save checkpoints during training
              - force at least one validation per epoch

            The brittle part for pose is metric extraction, so validate() is explicitly pose-aware.
            """

            def __init__(self, *trainer_args, **trainer_kwargs):
                super().__init__(*trainer_args, **trainer_kwargs)
                target_epochs = int(epochs_this_run)
                if int(getattr(self.args, "epochs", 0) or 0) != target_epochs:
                    self.args.epochs = target_epochs
                if int(getattr(self, "epochs", 0) or 0) != target_epochs:
                    self.epochs = target_epochs
                self._best_score_custom: float = float("-inf")
                self._best_sd_custom: Optional[Dict[str, torch.Tensor]] = None
                self._best_epoch_custom: Optional[int] = None
                self._last_val_epoch: int = -1
                self._logged_metric_keys_once = False
                self._search_ckpt_path: Optional[Path] = None
                self._resume_last_ckpt_path: Optional[Path] = None
                self._resume_best_ckpt_path: Optional[Path] = None
                self._setup_model_ckpt: Optional[Dict[str, Any]] = None

            def setup_model(self):
                if self.resume and is_resume_checkpoint_path(self.model):
                    self.model, ckpt = load_checkpoint_model_direct(Path(self.model))
                    self._setup_model_ckpt = ckpt
                    return ckpt

                ckpt = super().setup_model()
                self._setup_model_ckpt = ckpt
                return ckpt

            def _restore_custom_resume_state(self) -> None:
                ckpt = self._setup_model_ckpt if isinstance(self._setup_model_ckpt, dict) else {}
                if not ckpt:
                    return

                best_score = ckpt.get("custom_best_score", None)
                if best_score is not None:
                    try:
                        self._best_score_custom = float(best_score)
                    except Exception:
                        pass

                best_epoch = ckpt.get("custom_best_epoch", None)
                if best_epoch is not None:
                    try:
                        self._best_epoch_custom = int(best_epoch)
                    except Exception:
                        pass

                best_epoch_text = ckpt.get("custom_best_epoch_text", None)
                if best_epoch_text:
                    best_epoch_path = Path(self.save_dir) / "best_epoch.txt"
                    if not best_epoch_path.exists():
                        try:
                            best_epoch_path.write_text(str(best_epoch_text))
                        except Exception:
                            pass

            def _serializable_resume_model(self, model_obj: Optional[torch.nn.Module] = None):
                if model_obj is None:
                    model_obj = self.ema.ema if getattr(self, "ema", None) is not None else self.model
                return clone_serializable_model(model_obj)

            def _build_resume_checkpoint(self, *, include_optimizer: bool) -> Dict[str, Any]:
                from copy import deepcopy
                from datetime import datetime

                from ultralytics import __version__
                from ultralytics.utils.torch_utils import convert_optimizer_state_dict_to_fp16

                metrics_dict = self.metrics if isinstance(self.metrics, dict) else {}
                try:
                    results_dict = self.read_results_csv()
                except Exception:
                    results_dict = {}

                best_epoch_path = Path(self.save_dir) / "best_epoch.txt"
                best_epoch_text = best_epoch_path.read_text() if best_epoch_path.exists() else ""

                resume_model = self._serializable_resume_model(self.model)
                resume_ema = self._serializable_resume_model(
                    self.ema.ema if getattr(self, "ema", None) is not None else self.model
                )

                ckpt = {
                    "epoch": int(getattr(self, "epoch", -1)),
                    "best_fitness": self.best_fitness,
                    "model": resume_model,
                    "ema": resume_ema,
                    "updates": getattr(self.ema, "updates", 0) if getattr(self, "ema", None) is not None else 0,
                    "optimizer": None,
                    "train_args": vars(self.args),
                    "train_metrics": {**metrics_dict, **{"fitness": self.fitness}},
                    "train_results": results_dict,
                    "date": datetime.now().isoformat(),
                    "version": __version__,
                    "license": "AGPL-3.0 (https://ultralytics.com/license)",
                    "docs": "https://docs.ultralytics.com",
                    "custom_best_score": self._best_score_custom,
                    "custom_best_epoch": self._best_epoch_custom,
                    "custom_best_epoch_text": best_epoch_text,
                }
                if include_optimizer:
                    ckpt["optimizer"] = convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict()))
                return ckpt

            def _write_resume_checkpoint(self, path: Path, *, include_optimizer: bool) -> None:
                self._write_loadable_checkpoint(path, include_optimizer=include_optimizer)

            def _write_loadable_checkpoint(self, path: Path, *, include_optimizer: bool) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._build_resume_checkpoint(include_optimizer=include_optimizer), path)

            def save_model(self):
                from ultralytics.utils import LOGGER

                epoch_num = int(getattr(self, "epoch", -1)) + 1
                if epoch_num < 1:
                    return

                is_period_epoch = (epoch_num % args.resume_save_period) == 0
                is_final_epoch = epoch_num >= int(self.epochs)

                if is_final_epoch:
                    try:
                        self._write_loadable_checkpoint(self.last, include_optimizer=True)
                    except Exception as exc:
                        LOGGER.warning(f"⚠️ Failed to save final loadable checkpoint at epoch {epoch_num}: {exc}")

                if not args.resume_incomplete:
                    return

                if not (is_period_epoch or is_final_epoch):
                    return

                if self._resume_last_ckpt_path is None:
                    return

                try:
                    self._write_resume_checkpoint(self._resume_last_ckpt_path, include_optimizer=True)
                except Exception as exc:
                    LOGGER.warning(f"⚠️ Failed to save resumable checkpoint at epoch {epoch_num}: {exc}")

            def _setup_train(self):
                from ultralytics.utils import LOGGER
                from ultralytics.utils.torch_utils import ModelEMA

                super()._setup_train()

                self.save_dir = Path(self.save_dir)
                self.save_dir.mkdir(parents=True, exist_ok=True)
                weights_dir_local = self.save_dir / "weights"
                weights_dir_local.mkdir(parents=True, exist_ok=True)

                for standard_ckpt in (Path(self.last), Path(self.best)):
                    try:
                        if standard_ckpt.exists():
                            standard_ckpt.unlink()
                    except Exception:
                        pass

                self._resume_last_ckpt_path = weights_dir_local / RESUME_LAST_CKPT_NAME
                self._resume_best_ckpt_path = weights_dir_local / RESUME_BEST_CKPT_NAME

                search_ckpt_path = Path(args.search_ckpt)
                if not search_ckpt_path.is_absolute():
                    search_ckpt_path = self.save_dir / search_ckpt_path.name
                self._search_ckpt_path = search_ckpt_path

                if self.resume:
                    self._restore_custom_resume_state()
                    LOGGER.info(
                        f"🔁 Resumed fine-tuning for {run_name} from "
                        f"{self._resume_last_ckpt_path if self._resume_last_ckpt_path else 'checkpoint'}"
                    )
                    return

                dummy = torch.randn(1, 3, args.imgsz, args.imgsz).to(self.device)

                def collect_func(batch):
                    return self.preprocess_batch(batch)["img"]

                def score_func(model_for_search):
                    """
                    FastNAS search score.
                    For pose pruning, use pose mAP50-95 whenever it is available.
                    Only fall back to generic fitness when pose extraction is impossible.
                    """
                    model_for_search.eval()
                    self.validator.args.save = False
                    self.validator.args.plots = False
                    self.validator.args.verbose = False

                    out = self.validator(model=model_for_search)
                    metrics_obj, results_dict, fitness = parse_validation_output(out, trainer=self)

                    try:
                        return extract_metric_score(metrics_obj, results_dict, args.metric, args.task)
                    except Exception as exc:
                        if fitness is not None:
                            LOGGER.warning(
                                f"⚠️ FastNAS score_func fell back to fitness for {run_name}: {exc}"
                            )
                            return float(fitness)
                        raise

                self.model.is_fused = always_true

                try:
                    prune_result = mtp.prune(
                        model=self.model,
                        mode="fastnas",
                        constraints={"flops": flops_target},
                        dummy_input=dummy,
                        config={
                            "score_func": score_func,
                            "data_loader": self.train_loader,
                            "collect_func": collect_func,
                            "max_iter_data_loader": args.max_iter_data_loader,
                            "checkpoint": str(search_ckpt_path),
                        },
                    )
                    self.model, prune_info = normalize_modelopt_prune_result(prune_result)
                except ValueError as exc:
                    msg = str(exc)
                    if "NOT all constraints can be satisfied" in msg or "cannot be satisfied" in msg:
                        raise RuntimeError(
                            f"Unachievable FLOPs target for run={run_name}, target={flops_target}. {msg}"
                        ) from exc
                    raise
                finally:
                    clear_instance_override(self.model, "is_fused")

                self.model = finalize_pruned_model(self.model)

                self.model.to(self.device)
                self.ema = ModelEMA(self.model)

                # Rebuild optimizer and scheduler because the model graph changed after pruning.
                weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
                try:
                    train_dataset = self.train_loader.dataset
                    train_dataset_len = len(train_dataset)
                except Exception:
                    train_dataset_len = None

                if train_dataset_len is not None:
                    LOGGER.info(
                        f"🧩 Train fraction active: fraction={args.train_fraction:.4f} | "
                        f"train_samples_seen_per_epoch={train_dataset_len}"
                    )
                iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
                self.optimizer = self.build_optimizer(
                    model=self.model,
                    name=self.args.optimizer,
                    lr=self.args.lr0,
                    momentum=self.args.momentum,
                    decay=weight_decay,
                    iterations=iterations,
                )
                self._setup_scheduler()
                LOGGER.info(f"✅ Pruning applied | target FLOPs={flops_target} | info={prune_info}")
                LOGGER.info("📦 Exported pruned subnet to a regular Ultralytics model for checkpointing.")

                if args.resume_incomplete and self._resume_last_ckpt_path is not None:
                    try:
                        self._write_resume_checkpoint(self._resume_last_ckpt_path, include_optimizer=True)
                    except Exception as exc:
                        LOGGER.warning(f"⚠️ Failed to save post-prune resumable checkpoint: {exc}")

            def validate(self):
                from ultralytics.utils import LOGGER

                out = super().validate()
                epoch_i = int(getattr(self, "epoch", -1))
                self._last_val_epoch = epoch_i

                metrics_obj, results_dict, fitness = parse_validation_output(out, trainer=self)

                if not self._logged_metric_keys_once:
                    self._logged_metric_keys_once = True
                    LOGGER.info(
                        f"🔎 validate() types | out={type(out)} | metrics_obj={type(metrics_obj)} | "
                        f"results_keys={sorted(list(results_dict.keys()))}"
                    )

                score: Optional[float] = None

                try:
                    score = extract_metric_score(metrics_obj, results_dict, args.metric, args.task)
                except Exception as exc_attr_dict:
                    csv_score = score_from_results_csv(
                        Path(self.save_dir),
                        csv_metric_keys(args.task, args.metric),
                        epoch_i,
                    )
                    if csv_score is not None:
                        score = csv_score
                        LOGGER.info(
                            f"📄 Using results.csv fallback at epoch {epoch_i}: "
                            f"metric={args.metric} score={score:.6f}"
                        )
                    elif fitness is not None:
                        # This is intentionally last-resort only for pose.
                        score = float(fitness)
                        LOGGER.warning(
                            f"⚠️ Using fitness fallback at epoch {epoch_i} for {run_name}: {exc_attr_dict}"
                        )
                    else:
                        LOGGER.warning(
                            f"⚠️ Could not extract pose score at epoch {epoch_i} for {run_name}: {exc_attr_dict}. "
                            f"No best update this epoch."
                        )
                        return out

                LOGGER.info(
                    f"📏 val @ epoch {epoch_i}: {args.metric}={score:.6f} "
                    f"(best={self._best_score_custom:.6f})"
                )

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
                    LOGGER.info(f"🏁 BEST UPDATE @ epoch {epoch_i}: {previous:.6f} -> {score:.6f}")

                    if args.resume_incomplete and self._resume_best_ckpt_path is not None:
                        try:
                            self._write_resume_checkpoint(self._resume_best_ckpt_path, include_optimizer=False)
                        except Exception as exc:
                            LOGGER.warning(
                                f"⚠️ Failed to save best resumable checkpoint at epoch {epoch_i}: {exc}"
                            )

                return out

            def on_train_epoch_end(self):
                """
                Guarantee at least one validation per epoch.
                This mirrors the attached script's safeguard against missing a best-model update.
                """
                super().on_train_epoch_end()
                epoch_i = int(getattr(self, "epoch", -1))
                if self._last_val_epoch != epoch_i:
                    self.validate()

        train_kwargs = dict(
            task=base_task,
            data=args.data,
            epochs=epochs_this_run,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            cache=args.cache,
            plots=args.plots,
            verbose=args.verbose,
            fraction=args.train_fraction,
            name=run_name,
            trainer=PrunedPoseTrainer,
            exist_ok=True,
            save=args.resume_incomplete,
            save_period=0,
        )
        if args.project:
            train_kwargs["project"] = args.project
        if args.device:
            train_kwargs["device"] = args.device
        if resume_training:
            train_kwargs["resume"] = True

        try:
            y.train(**train_kwargs)

            trainer = y.trainer
            out_dir = Path(trainer.save_dir)
            weights_dir = out_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)
            base_pt = weights_dir / "last.pt"
            if not base_pt.exists():
                export_model = getattr(getattr(trainer, "ema", None), "ema", None) or trainer.model
                y.model = clone_serializable_model(export_model, half=False)
                clear_instance_override(y.model, "is_fused")
                save_yolo_checkpoint(y, base_pt)

            last_sd = get_ema_state_dict(trainer)
            best_sd = getattr(trainer, "_best_sd_custom", None)
            if best_sd is None and args.resume_incomplete and resume_best_path.exists():
                try:
                    best_sd = load_state_dict_from_ultralytics_pt(resume_best_path)
                except Exception as exc:
                    print(f"[WARN] Could not load resumable best checkpoint for {run_name}: {exc}")
            finalize_run_artifacts(
                run_name=run_name,
                model_path=model_path,
                task=args.task,
                flops_target=flops_target,
                metric=args.metric,
                epochs=epochs_this_run,
                train_fraction=args.train_fraction,
                out_dir=out_dir,
                base_pt=base_pt,
                last_sd=last_sd,
                best_sd=best_sd,
                resume_last_path=resume_last_path,
                resume_best_path=resume_best_path,
                search_ckpt_path=getattr(trainer, "_search_ckpt_path", None),
                best_score=getattr(trainer, "_best_score_custom", None),
                last_epoch=getattr(trainer, "epoch", None),
            )

            print(f"[INFO] Completed run: {run_name}")

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            status = "skipped" if "Unachievable FLOPs target" in reason else "failed"

            print(f"[WARN] {status.upper()} run {run_name}: {reason}")
            print(traceback.format_exc())

            # Keep resumable artifacts for failed runs when explicitly requested.
            if status == "skipped" or not args.resume_incomplete:
                try:
                    if search_ckpt_path.exists():
                        search_ckpt_path.unlink()
                except Exception:
                    pass

            write_manifest(
                (run_dir / "weights" / "artifacts_manifest.txt"),
                status=status,
                run_name=run_name,
                model_path=model_path,
                task=args.task,
                flops_target=flops_target,
                metric=args.metric,
                epochs=epochs_this_run,
                train_fraction=args.train_fraction,
                reason=reason,
            )
            continue


# ---------- entrypoint ----------


def main() -> None:
    args = parse_args()
    for model in args.models:
        run_one_model(model, args, args.flops)


if __name__ == "__main__":
    main()
