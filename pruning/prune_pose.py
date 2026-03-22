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
    --cache
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
import modelopt.torch.prune as mtp


# ---------- pickling-safe / top-level helpers ----------


def always_true() -> bool:
    """Small top-level helper used to mimic the attached script's fused-model workaround."""
    return True


def safe_tag(text: str) -> str:
    return text.replace("%", "p").replace("/", "_").replace("-", "_").replace(" ", "_")


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
      90% -> 40
      80% -> 60
      70% -> 80
    Any other target falls back to 60.
    """
    if args.epochs is not None:
        return int(args.epochs)

    pct = parse_target_percent(flops_target)
    if pct == 90:
        return 40
    if pct == 80:
        return 60
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


def cleanup_stale_run_outputs(run_dir: Path, search_ckpt_name: str) -> None:
    """
    Keep the run directory stable for restart-friendliness, but remove stale files that can
    confuse later validation/manifest generation on rerun.
    """
    weights_dir = run_dir / "weights"
    if weights_dir.exists():
        shutil.rmtree(weights_dir, ignore_errors=True)

    for path in [
        run_dir / "results.csv",
        run_dir / "best_epoch.txt",
        run_dir / search_ckpt_name,
    ]:
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


def repack_to_ultralytics_pt(base_pt: Path, state_dict: Dict[str, torch.Tensor], out_pt: Path) -> None:
    """
    Rebuild a normal Ultralytics .pt using a saved pruned-architecture base checkpoint.
    This preserves a loadable Ultralytics artifact instead of a raw state_dict-only file.
    """
    y = YOLO(str(base_pt))
    y.model.load_state_dict(state_dict, strict=True)
    y.save(str(out_pt))


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
        help="Fine-tune epochs for every run. If omitted, use per-target defaults: 90%%->40, 80%%->60, 70%%->80.",
    )
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)

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
        "--metric",
        choices=["map5095", "map50", "precision", "recall"],
        default="map5095",
        help="Primary validation metric. For this project, keep this at pose mAP50-95.",
    )
    return parser.parse_args()


# ---------- main pruning logic ----------


def run_one_model(model_path: str, args: argparse.Namespace, flops_targets: Sequence[str]) -> None:
    model_path = str(Path(model_path))
    src_model_tag = model_run_tag(model_path)

    for flops_target in flops_targets:
        run_name = f"{args.name_prefix}_{src_model_tag}_flops{safe_tag(flops_target)}"
        run_dir = expected_run_dir(args, run_name)
        weights_dir = run_dir / "weights"
        epochs_this_run = resolve_epochs_for_target(flops_target, args)

        if run_is_complete(run_dir) and not args.force:
            print(f"[INFO] Skipping completed run: {run_name}")
            continue

        if run_dir.exists():
            cleanup_stale_run_outputs(run_dir, Path(args.search_ckpt).name)

        y = YOLO(model_path)
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
                self._best_score_custom: float = float("-inf")
                self._best_sd_custom: Optional[Dict[str, torch.Tensor]] = None
                self._best_epoch_custom: Optional[int] = None
                self._last_val_epoch: int = -1
                self._logged_metric_keys_once = False
                self._search_ckpt_path: Optional[Path] = None

            def _setup_train(self):
                from ultralytics.utils import LOGGER
                from ultralytics.utils.torch_utils import ModelEMA

                super()._setup_train()

                self.save_dir = Path(self.save_dir)
                self.save_dir.mkdir(parents=True, exist_ok=True)

                search_ckpt_path = Path(args.search_ckpt)
                if not search_ckpt_path.is_absolute():
                    search_ckpt_path = self.save_dir / search_ckpt_path.name
                self._search_ckpt_path = search_ckpt_path

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
                    self.model, prune_info = mtp.prune(
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
                except ValueError as exc:
                    msg = str(exc)
                    if "NOT all constraints can be satisfied" in msg or "cannot be satisfied" in msg:
                        raise RuntimeError(
                            f"Unachievable FLOPs target for run={run_name}, target={flops_target}. {msg}"
                        ) from exc
                    raise

                self.model.to(self.device)
                self.ema = ModelEMA(self.model)

                # Rebuild optimizer and scheduler because the model graph changed after pruning.
                weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
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
            name=run_name,
            trainer=PrunedPoseTrainer,
            exist_ok=True,
            save=False,
            save_period=0,
        )
        if args.project:
            train_kwargs["project"] = args.project
        if args.device:
            train_kwargs["device"] = args.device

        try:
            y.train(**train_kwargs)

            trainer = y.trainer
            out_dir = Path(trainer.save_dir)
            weights_dir = out_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)

            base_last_tmp = weights_dir / "_base_last_tmp.pt"
            y.save(str(base_last_tmp))

            last_sd = get_ema_state_dict(trainer)
            best_sd = getattr(trainer, "_best_sd_custom", None) or last_sd

            repack_to_ultralytics_pt(base_last_tmp, last_sd, weights_dir / "last.pt")
            repack_to_ultralytics_pt(base_last_tmp, best_sd, weights_dir / "best.pt")

            try:
                base_last_tmp.unlink()
            except Exception:
                pass

            search_ckpt_path = getattr(trainer, "_search_ckpt_path", None)
            if search_ckpt_path is not None:
                try:
                    Path(search_ckpt_path).unlink()
                except Exception:
                    pass

            best_score = getattr(trainer, "_best_score_custom", None)
            last_epoch = getattr(trainer, "epoch", None)
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
                task=args.task,
                flops_target=flops_target,
                metric=args.metric,
                epochs=epochs_this_run,
                best_score=best_score,
                best_epoch_text=best_epoch_text,
                last_epoch=last_epoch,
            )

            print(f"[INFO] Completed run: {run_name}")

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            status = "skipped" if "Unachievable FLOPs target" in reason else "failed"

            print(f"[WARN] {status.upper()} run {run_name}: {reason}")
            print(traceback.format_exc())

            # Best-effort cleanup of the temporary search checkpoint for failed/skipped runs.
            search_ckpt_path = run_dir / Path(args.search_ckpt).name
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
