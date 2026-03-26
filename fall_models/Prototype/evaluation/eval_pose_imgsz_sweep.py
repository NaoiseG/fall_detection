#!/usr/bin/env python3
"""Sweep YOLO11 pose validation over multiple image sizes on COCO-pose.

This script forces square validation inputs via ``rect=False`` so the sweep
isolates the speed/accuracy tradeoff from reducing model input pixels alone.
The local Ultralytics fork automatically enables COCO JSON evaluation on the
official val split, so pose mAP comes from the standard COCO validation path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

DEFAULT_DATA = "/home/people/21376026/fall_detection/pruning/coco-pose.yaml"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def default_project_dir(repo_root: Path) -> Path:
    return repo_root / "fall_models" / "Prototype" / "eval_outputs" / "pose_imgsz_sweep"


def unique_preserve_order(values: Sequence[int]) -> List[int]:
    seen = set()
    unique_values: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return slug or "run"


def import_local_yolo(fork_root: Path):
    fork_root = fork_root.resolve()
    existing = sys.modules.get("ultralytics")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        existing_path = Path(existing_file).resolve() if existing_file else None
        if existing_path is None or fork_root not in existing_path.parents:
            for module_name in list(sys.modules):
                if module_name == "ultralytics" or module_name.startswith("ultralytics."):
                    sys.modules.pop(module_name, None)

    fork_root_str = str(fork_root)
    if fork_root_str in sys.path:
        sys.path.remove(fork_root_str)
    sys.path.insert(0, fork_root_str)
    importlib.invalidate_caches()

    ultralytics = importlib.import_module("ultralytics")
    imported_root = Path(ultralytics.__file__).resolve().parents[1]
    if imported_root != fork_root:
        raise RuntimeError(
            f"Expected local Ultralytics fork at {fork_root}, but imported {imported_root} instead."
        )
    return ultralytics.YOLO


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    fork_root = repo_root / "pruning" / "yolov11-prune"

    parser = argparse.ArgumentParser(
        description="Evaluate multiple YOLO11 pose models on COCO-pose while sweeping imgsz."
    )
    parser.add_argument("--models", nargs="+", required=True, help="One or more model weight paths.")
    parser.add_argument("--imgsz", nargs="+", type=int, required=True, help="One or more square input sizes.")
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"Dataset YAML path (default: {DEFAULT_DATA}).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Validation device passed to Ultralytics (default: 0).",
    )
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size (default: 16).")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers (default: 8).")
    parser.add_argument("--half", action="store_true", help="Enable FP16 validation.")
    parser.add_argument(
        "--project",
        default=str(default_project_dir(repo_root)),
        help="Output directory for per-run Ultralytics validation artifacts.",
    )
    parser.add_argument("--name", default="pose_imgsz_sweep", help="Base run name.")
    parser.add_argument("--split", default="val", help="Dataset split to validate (default: val).")
    parser.add_argument(
        "--save-csv",
        dest="save_csv",
        default=None,
        help="Path to the summary CSV. Defaults to <project>/<name>_summary.csv.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose Ultralytics output.")
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics validation plots.")
    args = parser.parse_args()

    args.repo_root = repo_root
    args.fork_root = fork_root
    args.models = [Path(model).expanduser() for model in args.models]
    args.imgsz = unique_preserve_order(args.imgsz)
    args.data = Path(args.data).expanduser()
    args.project = Path(args.project).expanduser()
    args.save_csv = (
        Path(args.save_csv).expanduser()
        if args.save_csv
        else args.project / f"{slugify(args.name)}_summary.csv"
    )
    args.baseline_imgsz = max(args.imgsz)

    if not args.fork_root.exists():
        parser.error(f"Local Ultralytics fork not found: {args.fork_root}")
    if any(size <= 0 for size in args.imgsz):
        parser.error("All --imgsz values must be positive integers.")
    if args.batch <= 0:
        parser.error("--batch must be a positive integer.")
    if args.workers < 0:
        parser.error("--workers must be zero or greater.")

    return args


def make_run_name(base_name: str, model_path: Path, imgsz: int) -> str:
    model_hash = hashlib.sha1(str(model_path).encode("utf-8")).hexdigest()[:8]
    return f"{slugify(base_name)}_{slugify(model_path.stem)}_{model_hash}_imgsz{imgsz}"


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def pick_metric(results_dict: Dict[str, Any], keys: Sequence[str]) -> Tuple[float | None, str | None]:
    for key in keys:
        if key not in results_dict:
            continue
        value = safe_float(results_dict.get(key))
        if value is not None:
            return value, key
    return None, None


def extract_metrics(metrics: Any) -> Dict[str, Any]:
    results_dict = getattr(metrics, "results_dict", {}) or {}
    if not isinstance(results_dict, dict):
        results_dict = {}
    speed = getattr(metrics, "speed", {}) or {}
    if not isinstance(speed, dict):
        speed = {}

    pose_map50_95, pose_map50_95_key = pick_metric(
        results_dict,
        ("metrics/mAP50-95(P)", "metrics/mAP50-95"),
    )
    pose_map50, pose_map50_key = pick_metric(
        results_dict,
        ("metrics/mAP50(P)", "metrics/mAP50"),
    )
    pose_precision, pose_precision_key = pick_metric(
        results_dict,
        ("metrics/precision(P)", "metrics/precision"),
    )
    pose_recall, pose_recall_key = pick_metric(
        results_dict,
        ("metrics/recall(P)", "metrics/recall"),
    )

    return {
        "pose_map50_95": pose_map50_95,
        "pose_map50_95_key_used": pose_map50_95_key,
        "pose_map50": pose_map50,
        "pose_map50_key_used": pose_map50_key,
        "pose_precision": pose_precision,
        "pose_precision_key_used": pose_precision_key,
        "pose_recall": pose_recall,
        "pose_recall_key_used": pose_recall_key,
        "preprocess_ms": safe_float(speed.get("preprocess")),
        "inference_ms": safe_float(speed.get("inference")),
        "postprocess_ms": safe_float(speed.get("postprocess")),
    }


def base_row(args: argparse.Namespace, model_path: Path, imgsz: int) -> Dict[str, Any]:
    ratio = float(imgsz**2) / float(args.baseline_imgsz**2)
    return {
        "status": "ok",
        "task": "pose",
        "model_path": str(model_path),
        "model_name": model_path.name,
        "imgsz": imgsz,
        "baseline_imgsz": args.baseline_imgsz,
        "pixel_ratio_vs_baseline": ratio,
        "est_flop_ratio_vs_baseline": ratio,
        "delta_map50_95_vs_baseline": None,
        "baseline_pose_map50_95": None,
        "baseline_inference_ms": None,
        "inference_ratio_vs_baseline": None,
        "inference_reduction_vs_baseline": None,
        "data": str(args.data),
        "split": args.split,
        "device": str(args.device),
        "batch": args.batch,
        "workers": args.workers,
        "half": args.half,
        "project": str(args.project),
        "run_name": make_run_name(args.name, model_path, imgsz),
        "run_dir": None,
        "pose_map50_95": None,
        "pose_map50": None,
        "pose_precision": None,
        "pose_recall": None,
        "preprocess_ms": None,
        "inference_ms": None,
        "postprocess_ms": None,
        "pose_map50_95_key_used": None,
        "pose_map50_key_used": None,
        "pose_precision_key_used": None,
        "pose_recall_key_used": None,
        "error": None,
    }


def run_single_eval(YOLO, args: argparse.Namespace, model_path: Path, imgsz: int) -> Dict[str, Any]:
    row = base_row(args, model_path, imgsz)
    if not model_path.exists():
        row["status"] = "failed"
        row["error"] = f"FileNotFoundError: weights not found: {model_path}"
        return row

    try:
        model = YOLO(str(model_path), task="pose")
        metrics = model.val(
            data=str(args.data),
            task="pose",
            imgsz=imgsz,
            rect=False,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            half=args.half,
            split=args.split,
            plots=args.plots,
            verbose=args.verbose,
            project=str(args.project),
            name=row["run_name"],
        )
        row.update(extract_metrics(metrics))
        save_dir = getattr(metrics, "save_dir", None)
        row["run_dir"] = str(save_dir) if save_dir is not None else None
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        if args.verbose:
            traceback.print_exc()

    return row


def apply_baseline_deltas(rows: List[Dict[str, Any]], baseline_imgsz: int) -> None:
    baseline_map_by_model: Dict[str, float] = {}
    baseline_inference_by_model: Dict[str, float] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        if row["imgsz"] != baseline_imgsz:
            continue
        if row["pose_map50_95"] is not None:
            baseline_map_by_model[row["model_path"]] = row["pose_map50_95"]
        if row["inference_ms"] is not None and row["inference_ms"] > 0:
            baseline_inference_by_model[row["model_path"]] = row["inference_ms"]

    for row in rows:
        baseline_map_value = baseline_map_by_model.get(row["model_path"])
        row["baseline_pose_map50_95"] = baseline_map_value
        if baseline_map_value is None or row["pose_map50_95"] is None:
            row["delta_map50_95_vs_baseline"] = None
        else:
            row["delta_map50_95_vs_baseline"] = row["pose_map50_95"] - baseline_map_value

        baseline_inference_value = baseline_inference_by_model.get(row["model_path"])
        row["baseline_inference_ms"] = baseline_inference_value
        if baseline_inference_value is None or row["inference_ms"] is None:
            row["inference_ratio_vs_baseline"] = None
            row["inference_reduction_vs_baseline"] = None
        else:
            ratio = row["inference_ms"] / baseline_inference_value
            row["inference_ratio_vs_baseline"] = ratio
            row["inference_reduction_vs_baseline"] = 1.0 - ratio


def sort_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["model_path"], -int(row["imgsz"])))


def csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_csv(rows: Sequence[Dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "status",
        "task",
        "model_name",
        "model_path",
        "imgsz",
        "baseline_imgsz",
        "pixel_ratio_vs_baseline",
        "est_flop_ratio_vs_baseline",
        "pose_map50_95",
        "pose_map50_95_key_used",
        "baseline_pose_map50_95",
        "delta_map50_95_vs_baseline",
        "pose_map50",
        "pose_map50_key_used",
        "pose_precision",
        "pose_precision_key_used",
        "pose_recall",
        "pose_recall_key_used",
        "preprocess_ms",
        "inference_ms",
        "baseline_inference_ms",
        "inference_ratio_vs_baseline",
        "inference_reduction_vs_baseline",
        "postprocess_ms",
        "data",
        "split",
        "device",
        "batch",
        "workers",
        "half",
        "project",
        "run_name",
        "run_dir",
        "error",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sort_rows(rows):
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def format_float(value: float | None, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return ""
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return fmt.format(value)


def print_summary_table(rows: Sequence[Dict[str, Any]]) -> None:
    headers = [
        "model",
        "imgsz",
        "mAP50-95(P)",
        "mAP50(P)",
        "P",
        "R",
        "infer_ms",
        "infer_ratio",
        "infer_red",
        "pix_ratio",
        "delta",
        "status",
    ]
    display_rows: List[List[str]] = []
    for row in sort_rows(rows):
        display_rows.append(
            [
                row["model_name"],
                str(row["imgsz"]),
                format_float(row["pose_map50_95"], digits=4),
                format_float(row["pose_map50"], digits=4),
                format_float(row["pose_precision"], digits=4),
                format_float(row["pose_recall"], digits=4),
                format_float(row["inference_ms"], digits=2),
                format_float(row["inference_ratio_vs_baseline"], digits=3),
                format_float(row["inference_reduction_vs_baseline"], digits=3, signed=True),
                format_float(row["pixel_ratio_vs_baseline"], digits=3),
                format_float(row["delta_map50_95_vs_baseline"], digits=4, signed=True),
                row["status"],
            ]
        )

    widths = [len(header) for header in headers]
    for row in display_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    print("Summary")
    print("  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("  ".join("-" * widths[idx] for idx in range(len(headers))))
    for row in display_rows:
        print("  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))

    failures = [row for row in sort_rows(rows) if row["status"] != "ok"]
    if failures:
        print("\nFailures")
        for row in failures:
            print(f"- {row['model_path']} imgsz={row['imgsz']}: {row['error']}")


def main() -> int:
    args = parse_args()
    YOLO = import_local_yolo(args.fork_root)

    rows: List[Dict[str, Any]] = []
    for model_path in args.models:
        for imgsz in args.imgsz:
            print(f"[val] model={model_path} imgsz={imgsz}")
            rows.append(run_single_eval(YOLO, args, model_path, imgsz))

    apply_baseline_deltas(rows, args.baseline_imgsz)
    write_csv(rows, args.save_csv)
    print_summary_table(rows)
    print(f"\nSaved CSV: {args.save_csv}")

    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
