#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from e

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("Missing dependency: ultralytics. Install with `pip install ultralytics`.") from e


PRIMARY_POSE_MAP_KEY = "metrics/mAP50-95(P)"
EXPECTED_PRECISIONS = ("fp32", "fp16", "int8")
DEFAULT_MODELS_ROOT = Path("/home/people/21376026/scratch/pose_models/prune_models/full_pruned")
DEFAULT_DATA_YAML = Path("/home/people/21376026/fall_detection/pruning/coco-pose.yaml")
DEFAULT_OUTPUT_DIR = Path("/home/people/21376026/scratch/final_results/pose_models_map5095")
DEFAULT_JSON_OUT = DEFAULT_OUTPUT_DIR / "pruned_quantised_map5095.json"
DEFAULT_PROJECT = DEFAULT_OUTPUT_DIR / "runs"

PRUNED_ENGINE_RE = re.compile(
    r"^(?P<architecture>yolo\d+[nslmx])_pruned_(?P<prune_percent>\d+)_"
    r"(?P<precision>fp32|fp16|int8)\.engine$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate pruned TensorRT YOLO pose .engine models on the COCO-Pose validation split "
            "and save pose mAP50-95 results to JSON."
        )
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help=f"Root directory to recursively search for pruned .engine files (default: {DEFAULT_MODELS_ROOT}).",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help=f"Path to COCO-Pose dataset YAML (default: {DEFAULT_DATA_YAML}).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_JSON_OUT,
        help=f"Combined JSON output path (default: {DEFAULT_JSON_OUT}).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to validate on. Defaults to val.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Validation image size. Must be compatible with the TensorRT engine profile.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Validation batch size. Default 1 is conservative for TensorRT validation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Ultralytics device argument. Defaults to GPU 0.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers. Default 0 is conservative on shared/HPC nodes.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold for validation.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold for NMS during validation.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT,
        help=f"Ultralytics project directory for per-engine validation logs (default: {DEFAULT_PROJECT}).",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="pruned_",
        help="Optional prefix added to each Ultralytics run name.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore prior JSON results and re-run every discovered engine.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Retry engines that previously failed, while still skipping prior successes.",
    )
    parser.add_argument(
        "--allow-unexpected-engines",
        action="store_true",
        help="Also validate .engine files that do not match the expected *_fp32/fp16/int8.engine naming pattern.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def stable_engine_key(engine_path: Path) -> str:
    return normalize_path(engine_path)


def infer_engine_metadata(engine_path: Path) -> dict[str, Any]:
    name = engine_path.name
    match = PRUNED_ENGINE_RE.match(name)
    if match:
        architecture = match.group("architecture").lower()
        prune_percent = int(match.group("prune_percent"))
        precision = match.group("precision").lower()
        model_variant = f"{architecture}_pruned_{prune_percent}"
    else:
        stem_parts = engine_path.stem.split("_")
        precision = stem_parts[-1].lower() if stem_parts else "unknown"
        architecture = stem_parts[0].lower() if stem_parts else "unknown"
        prune_percent = None
        if len(stem_parts) >= 3 and stem_parts[1].lower() == "pruned":
            try:
                prune_percent = int(stem_parts[2])
            except ValueError:
                prune_percent = None
        model_variant = "_".join(stem_parts[:-1]) if len(stem_parts) > 1 else engine_path.stem

    weights_dir = engine_path.parent
    model_dir = weights_dir.parent if weights_dir.name == "weights" else weights_dir
    relative_model_dir = model_dir.name

    return {
        "architecture": architecture,
        "model_variant": model_variant,
        "prune_percent": prune_percent,
        "precision": precision,
        "engine_name": name,
        "engine_path": normalize_path(engine_path),
        "engine_key": stable_engine_key(engine_path),
        "model_dir": normalize_path(model_dir),
        "model_dir_name": relative_model_dir,
        "weights_dir": normalize_path(weights_dir),
    }


def is_expected_engine(path: Path) -> bool:
    return bool(PRUNED_ENGINE_RE.match(path.name))


def discover_engines(models_root: Path, allow_unexpected_engines: bool = False) -> list[Path]:
    root = models_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Models root does not exist: {root}")

    engines = []
    for path in sorted(root.rglob("*.engine")):
        if not path.is_file():
            continue
        if allow_unexpected_engines or is_expected_engine(path):
            engines.append(path)
    return engines


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML is not a mapping: {path}")
    return data


def _looks_like_dataset_root(candidate: Path, data: dict[str, Any]) -> bool:
    if not candidate.exists():
        return False
    marker_names = ["images", "labels", "val2017.txt", "train2017.txt", "test-dev2017.txt"]
    if any((candidate / name).exists() for name in marker_names):
        return True
    for key in ("train", "val", "test", "validation"):
        entry = data.get(key)
        if isinstance(entry, str) and (candidate / entry).exists():
            return True
    return False


def resolve_dataset_root(yaml_path: Path, data: dict[str, Any]) -> Path:
    yaml_dir = yaml_path.parent.resolve()
    path_value = data.get("path")
    if path_value in (None, ""):
        return yaml_dir

    raw = Path(os.path.expanduser(str(path_value)))
    if raw.is_absolute():
        return raw

    nested = (yaml_dir / raw).resolve()
    if nested.exists():
        return nested

    if yaml_dir.name == raw.name or _looks_like_dataset_root(yaml_dir, data):
        return yaml_dir

    return nested


def resolve_split_entry(entry: Any, dataset_root: Path) -> Any:
    if isinstance(entry, str):
        p = Path(os.path.expanduser(entry))
        if p.is_absolute():
            return str(p)
        return str((dataset_root / p).resolve())
    if isinstance(entry, list):
        return [resolve_split_entry(x, dataset_root) for x in entry]
    if entry is None:
        return None
    return entry


def create_val_only_yaml(source_yaml: Path, split: str) -> tuple[Path, dict[str, Any]]:
    source_yaml = source_yaml.expanduser().resolve()
    if not source_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")

    original = load_yaml(source_yaml)
    dataset_root = resolve_dataset_root(source_yaml, original)

    if split not in original and not (split == "val" and "validation" in original):
        raise KeyError(f"Requested split '{split}' is missing from dataset YAML: {source_yaml}")

    split_key = "validation" if split == "val" and "val" not in original and "validation" in original else split
    original_split_entry = original.get(split_key)
    if original_split_entry is None:
        raise ValueError(f"Dataset YAML split '{split_key}' is empty in {source_yaml}")

    resolved_split = resolve_split_entry(original_split_entry, dataset_root)

    temp_yaml = deepcopy(original)
    temp_yaml["path"] = str(dataset_root)
    temp_yaml["train"] = resolved_split
    temp_yaml["val"] = resolved_split if split == "val" else resolve_split_entry(
        original.get("val", original_split_entry), dataset_root
    )
    if "validation" in temp_yaml:
        temp_yaml.pop("validation", None)
    if split == "test":
        temp_yaml["test"] = resolved_split
    if split == "train":
        temp_yaml["train"] = resolved_split
        temp_yaml["val"] = resolved_split

    temp_yaml.pop("download", None)

    fd, temp_path_str = tempfile.mkstemp(prefix="coco_pose_val_only_", suffix=".yaml")
    os.close(fd)
    temp_path = Path(temp_path_str)
    with temp_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(temp_yaml, f, sort_keys=False)

    info = {
        "source_yaml": str(source_yaml),
        "temp_yaml": str(temp_path),
        "dataset_root": str(dataset_root),
        "requested_split": split,
        "resolved_split_entry": resolved_split,
    }
    return temp_path, info


def ensure_parent_dir(path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return None


def read_json_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}

    rows = data.get("results", []) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("engine_key") or row.get("engine_path")
        if not key:
            continue
        pose_value = row.get("pose_map50_95")
        if pose_value not in (None, ""):
            row["pose_map50_95"] = safe_float(pose_value)
        out[str(key)] = row
    return out


def extract_metrics_dict(val_result: Any) -> dict[str, Any]:
    if isinstance(val_result, dict):
        return dict(val_result)

    for attr in ("results_dict", "stats"):
        value = getattr(val_result, attr, None)
        if isinstance(value, dict):
            return dict(value)

    metrics = getattr(val_result, "metrics", None)
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        return dict(results_dict)

    extracted: dict[str, Any] = {}
    metrics_obj = metrics if metrics is not None else val_result
    for parent_name, child_name, key in (
        ("pose", "map", PRIMARY_POSE_MAP_KEY),
        ("pose", "map50", "metrics/mAP50(P)"),
        ("pose", "mp", "metrics/precision(P)"),
        ("pose", "mr", "metrics/recall(P)"),
        ("box", "map", "metrics/mAP50-95(B)"),
    ):
        parent = getattr(metrics_obj, parent_name, None)
        if parent is not None and hasattr(parent, child_name):
            extracted[key] = getattr(parent, child_name)
    return extracted


def extract_pose_map50_95(metrics_dict: dict[str, Any]) -> float | None:
    candidate_keys = [
        PRIMARY_POSE_MAP_KEY,
        "pose/mAP50-95",
        "pose_map50_95",
        "pose_map",
    ]
    for key in candidate_keys:
        if key in metrics_dict:
            value = safe_float(metrics_dict[key])
            if value is not None:
                return value
    return None


def sanitize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    clean = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def evaluate_engine(engine_path: Path, data_yaml: Path, args: argparse.Namespace) -> dict[str, Any]:
    meta = infer_engine_metadata(engine_path)
    run_name = f"{args.name_prefix}{meta['model_variant']}_{meta['precision']}"
    result: dict[str, Any] = {
        **meta,
        "status": "failed",
        "pose_map50_95": None,
        "metrics": {},
        "error": None,
        "run_at": now_iso(),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "split": args.split,
    }

    try:
        model = YOLO(str(engine_path))
        val_output = model.val(
            data=str(data_yaml),
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            conf=args.conf,
            iou=args.iou,
            project=str(args.project),
            name=run_name,
            exist_ok=True,
            verbose=False,
            plots=False,
            save=False,
        )
        metrics = sanitize_metrics(extract_metrics_dict(val_output))
        result["status"] = "success"
        result["pose_map50_95"] = extract_pose_map50_95(metrics)
        result["metrics"] = metrics
    except Exception as e:
        result["status"] = "failed"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback_tail"] = traceback.format_exc(limit=8)

    return result


def merge_results(existing: dict[str, dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged = dict(existing)
    for row in new_rows:
        merged[row["engine_key"]] = row
    return merged


def result_sort_key(row: dict[str, Any]) -> tuple:
    return (
        str(row.get("architecture", "")),
        int(row.get("prune_percent") or 0),
        str(row.get("precision", "")),
        str(row.get("engine_path", "")),
    )


def write_json(path: Path, rows: list[dict[str, Any]], dataset_info: dict[str, Any], args: argparse.Namespace) -> None:
    ensure_parent_dir(path)
    payload = {
        "generated_at": now_iso(),
        "primary_metric": PRIMARY_POSE_MAP_KEY,
        "models_root": normalize_path(args.models_root),
        "expected_precisions": list(EXPECTED_PRECISIONS),
        "dataset": dataset_info,
        "validation": {
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "conf": args.conf,
            "iou": args.iou,
        },
        "results": rows,
    }
    with path.expanduser().resolve().open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def format_float(value: Any, decimals: int = 4) -> str:
    v = safe_float(value)
    return "" if v is None else f"{v:.{decimals}f}"


def print_table(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}")
    if not rows:
        print("(none)")
        return

    columns = [
        ("architecture", "architecture"),
        ("prune", "prune_percent"),
        ("precision", "precision"),
        ("status", "status"),
        ("mAP50-95(P)", "pose_map50_95"),
        ("engine", "engine_path"),
    ]

    display_rows: list[dict[str, str]] = []
    for row in rows:
        display_rows.append(
            {
                "architecture": str(row.get("architecture", "")),
                "prune_percent": str(row.get("prune_percent", "")),
                "precision": str(row.get("precision", "")),
                "status": str(row.get("status", "")),
                "pose_map50_95": format_float(row.get("pose_map50_95")),
                "engine_path": str(row.get("engine_path", "")),
            }
        )

    widths = {}
    for header, key in columns:
        widths[key] = max(len(header), *(len(row[key]) for row in display_rows))

    header_line = " | ".join(header.ljust(widths[key]) for header, key in columns)
    sep_line = "-+-".join("-" * widths[key] for _, key in columns)
    print(header_line)
    print(sep_line)
    for row in display_rows:
        print(" | ".join(row[key].ljust(widths[key]) for _, key in columns))


def main() -> int:
    args = parse_args()

    args.models_root = args.models_root.expanduser().resolve()
    args.json_out = args.json_out.expanduser().resolve()
    args.project = args.project.expanduser().resolve()

    if args.force and args.rerun_failed:
        print("Warning: --force overrides normal resume behavior; all engines will be re-run.")

    temp_yaml = None
    dataset_info: dict[str, Any] = {}
    try:
        temp_yaml, dataset_info = create_val_only_yaml(args.data_yaml, args.split)
        engines = discover_engines(args.models_root, args.allow_unexpected_engines)
        if not engines:
            print(f"No expected .engine files found under: {args.models_root}")
            print("Expected filenames like: yolo11l_pruned_90_fp16.engine")
            return 1

        existing = {} if args.force else read_json_results(args.json_out)

        discovered_meta = [infer_engine_metadata(path) for path in engines]
        skipped_completed: list[dict[str, Any]] = []
        skipped_prev_failed: list[dict[str, Any]] = []
        to_run: list[Path] = []

        for engine_path, meta in zip(engines, discovered_meta):
            prior = existing.get(meta["engine_key"])
            if args.force or prior is None:
                to_run.append(engine_path)
                continue

            prior_status = str(prior.get("status", "")).lower()
            if prior_status == "success":
                skipped_completed.append(prior)
            elif prior_status == "failed" and not args.rerun_failed:
                skipped_prev_failed.append(prior)
            else:
                to_run.append(engine_path)

        print(f"Discovered {len(engines)} expected .engine files under {args.models_root}")
        print(f"Using temporary validation-only YAML: {temp_yaml}")
        print(f"Resolved dataset root: {dataset_info.get('dataset_root')}")
        print(f"Resolved {args.split} entry: {dataset_info.get('resolved_split_entry')}")
        print(f"Will run {len(to_run)} engine(s), skip {len(skipped_completed)} completed, hold {len(skipped_prev_failed)} prior failures.")

        run_results: list[dict[str, Any]] = []
        for idx, engine_path in enumerate(to_run, start=1):
            print(f"\n[{idx}/{len(to_run)}] Validating {engine_path}")
            row = evaluate_engine(engine_path=engine_path, data_yaml=temp_yaml, args=args)
            run_results.append(row)

            merged = merge_results(existing, run_results)
            checkpoint_rows = sorted(merged.values(), key=result_sort_key)
            write_json(args.json_out, checkpoint_rows, dataset_info, args)

            if row["status"] == "success":
                print(f"  success: pose mAP50-95 = {format_float(row['pose_map50_95'])}")
            else:
                print(f"  failed: {row.get('error')}")
            print(f"  checkpoint saved to: {args.json_out}")

        merged = merge_results(existing, run_results)
        final_rows = sorted(merged.values(), key=result_sort_key)
        write_json(args.json_out, final_rows, dataset_info, args)

        new_successes = [row for row in run_results if row.get("status") == "success"]
        new_failures = [row for row in run_results if row.get("status") == "failed"]

        print_table(final_rows, "Combined summary")

        print("\nRun summary")
        print(f"- newly completed successes: {len(new_successes)}")
        print(f"- skipped already-completed engines: {len(skipped_completed)}")
        print(f"- failures from this run: {len(new_failures)}")
        print(f"- previously failed but not retried: {len(skipped_prev_failed)}")
        print(f"- JSON saved to: {args.json_out}")

        if new_failures:
            print_table(new_failures, "Failures from this run")
        if skipped_prev_failed:
            print_table(skipped_prev_failed, "Previously failed but not retried")

        return 0 if not new_failures else 2
    finally:
        if temp_yaml is not None:
            try:
                temp_yaml.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
