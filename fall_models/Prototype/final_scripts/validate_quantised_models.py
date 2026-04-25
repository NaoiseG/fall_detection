#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import re
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import yaml
except ImportError as e:
    raise SystemExit("Missing dependency: pyyaml. Install with `pip install pyyaml`.") from e

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("Missing dependency: ultralytics. Install with `pip install ultralytics`.") from e


YOLO_MODEL_RE = re.compile(
    r"^(?P<arch>yolo\d+[nslmx]-pose)(?:_(?P<version>[^.]+))?\.(?P<suffix>engine|pt)$",
    re.IGNORECASE,
)
PRIMARY_POSE_MAP_KEY = "metrics/mAP50-95(P)"
DEFAULT_ALPHAPOSE_CFG = "configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
DEFAULT_ALPHAPOSE_DETECTOR_CFG = "detector/yolo/cfg/yolov3-spp.cfg"
DEFAULT_ALPHAPOSE_CHECKPOINT = "pretrained_models/fast_res50_256x192.pth"
DEFAULT_ALPHAPOSE_DETECTOR_WEIGHTS = "detector/yolo/data/yolov3-spp.weights"
DEFAULT_VITPOSE_DETECTOR_MODEL = "PekingU/rtdetr_r50vd_coco_o365"
DEFAULT_VITPOSE_POSE_MODEL = "usyd-community/vitpose-base"
ARCHITECTURE_ORDER = ("yolo", "alphapose", "vitpose")
ARCHITECTURE_ALIASES = {
    "all": "all",
    "yolo": "yolo",
    "alphapose": "alphapose",
    "alpha": "alphapose",
    "alpha-pose": "alphapose",
    "vitpose": "vitpose",
    "vit": "vitpose",
    "vit-pose": "vitpose",
}
OUTPUT_ARCH_LABELS = {
    "yolo": "yolo",
    "alphapose": "alpha",
    "vitpose": "vit",
}
COCO_IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
ALPHAPOSE_VARIANTS = (
    ("fp32-fp32", "fp32", "fp32"),
    ("fp16-fp16", "fp16", "fp16"),
    ("int8-fp16", "int8", "fp16"),
)
VITPOSE_VARIANTS = (
    ("fp32-fp32", "fp32", "fp32"),
    ("fp16-fp16", "fp16", "fp16"),
)
ARCHITECTURE_SORT_ORDER = {
    "yolo": 0,
    "alphapose": 1,
    "vitpose": 2,
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def prototype_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_models_root() -> Path:
    return repo_root_from_script() / "pose_models" / "quantised"


def default_data_yaml() -> Path:
    return repo_root_from_script() / "pruning" / "coco-pose.yaml"


def default_project_dir() -> Path:
    return repo_root_from_script() / "runs" / "engine_pose_val"


def default_alphapose_root() -> Path:
    return prototype_root_from_script() / "pose_models" / "AlphaPose"


def default_base_yolo_root() -> Path:
    return prototype_root_from_script() / "pose_models" / "ultralytics"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def format_float(value: Any, decimals: int = 4) -> str:
    numeric = safe_float(value)
    return "" if numeric is None else f"{numeric:.{decimals}f}"


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return slug or "run"


def normalize_architectures(values: Sequence[str] | None) -> list[str]:
    if not values:
        return list(ARCHITECTURE_ORDER)

    normalized: list[str] = []
    for value in values:
        key = ARCHITECTURE_ALIASES.get(str(value).strip().lower())
        if key is None:
            choices = ", ".join(sorted(ARCHITECTURE_ALIASES))
            raise ValueError(f"Unsupported architecture '{value}'. Choices: {choices}")
        if key == "all":
            return list(ARCHITECTURE_ORDER)
        if key not in normalized:
            normalized.append(key)
    return normalized


def output_name_for_architectures(architectures: Sequence[str], suffix: str) -> Path:
    tokens = [OUTPUT_ARCH_LABELS[arch] for arch in ARCHITECTURE_ORDER if arch in architectures]
    joined = "_".join(tokens) if tokens else "none"
    return Path(f"quantised_{joined}_map5095{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate quantised pose models on the COCO-Pose validation split. "
            "Supports YOLO TensorRT .engine files plus AlphaPose and ViTPose detector+pose engine pairs."
        )
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=default_models_root(),
        help="Root directory containing quantised pose model folders.",
    )
    parser.add_argument(
        "--base-yolo-root",
        type=Path,
        default=default_base_yolo_root(),
        help="Root directory searched for base Ultralytics YOLO pose .pt models.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=default_data_yaml(),
        help="Path to the COCO-Pose dataset YAML.",
    )
    parser.add_argument(
        "--annotations-json",
        type=Path,
        default=None,
        help="Optional override for the COCO person-keypoints annotation JSON used by AlphaPose/ViTPose evaluation.",
    )
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=list(ARCHITECTURE_ORDER),
        help="Architectures to evaluate. Choices: yolo, alphapose, vitpose, all.",
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
        help="Validation image size for the YOLO path. Included in the result rows for all backends.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Validation batch size for the Ultralytics YOLO path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device argument. '0' maps to CUDA:0 for AlphaPose/ViTPose and is passed through to Ultralytics for YOLO.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Dataloader workers for the YOLO path.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold. Reused as the detector threshold for AlphaPose/ViTPose.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="NMS IoU threshold for YOLO and AlphaPose.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=20,
        help="Max people detections per image for AlphaPose/ViTPose evaluation. COCO keypoint eval uses maxDets=20.",
    )
    parser.add_argument(
        "--vitpose-pose-threshold",
        type=float,
        default=0.001,
        help="Keypoint post-process threshold for ViTPose.",
    )
    parser.add_argument(
        "--alphapose-root",
        type=Path,
        default=default_alphapose_root(),
        help="AlphaPose repository root containing alphapose/ and configs/.",
    )
    parser.add_argument(
        "--alphapose-cfg",
        type=str,
        default=DEFAULT_ALPHAPOSE_CFG,
        help="AlphaPose config relative to --alphapose-root.",
    )
    parser.add_argument(
        "--alphapose-detector-cfg",
        type=str,
        default=DEFAULT_ALPHAPOSE_DETECTOR_CFG,
        help="AlphaPose detector cfg relative to --alphapose-root.",
    )
    parser.add_argument(
        "--alphapose-base-checkpoint",
        type=str,
        default=DEFAULT_ALPHAPOSE_CHECKPOINT,
        help="Base AlphaPose checkpoint path, absolute or relative to --alphapose-root.",
    )
    parser.add_argument(
        "--alphapose-base-detector-weights",
        type=str,
        default=DEFAULT_ALPHAPOSE_DETECTOR_WEIGHTS,
        help="Base AlphaPose detector weights path, absolute or relative to --alphapose-root.",
    )
    parser.add_argument(
        "--vitpose-base-detector-model",
        type=str,
        default=DEFAULT_VITPOSE_DETECTOR_MODEL,
        help="Base ViTPose detector model source.",
    )
    parser.add_argument(
        "--vitpose-base-pose-model",
        type=str,
        default=DEFAULT_VITPOSE_POSE_MODEL,
        help="Base ViTPose pose model source.",
    )
    parser.add_argument(
        "--vitpose-detector-processor",
        type=str,
        default=None,
        help="Optional processor source override for ViTPose RT-DETR detector engines.",
    )
    parser.add_argument(
        "--vitpose-pose-processor",
        type=str,
        default=None,
        help="Optional processor source override for ViTPose pose engines.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Combined CSV summary output path. Defaults to quantised_<models>_map5095.csv.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Combined JSON summary output path. Defaults to quantised_<models>_map5095.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore prior results and re-run every discovered target.",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="Retry targets that previously failed, while still skipping prior successes.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=default_project_dir(),
        help="Project directory for per-target validation artifacts.",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default="",
        help="Optional prefix added to each run directory name.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return {}

    if isinstance(data, dict):
        rows = data.get("results", [])
    elif isinstance(data, list):
        rows = data
    else:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = row.get("engine_key") or row.get("engine_path")
        if key:
            out[str(key)] = row
    return out


def read_csv_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row.get("engine_key") or row.get("engine_path")
            if not key:
                continue
            metrics_json = row.get("metrics_json")
            if metrics_json:
                try:
                    row["metrics"] = json.loads(metrics_json)
                except json.JSONDecodeError:
                    row["metrics"] = {}
            pose_value = row.get("pose_map50_95")
            if pose_value not in (None, ""):
                try:
                    row["pose_map50_95"] = float(pose_value)
                except ValueError:
                    pass
            out[str(key)] = row
    return out


def load_existing_results(csv_path: Path, json_path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    existing.update(read_csv_results(csv_path.expanduser().resolve()))
    existing.update(read_json_results(json_path.expanduser().resolve()))
    return existing


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset YAML is not a mapping: {path}")
    return data


def _looks_like_dataset_root(candidate: Path, data: dict[str, Any]) -> bool:
    if not candidate.exists():
        return False
    marker_names = ["images", "labels", "annotations", "val2017.txt", "train2017.txt", "test-dev2017.txt"]
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


def resolve_split_entry(entry: Any, dataset_root: Path, yaml_dir: Path) -> Any:
    if isinstance(entry, str):
        path = Path(os.path.expanduser(entry))
        if path.is_absolute():
            return str(path)
        return str((dataset_root / path).resolve())
    if isinstance(entry, list):
        return [resolve_split_entry(item, dataset_root, yaml_dir) for item in entry]
    if entry is None:
        return None
    return entry


def create_val_only_yaml(source_yaml: Path, split: str) -> tuple[Path, dict[str, Any]]:
    source_yaml = source_yaml.expanduser().resolve()
    if not source_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")

    original = load_yaml(source_yaml)
    yaml_dir = source_yaml.parent.resolve()
    dataset_root = resolve_dataset_root(source_yaml, original)

    if split not in original and not (split == "val" and "validation" in original):
        raise KeyError(f"Requested split '{split}' is missing from dataset YAML: {source_yaml}")

    split_key = "validation" if split == "val" and "val" not in original and "validation" in original else split
    original_split_entry = original.get(split_key)
    if original_split_entry is None:
        raise ValueError(f"Dataset YAML split '{split_key}' is empty in {source_yaml}")

    resolved_split = resolve_split_entry(original_split_entry, dataset_root, yaml_dir)

    temp_yaml = deepcopy(original)
    temp_yaml["path"] = str(dataset_root)
    temp_yaml["train"] = resolved_split
    temp_yaml["val"] = resolved_split if split == "val" else resolve_split_entry(
        original.get("val", original_split_entry), dataset_root, yaml_dir
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
    with temp_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(temp_yaml, handle, sort_keys=False)

    info = {
        "source_yaml": str(source_yaml),
        "temp_yaml": str(temp_path),
        "dataset_root": str(dataset_root),
        "requested_split": split,
        "split_key": split_key,
        "resolved_split_entry": resolved_split,
    }
    return temp_path, info


def resolve_reference_path(reference: str, *, dataset_root: Path, reference_parent: Path) -> Path:
    raw = Path(os.path.expanduser(str(reference)))
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        (reference_parent / raw).resolve(),
        (dataset_root / raw).resolve(),
        (Path.cwd() / raw).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def list_images_in_directory(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    paths = [path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in COCO_IMAGE_EXTS]
    if not paths:
        raise FileNotFoundError(f"No images found in directory: {directory}")
    return paths


def load_split_image_paths(split_entry: Any, *, dataset_root: Path) -> list[Path]:
    paths: list[Path] = []

    def consume(entry: Any) -> None:
        if entry is None:
            return
        if isinstance(entry, list):
            for item in entry:
                consume(item)
            return
        if not isinstance(entry, str):
            raise TypeError(f"Unsupported split entry type: {type(entry)!r}")

        resolved = Path(os.path.expanduser(entry)).resolve()
        if resolved.is_file():
            if resolved.suffix.lower() == ".txt":
                with resolved.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        paths.append(
                            resolve_reference_path(line, dataset_root=dataset_root, reference_parent=resolved.parent)
                        )
                return
            paths.append(resolved)
            return
        if resolved.is_dir():
            paths.extend(list_images_in_directory(resolved))
            return
        raise FileNotFoundError(f"Resolved split path does not exist: {resolved}")

    consume(split_entry)
    if not paths:
        raise FileNotFoundError("No images resolved from the requested split entry.")
    return paths


def split_name_candidates(split: str, resolved_split_entry: Any) -> list[str]:
    tokens: list[str] = []

    def add(token: str | None) -> None:
        if token is None:
            return
        token = str(token).strip()
        if token and token not in tokens:
            tokens.append(token)

    def consume(entry: Any) -> None:
        if isinstance(entry, list):
            for item in entry:
                consume(item)
            return
        if not isinstance(entry, str):
            return
        path = Path(entry)
        if path.suffix.lower() == ".txt":
            add(path.stem)
        elif path.suffix.lower() in COCO_IMAGE_EXTS:
            add(path.parent.name)
        else:
            add(path.name)

    consume(resolved_split_entry)
    if split == "val":
        add("val2017")
        add("val")
    elif split == "train":
        add("train2017")
        add("train")
    elif split == "test":
        add("test-dev2017")
        add("test2017")
        add("test")
    add(split)
    return tokens


def resolve_annotations_json(
    *,
    source_yaml: Path,
    dataset_root: Path,
    split: str,
    resolved_split_entry: Any,
    override: Path | None,
) -> Path:
    if override is not None:
        resolved = override.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Annotation JSON not found: {resolved}")
        return resolved

    candidate_dirs = []
    for directory in (dataset_root / "annotations", source_yaml.parent / "annotations"):
        if directory.exists() and directory not in candidate_dirs:
            candidate_dirs.append(directory)

    split_tokens = split_name_candidates(split, resolved_split_entry)
    file_candidates = [f"person_keypoints_{token}.json" for token in split_tokens]

    for directory in candidate_dirs:
        for filename in file_candidates:
            candidate = directory / filename
            if candidate.exists():
                return candidate.resolve()

    for directory in candidate_dirs:
        globbed = sorted(directory.glob("person_keypoints*.json"))
        if len(globbed) == 1:
            return globbed[0].resolve()
        for token in split_tokens:
            for candidate in globbed:
                if token in candidate.stem:
                    return candidate.resolve()

    searched = ", ".join(str(path) for path in candidate_dirs) or "<none>"
    names = ", ".join(file_candidates) or "<none>"
    raise FileNotFoundError(
        "Could not resolve the COCO person-keypoints annotation JSON. "
        f"Searched for [{names}] under: {searched}"
    )


def build_dataset_context(
    *,
    source_yaml: Path,
    split: str,
    annotations_json_override: Path | None,
) -> dict[str, Any]:
    source_yaml = source_yaml.expanduser().resolve()
    if not source_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")

    original = load_yaml(source_yaml)
    yaml_dir = source_yaml.parent.resolve()
    dataset_root = resolve_dataset_root(source_yaml, original)

    if split not in original and not (split == "val" and "validation" in original):
        raise KeyError(f"Requested split '{split}' is missing from dataset YAML: {source_yaml}")

    split_key = "validation" if split == "val" and "val" not in original and "validation" in original else split
    raw_split_entry = original.get(split_key)
    if raw_split_entry is None:
        raise ValueError(f"Dataset YAML split '{split_key}' is empty in {source_yaml}")

    resolved_split_entry = resolve_split_entry(raw_split_entry, dataset_root, yaml_dir)
    image_paths = load_split_image_paths(resolved_split_entry, dataset_root=dataset_root)
    annotations_json = resolve_annotations_json(
        source_yaml=source_yaml,
        dataset_root=dataset_root,
        split=split,
        resolved_split_entry=resolved_split_entry,
        override=annotations_json_override,
    )

    return {
        "source_yaml": str(source_yaml),
        "dataset_root": str(dataset_root),
        "requested_split": split,
        "split_key": split_key,
        "resolved_split_entry": resolved_split_entry,
        "annotations_json": str(annotations_json),
        "num_images": len(image_paths),
        "image_paths": [str(path) for path in image_paths],
    }


def infer_yolo_metadata(model_path: Path) -> dict[str, Any]:
    match = YOLO_MODEL_RE.match(model_path.name)
    if match:
        architecture = str(match.group("arch"))
        version = str(match.group("version") or ("base" if match.group("suffix").lower() == "pt" else "engine"))
    else:
        stem_parts = model_path.stem.split("_")
        architecture = stem_parts[0] if stem_parts else model_path.parent.name
        version = stem_parts[1] if len(stem_parts) > 1 else "unknown"
    return {
        "backend": "yolo",
        "architecture": architecture,
        "version": version,
        "engine_name": model_path.name,
        "engine_path": normalize_path(model_path),
        "engine_key": normalize_path(model_path),
        "detector_engine_path": None,
        "pose_engine_path": None,
    }


def discover_yolo_targets(models_root: Path) -> list[dict[str, Any]]:
    root = models_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Models root does not exist: {root}")
    targets = []
    for path in sorted(root.rglob("*.engine")):
        if not path.is_file():
            continue
        if not YOLO_MODEL_RE.match(path.name):
            continue
        targets.append(infer_yolo_metadata(path))
    return targets


def discover_base_yolo_targets(base_yolo_root: Path) -> list[dict[str, Any]]:
    root = base_yolo_root.expanduser().resolve()
    if not root.exists():
        return []
    targets = []
    for path in sorted(root.rglob("*.pt")):
        if path.is_file() and YOLO_MODEL_RE.match(path.name):
            targets.append(infer_yolo_metadata(path))
    return targets


def resolve_optional_local_model_source(source: str, *, root: Path) -> str:
    raw = Path(os.path.expanduser(str(source)))
    if raw.is_absolute():
        return normalize_path(raw)
    candidate = (root / raw).expanduser().resolve()
    if candidate.exists():
        return normalize_path(candidate)
    return str(source)


def build_alphapose_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    alpha_root = args.models_root.expanduser().resolve() / "alphapose"
    base_detector_path = resolve_optional_local_model_source(
        args.alphapose_base_detector_weights,
        root=args.alphapose_root,
    )
    base_pose_path = resolve_optional_local_model_source(
        args.alphapose_base_checkpoint,
        root=args.alphapose_root,
    )
    targets: list[dict[str, Any]] = [
        {
            "backend": "alphapose",
            "architecture": "alphapose",
            "version": "base",
            "engine_name": "alphapose_base",
            "engine_path": f"detector={base_detector_path} ; pose={base_pose_path}",
            "engine_key": f"alphapose::base::{base_detector_path}::{base_pose_path}",
            "detector_engine_path": base_detector_path,
            "pose_engine_path": base_pose_path,
        }
    ]
    for version, detector_precision, pose_precision in ALPHAPOSE_VARIANTS:
        detector_path = alpha_root / f"yolov3_spp_{detector_precision}.engine"
        pose_path = alpha_root / f"fastpose_{pose_precision}.engine"
        targets.append(
            {
                "backend": "alphapose",
                "architecture": "alphapose",
                "version": version,
                "engine_name": f"alphapose_{version}",
                "engine_path": f"detector={normalize_path(detector_path)} ; pose={normalize_path(pose_path)}",
                "engine_key": f"alphapose::{version}::{normalize_path(detector_path)}::{normalize_path(pose_path)}",
                "detector_engine_path": normalize_path(detector_path),
                "pose_engine_path": normalize_path(pose_path),
            }
        )
    return targets


def build_vitpose_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    vit_root = args.models_root.expanduser().resolve() / "vitpose_trt" / "engines"
    base_detector_model = str(args.vitpose_base_detector_model)
    base_pose_model = str(args.vitpose_base_pose_model)
    targets: list[dict[str, Any]] = [
        {
            "backend": "vitpose",
            "architecture": "vitpose",
            "version": "base",
            "engine_name": "vitpose_base",
            "engine_path": f"detector={base_detector_model} ; pose={base_pose_model}",
            "engine_key": f"vitpose::base::{base_detector_model}::{base_pose_model}",
            "detector_engine_path": base_detector_model,
            "pose_engine_path": base_pose_model,
        }
    ]
    for version, detector_precision, pose_precision in VITPOSE_VARIANTS:
        detector_path = vit_root / f"detector_pekingu_rtdetr_r50vd_coco_o365_{detector_precision}.engine"
        pose_path = vit_root / f"pose_usyd_community_vitpose_base_{pose_precision}.engine"
        targets.append(
            {
                "backend": "vitpose",
                "architecture": "vitpose",
                "version": version,
                "engine_name": f"vitpose_{version}",
                "engine_path": f"detector={normalize_path(detector_path)} ; pose={normalize_path(pose_path)}",
                "engine_key": f"vitpose::{version}::{normalize_path(detector_path)}::{normalize_path(pose_path)}",
                "detector_engine_path": normalize_path(detector_path),
                "pose_engine_path": normalize_path(pose_path),
            }
        )
    return targets


def build_requested_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for architecture in ARCHITECTURE_ORDER:
        if architecture not in args.architectures:
            continue
        if architecture == "yolo":
            targets.extend(discover_base_yolo_targets(args.base_yolo_root))
            targets.extend(discover_yolo_targets(args.models_root))
        elif architecture == "alphapose":
            targets.extend(build_alphapose_targets(args))
        elif architecture == "vitpose":
            targets.extend(build_vitpose_targets(args))
    return targets


def normalize_runtime_device(device: str) -> str:
    raw = str(device).strip()
    lowered = raw.lower()
    if lowered in {"cpu", "mps"}:
        return lowered
    if lowered.startswith("cuda"):
        return lowered
    if raw.isdigit():
        return f"cuda:{raw}"
    return raw


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


class JsonArrayWriter:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.handle = None
        self.count = 0
        self._first = True

    def __enter__(self) -> "JsonArrayWriter":
        ensure_parent_dir(self.path)
        self.handle = self.path.open("w", encoding="utf-8")
        self.handle.write("[\n")
        return self

    def write(self, item: dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("JsonArrayWriter is not open.")
        if not self._first:
            self.handle.write(",\n")
        json.dump(item, self.handle, ensure_ascii=False)
        self._first = False
        self.count += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.write("\n]\n")
            self.handle.close()
            self.handle = None


def runtime_memory_summary() -> str:
    parts: list[str] = []

    status_path = Path("/proc/self/status")
    if status_path.exists():
        try:
            metrics: dict[str, str] = {}
            with status_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    if key in {"VmRSS", "VmHWM", "VmSwap"}:
                        metrics[key] = value.strip()
            for key in ("VmRSS", "VmHWM", "VmSwap"):
                if key in metrics:
                    parts.append(f"{key}={metrics[key]}")
        except Exception:
            pass

    try:
        import torch

        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                used_mb = (int(total_bytes) - int(free_bytes)) / (1024 * 1024)
                total_mb = int(total_bytes) / (1024 * 1024)
                parts.append(f"cuda_used={used_mb:.0f}MiB/{total_mb:.0f}MiB")
            except Exception:
                pass
    except Exception:
        pass

    return ", ".join(parts) if parts else "memory=n/a"


def maybe_release_runtime_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def evaluate_yolo_target(
    *,
    target: dict[str, Any],
    data_yaml: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    model = YOLO(str(target["engine_path"]))
    run_name = f"{args.name_prefix}{Path(target['engine_path']).stem}"
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
    save_dir = getattr(val_output, "save_dir", None)
    metrics = sanitize_metrics(extract_metrics_dict(val_output))
    return metrics, (str(save_dir) if save_dir is not None else "")


def ensure_prototype_root_on_path() -> None:
    prototype_root = prototype_root_from_script()
    prototype_root_str = str(prototype_root)
    if prototype_root_str not in sys.path:
        sys.path.insert(0, prototype_root_str)


def create_alphapose_runner(target: dict[str, Any], args: argparse.Namespace):
    ensure_prototype_root_on_path()
    from dataset_helpers.pose_alphapose import AlphaPoseExportConfig, AlphaPoseRunner

    detector_path = Path(str(target["detector_engine_path"]))
    pose_path = Path(str(target["pose_engine_path"]))
    missing = [str(path) for path in (detector_path, pose_path) if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing AlphaPose engine file(s): {missing_str}")

    cfg = AlphaPoseExportConfig(
        alphapose_root=str(args.alphapose_root.expanduser().resolve()),
        cfg_path=str(args.alphapose_cfg),
        checkpoint=str(pose_path),
        detector_cfg=str(args.alphapose_detector_cfg),
        detector_weights=str(detector_path),
        conf_thres=float(args.conf),
        nms_thres=float(args.iou),
        max_people=max(1, int(args.max_det)),
        render_video=False,
        save_csv=False,
        device=normalize_runtime_device(args.device),
    )
    return AlphaPoseRunner(cfg)


def create_vitpose_runner(target: dict[str, Any], args: argparse.Namespace):
    ensure_prototype_root_on_path()
    from dataset_helpers.get_keypoints_files_ViTpose import VitPoseExportConfig, VitPoseRunner

    detector_source = str(target["detector_engine_path"])
    pose_source = str(target["pose_engine_path"])
    detector_path = Path(detector_source)
    pose_path = Path(pose_source)
    missing = [
        str(path)
        for path in (detector_path, pose_path)
        if path.suffix.lower() == ".engine" and not path.exists()
    ]
    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(f"Missing ViTPose engine file(s): {missing_str}")

    cfg = VitPoseExportConfig(
        detector_model=detector_source,
        detector_processor=args.vitpose_detector_processor,
        pose_model=pose_source,
        pose_processor=args.vitpose_pose_processor,
        person_threshold=float(args.conf),
        pose_threshold=float(args.vitpose_pose_threshold),
        detector_max_det=max(1, int(args.max_det)),
        max_people=max(1, int(args.max_det)),
        render_video=False,
        save_csv=False,
        device=normalize_runtime_device(args.device),
    )
    return VitPoseRunner(cfg)


def load_coco_apis():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycocotools is required for AlphaPose/ViTPose COCO evaluation. "
            "Install with `pip install pycocotools>=2.0.6`."
        ) from exc
    return COCO, COCOeval


def build_coco_image_id_lookup(coco_gt: Any) -> tuple[dict[str, int], set[int]]:
    file_name_to_id: dict[str, int] = {}
    valid_ids: set[int] = set()
    for image in coco_gt.dataset.get("images", []):
        image_id = int(image["id"])
        file_name = str(image.get("file_name", ""))
        valid_ids.add(image_id)
        if file_name:
            file_name_to_id[file_name] = image_id
            file_name_to_id[Path(file_name).name] = image_id
    return file_name_to_id, valid_ids


def resolve_image_id(image_path: Path, *, file_name_to_id: dict[str, int], valid_ids: set[int]) -> int:
    stem = image_path.stem
    if stem.isdigit():
        numeric = int(stem)
        if numeric in valid_ids:
            return numeric

    candidates = [image_path.name, "/".join(image_path.parts[-2:]), image_path.as_posix()]
    for candidate in candidates:
        image_id = file_name_to_id.get(candidate)
        if image_id is not None:
            return image_id

    raise KeyError(f"Could not resolve COCO image id for {image_path}")


def xyxy_to_xywh(box_xyxy: Sequence[float] | None) -> list[float] | None:
    if box_xyxy is None:
        return None
    values = list(box_xyxy)
    if len(values) < 4:
        return None
    x1, y1, x2, y2 = (safe_float(value) for value in values[:4])
    if None in (x1, y1, x2, y2):
        return None
    width = max(0.0, float(x2) - float(x1))
    height = max(0.0, float(y2) - float(y1))
    return [float(x1), float(y1), width, height]


def keypoints_to_coco_format(xy: Any, conf: Any) -> list[float]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - ultralytics environments should already include numpy
        raise ModuleNotFoundError("numpy is required for AlphaPose/ViTPose evaluation.") from exc

    xy_arr = np.asarray(xy, dtype=np.float32)
    conf_arr = np.asarray(conf, dtype=np.float32).reshape(-1)
    if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
        raise ValueError(f"Invalid keypoint array shape: {tuple(xy_arr.shape)}")

    flat: list[float] = []
    count = int(xy_arr.shape[0])
    for idx in range(count):
        x = float(xy_arr[idx, 0])
        y = float(xy_arr[idx, 1])
        score = float(conf_arr[idx]) if idx < int(conf_arr.shape[0]) else 0.0
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(score) and score > 0.0):
            flat.extend([0.0, 0.0, 0.0])
            continue
        flat.extend([x, y, max(0.0, score)])
    return flat


def fallback_box_from_keypoints(xy: Any, conf: Any) -> list[float] | None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ModuleNotFoundError("numpy is required for AlphaPose/ViTPose evaluation.") from exc

    xy_arr = np.asarray(xy, dtype=np.float32)
    conf_arr = np.asarray(conf, dtype=np.float32).reshape(-1)
    if conf_arr.shape[0] < xy_arr.shape[0]:
        padded = np.zeros((xy_arr.shape[0],), dtype=np.float32)
        padded[: conf_arr.shape[0]] = conf_arr
        conf_arr = padded
    if xy_arr.ndim != 2 or xy_arr.shape[1] != 2:
        return None
    valid = (
        np.isfinite(xy_arr[:, 0])
        & np.isfinite(xy_arr[:, 1])
        & np.isfinite(conf_arr[: xy_arr.shape[0]])
        & (conf_arr[: xy_arr.shape[0]] > 0.0)
    )
    if not np.any(valid):
        return None
    valid_xy = xy_arr[valid]
    x1 = float(np.min(valid_xy[:, 0]))
    y1 = float(np.min(valid_xy[:, 1]))
    x2 = float(np.max(valid_xy[:, 0]))
    y2 = float(np.max(valid_xy[:, 1]))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def select_instance_score(person: dict[str, Any]) -> float:
    person_conf = safe_float(person.get("person_conf"))
    box_conf = safe_float(person.get("box_conf"))

    mean_kpt_conf: float | None = None
    try:
        import numpy as np

        conf_arr = np.asarray(person.get("kpts_conf"), dtype=np.float32).reshape(-1)
        valid = np.isfinite(conf_arr) & (conf_arr > 0.0)
        if np.any(valid):
            mean_kpt_conf = float(np.mean(conf_arr[valid]))
    except Exception:
        mean_kpt_conf = None

    for candidate in (
        person_conf,
        (box_conf * mean_kpt_conf) if box_conf is not None and mean_kpt_conf is not None else None,
        box_conf,
        mean_kpt_conf,
        0.0,
    ):
        if candidate is not None:
            return max(0.0, float(candidate))
    return 0.0


def person_to_coco_prediction(person: dict[str, Any], *, image_id: int) -> dict[str, Any] | None:
    xy = person.get("kpts_xy")
    conf = person.get("kpts_conf")
    if xy is None or conf is None:
        return None

    keypoints = keypoints_to_coco_format(xy, conf)
    bbox = xyxy_to_xywh(person.get("box_xyxy"))
    if bbox is None:
        bbox = fallback_box_from_keypoints(xy, conf)

    prediction = {
        "image_id": int(image_id),
        "category_id": 1,
        "keypoints": keypoints,
        "score": round(select_instance_score(person), 5),
    }
    if bbox is not None:
        prediction["bbox"] = [round(value, 3) for value in bbox]
    return prediction


def evaluate_predictions_with_coco(
    *,
    prediction_count: int,
    image_ids: Sequence[int],
    annotations_json: Path,
    pred_json_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    COCO, COCOeval = load_coco_apis()
    coco_gt = COCO(str(annotations_json))

    if prediction_count <= 0:
        summary_path.write_text("No predictions were produced.\n", encoding="utf-8")
        return {
            PRIMARY_POSE_MAP_KEY: 0.0,
            "metrics/mAP50(P)": 0.0,
            "counts/predictions": 0,
            "counts/images_evaluated": len(image_ids),
        }

    coco_pred = coco_gt.loadRes(str(pred_json_path))
    coco_eval = COCOeval(coco_gt, coco_pred, "keypoints")
    coco_eval.params.imgIds = list(image_ids)
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    summary_text = buffer.getvalue()
    summary_path.write_text(summary_text, encoding="utf-8")

    stats = getattr(coco_eval, "stats", None)
    map50_95 = float(stats[0]) if stats is not None and len(stats) > 0 else 0.0
    map50 = float(stats[1]) if stats is not None and len(stats) > 1 else 0.0
    return {
        PRIMARY_POSE_MAP_KEY: map50_95,
        "metrics/mAP50(P)": map50,
        "counts/predictions": int(prediction_count),
        "counts/images_evaluated": len(image_ids),
    }


def evaluate_backend_target(
    *,
    target: dict[str, Any],
    dataset_context: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    try:
        import cv2
    except ImportError as exc:
        raise ModuleNotFoundError("opencv-python is required for AlphaPose/ViTPose evaluation.") from exc

    annotations_json = Path(str(dataset_context["annotations_json"])).resolve()
    image_paths = [Path(path).resolve() for path in dataset_context["image_paths"]]
    run_name = f"{args.name_prefix}{slugify(target['engine_name'])}"
    run_dir = args.project / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    pred_json_path = run_dir / "predictions.json"
    summary_path = run_dir / "cocoeval_summary.txt"

    COCO, _ = load_coco_apis()
    coco_gt = COCO(str(annotations_json))
    file_name_to_id, valid_ids = build_coco_image_id_lookup(coco_gt)
    image_records = [
        (resolve_image_id(path, file_name_to_id=file_name_to_id, valid_ids=valid_ids), path) for path in image_paths
    ]

    backend = str(target["backend"])
    if backend == "alphapose":
        runner = create_alphapose_runner(target, args)
    elif backend == "vitpose":
        runner = create_vitpose_runner(target, args)
    else:
        raise ValueError(f"Unsupported backend for manual COCO evaluation: {backend}")

    total = len(image_records)
    keep_people = max(1, int(args.max_det))
    cleanup_interval = 25
    memory_report_interval = 200
    with JsonArrayWriter(pred_json_path) as prediction_writer:
        for index, (image_id, image_path) in enumerate(image_records, start=1):
            if index == 1 or index % 50 == 0 or index == total:
                line = f"  progress: {index}/{total} images"
                if index == 1 or index % memory_report_interval == 0 or index == total:
                    line = f"{line} | {runtime_memory_summary()}"
                print(line)

            frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"Failed to read image: {image_path}")

            frame_rgb = None
            infer_out = None
            if backend == "alphapose":
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                infer_out = runner.infer(frame_rgb, image_name=image_path.name)
                people = list(infer_out.get("people", []))
            else:
                people = list(runner.infer(frame_bgr))

            if people:
                people.sort(key=select_instance_score, reverse=True)
                people = people[:keep_people]

            for person in people:
                prediction = person_to_coco_prediction(person, image_id=image_id)
                if prediction is not None:
                    prediction_writer.write(prediction)

            del people
            del infer_out
            del frame_rgb
            del frame_bgr
            if index % cleanup_interval == 0:
                maybe_release_runtime_memory()

    metrics = evaluate_predictions_with_coco(
        prediction_count=prediction_writer.count,
        image_ids=[image_id for image_id, _ in image_records],
        annotations_json=annotations_json,
        pred_json_path=pred_json_path,
        summary_path=summary_path,
    )
    metrics["annotations_json"] = str(annotations_json)
    metrics["predictions_json"] = str(pred_json_path)
    metrics["cocoeval_summary_txt"] = str(summary_path)
    return sanitize_metrics(metrics), str(run_dir)


def base_result_row(target: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        **target,
        "status": "failed",
        "pose_map50_95": None,
        "metrics": {},
        "error": None,
        "run_at": now_iso(),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "split": args.split,
        "run_dir": None,
    }


def evaluate_target(
    *,
    target: dict[str, Any],
    data_yaml: Path,
    dataset_context: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    result = base_result_row(target, args)
    try:
        backend = str(target["backend"])
        if backend == "yolo":
            metrics, run_dir = evaluate_yolo_target(target=target, data_yaml=data_yaml, args=args)
        else:
            metrics, run_dir = evaluate_backend_target(target=target, dataset_context=dataset_context, args=args)
        pose_map = extract_pose_map50_95(metrics)
        result["status"] = "success"
        result["pose_map50_95"] = pose_map
        result["metrics"] = metrics
        result["run_dir"] = run_dir
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback_tail"] = traceback.format_exc(limit=8)
    return result


def merge_results(existing: dict[str, dict[str, Any]], new_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged = dict(existing)
    for row in new_rows:
        merged[str(row["engine_key"])] = row
    return merged


def result_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    backend = str(row.get("backend", ""))
    return (
        ARCHITECTURE_SORT_ORDER.get(backend, 99),
        str(row.get("architecture", "")),
        str(row.get("version", "")),
        str(row.get("engine_name", "")),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    fieldnames = [
        "backend",
        "architecture",
        "version",
        "engine_name",
        "engine_path",
        "engine_key",
        "detector_engine_path",
        "pose_engine_path",
        "status",
        "pose_map50_95",
        "split",
        "imgsz",
        "batch",
        "device",
        "run_at",
        "run_dir",
        "error",
        "metrics_json",
    ]
    with path.expanduser().resolve().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            out["metrics_json"] = json.dumps(row.get("metrics", {}), ensure_ascii=False, sort_keys=True)
            writer.writerow(out)


def write_json(
    path: Path,
    rows: list[dict[str, Any]],
    dataset_context: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    ensure_parent_dir(path)
    payload = {
        "generated_at": now_iso(),
        "primary_metric": PRIMARY_POSE_MAP_KEY,
        "architectures": list(args.architectures),
        "models_root": str(args.models_root),
        "dataset": {
            key: value
            for key, value in dataset_context.items()
            if key not in {"image_paths"}
        },
        "results": rows,
    }
    with path.expanduser().resolve().open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def print_table(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}")
    if not rows:
        print("(none)")
        return

    columns = [
        ("backend", "backend"),
        ("architecture", "architecture"),
        ("version", "version"),
        ("status", "status"),
        ("pose_map50_95", "pose_map50_95"),
    ]

    display_rows: list[dict[str, str]] = []
    for row in rows:
        display_rows.append(
            {
                "backend": str(row.get("backend", "")),
                "architecture": str(row.get("architecture", "")),
                "version": str(row.get("version", "")),
                "status": str(row.get("status", "")),
                "pose_map50_95": format_float(row.get("pose_map50_95")),
            }
        )

    widths = {}
    for header, key in columns:
        widths[key] = max(len(header), *(len(display_row[key]) for display_row in display_rows))

    header_line = " | ".join(header.ljust(widths[key]) for header, key in columns)
    sep_line = "-+-".join("-" * widths[key] for _, key in columns)
    print(header_line)
    print(sep_line)
    for row in display_rows:
        print(" | ".join(row[key].ljust(widths[key]) for _, key in columns))


def main() -> int:
    args = parse_args()
    args.architectures = normalize_architectures(args.architectures)
    args.models_root = args.models_root.expanduser().resolve()
    args.base_yolo_root = args.base_yolo_root.expanduser().resolve()
    args.data_yaml = args.data_yaml.expanduser().resolve()
    args.project = args.project.expanduser().resolve()
    args.alphapose_root = args.alphapose_root.expanduser().resolve()
    args.annotations_json = args.annotations_json.expanduser().resolve() if args.annotations_json else None
    args.csv_out = (
        args.csv_out.expanduser().resolve()
        if args.csv_out is not None
        else output_name_for_architectures(args.architectures, ".csv").expanduser().resolve()
    )
    args.json_out = (
        args.json_out.expanduser().resolve()
        if args.json_out is not None
        else output_name_for_architectures(args.architectures, ".json").expanduser().resolve()
    )

    if args.force and args.rerun_failed:
        print("Warning: --force overrides normal resume behavior; all targets will be re-run.")

    temp_yaml = None
    try:
        temp_yaml, val_yaml_info = create_val_only_yaml(args.data_yaml, args.split)
        dataset_context = build_dataset_context(
            source_yaml=args.data_yaml,
            split=args.split,
            annotations_json_override=args.annotations_json,
        )
        dataset_context["temp_yaml"] = str(temp_yaml)

        targets = build_requested_targets(args)
        if not targets:
            print(f"No matching validation targets found under: {args.models_root}")
            return 1

        existing = {} if args.force else load_existing_results(args.csv_out, args.json_out)

        skipped_completed: list[dict[str, Any]] = []
        skipped_prev_failed: list[dict[str, Any]] = []
        to_run: list[dict[str, Any]] = []

        for target in targets:
            prior = existing.get(str(target["engine_key"]))
            if args.force or prior is None:
                to_run.append(target)
                continue

            prior_status = str(prior.get("status", "")).lower()
            if prior_status == "success":
                skipped_completed.append(prior)
            elif prior_status == "failed" and not args.rerun_failed:
                skipped_prev_failed.append(prior)
            else:
                to_run.append(target)

        print(f"Architectures: {', '.join(args.architectures)}")
        print(f"Discovered {len(targets)} target(s)")
        print(f"Using temporary YOLO validation YAML: {temp_yaml}")
        print(f"Resolved dataset root: {dataset_context.get('dataset_root')}")
        print(f"Resolved {args.split} entry: {dataset_context.get('resolved_split_entry')}")
        print(f"Resolved annotations JSON: {dataset_context.get('annotations_json')}")
        print(
            f"Will run {len(to_run)} target(s), "
            f"skip {len(skipped_completed)} completed, "
            f"hold {len(skipped_prev_failed)} prior failures."
        )

        run_results: list[dict[str, Any]] = []
        for index, target in enumerate(to_run, start=1):
            print(f"\n[{index}/{len(to_run)}] Validating {target['engine_name']} ({target['backend']} / {target['version']})")
            row = evaluate_target(
                target=target,
                data_yaml=temp_yaml,
                dataset_context=dataset_context,
                args=args,
            )
            run_results.append(row)
            if row["status"] == "success":
                print(f"  success: pose mAP50-95 = {format_float(row['pose_map50_95'])}")
            else:
                print(f"  failed: {row.get('error')}")

        merged = merge_results(existing, run_results)
        final_rows = sorted(merged.values(), key=result_sort_key)
        write_csv(args.csv_out, final_rows)
        write_json(args.json_out, final_rows, dataset_context, args)

        new_successes = [row for row in run_results if row.get("status") == "success"]
        new_failures = [row for row in run_results if row.get("status") == "failed"]

        print_table(final_rows, "Combined summary")

        print("\nRun summary")
        print(f"- newly completed successes: {len(new_successes)}")
        print(f"- skipped already-completed targets: {len(skipped_completed)}")
        print(f"- failures from this run: {len(new_failures)}")
        print(f"- previously failed but not retried: {len(skipped_prev_failed)}")
        print(f"- CSV saved to: {args.csv_out}")
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
