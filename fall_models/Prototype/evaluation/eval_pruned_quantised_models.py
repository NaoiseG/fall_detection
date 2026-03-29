#!/usr/bin/env python3
"""Resumable UP-Fall evaluation sweep over fully-pruned quantised YOLO variants."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CLASSIFIER_MODELS: Tuple[str, ...] = ("cnnlstm", "stgcn", "MotionBERT")
POSE_MODEL_ORDER: Tuple[str, ...] = ("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x")
KNOWN_PRECISIONS: Tuple[str, ...] = ("base", "fp16", "int8", "fp32")
POSE_MODEL_PATTERN = re.compile(r"^yolo11[nsmlx]$")
PRUNE_VARIANT_PATTERN = re.compile(r"^pruned_(\d+)$")

DEFAULT_KEYPOINTS_ROOT = Path("/home/people/21376026/scratch/keypoints/pruned_keypoints/full_pruned")
DEFAULT_CLASSIFICATION_ROOT = Path("/home/people/21376026/scratch/classification_models")
DEFAULT_OUTPUT_ROOT = Path("/home/people/21376026/scratch/evaluations/pruned_quantised")
DEFAULT_MOTIONBERT_PKL_ROOT = Path("/home/people/21376026/scratch/MotionBERT_pkls")

TEST_SUBJECTS = "16-17"
TRAIN_SUBJECTS = TEST_SUBJECTS
VAL_SUBJECTS = "13-15"
CAMERAS: Tuple[int, ...] = (1, 2)
WINDOW_T = 64
WINDOW_STRIDE = 32
LABEL_MODE = "center"
MASTER_JSON_NAME = "pruned_quantised_metrics.json"

MODEL_ALIASES = {
    "cnnlstm": "cnnlstm",
    "stgcn": "stgcn",
    "motionbert": "MotionBERT",
    "motionbert_action": "MotionBERT",
    "MotionBERT": "MotionBERT",
}

pd = None


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: List[List[List[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_table: List[List[str]] = []
        self._current_row: List[str] = []
        self._current_cell_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._current_cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_row and tag in {"td", "th"} and self._in_cell:
            cell = " ".join(part for part in self._current_cell_parts if part).strip()
            self._current_row.append(cell)
            self._current_cell_parts = []
            self._in_cell = False
        elif self._in_table and tag == "tr" and self._in_row:
            if any(cell.strip() for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = []
            self._in_row = False
        elif tag == "table" and self._in_table:
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = []
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            text = " ".join(data.split())
            if text:
                self._current_cell_parts.append(text)


@dataclass(frozen=True)
class SweepPaths:
    script_path: Path
    eval_dir: Path
    repo_root: Path
    prepare_motionbert_script: Path
    motionbert_code_root: Path
    keypoints_root: Path
    classification_root: Path
    output_root: Path
    motionbert_pkl_root: Path


@dataclass(frozen=True)
class DiscoveredVariant:
    pose_model_size: str
    prune_variant: str
    precision: str
    npz_root: Path

    @property
    def variant_id(self) -> str:
        return f"{self.pose_model_size}__{self.prune_variant}__{self.precision}"


@dataclass(frozen=True)
class RunSpec:
    classifier_model: str
    pose_model_size: str
    prune_variant: str
    precision: str
    npz_root: Path
    checkpoint_path: Path
    run_dir: Path
    stdout_log_path: Path
    stderr_log_path: Path
    run_status_path: Path
    motionbert_repo_root: Optional[Path] = None
    motionbert_pkl_path: Optional[Path] = None
    motionbert_label_map_path: Optional[Path] = None

    @property
    def run_id(self) -> str:
        return (
            f"{self.classifier_model}__{self.pose_model_size}"
            f"__{self.prune_variant}__{self.precision}"
        )


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def quote_command(cmd: Sequence[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def require_pandas():
    global pd
    if pd is not None:
        return pd
    try:
        import pandas as pandas_module
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to parse evaluator outputs. "
            "Install pandas in the active Python environment and retry."
        ) from exc
    pd = pandas_module
    return pd


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_log_tail(path: Path, max_lines: int = 40, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError:
        return ""
    tail = "".join(lines[-max_lines:]).strip()
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable UP-Fall evaluation sweep over fully-pruned quantised YOLO keypoints."
    )
    parser.add_argument("--only-models", nargs="+", default=None)
    parser.add_argument("--only-pose-sizes", nargs="+", default=None)
    parser.add_argument("--only-prune-variants", nargs="+", default=None)
    parser.add_argument("--only-precisions", nargs="+", default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-regenerate-motionbert-pkl", action="store_true")
    parser.add_argument("--keypoints-root", type=Path, default=DEFAULT_KEYPOINTS_ROOT)
    parser.add_argument("--classification-root", type=Path, default=DEFAULT_CLASSIFICATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--motionbert-pkl-root", type=Path, default=DEFAULT_MOTIONBERT_PKL_ROOT)
    parser.add_argument("--motionbert-code-root", type=Path, default=None)
    return parser.parse_args()


def normalize_model_token(token: str) -> str:
    key = str(token).strip()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    lowered = key.lower()
    if lowered in MODEL_ALIASES:
        return MODEL_ALIASES[lowered]
    raise SystemExit(
        f"Unknown model token '{token}'. Expected one of: {', '.join(CLASSIFIER_MODELS)}"
    )


def normalize_filter_tokens(
    values: Optional[Iterable[str]],
    *,
    allowed: Sequence[str],
    kind: str,
    transform=str.lower,
) -> Tuple[str, ...]:
    if values is None:
        return tuple(allowed)
    seen: List[str] = []
    allowed_set = set(allowed)
    for raw in values:
        token = transform(str(raw).strip())
        if token not in allowed_set:
            raise SystemExit(
                f"Unknown {kind} '{raw}'. Expected one of: {', '.join(allowed)}"
            )
        if token not in seen:
            seen.append(token)
    return tuple(seen)


def pose_sort_key(pose_model_size: str) -> Tuple[int, str]:
    index = POSE_MODEL_ORDER.index(pose_model_size) if pose_model_size in POSE_MODEL_ORDER else 999
    return index, pose_model_size


def prune_sort_key(prune_variant: str) -> Tuple[int, str]:
    match = PRUNE_VARIANT_PATTERN.fullmatch(prune_variant)
    if match is None:
        return 999, prune_variant
    return int(match.group(1)), prune_variant


def precision_sort_key(precision: str) -> Tuple[int, str]:
    index = KNOWN_PRECISIONS.index(precision) if precision in KNOWN_PRECISIONS else 999
    return index, precision


def discovered_variant_sort_key(
    variant: DiscoveredVariant,
) -> Tuple[Tuple[int, str], Tuple[int, str], Tuple[int, str]]:
    return (
        pose_sort_key(variant.pose_model_size),
        prune_sort_key(variant.prune_variant),
        precision_sort_key(variant.precision),
    )


def discover_variants(keypoints_root: Path) -> Tuple[List[DiscoveredVariant], List[str]]:
    issues: List[str] = []
    variants: List[DiscoveredVariant] = []

    if not keypoints_root.is_dir():
        issues.append(f"Keypoints root does not exist or is not a directory: {keypoints_root.as_posix()}")
        return variants, issues

    for pose_dir in sorted(keypoints_root.iterdir(), key=lambda item: item.name):
        if not pose_dir.is_dir():
            continue
        pose_model_size = pose_dir.name
        if POSE_MODEL_PATTERN.fullmatch(pose_model_size) is None:
            issues.append(f"Skipping unrecognized pose directory: {pose_dir.as_posix()}")
            continue

        for prune_dir in sorted(pose_dir.iterdir(), key=lambda item: item.name):
            if not prune_dir.is_dir():
                continue
            prune_variant = prune_dir.name
            if PRUNE_VARIANT_PATTERN.fullmatch(prune_variant) is None:
                issues.append(f"Skipping unrecognized prune directory: {prune_dir.as_posix()}")
                continue

            for precision_dir in sorted(prune_dir.iterdir(), key=lambda item: item.name):
                if not precision_dir.is_dir():
                    continue
                precision = precision_dir.name.lower()
                if precision not in KNOWN_PRECISIONS:
                    issues.append(f"Skipping unrecognized precision directory: {precision_dir.as_posix()}")
                    continue
                variants.append(
                    DiscoveredVariant(
                        pose_model_size=pose_model_size,
                        prune_variant=prune_variant,
                        precision=precision,
                        npz_root=precision_dir,
                    )
                )

    variants.sort(key=discovered_variant_sort_key)
    if not variants:
        issues.append(f"No valid pose/prune/precision directories were discovered under: {keypoints_root.as_posix()}")
    return variants, issues


def get_checkpoint_subdir_name(pose_size: str) -> str:
    return f"{pose_size}-pose"


def get_checkpoint_path(paths: SweepPaths, classifier_model: str, pose_size: str) -> Path:
    checkpoint_subdir = get_checkpoint_subdir_name(pose_size)
    if classifier_model in ("cnnlstm", "stgcn"):
        return (
            paths.classification_root
            / classifier_model
            / checkpoint_subdir
            / f"{classifier_model}_best.pt"
        )
    if classifier_model == "MotionBERT":
        return (
            paths.classification_root
            / "MotionBERT"
            / checkpoint_subdir
            / "FT_MB_release_MB_ft_UPFall_xsub"
            / "best_epoch.bin"
        )
    raise ValueError(f"Unsupported classifier model: {classifier_model}")


def get_run_dir(
    paths: SweepPaths,
    classifier_model: str,
    pose_size: str,
    prune_variant: str,
    precision: str,
) -> Path:
    return paths.output_root / classifier_model / pose_size / prune_variant / precision


def get_motionbert_pkl_paths(
    paths: SweepPaths,
    pose_size: str,
    prune_variant: str,
    precision: str,
) -> Tuple[Path, Path]:
    out_dir = paths.motionbert_pkl_root / pose_size / prune_variant / precision
    stem = (
        f"upfall_motionbert_{pose_size}_{prune_variant}_{precision}"
        f"_cam_{'_'.join(str(c) for c in CAMERAS)}"
        f"_train_{TRAIN_SUBJECTS.replace('-', '_')}"
        f"_val_{VAL_SUBJECTS.replace('-', '_')}"
        f"_label_{LABEL_MODE}_win_{WINDOW_T}_step_{WINDOW_STRIDE}"
    )
    return out_dir / f"{stem}.pkl", out_dir / f"{stem}_label_map.json"


def build_run_matrix(paths: SweepPaths, variants: Sequence[DiscoveredVariant]) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for variant in variants:
        for classifier_model in CLASSIFIER_MODELS:
            checkpoint_path = get_checkpoint_path(paths, classifier_model, variant.pose_model_size)
            run_dir = get_run_dir(
                paths,
                classifier_model,
                variant.pose_model_size,
                variant.prune_variant,
                variant.precision,
            )
            kwargs: Dict[str, Any] = {}
            if classifier_model == "MotionBERT":
                pkl_path, label_map_path = get_motionbert_pkl_paths(
                    paths,
                    variant.pose_model_size,
                    variant.prune_variant,
                    variant.precision,
                )
                kwargs["motionbert_repo_root"] = paths.motionbert_code_root
                kwargs["motionbert_pkl_path"] = pkl_path
                kwargs["motionbert_label_map_path"] = label_map_path
            runs.append(
                RunSpec(
                    classifier_model=classifier_model,
                    pose_model_size=variant.pose_model_size,
                    prune_variant=variant.prune_variant,
                    precision=variant.precision,
                    npz_root=variant.npz_root,
                    checkpoint_path=checkpoint_path,
                    run_dir=run_dir,
                    stdout_log_path=run_dir / "stdout.log",
                    stderr_log_path=run_dir / "stderr.log",
                    run_status_path=run_dir / "run_status.json",
                    **kwargs,
                )
            )
    return runs


def is_selected_run(
    run: RunSpec,
    selected_models: Sequence[str],
    selected_pose_sizes: Sequence[str],
    selected_prune_variants: Sequence[str],
    selected_precisions: Sequence[str],
) -> bool:
    return (
        run.classifier_model in selected_models
        and run.pose_model_size in selected_pose_sizes
        and run.prune_variant in selected_prune_variants
        and run.precision in selected_precisions
    )


def default_metrics() -> Dict[str, Optional[float]]:
    return {
        "macro_accuracy": None,
        "macro_recall": None,
        "macro_f1": None,
        "fall_f1": None,
    }


def base_master_entry(run: RunSpec) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "classifier_model": run.classifier_model,
        "pose_model_size": run.pose_model_size,
        "prune_variant": run.prune_variant,
        "precision": run.precision,
        "npz_root": run.npz_root.as_posix(),
        "checkpoint_path": run.checkpoint_path.as_posix(),
        "evaluator_output_dir": None,
        "stdout_log_path": run.stdout_log_path.as_posix(),
        "stderr_log_path": run.stderr_log_path.as_posix(),
        "status": "pending",
        "skip_reason": None,
        "error_message": None,
        "metrics": default_metrics(),
    }


def merge_master_entry(base: Dict[str, Any], existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if not existing:
        return merged
    for key, value in existing.items():
        if key == "metrics" and isinstance(value, dict):
            metrics = default_metrics()
            metrics.update(value)
            merged["metrics"] = metrics
        elif key not in merged:
            merged[key] = value
        else:
            merged[key] = value
    if not isinstance(merged.get("metrics"), dict):
        merged["metrics"] = default_metrics()
    else:
        metrics = default_metrics()
        metrics.update(merged["metrics"])
        merged["metrics"] = metrics
    return merged


def load_master_entries(master_json_path: Path) -> Dict[str, Dict[str, Any]]:
    raw = read_json(master_json_path)
    if not raw:
        return {}
    entries = raw.get("runs")
    if not isinstance(entries, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        out[run_id] = item
    return out


def write_master_json(
    master_json_path: Path,
    paths: SweepPaths,
    all_runs: Sequence[RunSpec],
    master_entries: Dict[str, Dict[str, Any]],
    discovered_variants: Sequence[DiscoveredVariant],
) -> None:
    ordered_entries = [master_entries[run.run_id] for run in all_runs]
    counts = Counter(str(entry.get("status", "unknown")) for entry in ordered_entries)
    payload = {
        "updated_at": now_iso(),
        "script_path": paths.script_path.as_posix(),
        "keypoints_root": paths.keypoints_root.as_posix(),
        "classification_root": paths.classification_root.as_posix(),
        "output_root": paths.output_root.as_posix(),
        "motionbert_pkl_root": paths.motionbert_pkl_root.as_posix(),
        "discovered_variants": [
            {
                "pose_model_size": variant.pose_model_size,
                "prune_variant": variant.prune_variant,
                "precision": variant.precision,
                "npz_root": variant.npz_root.as_posix(),
            }
            for variant in discovered_variants
        ],
        "status_counts": dict(sorted(counts.items())),
        "runs": ordered_entries,
    }
    write_json_atomic(master_json_path, payload)


def eval_models_supports_weights_path(paths: SweepPaths) -> bool:
    eval_models_path = paths.repo_root / "evaluation" / "eval_models.py"
    try:
        text = eval_models_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "--weights-path" in text


def build_eval_models_command(paths: SweepPaths, run: RunSpec) -> List[str]:
    command = [
        sys.executable,
        "-m",
        "evaluation.eval_models",
        "--models",
        run.classifier_model,
        "--test-subjects",
        TEST_SUBJECTS,
        "--npz-root",
        run.npz_root.as_posix(),
        "--camera",
        *(str(camera) for camera in CAMERAS),
        "--label-mode",
        LABEL_MODE,
        "--drop-ambig-share",
        "0",
        "--T",
        str(WINDOW_T),
        "--stride",
        str(WINDOW_STRIDE),
        "--normalize",
        "1",
        "--normalize-mode",
        "paper_rp",
        "--rp-center-mode",
        "pixel",
        "--rp-img-w",
        "640",
        "--rp-img-h",
        "480",
        "--missing-mode",
        "zeros_only",
        "--interp-mode",
        "paper_group_linear",
        "--interp-group",
        "100",
        "--out-dir",
        run.run_dir.as_posix(),
    ]
    if eval_models_supports_weights_path(paths):
        command.extend(["--weights-path", run.checkpoint_path.as_posix()])
    else:
        command.extend(
            [
                "--ckpt-root",
                paths.classification_root.as_posix(),
                "--ckpt",
                f"{run.classifier_model}={run.checkpoint_path.parent.name}",
                "--weights-name",
                run.checkpoint_path.name,
            ]
        )
    return command


def build_motionbert_prepare_command(paths: SweepPaths, run: RunSpec) -> List[str]:
    if run.motionbert_pkl_path is None or run.motionbert_label_map_path is None:
        raise ValueError("MotionBERT prepare command requested for a non-MotionBERT run.")
    return [
        sys.executable,
        paths.prepare_motionbert_script.as_posix(),
        "--outputs-npz-root",
        run.npz_root.as_posix(),
        "--camera",
        *(str(camera) for camera in CAMERAS),
        "--train-subjects",
        TRAIN_SUBJECTS,
        "--val-subjects",
        VAL_SUBJECTS,
        "--out-pkl",
        run.motionbert_pkl_path.as_posix(),
        "--out-label-map",
        run.motionbert_label_map_path.as_posix(),
        "--label-mode",
        LABEL_MODE,
        "--win-len",
        str(WINDOW_T),
        "--win-step",
        str(WINDOW_STRIDE),
    ]


def build_motionbert_eval_command(run: RunSpec) -> List[str]:
    if run.motionbert_pkl_path is None:
        raise ValueError("MotionBERT eval command requested for a non-MotionBERT run.")
    return [
        sys.executable,
        "eval_motionbert_action.py",
        "--config",
        "configs/action/MB_ft_UPFall_xsub.yaml",
        "--checkpoint",
        run.checkpoint_path.as_posix(),
        "--subjects",
        TEST_SUBJECTS,
        "--camera",
        *(str(camera) for camera in CAMERAS),
        "--out-dir",
        run.run_dir.as_posix(),
        "--batch-size",
        "64",
        "--num-workers",
        "0",
        "--device",
        "cuda",
        "--ckpt-metric",
        "composite",
        "--ckpt-w",
        "0.7",
        "--ckpt-beta",
        "2.0",
        "--fall-class-idx",
        "0",
        "--fall-class-ids",
        "0",
        "--split-base",
        "xsub_train",
        "--no-class-weights",
        "--data-pkl",
        run.motionbert_pkl_path.as_posix(),
    ]


def valid_motionbert_pkl(pkl_path: Optional[Path], label_map_path: Optional[Path]) -> bool:
    if pkl_path is None or label_map_path is None:
        return False
    if not pkl_path.is_file() or pkl_path.stat().st_size <= 0:
        return False
    if not label_map_path.is_file() or label_map_path.stat().st_size <= 0:
        return False
    try:
        with label_map_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict)


def record_log_header(handle: Any, *, title: str, cwd: Path, cmd: Sequence[str]) -> None:
    handle.write(f"[{now_iso()}] {title}\n")
    handle.write(f"cwd: {cwd.as_posix()}\n")
    handle.write(f"command: {quote_command(cmd)}\n\n")
    handle.flush()


def run_subprocess_to_logs(
    cmd: Sequence[str],
    *,
    cwd: Path,
    stdout_handle: Any,
    stderr_handle: Any,
    title: str,
) -> int:
    record_log_header(stdout_handle, title=title, cwd=cwd, cmd=cmd)
    record_log_header(stderr_handle, title=title, cwd=cwd, cmd=cmd)
    completed = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd),
        stdout=stdout_handle,
        stderr=stderr_handle,
        check=False,
        env=os.environ.copy(),
    )
    stdout_handle.write(f"\n[{now_iso()}] return_code={completed.returncode}\n")
    stderr_handle.write(f"\n[{now_iso()}] return_code={completed.returncode}\n")
    stdout_handle.flush()
    stderr_handle.flush()
    return int(completed.returncode)


def list_child_dirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    try:
        return [child for child in path.iterdir() if child.is_dir()]
    except OSError:
        return []


def has_complete_evaluator_artifacts(output_dir: Optional[Path]) -> bool:
    if output_dir is None or not output_dir.is_dir():
        return False
    return (output_dir / "metrics_summary.csv").is_file() and (output_dir / "report.html").is_file()


def find_latest_evaluator_output(run_dir: Path, candidate_names: Optional[Iterable[str]] = None) -> Optional[Path]:
    allow = set(candidate_names) if candidate_names is not None else None
    candidates: List[Path] = []
    for child in list_child_dirs(run_dir):
        if allow is not None and child.name not in allow:
            continue
        if has_complete_evaluator_artifacts(child):
            candidates.append(child)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.name)[-1]


def write_run_status(run: RunSpec, payload: Dict[str, Any]) -> None:
    write_json_atomic(run.run_status_path, payload)


def detect_completed_run(run: RunSpec) -> Optional[Dict[str, Any]]:
    payload = read_json(run.run_status_path)
    if not payload:
        return None
    if str(payload.get("status")) != "completed":
        return None
    if int(payload.get("return_code", -1)) != 0:
        return None
    discovered = payload.get("discovered_output_dir")
    if not isinstance(discovered, str) or not discovered:
        return None
    output_dir = Path(discovered)
    if not has_complete_evaluator_artifacts(output_dir):
        return None
    return payload


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def select_combined_row(
    df: pd.DataFrame,
    *,
    classifier_model: Optional[str] = None,
) -> pd.Series:
    norm = normalize_columns(df)
    if "eval_split" in norm.columns:
        norm = norm[norm["eval_split"].astype(str).str.strip().str.lower() == "combined"]
    if classifier_model is not None and "model" in norm.columns:
        norm = norm[norm["model"].astype(str).str.strip().str.lower() == classifier_model.lower()]
    if norm.empty:
        raise ValueError("No combined row matched the requested evaluator output.")
    return norm.iloc[0]


def fallback_read_html_tables(report_html: Path) -> List["pd.DataFrame"]:
    pandas = require_pandas()
    parser = _HTMLTableParser()
    parser.feed(report_html.read_text(encoding="utf-8"))
    tables: List["pd.DataFrame"] = []
    for rows in parser.tables:
        if len(rows) < 2:
            continue
        header = rows[0]
        body = rows[1:]
        width = len(header)
        cleaned_body = [row[:width] + [""] * max(0, width - len(row)) for row in body]
        tables.append(pandas.DataFrame(cleaned_body, columns=header))
    return tables


def parse_overall_metrics_table(report_html: Path) -> pd.DataFrame:
    pandas = require_pandas()
    read_html_errors: List[str] = []
    tables: Optional[List["pd.DataFrame"]] = None
    for flavor in (None, "lxml", "html5lib", "bs4"):
        try:
            kwargs = {} if flavor is None else {"flavor": flavor}
            tables = pandas.read_html(report_html, **kwargs)
            break
        except Exception as exc:
            label = "default" if flavor is None else flavor
            read_html_errors.append(f"{label}: {exc}")
    if tables is None:
        tables = fallback_read_html_tables(report_html)
        if not tables:
            raise RuntimeError(
                "Unable to parse evaluator HTML tables. "
                "pandas.read_html() failed and the fallback HTML table parser found no usable tables. "
                f"Errors: {' | '.join(read_html_errors)}"
            )
    for table in tables:
        norm = normalize_columns(table)
        cols = set(norm.columns)
        if {"accuracy", "recall"}.issubset(cols):
            return norm
    raise ValueError(f"Could not find the Overall metrics table in {report_html.as_posix()}")


def normalize_metric_0_to_100(value: Any) -> Optional[float]:
    pandas = require_pandas()
    if value is None:
        return None
    if pandas.isna(value):
        return None
    numeric = float(value)
    if -1.000001 <= numeric <= 1.000001:
        return numeric * 100.0
    return numeric


def parse_eval_models_metrics(output_dir: Path, classifier_model: str) -> Dict[str, Optional[float]]:
    pandas = require_pandas()
    summary_path = output_dir / "metrics_summary.csv"
    report_path = output_dir / "report.html"
    summary_df = pandas.read_csv(summary_path)
    summary_row = select_combined_row(summary_df, classifier_model=classifier_model)
    overall_df = parse_overall_metrics_table(report_path)
    overall_row = select_combined_row(overall_df, classifier_model=classifier_model)
    return {
        "macro_accuracy": normalize_metric_0_to_100(overall_row.get("accuracy")),
        "macro_recall": normalize_metric_0_to_100(overall_row.get("recall")),
        "macro_f1": normalize_metric_0_to_100(summary_row.get("macro_f1")),
        "fall_f1": normalize_metric_0_to_100(summary_row.get("binary_f1_fall")),
    }


def parse_motionbert_metrics(output_dir: Path) -> Dict[str, Optional[float]]:
    pandas = require_pandas()
    summary_path = output_dir / "metrics_summary.csv"
    summary_df = pandas.read_csv(summary_path)
    summary_row = select_combined_row(summary_df)
    return {
        "macro_accuracy": normalize_metric_0_to_100(summary_row.get("acc_top1")),
        "macro_recall": normalize_metric_0_to_100(summary_row.get("balanced_acc")),
        "macro_f1": normalize_metric_0_to_100(summary_row.get("macro_f1")),
        "fall_f1": normalize_metric_0_to_100(summary_row.get("bin_f1_fall")),
    }


def parse_run_metrics(run: RunSpec, output_dir: Path) -> Dict[str, Optional[float]]:
    if run.classifier_model in ("cnnlstm", "stgcn"):
        return parse_eval_models_metrics(output_dir, run.classifier_model)
    if run.classifier_model == "MotionBERT":
        return parse_motionbert_metrics(output_dir)
    raise ValueError(f"Unsupported classifier model: {run.classifier_model}")


def check_required_inputs(paths: SweepPaths, run: RunSpec) -> List[str]:
    missing: List[str] = []
    if not paths.prepare_motionbert_script.is_file():
        missing.append(f"prepare script missing: {paths.prepare_motionbert_script.as_posix()}")
    eval_models_path = paths.repo_root / "evaluation" / "eval_models.py"
    if not eval_models_path.is_file():
        missing.append(f"eval_models.py missing: {eval_models_path.as_posix()}")
    if not run.npz_root.is_dir():
        missing.append(f"npz root missing: {run.npz_root.as_posix()}")
    if not run.checkpoint_path.is_file():
        missing.append(f"checkpoint missing: {run.checkpoint_path.as_posix()}")
    if run.classifier_model == "MotionBERT":
        if run.motionbert_repo_root is None or not run.motionbert_repo_root.is_dir():
            repo_root_str = "" if run.motionbert_repo_root is None else run.motionbert_repo_root.as_posix()
            missing.append(f"MotionBERT repo root missing: {repo_root_str}")
        else:
            eval_script = run.motionbert_repo_root / "eval_motionbert_action.py"
            config_path = run.motionbert_repo_root / "configs" / "action" / "MB_ft_UPFall_xsub.yaml"
            if not eval_script.is_file():
                missing.append(f"MotionBERT eval script missing: {eval_script.as_posix()}")
            if not config_path.is_file():
                missing.append(f"MotionBERT config missing: {config_path.as_posix()}")
    return missing


def build_run_status_payload(
    run: RunSpec,
    *,
    status: str,
    started_at: Optional[str],
    finished_at: Optional[str],
    return_code: Optional[int],
    cwd: Optional[Path],
    command: Optional[Sequence[str]],
    discovered_output_dir: Optional[Path],
    metrics: Optional[Dict[str, Optional[float]]] = None,
    skip_reason: Optional[str] = None,
    error_message: Optional[str] = None,
    prepare_command: Optional[Sequence[str]] = None,
    prepare_return_code: Optional[int] = None,
    reused_motionbert_pkl: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "classifier_model": run.classifier_model,
        "pose_model_size": run.pose_model_size,
        "prune_variant": run.prune_variant,
        "precision": run.precision,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "cwd": None if cwd is None else cwd.as_posix(),
        "command": None if command is None else quote_command(command),
        "prepare_command": None if prepare_command is None else quote_command(prepare_command),
        "return_code": return_code,
        "prepare_return_code": prepare_return_code,
        "discovered_output_dir": None if discovered_output_dir is None else discovered_output_dir.as_posix(),
        "stdout_log_path": run.stdout_log_path.as_posix(),
        "stderr_log_path": run.stderr_log_path.as_posix(),
        "metrics": metrics if metrics is not None else default_metrics(),
        "skip_reason": skip_reason,
        "error_message": error_message,
        "npz_root": run.npz_root.as_posix(),
        "checkpoint_path": run.checkpoint_path.as_posix(),
        "reused_motionbert_pkl": reused_motionbert_pkl,
        "motionbert_pkl_path": None if run.motionbert_pkl_path is None else run.motionbert_pkl_path.as_posix(),
        "motionbert_label_map_path": None
        if run.motionbert_label_map_path is None
        else run.motionbert_label_map_path.as_posix(),
    }


def build_master_entry_from_run_status(run: RunSpec, run_status: Dict[str, Any]) -> Dict[str, Any]:
    entry = base_master_entry(run)
    entry["evaluator_output_dir"] = run_status.get("discovered_output_dir")
    entry["status"] = run_status.get("status", "unknown")
    entry["skip_reason"] = run_status.get("skip_reason")
    entry["error_message"] = run_status.get("error_message")
    metrics = run_status.get("metrics")
    if isinstance(metrics, dict):
        merged_metrics = default_metrics()
        merged_metrics.update(metrics)
        entry["metrics"] = merged_metrics
    return entry


def process_existing_completed_run(run: RunSpec) -> Dict[str, Any]:
    existing_status = detect_completed_run(run)
    if existing_status is None:
        raise ValueError("process_existing_completed_run called for a non-complete run.")
    output_dir = Path(str(existing_status["discovered_output_dir"]))
    error_message = existing_status.get("error_message")
    metrics = default_metrics()
    try:
        metrics = parse_run_metrics(run, output_dir)
        error_message = None
    except Exception as exc:
        error_message = f"Completed artifacts found but metric parsing failed: {exc}"

    command = None
    if existing_status.get("command") is not None:
        command = shlex.split(str(existing_status["command"]))
    prepare_command = None
    if existing_status.get("prepare_command") is not None:
        prepare_command = shlex.split(str(existing_status["prepare_command"]))

    updated_status = build_run_status_payload(
        run,
        status="completed",
        started_at=existing_status.get("started_at"),
        finished_at=existing_status.get("finished_at"),
        return_code=int(existing_status.get("return_code", 0)),
        cwd=Path(str(existing_status["cwd"])) if existing_status.get("cwd") else None,
        command=command,
        discovered_output_dir=output_dir,
        metrics=metrics,
        skip_reason=None,
        error_message=error_message,
        prepare_command=prepare_command,
        prepare_return_code=existing_status.get("prepare_return_code"),
        reused_motionbert_pkl=existing_status.get("reused_motionbert_pkl"),
    )
    write_run_status(run, updated_status)
    return build_master_entry_from_run_status(run, updated_status)


def ensure_parent_dirs_for_run(run: RunSpec) -> None:
    ensure_dir(run.run_dir)
    if run.motionbert_pkl_path is not None:
        ensure_dir(run.motionbert_pkl_path.parent)
    if run.motionbert_label_map_path is not None:
        ensure_dir(run.motionbert_label_map_path.parent)


def dry_run_status_label(
    *,
    force_rerun: bool,
    complete_now: bool,
    missing_inputs: Sequence[str],
) -> str:
    if missing_inputs:
        return "would_skip_missing_input"
    if complete_now and not force_rerun:
        return "would_skip_completed"
    return "would_run"


def execute_run(
    paths: SweepPaths,
    run: RunSpec,
    *,
    force_regenerate_motionbert_pkl: bool,
) -> Dict[str, Any]:
    ensure_parent_dirs_for_run(run)
    existing_dir_names = {child.name for child in list_child_dirs(run.run_dir)}
    started_at = now_iso()

    if run.classifier_model in ("cnnlstm", "stgcn"):
        command = build_eval_models_command(paths, run)
        command_cwd = paths.repo_root
        prepare_command: Optional[List[str]] = None
    else:
        command = build_motionbert_eval_command(run)
        if run.motionbert_repo_root is None:
            raise ValueError("MotionBERT run is missing motionbert_repo_root.")
        command_cwd = run.motionbert_repo_root
        prepare_command = build_motionbert_prepare_command(paths, run)

    running_payload = build_run_status_payload(
        run,
        status="running",
        started_at=started_at,
        finished_at=None,
        return_code=None,
        cwd=command_cwd,
        command=command,
        discovered_output_dir=None,
        metrics=default_metrics(),
        prepare_command=prepare_command,
    )
    write_run_status(run, running_payload)

    prepare_return_code: Optional[int] = None
    reused_motionbert_pkl: Optional[bool] = None
    return_code: Optional[int] = None
    discovered_output_dir: Optional[Path] = None
    metrics = default_metrics()
    error_message: Optional[str] = None

    try:
        with run.stdout_log_path.open("w", encoding="utf-8") as stdout_handle, run.stderr_log_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            if run.classifier_model == "MotionBERT":
                have_valid_pkl = valid_motionbert_pkl(run.motionbert_pkl_path, run.motionbert_label_map_path)
                should_prepare = force_regenerate_motionbert_pkl or not have_valid_pkl

                if should_prepare:
                    reused_motionbert_pkl = False
                    if force_regenerate_motionbert_pkl and have_valid_pkl:
                        stdout_handle.write(
                            f"[{now_iso()}] Regenerating MotionBERT PKL due to "
                            "--force-regenerate-motionbert-pkl\n\n"
                        )
                        stdout_handle.flush()
                    prepare_return_code = run_subprocess_to_logs(
                        prepare_command or [],
                        cwd=paths.repo_root,
                        stdout_handle=stdout_handle,
                        stderr_handle=stderr_handle,
                        title="prepare_motionbert_dataset",
                    )
                    if prepare_return_code != 0:
                        stderr_tail = read_log_tail(run.stderr_log_path)
                        tail_suffix = "" if not stderr_tail else f" Last stderr lines:\n{stderr_tail}"
                        raise RuntimeError(
                            f"MotionBERT dataset preparation failed with return code {prepare_return_code}."
                            f"{tail_suffix}"
                        )
                    if not valid_motionbert_pkl(run.motionbert_pkl_path, run.motionbert_label_map_path):
                        raise RuntimeError(
                            "MotionBERT dataset preparation completed but the expected PKL/label map files "
                            "were not produced or looked invalid."
                        )
                else:
                    reused_motionbert_pkl = True
                    stdout_handle.write(
                        f"[{now_iso()}] Reusing existing MotionBERT PKL: "
                        f"{run.motionbert_pkl_path.as_posix()}\n\n"
                    )
                    stdout_handle.flush()

            return_code = run_subprocess_to_logs(
                command,
                cwd=command_cwd,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                title="evaluation",
            )
    except Exception as exc:
        error_message = str(exc)

    new_dir_names = {child.name for child in list_child_dirs(run.run_dir)} - existing_dir_names
    discovered_output_dir = find_latest_evaluator_output(run.run_dir, candidate_names=new_dir_names)
    finished_at = now_iso()

    status = "failed"
    if error_message is None and return_code == 0 and has_complete_evaluator_artifacts(discovered_output_dir):
        status = "completed"
        try:
            metrics = parse_run_metrics(run, discovered_output_dir)
        except Exception as exc:
            error_message = f"Evaluation completed but metric parsing failed: {exc}"
    else:
        if error_message is None:
            if return_code != 0:
                stderr_tail = read_log_tail(run.stderr_log_path)
                tail_suffix = "" if not stderr_tail else f" Last stderr lines:\n{stderr_tail}"
                error_message = f"Evaluation subprocess failed with return code {return_code}.{tail_suffix}"
            else:
                error_message = (
                    "Evaluation subprocess finished without a new complete evaluator output directory "
                    "containing metrics_summary.csv and report.html."
                )

    payload = build_run_status_payload(
        run,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        return_code=return_code,
        cwd=command_cwd,
        command=command,
        discovered_output_dir=discovered_output_dir,
        metrics=metrics,
        skip_reason=None,
        error_message=error_message,
        prepare_command=prepare_command,
        prepare_return_code=prepare_return_code,
        reused_motionbert_pkl=reused_motionbert_pkl,
    )
    write_run_status(run, payload)
    return build_master_entry_from_run_status(run, payload)


def handle_missing_inputs(run: RunSpec, missing_inputs: Sequence[str]) -> Dict[str, Any]:
    ensure_parent_dirs_for_run(run)
    reason = "; ".join(missing_inputs)
    payload = build_run_status_payload(
        run,
        status="skipped_missing_input",
        started_at=now_iso(),
        finished_at=now_iso(),
        return_code=None,
        cwd=None,
        command=None,
        discovered_output_dir=None,
        metrics=default_metrics(),
        skip_reason=reason,
        error_message=None,
    )
    write_run_status(run, payload)
    return build_master_entry_from_run_status(run, payload)


def print_dry_run_line(
    run: RunSpec,
    *,
    label: str,
    command: Sequence[str],
    command_cwd: Path,
    prepare_command: Optional[Sequence[str]] = None,
) -> None:
    print(
        f"[DRY-RUN] {label}: {run.run_id}\n"
        f"  run_dir={run.run_dir.as_posix()}\n"
        f"  npz_root={run.npz_root.as_posix()}\n"
        f"  checkpoint={run.checkpoint_path.as_posix()}\n"
        f"  cwd={command_cwd.as_posix()}\n"
        f"  cmd={quote_command(command)}"
    )
    if prepare_command is not None:
        print(f"  prepare_cmd={quote_command(prepare_command)}")


def main() -> int:
    args = parse_args()

    script_path = Path(__file__).resolve()
    eval_dir = script_path.parent
    repo_root = eval_dir.parent.resolve()

    paths = SweepPaths(
        script_path=script_path,
        eval_dir=eval_dir,
        repo_root=repo_root,
        prepare_motionbert_script=(repo_root / "dataset_helpers" / "prepare_motionbert_dataset.py").resolve(),
        motionbert_code_root=(
            args.motionbert_code_root.expanduser().resolve()
            if args.motionbert_code_root is not None
            else (repo_root / "models" / "MotionBERT").resolve()
        ),
        keypoints_root=args.keypoints_root.expanduser(),
        classification_root=args.classification_root.expanduser(),
        output_root=args.output_root.expanduser(),
        motionbert_pkl_root=args.motionbert_pkl_root.expanduser(),
    )

    discovered_variants, discovery_issues = discover_variants(paths.keypoints_root)
    if not discovered_variants:
        joined = "\n".join(f"- {issue}" for issue in discovery_issues) if discovery_issues else "(no details)"
        raise SystemExit(f"No evaluable variants were discovered.\n{joined}")

    discovered_pose_sizes = tuple(dict.fromkeys(v.pose_model_size for v in discovered_variants))
    discovered_prune_variants = tuple(dict.fromkeys(v.prune_variant for v in discovered_variants))
    discovered_precisions = tuple(dict.fromkeys(v.precision for v in discovered_variants))

    selected_models = (
        tuple(normalize_model_token(token) for token in args.only_models)
        if args.only_models
        else CLASSIFIER_MODELS
    )
    selected_pose_sizes = normalize_filter_tokens(
        args.only_pose_sizes,
        allowed=discovered_pose_sizes,
        kind="pose size",
        transform=lambda value: value.lower(),
    )
    selected_prune_variants = normalize_filter_tokens(
        args.only_prune_variants,
        allowed=discovered_prune_variants,
        kind="prune variant",
        transform=lambda value: value.lower(),
    )
    selected_precisions = normalize_filter_tokens(
        args.only_precisions,
        allowed=discovered_precisions,
        kind="precision",
        transform=lambda value: value.lower(),
    )

    all_runs = build_run_matrix(paths, discovered_variants)
    selected_runs = [
        run
        for run in all_runs
        if is_selected_run(
            run,
            selected_models,
            selected_pose_sizes,
            selected_prune_variants,
            selected_precisions,
        )
    ]
    if not selected_runs:
        raise SystemExit("No runs selected after applying the CLI filters.")

    print(
        f"Discovered {len(discovered_variants)} keypoint variant(s) under {paths.keypoints_root.as_posix()} "
        f"covering pose_sizes={list(discovered_pose_sizes)} prune_variants={list(discovered_prune_variants)} "
        f"precisions={list(discovered_precisions)}"
    )
    if discovery_issues:
        preview = discovery_issues[:20]
        print(f"Discovery notes ({len(discovery_issues)}):")
        for issue in preview:
            print(f"  - {issue}")
        if len(preview) < len(discovery_issues):
            print(f"  - ... {len(discovery_issues) - len(preview)} more")

    master_json_path = paths.output_root / MASTER_JSON_NAME
    existing_master_entries = load_master_entries(master_json_path) if master_json_path.exists() else {}
    master_entries: Dict[str, Dict[str, Any]] = {}
    for run in all_runs:
        master_entries[run.run_id] = merge_master_entry(
            base_master_entry(run),
            existing_master_entries.get(run.run_id),
        )

    print(
        f"Selected {len(selected_runs)} run(s) out of {len(all_runs)} total. "
        f"force_rerun={bool(args.force_rerun)} dry_run={bool(args.dry_run)} "
        f"force_regenerate_motionbert_pkl={bool(args.force_regenerate_motionbert_pkl)}"
    )

    if not args.dry_run:
        ensure_dir(paths.output_root)
        write_master_json(master_json_path, paths, all_runs, master_entries, discovered_variants)

    for index, run in enumerate(selected_runs, start=1):
        prefix = f"[{index}/{len(selected_runs)}] {run.run_id}"
        missing_inputs = check_required_inputs(paths, run)
        is_complete = detect_completed_run(run) is not None

        if args.dry_run:
            if run.classifier_model in ("cnnlstm", "stgcn"):
                command = build_eval_models_command(paths, run)
                command_cwd = paths.repo_root
                prepare_command = None
            else:
                command = build_motionbert_eval_command(run)
                if run.motionbert_repo_root is None:
                    raise SystemExit(f"{run.run_id} is missing MotionBERT repo root.")
                command_cwd = run.motionbert_repo_root
                prepare_command = build_motionbert_prepare_command(paths, run)
            label = dry_run_status_label(
                force_rerun=args.force_rerun,
                complete_now=is_complete,
                missing_inputs=missing_inputs,
            )
            print_dry_run_line(
                run,
                label=label,
                command=command,
                command_cwd=command_cwd,
                prepare_command=prepare_command,
            )
            if missing_inputs:
                print(f"  reason={'; '.join(missing_inputs)}")
            continue

        if is_complete and not args.force_rerun:
            print(f"{prefix} -> skip existing completed run")
            entry = process_existing_completed_run(run)
        elif missing_inputs:
            print(f"{prefix} -> skip missing input")
            entry = handle_missing_inputs(run, missing_inputs)
        else:
            print(f"{prefix} -> start")
            try:
                entry = execute_run(
                    paths,
                    run,
                    force_regenerate_motionbert_pkl=bool(args.force_regenerate_motionbert_pkl),
                )
            except Exception as exc:
                ensure_parent_dirs_for_run(run)
                error_message = f"{exc}\n{traceback.format_exc()}"
                payload = build_run_status_payload(
                    run,
                    status="failed",
                    started_at=now_iso(),
                    finished_at=now_iso(),
                    return_code=None,
                    cwd=None,
                    command=None,
                    discovered_output_dir=None,
                    metrics=default_metrics(),
                    skip_reason=None,
                    error_message=error_message,
                )
                write_run_status(run, payload)
                entry = build_master_entry_from_run_status(run, payload)

        master_entries[run.run_id] = merge_master_entry(base_master_entry(run), entry)
        write_master_json(master_json_path, paths, all_runs, master_entries, discovered_variants)
        print(
            f"{prefix} -> {master_entries[run.run_id]['status']}"
            f"{'' if master_entries[run.run_id]['error_message'] is None else ' (with note)'}"
        )

    if args.dry_run:
        print("Dry-run complete. No files were written.")
        return 0

    counts = Counter(str(master_entries[run.run_id].get("status", "unknown")) for run in all_runs)
    print(f"Master JSON: {master_json_path.as_posix()}")
    print(f"Status counts: {dict(sorted(counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
