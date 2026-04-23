#!/usr/bin/env python3
"""Evaluate final classifiers on downsized YOLO11 keypoint variants.

This mirrors the evaluation settings used by ``final_scripts/eval_final_models.py``
but discovers keypoints from the downsized-image layout:

    /home/people/21376026/scratch/keypoints/downsized_keypoints/
      yolo11n/base_448
      yolo11n/base_512
      yolo11n/base_576
      ...

For each discovered keypoint variant, this runner evaluates:
  - paper_stgcn
  - cnnlstm
  - MotionBERT

Outputs are written under:

    /home/people/21376026/scratch/final_results/img_downsized_evals

and a combined JSON summary is written to:

    /home/people/21376026/scratch/final_results/img_downsized_evals/combined_results.json
"""

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


CLASSIFIER_MODELS: Tuple[str, ...] = ("paper_stgcn", "cnnlstm", "MotionBERT")
MODEL_BUCKETS: Dict[str, str] = {
    "paper_stgcn": "stgcn",
    "cnnlstm": "cnnlstm",
}

POSE_MODEL_ORDER: Tuple[str, ...] = ("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x")
YOLO_POSE_PATTERN = re.compile(r"^yolo11[nsmlx]$")
DOWNSIZED_VARIANT_PATTERN = re.compile(r"^base_(\d+)$")

DEFAULT_KEYPOINTS_ROOT = Path("/home/people/21376026/scratch/keypoints/downsized_keypoints")
DEFAULT_CLASSIFICATION_ROOT = Path("/home/people/21376026/scratch/final_classification_models")
DEFAULT_OUTPUT_ROOT = Path("/home/people/21376026/scratch/final_results/img_downsized_evals")

TEST_SUBJECTS = "16-17"
TRAIN_SUBJECTS = TEST_SUBJECTS
VAL_SUBJECTS = "13-15"
CAMERAS: Tuple[int, ...] = (1, 2)
WINDOW_T = 64
WINDOW_STRIDE = 48
LABEL_MODE = "center"
SUMMARY_JSON_NAME = "combined_results.json"
SUMMARY_VARIANT_NAME = "strict_label_mode"
DEFAULT_MOTIONBERT_EVAL_BATCH_SIZE = 32

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
    repo_root: Path
    eval_models_path: Path
    prepare_motionbert_script: Path
    motionbert_code_root: Path
    keypoints_root: Path
    classification_root: Path
    output_root: Path
    motionbert_pkl_root: Path


@dataclass(frozen=True)
class SweepConfig:
    python_executable: str
    motionbert_device: str
    batch_size: int
    num_workers: int
    force_regenerate_motionbert_pkl: bool


@dataclass(frozen=True)
class KeypointVariant:
    pose_model_size: str
    checkpoint_tag: str
    variant_name: str
    imgsz: int
    npz_root: Path

    @property
    def variant_id(self) -> str:
        return f"{self.pose_model_size}__{self.variant_name}"

    @property
    def keypoints_tag(self) -> str:
        return f"{self.pose_model_size}/{self.variant_name}"


@dataclass(frozen=True)
class RunSpec:
    classifier_model: str
    pose_model_size: str
    checkpoint_tag: str
    variant_name: str
    imgsz: int
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
        return f"{self.classifier_model}__{self.pose_model_size}__{self.variant_name}"

    @property
    def keypoints_tag(self) -> str:
        return f"{self.pose_model_size}/{self.variant_name}"


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
            "Install pandas in the active environment and retry."
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
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(path)


def normalize_model_token(token: str) -> str:
    cleaned = str(token).strip()
    aliases = {
        "motionbert": "MotionBERT",
        "MotionBERT": "MotionBERT",
        "paper_stgcn": "paper_stgcn",
        "cnnlstm": "cnnlstm",
    }
    if cleaned in aliases:
        return aliases[cleaned]
    lowered = cleaned.lower()
    if lowered in aliases:
        return aliases[lowered]
    raise SystemExit(
        f"Unknown model token '{token}'. Expected one of: {', '.join(CLASSIFIER_MODELS)}"
    )


def normalize_filter_tokens(
    values: Optional[Iterable[str]],
    *,
    allowed: Sequence[str],
    kind: str,
    transform=str.lower,
) -> Optional[Tuple[str, ...]]:
    if values is None:
        return None
    seen: List[str] = []
    allowed_set = set(allowed)
    for raw in values:
        token = transform(str(raw).strip())
        if token not in allowed_set:
            raise SystemExit(
                f"Unknown {kind} '{raw}'. Expected one of: {', '.join(str(item) for item in allowed)}"
            )
        if token not in seen:
            seen.append(token)
    return tuple(seen)


def pose_sort_key(pose_model_size: str) -> Tuple[int, str]:
    index = POSE_MODEL_ORDER.index(pose_model_size) if pose_model_size in POSE_MODEL_ORDER else 999
    return index, pose_model_size


def variant_sort_key(variant_name: str) -> Tuple[int, str]:
    match = DOWNSIZED_VARIANT_PATTERN.fullmatch(variant_name)
    if match is None:
        return 999999, variant_name
    return -int(match.group(1)), variant_name


def discover_variants(paths: SweepPaths) -> Tuple[List[KeypointVariant], List[str]]:
    issues: List[str] = []
    variants: List[KeypointVariant] = []

    if not paths.keypoints_root.is_dir():
        issues.append(f"Downsized keypoints root not found: {paths.keypoints_root.as_posix()}")
        return variants, issues

    for pose_dir in sorted(paths.keypoints_root.iterdir(), key=lambda item: item.name):
        if not pose_dir.is_dir():
            continue
        pose_name = pose_dir.name
        if YOLO_POSE_PATTERN.fullmatch(pose_name) is None:
            issues.append(f"Skipping unrecognized downsized pose directory: {pose_dir.as_posix()}")
            continue

        for variant_dir in sorted(pose_dir.iterdir(), key=lambda item: item.name):
            if not variant_dir.is_dir():
                continue
            variant_name = variant_dir.name
            match = DOWNSIZED_VARIANT_PATTERN.fullmatch(variant_name)
            if match is None:
                issues.append(f"Skipping unrecognized downsized variant directory: {variant_dir.as_posix()}")
                continue
            variants.append(
                KeypointVariant(
                    pose_model_size=pose_name,
                    checkpoint_tag=f"{pose_name}-pose",
                    variant_name=variant_name,
                    imgsz=int(match.group(1)),
                    npz_root=variant_dir,
                )
            )

    variants.sort(
        key=lambda item: (
            pose_sort_key(item.pose_model_size),
            variant_sort_key(item.variant_name),
        )
    )
    return variants, issues


def get_checkpoint_path(paths: SweepPaths, classifier_model: str, checkpoint_tag: str) -> Path:
    if classifier_model == "MotionBERT":
        return (
            paths.classification_root
            / "MotionBERT"
            / checkpoint_tag
            / "FT_MB_release_MB_ft_UPFall_xsub"
            / "best_epoch.bin"
        )
    bucket = MODEL_BUCKETS[classifier_model]
    return paths.classification_root / bucket / checkpoint_tag / f"{classifier_model}_best.pt"


def get_run_dir(paths: SweepPaths, classifier_model: str, variant: KeypointVariant) -> Path:
    return paths.output_root / classifier_model / variant.pose_model_size / variant.variant_name


def get_motionbert_pkl_paths(paths: SweepPaths, variant: KeypointVariant) -> Tuple[Path, Path]:
    out_dir = paths.motionbert_pkl_root / variant.pose_model_size / variant.variant_name
    stem = (
        f"upfall_motionbert_{variant.pose_model_size}_{variant.variant_name}"
        f"_cam_{'_'.join(str(c) for c in CAMERAS)}"
        f"_train_{TRAIN_SUBJECTS.replace('-', '_')}"
        f"_val_{VAL_SUBJECTS.replace('-', '_')}"
        f"_label_{LABEL_MODE}"
        f"_win_{WINDOW_T}"
        f"_step_{WINDOW_STRIDE}"
    )
    return out_dir / f"{stem}.pkl", out_dir / f"{stem}_label_map.json"


def build_run_matrix(paths: SweepPaths, variants: Sequence[KeypointVariant]) -> List[RunSpec]:
    runs: List[RunSpec] = []
    for variant in variants:
        for classifier_model in CLASSIFIER_MODELS:
            checkpoint_path = get_checkpoint_path(paths, classifier_model, variant.checkpoint_tag)
            run_dir = get_run_dir(paths, classifier_model, variant)
            kwargs: Dict[str, Any] = {}
            if classifier_model == "MotionBERT":
                pkl_path, label_map_path = get_motionbert_pkl_paths(paths, variant)
                kwargs["motionbert_repo_root"] = paths.motionbert_code_root
                kwargs["motionbert_pkl_path"] = pkl_path
                kwargs["motionbert_label_map_path"] = label_map_path
            runs.append(
                RunSpec(
                    classifier_model=classifier_model,
                    pose_model_size=variant.pose_model_size,
                    checkpoint_tag=variant.checkpoint_tag,
                    variant_name=variant.variant_name,
                    imgsz=variant.imgsz,
                    npz_root=variant.npz_root,
                    checkpoint_path=checkpoint_path,
                    run_dir=run_dir,
                    stdout_log_path=run_dir / "stdout.log",
                    stderr_log_path=run_dir / "stderr.log",
                    run_status_path=run_dir / "run_status.json",
                    **kwargs,
                )
            )
    runs.sort(
        key=lambda item: (
            CLASSIFIER_MODELS.index(item.classifier_model),
            pose_sort_key(item.pose_model_size),
            variant_sort_key(item.variant_name),
        )
    )
    return runs


def is_selected_run(
    run: RunSpec,
    *,
    selected_models: Optional[Sequence[str]],
    selected_pose_sizes: Optional[Sequence[str]],
    selected_imgsz: Optional[Sequence[int]],
) -> bool:
    if selected_models is not None and run.classifier_model not in selected_models:
        return False
    if selected_pose_sizes is not None and run.pose_model_size not in selected_pose_sizes:
        return False
    if selected_imgsz is not None and run.imgsz not in selected_imgsz:
        return False
    return True


def default_metrics() -> Dict[str, Optional[float]]:
    return {
        "accuracy": None,
        "recall": None,
        "macro_f1": None,
        "fall_f1": None,
    }


def round_metric(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 4)


def normalize_metrics(metrics: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    normalized = default_metrics()
    for key, value in metrics.items():
        if key in normalized:
            normalized[key] = round_metric(value)
    return normalized


def base_summary_entry(run: RunSpec) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "status": "pending",
        "classifier_model": run.classifier_model,
        "pose_model_size": run.pose_model_size,
        "checkpoint_tag": run.checkpoint_tag,
        "variant_name": run.variant_name,
        "imgsz": run.imgsz,
        "keypoints_tag": run.keypoints_tag,
        "npz_root": run.npz_root.as_posix(),
        "checkpoint_path": run.checkpoint_path.as_posix(),
        "evaluator_output_dir": None,
        "stdout_log_path": run.stdout_log_path.as_posix(),
        "stderr_log_path": run.stderr_log_path.as_posix(),
        "error_message": None,
        "metrics": default_metrics(),
    }


def build_summary_entry_from_status(run: RunSpec, run_status: Dict[str, Any]) -> Dict[str, Any]:
    entry = base_summary_entry(run)
    entry["status"] = run_status.get("status", "unknown")
    entry["evaluator_output_dir"] = run_status.get("discovered_output_dir")
    entry["error_message"] = run_status.get("error_message")
    metrics = run_status.get("metrics")
    if isinstance(metrics, dict):
        entry["metrics"] = normalize_metrics(metrics)
    return entry


def load_existing_summary_entries(all_runs: Sequence[RunSpec]) -> Dict[str, Dict[str, Any]]:
    entries: Dict[str, Dict[str, Any]] = {}
    for run in all_runs:
        run_status = read_json(run.run_status_path)
        if isinstance(run_status, dict) and str(run_status.get("run_id", run.run_id)) == run.run_id:
            entries[run.run_id] = build_summary_entry_from_status(run, run_status)
        else:
            entries[run.run_id] = base_summary_entry(run)
    return entries


def write_summary_json(
    summary_json_path: Path,
    paths: SweepPaths,
    all_runs: Sequence[RunSpec],
    summary_entries: Dict[str, Dict[str, Any]],
) -> None:
    ordered_entries = [summary_entries[run.run_id] for run in all_runs]
    counts = Counter(str(entry.get("status", "unknown")) for entry in ordered_entries)
    payload = {
        "updated_at": now_iso(),
        "output_root": paths.output_root.as_posix(),
        "classification_root": paths.classification_root.as_posix(),
        "keypoints_root": paths.keypoints_root.as_posix(),
        "test_subjects": TEST_SUBJECTS,
        "window": {
            "T": WINDOW_T,
            "stride": WINDOW_STRIDE,
            "label_mode": LABEL_MODE,
        },
        "summary_variant": {
            "eval_models": SUMMARY_VARIANT_NAME,
            "motionbert": "combined",
        },
        "total_runs": len(ordered_entries),
        "status_counts": dict(sorted(counts.items())),
        "results": ordered_entries,
    }
    write_json_atomic(summary_json_path, payload)


def eval_models_supports_weights_path(paths: SweepPaths) -> bool:
    try:
        text = paths.eval_models_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "--weights-path" in text


def build_eval_models_command(paths: SweepPaths, config: SweepConfig, run: RunSpec) -> List[str]:
    if not eval_models_supports_weights_path(paths):
        raise RuntimeError(
            "evaluation/eval_models.py does not support --weights-path in this checkout. "
            "This runner expects that flag so it can evaluate the flat final checkpoint layout."
        )
    return [
        config.python_executable,
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
        "--weights-path",
        run.checkpoint_path.as_posix(),
        "--out-dir",
        run.run_dir.as_posix(),
    ]


def build_motionbert_prepare_command(config: SweepConfig, paths: SweepPaths, run: RunSpec) -> List[str]:
    if run.motionbert_pkl_path is None or run.motionbert_label_map_path is None:
        raise ValueError("MotionBERT prepare command requested for a non-MotionBERT run.")
    return [
        config.python_executable,
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


def build_motionbert_eval_command(config: SweepConfig, run: RunSpec) -> List[str]:
    if run.motionbert_pkl_path is None:
        raise ValueError("MotionBERT eval command requested for a non-MotionBERT run.")
    return [
        config.python_executable,
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
        str(config.batch_size),
        "--num-workers",
        str(config.num_workers),
        "--device",
        config.motionbert_device,
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


def parse_subject_range_expr(expr: str) -> List[int]:
    subjects: List[int] = []
    for raw_part in str(expr).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                start, end = end, start
            subjects.extend(range(start, end + 1))
        else:
            subjects.append(int(part))
    return sorted(set(subjects))


def find_motionbert_source_npzs(npz_root: Path) -> List[Path]:
    subjects = sorted(set(parse_subject_range_expr(TRAIN_SUBJECTS) + parse_subject_range_expr(VAL_SUBJECTS)))
    camera_ids = sorted(set(int(camera) for camera in CAMERAS))
    npzs: List[Path] = []
    for subject in subjects:
        subject_root = npz_root / f"Subject{subject}"
        if not subject_root.is_dir():
            continue
        for camera in camera_ids:
            pattern = (
                f"Activity*/Trial*/"
                f"Subject{subject}Activity*Trial*Camera{camera}/keypoints.npz"
            )
            npzs.extend(path for path in subject_root.glob(pattern) if path.is_file())
    return sorted(set(npzs), key=lambda path: path.as_posix())


def motionbert_cache_is_newer_than_sources(
    pkl_path: Path,
    label_map_path: Path,
    npz_root: Path,
) -> bool:
    source_npzs = find_motionbert_source_npzs(npz_root)
    if not source_npzs:
        return False
    try:
        cache_mtime_ns = min(pkl_path.stat().st_mtime_ns, label_map_path.stat().st_mtime_ns)
    except OSError:
        return False
    for source_npz in source_npzs:
        try:
            if source_npz.stat().st_mtime_ns > cache_mtime_ns:
                return False
        except OSError:
            return False
    return True


def valid_motionbert_pkl(
    pkl_path: Optional[Path],
    label_map_path: Optional[Path],
    npz_root: Optional[Path] = None,
) -> bool:
    if pkl_path is None or label_map_path is None:
        return False
    if not pkl_path.is_file() or pkl_path.stat().st_size <= 0:
        return False
    if not label_map_path.is_file() or label_map_path.stat().st_size <= 0:
        return False
    data = read_json(label_map_path)
    if not isinstance(data, dict):
        return False
    if npz_root is not None and not motionbert_cache_is_newer_than_sources(
        pkl_path,
        label_map_path,
        npz_root,
    ):
        return False
    return True


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


def has_complete_evaluator_artifacts(output_dir: Optional[Path]) -> bool:
    if output_dir is None or not output_dir.is_dir():
        return False
    return (output_dir / "metrics_summary.csv").is_file() and (output_dir / "report.html").is_file()


def discover_complete_output_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    candidates: List[Path] = []
    try:
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            if has_complete_evaluator_artifacts(path):
                candidates.append(path)
    except OSError:
        return []
    return candidates


def latest_path_by_mtime(paths: Sequence[Path]) -> Optional[Path]:
    if not paths:
        return None

    def sort_key(path: Path) -> Tuple[float, str]:
        try:
            mtime = float(path.stat().st_mtime)
        except OSError:
            mtime = -1.0
        return mtime, path.as_posix()

    return sorted(paths, key=sort_key)[-1]


def choose_output_dir(run: RunSpec, candidates: Sequence[Path]) -> Optional[Path]:
    if not candidates:
        return None
    if run.classifier_model != "MotionBERT":
        strict_candidates = [path for path in candidates if path.name == SUMMARY_VARIANT_NAME]
        if strict_candidates:
            return latest_path_by_mtime(strict_candidates)
    return latest_path_by_mtime(candidates)


def find_new_output_dir(run: RunSpec, existing_output_dirs: Sequence[Path]) -> Optional[Path]:
    existing_keys = {path.resolve().as_posix() for path in existing_output_dirs}
    current_candidates = discover_complete_output_dirs(run.run_dir)
    new_candidates = [
        path
        for path in current_candidates
        if path.resolve().as_posix() not in existing_keys
    ]
    chosen = choose_output_dir(run, new_candidates)
    if chosen is not None:
        return chosen
    return choose_output_dir(run, current_candidates)


def write_run_status(run: RunSpec, payload: Dict[str, Any]) -> None:
    write_json_atomic(run.run_status_path, payload)


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
    error_message: Optional[str] = None,
    prepare_command: Optional[Sequence[str]] = None,
    prepare_return_code: Optional[int] = None,
    reused_motionbert_pkl: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "run_id": run.run_id,
        "status": status,
        "classifier_model": run.classifier_model,
        "pose_model_size": run.pose_model_size,
        "checkpoint_tag": run.checkpoint_tag,
        "variant_name": run.variant_name,
        "imgsz": run.imgsz,
        "started_at": started_at,
        "finished_at": finished_at,
        "return_code": return_code,
        "cwd": None if cwd is None else cwd.as_posix(),
        "command": None if command is None else quote_command(command),
        "prepare_command": None if prepare_command is None else quote_command(prepare_command),
        "prepare_return_code": prepare_return_code,
        "discovered_output_dir": None if discovered_output_dir is None else discovered_output_dir.as_posix(),
        "stdout_log_path": run.stdout_log_path.as_posix(),
        "stderr_log_path": run.stderr_log_path.as_posix(),
        "npz_root": run.npz_root.as_posix(),
        "checkpoint_path": run.checkpoint_path.as_posix(),
        "metrics": normalize_metrics(metrics or default_metrics()),
        "error_message": error_message,
        "reused_motionbert_pkl": reused_motionbert_pkl,
        "motionbert_pkl_path": None if run.motionbert_pkl_path is None else run.motionbert_pkl_path.as_posix(),
        "motionbert_label_map_path": None
        if run.motionbert_label_map_path is None
        else run.motionbert_label_map_path.as_posix(),
    }


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


def ensure_parent_dirs_for_run(run: RunSpec) -> None:
    ensure_dir(run.run_dir)
    if run.motionbert_pkl_path is not None:
        ensure_dir(run.motionbert_pkl_path.parent)
    if run.motionbert_label_map_path is not None:
        ensure_dir(run.motionbert_label_map_path.parent)


def normalize_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    out = df.copy()
    out.columns = [str(col).strip().lower() for col in out.columns]
    return out


def select_combined_row(
    df: "pd.DataFrame",
    *,
    classifier_model: Optional[str] = None,
) -> "pd.Series":
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


def parse_overall_metrics_table(report_html: Path) -> "pd.DataFrame":
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
                "pandas.read_html() failed and the fallback parser found no usable tables. "
                f"Errors: {' | '.join(read_html_errors)}"
            )
    for table in tables:
        norm = normalize_columns(table)
        if {"accuracy", "recall"}.issubset(set(norm.columns)):
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
    return normalize_metrics(
        {
            "accuracy": normalize_metric_0_to_100(overall_row.get("accuracy")),
            "recall": normalize_metric_0_to_100(overall_row.get("recall")),
            "macro_f1": normalize_metric_0_to_100(summary_row.get("macro_f1")),
            "fall_f1": normalize_metric_0_to_100(summary_row.get("binary_f1_fall")),
        }
    )


def parse_motionbert_metrics(output_dir: Path) -> Dict[str, Optional[float]]:
    pandas = require_pandas()
    summary_path = output_dir / "metrics_summary.csv"
    report_path = output_dir / "report.html"
    summary_df = pandas.read_csv(summary_path)
    summary_row = select_combined_row(summary_df)
    overall_df = parse_overall_metrics_table(report_path)
    overall_row = select_combined_row(overall_df)
    return normalize_metrics(
        {
            "accuracy": normalize_metric_0_to_100(overall_row.get("accuracy")),
            "recall": normalize_metric_0_to_100(overall_row.get("recall")),
            "macro_f1": normalize_metric_0_to_100(summary_row.get("macro_f1")),
            "fall_f1": normalize_metric_0_to_100(summary_row.get("bin_f1_fall")),
        }
    )


def parse_run_metrics(run: RunSpec, output_dir: Path) -> Dict[str, Optional[float]]:
    if run.classifier_model == "MotionBERT":
        return parse_motionbert_metrics(output_dir)
    return parse_eval_models_metrics(output_dir, run.classifier_model)


def check_required_inputs(paths: SweepPaths, run: RunSpec) -> List[str]:
    missing: List[str] = []
    if not run.npz_root.is_dir():
        missing.append(f"keypoints root missing: {run.npz_root.as_posix()}")
    if not run.checkpoint_path.is_file():
        missing.append(f"checkpoint missing: {run.checkpoint_path.as_posix()}")
    if run.classifier_model == "MotionBERT":
        if not paths.prepare_motionbert_script.is_file():
            missing.append(f"prepare script missing: {paths.prepare_motionbert_script.as_posix()}")
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
    else:
        if not paths.eval_models_path.is_file():
            missing.append(f"eval_models.py missing: {paths.eval_models_path.as_posix()}")
        if not eval_models_supports_weights_path(paths):
            missing.append("evaluation/eval_models.py is missing --weights-path support")
    return missing


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
        error_message=error_message,
        prepare_command=prepare_command,
        prepare_return_code=existing_status.get("prepare_return_code"),
        reused_motionbert_pkl=existing_status.get("reused_motionbert_pkl"),
    )
    write_run_status(run, updated_status)
    return build_summary_entry_from_status(run, updated_status)


def handle_missing_inputs(run: RunSpec, missing_inputs: Sequence[str]) -> Dict[str, Any]:
    ensure_parent_dirs_for_run(run)
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
        error_message="; ".join(missing_inputs),
    )
    write_run_status(run, payload)
    return build_summary_entry_from_status(run, payload)


def print_dry_run_line(
    run: RunSpec,
    *,
    command: Sequence[str],
    command_cwd: Path,
    prepare_command: Optional[Sequence[str]] = None,
) -> None:
    print(f"[dry-run] {run.run_id}")
    print(f"  keypoints: {run.keypoints_tag}")
    print(f"  checkpoint: {run.checkpoint_path.as_posix()}")
    if prepare_command is not None:
        print(f"  prepare: {quote_command(prepare_command)}")
    print(f"  eval ({command_cwd.as_posix()}): {quote_command(command)}")


def execute_run(paths: SweepPaths, config: SweepConfig, run: RunSpec) -> Dict[str, Any]:
    ensure_parent_dirs_for_run(run)
    existing_output_dirs = discover_complete_output_dirs(run.run_dir)
    started_at = now_iso()

    prepare_command: Optional[List[str]] = None
    prepare_cwd: Optional[Path] = None
    if run.classifier_model == "MotionBERT":
        if run.motionbert_repo_root is None:
            raise ValueError("MotionBERT run is missing motionbert_repo_root.")
        prepare_command = build_motionbert_prepare_command(config, paths, run)
        prepare_cwd = paths.repo_root
        command = build_motionbert_eval_command(config, run)
        command_cwd = run.motionbert_repo_root
    else:
        command = build_eval_models_command(paths, config, run)
        command_cwd = paths.repo_root

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
    metrics = default_metrics()
    error_message: Optional[str] = None

    try:
        with run.stdout_log_path.open("w", encoding="utf-8") as stdout_handle, run.stderr_log_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            if run.classifier_model == "MotionBERT":
                should_reuse_pkl = (
                    not config.force_regenerate_motionbert_pkl
                    and valid_motionbert_pkl(
                        run.motionbert_pkl_path,
                        run.motionbert_label_map_path,
                        run.npz_root,
                    )
                )
                if should_reuse_pkl:
                    reused_motionbert_pkl = True
                    stdout_handle.write(
                        f"[{now_iso()}] Reusing existing MotionBERT PKL: "
                        f"{run.motionbert_pkl_path.as_posix()}\n\n"
                    )
                    stdout_handle.flush()
                else:
                    reused_motionbert_pkl = False
                    prepare_return_code = run_subprocess_to_logs(
                        prepare_command or [],
                        cwd=prepare_cwd or paths.repo_root,
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
                    if not valid_motionbert_pkl(
                        run.motionbert_pkl_path,
                        run.motionbert_label_map_path,
                        run.npz_root,
                    ):
                        raise RuntimeError(
                            "MotionBERT dataset preparation completed but the expected PKL/label map files "
                            "were not produced or looked invalid."
                        )

            return_code = run_subprocess_to_logs(
                command,
                cwd=command_cwd,
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                title="evaluation",
            )
    except Exception as exc:
        error_message = str(exc)

    discovered_output_dir = find_new_output_dir(run, existing_output_dirs)
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
                    "Evaluation subprocess finished without a complete evaluator output directory "
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
        error_message=error_message,
        prepare_command=prepare_command,
        prepare_return_code=prepare_return_code,
        reused_motionbert_pkl=reused_motionbert_pkl,
    )
    write_run_status(run, payload)
    return build_summary_entry_from_status(run, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final classifiers on all discovered downsized YOLO11 keypoint variants."
    )
    parser.add_argument("--only-models", nargs="+", default=None)
    parser.add_argument("--only-pose-sizes", nargs="+", default=None)
    parser.add_argument("--only-imgsz", nargs="+", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--force-regenerate-motionbert-pkl", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--motionbert-device", type=str, default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_MOTIONBERT_EVAL_BATCH_SIZE,
        help=(
            "MotionBERT eval batch size. Default matches "
            "configs/action/MB_ft_UPFall_xsub.yaml (32)."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--keypoints-root", type=Path, default=DEFAULT_KEYPOINTS_ROOT)
    parser.add_argument("--classification-root", type=Path, default=DEFAULT_CLASSIFICATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--motionbert-pkl-root", type=Path, default=None)
    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> SweepPaths:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    output_root = args.output_root.expanduser()
    motionbert_pkl_root = (
        args.motionbert_pkl_root.expanduser()
        if args.motionbert_pkl_root is not None
        else output_root / "_motionbert_pkl_cache"
    )
    keypoints_root = args.keypoints_root.expanduser()
    return SweepPaths(
        script_path=script_path,
        repo_root=repo_root,
        eval_models_path=(repo_root / "evaluation" / "eval_models.py").resolve(),
        prepare_motionbert_script=(repo_root / "dataset_helpers" / "prepare_motionbert_dataset.py").resolve(),
        motionbert_code_root=(repo_root / "models" / "MotionBERT").resolve(),
        keypoints_root=keypoints_root,
        classification_root=args.classification_root.expanduser(),
        output_root=output_root,
        motionbert_pkl_root=motionbert_pkl_root,
    )


def build_config(args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        python_executable=str(args.python),
        motionbert_device=str(args.motionbert_device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        force_regenerate_motionbert_pkl=bool(args.force_regenerate_motionbert_pkl),
    )


def main() -> int:
    args = parse_args()
    paths = build_paths(args)
    config = build_config(args)

    variants, discovery_issues = discover_variants(paths)
    if not variants:
        print("No downsized keypoint variants were discovered.", file=sys.stderr)
        for issue in discovery_issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    all_runs = build_run_matrix(paths, variants)
    summary_json_path = paths.output_root / SUMMARY_JSON_NAME
    summary_entries = load_existing_summary_entries(all_runs)

    discovered_pose_sizes = tuple(sorted({variant.pose_model_size for variant in variants}, key=pose_sort_key))
    discovered_imgsz = tuple(sorted({variant.imgsz for variant in variants}, reverse=True))

    selected_models = normalize_filter_tokens(
        (normalize_model_token(token) for token in args.only_models) if args.only_models is not None else None,
        allowed=CLASSIFIER_MODELS,
        kind="model",
        transform=str,
    )
    selected_pose_sizes = normalize_filter_tokens(
        args.only_pose_sizes,
        allowed=discovered_pose_sizes,
        kind="pose size",
        transform=str.lower,
    )
    selected_imgsz = normalize_filter_tokens(
        args.only_imgsz,
        allowed=discovered_imgsz,
        kind="imgsz",
        transform=int,
    )

    selected_runs = [
        run
        for run in all_runs
        if is_selected_run(
            run,
            selected_models=selected_models,
            selected_pose_sizes=selected_pose_sizes,
            selected_imgsz=selected_imgsz,
        )
    ]

    ensure_dir(paths.output_root)
    write_summary_json(summary_json_path, paths, all_runs, summary_entries)

    print(f"Output root: {paths.output_root.as_posix()}")
    print(f"Summary JSON: {summary_json_path.as_posix()}")
    print(f"Discovered variants: {len(variants)}")
    print(f"Planned runs: {len(selected_runs)}")
    if discovery_issues:
        print("Discovery notes:")
        for issue in discovery_issues:
            print(f"  - {issue}")

    if not selected_runs:
        print("No runs matched the selected filters.")
        return 0

    completed = 0
    failed = 0
    skipped_missing = 0
    reused = 0

    for index, run in enumerate(selected_runs, start=1):
        print(
            f"[{index}/{len(selected_runs)}] {run.run_id} "
            f"-> keypoints={run.keypoints_tag}"
        )

        missing_inputs = check_required_inputs(paths, run)
        if missing_inputs:
            print("  missing inputs:")
            for item in missing_inputs:
                print(f"    - {item}")
            entry = handle_missing_inputs(run, missing_inputs)
            summary_entries[run.run_id] = entry
            skipped_missing += 1
            write_summary_json(summary_json_path, paths, all_runs, summary_entries)
            if args.stop_on_error:
                return 1
            continue

        if not args.force_rerun:
            existing = detect_completed_run(run)
            if existing is not None:
                print("  reusing completed run")
                entry = process_existing_completed_run(run)
                summary_entries[run.run_id] = entry
                completed += 1
                write_summary_json(summary_json_path, paths, all_runs, summary_entries)
                continue

        if run.classifier_model == "MotionBERT":
            prepare_command = build_motionbert_prepare_command(config, paths, run)
            eval_command = build_motionbert_eval_command(config, run)
            if args.dry_run:
                print_dry_run_line(
                    run,
                    command=eval_command,
                    command_cwd=run.motionbert_repo_root or paths.motionbert_code_root,
                    prepare_command=prepare_command,
                )
                continue
        else:
            eval_command = build_eval_models_command(paths, config, run)
            if args.dry_run:
                print_dry_run_line(
                    run,
                    command=eval_command,
                    command_cwd=paths.repo_root,
                )
                continue

        entry = execute_run(paths, config, run)
        summary_entries[run.run_id] = entry
        write_summary_json(summary_json_path, paths, all_runs, summary_entries)

        status = str(entry.get("status"))
        if status == "completed":
            completed += 1
            metrics = entry.get("metrics", {})
            if isinstance(metrics, dict):
                print(
                    "  completed "
                    f"(acc={metrics.get('accuracy')}, rec={metrics.get('recall')}, "
                    f"macro_f1={metrics.get('macro_f1')}, fall_f1={metrics.get('fall_f1')})"
                )
            else:
                print("  completed")

            run_status = read_json(run.run_status_path) or {}
            if bool(run_status.get("reused_motionbert_pkl")):
                reused += 1
        else:
            failed += 1
            print(f"  failed: {entry.get('error_message')}")
            if args.stop_on_error:
                return 1

    print(
        f"Finished: completed={completed} failed={failed} "
        f"skipped_missing={skipped_missing} motionbert_pkl_reused={reused}"
    )
    return 1 if failed or skipped_missing else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
