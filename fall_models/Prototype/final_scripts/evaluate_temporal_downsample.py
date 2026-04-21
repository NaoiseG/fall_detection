#!/usr/bin/env python3
"""evaluate_temporal_downsample.py

Evaluate cnnlstm, paper_stgcn and MotionBERT at k=1, k=2 and k=3 using
yolo11x-pose keypoints, under both temporal downsampling strategies:

  shrink      — take every k-th frame, shrink window T accordingly
  interpolate — take every k-th frame, linearly interpolate back to full T

Outputs per run go under:
  <output-root>/<mode>/<model>/k<k>/

Two summary JSON files are written to <output-root>:
  results_interpolate.json
  results_shrink.json

Completed runs are detected by the presence of metrics_summary.csv (for
cnnlstm/paper_stgcn) or a top1 entry in the MotionBERT summary JSON, and
skipped automatically unless --force is passed.

Usage (from Prototype/):
  python -m final_scripts.evaluate_temporal_downsample [options]
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSIFIER_MODELS: Tuple[str, ...] = ("cnnlstm", "paper_stgcn", "MotionBERT")
K_VALUES: Tuple[int, ...] = (1, 2, 3)
MODES: Tuple[str, ...] = ("interpolate", "shrink")

TEST_SUBJECTS = "16-17"
TRAIN_SUBJECTS = TEST_SUBJECTS
VAL_SUBJECTS = "13-15"
CAMERAS: Tuple[int, ...] = (1, 2)
WINDOW_T = 64
WINDOW_STRIDE = 48
LABEL_MODE = "center"

CHECKPOINT_TAG = "yolo11x-pose"
DEFAULT_NPZ_ROOT = Path("/home/people/21376026/scratch/keypoints/UPFall_keypoints/yolo11x/base")
DEFAULT_CLASSIFICATION_ROOT = Path("/home/people/21376026/scratch/final_classification_models")
DEFAULT_OUTPUT_ROOT = Path("/home/people/21376026/scratch/final_results/temporal_downsampling")
DEFAULT_MOTIONBERT_PKL_ROOT = Path("/home/people/21376026/scratch/final_results/motionbert_pkls/temporal_downsample")

MODEL_BUCKETS: Dict[str, str] = {
    "cnnlstm": "cnnlstm",
    "paper_stgcn": "stgcn",
}

SUMMARY_VARIANT_NAME = "strict_label_mode"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def quote_cmd(cmd: Sequence[str]) -> str:
    return shlex.join([str(p) for p in cmd])


def write_json_atomic(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_log_tail(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-max_lines:]).strip()
    except OSError:
        return ""


def run_subprocess(cmd: Sequence[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    ensure_dir(stdout_path.parent)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        header = f"[{now_iso()}] cwd={cwd}\ncommand={quote_cmd(cmd)}\n\n"
        out.write(header)
        err.write(header)
        out.flush()
        err.flush()
        rc = subprocess.run(
            [str(p) for p in cmd],
            cwd=str(cwd),
            stdout=out,
            stderr=err,
            check=False,
            env=os.environ.copy(),
        ).returncode
        footer = f"\n[{now_iso()}] return_code={rc}\n"
        out.write(footer)
        err.write(footer)
    return rc


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def get_checkpoint_path(classification_root: Path, model: str) -> Path:
    if model == "MotionBERT":
        return (
            classification_root / "MotionBERT" / CHECKPOINT_TAG
            / "FT_MB_release_MB_ft_UPFall_xsub" / "best_epoch.bin"
        )
    bucket = MODEL_BUCKETS[model]
    return classification_root / bucket / CHECKPOINT_TAG / f"{model}_best.pt"


def get_motionbert_pkl_paths(pkl_root: Path) -> Tuple[Path, Path]:
    stem = (
        f"upfall_motionbert_yolo11x_base"
        f"_cam_{'_'.join(str(c) for c in CAMERAS)}"
        f"_train_{TRAIN_SUBJECTS.replace('-', '_')}"
        f"_val_{VAL_SUBJECTS.replace('-', '_')}"
        f"_label_{LABEL_MODE}"
        f"_win_{WINDOW_T}"
        f"_step_{WINDOW_STRIDE}"
    )
    return pkl_root / f"{stem}.pkl", pkl_root / f"{stem}_label_map.json"


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    model: str
    k: int
    mode: str
    run_dir: Path
    checkpoint_path: Path
    npz_root: Path
    motionbert_pkl_path: Optional[Path] = None
    motionbert_label_map_path: Optional[Path] = None

    @property
    def run_id(self) -> str:
        return f"{self.model}__k{self.k}__{self.mode}"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "run_status.json"

    @property
    def stdout_path(self) -> Path:
        return self.run_dir / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.run_dir / "stderr.log"


def build_run_matrix(
    npz_root: Path,
    classification_root: Path,
    output_root: Path,
    pkl_root: Path,
) -> List[RunSpec]:
    runs: List[RunSpec] = []
    pkl_path, label_map_path = get_motionbert_pkl_paths(pkl_root)
    for mode in MODES:
        for model in CLASSIFIER_MODELS:
            ckpt = get_checkpoint_path(classification_root, model)
            for k in K_VALUES:
                run_dir = output_root / mode / model / f"k{k}"
                spec = RunSpec(
                    model=model,
                    k=k,
                    mode=mode,
                    run_dir=run_dir,
                    checkpoint_path=ckpt,
                    npz_root=npz_root,
                )
                if model == "MotionBERT":
                    spec.motionbert_pkl_path = pkl_path
                    spec.motionbert_label_map_path = label_map_path
                runs.append(spec)
    return runs


# ---------------------------------------------------------------------------
# Completion detection
# ---------------------------------------------------------------------------

def has_eval_models_output(run_dir: Path) -> bool:
    variant_dir = run_dir / SUMMARY_VARIANT_NAME
    return (variant_dir / "metrics_summary.csv").is_file()


def has_motionbert_output(run_dir: Path) -> bool:
    status = read_json(run_dir / "run_status.json")
    if not isinstance(status, dict):
        return False
    return str(status.get("status", "")) == "completed" and status.get("return_code") == 0


def is_completed(run: RunSpec) -> bool:
    if run.model == "MotionBERT":
        return has_motionbert_output(run.run_dir)
    return has_eval_models_output(run.run_dir)


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def build_motionbert_prepare_command(
    python: str,
    paths: "SweepPaths",
    run: RunSpec,
) -> List[str]:
    assert run.motionbert_pkl_path is not None
    assert run.motionbert_label_map_path is not None
    return [
        python,
        paths.prepare_motionbert_script.as_posix(),
        "--outputs-npz-root", run.npz_root.as_posix(),
        "--camera", *(str(c) for c in CAMERAS),
        "--train-subjects", TRAIN_SUBJECTS,
        "--val-subjects", VAL_SUBJECTS,
        "--out-pkl", run.motionbert_pkl_path.as_posix(),
        "--out-label-map", run.motionbert_label_map_path.as_posix(),
        "--label-mode", LABEL_MODE,
        "--win-len", str(WINDOW_T),
        "--win-step", str(WINDOW_STRIDE),
    ]


def build_eval_models_command(
    python: str,
    paths: "SweepPaths",
    run: RunSpec,
    device: str,
    batch_size: int,
    num_workers: int,
) -> List[str]:
    cmd = [
        python, "-m", "evaluation.eval_models",
        "--models", run.model,
        "--test-subjects", TEST_SUBJECTS,
        "--npz-root", run.npz_root.as_posix(),
        "--camera", *(str(c) for c in CAMERAS),
        "--label-mode", LABEL_MODE,
        "--drop-ambig-share", "0",
        "--T", str(WINDOW_T),
        "--stride", str(WINDOW_STRIDE),
        "--normalize", "1",
        "--normalize-mode", "paper_rp",
        "--rp-center-mode", "pixel",
        "--rp-img-w", "640",
        "--rp-img-h", "480",
        "--missing-mode", "zeros_only",
        "--interp-mode", "paper_group_linear",
        "--interp-group", "100",
        "--weights-path", run.checkpoint_path.as_posix(),
        "--out-dir", run.run_dir.as_posix(),
        "--device", device,
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
    ]
    if run.k > 1:
        cmd += ["--frame-step", str(run.k), "--temporal-downsample-mode", run.mode]
    return cmd


def build_motionbert_eval_command(
    python: str,
    paths: "SweepPaths",
    run: RunSpec,
    device: str,
    batch_size: int,
    num_workers: int,
) -> List[str]:
    assert run.motionbert_pkl_path is not None
    cmd = [
        python,
        "eval_motionbert_action.py",
        "--config", "configs/action/MB_ft_UPFall_xsub.yaml",
        "--checkpoint", run.checkpoint_path.as_posix(),
        "--subjects", TEST_SUBJECTS,
        "--camera", *(str(c) for c in CAMERAS),
        "--out-dir", run.run_dir.as_posix(),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--device", device,
        "--ckpt-metric", "composite",
        "--ckpt-w", "0.7",
        "--ckpt-beta", "2.0",
        "--fall-class-idx", "0",
        "--fall-class-ids", "0",
        "--split-base", "xsub_train",
        "--no-class-weights",
        "--data-pkl", run.motionbert_pkl_path.as_posix(),
    ]
    if run.k > 1:
        cmd += ["--frame-step", str(run.k), "--temporal-downsample-mode", run.mode]
    return cmd


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

def extract_eval_models_metrics(run: RunSpec) -> Dict[str, Any]:
    variant_dir = run.run_dir / SUMMARY_VARIANT_NAME
    csv_path = variant_dir / "metrics_summary.csv"
    if not csv_path.is_file():
        return {}
    try:
        import csv as _csv
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        # prefer combined split row; fall back to first row
        combined = [r for r in rows if str(r.get("eval_split", "")).strip().lower() == "combined"]
        row = combined[0] if combined else (rows[0] if rows else {})

        def _f(key: str) -> Optional[float]:
            for k in row:
                if k.strip().lower() == key:
                    try:
                        return float(row[k])
                    except (ValueError, TypeError):
                        return None
            return None

        return {
            "accuracy": _f("accuracy"),
            "recall": _f("recall"),
            "precision": _f("precision"),
            "macro_f1": _f("macro_f1"),
            "fall_f1": _f("fall_f1"),
        }
    except Exception:
        return {}


def extract_motionbert_metrics(run: RunSpec) -> Dict[str, Any]:
    status = read_json(run.run_dir / "run_status.json")
    if not isinstance(status, dict):
        return {}
    m = status.get("metrics", {})
    if not isinstance(m, dict):
        return {}
    return {
        "accuracy": m.get("top1"),
        "balanced_acc": m.get("balanced_acc"),
        "macro_f1": m.get("macro_f1"),
        "fall_fbeta": m.get("fall_fbeta"),
        "fall_recall": m.get("fall_recall"),
    }


def extract_metrics(run: RunSpec) -> Dict[str, Any]:
    if run.model == "MotionBERT":
        return extract_motionbert_metrics(run)
    return extract_eval_models_metrics(run)


# ---------------------------------------------------------------------------
# Path container
# ---------------------------------------------------------------------------

@dataclass
class SweepPaths:
    repo_root: Path
    eval_models_path: Path
    prepare_motionbert_script: Path
    motionbert_code_root: Path


def resolve_paths(repo_root: Path) -> SweepPaths:
    return SweepPaths(
        repo_root=repo_root,
        eval_models_path=repo_root / "evaluation" / "eval_models.py",
        prepare_motionbert_script=repo_root / "dataset_helpers" / "prepare_motionbert_dataset.py",
        motionbert_code_root=repo_root / "models" / "MotionBERT",
    )


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def run_eval_models(
    python: str,
    paths: SweepPaths,
    run: RunSpec,
    device: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[bool, str]:
    ensure_dir(run.run_dir)
    cmd = build_eval_models_command(python, paths, run, device, batch_size, num_workers)
    print(f"  CMD: {quote_cmd(cmd)}", flush=True)
    rc = run_subprocess(cmd, cwd=paths.repo_root, stdout_path=run.stdout_path, stderr_path=run.stderr_path)
    if rc != 0:
        tail = read_log_tail(run.stderr_path)
        return False, f"eval_models returned rc={rc}\n{tail}"
    return True, ""


def run_motionbert(
    python: str,
    paths: SweepPaths,
    run: RunSpec,
    device: str,
    batch_size: int,
    num_workers: int,
    force_prepare: bool,
) -> Tuple[bool, str]:
    assert run.motionbert_pkl_path is not None
    ensure_dir(run.run_dir)
    ensure_dir(run.motionbert_pkl_path.parent)

    # Prepare pkl if missing
    pkl_ok = (
        run.motionbert_pkl_path.is_file() and run.motionbert_pkl_path.stat().st_size > 0
        and run.motionbert_label_map_path is not None
        and run.motionbert_label_map_path.is_file()
    )
    if force_prepare or not pkl_ok:
        prep_cmd = build_motionbert_prepare_command(python, paths, run)
        print(f"  PREPARE CMD: {quote_cmd(prep_cmd)}", flush=True)
        prep_stdout = run.run_dir / "prepare_stdout.log"
        prep_stderr = run.run_dir / "prepare_stderr.log"
        rc = run_subprocess(prep_cmd, cwd=paths.repo_root, stdout_path=prep_stdout, stderr_path=prep_stderr)
        if rc != 0:
            tail = read_log_tail(prep_stderr)
            return False, f"prepare_motionbert returned rc={rc}\n{tail}"
    else:
        print(f"  Reusing existing pkl: {run.motionbert_pkl_path.as_posix()}", flush=True)

    cmd = build_motionbert_eval_command(python, paths, run, device, batch_size, num_workers)
    print(f"  CMD: {quote_cmd(cmd)}", flush=True)
    rc = run_subprocess(
        cmd,
        cwd=paths.motionbert_code_root,
        stdout_path=run.stdout_path,
        stderr_path=run.stderr_path,
    )
    if rc != 0:
        tail = read_log_tail(run.stderr_path)
        return False, f"eval_motionbert returned rc={rc}\n{tail}"

    # Write a simple run_status.json so completion can be detected next time
    status_payload = {
        "run_id": run.run_id,
        "status": "completed",
        "return_code": 0,
        "finished_at": now_iso(),
        "metrics": extract_motionbert_metrics(run),
    }
    write_json_atomic(run.status_path, status_payload)
    return True, ""


def execute_run(
    run: RunSpec,
    python: str,
    paths: SweepPaths,
    device: str,
    batch_size: int,
    num_workers: int,
    force_prepare: bool,
) -> Tuple[bool, str]:
    if run.model == "MotionBERT":
        return run_motionbert(python, paths, run, device, batch_size, num_workers, force_prepare)
    return run_eval_models(python, paths, run, device, batch_size, num_workers)


# ---------------------------------------------------------------------------
# Summary JSON construction
# ---------------------------------------------------------------------------

def build_result_entry(run: RunSpec, status: str, error: str = "") -> Dict[str, Any]:
    metrics = extract_metrics(run) if status == "completed" else {}
    return {
        "run_id": run.run_id,
        "status": status,
        "model": run.model,
        "k": run.k,
        "mode": run.mode,
        "checkpoint_path": run.checkpoint_path.as_posix(),
        "npz_root": run.npz_root.as_posix(),
        "run_dir": run.run_dir.as_posix(),
        "error": error,
        "metrics": metrics,
    }


def write_summary_jsons(
    output_root: Path,
    runs: List[RunSpec],
    results: Dict[str, Dict[str, Any]],
) -> None:
    for mode in MODES:
        mode_entries = [
            results.get(r.run_id, build_result_entry(r, "pending"))
            for r in runs
            if r.mode == mode
        ]
        payload = {
            "mode": mode,
            "generated_at": now_iso(),
            "k_values": list(K_VALUES),
            "models": list(CLASSIFIER_MODELS),
            "test_subjects": TEST_SUBJECTS,
            "window": {"T": WINDOW_T, "stride": WINDOW_STRIDE, "label_mode": LABEL_MODE},
            "keypoints": CHECKPOINT_TAG,
            "total_runs": len(mode_entries),
            "results": mode_entries,
        }
        out_path = output_root / f"results_{mode}.json"
        write_json_atomic(out_path, payload)
        print(f"Wrote summary: {out_path.as_posix()}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-root", type=str, default=str(DEFAULT_NPZ_ROOT),
                    help=f"yolo11x keypoint NPZ root (default: {DEFAULT_NPZ_ROOT})")
    ap.add_argument("--classification-root", type=str, default=str(DEFAULT_CLASSIFICATION_ROOT),
                    help=f"Classification model checkpoint root (default: {DEFAULT_CLASSIFICATION_ROOT})")
    ap.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT),
                    help=f"Results output root (default: {DEFAULT_OUTPUT_ROOT})")
    ap.add_argument("--motionbert-pkl-root", type=str, default=str(DEFAULT_MOTIONBERT_PKL_ROOT),
                    help="Root for MotionBERT prepared pkl files")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device string passed to eval scripts (default: cuda)")
    ap.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    ap.add_argument("--motionbert-batch-size", type=int, default=None,
                    help="Batch size for MotionBERT (default: same as --batch-size)")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python executable to use for subprocesses")
    ap.add_argument("--force", action="store_true",
                    help="Re-run all evaluations even if output already exists")
    ap.add_argument("--force-prepare", action="store_true",
                    help="Re-prepare MotionBERT pkl even if it already exists")
    ap.add_argument("--models", nargs="+", default=None,
                    choices=list(CLASSIFIER_MODELS),
                    help="Restrict to specific models (default: all)")
    ap.add_argument("--k", nargs="+", type=int, default=None,
                    help="Restrict to specific k values (default: 1 2 3)")
    ap.add_argument("--modes", nargs="+", default=None,
                    choices=list(MODES),
                    help="Restrict to specific modes (default: both)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    npz_root = Path(args.npz_root)
    classification_root = Path(args.classification_root)
    output_root = Path(args.output_root)
    pkl_root = Path(args.motionbert_pkl_root)
    mb_batch = args.motionbert_batch_size if args.motionbert_batch_size is not None else args.batch_size

    repo_root = Path(__file__).resolve().parents[1]
    paths = resolve_paths(repo_root)

    all_runs = build_run_matrix(npz_root, classification_root, output_root, pkl_root)

    # Apply filters
    selected_models = set(args.models) if args.models else set(CLASSIFIER_MODELS)
    selected_ks = set(args.k) if args.k else set(K_VALUES)
    selected_modes = set(args.modes) if args.modes else set(MODES)
    runs = [
        r for r in all_runs
        if r.model in selected_models and r.k in selected_ks and r.mode in selected_modes
    ]

    print(f"Temporal downsampling evaluation — {len(runs)} run(s) planned", flush=True)
    print(f"  Output root : {output_root}", flush=True)
    print(f"  NPZ root    : {npz_root}", flush=True)
    print(f"  Checkpoints : {classification_root}", flush=True)
    print(f"  Models      : {sorted(selected_models)}", flush=True)
    print(f"  K values    : {sorted(selected_ks)}", flush=True)
    print(f"  Modes       : {sorted(selected_modes)}", flush=True)

    # Check checkpoints exist
    for run in runs:
        if not run.checkpoint_path.exists():
            print(f"[WARN] Checkpoint not found for {run.run_id}: {run.checkpoint_path}", flush=True)

    results: Dict[str, Dict[str, Any]] = {}

    n_skip = n_ok = n_fail = 0
    for run in runs:
        print(f"\n{'='*60}", flush=True)
        print(f"Run: {run.run_id}", flush=True)
        print(f"  model={run.model}  k={run.k}  mode={run.mode}", flush=True)
        print(f"  run_dir={run.run_dir}", flush=True)

        # k=1: both modes are identical — record mode=none in both but only run once
        if run.k == 1 and run.mode == "shrink":
            # Results are identical to interpolate k=1; copy from there if available
            interp_run_id = f"{run.model}__k1__interpolate"
            if interp_run_id in results and results[interp_run_id]["status"] == "completed":
                entry = dict(results[interp_run_id])
                entry["run_id"] = run.run_id
                entry["mode"] = "shrink"
                entry["note"] = "k=1: identical to interpolate, result copied"
                results[run.run_id] = entry
                n_skip += 1
                print("  k=1 shrink: identical to interpolate result, skipping re-run.", flush=True)
                continue

        if not args.force and is_completed(run):
            print("  Already completed — skipping. (use --force to re-run)", flush=True)
            results[run.run_id] = build_result_entry(run, "completed")
            n_skip += 1
            continue

        ensure_dir(run.run_dir)
        started = now_iso()
        try:
            batch = mb_batch if run.model == "MotionBERT" else args.batch_size
            ok, err = execute_run(
                run, args.python, paths, args.device, batch, args.num_workers, args.force_prepare
            )
        except Exception:
            err = traceback.format_exc()
            ok = False

        if ok:
            print(f"  DONE: {run.run_id}", flush=True)
            results[run.run_id] = build_result_entry(run, "completed")
            n_ok += 1
        else:
            print(f"  FAILED: {run.run_id}\n  {err[:500]}", flush=True)
            results[run.run_id] = build_result_entry(run, "failed", error=err)
            n_fail += 1

        # Write summaries after every run so partial progress is preserved
        write_summary_jsons(output_root, all_runs, results)

    # Fill any runs not yet in results (e.g. filtered out but needed for summaries)
    for run in all_runs:
        if run.run_id not in results:
            results[run.run_id] = build_result_entry(run, "pending")

    write_summary_jsons(output_root, all_runs, results)

    print(f"\n{'='*60}", flush=True)
    print(f"Finished. completed={n_ok}  skipped={n_skip}  failed={n_fail}", flush=True)
    print(f"Summaries written to: {output_root}", flush=True)

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
