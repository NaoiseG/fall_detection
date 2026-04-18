#!/usr/bin/env python3
"""
Train the final temporal classifiers for each base keypoint backend.

This runner launches `python -m training.train_models` once per
    (model, pose-backend)
pair, then copies the resulting best checkpoint into the flat scratch layout
requested for the final models:

    ~/scratch/final_classification_models/<classifier_bucket>/<pose_tag>/<checkpoint>

Defaults:
  - models: paper_stgcn, cnnlstm, tcn
  - pose backends: yolo11n/s/m/l/x, alphapose, vitpose
  - subjects: train 1-12, val 13-15
  - cameras: 1 and 2
  - epochs: 300
  - skip jobs whose destination checkpoint already exists

Run from anywhere; the script resolves the Prototype root automatically.
Typical usage:

    python final_scripts/train_final_models.py

Dry-run a subset:

    python final_scripts/train_final_models.py --backbones yolo11l-pose vitpose --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


MODELS_DEFAULT: tuple[str, ...] = ("paper_stgcn", "cnnlstm", "tcn")
MODEL_OUTPUT_BUCKET: Dict[str, str] = {
    "paper_stgcn": "stgcn",
    "cnnlstm": "cnnlstm",
    "tcn": "tcn",
}
RUN_ID_RE = re.compile(r"Run ID:\s*(\S+)")


@dataclass(frozen=True)
class BackboneSpec:
    pose_tag: str
    npz_root: Path


@dataclass
class JobResult:
    pose_tag: str
    npz_root: str
    model_name: str
    output_bucket: str
    dest_checkpoint: str
    status: str
    run_id: Optional[str]
    source_run_dir: Optional[str]
    log_path: Optional[str]
    seconds: float
    command: List[str]
    note: str = ""


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_ts()}] {msg}", flush=True)


def command_string(cmd: Iterable[object]) -> str:
    return shlex.join([str(part) for part in cmd])


def prototype_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_backbone_specs(scratch_keypoints_root: Path) -> List[BackboneSpec]:
    root = scratch_keypoints_root.expanduser()
    return [
        BackboneSpec("yolo11n-pose", root / "UPFall_keypoints" / "yolo11n" / "base"),
        BackboneSpec("yolo11s-pose", root / "UPFall_keypoints" / "yolo11s" / "base"),
        BackboneSpec("yolo11m-pose", root / "UPFall_keypoints" / "yolo11m" / "base"),
        BackboneSpec("yolo11l-pose", root / "UPFall_keypoints" / "yolo11l" / "base"),
        BackboneSpec("yolo11x-pose", root / "UPFall_keypoints" / "yolo11x" / "base"),
        BackboneSpec("alphapose", root / "UPFall_keypoints_alpha" / "base"),
        BackboneSpec("vitpose", root / "UPFall_keypoints_vitpose" / "base"),
    ]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train final TCN/CNNLSTM/Paper-STGCN models for all base pose backends."
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS_DEFAULT),
        choices=sorted(MODEL_OUTPUT_BUCKET.keys()),
        help="Subset of classifier models to train.",
    )
    ap.add_argument(
        "--backbones",
        nargs="+",
        default=None,
        help="Optional subset of pose tags to train, e.g. yolo11l-pose vitpose.",
    )
    ap.add_argument(
        "--scratch-keypoints-root",
        type=Path,
        default=Path("~/scratch/keypoints"),
        help="Scratch root containing UPFall keypoint directories.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("~/scratch/final_classification_models"),
        help="Root directory where flat final checkpoints will be stored.",
    )
    ap.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch training.train_models.",
    )
    ap.add_argument("--train-subjects", type=str, default="1-12")
    ap.add_argument("--val-subjects", type=str, default="13-15")
    ap.add_argument(
        "--camera",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Camera ids passed through to training.train_models.",
    )
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and destinations without launching training.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing flat checkpoint instead of skipping it.",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed training job.",
    )
    ap.add_argument(
        "--cleanup-run-dirs",
        action="store_true",
        help="Delete the intermediate models/<model>/<run_id> directory after copying the checkpoint.",
    )
    return ap.parse_args()


def ensure_backbone_data_exists(spec: BackboneSpec) -> None:
    if not spec.npz_root.exists():
        raise FileNotFoundError(f"Missing keypoint root for {spec.pose_tag}: {spec.npz_root}")

    if not any(spec.npz_root.rglob("*.npz")):
        raise FileNotFoundError(f"No NPZ files found for {spec.pose_tag}: {spec.npz_root}")


def select_backbones(all_specs: Sequence[BackboneSpec], selected_tags: Optional[Sequence[str]]) -> List[BackboneSpec]:
    if not selected_tags:
        return list(all_specs)

    lookup = {spec.pose_tag: spec for spec in all_specs}
    missing = [tag for tag in selected_tags if tag not in lookup]
    if missing:
        raise SystemExit(
            f"Unknown --backbones value(s): {missing}. Valid: {sorted(lookup.keys())}"
        )
    return [lookup[tag] for tag in selected_tags]


def build_train_cmd(
    *,
    python_exe: str,
    model_name: str,
    backbone: BackboneSpec,
    train_subjects: str,
    val_subjects: str,
    camera_ids: Sequence[int],
    epochs: int,
) -> List[str]:
    cmd: List[str] = [
        python_exe,
        "-m",
        "training.train_models",
        "--model",
        model_name,
        "--train-subjects",
        train_subjects,
        "--val-subjects",
        val_subjects,
        "--npz-root",
        str(backbone.npz_root),
        "--camera",
        *[str(c) for c in camera_ids],
        "--label-mode",
        "center",
        "--drop-ambig-share",
        "0",
        "--T",
        "64",
        "--stride",
        "48",
        "--epochs",
        str(int(epochs)),
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
        "--selection-metric",
        "composite_fall_fbeta_macro_f1",
        "--selection-w",
        "0.7",
        "--selection-beta",
        "2.0",
        "--rare-class-boost",
        "1.5",
        "--weighted-sampler",
        "1",
        "--conf-thres",
        "0.05",
    ]
    return cmd


def detect_run_id(stdout: str) -> Optional[str]:
    match = RUN_ID_RE.search(stdout)
    return match.group(1).strip() if match else None


def detect_recent_run_dir(model_dir: Path, started_at: float) -> Optional[Path]:
    if not model_dir.exists():
        return None

    candidates: List[tuple[float, Path]] = []
    for child in model_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        if mtime >= started_at:
            candidates.append((mtime, child))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def run_command(cmd: Sequence[str], cwd: Path) -> tuple[int, str]:
    log(f"[run] {command_string(cmd)}")
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError(f"Failed to capture stdout for command: {command_string(cmd)}")

    lines: List[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)

    proc.wait()
    return int(proc.returncode), "".join(lines)


def copy_file_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2))


def cleanup_run_dir(run_dir: Path, model_name: str, repo_root: Path) -> None:
    expected_parent = (repo_root / "models" / model_name).resolve()
    resolved = run_dir.resolve()
    try:
        resolved.relative_to(expected_parent)
    except ValueError as exc:
        raise RuntimeError(f"Refusing to delete unexpected run dir: {resolved}") from exc
    shutil.rmtree(resolved)


def destination_paths(output_root: Path, model_name: str, pose_tag: str) -> tuple[str, Path, Path, Path]:
    bucket = MODEL_OUTPUT_BUCKET[model_name]
    dest_dir = output_root / bucket / pose_tag
    ckpt = dest_dir / f"{model_name}_best.pt"
    label_map = dest_dir / "label_map_fallmerged.json"
    meta = dest_dir / f"{model_name}_train_meta.json"
    return bucket, ckpt, label_map, meta


def write_summary(output_root: Path, results: Sequence[JobResult]) -> None:
    summary_json = output_root / "train_final_models_summary.json"
    summary_csv = output_root / "train_final_models_summary.csv"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "results": [asdict(result) for result in results],
    }
    write_json(summary_json, payload)

    fieldnames = list(JobResult.__dataclass_fields__.keys())
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["command"] = command_string(row["command"])
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    repo_root = prototype_root()
    output_root = args.output_root.expanduser()
    all_backbones = build_backbone_specs(args.scratch_keypoints_root)
    selected_backbones = select_backbones(all_backbones, args.backbones)

    log(f"Prototype root: {repo_root}")
    log(f"Output root: {output_root}")
    log(f"Models: {list(args.models)}")
    log(f"Backbones: {[spec.pose_tag for spec in selected_backbones]}")

    if args.dry_run:
        for spec in selected_backbones:
            if not spec.npz_root.exists():
                log(f"[dry-run] Keypoint root not found locally, skipping validation: {spec.npz_root}")
    else:
        for spec in selected_backbones:
            ensure_backbone_data_exists(spec)

    results: List[JobResult] = []
    failures = 0
    trained = 0
    skipped = 0
    total_jobs = len(args.models) * len(selected_backbones)
    job_index = 0

    for backbone in selected_backbones:
        for model_name in args.models:
            job_index += 1
            bucket, dest_ckpt, dest_label_map, dest_meta = destination_paths(
                output_root=output_root,
                model_name=model_name,
                pose_tag=backbone.pose_tag,
            )
            train_cmd = build_train_cmd(
                python_exe=args.python,
                model_name=model_name,
                backbone=backbone,
                train_subjects=args.train_subjects,
                val_subjects=args.val_subjects,
                camera_ids=args.camera,
                epochs=args.epochs,
            )

            log(
                f"[{job_index}/{total_jobs}] {model_name} on {backbone.pose_tag} "
                f"-> {dest_ckpt.as_posix()}"
            )

            if dest_ckpt.exists() and not args.force:
                skipped += 1
                note = "destination checkpoint already exists"
                log(f"Skipping: {note}")
                results.append(
                    JobResult(
                        pose_tag=backbone.pose_tag,
                        npz_root=str(backbone.npz_root),
                        model_name=model_name,
                        output_bucket=bucket,
                        dest_checkpoint=str(dest_ckpt),
                        status="skipped",
                        run_id=None,
                        source_run_dir=None,
                        log_path=None,
                        seconds=0.0,
                        command=train_cmd,
                        note=note,
                    )
                )
                continue

            if args.dry_run:
                log(f"Dry run only: {command_string(train_cmd)}")
                results.append(
                    JobResult(
                        pose_tag=backbone.pose_tag,
                        npz_root=str(backbone.npz_root),
                        model_name=model_name,
                        output_bucket=bucket,
                        dest_checkpoint=str(dest_ckpt),
                        status="dry_run",
                        run_id=None,
                        source_run_dir=None,
                        log_path=None,
                        seconds=0.0,
                        command=train_cmd,
                    )
                )
                continue

            started_at = time.time()
            returncode, stdout = run_command(train_cmd, cwd=repo_root)
            elapsed = time.time() - started_at
            run_id = detect_run_id(stdout)
            source_run_dir: Optional[Path] = None
            model_dir = repo_root / "models" / model_name
            if run_id:
                source_run_dir = model_dir / run_id
            if source_run_dir is None or not source_run_dir.exists():
                source_run_dir = detect_recent_run_dir(model_dir=model_dir, started_at=started_at)

            log_path: Optional[Path] = dest_ckpt.parent / f"{model_name}_train.log"
            write_text(log_path, stdout)

            if returncode != 0:
                failures += 1
                note = f"training command failed with rc={returncode}"
                log(f"FAILED: {note}")
                results.append(
                    JobResult(
                        pose_tag=backbone.pose_tag,
                        npz_root=str(backbone.npz_root),
                        model_name=model_name,
                        output_bucket=bucket,
                        dest_checkpoint=str(dest_ckpt),
                        status="failed",
                        run_id=run_id,
                        source_run_dir=str(source_run_dir) if source_run_dir is not None else None,
                        log_path=str(log_path),
                        seconds=elapsed,
                        command=train_cmd,
                        note=note,
                    )
                )
                if args.stop_on_error:
                    write_summary(output_root=output_root, results=results)
                    return 1
                continue

            if source_run_dir is None or not source_run_dir.exists():
                failures += 1
                note = "training finished but source run directory could not be located"
                log(f"FAILED: {note}")
                results.append(
                    JobResult(
                        pose_tag=backbone.pose_tag,
                        npz_root=str(backbone.npz_root),
                        model_name=model_name,
                        output_bucket=bucket,
                        dest_checkpoint=str(dest_ckpt),
                        status="failed",
                        run_id=run_id,
                        source_run_dir=None,
                        log_path=str(log_path),
                        seconds=elapsed,
                        command=train_cmd,
                        note=note,
                    )
                )
                if args.stop_on_error:
                    write_summary(output_root=output_root, results=results)
                    return 1
                continue

            source_ckpt = source_run_dir / f"{model_name}_best.pt"
            source_label_map = source_run_dir / "label_map_fallmerged.json"
            if not source_ckpt.exists():
                failures += 1
                note = f"expected checkpoint missing: {source_ckpt}"
                log(f"FAILED: {note}")
                results.append(
                    JobResult(
                        pose_tag=backbone.pose_tag,
                        npz_root=str(backbone.npz_root),
                        model_name=model_name,
                        output_bucket=bucket,
                        dest_checkpoint=str(dest_ckpt),
                        status="failed",
                        run_id=run_id,
                        source_run_dir=str(source_run_dir),
                        log_path=str(log_path),
                        seconds=elapsed,
                        command=train_cmd,
                        note=note,
                    )
                )
                if args.stop_on_error:
                    write_summary(output_root=output_root, results=results)
                    return 1
                continue

            copy_file_atomic(source_ckpt, dest_ckpt)
            if source_label_map.exists():
                copy_file_atomic(source_label_map, dest_label_map)

            meta_payload = {
                "generated_at": datetime.now().isoformat(),
                "model_name": model_name,
                "output_bucket": bucket,
                "pose_tag": backbone.pose_tag,
                "npz_root": str(backbone.npz_root),
                "run_id": run_id,
                "source_run_dir": str(source_run_dir),
                "source_checkpoint": str(source_ckpt),
                "dest_checkpoint": str(dest_ckpt),
                "train_log": str(log_path),
                "train_command": train_cmd,
                "python_executable": args.python,
                "seconds": elapsed,
            }
            write_json(dest_meta, meta_payload)

            if args.cleanup_run_dirs:
                cleanup_run_dir(run_dir=source_run_dir, model_name=model_name, repo_root=repo_root)

            trained += 1
            log("Completed and copied final checkpoint.")
            results.append(
                JobResult(
                    pose_tag=backbone.pose_tag,
                    npz_root=str(backbone.npz_root),
                    model_name=model_name,
                    output_bucket=bucket,
                    dest_checkpoint=str(dest_ckpt),
                    status="trained",
                    run_id=run_id,
                    source_run_dir=str(source_run_dir),
                    log_path=str(log_path),
                    seconds=elapsed,
                    command=train_cmd,
                )
            )

    try:
        write_summary(output_root=output_root, results=results)
        summary_note = f"summary={output_root / 'train_final_models_summary.json'}"
    except Exception as exc:
        if args.dry_run:
            summary_note = f"summary_not_written={exc}"
            log(f"[dry-run] Could not write summary files: {exc}")
        else:
            raise

    log(
        f"Finished: trained={trained} skipped={skipped} failed={failures} "
        f"{summary_note}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
