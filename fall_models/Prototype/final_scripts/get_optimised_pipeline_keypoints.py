#!/usr/bin/env python3
"""
Generate and clean the selected optimised YOLO pipeline keypoints for UP-Fall.

This script mirrors the generate-then-fix flow used by
`dataset_helpers/get_img_downsized_keypoints.sh`, but targets the specific
TensorRT engines consumed by `final_scripts/eval_optimised_pipelines.py`:

  - yolo11m-pose/fp32_576
  - yolo11l-pose/fp32_576
  - yolo11x-pose/fp16_576
  - yolo11x-pose/fp32_576

For each variant, the script:
  1. Generates missing keypoints with `dataset_helpers.get_keypoints_files`
  2. Runs `dataset_helpers/fix_bad_keypoints.sh`
  3. Writes `.fixed.ok` only after cleanup succeeds

Relative paths are resolved from the Prototype directory so the script can be
run from anywhere.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


SUBJECTS_DEFAULT = "1-17"
UPFALL_ROOT_DEFAULT = Path("/home/jetson/NaoiseG/fall_detection/Datasets/UPFall")
OUTPUT_BASE_DEFAULT = Path("/home/jetson/NaoiseG/fall_detection/Datasets/UPFall_keypoints_img_downsize")
MODEL_ROOT_DEFAULT = Path("/home/jetson/NaoiseG/fall_detection/pose_models/img_downsized")
CAMERA_RUNS = (
    {"camera": 2, "lock_settings": "strict_lock"},
    {"camera": 1, "lock_settings": "default"},
)


@dataclass(frozen=True)
class VariantSpec:
    pose_dir_name: str
    variant_name: str
    model_filename: str
    imgsz: int

    @property
    def output_relative_dir(self) -> Path:
        return Path(self.pose_dir_name) / self.variant_name


VARIANT_SPECS: tuple[VariantSpec, ...] = (
    VariantSpec(
        pose_dir_name="yolo11m-pose",
        variant_name="fp32_576",
        model_filename="yolo11m-pose_imgsz576_fp32.engine",
        imgsz=576,
    ),
    VariantSpec(
        pose_dir_name="yolo11l-pose",
        variant_name="fp32_576",
        model_filename="yolo11l-pose_imgsz576_fp32.engine",
        imgsz=576,
    ),
    VariantSpec(
        pose_dir_name="yolo11x-pose",
        variant_name="fp16_576",
        model_filename="yolo11x-pose_imgsz576_fp16.engine",
        imgsz=576,
    ),
    VariantSpec(
        pose_dir_name="yolo11x-pose",
        variant_name="fp32_576",
        model_filename="yolo11x-pose_imgsz576_fp32.engine",
        imgsz=576,
    ),
)


@dataclass(frozen=True)
class ResolvedVariant:
    spec: VariantSpec
    model_path: Path
    output_root: Path

    @property
    def display_name(self) -> str:
        return f"{self.spec.pose_dir_name}/{self.spec.variant_name}"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def format_command(cmd: Sequence[str | Path]) -> str:
    return shlex.join(str(part) for part in cmd)


def parse_subjects_arg(value: str) -> List[int]:
    subjects: List[int] = []
    chunks = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not chunks:
        raise argparse.ArgumentTypeError("Subjects cannot be empty.")

    for chunk in chunks:
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            try:
                start = int(start_text.strip())
                end = int(end_text.strip())
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid range '{chunk}'. Subject IDs must be integers."
                ) from exc
            if start <= 0 or end <= 0:
                raise argparse.ArgumentTypeError("Subject IDs must be positive integers.")
            if end < start:
                raise argparse.ArgumentTypeError(
                    f"Invalid range '{chunk}'. End must be >= start."
                )
            subjects.extend(range(start, end + 1))
            continue

        try:
            subject_id = int(chunk)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid subject '{chunk}'. Subject IDs must be integers."
            ) from exc
        if subject_id <= 0:
            raise argparse.ArgumentTypeError("Subject IDs must be positive integers.")
        subjects.append(subject_id)

    return sorted(set(subjects))


def resolve_from_prototype_root(path_value: Path, prototype_root: Path) -> Path:
    expanded = path_value.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (prototype_root / expanded).resolve()


def find_camera_folders_subjects(root: Path, camera: int, subjects: Sequence[int]) -> List[Path]:
    folders: List[Path] = []
    for subject_id in subjects:
        subject_root = root / f"Subject{subject_id}"
        if not subject_root.is_dir():
            continue
        folders.extend(
            path
            for path in subject_root.rglob(f"*Camera{camera}")
            if path.is_dir()
        )
    return sorted(set(folders), key=lambda path: path.as_posix())


def find_expected_relative_npz_paths(
    upfall_root: Path,
    camera: int,
    subjects: Sequence[int],
) -> List[Path]:
    expected_paths: List[Path] = []
    for frames_dir in find_camera_folders_subjects(upfall_root, camera, subjects):
        trial_dir = frames_dir.parent
        if not any(trial_dir.glob("*Features1&0.5.csv")):
            continue
        expected_paths.append(frames_dir.relative_to(upfall_root) / "keypoints.npz")
    return sorted(set(expected_paths), key=lambda path: path.as_posix())


def completion_counts(output_root: Path, expected_relative_npz_paths: Iterable[Path]) -> tuple[int, int, int]:
    expected_list = list(expected_relative_npz_paths)
    existing = sum(1 for rel_path in expected_list if (output_root / rel_path).is_file())
    missing = len(expected_list) - existing
    return len(expected_list), existing, missing


def build_generate_command(
    *,
    python_bin: str,
    subjects: str,
    camera: int,
    lock_settings: str,
    upfall_root: Path,
    output_root: Path,
    model_path: Path,
    imgsz: int,
) -> list[str]:
    return [
        python_bin,
        "-m",
        "dataset_helpers.get_keypoints_files",
        "--subjects",
        subjects,
        "--camera",
        str(camera),
        "--lock-settings",
        lock_settings,
        "--upfall-root",
        str(upfall_root),
        "--output-root",
        str(output_root),
        "--model-path",
        str(model_path),
        "--imgsz",
        str(imgsz),
    ]


def build_fix_command(
    *,
    bash_bin: str,
    fix_script: Path,
    keypoints_root: Path,
    upfall_root: Path,
    model_path: Path,
    imgsz: int,
    subjects: str,
) -> list[str]:
    return [
        bash_bin,
        str(fix_script),
        "--keypoints-root",
        str(keypoints_root),
        "--upfall-root",
        str(upfall_root),
        "--pose-backend",
        "yolo",
        "--model-path",
        str(model_path),
        "--imgsz",
        str(imgsz),
        "--subjects",
        subjects,
        "--camera1-lock-settings",
        "default",
        "--camera2-lock-settings",
        "strict_lock",
    ]


def run_command(cmd: Sequence[str], cwd: Path, dry_run: bool) -> int:
    log(f"CMD: {format_command(cmd)}")
    if dry_run:
        return 0
    completed = subprocess.run([str(part) for part in cmd], cwd=str(cwd), check=False)
    return int(completed.returncode)


def write_fixed_marker(
    *,
    marker_path: Path,
    subjects: str,
    variant: ResolvedVariant,
    upfall_root: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    marker_path.write_text(
        "\n".join(
            [
                f"completed_at={timestamp()}",
                f"subjects={subjects}",
                f"model={variant.spec.pose_dir_name}",
                f"variant={variant.spec.variant_name}",
                f"model_path={variant.model_path}",
                f"imgsz={variant.spec.imgsz}",
                f"upfall_root={upfall_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def resolve_variants(model_root: Path, output_base: Path) -> list[ResolvedVariant]:
    resolved: list[ResolvedVariant] = []
    for spec in VARIANT_SPECS:
        model_path = (model_root / spec.model_filename).resolve()
        output_root = (output_base / spec.output_relative_dir).resolve()
        resolved.append(
            ResolvedVariant(
                spec=spec,
                model_path=model_path,
                output_root=output_root,
            )
        )
    return resolved


def run_variant(
    *,
    variant: ResolvedVariant,
    subjects_arg: str,
    expected_all_paths: Sequence[Path],
    upfall_root: Path,
    prototype_root: Path,
    fix_script: Path,
    python_bin: str,
    bash_bin: str,
    force_fix: bool,
    dry_run: bool,
) -> None:
    marker_path = variant.output_root / ".fixed.ok"
    expected, existing, missing = completion_counts(variant.output_root, expected_all_paths)

    log(f"Processing {variant.display_name}")
    log(f"Model: {variant.model_path}")
    log(f"Output: {variant.output_root}")
    log(f"Current outputs: {existing}/{expected} expected keypoints present ({missing} missing).")

    if not variant.model_path.is_file():
        raise FileNotFoundError(f"Pose model weights not found: {variant.model_path}")

    if not dry_run:
        variant.output_root.mkdir(parents=True, exist_ok=True)

    if expected == 0 or missing != 0:
        if marker_path.exists() and not dry_run:
            marker_path.unlink()

        log(f"Generating missing native keypoints for {variant.display_name}.")
        for camera_cfg in CAMERA_RUNS:
            cmd = build_generate_command(
                python_bin=python_bin,
                subjects=subjects_arg,
                camera=camera_cfg["camera"],
                lock_settings=camera_cfg["lock_settings"],
                upfall_root=upfall_root,
                output_root=variant.output_root,
                model_path=variant.model_path,
                imgsz=variant.spec.imgsz,
            )
            returncode = run_command(cmd, prototype_root, dry_run)
            if returncode != 0:
                raise RuntimeError(
                    f"Generation failed for {variant.display_name} "
                    f"(camera {camera_cfg['camera']}, exit code {returncode})."
                )
    else:
        log("Native keypoint generation already complete; skipping generation commands.")

    expected_after, existing_after, missing_after = completion_counts(variant.output_root, expected_all_paths)
    log(
        f"After generation: {existing_after}/{expected_after} expected keypoints present "
        f"({missing_after} missing)."
    )

    if not dry_run and (expected_after == 0 or missing_after != 0):
        raise RuntimeError(f"Generation incomplete for {variant.display_name}.")

    if marker_path.exists() and not force_fix:
        log(f"Fix marker exists; skipping cleanup for {variant.display_name}.")
        return

    log(f"Running fix_bad_keypoints.sh for {variant.display_name}.")
    fix_cmd = build_fix_command(
        bash_bin=bash_bin,
        fix_script=fix_script,
        keypoints_root=variant.output_root,
        upfall_root=upfall_root,
        model_path=variant.model_path,
        imgsz=variant.spec.imgsz,
        subjects=subjects_arg,
    )
    returncode = run_command(fix_cmd, prototype_root, dry_run)
    if returncode != 0:
        raise RuntimeError(f"Cleanup failed for {variant.display_name} (exit code {returncode}).")

    write_fixed_marker(
        marker_path=marker_path,
        subjects=subjects_arg,
        variant=variant,
        upfall_root=upfall_root,
        dry_run=dry_run,
    )
    log(f"Finished {variant.display_name}.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and fix the selected optimised YOLO pipeline keypoints for UP-Fall."
    )
    parser.add_argument(
        "--subjects",
        default=SUBJECTS_DEFAULT,
        help=(
            "Subjects to process. Examples: 1-17 | 16-17 | 1-3,7 "
            f"(default: {SUBJECTS_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--upfall-root",
        type=Path,
        default=UPFALL_ROOT_DEFAULT,
        help="UP-Fall root. Relative paths are resolved from the Prototype directory.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=OUTPUT_BASE_DEFAULT,
        help="Base output directory for the optimised pipeline keypoints.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_ROOT_DEFAULT,
        help="Directory containing the optimised TensorRT engines.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used for keypoint generation.",
    )
    parser.add_argument(
        "--bash-bin",
        default="bash",
        help="Bash executable used for fix_bad_keypoints.sh.",
    )
    parser.add_argument(
        "--force-fix",
        action="store_true",
        help="Re-run fix_bad_keypoints.sh even when .fixed.ok already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them or writing markers.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    prototype_root = Path(__file__).resolve().parents[1]
    fix_script = (prototype_root / "dataset_helpers" / "fix_bad_keypoints.sh").resolve()
    subjects = parse_subjects_arg(args.subjects)
    upfall_root = resolve_from_prototype_root(args.upfall_root, prototype_root)
    output_base = resolve_from_prototype_root(args.output_base, prototype_root)
    model_root = resolve_from_prototype_root(args.model_root, prototype_root)

    log("Starting optimised pipeline keypoint generation")
    log(f"Prototype root: {prototype_root}")
    log(f"Python:         {args.python_bin}")
    log(f"Bash:           {args.bash_bin}")
    log(f"Subjects:       {args.subjects}")
    log(f"UP-Fall root:   {upfall_root}")
    log(f"Model root:     {model_root}")
    log(f"Output base:    {output_base}")
    log(f"Force fix:      {args.force_fix}")
    log(f"Dry run:        {args.dry_run}")

    if not upfall_root.is_dir():
        raise FileNotFoundError(f"UP-Fall root not found: {upfall_root}")
    if not model_root.is_dir():
        raise FileNotFoundError(f"Model root not found: {model_root}")
    if not fix_script.is_file():
        raise FileNotFoundError(f"Fix script not found: {fix_script}")

    if not args.dry_run:
        output_base.mkdir(parents=True, exist_ok=True)

    expected_all_paths: list[Path] = []
    for camera_cfg in CAMERA_RUNS:
        camera_paths = find_expected_relative_npz_paths(
            upfall_root=upfall_root,
            camera=camera_cfg["camera"],
            subjects=subjects,
        )
        log(f"Camera {camera_cfg['camera']} eligible sequences: {len(camera_paths)}")
        expected_all_paths.extend(camera_paths)
    expected_all_paths = sorted(set(expected_all_paths), key=lambda path: path.as_posix())
    log(f"Total eligible sequences across both cameras: {len(expected_all_paths)}")

    variants = resolve_variants(model_root=model_root, output_base=output_base)
    for variant in variants:
        log(f"Planned variant: {variant.display_name} -> {variant.model_path}")

    total_runs = len(variants)
    for index, variant in enumerate(variants, start=1):
        log(f"[{index}/{total_runs}] {variant.display_name}")
        run_variant(
            variant=variant,
            subjects_arg=args.subjects,
            expected_all_paths=expected_all_paths,
            upfall_root=upfall_root,
            prototype_root=prototype_root,
            fix_script=fix_script,
            python_bin=args.python_bin,
            bash_bin=args.bash_bin,
            force_fix=args.force_fix,
            dry_run=args.dry_run,
        )

    log("Optimised pipeline keypoint generation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
