#!/usr/bin/env python3
"""
Generate UP-Fall keypoints for subjects 16 and 17 using a fixed resized-input
YOLO pose sweep.

Runs:
- camera 1 with --lock-settings default
- camera 2 with --lock-settings strict_lock

Model order:
- yolo11m fp16 with imgsz 576
- yolo11x pruned_80 fp16 with imgsz 448
- yolo11x fp32 with imgsz 448
- yolo11l fp32 with imgsz 512

Outputs:
- ../../Datasets/UPFall_keypoints_img_downsize/<pose_model>/<variant_name>

Relative CLI paths are resolved from the Prototype directory so this script can
be run from anywhere.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SUBJECTS_DEFAULT = "16-17"
CAMERA_RUNS = (
    {"camera": 1, "lock_settings": "default"},
    {"camera": 2, "lock_settings": "strict_lock"},
)
MODEL_ROOT_DEFAULT = Path("../../quantisation/models/img_downsize")
OUTPUT_BASE_DEFAULT = Path("../../Datasets/UPFall_keypoints_img_downsize")
UPFALL_ROOT_DEFAULT = Path("../../Datasets/UPFall")


@dataclass(frozen=True)
class VariantSpec:
    run_name: str
    pose_model: str
    variant_name: str
    imgsz: int
    preferred_relative_paths: Tuple[Path, ...]
    search_patterns: Tuple[str, ...]
    fallback_paths: Tuple[Path, ...]
    output_relative_dir: Path


VARIANT_SPECS: Tuple[VariantSpec, ...] = (
    VariantSpec(
        run_name="yolo11m fp16 imgsz 576",
        pose_model="yolo11m",
        variant_name="fp16_imgsz576",
        imgsz=576,
        preferred_relative_paths=(
            Path("yolo11m-pose/yolo11m-pose_imgsz576_fp16.engine"),
            Path("yolo11m-pose_imgsz576_fp16.engine"),
        ),
        search_patterns=(
            "**/yolo11m-pose_imgsz576_fp16.engine",
            "**/*yolo11m*imgsz576*fp16*.engine",
        ),
        fallback_paths=(
            Path("../../quantisation/models/ultralytics/yolo11m-pose/yolo11m-pose_imgsz576_fp16.engine"),
        ),
        output_relative_dir=Path("yolo11m/fp16_imgsz576"),
    ),
    VariantSpec(
        run_name="yolo11x pruned_80 fp16 imgsz 448",
        pose_model="yolo11x",
        variant_name="pruned_80_fp16_imgsz448",
        imgsz=448,
        preferred_relative_paths=(
            Path("yolo11x_pruned_80/yolo11x_pruned_80_imgsz448_fp16.engine"),
            Path("yolo11x_pruned_80/weights/yolo11x_pruned_80_imgsz448_fp16.engine"),
            Path("yolo11x_pruned_80/yolo11x_pruned_80_fp16.engine"),
            Path("yolo11x_pruned_80/weights/yolo11x_pruned_80_fp16.engine"),
        ),
        search_patterns=(
            "**/yolo11x_pruned_80_imgsz448_fp16.engine",
            "**/*yolo11x*pruned*80*imgsz448*fp16*.engine",
            "**/yolo11x_pruned_80_fp16.engine",
            "**/*yolo11x*pruned*80*fp16*.engine",
        ),
        fallback_paths=(
            Path("../../pruning/full_pruned/yolo11x_pruned_80/weights/yolo11x_pruned_80_fp16.engine"),
        ),
        output_relative_dir=Path("yolo11x/pruned_80_fp16_imgsz448"),
    ),
    VariantSpec(
        run_name="yolo11x fp32 imgsz 448",
        pose_model="yolo11x",
        variant_name="fp32_imgsz448",
        imgsz=448,
        preferred_relative_paths=(
            Path("yolo11x-pose/yolo11x-pose_imgsz448_fp32.engine"),
            Path("yolo11x-pose_imgsz448_fp32.engine"),
        ),
        search_patterns=(
            "**/yolo11x-pose_imgsz448_fp32.engine",
            "**/*yolo11x*imgsz448*fp32*.engine",
        ),
        fallback_paths=(
            Path("../../quantisation/models/ultralytics/yolo11x-pose/yolo11x-pose_imgsz448_fp32.engine"),
        ),
        output_relative_dir=Path("yolo11x/fp32_imgsz448"),
    ),
    VariantSpec(
        run_name="yolo11l fp32 imgsz 512",
        pose_model="yolo11l",
        variant_name="fp32_imgsz512",
        imgsz=512,
        preferred_relative_paths=(
            Path("yolo11l-pose/yolo11l-pose_imgsz512_fp32.engine"),
            Path("yolo11l-pose_imgsz512_fp32.engine"),
        ),
        search_patterns=(
            "**/yolo11l-pose_imgsz512_fp32.engine",
            "**/*yolo11l*imgsz512*fp32*.engine",
        ),
        fallback_paths=(
            Path("../../quantisation/models/ultralytics/yolo11l-pose/yolo11l-pose_imgsz512_fp32.engine"),
        ),
        output_relative_dir=Path("yolo11l/fp32_imgsz512"),
    ),
)


@dataclass(frozen=True)
class ResolvedVariant:
    spec: VariantSpec
    model_path: Path
    output_root: Path


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


def find_camera_folders_subjects(
    root: Path,
    camera: int,
    subjects: Sequence[int],
) -> List[Path]:
    folders: List[Path] = []
    for subject_id in subjects:
        subject_root = root / f"Subject{subject_id}"
        if not subject_root.exists():
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

    for frames_dir in find_camera_folders_subjects(
        root=upfall_root,
        camera=camera,
        subjects=subjects,
    ):
        trial_dir = frames_dir.parent
        if not any(trial_dir.glob("*Features1&0.5.csv")):
            continue
        expected_paths.append(frames_dir.relative_to(upfall_root) / "keypoints.npz")

    return sorted(set(expected_paths), key=lambda path: path.as_posix())


def get_run_completion(
    output_root: Path,
    expected_relative_npz_paths: Sequence[Path],
) -> Dict[str, Any]:
    existing_paths = [
        rel_path for rel_path in expected_relative_npz_paths if (output_root / rel_path).is_file()
    ]
    missing_paths = [
        rel_path for rel_path in expected_relative_npz_paths if not (output_root / rel_path).is_file()
    ]

    return {
        "complete": bool(expected_relative_npz_paths) and not missing_paths,
        "expected_count": len(expected_relative_npz_paths),
        "existing_count": len(existing_paths),
        "missing_count": len(missing_paths),
        "first_missing": str(missing_paths[0]) if missing_paths else None,
    }


def format_command(cmd: Sequence[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def resolve_variant_model_path(
    spec: VariantSpec,
    *,
    model_root: Path,
    prototype_root: Path,
) -> Tuple[Optional[Path], List[str]]:
    checked_locations: List[str] = []

    for relative_path in spec.preferred_relative_paths:
        candidate = (model_root / relative_path).resolve()
        checked_locations.append(str(candidate))
        if candidate.is_file():
            return candidate, checked_locations

    if model_root.is_dir():
        seen_matches: List[Path] = []
        for pattern in spec.search_patterns:
            for match in sorted(model_root.glob(pattern), key=lambda path: path.as_posix()):
                resolved_match = match.resolve()
                if resolved_match.is_file() and resolved_match not in seen_matches:
                    seen_matches.append(resolved_match)
        if len(seen_matches) == 1:
            checked_locations.append(str(seen_matches[0]))
            return seen_matches[0], checked_locations
        if len(seen_matches) > 1:
            checked_locations.extend(str(match) for match in seen_matches)
            return None, checked_locations

    for fallback_path in spec.fallback_paths:
        candidate = resolve_from_prototype_root(fallback_path, prototype_root)
        checked_locations.append(str(candidate))
        if candidate.is_file():
            return candidate, checked_locations

    return None, checked_locations


def resolve_variants(
    *,
    prototype_root: Path,
    model_root: Path,
    output_base: Path,
) -> Tuple[List[ResolvedVariant], List[str]]:
    resolved_variants: List[ResolvedVariant] = []
    issues: List[str] = []

    for spec in VARIANT_SPECS:
        model_path, checked_locations = resolve_variant_model_path(
            spec,
            model_root=model_root,
            prototype_root=prototype_root,
        )
        if model_path is None:
            issue = (
                f"Could not resolve model for {spec.run_name}. "
                f"Checked: {', '.join(checked_locations) if checked_locations else '<none>'}"
            )
            issues.append(issue)
            continue

        resolved_variants.append(
            ResolvedVariant(
                spec=spec,
                model_path=model_path,
                output_root=output_base / spec.output_relative_dir,
            )
        )

    return resolved_variants, issues


def build_command(
    subjects_arg: str,
    upfall_root: Path,
    output_root: Path,
    camera: int,
    lock_settings: str,
    model_path: Path,
    imgsz: int,
) -> List[str]:
    return [
        sys.executable,
        "-m",
        "dataset_helpers.get_keypoints_files",
        "--subjects",
        subjects_arg,
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


def run_one(
    run: ResolvedVariant,
    subjects_arg: str,
    upfall_root: Path,
    camera: int,
    lock_settings: str,
    expected_relative_npz_paths: Sequence[Path],
    prototype_root: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "run_name": run.spec.run_name,
        "pose_model": run.spec.pose_model,
        "variant_name": run.spec.variant_name,
        "imgsz": run.spec.imgsz,
        "model_path": str(run.model_path),
        "camera": camera,
        "lock_settings": lock_settings,
        "output_dir": str(run.output_root),
        "status": None,
        "returncode": None,
        "error": None,
        "expected_sequences": len(expected_relative_npz_paths),
        "existing_sequences": 0,
        "missing_sequences": len(expected_relative_npz_paths),
        "skip_reason": None,
    }

    if not run.model_path.is_file():
        result["status"] = "failed"
        result["error"] = f"Model file does not exist: {run.model_path}"
        return result

    run.output_root.mkdir(parents=True, exist_ok=True)

    completion = get_run_completion(
        output_root=run.output_root,
        expected_relative_npz_paths=expected_relative_npz_paths,
    )
    result["existing_sequences"] = completion["existing_count"]
    result["missing_sequences"] = completion["missing_count"]

    if completion["complete"]:
        result["status"] = "skipped"
        result["skip_reason"] = (
            f"all {completion['expected_count']} expected native outputs already exist "
            f"for camera {camera}"
        )
        return result

    cmd = build_command(
        subjects_arg=subjects_arg,
        upfall_root=upfall_root,
        output_root=run.output_root,
        camera=camera,
        lock_settings=lock_settings,
        model_path=run.model_path,
        imgsz=run.spec.imgsz,
    )

    print("\n" + "=" * 100)
    print(f"RUN:         {run.spec.run_name}")
    print(f"POSE MODEL:  {run.spec.pose_model}")
    print(f"VARIANT:     {run.spec.variant_name}")
    print(f"IMGSZ:       {run.spec.imgsz}")
    print(f"CAMERA:      {camera}")
    print(f"LOCK:        {lock_settings}")
    print(f"MODEL:       {run.model_path}")
    print(f"OUTPUT:      {run.output_root}")
    if completion["expected_count"]:
        print(
            "STATE:       "
            f"{completion['existing_count']}/{completion['expected_count']} expected "
            f"native outputs already present"
        )
        if completion["first_missing"]:
            print(f"MISSING:     {completion['first_missing']}")
    else:
        print("STATE:       no eligible native outputs were discovered up front; skip disabled")
    print("COMMAND:")
    print(format_command(cmd))
    print("=" * 100)

    if dry_run:
        result["status"] = "planned"
        return result

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            cwd=str(prototype_root),
        )
        result["returncode"] = completed.returncode
        if completed.returncode == 0:
            result["status"] = "success"
        else:
            result["status"] = "failed"
            result["error"] = f"Command exited with return code {completed.returncode}"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate UP-Fall keypoints for a fixed resized-input YOLO pose sweep."
        )
    )
    parser.add_argument(
        "--subjects",
        default=SUBJECTS_DEFAULT,
        help=(
            "Subjects to process. Examples: 16-17 | 16,17 | 1-3,7 "
            f"(default: {SUBJECTS_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--upfall-root",
        type=Path,
        default=UPFALL_ROOT_DEFAULT,
        help="UP-Fall root, resolved relative to the Prototype directory.",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=OUTPUT_BASE_DEFAULT,
        help=(
            "Base output directory, resolved relative to the Prototype directory. "
            "Outputs are written under <base>/<pose_model>/<variant_name>."
        ),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=MODEL_ROOT_DEFAULT,
        help=(
            "Root containing the resized-input model artifacts, resolved relative to "
            "the Prototype directory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved runs and commands without executing them.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    prototype_root = Path(__file__).resolve().parents[1]
    subjects = parse_subjects_arg(args.subjects)
    upfall_root = resolve_from_prototype_root(args.upfall_root, prototype_root)
    output_base = resolve_from_prototype_root(args.output_base, prototype_root)
    model_root = resolve_from_prototype_root(args.model_root, prototype_root)

    print("Starting resized-input UP-Fall keypoint generation")
    print(f"Working directory:  {Path.cwd()}")
    print(f"Prototype root:     {prototype_root}")
    print(f"Python executable:  {sys.executable}")
    print(f"Subjects:           {args.subjects}")
    print(f"UP-Fall root:       {upfall_root}")
    print(f"Model root:         {model_root}")
    print(f"Output base:        {output_base}")
    print(f"Dry run:            {args.dry_run}")

    output_base.mkdir(parents=True, exist_ok=True)

    if not upfall_root.is_dir():
        print(f"ERROR: UP-Fall root does not exist: {upfall_root}")
        return 1

    expected_paths_by_camera = {
        camera_cfg["camera"]: find_expected_relative_npz_paths(
            upfall_root=upfall_root,
            camera=camera_cfg["camera"],
            subjects=subjects,
        )
        for camera_cfg in CAMERA_RUNS
    }
    for camera_cfg in CAMERA_RUNS:
        camera = camera_cfg["camera"]
        expected_count = len(expected_paths_by_camera[camera])
        print(f"Camera {camera} eligible sequences: {expected_count}")

    resolved_variants, resolution_issues = resolve_variants(
        prototype_root=prototype_root,
        model_root=model_root,
        output_base=output_base,
    )

    if resolution_issues:
        print("\nMODEL RESOLUTION ISSUES:")
        for issue in resolution_issues:
            print(f"  - {issue}")

    if not resolved_variants:
        print("\nNo runnable resized-input variants were resolved.")
        return 1

    print(f"\nResolved runnable variants: {len(resolved_variants)}")
    for run in resolved_variants:
        print(
            f"  - {run.spec.run_name} | imgsz {run.spec.imgsz} "
            f"| {run.model_path}"
        )

    all_results: List[Dict[str, Any]] = []

    for run in resolved_variants:
        for camera_cfg in CAMERA_RUNS:
            result = run_one(
                run=run,
                subjects_arg=args.subjects,
                upfall_root=upfall_root,
                camera=camera_cfg["camera"],
                lock_settings=camera_cfg["lock_settings"],
                expected_relative_npz_paths=expected_paths_by_camera[camera_cfg["camera"]],
                prototype_root=prototype_root,
                dry_run=args.dry_run,
            )
            all_results.append(result)

    successes = [result for result in all_results if result["status"] == "success"]
    failures = [result for result in all_results if result["status"] == "failed"]
    skipped = [result for result in all_results if result["status"] == "skipped"]
    planned = [result for result in all_results if result["status"] == "planned"]

    print("\n" + "#" * 100)
    print("SUMMARY")
    print("#" * 100)
    print(f"Total runs       : {len(all_results)}")
    print(f"Successful runs  : {len(successes)}")
    print(f"Failed runs      : {len(failures)}")
    print(f"Skipped runs     : {len(skipped)}")
    print(f"Planned only     : {len(planned)}")
    print(f"Resolution issues: {len(resolution_issues)}")

    if skipped:
        print("\nSKIPPED RUNS:")
        for result in skipped:
            print(
                f"  - {result['run_name']} | camera {result['camera']} | {result['output_dir']}"
            )
            if result["skip_reason"]:
                print(f"    Reason: {result['skip_reason']}")

    if failures:
        print("\nFAILED RUNS:")
        for result in failures:
            print(
                f"  - {result['run_name']} | camera {result['camera']} | {result['output_dir']}"
            )
            if result["error"]:
                print(f"    Error: {result['error']}")

    if planned:
        print("\nPLANNED RUNS:")
        for result in planned:
            print(
                f"  - {result['run_name']} | camera {result['camera']} | {result['output_dir']}"
            )

    return 1 if failures or resolution_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
