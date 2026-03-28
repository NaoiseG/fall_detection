#!/usr/bin/env python3
"""
Generate UP-Fall keypoints for subjects 16 and 17 using every discovered
fully-pruned base/fp16/int8 TensorRT engine.

Runs:
- camera 1 with --lock-settings default
- camera 2 with --lock-settings strict_lock

Outputs:
- ../../Datasets/pruned_keypoints/full_pruned/<pose_model>/<prune_variant>/<variant>

Relative CLI paths are resolved from the Prototype directory so this script can
be run from anywhere.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SUBJECTS_DEFAULT = "16-17"
POSE_MODEL_ORDER = ("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x")
VERSION_ORDER = ("base", "fp16", "int8")
ENGINE_SUFFIX_BY_VERSION = {
    "base": "fp32",
    "fp16": "fp16",
    "int8": "int8",
}
CAMERA_RUNS = (
    {"camera": 1, "lock_settings": "default"},
    {"camera": 2, "lock_settings": "strict_lock"},
)
FULL_PRUNED_ROOT_CANDIDATES = (
    Path("../../pruning/pruned_models/full_pruned"),
    Path("../../pruning/full_pruned"),
)
UPFALL_ROOT_DEFAULT = Path("../../Datasets/UPFall")
OUTPUT_BASE_DEFAULT = Path("../../Datasets/pruned_keypoints/full_pruned")


@dataclass(frozen=True)
class DiscoveredRun:
    model_dir_name: str
    weights_dir: Path
    pose_model: str
    prune_variant: str
    version: str
    engine_path: Path
    output_root: Path


def parse_subjects_arg(value: str) -> List[int]:
    """
    Parse subject expressions like "16-17" or "1,3,7".
    """
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
    """
    Resolve absolute paths directly and relative paths against the Prototype
    directory so the script is independent of the caller's cwd.
    """
    expanded = path_value.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (prototype_root / expanded).resolve()


def resolve_full_pruned_root(path_value: Optional[Path], prototype_root: Path) -> Path:
    if path_value is not None:
        return resolve_from_prototype_root(path_value, prototype_root)

    resolved_candidates = [
        resolve_from_prototype_root(candidate, prototype_root)
        for candidate in FULL_PRUNED_ROOT_CANDIDATES
    ]
    for candidate in resolved_candidates:
        if candidate.is_dir():
            return candidate
    return resolved_candidates[0]


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


def parse_model_dir_name(model_dir_name: str) -> Optional[Tuple[str, str]]:
    direct_match = re.fullmatch(r"(yolo11[nsmlx])_pruned_(\d+)", model_dir_name)
    if direct_match:
        pose_model, prune_percent = direct_match.groups()
        return pose_model, f"pruned_{prune_percent}"

    legacy_match = re.fullmatch(
        r"pruned_pose_(yolo11[nsmlx])_pose_flops(\d+)p(?:_trainfrac\d+p)?",
        model_dir_name,
    )
    if legacy_match:
        pose_model, prune_percent = legacy_match.groups()
        return pose_model, f"pruned_{prune_percent}"

    return None


def find_engine_path(weights_dir: Path, model_dir_name: str, version: str) -> Optional[Path]:
    engine_suffix = ENGINE_SUFFIX_BY_VERSION[version]
    expected = weights_dir / f"{model_dir_name}_{engine_suffix}.engine"
    if expected.is_file():
        return expected

    matches = sorted(weights_dir.glob(f"*_{engine_suffix}.engine"))
    if len(matches) == 1:
        return matches[0]
    return None


def discovered_run_sort_key(run: DiscoveredRun) -> Tuple[int, int, int, str]:
    pose_index = POSE_MODEL_ORDER.index(run.pose_model) if run.pose_model in POSE_MODEL_ORDER else 999
    if run.prune_variant.startswith("pruned_"):
        prune_value_text = run.prune_variant[len("pruned_") :]
    else:
        prune_value_text = run.prune_variant
    try:
        prune_value = int(prune_value_text)
    except ValueError:
        prune_value = 999
    version_index = VERSION_ORDER.index(run.version) if run.version in VERSION_ORDER else 999
    return pose_index, prune_value, version_index, run.model_dir_name


def discover_full_pruned_runs(
    full_pruned_root: Path,
    output_base: Path,
    versions: Sequence[str],
) -> Tuple[List[DiscoveredRun], List[str]]:
    issues: List[str] = []
    discovered_runs: List[DiscoveredRun] = []

    if not full_pruned_root.is_dir():
        issues.append(f"Full-pruned root does not exist: {full_pruned_root}")
        return discovered_runs, issues

    weights_dirs = sorted(
        path
        for path in full_pruned_root.rglob("weights")
        if path.is_dir()
    )
    if not weights_dirs:
        issues.append(f"No weights directories were found under: {full_pruned_root}")
        return discovered_runs, issues

    for weights_dir in weights_dirs:
        model_dir_name = weights_dir.parent.name
        parsed = parse_model_dir_name(model_dir_name)
        if parsed is None:
            issues.append(
                f"Skipping unrecognized full-pruned model directory name: {weights_dir.parent}"
            )
            continue

        pose_model, prune_variant = parsed
        for version in versions:
            engine_path = find_engine_path(weights_dir, model_dir_name, version)
            if engine_path is None:
                issues.append(
                    f"Missing {version} engine for {model_dir_name} under {weights_dir}"
                )
                continue

            discovered_runs.append(
                DiscoveredRun(
                    model_dir_name=model_dir_name,
                    weights_dir=weights_dir,
                    pose_model=pose_model,
                    prune_variant=prune_variant,
                    version=version,
                    engine_path=engine_path,
                    output_root=output_base / pose_model / prune_variant / version,
                )
            )

    discovered_runs.sort(key=discovered_run_sort_key)
    return discovered_runs, issues


def build_command(
    subjects_arg: str,
    upfall_root: Path,
    output_root: Path,
    camera: int,
    lock_settings: str,
    model_path: Path,
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
    ]


def run_one(
    run: DiscoveredRun,
    subjects_arg: str,
    upfall_root: Path,
    camera: int,
    lock_settings: str,
    expected_relative_npz_paths: Sequence[Path],
    prototype_root: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "model_dir_name": run.model_dir_name,
        "pose_model": run.pose_model,
        "prune_variant": run.prune_variant,
        "version": run.version,
        "engine_path": str(run.engine_path),
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

    if not run.engine_path.is_file():
        result["status"] = "failed"
        result["error"] = f"Model file does not exist: {run.engine_path}"
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
        model_path=run.engine_path,
    )

    print("\n" + "=" * 100)
    print(f"MODEL DIR:   {run.model_dir_name}")
    print(f"POSE MODEL:  {run.pose_model}")
    print(f"PRUNE:       {run.prune_variant}")
    print(f"VERSION:     {run.version}")
    print(f"CAMERA:      {camera}")
    print(f"LOCK:        {lock_settings}")
    print(f"ENGINE:      {run.engine_path}")
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
            "Generate UP-Fall keypoints for subjects 16-17 using every discovered "
            "fully-pruned base/fp16/int8 TensorRT engine."
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
            "Outputs are written under <base>/<pose_model>/<pruned_variant>/<variant>."
        ),
    )
    parser.add_argument(
        "--full-pruned-root",
        type=Path,
        default=None,
        help=(
            "Root containing fully-pruned model directories. Defaults to the first "
            "existing path from ../../pruning/pruned_models/full_pruned and "
            "../../pruning/full_pruned."
        ),
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=VERSION_ORDER,
        default=list(VERSION_ORDER),
        help="Engine variants to process (default: base fp16 int8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the discovered runs and commands without executing them.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    prototype_root = Path(__file__).resolve().parents[1]
    subjects = parse_subjects_arg(args.subjects)
    upfall_root = resolve_from_prototype_root(args.upfall_root, prototype_root)
    output_base = resolve_from_prototype_root(args.output_base, prototype_root)
    full_pruned_root = resolve_full_pruned_root(args.full_pruned_root, prototype_root)

    print("Starting full-pruned UP-Fall keypoint generation")
    print(f"Working directory:  {Path.cwd()}")
    print(f"Prototype root:     {prototype_root}")
    print(f"Python executable:  {sys.executable}")
    print(f"Subjects:           {args.subjects}")
    print(f"UP-Fall root:       {upfall_root}")
    print(f"Full-pruned root:   {full_pruned_root}")
    print(f"Output base:        {output_base}")
    print(f"Versions:           {', '.join(args.versions)}")
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

    discovered_runs, discovery_issues = discover_full_pruned_runs(
        full_pruned_root=full_pruned_root,
        output_base=output_base,
        versions=args.versions,
    )

    if discovery_issues:
        print("\nDISCOVERY ISSUES:")
        for issue in discovery_issues:
            print(f"  - {issue}")

    if not discovered_runs:
        print("\nNo runnable full-pruned engine variants were discovered.")
        return 1

    print(f"\nDiscovered runnable engine variants: {len(discovered_runs)}")
    for run in discovered_runs:
        print(
            f"  - {run.pose_model} | {run.prune_variant} | {run.version} "
            f"| {run.engine_path}"
        )

    all_results: List[Dict[str, Any]] = []

    for run in discovered_runs:
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
    print(f"Discovery issues : {len(discovery_issues)}")

    if skipped:
        print("\nSKIPPED RUNS:")
        for result in skipped:
            print(
                f"  - {result['pose_model']} | {result['prune_variant']} | {result['version']} "
                f"| camera {result['camera']} | {result['output_dir']}"
            )
            if result["skip_reason"]:
                print(f"    Reason: {result['skip_reason']}")

    if failures:
        print("\nFAILED RUNS:")
        for result in failures:
            print(
                f"  - {result['pose_model']} | {result['prune_variant']} | {result['version']} "
                f"| camera {result['camera']} | {result['output_dir']}"
            )
            if result["error"]:
                print(f"    Error: {result['error']}")

    if planned:
        print("\nPLANNED RUNS:")
        for result in planned:
            print(
                f"  - {result['pose_model']} | {result['prune_variant']} | {result['version']} "
                f"| camera {result['camera']} | {result['output_dir']}"
            )

    return 1 if failures or discovery_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
