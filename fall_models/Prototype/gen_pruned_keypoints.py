#!/usr/bin/env python3
"""
Generate UP-Fall keypoints for subjects 16 and 17 using all pruned YOLO pose models.

Runs:
- camera 1 with --lock-settings default
- camera 2 with --lock-settings strict_lock

Outputs:
- /home/people/21376026/scratch/keypoints/pruned_keypoints/<model_name>

Behaviour:
- Prints each command before running it
- Continues if one run fails
- Skips a camera run only when all expected native helper outputs for that
  specific camera already exist under the model output root
- Prints a success / failure / skipped summary at the end

Run this script from:
    /home/people/21376026/fall_detection/fall_models/Prototype
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

UPFALL_ROOT = "/home/people/21376026/scratch/UPFall"
OUTPUT_BASE = Path("/home/people/21376026/scratch/keypoints/pruned_keypoints")

# Use a range to match the example command style.
SUBJECTS_ARG = "16-17"

MODELS = {
    "yolo11n_pruned_80": "/home/people/21376026/scratch/prune_models/yolo11n-pose/pruned/pruned_pose_yolo11n_pose_flops80p_trainfrac25p/weights/best.pt",
    "yolo11n_pruned_90": "/home/people/21376026/scratch/prune_models/yolo11n-pose/pruned/pruned_pose_yolo11n_pose_flops90p_trainfrac25p/weights/best.pt",
    "yolo11s_pruned_80": "/home/people/21376026/scratch/prune_models/yolo11s-pose/pruned/pruned_pose_yolo11s_pose_flops80p_trainfrac25p/weights/best.pt",
    "yolo11s_pruned_90": "/home/people/21376026/scratch/prune_models/yolo11s-pose/pruned/pruned_pose_yolo11s_pose_flops90p_trainfrac25p/weights/best.pt",
    "yolo11m_pruned_80": "/home/people/21376026/scratch/prune_models/yolo11m-pose/pruned/pruned_pose_yolo11m_pose_flops80p_trainfrac25p/weights/best.pt",
    "yolo11m_pruned_90": "/home/people/21376026/scratch/prune_models/yolo11m-pose/pruned/pruned_pose_yolo11m_pose_flops90p_trainfrac25p/weights/best.pt",
    "yolo11l_pruned_80": "/home/people/21376026/scratch/prune_models/yolo11l-pose/pruned/pruned_pose_yolo11l_pose_flops80p_trainfrac25p/weights/best.pt",
    "yolo11l_pruned_90": "/home/people/21376026/scratch/prune_models/yolo11l-pose/pruned/pruned_pose_yolo11l_pose_flops90p_trainfrac25p/weights/best.pt",
    "yolo11x_pruned_80": "/home/people/21376026/scratch/prune_models/yolo11x-pose/pruned/pruned_pose_yolo11x_pose_flops80p_trainfrac25p/weights/best.pt",
    "yolo11x_pruned_90": "/home/people/21376026/scratch/prune_models/yolo11x-pose/pruned/pruned_pose_yolo11x_pose_flops90p_trainfrac25p/weights/best.pt",
}

CAMERA_RUNS = [
    {"camera": 1, "lock_settings": "default"},
    {"camera": 2, "lock_settings": "strict_lock"},
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_subjects_arg(value: str) -> List[int]:
    """Parse subject expressions like '16-17' or '1,3,7'."""
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


def find_camera_folders_subjects(
    root: Path,
    camera: int,
    subjects: Sequence[int],
) -> List[Path]:
    """Match the camera-folder discovery used by dataset_helpers.get_keypoints_files."""
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
    """
    Compute the native relative keypoint output paths produced by
    dataset_helpers.get_keypoints_files for one camera run.
    """
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
    """
    A run is complete only if every expected native keypoint file for that
    specific camera already exists under the shared model output root.
    """
    existing_paths = [
        rel_path for rel_path in expected_relative_npz_paths if (output_root / rel_path).is_file()
    ]
    missing_paths = [
        rel_path for rel_path in expected_relative_npz_paths if (output_root / rel_path).is_file() is False
    ]

    return {
        "complete": bool(expected_relative_npz_paths) and not missing_paths,
        "expected_count": len(expected_relative_npz_paths),
        "existing_count": len(existing_paths),
        "missing_count": len(missing_paths),
        "first_missing": str(missing_paths[0]) if missing_paths else None,
    }


def format_command(cmd: Sequence[str]) -> str:
    """Render a shell-friendly command line for logging."""
    return shlex.join(str(part) for part in cmd)


def build_command(
    model_path: str,
    output_root: Path,
    camera: int,
    lock_settings: str,
) -> List[str]:
    """Build the subprocess command."""
    return [
        sys.executable,
        "-m",
        "dataset_helpers.get_keypoints_files",
        "--subjects",
        SUBJECTS_ARG,
        "--camera",
        str(camera),
        "--lock-settings",
        lock_settings,
        "--upfall-root",
        UPFALL_ROOT,
        "--output-root",
        str(output_root),
        "--model-path",
        model_path,
    ]


def run_one(
    model_name: str,
    model_path: str,
    camera: int,
    lock_settings: str,
    output_root: Path,
    expected_relative_npz_paths: Sequence[Path],
) -> Dict[str, Any]:
    """Run one keypoint generation job."""
    result: Dict[str, Any] = {
        "model_name": model_name,
        "model_path": model_path,
        "camera": camera,
        "lock_settings": lock_settings,
        "output_dir": str(output_root),
        "status": None,
        "returncode": None,
        "error": None,
        "expected_sequences": len(expected_relative_npz_paths),
        "existing_sequences": 0,
        "missing_sequences": len(expected_relative_npz_paths),
        "skip_reason": None,
    }

    if not Path(model_path).is_file():
        result["status"] = "failed"
        result["error"] = f"Model file does not exist: {model_path}"
        return result

    output_root.mkdir(parents=True, exist_ok=True)

    completion = get_run_completion(
        output_root=output_root,
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
        model_path=model_path,
        output_root=output_root,
        camera=camera,
        lock_settings=lock_settings,
    )

    print("\n" + "=" * 100)
    print(f"MODEL:  {model_name}")
    print(f"CAMERA: {camera}")
    print(f"LOCK:   {lock_settings}")
    print(f"OUTPUT: {output_root}")
    if completion["expected_count"]:
        print(
            "STATE:  "
            f"{completion['existing_count']}/{completion['expected_count']} expected "
            f"native outputs already present"
        )
        if completion["first_missing"]:
            print(f"MISSING SAMPLE: {completion['first_missing']}")
    else:
        print("STATE:  no eligible native outputs were discovered up front; skip disabled")
    print("COMMAND:")
    print(format_command(cmd))
    print("=" * 100)

    try:
        completed = subprocess.run(cmd, check=False)
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    print("Starting pruned UP-Fall keypoint generation")
    print(f"Working directory: {Path.cwd()}")
    print(f"Python executable:  {sys.executable}")
    print(f"Subjects:           {SUBJECTS_ARG}")
    print(f"UP-Fall root:       {UPFALL_ROOT}")
    print(f"Output base:        {OUTPUT_BASE}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    subjects = parse_subjects_arg(SUBJECTS_ARG)
    upfall_root = Path(UPFALL_ROOT)
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

    all_results: List[Dict[str, Any]] = []

    for model_name, model_path in MODELS.items():
        model_root = OUTPUT_BASE / model_name
        model_root.mkdir(parents=True, exist_ok=True)

        for camera_cfg in CAMERA_RUNS:
            camera = camera_cfg["camera"]
            lock_settings = camera_cfg["lock_settings"]

            result = run_one(
                model_name=model_name,
                model_path=model_path,
                camera=camera,
                lock_settings=lock_settings,
                output_root=model_root,
                expected_relative_npz_paths=expected_paths_by_camera[camera],
            )
            all_results.append(result)

    successes = [r for r in all_results if r["status"] == "success"]
    failures = [r for r in all_results if r["status"] == "failed"]
    skipped = [r for r in all_results if r["status"] == "skipped"]

    print("\n" + "#" * 100)
    print("SUMMARY")
    print("#" * 100)
    print(f"Total runs : {len(all_results)}")
    print(f"Successes  : {len(successes)}")
    print(f"Failures   : {len(failures)}")
    print(f"Skipped    : {len(skipped)}")

    if successes:
        print("\nSUCCESSFUL RUNS:")
        for r in successes:
            print(
                f"  - {r['model_name']} | camera {r['camera']} | {r['output_dir']} "
                f"| existing before run: {r['existing_sequences']}/{r['expected_sequences']}"
            )

    if skipped:
        print("\nSKIPPED RUNS (camera-specific native outputs already complete):")
        for r in skipped:
            print(f"  - {r['model_name']} | camera {r['camera']} | {r['output_dir']}")
            if r["skip_reason"]:
                print(f"    Reason: {r['skip_reason']}")

    if failures:
        print("\nFAILED RUNS:")
        for r in failures:
            print(f"  - {r['model_name']} | camera {r['camera']} | {r['output_dir']}")
            if r["error"]:
                print(f"    Error: {r['error']}")

    # Return non-zero only if at least one run failed.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
