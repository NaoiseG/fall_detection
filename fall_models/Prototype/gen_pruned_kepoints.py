#!/usr/bin/env python3
"""
Generate UP-Fall keypoints for subjects 16 and 17 using all pruned YOLO pose models.

Runs:
- camera 1 with --lock-settings default
- camera 2 with --lock-settings strict_lock

Outputs:
- /home/people/21376026/scratch/keypoints/pruned_keypoints/<model_name>/camera_1
- /home/people/21376026/scratch/keypoints/pruned_keypoints/<model_name>/camera_2

Behaviour:
- Prints each command before running it
- Continues if one run fails
- Skips runs whose output directory already exists and is non-empty
- Prints a success / failure / skipped summary at the end

Run this script from:
    /home/people/21376026/fall_detection/fall_models/Prototype
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

UPFALL_ROOT = "../../Datasets/UPFall"
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
    {"camera": 1, "lock_settings": "default", "subdir": "camera_1"},
    {"camera": 2, "lock_settings": "strict_lock", "subdir": "camera_2"},
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def is_non_empty_dir(path: Path) -> bool:
    """Return True if path exists, is a directory, and contains at least one item."""
    return path.is_dir() and any(path.iterdir())


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
    output_dir: Path,
) -> Dict[str, Any]:
    """Run one keypoint generation job."""
    result: Dict[str, Any] = {
        "model_name": model_name,
        "model_path": model_path,
        "camera": camera,
        "lock_settings": lock_settings,
        "output_dir": str(output_dir),
        "status": None,
        "returncode": None,
        "error": None,
    }

    if not Path(model_path).is_file():
        result["status"] = "failed"
        result["error"] = f"Model file does not exist: {model_path}"
        return result

    output_dir.mkdir(parents=True, exist_ok=True)

    if is_non_empty_dir(output_dir):
        result["status"] = "skipped"
        return result

    cmd = build_command(
        model_path=model_path,
        output_root=output_dir,
        camera=camera,
        lock_settings=lock_settings,
    )

    print("\n" + "=" * 100)
    print(f"MODEL:  {model_name}")
    print(f"CAMERA: {camera}")
    print(f"LOCK:   {lock_settings}")
    print(f"OUTPUT: {output_dir}")
    print("COMMAND:")
    print(" ".join(subprocess.list2cmdline([part]) if " " in part else part for part in cmd))
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
    print(f"Working directory: {os.getcwd()}")
    print(f"Python executable:  {sys.executable}")
    print(f"Subjects:           {SUBJECTS_ARG}")
    print(f"UP-Fall root:       {UPFALL_ROOT}")
    print(f"Output base:        {OUTPUT_BASE}")

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for model_name, model_path in MODELS.items():
        model_root = OUTPUT_BASE / model_name
        model_root.mkdir(parents=True, exist_ok=True)

        for camera_cfg in CAMERA_RUNS:
            camera = camera_cfg["camera"]
            lock_settings = camera_cfg["lock_settings"]
            subdir = camera_cfg["subdir"]
            output_dir = model_root / subdir

            result = run_one(
                model_name=model_name,
                model_path=model_path,
                camera=camera,
                lock_settings=lock_settings,
                output_dir=output_dir,
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
            print(f"  - {r['model_name']} | camera {r['camera']} | {r['output_dir']}")

    if skipped:
        print("\nSKIPPED RUNS (output dir already exists and is non-empty):")
        for r in skipped:
            print(f"  - {r['model_name']} | camera {r['camera']} | {r['output_dir']}")

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