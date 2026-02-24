"""
Entry-point script to run AlphaPose extraction on the full UP-Fall directory tree.

This file ONLY contains the main() logic.
All pose logic, batching, timestamp parsing, etc. remain in pose_alphapose.py unchanged.

Usage examples:
  python dataset_helpers/get_keypoints_files_alphapose.py --camera 1
  python dataset_helpers/get_keypoints_files_alphapose.py --subjects 12-12
  python dataset_helpers/get_keypoints_files_alphapose.py --subjects 2,4,7
  python dataset_helpers/get_keypoints_files_alphapose.py --camera 2 --subjects 1-3
"""

from pathlib import Path
import argparse
import glob
import os

try:
    from .pose_alphapose import AlphaPoseExportConfig, AlphaPoseRunner, run_pose_on_frames_alphapose
except ImportError:
    from pose_alphapose import AlphaPoseExportConfig, AlphaPoseRunner, run_pose_on_frames_alphapose


def find_camera_folders_subjects(root, camera=1, subjects=range(1, 6)):
    folders = []
    for s in subjects:
        subj_root = Path(root) / f"Subject{s}"
        if not subj_root.exists():
            continue
        pat = subj_root / "**" / f"*Camera{camera}"
        folders.extend([str(p) for p in glob.glob(str(pat), recursive=True) if os.path.isdir(p)])
    return folders


def parse_subjects(subjects_str):
    if subjects_str is None or str(subjects_str).strip() == "":
        return range(1, 6)

    raw = str(subjects_str).strip()
    if "," in raw and "-" in raw:
        raise ValueError("subjects must be a comma list or a range, not both")

    if "-" in raw:
        parts = [p.strip() for p in raw.split("-")]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError("invalid subjects range, expected START-END")
        if not parts[0].isdigit() or not parts[1].isdigit():
            raise ValueError("subjects range must be numeric")
        start = int(parts[0])
        end = int(parts[1])
        if start <= 0 or end <= 0:
            raise ValueError("subjects must be positive integers")
        if start > end:
            raise ValueError("subjects range start must be <= end")
        return range(start, end + 1)

    parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("subjects list cannot be empty")
    subjects = []
    for p in parts:
        if not p.isdigit():
            raise ValueError("subjects list must be numeric")
        val = int(p)
        if val <= 0:
            raise ValueError("subjects must be positive integers")
        subjects.append(val)
    return sorted(set(subjects))


def build_arg_parser(default_upfall_root, default_output_root):
    parser = argparse.ArgumentParser(
        description="Run AlphaPose extraction on UP-Fall frames."
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="Comma list (e.g., 1,2,3) or range (e.g., 1-5). Default: 1-5.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        required=True,
        help="Camera index to process (e.g., 1 for Camera1).",
    )
    parser.add_argument(
        "--upfall-root",
        type=str,
        default=str(default_upfall_root),
        help="Root directory of the UP-Fall dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(default_output_root),
        help="Root directory where keypoint outputs are written.",
    )
    return parser


def main():
    # --- Configure paths for your PC ---
    UPFALL_ROOT = Path("../../../scratch/UPFall")  # change if needed
    OUTPUT_ROOT = Path("../../Datasets/UPFall_keypoints_alpha/outputs_npz")  # change if needed

    parser = build_arg_parser(UPFALL_ROOT, OUTPUT_ROOT)
    args = parser.parse_args()

    try:
        subjects = parse_subjects(args.subjects)
    except ValueError as exc:
        parser.error(str(exc))

    upfall_root = Path(args.upfall_root)
    output_root = Path(args.output_root)

    cfg = AlphaPoseExportConfig(
        alphapose_root="pose_models/AlphaPose",
        cfg_path="configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml",
        checkpoint="pretrained_models/fast_res50_256x192.pth",
        detector_cfg="detector/yolo/cfg/yolov3-spp.cfg",
        detector_weights="detector/yolo/data/yolov3-spp.weights",
        conf_thres=0.1,
        nms_thres=0.6,
        fps=30,
        max_people=1,
        save_csv=False,
        render_video=True,
    )

    runner = AlphaPoseRunner(cfg)

    camera_folders = find_camera_folders_subjects(
        root=str(upfall_root),
        camera=args.camera,
        subjects=subjects,
    )

    print("Camera folders found:", len(camera_folders))
    total = len(camera_folders)
    results = []

    for i, frames_dir in enumerate(camera_folders, 1):
        print(f"\n[{i}/{total}] Processing: {frames_dir}")

        trial_dir = os.path.dirname(frames_dir)
        matches = glob.glob(os.path.join(trial_dir, "*Features1&0.5.csv"))
        if not matches:
            print("  -> no Features1&0.5.csv found, skipping")
            continue
        windows_csv = matches[0]

        rel = os.path.relpath(frames_dir, str(upfall_root))
        out_dir = output_root / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        if (out_dir / "keypoints.npz").exists():
            print("  -> already exists, skipping")
            continue

        out_video, out_npz, _ = run_pose_on_frames_alphapose(
            frames_dir=frames_dir,
            out_dir=str(out_dir),
            windows_csv=windows_csv,
            config=cfg,
            pattern="*.png",
            runner=runner,
        )

        print(f"  OK wrote {out_npz}")
        results.append(out_npz)

    print("\nDone.")
    print("Processed sequences:", len(results))


if __name__ == "__main__":
    main()
