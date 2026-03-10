"""
Entry-point script to run AlphaPose extraction on the full UP-Fall directory tree.

This file ONLY contains the main() logic.
All pose logic, batching, timestamp parsing, etc. remain in pose_alphapose.py unchanged.
"""

from pathlib import Path
import argparse
import glob
import os


def parse_subjects_arg(value: str):
    """
    Parse subjects from:
    - single value: "12"
    - comma list: "1,3,7"
    - ranges: "1-5"
    - mixed: "1-3,7,10-12"
    """
    subjects = []
    chunks = [c.strip() for c in value.split(",") if c.strip()]
    if not chunks:
        raise argparse.ArgumentTypeError("Subjects cannot be empty.")

    for chunk in chunks:
        if "-" in chunk:
            parts = chunk.split("-", 1)
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(
                    f"Invalid range '{chunk}'. Use start-end, e.g. 1-5."
                )
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
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
        else:
            try:
                sid = int(chunk)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"Invalid subject '{chunk}'. Subject IDs must be integers."
                ) from exc
            if sid <= 0:
                raise argparse.ArgumentTypeError("Subject IDs must be positive integers.")
            subjects.append(sid)

    return sorted(set(subjects))


def find_camera_folders_subjects(root, camera=1, subjects=range(1, 6)):
    folders = []
    for s in subjects:
        subj_root = Path(root) / f"Subject{s}"
        if not subj_root.exists():
            continue
        pat = subj_root / "**" / f"*Camera{camera}"
        folders.extend([str(p) for p in glob.glob(str(pat), recursive=True) if os.path.isdir(p)])
    return sorted(set(folders))


def main():
    ap = argparse.ArgumentParser(description="Extract keypoints with AlphaPose from UP-Fall frame folders.")
    ap.add_argument(
        "--camera",
        type=int,
        required=True,
        help="UP-Fall camera number to process (e.g., 1 for Camera1).",
    )
    ap.add_argument(
        "--subjects",
        type=parse_subjects_arg,
        default=[12],
        help="Subjects to process. Examples: 12 | 1,3,7 | 1-5 | 1-3,7,10-12 (default: 12).",
    )
    ap.add_argument(
        "--upfall-root",
        type=Path,
        default=Path("../../Datasets/UPFall"),
        help="Root of UP-Fall dataset (default: ../../Datasets/UPFall).",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("../../Datasets/UPFall_keypoints_alpha/outputs_npz"),
        help="Root where outputs are written (default: ../../Datasets/UPFall_keypoints_alpha/outputs_npz).",
    )
    args = ap.parse_args()

    upfall_root = args.upfall_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not upfall_root.exists() or not upfall_root.is_dir():
        raise SystemExit(f"UP-Fall root does not exist or is not a directory: {upfall_root}")

    if __package__:
        from .pose_alphapose import AlphaPoseExportConfig, AlphaPoseRunner, run_pose_on_frames_alphapose
    else:
        from pose_alphapose import AlphaPoseExportConfig, AlphaPoseRunner, run_pose_on_frames_alphapose

    cfg = AlphaPoseExportConfig(
        alphapose_root="pose_models/AlphaPose",
        cfg_path="configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml",
        checkpoint="pretrained_models/fast_res50_256x192.pth",
        detector_cfg="detector/yolo/cfg/yolov3-spp.cfg",
        detector_weights="detector/yolo/data/yolov3-spp.weights",
        conf_thres=0.01,
        conf_min=0.01,
        nms_thres=0.6,
        fps=30,
        max_people=1,
        max_jump_px=None,  # None => use max_jump_diag_frac * image_diagonal
        max_jump_diag_frac=0.12,
        max_lost=60,
        switch_margin_px=9999.0,
        reset_on_max_lost=False,
        lock_first_target=True,
        strict_reacquire=True,
        min_iou_same_track=0.05,
        max_box_area_ratio=2.5,
        save_csv=False,
        render_video=True,
    )
    runner = AlphaPoseRunner(cfg)

    camera_folders = find_camera_folders_subjects(
        root=str(upfall_root),
        camera=args.camera,
        subjects=args.subjects,
    )

    print(f"UP-Fall root: {upfall_root}")
    print(f"Output root: {output_root}")
    print(f"Subjects: {args.subjects}")
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

        _, out_npz, _ = run_pose_on_frames_alphapose(
            frames_dir=frames_dir,
            out_dir=str(out_dir),
            windows_csv=windows_csv,
            config=cfg,
            pattern="*.png",
            runner=runner,
        )

        print(f"  -> wrote {out_npz}")
        results.append(out_npz)

    print("\nDone.")
    print("Processed sequences:", len(results))


if __name__ == "__main__":
    main()
