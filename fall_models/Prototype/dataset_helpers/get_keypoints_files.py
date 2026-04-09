"""
Entry-point script to run pose extraction on the full UP-Fall directory tree.

This file ONLY contains the main() logic.
All pose logic, batching, timestamp parsing, etc. remain in pose.py unchanged.
"""

from pathlib import Path
import argparse
import glob
import os


LOCK_SETTINGS_CHOICES = ("strict_lock", "default")


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
    ap = argparse.ArgumentParser(description="Extract keypoints from UP-Fall frame folders.")
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
        default=Path("../../Datasets/UPFall_keypoints/outputs_npz"),
        help="Root where outputs are written (default: ../../Datasets/UPFall_keypoints/outputs_npz).",
    )
    ap.add_argument(
        "--model-path",
        type=Path,
        default=Path("pose_models/ultralytics/yolo11l-pose.pt"),
        help="Path to YOLO pose weights (.pt or TensorRT .engine).",
    )
    ap.add_argument(
        "--imgsz",
        type=float,
        default=None,
        help=(
            "Optional YOLO inference input size control. "
            "Values in (0,1] are interpreted as a fraction of original input pixels, "
            "e.g. --imgsz 0.9 targets 90%% of the original input pixels. "
            "Values >1 are treated as an explicit square YOLO imgsz."
        ),
    )

    ap.add_argument(
        "--lock-settings",
        choices=LOCK_SETTINGS_CHOICES,
        default="strict_lock",
        help=(
            "Tracking/lock preset. "
            "'strict_lock' preserves the settings that were previously hard-coded here."
        ),
    )
    ap.add_argument(
        "--camera1-foreground-guard",
        dest="camera1_foreground_guard",
        action="store_true",
        help=(
            "For Camera1, prefer larger/lower foreground candidates before the track is "
            "permanently locked, which helps avoid sticking to reflections."
        ),
    )
    ap.add_argument(
        "--no-camera1-foreground-guard",
        dest="camera1_foreground_guard",
        action="store_false",
        help="Disable the Camera1 foreground-acquisition guard.",
    )
    ap.add_argument("--conf-thres", type=float, default=None, help="Override detector confidence threshold.")
    ap.add_argument("--conf-min", type=float, default=None, help="Override minimum confidence used when selecting the tracked target.")
    ap.add_argument("--detector-max-det", type=int, default=None, help="Override the maximum number of detector candidates considered per frame.")
    ap.add_argument("--max-jump-px", type=float, default=None, help="Override the maximum allowed target-center jump in pixels.")
    ap.add_argument(
        "--max-jump-diag-frac",
        type=float,
        default=None,
        help="Override the maximum allowed target-center jump as a fraction of the image diagonal when --max-jump-px is unset.",
    )
    ap.add_argument("--max-lost", type=int, default=None, help="Override the number of consecutive lost frames tolerated before reset logic applies.")
    ap.add_argument("--min-iou-same-track", type=float, default=None, help="Override the minimum IoU required to stay on the same track after lock.")
    ap.add_argument("--max-box-area-ratio", type=float, default=None, help="Override the allowed box-area ratio change when staying on the same track.")
    ap.add_argument(
        "--acquire-min-box-area-ratio",
        type=float,
        default=None,
        help="Override the minimum prelock candidate box-area ratio used by the Camera1 foreground guard.",
    )
    ap.add_argument(
        "--acquire-bottom-margin-px",
        type=float,
        default=None,
        help="Override the bottom-edge margin used by the Camera1 foreground guard.",
    )
    ap.add_argument(
        "--lock-delay-frames",
        type=int,
        default=None,
        help="Override the number of initial frames to wait before making the first lock permanent.",
    )
    ap.add_argument("--target-x-frac", type=float, default=None, help="Override the horizontal target-acquisition anchor as a fraction of image width.")
    ap.add_argument("--target-y-frac", type=float, default=None, help="Override the vertical target-acquisition anchor as a fraction of image height.")
    ap.add_argument(
        "--no-suspicious",
        action="store_true",
        help=(
            "Treat known Camera2 background regions as a switch barrier. "
            "Tracking prefers candidates outside those regions, and will leave frames empty "
            "instead of switching into them unless continuity is very strong."
        ),
    )
    ap.add_argument(
        "--allow-region1-start",
        action="store_true",
        help=(
            "When --no-suspicious is enabled, allow initial target acquisition to start "
            "inside suspicious Region 1 (the right-side box) instead of waiting for a "
            "non-suspicious target."
        ),
    )
    ap.add_argument(
        "--allow-region2-start",
        action="store_true",
        help=(
            "When --no-suspicious is enabled, allow initial target acquisition to start "
            "inside suspicious Region 2 (the middle/early-clip box) instead of waiting for a "
            "non-suspicious target."
        ),
    )
    ap.add_argument("--lock-first-target", dest="lock_first_target", action="store_true", help="Lock onto the first acquired target.")
    ap.add_argument("--no-lock-first-target", dest="lock_first_target", action="store_false", help="Disable permanent first-target locking.")
    ap.add_argument("--strict-reacquire", dest="strict_reacquire", action="store_true", help="Require IoU and area-ratio consistency when reacquiring the locked target.")
    ap.add_argument("--no-strict-reacquire", dest="strict_reacquire", action="store_false", help="Disable strict locked-target reacquisition checks.")
    ap.add_argument("--reset-on-max-lost", dest="reset_on_max_lost", action="store_true", help="Reset the tracked target after too many consecutive lost frames.")
    ap.add_argument("--no-reset-on-max-lost", dest="reset_on_max_lost", action="store_false", help="Disable reset after too many consecutive lost frames.")
    ap.set_defaults(
        lock_first_target=None,
        strict_reacquire=None,
        reset_on_max_lost=None,
        camera1_foreground_guard=None,
    )
    args = ap.parse_args()

    upfall_root = args.upfall_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()

    if not upfall_root.exists() or not upfall_root.is_dir():
        raise SystemExit(f"UP-Fall root does not exist or is not a directory: {upfall_root}")

    if __package__:
        from .pose import (
            PoseExportConfig,
            apply_pose_lock_settings,
            run_pose_on_frames,
        )
    else:
        # Allow direct execution: `python dataset_helpers/get_keypoints_files.py ...`
        from pose import (
            PoseExportConfig,
            apply_pose_lock_settings,
            run_pose_on_frames,
        )

    cfg = PoseExportConfig(
        model_path=str(model_path),
        fps=30,
        max_people=1,  # export one tracked person
        save_csv=False,
        imgsz=args.imgsz,
    )
    apply_pose_lock_settings(cfg, args.lock_settings)
    cfg.no_suspicious = bool(args.no_suspicious and int(args.camera) == 2)
    cfg.allow_region1_start = bool(args.allow_region1_start and int(args.camera) == 2)
    cfg.allow_region2_start = bool(args.allow_region2_start and int(args.camera) == 2)

    if int(args.camera) == 1:
        if args.camera1_foreground_guard is None:
            cfg.prefer_foreground_on_acquire = True
            if args.lock_delay_frames is None:
                cfg.lock_delay_frames = 30
        else:
            cfg.prefer_foreground_on_acquire = bool(args.camera1_foreground_guard)

    overrides = {
        "conf_thres": args.conf_thres,
        "conf_min": args.conf_min,
        "detector_max_det": args.detector_max_det,
        "max_jump_px": args.max_jump_px,
        "max_jump_diag_frac": args.max_jump_diag_frac,
        "max_lost": args.max_lost,
        "min_iou_same_track": args.min_iou_same_track,
        "max_box_area_ratio": args.max_box_area_ratio,
        "acquire_min_box_area_ratio": args.acquire_min_box_area_ratio,
        "acquire_bottom_margin_px": args.acquire_bottom_margin_px,
        "lock_delay_frames": args.lock_delay_frames,
        "target_x_frac": args.target_x_frac,
        "target_y_frac": args.target_y_frac,
        "lock_first_target": args.lock_first_target,
        "strict_reacquire": args.strict_reacquire,
        "reset_on_max_lost": args.reset_on_max_lost,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(cfg, field_name, value)
    cfg.model_path = str(model_path)

    camera_folders = find_camera_folders_subjects(
        root=str(upfall_root),
        camera=args.camera,
        subjects=args.subjects,
    )

    print(f"UP-Fall root: {upfall_root}")
    print(f"Output root: {output_root}")
    print(f"Model path: {model_path}")
    print(f"Lock settings preset: {args.lock_settings}")
    print(f"No suspicious switching: {cfg.no_suspicious}")
    print(f"Allow Region1 start: {cfg.allow_region1_start}")
    print(f"Allow Region2 start: {cfg.allow_region2_start}")
    print(f"Prefer foreground on acquire: {cfg.prefer_foreground_on_acquire}")
    print(f"Acquire min box-area ratio: {cfg.acquire_min_box_area_ratio}")
    print(f"Acquire bottom margin px: {cfg.acquire_bottom_margin_px}")
    print(f"Lock delay frames: {cfg.lock_delay_frames}")
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

        _, out_npz, _ = run_pose_on_frames(
            frames_dir=frames_dir,
            out_dir=str(out_dir),
            windows_csv=windows_csv,
            config=cfg,
            pattern="*.png",
        )

        print(f"  -> wrote {out_npz}")
        results.append(out_npz)

    print("\nDone.")
    print("Processed sequences:", len(results))


if __name__ == "__main__":
    main()
