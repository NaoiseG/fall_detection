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
    ap.add_argument(
        "--alphapose-root",
        type=Path,
        default=Path("pose_models/AlphaPose"),
        help="Path to the AlphaPose repo root (default: pose_models/AlphaPose).",
    )
    ap.add_argument(
        "--cfg-path",
        type=Path,
        default=Path("configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"),
        help="AlphaPose config path, absolute or relative to --alphapose-root.",
    )
    ap.add_argument(
        "--fastpose-weights",
        type=Path,
        default=Path("pretrained_models/fast_res50_256x192.pth"),
        help="FastPose weights path (.pth or TensorRT .engine), absolute or relative to --alphapose-root.",
    )
    ap.add_argument(
        "--detector-cfg",
        type=Path,
        default=Path("detector/yolo/cfg/yolov3-spp.cfg"),
        help="YOLOv3-SPP detector cfg path, absolute or relative to --alphapose-root.",
    )
    ap.add_argument(
        "--detector-weights",
        type=Path,
        default=Path("detector/yolo/data/yolov3-spp.weights"),
        help="YOLOv3-SPP detector weights path (.weights or TensorRT .engine), absolute or relative to --alphapose-root.",
    )
    ap.add_argument(
        "--lock-settings",
        choices=("strict_lock", "default"),
        default="strict_lock",
        help="Tracking/lock preset. 'strict_lock' matches the settings used by get_keypoints_files.py. Default: strict_lock.",
    )
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
    ap.add_argument(
        "--lock-after-foreground-frames",
        type=int,
        default=None,
        help=(
            "Override the number of consecutive prelock foreground selections required "
            "before making the first lock permanent."
        ),
    )
    ap.add_argument(
        "--lock-delay-frames",
        type=int,
        default=None,
        help="Override an additional minimum frame index before making the first lock permanent.",
    )
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
    ap.set_defaults(camera1_foreground_guard=None)
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
        alphapose_root=str(args.alphapose_root),
        cfg_path=str(args.cfg_path),
        checkpoint=str(args.fastpose_weights),
        detector_cfg=str(args.detector_cfg),
        detector_weights=str(args.detector_weights),
        fps=30,
        max_people=1,
        save_csv=False,
        render_video=True,
    )

    # Apply lock-settings preset first, then per-field overrides.
    _ALPHAPOSE_LOCK_PRESETS = {
        "default": {},
        "strict_lock": {
            "conf_thres": 0.01,
            "conf_min": 0.01,
            "max_jump_px": None,
            "max_jump_diag_frac": 0.12,
            "max_lost": 60,
            "switch_margin_px": 9999.0,
            "reset_on_max_lost": False,
            "lock_first_target": True,
            "strict_reacquire": True,
            "min_iou_same_track": 0.05,
            "max_box_area_ratio": 2.5,
        },
    }
    for field_name, value in _ALPHAPOSE_LOCK_PRESETS[args.lock_settings].items():
        setattr(cfg, field_name, value)

    cfg.no_suspicious = bool(args.no_suspicious and int(args.camera) == 2)
    cfg.allow_region1_start = bool(args.allow_region1_start and int(args.camera) == 2)
    cfg.allow_region2_start = bool(args.allow_region2_start and int(args.camera) == 2)

    if int(args.camera) == 1:
        if args.camera1_foreground_guard is None:
            cfg.prefer_foreground_on_acquire = True
            if args.lock_after_foreground_frames is None:
                cfg.lock_after_foreground_frames = 10
        else:
            cfg.prefer_foreground_on_acquire = bool(args.camera1_foreground_guard)

    overrides = {
        "lock_after_foreground_frames": args.lock_after_foreground_frames,
        "lock_delay_frames": args.lock_delay_frames,
        "acquire_min_box_area_ratio": args.acquire_min_box_area_ratio,
        "acquire_bottom_margin_px": args.acquire_bottom_margin_px,
    }
    for field_name, value in overrides.items():
        if value is not None:
            setattr(cfg, field_name, value)

    runner = AlphaPoseRunner(cfg)

    camera_folders = find_camera_folders_subjects(
        root=str(upfall_root),
        camera=args.camera,
        subjects=args.subjects,
    )

    print(f"UP-Fall root: {upfall_root}")
    print(f"Output root: {output_root}")
    print(f"AlphaPose root: {args.alphapose_root}")
    print(f"FastPose weights: {args.fastpose_weights}")
    print(f"Detector weights: {args.detector_weights}")
    print(f"Lock settings preset: {args.lock_settings}")
    print(f"No suspicious switching: {cfg.no_suspicious}")
    print(f"Allow Region1 start: {cfg.allow_region1_start}")
    print(f"Allow Region2 start: {cfg.allow_region2_start}")
    print(f"Prefer foreground on acquire: {cfg.prefer_foreground_on_acquire}")
    print(f"Lock after foreground frames: {cfg.lock_after_foreground_frames}")
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
