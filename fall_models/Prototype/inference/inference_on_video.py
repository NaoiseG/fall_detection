#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

from inference.benchmark_core import (
    BenchmarkRunConfig,
    assert_benchmark_device_ok,
    get_benchmark_duration_s,
    pick_profile_out_dir,
    run_shared_benchmark,
)
from inference.classifier_adapters import GenericAdapterConfig, GenericTemporalAdapter, KNOWN_ARCHES, pick_device
from inference.pose_pipeline import PosePipeline, PosePipelineConfig


def _resolve_save_path(save_arg: Optional[str], video_path: Path) -> Optional[Path]:
    if not save_arg:
        return None
    p = Path(save_arg).expanduser()
    if str(save_arg).endswith(("/", "\\")) or (p.exists() and p.is_dir()):
        p = p / f"{video_path.stem}_annotated.mp4"
    if p.suffix == "":
        p = p.with_suffix(".mp4")
    return p


def _build_imgsz_run_tag(imgsz_value: Optional[float]) -> Optional[str]:
    if imgsz_value is None:
        return None
    value = float(imgsz_value)
    if not np.isfinite(value) or value <= 0.0:
        return None
    if value <= 1.0:
        return f"imgsz_{value:g}"
    rounded = int(round(value))
    if rounded == 640:
        return None
    return f"imgsz_{rounded}"


def _persist_pose_imgsz_summary(result, pose_pipeline: PosePipeline) -> None:
    info = pose_pipeline.predict_imgsz_info
    applied_hw = info.get("applied_hw")
    result.summary["pose_predict_stride"] = int(pose_pipeline.predict_stride)
    result.summary["pose_predict_imgsz_cli"] = (
        float(pose_pipeline.config.imgsz) if pose_pipeline.config.imgsz is not None else None
    )
    result.summary["pose_predict_imgsz_mode"] = str(info.get("mode", "default"))
    result.summary["pose_predict_imgsz_requested"] = info.get("requested_value")
    result.summary["pose_predict_imgsz_applied_hw"] = (
        [int(applied_hw[0]), int(applied_hw[1])] if applied_hw is not None else None
    )
    result.summary["pose_predict_pixel_ratio_applied"] = float(info.get("applied_ratio", 1.0))
    result.summary["pose_predict_manual_letterbox"] = bool(applied_hw is not None)
    if result.summary_json is not None:
        with result.summary_json.open("w", encoding="utf-8") as f:
            json.dump(result.summary, f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Shared temporal inference wrapper (CNN-LSTM / ST-GCN / compatible arches).")
    ap.add_argument("--video", type=str, required=True, help="Path to input .mp4")
    ap.add_argument("--model", type=str, required=True, help="Checkpoint *.pt/*.pkl OR model folder OR model .py")
    ap.add_argument("--arch", type=str, default=None, choices=KNOWN_ARCHES, help="Override model architecture if needed")

    ap.add_argument("--yolo-weights", type=str, default="pose_models/ultralytics/yolo11l-pose.pt")
    ap.add_argument(
        "--imgsz",
        type=float,
        default=640,
        help=(
            "Optional YOLO inference input size control. "
            "Values in (0,1] are interpreted as a fraction of original input pixels, "
            "e.g. --imgsz 0.9 targets 90%% of the original input pixels. "
            "Values >1 are treated as an explicit square YOLO imgsz."
        ),
    )
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--yolo-iou", type=float, default=None)
    ap.add_argument("--max-people", type=int, default=10)
    ap.add_argument(
        "--max-det",
        type=int,
        default=0,
        help="YOLO max_det override. <=0 uses --max-people for backward compatibility.",
    )

    ap.add_argument("--track-conf-min", type=float, default=0.75)
    ap.add_argument("--track-max-jump-px", type=float, default=0.0)
    ap.add_argument("--track-max-jump-diag-frac", type=float, default=0.25)
    ap.add_argument("--track-max-lost", type=int, default=10)
    ap.add_argument("--track-target-x-frac", type=float, default=0.5)
    ap.add_argument("--track-target-y-frac", type=float, default=0.5)

    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--half", type=int, default=0)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--frame-step", "--k", type=int, default=1)

    ap.add_argument("--normalize-mode", type=str, default=None, choices=["center_scale", "paper_rp"])
    ap.add_argument("--missing-mode", type=str, default=None, choices=["conf_thres", "zeros_only", "conf_or_zeros"])
    ap.add_argument("--interp-mode", type=str, default=None, choices=["short_gap_hold", "paper_group_linear"])
    ap.add_argument("--interp-group", type=int, default=0)
    ap.add_argument("--rp-center-mode", type=str, default=None, choices=["auto", "normalized_01", "pixel"])
    ap.add_argument("--rp-img-w", type=int, default=0)
    ap.add_argument("--rp-img-h", type=int, default=0)
    ap.add_argument("--labels-file", type=str, default=None)

    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--display-fps", type=float, default=0.0)
    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--profile-out", type=str, default=None)
    ap.add_argument("--profile-duration-s", type=float, default=0.0)

    ap.add_argument("--benchmark", type=int, default=0, help="Legacy benchmark loop mode (duration-controlled).")
    ap.add_argument("--benchmark-mode", type=int, default=0, help="Headless thesis benchmarking mode.")
    ap.add_argument("--hw-sample-hz", type=float, default=1.0, help="Hardware metrics sample rate in Hz for benchmark/profile runs.")
    ap.add_argument("--no-display", type=int, default=0)

    ap.add_argument("--warmup-frames", type=int, default=0)
    ap.add_argument("--warmup-windows", type=int, default=0)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    frame_step = int(args.frame_step)
    if frame_step <= 0:
        raise ValueError("--frame-step must be >= 1.")
    if int(args.max_people) <= 0:
        raise ValueError("--max-people must be >= 1.")
    if int(args.max_det) < 0:
        raise ValueError("--max-det must be >= 0.")
    imgsz_value = float(args.imgsz)
    if (not np.isfinite(imgsz_value)) or imgsz_value <= 0.0:
        raise ValueError("--imgsz must be a finite number > 0.")

    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"--video not found: {video_path}")

    yolo_weights_path = Path(args.yolo_weights).expanduser()
    if not yolo_weights_path.exists():
        raise FileNotFoundError(f"--yolo-weights not found: {yolo_weights_path}")

    device = pick_device(args.device)

    benchmark_loop = bool(int(args.benchmark))
    benchmark_mode = bool(int(args.benchmark_mode)) or benchmark_loop
    profile_enabled = bool(int(args.profile)) or benchmark_mode

    try:
        assert_benchmark_device_ok(benchmark=benchmark_loop, device=device)
    except ValueError as e:
        print(f"[benchmark][ERROR] {e}", file=sys.stderr)
        return 2

    if benchmark_loop:
        try:
            profile_duration_s = get_benchmark_duration_s(default_s=600.0)
        except ValueError as e:
            print(f"[benchmark][ERROR] {e}", file=sys.stderr)
            return 2
    else:
        profile_duration_s = max(0.0, float(args.profile_duration_s))

    save_path = _resolve_save_path(args.save, video_path)

    # Benchmark mode is headless by default. Saving is still allowed when explicitly requested via --save.
    no_display = bool(int(args.no_display)) or benchmark_mode

    adapter = GenericTemporalAdapter(
        GenericAdapterConfig(
            model_arg=str(args.model),
            arch_arg=args.arch,
            device=device,
            frame_step=int(frame_step),
            half=bool(int(args.half)),
            T_override=int(args.T),
            stride_override=int(args.stride),
            normalize_mode_override=args.normalize_mode,
            missing_mode_override=args.missing_mode,
            interp_mode_override=args.interp_mode,
            interp_group_override=int(args.interp_group),
            rp_center_mode_override=args.rp_center_mode,
            rp_img_w_override=int(args.rp_img_w),
            rp_img_h_override=int(args.rp_img_h),
            labels_file=args.labels_file,
        )
    )

    profile_out_dir: Optional[Path] = None
    if profile_enabled:
        profile_out_dir = pick_profile_out_dir(
            profile_out_arg=args.profile_out,
            save_path=save_path,
            model_tag=str(adapter.name),
            yolo_weights_path=yolo_weights_path,
            run_tag=_build_imgsz_run_tag(imgsz_value),
        )

    use_half = bool(int(args.half)) and str(device).startswith("cuda")
    max_det = int(args.max_det) if int(args.max_det) > 0 else int(args.max_people)

    pose_pipeline = PosePipeline(
        PosePipelineConfig(
            yolo_weights=yolo_weights_path,
            device=str(device),
            imgsz=imgsz_value,
            yolo_conf=float(args.yolo_conf),
            yolo_iou=float(args.yolo_iou) if args.yolo_iou is not None else None,
            max_det=int(max_det),
            use_half=bool(use_half),
            frame_step=int(frame_step),
            track_conf_min=float(args.track_conf_min),
            track_max_jump_px=float(args.track_max_jump_px),
            track_max_jump_diag_frac=float(args.track_max_jump_diag_frac),
            track_max_lost=int(args.track_max_lost),
            track_target_x_frac=float(args.track_target_x_frac),
            track_target_y_frac=float(args.track_target_y_frac),
        )
    )

    run_cfg = BenchmarkRunConfig(
        video_path=video_path,
        benchmark_name="inference_on_video",
        profile_enabled=bool(profile_enabled),
        profile_out_dir=profile_out_dir,
        profile_duration_s=float(profile_duration_s),
        benchmark_mode=bool(benchmark_mode),
        benchmark_loop_video=bool(benchmark_loop),
        no_display=bool(no_display),
        display_fps=float(args.display_fps),
        save_path=save_path,
        draw_conf_thres=float(getattr(adapter, "conf_thres", 0.2)),
        warmup_frames=max(0, int(args.warmup_frames)),
        warmup_windows=max(0, int(args.warmup_windows)),
        limit_frames=None,
        pad_tail=False,
        retain_window_payloads=False,
        hw_sample_hz=float(args.hw_sample_hz),
    )

    result = run_shared_benchmark(
        config=run_cfg,
        pose_pipeline=pose_pipeline,
        classifier=adapter,
    )
    _persist_pose_imgsz_summary(result, pose_pipeline)

    if profile_enabled and result.profile_out_dir is not None:
        print(f"[profile] wrote outputs to: {result.profile_out_dir.as_posix()}")

    avg_fps = result.summary.get("avg_fps", None)
    if avg_fps is not None:
        try:
            print(f"[summary] avg_fps={float(avg_fps):.3f} windows={int(result.summary.get('num_windows_evaluated', 0))}")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
