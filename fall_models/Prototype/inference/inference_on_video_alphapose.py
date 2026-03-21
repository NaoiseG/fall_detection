#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

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
from inference.pose_pipeline_alphapose import AlphaPosePipeline, AlphaPosePipelineConfig


def _resolve_save_path(save_arg: Optional[str], video_path: Path) -> Optional[Path]:
    if not save_arg:
        return None
    p = Path(save_arg).expanduser()
    if str(save_arg).endswith(("/", "\\")) or (p.exists() and p.is_dir()):
        p = p / f"{video_path.stem}_annotated.mp4"
    if p.suffix == "":
        p = p.with_suffix(".mp4")
    return p


def _resolve_repo_path(path_arg: str, desc: str) -> Path:
    p = Path(path_arg).expanduser()
    if p.exists():
        return p
    repo_rel = (_REPO_ROOT / path_arg).expanduser()
    if repo_rel.exists():
        return repo_rel
    raise FileNotFoundError(f"{desc} not found: {path_arg}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Shared temporal inference wrapper using AlphaPose keypoints.")
    ap.add_argument("--video", type=str, required=True, help="Path to input .mp4")
    ap.add_argument("--model", type=str, required=True, help="Checkpoint *.pt/*.pkl OR model folder OR model .py")
    ap.add_argument("--arch", type=str, default=None, choices=KNOWN_ARCHES, help="Override model architecture if needed")

    ap.add_argument("--alphapose-root", type=str, default="pose_models/AlphaPose")
    ap.add_argument("--alphapose-cfg", type=str, default="configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml")
    ap.add_argument("--alphapose-checkpoint", type=str, default="pretrained_models/fast_res50_256x192.pth")
    ap.add_argument("--alphapose-detector-cfg", type=str, default="detector/yolo/cfg/yolov3-spp.cfg")
    ap.add_argument("--alphapose-detector-weights", type=str, default="detector/yolo/data/yolov3-spp.weights")
    ap.add_argument("--alphapose-conf", "--conf-thres", dest="alphapose_conf", type=float, default=0.10)
    ap.add_argument("--alphapose-nms", "--nms-thres", dest="alphapose_nms", type=float, default=0.60)
    ap.add_argument("--alphapose-flip", type=int, default=0)
    ap.add_argument("--min-box-area", type=int, default=0)
    ap.add_argument(
        "--max-det",
        type=int,
        default=0,
        help="AlphaPose post-NMS candidate cap. <=0 keeps all candidates.",
    )

    ap.add_argument("--track-conf-min", type=float, default=0.75)
    ap.add_argument("--track-max-jump-px", type=float, default=0.0)
    ap.add_argument("--track-max-jump-diag-frac", type=float, default=0.25)
    ap.add_argument("--track-max-lost", type=int, default=10)
    ap.add_argument("--track-target-x-frac", type=float, default=0.5)
    ap.add_argument("--track-target-y-frac", type=float, default=0.5)

    ap.add_argument("--device", type=str, default=None)
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
    ap.add_argument("--hw-sample-hz", type=float, default=1.0, help="Accepted for CLI compatibility (not used in shared core).")
    ap.add_argument("--no-display", type=int, default=0)

    ap.add_argument("--warmup-frames", type=int, default=0)
    ap.add_argument("--warmup-windows", type=int, default=0)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    frame_step = int(args.frame_step)
    if frame_step <= 0:
        raise ValueError("--frame-step must be >= 1.")
    if int(args.max_det) < 0:
        raise ValueError("--max-det must be >= 0.")

    video_path = _resolve_repo_path(args.video, desc="--video")
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
    no_display = bool(int(args.no_display)) or benchmark_mode

    adapter = GenericTemporalAdapter(
        GenericAdapterConfig(
            model_arg=str(args.model),
            arch_arg=args.arch,
            device=device,
            frame_step=int(frame_step),
            half=False,
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

    pose_pipeline = AlphaPosePipeline(
        AlphaPosePipelineConfig(
            alphapose_root=_resolve_repo_path(args.alphapose_root, desc="--alphapose-root"),
            device=str(device),
            cfg_path=str(args.alphapose_cfg),
            checkpoint=str(args.alphapose_checkpoint),
            detector_cfg=str(args.alphapose_detector_cfg),
            detector_weights=str(args.alphapose_detector_weights),
            alphapose_conf=float(args.alphapose_conf),
            alphapose_nms=float(args.alphapose_nms),
            min_box_area=int(args.min_box_area),
            flip=bool(int(args.alphapose_flip)),
            max_det=int(args.max_det),
            frame_step=int(frame_step),
            track_conf_min=float(args.track_conf_min),
            track_max_jump_px=float(args.track_max_jump_px),
            track_max_jump_diag_frac=float(args.track_max_jump_diag_frac),
            track_max_lost=int(args.track_max_lost),
            track_target_x_frac=float(args.track_target_x_frac),
            track_target_y_frac=float(args.track_target_y_frac),
        )
    )

    profile_out_dir: Optional[Path] = None
    if profile_enabled:
        profile_out_dir = pick_profile_out_dir(
            profile_out_arg=args.profile_out,
            save_path=save_path,
            model_tag=f"{adapter.name}_alphapose",
            yolo_weights_path=pose_pipeline.checkpoint_path,
        )

    run_cfg = BenchmarkRunConfig(
        video_path=video_path,
        benchmark_name="inference_on_video_alphapose",
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
    )

    result = run_shared_benchmark(
        config=run_cfg,
        pose_pipeline=pose_pipeline,
        classifier=adapter,
    )

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
