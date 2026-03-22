#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import pickle
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
from inference.classifier_adapters import (
    MotionBERTAdapter,
    MotionBERTAdapterConfig,
    pick_device,
    resolve_motionbert_path,
    resolve_motionbert_root,
)
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


def _wants_output(path_arg: Optional[str]) -> bool:
    return path_arg is not None and str(path_arg).strip() != ""


def _build_motionbert_pose_pipeline(
    *,
    args: argparse.Namespace,
    device: str,
    yolo_weights_path: Path,
    frame_step: int,
) -> PosePipeline:
    use_half = bool(int(args.half)) and str(device).startswith("cuda")
    max_det = int(args.max_det) if int(args.max_det) > 0 else int(args.max_people)

    return PosePipeline(
        PosePipelineConfig(
            yolo_weights=yolo_weights_path,
            device=str(device),
            imgsz=int(args.imgsz),
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


def _write_legacy_motionbert_csv(path: Path, window_records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_dir", "start_frame", "end_frame", "pred_id", "pred_name", "pred_conf", "p_fall"])
        for rec in window_records:
            p_fall = rec.prediction.extra.get("p_fall", 0.0) if isinstance(rec.prediction.extra, dict) else 0.0
            writer.writerow(
                [
                    rec.frame_dir,
                    int(rec.window_start_raw),
                    int(rec.window_end_raw),
                    int(rec.prediction.pred_id),
                    str(rec.prediction.pred_label),
                    f"{float(rec.prediction.confidence if rec.prediction.confidence is not None else 0.0):.6f}",
                    f"{float(p_fall):.6f}",
                ]
            )


def _write_legacy_motionbert_pkl(path: Path, window_records) -> None:
    split_list = []
    annotations = []
    for rec in window_records:
        if rec.xy_seq is None or rec.conf_seq is None:
            continue
        frame_dir = str(rec.frame_dir)
        split_list.append(frame_dir)
        annotations.append(
            {
                "frame_dir": frame_dir,
                "total_frames": int(rec.xy_seq.shape[0]),
                "img_shape": (int(rec.image_shape[0]), int(rec.image_shape[1])),
                "keypoint": rec.xy_seq[None, ...].astype(np.float32),
                "keypoint_score": rec.conf_seq[None, ...].astype(np.float32),
                "label": 0,
            }
        )

    dataset = {"split": {"xsub_train": [], "xsub_val": split_list}, "annotations": annotations}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MotionBERT wrapper on shared benchmark core.")
    ap.add_argument("--model", type=str, required=True, help="MotionBERT checkpoint (*.bin) or checkpoint directory")
    ap.add_argument("--config", type=str, default="configs/action/MB_ft_UPFall_xsub_LITE.yaml")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--yolo-weights", type=str, default="pose_models/ultralytics/yolo11l-pose.pt")
    ap.add_argument("--device", type=str, default=None)

    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--conf-thres",
        "--yolo-conf",
        dest="yolo_conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold. --conf-thres kept for backward compatibility.",
    )
    ap.add_argument("--yolo-iou", type=float, default=None)
    ap.add_argument("--half", type=int, default=0, help="Use FP16 for shared YOLO pose path on CUDA (0/1).")

    ap.add_argument("--win-len", type=int, default=None)
    ap.add_argument("--win-step", type=int, default=16)
    ap.add_argument("--frame-step", "--k", type=int, default=1)
    ap.add_argument("--pad-tail", action="store_true")

    ap.add_argument("--missing-conf-thres", type=float, default=0.0)
    ap.add_argument("--keep-empty-windows", action="store_true", default=False)

    ap.add_argument("--out-pkl", type=str, default="outputs/motionbert_video.pkl")
    ap.add_argument("--out-csv", type=str, default="outputs/motionbert_video_preds.csv")
    ap.add_argument("--labels-file", type=str, default=None)
    ap.add_argument("--limit-frames", type=int, default=None)

    ap.add_argument("--display", action="store_true")
    ap.add_argument("--display-conf-thres", type=float, default=0.2)
    ap.add_argument("--display-fps", type=float, default=None)

    ap.add_argument("--profile", type=int, default=0)
    ap.add_argument("--profile-out", type=str, default=None)
    ap.add_argument("--profile-duration-s", type=float, default=0.0)

    ap.add_argument("--benchmark", type=int, default=0, help="Legacy benchmark loop mode (duration-controlled).")
    ap.add_argument("--benchmark-mode", type=int, default=0, help="Headless thesis benchmark mode.")
    ap.add_argument("--hw-sample-hz", type=float, default=1.0, help="Hardware metrics sample rate in Hz for benchmark/profile runs.")
    ap.add_argument("--no-display", type=int, default=0)

    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--no-merge-fall", action="store_true")

    ap.add_argument("--track-conf-min", type=float, default=0.75)
    ap.add_argument("--track-max-jump-px", type=float, default=0.0)
    ap.add_argument("--track-max-jump-diag-frac", type=float, default=0.25)
    ap.add_argument("--track-max-lost", type=int, default=10)
    ap.add_argument("--track-target-x-frac", type=float, default=0.5)
    ap.add_argument("--track-target-y-frac", type=float, default=0.5)
    ap.add_argument("--max-people", type=int, default=10)
    ap.add_argument(
        "--max-det",
        type=int,
        default=0,
        help="YOLO max_det override. <=0 uses --max-people for backward compatibility.",
    )

    ap.add_argument("--warmup-frames", type=int, default=0)
    ap.add_argument("--warmup-windows", type=int, default=0)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()

    frame_step = int(args.frame_step)
    if frame_step <= 0:
        raise ValueError("--frame-step/--k must be >= 1.")
    if int(args.max_people) <= 0:
        raise ValueError("--max-people must be >= 1.")
    if int(args.max_det) < 0:
        raise ValueError("--max-det must be >= 0.")

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

    mb_root = resolve_motionbert_root(_REPO_ROOT)
    video_path = resolve_motionbert_path(args.video, repo_root=_REPO_ROOT, mb_root=mb_root, desc="Video")
    yolo_weights_path = resolve_motionbert_path(args.yolo_weights, repo_root=_REPO_ROOT, mb_root=mb_root, desc="YOLO weights")

    save_path = _resolve_save_path(args.save, video_path)

    # Headless by default unless --display was requested outside benchmark mode.
    no_display = bool(int(args.no_display)) or benchmark_mode or (not bool(args.display))

    adapter = MotionBERTAdapter(
        MotionBERTAdapterConfig(
            model_arg=str(args.model),
            config_arg=str(args.config),
            device=device,
            frame_step=int(frame_step),
            win_len_raw=int(args.win_len) if args.win_len is not None else None,
            win_step_raw=int(args.win_step),
            labels_file=args.labels_file,
            no_merge_fall=bool(args.no_merge_fall),
            missing_conf_thres=float(args.missing_conf_thres),
            repo_root=_REPO_ROOT,
        )
    )

    profile_out_dir: Optional[Path] = None
    if profile_enabled:
        profile_out_dir = pick_profile_out_dir(
            profile_out_arg=args.profile_out,
            save_path=save_path,
            model_tag="motionbert",
            yolo_weights_path=yolo_weights_path,
        )

    pose_pipeline = _build_motionbert_pose_pipeline(
        args=args,
        device=device,
        yolo_weights_path=yolo_weights_path,
        frame_step=int(frame_step),
    )

    display_fps = float(args.display_fps) if args.display_fps is not None else 0.0

    wants_legacy_csv = _wants_output(args.out_csv)
    wants_legacy_pkl = _wants_output(args.out_pkl)
    # Fairness: avoid MotionBERT-only payload copying in benchmark mode.
    retain_payloads_in_primary_run = bool(wants_legacy_pkl and (not benchmark_mode))

    run_cfg = BenchmarkRunConfig(
        video_path=video_path,
        benchmark_name="MotionBERT Inference",
        profile_enabled=bool(profile_enabled),
        profile_out_dir=profile_out_dir,
        profile_duration_s=float(profile_duration_s),
        benchmark_mode=bool(benchmark_mode),
        benchmark_loop_video=bool(benchmark_loop),
        no_display=bool(no_display),
        display_fps=float(display_fps),
        save_path=save_path,
        draw_conf_thres=float(args.display_conf_thres),
        warmup_frames=max(0, int(args.warmup_frames)),
        warmup_windows=max(0, int(args.warmup_windows)),
        limit_frames=None if benchmark_loop else (int(args.limit_frames) if args.limit_frames is not None else None),
        pad_tail=bool(args.pad_tail),
        retain_window_payloads=bool(retain_payloads_in_primary_run),
        hw_sample_hz=float(args.hw_sample_hz),
    )

    result = run_shared_benchmark(
        config=run_cfg,
        pose_pipeline=pose_pipeline,
        classifier=adapter,
    )

    out_csv = Path(args.out_csv).expanduser() if wants_legacy_csv else None
    out_pkl = Path(args.out_pkl).expanduser() if wants_legacy_pkl else None

    if out_csv is not None:
        _write_legacy_motionbert_csv(out_csv, result.window_records)
        print(f"Saved predictions: {out_csv.as_posix()}")

    if out_pkl is not None:
        payload_ready = all((rec.xy_seq is not None and rec.conf_seq is not None) for rec in result.window_records)
        if payload_ready:
            _write_legacy_motionbert_pkl(out_pkl, result.window_records)
            print(f"Saved pkl: {out_pkl.as_posix()}")
        else:
            # Compatibility path: generate payload-heavy artifacts outside timed benchmark run.
            print("[compat] running deferred export pass for --out-pkl (outside benchmark timing).")
            export_pose_pipeline = _build_motionbert_pose_pipeline(
                args=args,
                device=device,
                yolo_weights_path=yolo_weights_path,
                frame_step=int(frame_step),
            )
            export_cfg = BenchmarkRunConfig(
                video_path=video_path,
                benchmark_name="MotionBERT Compatibility Export",
                profile_enabled=False,
                profile_out_dir=None,
                profile_duration_s=0.0,
                benchmark_mode=False,
                benchmark_loop_video=False,
                no_display=True,
                display_fps=0.0,
                save_path=None,
                draw_conf_thres=float(args.display_conf_thres),
                warmup_frames=0,
                warmup_windows=0,
                limit_frames=None if benchmark_loop else (int(args.limit_frames) if args.limit_frames is not None else None),
                pad_tail=bool(args.pad_tail),
                retain_window_payloads=True,
                hw_sample_hz=0.0,
            )
            export_result = run_shared_benchmark(
                config=export_cfg,
                pose_pipeline=export_pose_pipeline,
                classifier=adapter,
            )
            _write_legacy_motionbert_pkl(out_pkl, export_result.window_records)
            print(f"Saved pkl: {out_pkl.as_posix()} (deferred export pass)")

    print(f"Windows: {len(result.window_records)}")

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
