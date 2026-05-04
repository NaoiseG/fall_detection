#!/usr/bin/env python3
"""
Rank URFD fall videos by timing-aware window-label accuracy.

This is a focused companion to evaluate_urfd.py. It reuses the same URFD
sequence discovery, YOLO/keypoint pipeline, classifier adapters, temporal
windowing, and URFD fall timing labels, then prints the top fall videos by
per-window predicted-label accuracy.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_repo_root_str = str(_REPO_ROOT)
if _repo_root_str not in sys.path:
    sys.path.insert(0, _repo_root_str)

import evaluation.evaluate_urfd as urfd_eval

LOGGER = logging.getLogger("top_urfd_fall_videos")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing final keypoint + temporal classifier pipeline on "
            "URFD and print the fall videos with highest timing-aware "
            "window-label accuracy."
        )
    )
    parser.add_argument(
        "--urfd-root",
        type=Path,
        required=True,
        help="Path to the URFD root containing ADLs/, Falls/, and optionally cvat/.",
    )
    parser.add_argument(
        "--keypoint-weights",
        type=Path,
        required=True,
        help="Path to pose/keypoint weights used by the final pipeline (.pt or .engine).",
    )
    parser.add_argument(
        "--classifier-model",
        type=Path,
        required=True,
        help="Path to the trained temporal classifier checkpoint (.pt).",
    )
    parser.add_argument("--device", type=str, default=None, help="Inference device. Defaults to CUDA if available, else CPU.")
    parser.add_argument("--window-size", type=int, default=None, help="Optional raw-frame temporal window length override.")
    parser.add_argument("--stride", type=int, default=None, help="Optional raw-frame temporal stride override.")
    parser.add_argument("--fps", type=float, default=urfd_eval.DEFAULT_FPS, help="URFD FPS used for timestamps. Default: 30.")
    parser.add_argument("--threshold", type=float, default=urfd_eval.DEFAULT_THRESHOLD, help="Fall-score threshold. Default: 0.5.")
    parser.add_argument(
        "--gt-phase-mode",
        type=str,
        default="transition_and_post",
        choices=urfd_eval.GT_PHASE_MODE_CHOICES,
        help="URFD timing CSV phase rule. Default: transition_and_post.",
    )
    parser.add_argument(
        "--pad-short-clips",
        type=str,
        default="none",
        choices=urfd_eval.PAD_SHORT_CLIPS_CHOICES,
        help="Optional padding for clips shorter than the classifier window length. Default: none.",
    )
    parser.add_argument("--frame-exts", nargs="*", default=None, help="Optional frame extensions. Defaults to png jpg jpeg.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of fall videos to print. Default: 10.")
    parser.add_argument("--test", action="store_true", help="Run a small sanity-check subset of URFD videos.")
    parser.add_argument("--arch", type=str, default=None, help="Optional classifier architecture override.")
    parser.add_argument(
        "--motionbert-config",
        type=str,
        default=urfd_eval.DEFAULT_MOTIONBERT_CONFIG,
        help=f"MotionBERT config path. Default: {urfd_eval.DEFAULT_MOTIONBERT_CONFIG}.",
    )
    parser.add_argument("--imgsz", type=float, default=None, help="Optional YOLO predict size override.")
    parser.add_argument("--half", type=int, choices=(0, 1), default=None, help="Optional FP16 toggle for the pose model.")
    parser.add_argument("--frame-step", type=int, default=1, help="Raw-frame subsampling factor before windowing. Default: 1.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO pose confidence threshold. Default: 0.25.")
    parser.add_argument("--yolo-iou", type=float, default=None, help="Optional YOLO NMS IoU threshold.")
    parser.add_argument("--max-people", type=int, default=10, help="Maximum people to consider. Default: 10.")
    parser.add_argument("--max-det", type=int, default=0, help="Optional YOLO max_det override. Values <= 0 reuse --max-people.")
    parser.add_argument("--track-conf-min", type=float, default=0.75, help="Tracking confidence minimum. Default: 0.75.")
    parser.add_argument("--track-max-jump-px", type=float, default=0.0, help="Absolute max tracking jump in pixels.")
    parser.add_argument("--track-max-jump-diag-frac", type=float, default=0.25, help="Fallback max jump as frame-diagonal fraction.")
    parser.add_argument("--track-max-lost", type=int, default=10, help="Sampled misses before tracker reset. Default: 10.")
    parser.add_argument("--track-target-x-frac", type=float, default=0.5, help="Tracking target x-position fraction. Default: 0.5.")
    parser.add_argument("--track-target-y-frac", type=float, default=0.5, help="Tracking target y-position fraction. Default: 0.5.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Console log verbosity. Default: INFO.",
    )
    return parser


def normalize_prediction_label(row: Dict[str, Any], threshold: float) -> str:
    """Use the evaluator's predicted label, with score threshold as fallback."""
    predicted_label = str(row.get("predicted_label", "")).strip().lower()
    if predicted_label in {"fall", "non_fall"}:
        return predicted_label
    return "fall" if float(row.get("fall_score", 0.0) or 0.0) >= float(threshold) else "non_fall"


def timing_aware_true_label(row: Dict[str, Any]) -> Optional[str]:
    """Match evaluate_urfd.py's timing-aware window ground-truth rule."""
    video_label = str(row.get("video_label", "")).strip().lower()
    if video_label == "non_fall":
        return "non_fall"
    if video_label != "fall":
        return None
    if not bool(row.get("timing_annotation_available", False)):
        return None
    return "fall" if bool(row.get("overlaps_annotated_fall_event", False)) else "non_fall"


def summarize_fall_video_accuracy(
    sequence: urfd_eval.URFDSequence,
    window_rows: Sequence[Dict[str, Any]],
    *,
    threshold: float,
) -> Dict[str, Any]:
    eligible_rows: List[Tuple[str, str]] = []
    gt_fall_windows = 0
    predicted_fall_windows = 0

    for row in window_rows:
        true_label = timing_aware_true_label(row)
        if true_label is None:
            continue
        predicted_label = normalize_prediction_label(row, threshold)
        eligible_rows.append((true_label, predicted_label))
        if true_label == "fall":
            gt_fall_windows += 1
        if predicted_label == "fall":
            predicted_fall_windows += 1

    correct = sum(1 for true_label, predicted_label in eligible_rows if true_label == predicted_label)
    total = len(eligible_rows)
    accuracy = (float(correct) / float(total)) if total > 0 else None

    return {
        "video_id": sequence.video_id,
        "video_path": str(sequence.frame_dir),
        "accuracy": accuracy,
        "correct_windows": int(correct),
        "total_windows": int(total),
        "gt_fall_windows": int(gt_fall_windows),
        "predicted_fall_windows": int(predicted_fall_windows),
    }


def print_top_table(rows: Sequence[Dict[str, Any]], *, top_k: int) -> None:
    headers = [
        "rank",
        "video_id",
        "accuracy",
        "correct/total",
        "gt_fall_windows",
        "pred_fall_windows",
        "video_path",
    ]
    table_rows: List[List[str]] = []
    for idx, row in enumerate(rows[: max(0, int(top_k))], start=1):
        accuracy = row.get("accuracy")
        table_rows.append(
            [
                str(idx),
                str(row["video_id"]),
                "n/a" if accuracy is None else f"{float(accuracy) * 100.0:.2f}%",
                f"{int(row['correct_windows'])}/{int(row['total_windows'])}",
                str(row["gt_fall_windows"]),
                str(row["predicted_fall_windows"]),
                str(row["video_path"]),
            ]
        )

    widths = [
        max(len(headers[col]), *(len(row[col]) for row in table_rows)) if table_rows else len(headers[col])
        for col in range(len(headers))
    ]
    print()
    print("Top URFD Fall Videos by Window Accuracy")
    print("  Ground truth: fall windows overlap the URFD annotated fall interval; other windows are non_fall.")
    print()
    print("  " + "  ".join(headers[col].ljust(widths[col]) for col in range(len(headers))))
    print("  " + "  ".join("-" * widths[col] for col in range(len(headers))))
    for row in table_rows:
        print("  " + "  ".join(row[col].ljust(widths[col]) for col in range(len(headers))))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    urfd_eval.configure_logging(str(args.log_level))

    if urfd_eval.IMPORT_ERROR is not None:
        raise RuntimeError(f"Failed to import runtime dependencies required by evaluate_urfd.py: {urfd_eval.IMPORT_ERROR}")

    urfd_root = args.urfd_root.expanduser().resolve()
    keypoint_weights = args.keypoint_weights.expanduser().resolve()
    classifier_model = args.classifier_model.expanduser().resolve()
    args.urfd_root = urfd_root
    args.keypoint_weights = keypoint_weights
    args.classifier_model = classifier_model

    frame_exts = urfd_eval.normalize_frame_exts(args.frame_exts)
    sequences_all = urfd_eval.discover_urfd_sequences(urfd_root, frame_exts)
    if not sequences_all:
        raise RuntimeError(f"No URFD sequences found under {urfd_root}")
    sequences = urfd_eval.select_test_sequences(sequences_all) if bool(args.test) else sequences_all

    fall_sequences = [seq for seq in sequences if seq.video_label == "fall"]
    LOGGER.info("Selected %d URFD videos (%d fall videos).", len(sequences), len(fall_sequences))

    timing_annotations = urfd_eval.load_urfd_fall_timing_annotations(
        urfd_root,
        gt_phase_mode=str(args.gt_phase_mode),
    )
    if not timing_annotations:
        LOGGER.warning(
            "No URFD timing annotations were loaded. Fall videos need timing annotations for per-window accuracy."
        )

    device = str(args.device or urfd_eval.pick_device())
    resolved_imgsz = float(args.imgsz) if args.imgsz is not None else float(urfd_eval.infer_imgsz_from_path(keypoint_weights) or 640.0)
    classifier = urfd_eval.build_classifier_adapter(args, device=device)
    pose_pipeline = urfd_eval.build_pose_pipeline(args, device=device, keypoint_weights=keypoint_weights, imgsz=resolved_imgsz)
    strict_late_tolerance_frames = urfd_eval.resolve_strict_late_tolerance_frames(
        None,
        int(classifier.window_policy.raw_window_len),
    )

    start_time = time.perf_counter()
    summaries: List[Dict[str, Any]] = []
    for sequence in urfd_eval.iter_progress(sequences, desc="URFD videos", total=len(sequences)):
        timing_annotation = None
        if sequence.video_label == "fall":
            timing_annotation = timing_annotations.get(urfd_eval.infer_urfd_csv_video_id(sequence.video_id))
        window_rows, _video_summary = urfd_eval.evaluate_sequence(
            sequence,
            pose_pipeline=pose_pipeline,
            classifier=classifier,
            timing_annotation=timing_annotation,
            fps=float(args.fps),
            threshold=float(args.threshold),
            min_consecutive_positive=1,
            gt_phase_mode=str(args.gt_phase_mode),
            decision_mode="max_score",
            score_smoothing="none",
            score_smoothing_window=1,
            score_smoothing_alpha=0.4,
            pad_short_clips=str(args.pad_short_clips),
            strict_early_tolerance_frames=0,
            strict_late_tolerance_frames=int(strict_late_tolerance_frames),
        )
        if sequence.video_label == "fall":
            summaries.append(
                summarize_fall_video_accuracy(
                    sequence,
                    window_rows,
                    threshold=float(args.threshold),
                )
            )

    ranked = [
        row for row in summaries
        if row.get("accuracy") is not None and int(row.get("total_windows", 0)) > 0
    ]
    ranked.sort(
        key=lambda row: (
            -float(row["accuracy"]),
            -int(row["correct_windows"]),
            -int(row["total_windows"]),
            str(row["video_id"]),
        )
    )

    print_top_table(ranked, top_k=int(args.top_k))
    print()
    print(f"Evaluated videos: {len(sequences)}")
    print(f"Fall videos ranked: {len(ranked)} / {len(fall_sequences)}")
    print(f"Elapsed: {time.perf_counter() - start_time:.2f}s")
    if len(ranked) < len(fall_sequences):
        print("Skipped fall videos had no timing-aware window ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
