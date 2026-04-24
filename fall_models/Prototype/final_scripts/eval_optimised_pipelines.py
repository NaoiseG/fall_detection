#!/usr/bin/env python3
"""Evaluate selected optimised YOLO pose pipelines across final classifiers.

This reuses the same classifier evaluation flow as
``final_scripts/eval_downsized_models.py`` but restricts discovery to the
requested optimised keypoint variants:

    /home/jetson/NaoiseG/fall_detection/Datasets/UPFall_keypoints_img_downsize/
      yolo11x-pose/fp32_576
      yolo11x-pose/fp16_576
      yolo11l-pose/fp32_576
      yolo11m-pose/fp32_576

For each discovered variant, this runner evaluates:
  - paper_stgcn
  - cnnlstm
  - MotionBERT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import eval_downsized_models as base


CLASSIFIER_MODELS: Tuple[str, ...] = base.CLASSIFIER_MODELS
SUMMARY_JSON_NAME = base.SUMMARY_JSON_NAME
DEFAULT_MOTIONBERT_EVAL_BATCH_SIZE = base.DEFAULT_MOTIONBERT_EVAL_BATCH_SIZE

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PROTOTYPE_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_KEYPOINTS_ROOT = Path(
    "/home/jetson/NaoiseG/fall_detection/Datasets/UPFall_keypoints_img_downsize"
)
DEFAULT_CLASSIFICATION_ROOT = WORKSPACE_ROOT / "web_app" / "models" / "classification"
DEFAULT_OUTPUT_ROOT = PROTOTYPE_ROOT / "eval_outputs" / "optimised_pipeline_evals"

PIPELINE_SPECS: Tuple[Tuple[str, str, int], ...] = (
    ("yolo11m-pose", "fp32_576", 576),
    ("yolo11l-pose", "fp32_576", 576),
    ("yolo11x-pose", "fp16_576", 576),
    ("yolo11x-pose", "fp32_576", 576),
)


def pipeline_sort_key(variant: base.KeypointVariant) -> Tuple[int, str, str]:
    pose_order = {
        "yolo11m-pose": 0,
        "yolo11l-pose": 1,
        "yolo11x-pose": 2,
    }
    precision_order = {
        "fp32_576": 0,
        "fp16_576": 1,
    }
    return (
        pose_order.get(variant.pose_model_size, 999),
        precision_order.get(variant.variant_name, 999),
        variant.variant_name,
    )


def discover_variants(paths: base.SweepPaths) -> Tuple[List[base.KeypointVariant], List[str]]:
    issues: List[str] = []
    variants: List[base.KeypointVariant] = []

    if not paths.keypoints_root.is_dir():
        issues.append(f"Optimised keypoints root not found: {paths.keypoints_root.as_posix()}")
        return variants, issues

    for pose_dir_name, variant_name, imgsz in PIPELINE_SPECS:
        npz_root = paths.keypoints_root / pose_dir_name / variant_name
        if not npz_root.is_dir():
            issues.append(f"Missing optimised keypoint variant: {npz_root.as_posix()}")
            continue

        variants.append(
            base.KeypointVariant(
                pose_model_size=pose_dir_name,
                checkpoint_tag=pose_dir_name,
                variant_name=variant_name,
                imgsz=imgsz,
                npz_root=npz_root,
            )
        )

    variants.sort(key=pipeline_sort_key)
    return variants, issues


def build_run_matrix(paths: base.SweepPaths, variants: Sequence[base.KeypointVariant]) -> List[base.RunSpec]:
    runs: List[base.RunSpec] = []
    for variant in variants:
        for classifier_model in CLASSIFIER_MODELS:
            checkpoint_path = base.get_checkpoint_path(paths, classifier_model, variant.checkpoint_tag)
            run_dir = paths.output_root / classifier_model / variant.pose_model_size / variant.variant_name
            kwargs = {}
            if classifier_model == "MotionBERT":
                pkl_path, label_map_path = base.get_motionbert_pkl_paths(paths, variant)
                kwargs["motionbert_repo_root"] = paths.motionbert_code_root
                kwargs["motionbert_pkl_path"] = pkl_path
                kwargs["motionbert_label_map_path"] = label_map_path
            runs.append(
                base.RunSpec(
                    classifier_model=classifier_model,
                    pose_model_size=variant.pose_model_size,
                    checkpoint_tag=variant.checkpoint_tag,
                    variant_name=variant.variant_name,
                    imgsz=variant.imgsz,
                    npz_root=variant.npz_root,
                    checkpoint_path=checkpoint_path,
                    run_dir=run_dir,
                    stdout_log_path=run_dir / "stdout.log",
                    stderr_log_path=run_dir / "stderr.log",
                    run_status_path=run_dir / "run_status.json",
                    **kwargs,
                )
            )

    classifier_order = {name: index for index, name in enumerate(CLASSIFIER_MODELS)}
    runs.sort(
        key=lambda item: (
            classifier_order[item.classifier_model],
            pipeline_sort_key(
                base.KeypointVariant(
                    pose_model_size=item.pose_model_size,
                    checkpoint_tag=item.checkpoint_tag,
                    variant_name=item.variant_name,
                    imgsz=item.imgsz,
                    npz_root=item.npz_root,
                )
            ),
        )
    )
    return runs


def normalize_pose_filters(values: Sequence[str] | None) -> Tuple[str, ...] | None:
    if values is None:
        return None
    allowed = tuple(sorted({pose for pose, _, _ in PIPELINE_SPECS}))
    return base.normalize_filter_tokens(
        values,
        allowed=allowed,
        kind="pose size",
        transform=str,
    )


def normalize_variant_filters(values: Sequence[str] | None) -> Tuple[str, ...] | None:
    if values is None:
        return None
    allowed = tuple(sorted({variant for _, variant, _ in PIPELINE_SPECS}))
    return base.normalize_filter_tokens(
        values,
        allowed=allowed,
        kind="pipeline variant",
        transform=str,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the selected optimised YOLO pose pipelines across final classifiers."
    )
    parser.add_argument("--only-models", nargs="+", default=None)
    parser.add_argument("--only-pose-sizes", nargs="+", default=None)
    parser.add_argument("--only-variants", nargs="+", default=None, help="Subset of fp32_576 fp16_576")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--force-regenerate-motionbert-pkl", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--motionbert-device", type=str, default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_MOTIONBERT_EVAL_BATCH_SIZE,
        help="MotionBERT eval batch size. Defaults to 32.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--keypoints-root", type=Path, default=DEFAULT_KEYPOINTS_ROOT)
    parser.add_argument("--classification-root", type=Path, default=DEFAULT_CLASSIFICATION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--motionbert-pkl-root", type=Path, default=None)
    return parser.parse_args()


def build_paths(args: argparse.Namespace) -> base.SweepPaths:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    output_root = args.output_root.expanduser()
    motionbert_pkl_root = (
        args.motionbert_pkl_root.expanduser()
        if args.motionbert_pkl_root is not None
        else output_root / "_motionbert_pkl_cache"
    )
    keypoints_root = args.keypoints_root.expanduser()
    return base.SweepPaths(
        script_path=script_path,
        repo_root=repo_root,
        eval_models_path=(repo_root / "evaluation" / "eval_models.py").resolve(),
        prepare_motionbert_script=(repo_root / "dataset_helpers" / "prepare_motionbert_dataset.py").resolve(),
        motionbert_code_root=(repo_root / "models" / "MotionBERT").resolve(),
        keypoints_root=keypoints_root,
        classification_root=args.classification_root.expanduser(),
        output_root=output_root,
        motionbert_pkl_root=motionbert_pkl_root,
    )


def build_config(args: argparse.Namespace) -> base.SweepConfig:
    return base.SweepConfig(
        python_executable=str(args.python),
        motionbert_device=str(args.motionbert_device),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        force_regenerate_motionbert_pkl=bool(args.force_regenerate_motionbert_pkl),
    )


def main() -> int:
    args = parse_args()
    paths = build_paths(args)
    config = build_config(args)

    variants, discovery_issues = discover_variants(paths)
    if not variants:
        print("No optimised pipeline variants were discovered.", file=sys.stderr)
        for issue in discovery_issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    all_runs = build_run_matrix(paths, variants)
    summary_json_path = paths.output_root / SUMMARY_JSON_NAME
    summary_entries = base.load_existing_summary_entries(all_runs)

    selected_models = base.normalize_filter_tokens(
        (base.normalize_model_token(token) for token in args.only_models) if args.only_models is not None else None,
        allowed=CLASSIFIER_MODELS,
        kind="model",
        transform=str,
    )
    selected_pose_sizes = normalize_pose_filters(args.only_pose_sizes)
    selected_variants = normalize_variant_filters(args.only_variants)

    selected_runs = [
        run
        for run in all_runs
        if (selected_models is None or run.classifier_model in selected_models)
        and (selected_pose_sizes is None or run.pose_model_size in selected_pose_sizes)
        and (selected_variants is None or run.variant_name in selected_variants)
    ]

    base.ensure_dir(paths.output_root)
    base.write_summary_json(summary_json_path, paths, all_runs, summary_entries)

    print(f"Output root: {paths.output_root.as_posix()}")
    print(f"Summary JSON: {summary_json_path.as_posix()}")
    print(f"Discovered variants: {len(variants)}")
    print(f"Planned runs: {len(selected_runs)}")
    if discovery_issues:
        print("Discovery notes:")
        for issue in discovery_issues:
            print(f"  - {issue}")

    if not selected_runs:
        print("No runs matched the selected filters.")
        return 0

    completed = 0
    failed = 0
    skipped_missing = 0
    reused = 0

    for index, run in enumerate(selected_runs, start=1):
        print(f"[{index}/{len(selected_runs)}] {run.run_id} -> keypoints={run.keypoints_tag}")

        missing_inputs = base.check_required_inputs(paths, run)
        if missing_inputs:
            print("  missing inputs:")
            for item in missing_inputs:
                print(f"    - {item}")
            entry = base.handle_missing_inputs(run, missing_inputs)
            summary_entries[run.run_id] = entry
            skipped_missing += 1
            base.write_summary_json(summary_json_path, paths, all_runs, summary_entries)
            if args.stop_on_error:
                return 1
            continue

        if not args.force_rerun:
            existing = base.detect_completed_run(run)
            if existing is not None:
                print("  reusing completed run")
                entry = base.process_existing_completed_run(run)
                summary_entries[run.run_id] = entry
                completed += 1
                base.write_summary_json(summary_json_path, paths, all_runs, summary_entries)
                continue

        if run.classifier_model == "MotionBERT":
            prepare_command = base.build_motionbert_prepare_command(config, paths, run)
            eval_command = base.build_motionbert_eval_command(config, run)
            if args.dry_run:
                base.print_dry_run_line(
                    run,
                    command=eval_command,
                    command_cwd=run.motionbert_repo_root or paths.motionbert_code_root,
                    prepare_command=prepare_command,
                )
                continue
        else:
            eval_command = base.build_eval_models_command(paths, config, run)
            if args.dry_run:
                base.print_dry_run_line(
                    run,
                    command=eval_command,
                    command_cwd=paths.repo_root,
                )
                continue

        entry = base.execute_run(paths, config, run)
        summary_entries[run.run_id] = entry
        base.write_summary_json(summary_json_path, paths, all_runs, summary_entries)

        status = str(entry.get("status"))
        if status == "completed":
            completed += 1
            metrics = entry.get("metrics", {})
            if isinstance(metrics, dict):
                print(
                    "  completed "
                    f"(acc={metrics.get('accuracy')}, rec={metrics.get('recall')}, "
                    f"macro_f1={metrics.get('macro_f1')}, fall_f1={metrics.get('fall_f1')})"
                )
            else:
                print("  completed")

            run_status = base.read_json(run.run_status_path) or {}
            if bool(run_status.get("reused_motionbert_pkl")):
                reused += 1
        else:
            failed += 1
            print(f"  failed: {entry.get('error_message')}")
            if args.stop_on_error:
                return 1

    print(
        f"Finished: completed={completed} failed={failed} "
        f"skipped_missing={skipped_missing} motionbert_pkl_reused={reused}"
    )
    return 1 if failed or skipped_missing else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
    except SystemExit:
        raise
    except Exception:
        base.traceback.print_exc()
        raise SystemExit(1)
