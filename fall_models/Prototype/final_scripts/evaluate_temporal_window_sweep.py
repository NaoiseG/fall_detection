from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


MODELS = ("cnnlstm", "stgcn")
WINDOW_SIZES = (4, 8, 16, 32, 64, 128)
OVERLAPS_PCT = (25, 50, 75)
EVAL_VARIANTS = ("strict_label_mode", "two_label_transition_relaxed")
DEFAULT_LABEL_MODE = "center"
WINDOW_LABEL_MODE_OVERRIDES = {
    128: "majority",
}


@dataclass(frozen=True)
class SweepSetup:
    window_size: int
    overlap_pct: int
    stride: int
    label_mode: str
    name: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_out_root(repo_root: Path) -> Path:
    return (repo_root / ".." / ".." / "Conference" / "Results" / "window_evals").resolve()


def expand_user_path(path_text: str) -> str:
    return os.path.expanduser(path_text)


def validate_yolo11l_npz_root(npz_root: str) -> None:
    normalized = npz_root.replace("\\", "/").lower()
    if "yolo11l" not in normalized:
        raise SystemExit(
            "This script is restricted to yolo11l-pose keypoints. "
            "--npz-root must contain 'yolo11l'."
        )
    forbidden_tokens = ("yolo11n", "yolo11s", "yolo11m", "yolo11x", "alphapose", "vitpose")
    for token in forbidden_tokens:
        if token in normalized:
            raise SystemExit(
                "This script is restricted to yolo11l-pose keypoints only. "
                f"Found incompatible token '{token}' in --npz-root: {npz_root}"
            )


def label_mode_for_window_size(window_size: int) -> str:
    return str(WINDOW_LABEL_MODE_OVERRIDES.get(int(window_size), DEFAULT_LABEL_MODE))


def build_setups() -> List[SweepSetup]:
    setups: List[SweepSetup] = []
    for window_size in WINDOW_SIZES:
        for overlap_pct in OVERLAPS_PCT:
            stride_float = float(window_size) * (1.0 - (float(overlap_pct) / 100.0))
            stride_rounded = int(round(stride_float))
            if not math.isclose(stride_float, float(stride_rounded), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"Non-integer stride computed for T={window_size}, overlap={overlap_pct}%: {stride_float}"
                )
            if stride_rounded <= 0:
                raise ValueError(
                    f"Invalid stride computed for T={window_size}, overlap={overlap_pct}%: {stride_rounded}"
                )
            setups.append(
                SweepSetup(
                    window_size=window_size,
                    overlap_pct=overlap_pct,
                    stride=stride_rounded,
                    label_mode=label_mode_for_window_size(window_size),
                    name=f"T{window_size:02d}_overlap{overlap_pct:02d}_stride{stride_rounded:02d}",
                )
            )
    return setups


def command_string(cmd: Iterable[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def run_command(cmd: List[str], cwd: Path) -> str:
    print(f"\n[run] {command_string(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if proc.stdout is None:
        raise RuntimeError(f"Failed to capture stdout for command: {command_string(cmd)}")

    output_lines: List[str] = []
    for line in proc.stdout:
        print(line, end="")
        output_lines.append(line)

    proc.wait()
    output = "".join(output_lines)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=output)
    return output


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_flag_value_from_command(command: Any, flag: str) -> str | None:
    if not isinstance(command, list):
        return None
    for idx, token in enumerate(command):
        if str(token) != str(flag):
            continue
        if idx + 1 >= len(command):
            return None
        value = command[idx + 1]
        if value is None:
            return None
        text = str(value).strip()
        return text or None
    return None


def infer_epochs_from_train_command(train_command: Any) -> int | None:
    value = infer_flag_value_from_command(train_command, "--epochs")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def infer_label_mode_from_command(command: Any) -> str | None:
    return infer_flag_value_from_command(command, "--label-mode")


def normalize_setup_summary(summary: Dict[str, Any], fallback_epochs: int) -> Dict[str, Any]:
    normalized = dict(summary)
    if normalized.get("epochs") is None:
        inferred_epochs = infer_epochs_from_train_command(normalized.get("train_command"))
        normalized["epochs"] = int(inferred_epochs) if inferred_epochs is not None else int(fallback_epochs)
    else:
        normalized["epochs"] = int(normalized["epochs"])
    label_mode = normalized.get("label_mode")
    if label_mode is None:
        label_mode = infer_label_mode_from_command(normalized.get("train_command"))
    normalized["label_mode"] = str(label_mode or DEFAULT_LABEL_MODE)
    return normalized


def is_complete_setup_summary(
    summary: Any,
    expected_setup_name: str | None = None,
    expected_label_mode: str | None = None,
) -> bool:
    if not isinstance(summary, dict):
        return False
    if expected_setup_name is not None and str(summary.get("setup_name")) != expected_setup_name:
        return False
    if expected_label_mode is not None and str(summary.get("label_mode") or DEFAULT_LABEL_MODE) != str(expected_label_mode):
        return False

    training_results = summary.get("training_results")
    if not isinstance(training_results, list):
        return False
    training_models = {
        str(row.get("model"))
        for row in training_results
        if isinstance(row, dict) and row.get("model") is not None
    }
    if training_models != set(MODELS):
        return False

    weights = summary.get("weights")
    if not isinstance(weights, dict):
        return False
    if {str(k) for k in weights.keys()} != set(MODELS):
        return False

    evaluation_metrics = summary.get("evaluation_metrics")
    if not isinstance(evaluation_metrics, dict):
        return False
    if set(EVAL_VARIANTS) - set(str(k) for k in evaluation_metrics.keys()):
        return False
    for variant in EVAL_VARIANTS:
        rows = evaluation_metrics.get(variant)
        if not isinstance(rows, list) or len(rows) == 0:
            return False

    return True


def load_completed_setup_summaries(
    combined_results_path: Path,
    fallback_epochs: int,
    expected_setups_by_name: Dict[str, SweepSetup],
) -> Dict[str, Dict[str, Any]]:
    if not combined_results_path.exists():
        return {}

    payload = load_json(combined_results_path)
    payload_epochs = payload.get("epochs", fallback_epochs)
    raw_setups = payload.get("setups", [])
    if not isinstance(raw_setups, list):
        raise RuntimeError(
            f"Expected 'setups' to be a list in combined results JSON: {combined_results_path}"
        )

    completed: Dict[str, Dict[str, Any]] = {}
    for raw_summary in raw_setups:
        if not isinstance(raw_summary, dict):
            continue
        setup_name = str(raw_summary.get("setup_name", "")).strip()
        if not setup_name:
            continue
        if setup_name in completed:
            print(f"[resume] Ignoring duplicate setup entry in combined results: {setup_name}")
            continue

        summary = normalize_setup_summary(raw_summary, fallback_epochs=int(payload_epochs))
        expected_setup = expected_setups_by_name.get(setup_name)
        if expected_setup is None:
            print(f"[resume] Ignoring unknown setup entry in combined results: {setup_name}")
            continue
        if not is_complete_setup_summary(
            summary,
            expected_setup_name=setup_name,
            expected_label_mode=expected_setup.label_mode,
        ):
            print(f"[resume] Ignoring incomplete setup entry in combined results: {setup_name}")
            continue
        completed[setup_name] = summary

    print(f"[resume] Loaded {len(completed)} completed setup(s) from: {combined_results_path}")
    return completed


def remove_setup_dir(setup_dir: Path, out_root: Path) -> None:
    if not setup_dir.exists():
        return

    resolved_out_root = out_root.resolve()
    resolved_setup_dir = setup_dir.resolve()
    try:
        resolved_setup_dir.relative_to(resolved_out_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to delete setup directory outside output root: {resolved_setup_dir}"
        ) from exc

    if resolved_setup_dir == resolved_out_root:
        raise RuntimeError(f"Refusing to delete output root directly: {resolved_out_root}")

    shutil.rmtree(resolved_setup_dir)


def coerce_scalar(value: str) -> Any:
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"none", "nan", "null"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if all(ch not in text for ch in (".", "e", "E")):
            return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: coerce_scalar(val) for key, val in row.items()} for row in reader]


def resolve_path_from_repo(repo_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def flatten_summary_rows(setup_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat_rows: List[Dict[str, Any]] = []
    for setup_summary in setup_summaries:
        for variant, rows in setup_summary.get("evaluation_metrics", {}).items():
            for row in rows:
                merged = {
                    "setup_name": setup_summary["setup_name"],
                    "setup_dir": setup_summary["setup_dir"],
                    "window_size": setup_summary["window_size"],
                    "overlap_pct": setup_summary["overlap_pct"],
                    "stride": setup_summary["stride"],
                    "label_mode": setup_summary["label_mode"],
                    "variant": variant,
                }
                merged.update(row)
                flat_rows.append(merged)
    return flat_rows


def pick_best_rows(flat_rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    best_by_group: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for row in flat_rows:
        metric_value = row.get(metric)
        if metric_value is None:
            continue
        key = (str(row.get("variant")), str(row.get("model")), str(row.get("eval_split")))
        current = best_by_group.get(key)
        if current is None or float(metric_value) > float(current.get(metric, float("-inf"))):
            best_by_group[key] = row

    return [best_by_group[key] for key in sorted(best_by_group)]


def build_combined_payload(
    *,
    repo_root: Path,
    out_root: Path,
    npz_root: str,
    epochs: int,
    setup_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    flat_rows = flatten_summary_rows(setup_summaries)
    setup_epochs = sorted({int(summary.get("epochs", epochs)) for summary in setup_summaries})
    return {
        "generated_at": utc_now_iso(),
        "project_root": str(repo_root.resolve()),
        "out_root": str(out_root.resolve()),
        "npz_root": npz_root,
        "models": list(MODELS),
        "window_sizes": list(WINDOW_SIZES),
        "overlaps_pct": list(OVERLAPS_PCT),
        "epochs": int(epochs),
        "setup_epochs": setup_epochs,
        "fixed_config": {
            "train_subjects": "1-12",
            "val_subjects": "13-15",
            "test_subjects": "16-17",
            "camera": [1, 2],
            "label_mode_default": DEFAULT_LABEL_MODE,
            "label_mode_overrides": {
                str(window_size): str(label_mode)
                for window_size, label_mode in sorted(WINDOW_LABEL_MODE_OVERRIDES.items())
            },
            "drop_ambig_share": 0,
            "normalize": 1,
            "normalize_mode": "paper_rp",
            "rp_center_mode": "pixel",
            "rp_img_w": 640,
            "rp_img_h": 480,
            "missing_mode": "zeros_only",
            "interp_mode": "paper_group_linear",
            "interp_group": 100,
            "selection_metric": "composite_fall_fbeta_macro_f1",
            "selection_w": 0.7,
            "selection_beta": 2.0,
            "rare_class_boost": 1.5,
            "weighted_sampler": 1,
            "conf_thres": 0.05,
        },
        "completed_setup_count": len(setup_summaries),
        "setups": setup_summaries,
        "all_summary_rows": flat_rows,
        "best_macro_f1": pick_best_rows(flat_rows, "macro_f1"),
        "best_binary_f1_fall": pick_best_rows(flat_rows, "binary_f1_fall"),
    }


def build_train_command(
    *,
    npz_root: str,
    epochs: int,
    setup: SweepSetup,
    train_results_csv: Path,
) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "training.train_models",
        "--model",
        *MODELS,
        "--train-subjects",
        "1-12",
        "--val-subjects",
        "13-15",
        "--npz-root",
        npz_root,
        "--camera",
        "1",
        "2",
        "--label-mode",
        setup.label_mode,
        "--drop-ambig-share",
        "0",
        "--T",
        str(setup.window_size),
        "--stride",
        str(setup.stride),
        "--epochs",
        str(int(epochs)),
        "--normalize",
        "1",
        "--normalize-mode",
        "paper_rp",
        "--rp-center-mode",
        "pixel",
        "--rp-img-w",
        "640",
        "--rp-img-h",
        "480",
        "--missing-mode",
        "zeros_only",
        "--interp-mode",
        "paper_group_linear",
        "--interp-group",
        "100",
        "--selection-metric",
        "composite_fall_fbeta_macro_f1",
        "--selection-w",
        "0.7",
        "--selection-beta",
        "2.0",
        "--rare-class-boost",
        "1.5",
        "--weighted-sampler",
        "1",
        "--conf-thres",
        "0.05",
        "--save-results",
        str(train_results_csv),
    ]


def build_eval_command(
    *,
    npz_root: str,
    setup: SweepSetup,
    eval_out_dir: Path,
    weights_by_model: Dict[str, str],
) -> List[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "evaluation.eval_models",
        "--models",
        *MODELS,
        "--test-subjects",
        "16-17",
        "--npz-root",
        npz_root,
        "--camera",
        "1",
        "2",
        "--label-mode",
        setup.label_mode,
        "--drop-ambig-share",
        "0",
        "--T",
        str(setup.window_size),
        "--stride",
        str(setup.stride),
        "--normalize",
        "1",
        "--normalize-mode",
        "paper_rp",
        "--rp-center-mode",
        "pixel",
        "--rp-img-w",
        "640",
        "--rp-img-h",
        "480",
        "--missing-mode",
        "zeros_only",
        "--interp-mode",
        "paper_group_linear",
        "--interp-group",
        "100",
        "--conf-thres",
        "0.05",
        "--weights-path",
        f"cnnlstm={weights_by_model['cnnlstm']}",
        f"stgcn={weights_by_model['stgcn']}",
        "--out-dir",
        str(eval_out_dir),
    ]


def collect_training_summary(repo_root: Path, train_results_csv: Path) -> tuple[List[Dict[str, Any]], Dict[str, str], str]:
    if not train_results_csv.exists():
        raise FileNotFoundError(f"Training results CSV was not created: {train_results_csv}")

    training_rows = read_csv_rows(train_results_csv)
    found_models = {str(row.get("model")) for row in training_rows}
    expected_models = set(MODELS)
    if found_models != expected_models:
        raise RuntimeError(
            f"Training results CSV did not contain the expected models. Found={sorted(found_models)} "
            f"Expected={sorted(expected_models)}"
        )

    weights_by_model: Dict[str, str] = {}
    run_ids: set[str] = set()
    for row in training_rows:
        model_name = str(row["model"])
        ckpt_path = resolve_path_from_repo(repo_root, str(row["ckpt_path"]))
        row["ckpt_path"] = str(ckpt_path)
        weights_by_model[model_name] = str(ckpt_path)
        run_ids.add(ckpt_path.parent.name)

    if len(run_ids) != 1:
        raise RuntimeError(f"Expected one shared training run id, found: {sorted(run_ids)}")

    return training_rows, weights_by_model, next(iter(run_ids))


def finalize_evaluation_dir(eval_temp_root: Path, final_eval_dir: Path) -> Path:
    child_dirs = sorted([path for path in eval_temp_root.iterdir() if path.is_dir()])
    if len(child_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one evaluation run directory under {eval_temp_root}, found {len(child_dirs)}"
        )
    produced_dir = child_dirs[0]
    shutil.move(str(produced_dir), str(final_eval_dir))
    eval_temp_root.rmdir()
    return final_eval_dir


def collect_eval_metrics(final_eval_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    evaluation_metrics: Dict[str, List[Dict[str, Any]]] = {}
    for variant in EVAL_VARIANTS:
        summary_csv = final_eval_dir / variant / "metrics_summary.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(f"Missing evaluation summary CSV: {summary_csv}")
        evaluation_metrics[variant] = read_csv_rows(summary_csv)
    return evaluation_metrics


def run_setup(
    *,
    repo_root: Path,
    out_root: Path,
    npz_root: str,
    epochs: int,
    setup: SweepSetup,
    force_restart: bool = False,
) -> Dict[str, Any]:
    setup_dir = out_root / setup.name
    summary_path = setup_dir / "setup_summary.json"

    if force_restart and setup_dir.exists():
        print(f"[overwrite] Removing existing setup directory before rerun: {setup_dir}")
        remove_setup_dir(setup_dir, out_root)

    if summary_path.exists():
        print(f"[resume] Reusing existing setup summary: {summary_path}")
        summary = normalize_setup_summary(load_json(summary_path), fallback_epochs=int(epochs))
        if not is_complete_setup_summary(
            summary,
            expected_setup_name=setup.name,
            expected_label_mode=setup.label_mode,
        ):
            print(f"[resume] Existing setup summary is incomplete; restarting: {summary_path}")
            remove_setup_dir(setup_dir, out_root)
        else:
            return summary

    if setup_dir.exists():
        print(f"[resume] Partial setup detected; deleting and restarting: {setup_dir}")
        remove_setup_dir(setup_dir, out_root)

    training_dir = setup_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=False)

    train_results_csv = training_dir / "training_results.csv"
    train_command = build_train_command(
        npz_root=npz_root,
        epochs=epochs,
        setup=setup,
        train_results_csv=train_results_csv,
    )
    write_text(training_dir / "train_command.txt", command_string(train_command) + "\n")
    train_output = run_command(train_command, cwd=repo_root)
    write_text(training_dir / "train.log", train_output)

    training_rows, weights_by_model, run_id = collect_training_summary(repo_root, train_results_csv)

    eval_temp_root = setup_dir / "_eval_outputs"
    eval_temp_root.mkdir(parents=True, exist_ok=False)
    eval_command = build_eval_command(
        npz_root=npz_root,
        setup=setup,
        eval_out_dir=eval_temp_root,
        weights_by_model=weights_by_model,
    )
    write_text(setup_dir / "eval_command.txt", command_string(eval_command) + "\n")
    eval_output = run_command(eval_command, cwd=repo_root)
    write_text(setup_dir / "eval.log", eval_output)

    final_eval_dir = finalize_evaluation_dir(eval_temp_root, setup_dir / "evaluation")
    evaluation_metrics = collect_eval_metrics(final_eval_dir)

    setup_summary = {
        "generated_at": utc_now_iso(),
        "setup_name": setup.name,
        "setup_dir": str(setup_dir.resolve()),
        "epochs": int(epochs),
        "window_size": int(setup.window_size),
        "overlap_pct": int(setup.overlap_pct),
        "overlap_fraction": float(setup.overlap_pct) / 100.0,
        "stride": int(setup.stride),
        "label_mode": str(setup.label_mode),
        "train_run_id": run_id,
        "training_results_csv": str(train_results_csv.resolve()),
        "evaluation_dir": str(final_eval_dir.resolve()),
        "weights": weights_by_model,
        "train_command": train_command,
        "eval_command": eval_command,
        "training_results": training_rows,
        "evaluation_metrics": evaluation_metrics,
    }
    write_json(summary_path, setup_summary)
    return setup_summary


def parse_window_sizes_arg(raw_window_sizes: List[int] | None) -> set[int] | None:
    if raw_window_sizes is None:
        return None

    selected = {int(window_size) for window_size in raw_window_sizes}
    invalid = sorted(selected - set(WINDOW_SIZES))
    if invalid:
        raise SystemExit(
            f"Unsupported --window-sizes value(s): {invalid}. Supported values: {list(WINDOW_SIZES)}"
        )
    return selected


def filter_setups_by_window_size(
    setups: List[SweepSetup],
    selected_window_sizes: set[int] | None,
) -> List[SweepSetup]:
    if selected_window_sizes is None:
        return list(setups)
    return [setup for setup in setups if int(setup.window_size) in selected_window_sizes]


def ordered_setup_summaries(
    ordered_setups: List[SweepSetup],
    summaries_by_name: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [summaries_by_name[setup.name] for setup in ordered_setups if setup.name in summaries_by_name]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate cnnlstm and stgcn over temporal window sizes "
            "T={4,8,16,32,64,128} and overlaps {25,50,75}% for yolo11l keypoints. "
            "T=128 uses majority window labels; smaller windows use center labels."
        )
    )
    parser.add_argument(
        "--npz-root",
        type=str,
        default="~/scratch/keypoints/UPFall_keypoints/yolo11l/base",
        help="Root directory containing yolo11l UP-Fall keypoint NPZs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(default_out_root(repo_root)),
        help="Output root for per-setup folders and the combined JSON summary.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Epochs per model for each setup (default: 100).",
    )
    parser.add_argument(
        "--window-sizes",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional subset of window sizes to run, e.g. --window-sizes 128 or "
            "--window-sizes 32 64 128."
        ),
    )
    parser.add_argument(
        "--overwrite-selected",
        action="store_true",
        help=(
            "Delete and rerun the selected setups even if completed results already exist. "
            "Use with --window-sizes 128 to refresh only the T=128 runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    npz_root = expand_user_path(str(args.npz_root))
    validate_yolo11l_npz_root(npz_root)
    all_setups = build_setups()
    setup_by_name = {setup.name: setup for setup in all_setups}
    selected_window_sizes = parse_window_sizes_arg(args.window_sizes)
    selected_setups = filter_setups_by_window_size(all_setups, selected_window_sizes)

    if not selected_setups:
        raise SystemExit("No setups matched the requested --window-sizes filter.")

    out_root = Path(expand_user_path(str(args.out))).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    combined_results_path = out_root / "combined_results.json"
    completed_by_name = load_completed_setup_summaries(
        combined_results_path,
        fallback_epochs=int(args.epochs),
        expected_setups_by_name=setup_by_name,
    )
    setup_summaries_by_name: Dict[str, Dict[str, Any]] = dict(completed_by_name)

    if args.overwrite_selected:
        for setup in selected_setups:
            if setup.name in setup_summaries_by_name:
                print(f"[overwrite] Dropping cached combined-results entry: {setup.name}")
                setup_summaries_by_name.pop(setup.name, None)

    for setup in selected_setups:
        print(
            f"\n[setup] {setup.name} | T={setup.window_size} | overlap={setup.overlap_pct}% "
            f"| stride={setup.stride} | label_mode={setup.label_mode}"
        )
        if setup.name in completed_by_name and not args.overwrite_selected:
            print(f"[resume] Reusing completed setup from combined results: {setup.name}")
            summary = completed_by_name[setup.name]
        else:
            summary = run_setup(
                repo_root=repo_root,
                out_root=out_root,
                npz_root=npz_root,
                epochs=int(args.epochs),
                setup=setup,
                force_restart=bool(args.overwrite_selected),
            )
        setup_summaries_by_name[setup.name] = summary
        ordered_summaries = ordered_setup_summaries(all_setups, setup_summaries_by_name)

        combined_payload = build_combined_payload(
            repo_root=repo_root,
            out_root=out_root,
            npz_root=npz_root,
            epochs=int(args.epochs),
            setup_summaries=ordered_summaries,
        )
        write_json(combined_results_path, combined_payload)
        print(f"[write] Updated combined results: {combined_results_path}")

    print(f"\n[done] Wrote combined results to: {combined_results_path}")


if __name__ == "__main__":
    main()
