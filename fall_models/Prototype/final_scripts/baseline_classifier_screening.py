#!/usr/bin/env python3
"""
Resumable baseline classifier screening runner.

Trains each backbone once over the full model list, moves checkpoints to
~/scratch/evaluations/classifier_screening/weights/<backbone>/<model>/<run_id>/,
evaluates with explicit --weights-path overrides, then extracts strict-label
metrics from strict_label_mode/metrics_summary.csv.

Run from fall_models/Prototype/ (project root for module imports):
    python final_scripts/baseline_classifier_screening.py [options]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────── constants ───────────────────────────

MODELS_ALL: List[str] = [
    "cnnlstm", "tcn", "lstm", "paper_lstm", "gru", "stgcn", "paper_stgcn", "rf",
]

BACKBONES: Dict[str, Path] = {
    "yolo11x":   Path("~/scratch/keypoints/UPFall_keypoints/yolo11x/base").expanduser(),
    "alphapose": Path("~/scratch/keypoints/UPFall_keypoints_alpha/base").expanduser(),
    "vitpose":   Path("~/scratch/keypoints/UPFall_keypoints_vitpose/base").expanduser(),
}

OUTPUT_ROOT = Path("~/scratch/evaluations/classifier_screening").expanduser()
STATE_FILE  = OUTPUT_ROOT / "run_state.json"
LOG_FILE    = OUTPUT_ROOT / "run_log.txt"
JSON_OUT    = OUTPUT_ROOT / "baseline_classidier_screening.json"
CSV_OUT     = OUTPUT_ROOT / "baseline_classidier_screening.csv"

# ─────────────────────────── logging ───────────────────────────

def log_event(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ─────────────────────────── state ───────────────────────────

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            log_event(f"WARNING: could not load state file ({e}), starting fresh.")
    return {}

def save_state_atomic(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)

def get_backbone_state(state: Dict[str, Any], backbone: str) -> Dict[str, Any]:
    if backbone not in state:
        state[backbone] = {
            "train_done": False,
            "run_id": None,
            "moves_done": {},
            "eval_done": False,
            "eval_run_dir": None,
            "metrics_done": {},
        }
    return state[backbone]

# ─────────────────────────── filesystem helpers ───────────────────────────

def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "weights").mkdir(exist_ok=True)
    (OUTPUT_ROOT / "eval").mkdir(exist_ok=True)

def verify_npz_root(path: Path) -> bool:
    if not path.exists():
        return False
    return any(path.rglob("*.npz"))

def find_checkpoint_file(run_dir: Path, model: str) -> Optional[Path]:
    """Find <model>_best.{pt,pkl,pth,...} in run_dir without hardcoding extension."""
    for ext in (".pt", ".pkl", ".pth"):
        p = run_dir / f"{model}_best{ext}"
        if p.is_file() and p.stat().st_size > 0:
            return p
    # Fallback: any file matching pattern
    for p in sorted(run_dir.glob(f"{model}_best.*")):
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None

def is_valid_run_dir(run_dir: Path, model: str) -> bool:
    return find_checkpoint_file(run_dir, model) is not None

def find_moved_run_dir(backbone: str, model: str) -> Optional[Path]:
    """Return the first valid moved run dir for backbone/model, or None."""
    base = OUTPUT_ROOT / "weights" / backbone / model
    if not base.exists():
        return None
    for d in sorted(base.iterdir()):
        if d.is_dir() and is_valid_run_dir(d, model):
            return d
    return None

def move_run_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        log_event(f"    dst already exists, skipping: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    log_event(f"    moved: {src.name} → {dst}")

def find_eval_out_dir(backbone_eval_dir: Path, after_mtime: float) -> Optional[Path]:
    """Return most recent timestamped dir with strict_label_mode/metrics_summary.csv
    created at or after after_mtime."""
    if not backbone_eval_dir.exists():
        return None
    candidates = []
    for d in backbone_eval_dir.iterdir():
        if not d.is_dir():
            continue
        csv_p = d / "strict_label_mode" / "metrics_summary.csv"
        if not csv_p.exists():
            continue
        try:
            mtime = d.stat().st_mtime
        except OSError:
            continue
        if mtime >= after_mtime:
            candidates.append((mtime, d))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]

def find_any_eval_out_dir(backbone_eval_dir: Path) -> Optional[Path]:
    """Return the most recent eval output dir with a valid metrics_summary.csv."""
    return find_eval_out_dir(backbone_eval_dir, after_mtime=0.0)

# ─────────────────────────── subprocess ───────────────────────────

def run_subprocess(cmd: List[str], cwd: Path, label: str) -> Tuple[int, str]:
    """Run cmd, stream stdout to console and return (returncode, full_stdout)."""
    import subprocess
    log_event(f"[{label}] CMD: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    dt = time.time() - t0
    log_event(f"[{label}] done in {dt:.1f}s  rc={proc.returncode}")
    return proc.returncode, "".join(lines)

# ─────────────────────────── run_id detection ───────────────────────────

def detect_run_id_from_stdout(stdout: str) -> Optional[str]:
    """Parse 'Run ID: <value>' printed by train_models.py."""
    m = re.search(r"Run ID:\s*(\S+)", stdout)
    return m.group(1).strip() if m else None

def detect_run_dirs_by_mtime(model_dir: Path, after_mtime: float) -> List[Path]:
    """List subdirs of model_dir with mtime >= after_mtime."""
    if not model_dir.exists():
        return []
    result = []
    for d in model_dir.iterdir():
        if d.is_dir():
            try:
                if d.stat().st_mtime >= after_mtime:
                    result.append(d)
            except OSError:
                pass
    return result

# ─────────────────────────── command builders ───────────────────────────

def build_train_cmd(models: List[str], npz_root: Path) -> List[str]:
    return [
        sys.executable, "-m", "training.train_models",
        "--model", *models,
        "--train-subjects", "1-12",
        "--val-subjects", "13-15",
        "--npz-root", str(npz_root),
        "--camera", "1", "2",
        "--label-mode", "center",
        "--drop-ambig-share", "0",
        "--T", "64",
        "--stride", "48",
        "--epochs", "100",
        "--normalize", "1",
        "--normalize-mode", "paper_rp",
        "--rp-center-mode", "pixel",
        "--rp-img-w", "640",
        "--rp-img-h", "480",
        "--missing-mode", "zeros_only",
        "--interp-mode", "paper_group_linear",
        "--interp-group", "100",
        "--selection-metric", "composite_fall_fbeta_macro_f1",
        "--selection-w", "0.7",
        "--selection-beta", "2.0",
        "--rare-class-boost", "1.5",
        "--weighted-sampler", "1",
        "--conf-thres", "0.05",
    ]

def build_eval_cmd(
    models: List[str],
    npz_root: Path,
    out_dir: Path,
    weights: Dict[str, Path],
) -> List[str]:
    cmd = [
        sys.executable, "-m", "evaluation.eval_models",
        "--models", *models,
        "--test-subjects", "16-17",
        "--npz-root", str(npz_root),
        "--label-mode", "center",
        "--drop-ambig-share", "0",
        "--T", "64",
        "--stride", "48",
        "--normalize", "1",
        "--normalize-mode", "paper_rp",
        "--rp-center-mode", "pixel",
        "--rp-img-w", "640",
        "--rp-img-h", "480",
        "--missing-mode", "zeros_only",
        "--interp-mode", "paper_group_linear",
        "--interp-group", "100",
        "--out-dir", str(out_dir),
        "--weights-path",
    ]
    for model, ckpt_path in weights.items():
        cmd.append(f"{model}={ckpt_path.as_posix()}")
    return cmd

# ─────────────────────────── metrics extraction ───────────────────────────

def extract_strict_metrics(
    eval_run_dir: Path,
    models: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Parse strict_label_mode/metrics_summary.csv.
    Uses 'combined' eval_split row.  Accuracy = sum(recall_i * support_i) / n_samples.
    """
    csv_path = eval_run_dir / "strict_label_mode" / "metrics_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics_summary.csv not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results: Dict[str, Dict[str, float]] = {}
    for model in models:
        model_rows = [r for r in rows if r.get("model", "").lower().strip() == model]
        if not model_rows:
            log_event(f"    [metrics] no row for model={model} in {csv_path.name}")
            continue

        # Prefer 'combined' split; fall back to the row with most samples
        combined = [r for r in model_rows if r.get("eval_split", "").lower() == "combined"]
        if combined:
            row = combined[0]
        else:
            # Sort by n_samples desc, take best
            def _ns(r: Dict[str, str]) -> int:
                try:
                    return int(r.get("n_samples", 0))
                except ValueError:
                    return 0
            row = max(model_rows, key=_ns)
            log_event(f"    [metrics] no 'combined' split for {model}, using eval_split={row.get('eval_split','?')}")

        # macro_f1
        try:
            macro_f1 = float(row["macro_f1"])
        except (KeyError, ValueError):
            log_event(f"    [metrics] macro_f1 missing for {model}")
            macro_f1 = float("nan")

        # fall_f1  (column: binary_f1_fall)
        try:
            fall_f1 = float(row["binary_f1_fall"])
        except (KeyError, ValueError):
            log_event(f"    [metrics] binary_f1_fall missing for {model}")
            fall_f1 = float("nan")

        # accuracy from per-class recall * support / n_samples
        accuracy = float("nan")
        try:
            n_samples = int(row["n_samples"])
            if n_samples > 0:
                recall_suffixes = {
                    k[len("recall_"):]: float(v)
                    for k, v in row.items()
                    if k.startswith("recall_")
                }
                support_lookup = {
                    k[len("support_"):]: float(v)
                    for k, v in row.items()
                    if k.startswith("support_")
                }
                tp_sum = sum(
                    recall_suffixes[s] * support_lookup[s]
                    for s in recall_suffixes
                    if s in support_lookup
                )
                if tp_sum > 0:
                    accuracy = tp_sum / n_samples
        except Exception as e:
            log_event(f"    [metrics] accuracy computation failed for {model}: {e}")

        results[model] = {"accuracy": accuracy, "fall_f1": fall_f1, "macro_f1": macro_f1}
    return results

# ─────────────────────────── summary output ───────────────────────────

def write_summary_outputs(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    JSON_OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log_event(f"JSON: {JSON_OUT}")

    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backbone", "model", "accuracy", "fall_f1", "macro_f1"])
        for bb in sorted(results):
            for model in sorted(results[bb]):
                m = results[bb][model]
                w.writerow([bb, model, m.get("accuracy"), m.get("fall_f1"), m.get("macro_f1")])
    log_event(f"CSV: {CSV_OUT}")

# ─────────────────────────── per-backbone pipeline ───────────────────────────

def process_backbone(
    backbone: str,
    npz_root: Path,
    models: List[str],
    state: Dict[str, Any],
    project_root: Path,
    force_train: bool,
    force_eval: bool,
) -> Dict[str, Dict[str, float]]:

    bs = get_backbone_state(state, backbone)

    if not verify_npz_root(npz_root):
        raise RuntimeError(f"NPZ root not found or empty: {npz_root}")

    # ── 1. Check which checkpoints are already moved ──────────────
    existing = {m: find_moved_run_dir(backbone, m) for m in models}
    all_moved = all(v is not None for v in existing.values())

    # ── 2. Training ───────────────────────────────────────────────
    if force_train or not all_moved:
        if force_train:
            log_event(f"[{backbone}] force-train requested.")
        else:
            missing = [m for m, v in existing.items() if v is None]
            log_event(f"[{backbone}] Missing moved checkpoints: {missing}")

        if bs.get("train_done") and bs.get("run_id") and not force_train:
            log_event(f"[{backbone}] Training already recorded (run_id={bs['run_id']}), skipping re-train.")
            run_id: Optional[str] = bs["run_id"]
            t_train = 0.0
        else:
            log_event(f"[{backbone}] Training: {models}")
            t_train = time.time()
            rc, stdout = run_subprocess(
                build_train_cmd(models, npz_root), project_root, f"{backbone}/train"
            )
            if rc != 0:
                raise RuntimeError(f"Training subprocess failed (rc={rc})")

            run_id = detect_run_id_from_stdout(stdout)
            if run_id:
                log_event(f"[{backbone}] run_id from stdout: {run_id}")
            else:
                log_event(f"[{backbone}] run_id not found in stdout; will scan directories.")
            bs["run_id"] = run_id
            bs["train_done"] = True
            save_state_atomic(state)

        # Refresh existing after training
        existing = {m: find_moved_run_dir(backbone, m) for m in models}

    # ── 3. Move checkpoints ───────────────────────────────────────
    run_id = bs.get("run_id")

    for model in models:
        if existing.get(model) is not None and not force_train:
            log_event(f"[{backbone}/{model}] already moved → {existing[model]}")
            bs.setdefault("moves_done", {})[model] = True
            continue

        # Locate source run dir
        src: Optional[Path] = None
        if run_id:
            candidate = project_root / "models" / model / run_id
            if candidate.is_dir() and is_valid_run_dir(candidate, model):
                src = candidate

        if src is None:
            # Scan for the most recent valid run dir (e.g. run_id wasn't captured)
            model_dir = project_root / "models" / model
            candidates = []
            if model_dir.is_dir():
                for d in model_dir.iterdir():
                    if d.is_dir() and is_valid_run_dir(d, model):
                        try:
                            candidates.append((d.stat().st_mtime, d))
                        except OSError:
                            pass
            if candidates:
                src = max(candidates, key=lambda x: x[0])[1]
                log_event(f"[{backbone}/{model}] located run dir by scan: {src.name}")
                if not run_id:
                    run_id = src.name
                    bs["run_id"] = run_id

        if src is None:
            raise RuntimeError(
                f"[{backbone}/{model}] Cannot find source run dir "
                f"(run_id={run_id}, models/dir={project_root/'models'/model})"
            )

        dst = OUTPUT_ROOT / "weights" / backbone / model / src.name
        if not dst.exists():
            move_run_dir(src, dst)
        else:
            log_event(f"[{backbone}/{model}] dst exists: {dst}")

        if not is_valid_run_dir(dst, model):
            raise RuntimeError(f"[{backbone}/{model}] checkpoint invalid after move: {dst}")

        existing[model] = dst
        bs.setdefault("moves_done", {})[model] = True
        save_state_atomic(state)
        log_event(f"[{backbone}/{model}] move OK")

    # ── 4. Evaluation ─────────────────────────────────────────────
    backbone_eval_dir = OUTPUT_ROOT / "eval" / backbone
    backbone_eval_dir.mkdir(parents=True, exist_ok=True)

    # Resolve eval_run_dir from state or filesystem
    eval_run_dir: Optional[Path] = None
    stored_erd = bs.get("eval_run_dir")
    if stored_erd:
        candidate_erd = backbone_eval_dir / stored_erd
        if (candidate_erd / "strict_label_mode" / "metrics_summary.csv").exists():
            eval_run_dir = candidate_erd

    if eval_run_dir is None and not force_eval:
        eval_run_dir = find_any_eval_out_dir(backbone_eval_dir)
        if eval_run_dir:
            log_event(f"[{backbone}] found existing eval output: {eval_run_dir.name}")
            bs["eval_run_dir"] = eval_run_dir.name
            bs["eval_done"] = True
            save_state_atomic(state)

    eval_needed = (eval_run_dir is None) or force_eval
    if eval_needed:
        # Build per-model weights dict
        weights: Dict[str, Path] = {}
        eval_models: List[str] = []
        for model in models:
            run_dir = existing.get(model)
            if run_dir is None:
                log_event(f"[{backbone}/{model}] WARNING: no moved checkpoint, skipping eval")
                continue
            ckpt = find_checkpoint_file(run_dir, model)
            if ckpt is None:
                log_event(f"[{backbone}/{model}] WARNING: no checkpoint file in {run_dir}")
                continue
            weights[model] = ckpt
            eval_models.append(model)

        if not eval_models:
            raise RuntimeError(f"[{backbone}] No models available for eval")

        log_event(f"[{backbone}] Evaluating: {eval_models}")
        t_before = time.time()
        rc, _stdout = run_subprocess(
            build_eval_cmd(eval_models, npz_root, backbone_eval_dir, weights),
            project_root,
            f"{backbone}/eval",
        )
        if rc != 0:
            raise RuntimeError(f"Eval subprocess failed (rc={rc})")

        eval_run_dir = find_eval_out_dir(backbone_eval_dir, after_mtime=t_before - 10)
        if eval_run_dir is None:
            raise RuntimeError(
                f"[{backbone}] Eval succeeded but could not find output dir in {backbone_eval_dir}"
            )

        bs["eval_done"] = True
        bs["eval_run_dir"] = eval_run_dir.name
        save_state_atomic(state)
        log_event(f"[{backbone}] eval output: {eval_run_dir.name}")

    assert eval_run_dir is not None

    # ── 5. Extract metrics ────────────────────────────────────────
    eval_models_present = [m for m in models if existing.get(m) is not None]
    metrics = extract_strict_metrics(eval_run_dir, eval_models_present)

    if not metrics:
        raise RuntimeError(f"[{backbone}] No metrics extracted from {eval_run_dir}")

    for model, vals in metrics.items():
        bs.setdefault("metrics_done", {})[model] = True
        log_event(
            f"[{backbone}/{model}]  acc={vals['accuracy']:.4f}"
            f"  fall_f1={vals['fall_f1']:.4f}"
            f"  macro_f1={vals['macro_f1']:.4f}"
        )
    save_state_atomic(state)
    return metrics

# ─────────────────────────── CLI entry point ───────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable baseline classifier screening.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from state file (default: on).")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore state file, check filesystem only.")
    parser.add_argument("--force-train",   action="store_true", help="Retrain even if checkpoints exist.")
    parser.add_argument("--force-eval",    action="store_true", help="Re-evaluate even if eval output exists.")
    parser.add_argument("--force-summary", action="store_true", help="Rewrite summary files.")
    parser.add_argument("--force-all",     action="store_true", help="Implies --force-train --force-eval --force-summary.")
    parser.add_argument("--backbones", nargs="+", default=list(BACKBONES.keys()),
                        help="Backbones to process (default: all).")
    parser.add_argument("--models", nargs="+", default=MODELS_ALL,
                        help="Models to run (default: all).")
    args = parser.parse_args()

    if args.force_all:
        args.force_train = args.force_eval = args.force_summary = True

    backbone_names = [b for b in args.backbones if b in BACKBONES]
    unknown_bb = [b for b in args.backbones if b not in BACKBONES]
    if unknown_bb:
        print(f"WARNING: unknown backbones ignored: {unknown_bb}", file=sys.stderr)

    model_names = [m for m in args.models if m in MODELS_ALL]
    unknown_m = [m for m in args.models if m not in MODELS_ALL]
    if unknown_m:
        print(f"WARNING: unknown models ignored: {unknown_m}", file=sys.stderr)

    # Project root: fall_models/Prototype/ (parent of final_scripts/)
    project_root = Path(__file__).resolve().parent.parent

    ensure_dirs()
    log_event("=" * 70)
    log_event("Baseline classifier screening started")
    log_event(f"  project root : {project_root}")
    log_event(f"  output root  : {OUTPUT_ROOT}")
    log_event(f"  backbones    : {backbone_names}")
    log_event(f"  models       : {model_names}")
    log_event(f"  resume       : {args.resume}")
    log_event("=" * 70)

    state = load_state() if args.resume else {}

    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    failures: List[str] = []

    for backbone in backbone_names:
        log_event(f"\n{'─'*60}")
        log_event(f"BACKBONE: {backbone}")
        log_event(f"{'─'*60}")
        try:
            metrics = process_backbone(
                backbone=backbone,
                npz_root=BACKBONES[backbone],
                models=model_names,
                state=state,
                project_root=project_root,
                force_train=args.force_train,
                force_eval=args.force_eval,
            )
            all_results[backbone] = metrics
        except Exception as e:
            log_event(f"[{backbone}] FAILED: {e}")
            traceback.print_exc()
            failures.append(f"{backbone}: {e}")

    # Write summary (always if we have any results, or if forced)
    if all_results or args.force_summary:
        try:
            write_summary_outputs(all_results)
        except Exception as e:
            log_event(f"Summary write failed: {e}")
            failures.append(f"summary_write: {e}")

    log_event("\n" + "=" * 70)
    if failures:
        log_event(f"COMPLETED WITH {len(failures)} FAILURE(S):")
        for f in failures:
            log_event(f"  ✗ {f}")
        sys.exit(1)
    else:
        log_event("ALL BACKBONES SUCCEEDED")
        log_event(f"  JSON → {JSON_OUT}")
        log_event(f"  CSV  → {CSV_OUT}")
    log_event("=" * 70)


if __name__ == "__main__":
    main()
