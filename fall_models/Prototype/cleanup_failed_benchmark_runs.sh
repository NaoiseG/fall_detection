#!/usr/bin/env bash

set -euo pipefail

BENCH_DIR="benchmarks"
DRY_RUN=0
PYTHON_BIN=""

usage() {
  cat <<'EOF'
Usage:
  ./cleanup_failed_benchmark_runs.sh [--bench-dir PATH] [--dry-run]

Deletes failed top-level benchmark run folders under the benchmarks directory.

A run is treated as failed if:
  1) summary.json is missing, or
  2) summary.json has total_frames_processed <= 0

Only top-level timestamped run folders are considered, e.g.:
  benchmarks/2026-03-11_14-01-21_026661__model_cnnlstm__kpts_yolo11s-pose.pt
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bench-dir)
      BENCH_DIR="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${BENCH_DIR}" || ! -d "${BENCH_DIR}" ]]; then
  echo "ERROR: Benchmark directory not found: ${BENCH_DIR}" >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "ERROR: python3/python not found in PATH." >&2
  exit 1
fi

is_failed_summary() {
  local summary_json="$1"

  "${PYTHON_BIN}" - "$summary_json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
try:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
except Exception:
    print("failed")
    raise SystemExit(0)

frames = data.get("total_frames_processed")
try:
    frames = float(frames)
except (TypeError, ValueError):
    frames = 0.0

print("failed" if frames <= 0 else "ok")
PY
}

deleted_count=0
kept_count=0
candidate_count=0

shopt -s nullglob

for run_dir in "${BENCH_DIR}"/*; do
  [[ -d "${run_dir}" ]] || continue

  run_name="$(basename "${run_dir}")"

  # Only inspect top-level timestamped benchmark output folders.
  if [[ ! "${run_name}" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]+__model_ ]]; then
    continue
  fi

  candidate_count=$((candidate_count + 1))

  summary_json="${run_dir}/summary.json"
  should_delete=0

  if [[ ! -f "${summary_json}" ]]; then
    should_delete=1
    reason="missing summary.json"
  else
    status="$(is_failed_summary "${summary_json}")"
    if [[ "${status}" == "failed" ]]; then
      should_delete=1
      reason="total_frames_processed <= 0"
    else
      reason="valid run (kept)"
    fi
  fi

  if [[ "${should_delete}" -eq 1 ]]; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "[DRY-RUN] delete ${run_dir} (${reason})"
    else
      rm -rf -- "${run_dir}"
      echo "[DELETED] ${run_dir} (${reason})"
    fi
    deleted_count=$((deleted_count + 1))
  else
    echo "[KEPT] ${run_dir} (${reason})"
    kept_count=$((kept_count + 1))
  fi
done

echo
echo "Cleanup complete."
echo "  Benchmark dir: ${BENCH_DIR}"
echo "  Candidates scanned: ${candidate_count}"
echo "  Deleted: ${deleted_count}"
echo "  Kept: ${kept_count}"
echo "  Mode: $([[ ${DRY_RUN} -eq 1 ]] && echo dry-run || echo apply)"
