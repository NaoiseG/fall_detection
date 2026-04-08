#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage:
  ./fix_bad_keypoints.sh <pose-model>

Example:
  ./fix_bad_keypoints.sh yolo11l

Supported pose models:
  yolo11n
  yolo11s
  yolo11m
  yolo11l
  yolo11x

Environment overrides:
  PROTOTYPE_DIR       Prototype root. Defaults to the parent of this script.
  SCRATCH_ROOT        Scratch root. Defaults to $HOME/scratch.
  PYTHON_BIN          Python executable. Defaults to python, then python3.
  MODEL_PATH          Defaults to $PROTOTYPE_DIR/pose_models/ultralytics/<pose-model>-pose.pt
  BAD_REPORT_PATH     Defaults to $SCRATCH_ROOT/keypoints/bad_keypoints_report.csv
  EMPTY_REPORT_PATH   Defaults to $SCRATCH_ROOT/keypoints/empty_windows_report.csv
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

quote_cmd() {
  local parts=()
  local arg
  for arg in "$@"; do
    parts+=("$(printf '%q' "$arg")")
  done
  local IFS=' '
  printf '%s' "${parts[*]}"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

run_cmd() {
  log "CMD: $(quote_cmd "$@")"
  "$@"
}

remove_stale_report() {
  local report_path="$1"
  if [[ -f "$report_path" ]]; then
    rm -f -- "$report_path"
  fi
}

delete_if_report_exists() {
  local report_path="$1"
  shift

  if [[ ! -f "$report_path" ]]; then
    log "No new report at $report_path; skipping delete step."
    return 0
  fi

  run_cmd "$@"
}

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s' "$PYTHON_BIN"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    printf '%s' "python"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "python3"
    return 0
  fi

  echo "ERROR: python/python3 not found in PATH." >&2
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 1
fi

case "$1" in
  -h|--help)
    usage
    exit 0
    ;;
esac

POSE_MODEL="${1%-pose}"

case "$POSE_MODEL" in
  yolo11n|yolo11s|yolo11m|yolo11l|yolo11x)
    ;;
  *)
    echo "ERROR: Unsupported pose model: $1" >&2
    usage >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export PROTOTYPE_DIR

SCRATCH_ROOT="${SCRATCH_ROOT:-$HOME/scratch}"
KEYPOINTS_ROOT="${SCRATCH_ROOT}/keypoints/UPFall_keypoints/${POSE_MODEL}/base"
UPFALL_ROOT="${SCRATCH_ROOT}/UPFall"
MODEL_PATH="${MODEL_PATH:-${PROTOTYPE_DIR}/pose_models/ultralytics/${POSE_MODEL}-pose.pt}"
BAD_REPORT_PATH="${BAD_REPORT_PATH:-${SCRATCH_ROOT}/keypoints/bad_keypoints_report.csv}"
EMPTY_REPORT_PATH="${EMPTY_REPORT_PATH:-${SCRATCH_ROOT}/keypoints/empty_windows_report.csv}"
PYTHON_BIN="$(resolve_python_bin)"

mkdir -p -- "$(dirname -- "$BAD_REPORT_PATH")"
mkdir -p -- "$(dirname -- "$EMPTY_REPORT_PATH")"

if [[ ! -d "$PROTOTYPE_DIR" ]]; then
  echo "ERROR: Prototype directory not found: $PROTOTYPE_DIR" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: Pose model weights not found: $MODEL_PATH" >&2
  exit 1
fi

cd "$PROTOTYPE_DIR"

log "Starting bad keypoint cleanup pipeline for ${POSE_MODEL}"
log "Prototype root: ${PROTOTYPE_DIR}"
log "Keypoints root: ${KEYPOINTS_ROOT}"
log "UP-Fall root:   ${UPFALL_ROOT}"
log "Model path:     ${MODEL_PATH}"

log "Step 1/7: Scan for bad keypoint tracks."
remove_stale_report "$BAD_REPORT_PATH"
run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/check_keypoints_tracking.py" \
  --root "$KEYPOINTS_ROOT" \
  --output "$BAD_REPORT_PATH"

log "Step 2/7: Delete reported bad keypoint entries."
delete_if_report_exists \
  "$BAD_REPORT_PATH" \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/delete_reported_entries.py" \
  "$BAD_REPORT_PATH" \
  --root "$KEYPOINTS_ROOT" \
  --execute

log "Step 3/7: Regenerate keypoints."
run_cmd \
  "$PYTHON_BIN" -m dataset_helpers.get_keypoints_files \
  --subjects 1-17 \
  --camera 2 \
  --lock-settings strict_lock \
  --upfall-root "$UPFALL_ROOT" \
  --output-root "$KEYPOINTS_ROOT" \
  --model-path "$MODEL_PATH" \
  --no-suspicious

log "Step 4/7: Scan for empty windows."
remove_stale_report "$EMPTY_REPORT_PATH"
run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/check_keypoints_empty_windows.py" \
  --root "$KEYPOINTS_ROOT" \
  --camera-dirs Camera2 \
  --output "$EMPTY_REPORT_PATH"

log "Step 5/7: Delete reported empty-window entries."
delete_if_report_exists \
  "$EMPTY_REPORT_PATH" \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/delete_reported_entries.py" \
  "$EMPTY_REPORT_PATH" \
  --root "$KEYPOINTS_ROOT" \
  --execute

log "Step 6/7: Regenerate keypoints again with --allow-region2-start."
run_cmd \
  "$PYTHON_BIN" -m dataset_helpers.get_keypoints_files \
  --subjects 1-17 \
  --camera 2 \
  --lock-settings strict_lock \
  --upfall-root "$UPFALL_ROOT" \
  --output-root "$KEYPOINTS_ROOT" \
  --model-path "$MODEL_PATH" \
  --no-suspicious \
  --allow-region2-start

log "Step 7/7: Re-scan empty windows, delete any remaining empties, then fix bad labels."
remove_stale_report "$EMPTY_REPORT_PATH"
run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/check_keypoints_empty_windows.py" \
  --root "$KEYPOINTS_ROOT" \
  --camera-dirs Camera2 \
  --output "$EMPTY_REPORT_PATH"

delete_if_report_exists \
  "$EMPTY_REPORT_PATH" \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/delete_reported_entries.py" \
  "$EMPTY_REPORT_PATH" \
  --root "$KEYPOINTS_ROOT" \
  --execute

run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/fix_label_20_to_2.py" \
  --root "$KEYPOINTS_ROOT"

log "Pipeline complete for ${POSE_MODEL}."
