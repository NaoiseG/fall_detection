#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage:
  ./count_keypoint_anomalies.sh --root <path> [options]

Example:
  ./count_keypoint_anomalies.sh \
    --root "$HOME/scratch/keypoints/UPFall_keypoints/yolo11l/base"

Required arguments:
  --root PATH          Root directory to scan recursively for keypoint NPZ files.

Optional arguments:
  --activities NAME... Restrict all three checks to the listed activities.
  --python-bin PATH    Python executable to use. Defaults to python, then python3.
  -h, --help           Show this help text.

Notes:
  - No CSV files are written.
  - The empty-window check is run on Camera2 only, to match fix_bad_keypoints.sh.
  - The final total is not deduplicated across checks; one file may be counted by
    more than one checker.
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" >&2
}

require_value() {
  local flag_name="$1"
  local flag_value="${2:-}"
  if [[ -z "$flag_value" ]]; then
    echo "ERROR: Missing value for ${flag_name}" >&2
    usage >&2
    exit 1
  fi
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

extract_summary_count() {
  local summary_pattern="$1"
  local output_file="$2"
  local summary_line=""

  summary_line="$(grep -E "$summary_pattern" "$output_file" | tail -n 1 || true)"
  if [[ -z "$summary_line" ]]; then
    echo "ERROR: Could not parse checker summary." >&2
    cat "$output_file" >&2
    return 1
  fi

  sed -E 's/.*: ([0-9]+) \/ .*/\1/' <<<"$summary_line"
}

run_checker() {
  local label="$1"
  local summary_pattern="$2"
  shift 2

  local output_file=""
  local count=""
  output_file="$(mktemp)"

  log "Running ${label}"
  if ! "$@" >"$output_file" 2>&1; then
    echo "ERROR: ${label} failed." >&2
    cat "$output_file" >&2
    rm -f -- "$output_file"
    return 1
  fi

  count="$(extract_summary_count "$summary_pattern" "$output_file")" || {
    rm -f -- "$output_file"
    return 1
  }

  rm -f -- "$output_file"
  printf '%s' "$count"
}

ROOT=""
PYTHON_BIN="${PYTHON_BIN:-}"
declare -a ACTIVITIES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      require_value "$1" "${2:-}"
      ROOT="$2"
      shift 2
      ;;
    --activities)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        ACTIVITIES+=("$1")
        shift
      done
      if [[ ${#ACTIVITIES[@]} -eq 0 ]]; then
        echo "ERROR: --activities requires at least one activity name." >&2
        usage >&2
        exit 1
      fi
      ;;
    --python-bin)
      require_value "$1" "${2:-}"
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unrecognized argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ROOT" ]]; then
  echo "ERROR: --root is required." >&2
  usage >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROTOTYPE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="$(resolve_python_bin)"

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: Root directory not found: $ROOT" >&2
  exit 1
fi

declare -a ACTIVITY_ARGS=()
if [[ ${#ACTIVITIES[@]} -gt 0 ]]; then
  ACTIVITY_ARGS=(--activities "${ACTIVITIES[@]}")
fi

declare -a CAMERA1_CMD=(
  "$PYTHON_BIN"
  "$PROTOTYPE_DIR/dataset_helpers/check_camera1_background_switches.py"
  --root "$ROOT"
)
declare -a TRACKING_CMD=(
  "$PYTHON_BIN"
  "$PROTOTYPE_DIR/dataset_helpers/check_keypoints_tracking.py"
  --root "$ROOT"
)
declare -a EMPTY_CMD=(
  "$PYTHON_BIN"
  "$PROTOTYPE_DIR/dataset_helpers/check_keypoints_empty_windows.py"
  --root "$ROOT"
  --camera-dirs Camera2
)

if [[ ${#ACTIVITY_ARGS[@]} -gt 0 ]]; then
  CAMERA1_CMD+=("${ACTIVITY_ARGS[@]}")
  TRACKING_CMD+=("${ACTIVITY_ARGS[@]}")
  EMPTY_CMD+=("${ACTIVITY_ARGS[@]}")
fi

CAMERA1_COUNT="$(
  run_checker \
    "Camera1 background/reflection switches" \
    'Suspicious files: [0-9]+ / [0-9]+' \
    "${CAMERA1_CMD[@]}"
)"

TRACKING_COUNT="$(
  run_checker \
    "Camera2 bad tracking" \
    'Suspicious files: [0-9]+ / [0-9]+' \
    "${TRACKING_CMD[@]}"
)"

EMPTY_COUNT="$(
  run_checker \
    "Camera2 empty windows" \
    'Flagged files: [0-9]+ / [0-9]+' \
    "${EMPTY_CMD[@]}"
)"

TOTAL_COUNT=$((CAMERA1_COUNT + TRACKING_COUNT + EMPTY_COUNT))

printf 'Anomaly counts for %s\n' "$ROOT"
printf 'Camera1 background/reflection switches: %s\n' "$CAMERA1_COUNT"
printf 'Camera2 bad tracking: %s\n' "$TRACKING_COUNT"
printf 'Camera2 empty windows: %s\n' "$EMPTY_COUNT"
printf 'Total flagged reports (not deduplicated): %s\n' "$TOTAL_COUNT"
