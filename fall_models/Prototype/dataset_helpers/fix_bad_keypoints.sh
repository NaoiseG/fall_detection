#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage:
  ./fix_bad_keypoints.sh --keypoints-root <path> --upfall-root <path> --model-path <path> [options]

Example:
  ./fix_bad_keypoints.sh \
    --keypoints-root "$HOME/scratch/keypoints/UPFall_keypoints/yolo11l/base" \
    --upfall-root "$HOME/scratch/UPFall" \
    --model-path "$HOME/models/yolo11l-pose.pt"

Required arguments:
  --keypoints-root PATH  Root directory containing the keypoint tree to clean.
  --upfall-root PATH     Root directory of the UP-Fall frames tree used for regeneration.
  --model-path PATH      Pose model weights passed through to get_keypoints_files.py.

Optional arguments:
  --subjects SPEC        Subject list/range to regenerate, e.g. 1-17 or 1,3,7.
                         Defaults to subjects auto-detected from --keypoints-root,
                         then --upfall-root.
  --lock-settings NAME   Lock preset for regeneration. Defaults to strict_lock.
  --report-dir PATH      Directory for intermediate CSV reports.
  -h, --help             Show this help text.

Environment overrides:
  PROTOTYPE_DIR       Prototype root. Defaults to the parent of this script.
  PYTHON_BIN          Python executable. Defaults to python, then python3.
  REPORT_DIR          Default report directory when --report-dir is omitted.
  CAMERA1_REPORT_PATH Override the Camera1 report path.
  BAD_REPORT_PATH     Override the Camera2 tracking report path.
  EMPTY_REPORT_PATH   Override the Camera2 empty-window report path.

Notes:
  - --keypoints-root should be the directory that directly contains Subject* folders.
  - The cleanup pipeline still targets the UP-Fall Camera1 and Camera2 layout.
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

require_value() {
  local flag_name="$1"
  local flag_value="${2:-}"
  if [[ -z "$flag_value" ]]; then
    echo "ERROR: Missing value for ${flag_name}" >&2
    usage >&2
    exit 1
  fi
}

validate_lock_settings() {
  case "$1" in
    strict_lock|default)
      ;;
    *)
      echo "ERROR: Unsupported lock setting: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

discover_subjects() {
  local root="$1"
  local -a subject_ids=()
  local subject_dir=""
  local base_name=""
  local subject_id=""
  local joined=""

  [[ -d "$root" ]] || return 1

  while IFS= read -r -d '' subject_dir; do
    base_name="$(basename -- "$subject_dir")"
    if [[ "$base_name" =~ ^Subject([0-9]+)$ ]]; then
      subject_ids+=("${BASH_REMATCH[1]}")
    fi
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -name 'Subject*' -print0 2>/dev/null)

  [[ ${#subject_ids[@]} -gt 0 ]] || return 1

  while IFS= read -r subject_id; do
    [[ -n "$subject_id" ]] || continue
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="$subject_id"
  done < <(printf '%s\n' "${subject_ids[@]}" | sort -n -u)

  [[ -n "$joined" ]] || return 1
  printf '%s' "$joined"
}

KEYPOINTS_ROOT=""
UPFALL_ROOT=""
MODEL_PATH=""
SUBJECTS="${SUBJECTS:-}"
LOCK_SETTINGS="${LOCK_SETTINGS:-strict_lock}"
REPORT_DIR="${REPORT_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keypoints-root)
      require_value "$1" "${2:-}"
      KEYPOINTS_ROOT="$2"
      shift 2
      ;;
    --upfall-root)
      require_value "$1" "${2:-}"
      UPFALL_ROOT="$2"
      shift 2
      ;;
    --model-path)
      require_value "$1" "${2:-}"
      MODEL_PATH="$2"
      shift 2
      ;;
    --subjects)
      require_value "$1" "${2:-}"
      SUBJECTS="$2"
      shift 2
      ;;
    --lock-settings)
      require_value "$1" "${2:-}"
      LOCK_SETTINGS="$2"
      shift 2
      ;;
    --report-dir)
      require_value "$1" "${2:-}"
      REPORT_DIR="$2"
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

if [[ -z "$KEYPOINTS_ROOT" || -z "$UPFALL_ROOT" || -z "$MODEL_PATH" ]]; then
  echo "ERROR: --keypoints-root, --upfall-root, and --model-path are required." >&2
  usage >&2
  exit 1
fi

validate_lock_settings "$LOCK_SETTINGS"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export PROTOTYPE_DIR

if [[ ! -d "$PROTOTYPE_DIR" ]]; then
  echo "ERROR: Prototype directory not found: $PROTOTYPE_DIR" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: Pose model weights not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -d "$KEYPOINTS_ROOT" ]]; then
  echo "ERROR: Keypoints root not found: $KEYPOINTS_ROOT" >&2
  exit 1
fi

if [[ ! -d "$UPFALL_ROOT" ]]; then
  echo "ERROR: UP-Fall root not found: $UPFALL_ROOT" >&2
  exit 1
fi

if [[ -z "$REPORT_DIR" ]]; then
  REPORT_DIR="${KEYPOINTS_ROOT}/_fix_bad_keypoints_reports"
fi

CAMERA1_REPORT_PATH="${CAMERA1_REPORT_PATH:-${REPORT_DIR}/camera1_background_report.csv}"
BAD_REPORT_PATH="${BAD_REPORT_PATH:-${REPORT_DIR}/bad_keypoints_report.csv}"
EMPTY_REPORT_PATH="${EMPTY_REPORT_PATH:-${REPORT_DIR}/empty_windows_report.csv}"
PYTHON_BIN="$(resolve_python_bin)"

mkdir -p -- "$(dirname -- "$CAMERA1_REPORT_PATH")"
mkdir -p -- "$(dirname -- "$BAD_REPORT_PATH")"
mkdir -p -- "$(dirname -- "$EMPTY_REPORT_PATH")"

SUBJECT_SOURCE="cli"
if [[ -z "$SUBJECTS" ]]; then
  if SUBJECTS="$(discover_subjects "$KEYPOINTS_ROOT")"; then
    SUBJECT_SOURCE="keypoints-root"
  elif SUBJECTS="$(discover_subjects "$UPFALL_ROOT")"; then
    SUBJECT_SOURCE="upfall-root"
  else
    echo "ERROR: Could not infer subjects from $KEYPOINTS_ROOT or $UPFALL_ROOT. Pass --subjects explicitly." >&2
    exit 1
  fi
fi

cd "$PROTOTYPE_DIR"

log "Starting bad keypoint cleanup pipeline"
log "Prototype root: ${PROTOTYPE_DIR}"
log "Keypoints root: ${KEYPOINTS_ROOT}"
log "UP-Fall root:   ${UPFALL_ROOT}"
log "Model path:     ${MODEL_PATH}"
log "Subjects:       ${SUBJECTS} (${SUBJECT_SOURCE})"
log "Lock settings:  ${LOCK_SETTINGS}"
log "Report dir:     ${REPORT_DIR}"

log "Step 1/10: Scan Camera1 for background/reflection switches."
remove_stale_report "$CAMERA1_REPORT_PATH"
run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/check_camera1_background_switches.py" \
  --root "$KEYPOINTS_ROOT" \
  --output "$CAMERA1_REPORT_PATH"

log "Step 2/10: Delete reported Camera1 entries."
delete_if_report_exists \
  "$CAMERA1_REPORT_PATH" \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/delete_reported_entries.py" \
  "$CAMERA1_REPORT_PATH" \
  --root "$KEYPOINTS_ROOT" \
  --execute

log "Step 3/10: Regenerate Camera1 keypoints."
run_cmd \
  "$PYTHON_BIN" -m dataset_helpers.get_keypoints_files \
  --subjects "$SUBJECTS" \
  --camera 1 \
  --lock-settings "$LOCK_SETTINGS" \
  --upfall-root "$UPFALL_ROOT" \
  --output-root "$KEYPOINTS_ROOT" \
  --model-path "$MODEL_PATH"

log "Step 4/10: Scan Camera2 for bad keypoint tracks."
remove_stale_report "$BAD_REPORT_PATH"
run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/check_keypoints_tracking.py" \
  --root "$KEYPOINTS_ROOT" \
  --output "$BAD_REPORT_PATH"

log "Step 5/10: Delete reported Camera2 bad keypoint entries."
delete_if_report_exists \
  "$BAD_REPORT_PATH" \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/delete_reported_entries.py" \
  "$BAD_REPORT_PATH" \
  --root "$KEYPOINTS_ROOT" \
  --execute

log "Step 6/10: Regenerate Camera2 keypoints."
run_cmd \
  "$PYTHON_BIN" -m dataset_helpers.get_keypoints_files \
  --subjects "$SUBJECTS" \
  --camera 2 \
  --lock-settings "$LOCK_SETTINGS" \
  --upfall-root "$UPFALL_ROOT" \
  --output-root "$KEYPOINTS_ROOT" \
  --model-path "$MODEL_PATH" \
  --no-suspicious

log "Step 7/10: Scan Camera2 for empty windows."
remove_stale_report "$EMPTY_REPORT_PATH"
run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/check_keypoints_empty_windows.py" \
  --root "$KEYPOINTS_ROOT" \
  --camera-dirs Camera2 \
  --output "$EMPTY_REPORT_PATH"

log "Step 8/10: Delete reported Camera2 empty-window entries."
delete_if_report_exists \
  "$EMPTY_REPORT_PATH" \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/delete_reported_entries.py" \
  "$EMPTY_REPORT_PATH" \
  --root "$KEYPOINTS_ROOT" \
  --execute

log "Step 9/10: Regenerate Camera2 keypoints again with --allow-region2-start."
run_cmd \
  "$PYTHON_BIN" -m dataset_helpers.get_keypoints_files \
  --subjects "$SUBJECTS" \
  --camera 2 \
  --lock-settings "$LOCK_SETTINGS" \
  --upfall-root "$UPFALL_ROOT" \
  --output-root "$KEYPOINTS_ROOT" \
  --model-path "$MODEL_PATH" \
  --no-suspicious \
  --allow-region2-start

log "Step 10/10: Re-scan Camera2 empty windows, delete remaining empties, then fix bad labels."
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

log "Pipeline complete."
