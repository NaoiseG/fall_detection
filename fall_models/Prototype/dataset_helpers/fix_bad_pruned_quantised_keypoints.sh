#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage:
  bash dataset_helpers/fix_bad_pruned_quantised_keypoints.sh --arch yolo|all [options]

Required arguments:
  --arch NAME           Backend family to run: yolo or all.
                        `all` is accepted as a compatibility alias and runs
                        every full-pruned YOLO engine job.

Optional arguments:
  --subjects SPEC       Pass through to fix_bad_keypoints.sh, e.g. 16-17 or 16,17.
                        Default: 16-17
  --lock-settings NAME  Pass through to fix_bad_keypoints.sh.
  --dry-run             Print planned commands without executing them.
  --force               Ignore existing .done markers and rerun matching jobs.
  --continue-on-error   Keep going after job failures (default; accepted explicitly for clarity).
  --state-dir PATH      Persistent state directory.
                        Default: /home/jetson/NaoiseG/fall_detection/dataset_helpers/.fix_bad_pruned_quantised_keypoints_state
  --datasets-root PATH  Root containing the full-pruned keypoint trees.
                        Default: /home/jetson/NaoiseG/fall_detection/Datasets/full_pruned
  --models-root PATH    Root containing the full-pruned engine folders.
                        Default: /home/jetson/NaoiseG/fall_detection/pose_models/full_pruned
  --upfall-root PATH    UP-Fall root passed to fix_bad_keypoints.sh.
                        Default: sibling UPFall directory next to --datasets-root
                        (normally /home/jetson/NaoiseG/fall_detection/Datasets/UPFall)
  -h, --help            Show this help text.

Examples:
  Run every full-pruned job:
    bash dataset_helpers/fix_bad_pruned_quantised_keypoints.sh --arch all

  Run the YOLO-only job set:
    bash dataset_helpers/fix_bad_pruned_quantised_keypoints.sh --arch yolo

  Force rerun everything:
    bash dataset_helpers/fix_bad_pruned_quantised_keypoints.sh --arch all --force

  Dry run:
    bash dataset_helpers/fix_bad_pruned_quantised_keypoints.sh --arch all --dry-run

Notes:
  - Jobs cover yolo11n/yolo11s/yolo11m/yolo11l with pruned_80 and pruned_90.
  - yolo11x uses pruned_70 and pruned_80.
  - Only engine-backed fp32/fp16/int8 jobs are included.
  - Successful jobs create <state-dir>/<job-id>.done markers.
  - Per-job logs are written to <state-dir>/logs/<job-id>.log.
  - Existing .done markers are skipped by default and only ignored with --force.
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

warn() {
  printf '[%s] WARNING: %s\n' "$(timestamp)" "$*" >&2
}

fail() {
  printf '[%s] ERROR: %s\n' "$(timestamp)" "$*" >&2
  exit 1
}

require_value() {
  local flag_name="$1"
  local flag_value="${2:-}"
  if [[ -z "$flag_value" ]]; then
    printf 'ERROR: Missing value for %s\n' "$flag_name" >&2
    usage >&2
    exit 1
  fi
}

trim_field() {
  local value="$1"
  local width="$2"
  local limit=0

  if (( ${#value} <= width )); then
    printf '%s' "$value"
    return 0
  fi

  if (( width <= 3 )); then
    printf '%s' "${value:0:width}"
    return 0
  fi

  limit=$((width - 3))
  printf '%s...' "${value:0:limit}"
}

make_job_id() {
  local id="$1"
  local part=""
  shift

  for part in "$@"; do
    [[ -n "$part" ]] || continue
    id+="__${part}"
  done

  printf '%s' "$id"
}

job_done_path() {
  local job_id="$1"
  printf '%s/%s.done' "$STATE_DIR" "$job_id"
}

job_log_path() {
  local job_id="$1"
  printf '%s/logs/%s.log' "$STATE_DIR" "$job_id"
}

write_simple_log() {
  local log_path="$1"
  shift

  {
    printf '[%s] %s\n' "$(timestamp)" "$1"
    shift || true
    if (( $# > 0 )); then
      printf '%s\n' "$@"
    fi
  } >"$log_path"
}

add_job() {
  local job_id="$1"
  local backend="$2"
  local variant="$3"
  local keypoints_root="$4"
  local model_info="$5"

  JOB_ORDER+=("$job_id")
  JOB_BACKEND["$job_id"]="$backend"
  JOB_VARIANT["$job_id"]="$variant"
  JOB_KEYPOINTS_ROOT["$job_id"]="$keypoints_root"
  JOB_MODEL_INFO["$job_id"]="$model_info"
}

add_yolo_jobs() {
  local -a models=(yolo11n yolo11s yolo11m yolo11l yolo11x)
  local -a precisions=(fp32 fp16 int8)
  local -a prune_levels=()
  local model=""
  local prune_level=""
  local precision=""
  local keypoints_root=""
  local engine_path=""
  local job_id=""

  for model in "${models[@]}"; do
    case "$model" in
      yolo11x)
        prune_levels=(70 80)
        ;;
      *)
        prune_levels=(80 90)
        ;;
    esac

    for prune_level in "${prune_levels[@]}"; do
      for precision in "${precisions[@]}"; do
        keypoints_root="${DATASETS_ROOT}/${model}/pruned_${prune_level}/${precision}"
        engine_path="${MODELS_ROOT}/${model}_pruned_${prune_level}/weights/${model}_pruned_${prune_level}_${precision}.engine"
        job_id="$(make_job_id yolo "$model" "pruned_${prune_level}" "$precision")"

        add_job \
          "$job_id" \
          "yolo" \
          "${model}/pruned_${prune_level}/${precision}" \
          "$keypoints_root" \
          "$(basename -- "$engine_path")"

        JOB_MODEL_PATH["$job_id"]="$engine_path"
      done
    done
  done
}

build_command() {
  local job_id="$1"
  local -n cmd_ref="$2"

  cmd_ref=(
    bash "$FIX_BAD_KEYPOINTS_SCRIPT"
    --keypoints-root "${JOB_KEYPOINTS_ROOT[$job_id]}"
    --upfall-root "$UPFALL_ROOT"
    --model-path "${JOB_MODEL_PATH[$job_id]}"
  )

  if [[ -n "$SUBJECTS" ]]; then
    cmd_ref+=(--subjects "$SUBJECTS")
  fi

  if [[ -n "$LOCK_SETTINGS" ]]; then
    cmd_ref+=(--lock-settings "$LOCK_SETTINGS")
  fi
}

collect_missing_paths() {
  local job_id="$1"
  local -n missing_ref="$2"

  missing_ref=()

  [[ -f "$FIX_BAD_KEYPOINTS_SCRIPT" ]] || missing_ref+=("$FIX_BAD_KEYPOINTS_SCRIPT (fix_bad_keypoints.sh)")
  [[ -d "${JOB_KEYPOINTS_ROOT[$job_id]}" ]] || missing_ref+=("${JOB_KEYPOINTS_ROOT[$job_id]} (keypoints_root)")
  [[ -d "$UPFALL_ROOT" ]] || missing_ref+=("$UPFALL_ROOT (upfall_root)")
  [[ -f "${JOB_MODEL_PATH[$job_id]}" ]] || missing_ref+=("${JOB_MODEL_PATH[$job_id]} (model_path)")
}

record_status() {
  local job_id="$1"
  local status="$2"
  local message="${3:-}"

  JOB_STATUS["$job_id"]="$status"
  JOB_MESSAGE["$job_id"]="$message"
}

run_job() {
  local job_id="$1"
  local done_path=""
  local log_path=""
  local command_str=""
  local -a cmd=()
  local -a missing_paths=()
  local rc=0

  done_path="$(job_done_path "$job_id")"
  log_path="$(job_log_path "$job_id")"
  build_command "$job_id" cmd
  command_str="$(quote_cmd "${cmd[@]}")"

  if [[ -f "$done_path" && "$FORCE" -ne 1 ]]; then
    log "Skipping ${job_id}: already completed (${done_path})"
    record_status "$job_id" "skipped_already_done" "done marker exists"
    write_simple_log "$log_path" "Skipping ${job_id}: already completed." "Done marker: ${done_path}"
    SKIPPED_ALREADY_DONE_COUNT=$((SKIPPED_ALREADY_DONE_COUNT + 1))
    return 0
  fi

  if [[ "$FORCE" -eq 1 && "$DRY_RUN" -ne 1 && -f "$done_path" ]]; then
    rm -f -- "$done_path"
  fi

  collect_missing_paths "$job_id" missing_paths
  if (( ${#missing_paths[@]} > 0 )); then
    warn "Skipping ${job_id}: missing required paths."
    record_status "$job_id" "skipped_missing_paths" "missing required paths"
    {
      printf '[%s] Skipping %s: missing required paths.\n' "$(timestamp)" "$job_id"
      printf 'Command: %s\n' "$command_str"
      printf 'Missing paths:\n'
      printf '  - %s\n' "${missing_paths[@]}"
    } >"$log_path"
    SKIPPED_MISSING_PATHS_COUNT=$((SKIPPED_MISSING_PATHS_COUNT + 1))
    return 0
  fi

  log "Job ${job_id}"
  log "CMD: ${command_str}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    record_status "$job_id" "dry_run" "command printed only"
    write_simple_log "$log_path" "Dry run for ${job_id}" "Command: ${command_str}"
    return 0
  fi

  if {
    printf '[%s] CMD: %s\n' "$(timestamp)" "$command_str"
    "${cmd[@]}"
  } > >(tee "$log_path") 2>&1; then
    touch "$done_path"
    record_status "$job_id" "completed" "success"
    COMPLETED_NOW_COUNT=$((COMPLETED_NOW_COUNT + 1))
    return 0
  fi

  rc=$?
  rm -f -- "$done_path"
  warn "Job ${job_id} failed with exit code ${rc}. See ${log_path}"
  record_status "$job_id" "failed" "exit_code=${rc}"
  FAILED_COUNT=$((FAILED_COUNT + 1))

  if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then
    return 0
  fi

  return "$rc"
}

print_summary() {
  local job_id=""
  local backend=""
  local variant=""
  local keypoints_root=""
  local model_info=""
  local status=""

  printf '\n'
  printf '%-10s | %-28s | %-72s | %-40s | %s\n' "backend" "variant" "keypoints_root" "model_info" "status"
  printf '%-10s-+-%-28s-+-%-72s-+-%-40s-+-%s\n' \
    "----------" \
    "----------------------------" \
    "------------------------------------------------------------------------" \
    "----------------------------------------" \
    "--------------------"

  for job_id in "${JOB_ORDER[@]}"; do
    backend="$(trim_field "${JOB_BACKEND[$job_id]}" 10)"
    variant="$(trim_field "${JOB_VARIANT[$job_id]}" 28)"
    keypoints_root="$(trim_field "${JOB_KEYPOINTS_ROOT[$job_id]}" 72)"
    model_info="$(trim_field "${JOB_MODEL_INFO[$job_id]}" 40)"
    status="${JOB_STATUS[$job_id]:-pending}"

    printf '%-10s | %-28s | %-72s | %-40s | %s\n' \
      "$backend" \
      "$variant" \
      "$keypoints_root" \
      "$model_info" \
      "$status"
  done

  printf '\n'
  printf 'total_jobs:            %d\n' "${#JOB_ORDER[@]}"
  printf 'completed_now:         %d\n' "$COMPLETED_NOW_COUNT"
  printf 'failed:                %d\n' "$FAILED_COUNT"
  printf 'skipped_already_done:  %d\n' "$SKIPPED_ALREADY_DONE_COUNT"
  printf 'skipped_missing_paths: %d\n' "$SKIPPED_MISSING_PATHS_COUNT"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'dry_run:               yes\n'
  fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROTOTYPE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
FIX_BAD_KEYPOINTS_SCRIPT="${SCRIPT_DIR}/fix_bad_keypoints.sh"

DEFAULT_REPO_ROOT="/home/jetson/NaoiseG/fall_detection"
ARCH=""
SUBJECTS="16-17"
LOCK_SETTINGS=""
DRY_RUN=0
FORCE=0
CONTINUE_ON_ERROR=1
DATASETS_ROOT="${DEFAULT_REPO_ROOT}/Datasets/full_pruned"
MODELS_ROOT="${DEFAULT_REPO_ROOT}/pose_models/full_pruned"
UPFALL_ROOT=""
STATE_DIR="${DEFAULT_REPO_ROOT}/dataset_helpers/.fix_bad_pruned_quantised_keypoints_state"

declare -a JOB_ORDER=()
declare -A JOB_BACKEND=()
declare -A JOB_VARIANT=()
declare -A JOB_KEYPOINTS_ROOT=()
declare -A JOB_MODEL_INFO=()
declare -A JOB_STATUS=()
declare -A JOB_MESSAGE=()
declare -A JOB_MODEL_PATH=()

COMPLETED_NOW_COUNT=0
FAILED_COUNT=0
SKIPPED_ALREADY_DONE_COUNT=0
SKIPPED_MISSING_PATHS_COUNT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --arch)
      require_value "$1" "${2:-}"
      ARCH="$2"
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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    --state-dir)
      require_value "$1" "${2:-}"
      STATE_DIR="$2"
      shift 2
      ;;
    --datasets-root)
      require_value "$1" "${2:-}"
      DATASETS_ROOT="$2"
      shift 2
      ;;
    --models-root)
      require_value "$1" "${2:-}"
      MODELS_ROOT="$2"
      shift 2
      ;;
    --upfall-root)
      require_value "$1" "${2:-}"
      UPFALL_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'ERROR: Unrecognized argument: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "$ARCH" in
  yolo|all)
    ;;
  "")
    fail "--arch is required."
    ;;
  *)
    fail "Unsupported --arch: ${ARCH}. Choose from yolo or all."
    ;;
esac

if [[ -z "$UPFALL_ROOT" ]]; then
  UPFALL_ROOT="$(dirname -- "$DATASETS_ROOT")/UPFall"
fi

mkdir -p -- "$STATE_DIR" "${STATE_DIR}/logs"

[[ -d "$PROTOTYPE_DIR" ]] || fail "Prototype directory not found: ${PROTOTYPE_DIR}"
[[ -f "$FIX_BAD_KEYPOINTS_SCRIPT" ]] || fail "fix_bad_keypoints.sh not found: ${FIX_BAD_KEYPOINTS_SCRIPT}"

add_yolo_jobs

if (( ${#JOB_ORDER[@]} == 0 )); then
  fail "No jobs were generated for --arch ${ARCH}."
fi

log "Starting full-pruned bad keypoint batch cleanup"
log "Prototype root: ${PROTOTYPE_DIR}"
log "Arch:           ${ARCH}"
log "Datasets root:  ${DATASETS_ROOT}"
log "Models root:    ${MODELS_ROOT}"
log "UP-Fall root:   ${UPFALL_ROOT}"
log "State dir:      ${STATE_DIR}"
log "Dry run:        ${DRY_RUN}"
log "Force:          ${FORCE}"
log "Total jobs:     ${#JOB_ORDER[@]}"
if [[ -n "$SUBJECTS" ]]; then
  log "Subjects:       ${SUBJECTS}"
fi
if [[ -n "$LOCK_SETTINGS" ]]; then
  log "Lock settings:  ${LOCK_SETTINGS}"
fi

for job_id in "${JOB_ORDER[@]}"; do
  run_job "$job_id"
done

print_summary

if (( FAILED_COUNT > 0 )); then
  exit 1
fi
