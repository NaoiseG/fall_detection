#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage:
  bash dataset_helpers/run_fix_bad_keypoints_batch.sh --arch yolo|alphapose|vitpose|all [options]

Required arguments:
  --arch NAME           Backend family to run: yolo, alphapose, vitpose, or all.

Optional arguments:
  --subjects SPEC       Pass through to fix_bad_keypoints.sh, e.g. 1-17 or 1,3,7.
  --lock-settings NAME  Pass through to fix_bad_keypoints.sh.
  --dry-run             Print planned commands without executing them.
  --force               Ignore existing .done markers and rerun matching jobs.
  --continue-on-error   Keep going after job failures (default; accepted explicitly for clarity).
  --state-dir PATH      Persistent state directory.
                        Default: /home/jetson/NaoiseG/fall_detection/dataset_helpers/.fix_bad_keypoints_state
  --alphapose-root PATH AlphaPose repo root passed to fix_bad_keypoints.sh.
                        Default: /home/jetson/NaoiseG/fall_detection/pose_models/Alphapose
  --datasets-root PATH  Root containing UPFall, UPFall_keypoints*, etc.
                        Default: /home/jetson/NaoiseG/fall_detection/Datasets
  --models-root PATH    Root containing quantised engine folders.
                        Default: /home/jetson/NaoiseG/fall_detection/pose_models/quantised
  --upfall-root PATH    UP-Fall root passed to fix_bad_keypoints.sh.
                        Default: /home/jetson/NaoiseG/fall_detection/Datasets/UPFall
  -h, --help            Show this help text.

Examples:
  Run everything:
    bash dataset_helpers/run_fix_bad_keypoints_batch.sh --arch all

  Run only YOLO:
    bash dataset_helpers/run_fix_bad_keypoints_batch.sh --arch yolo

  Resume only AlphaPose:
    bash dataset_helpers/run_fix_bad_keypoints_batch.sh --arch alphapose

  Force rerun all ViTPose jobs:
    bash dataset_helpers/run_fix_bad_keypoints_batch.sh --arch vitpose --force

  Dry run:
    bash dataset_helpers/run_fix_bad_keypoints_batch.sh --arch all --dry-run

Notes:
  - Only quantised / engine-backed jobs are included. Base jobs are excluded.
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
  local model=""
  local precision=""
  local keypoints_folder=""
  local keypoints_root=""
  local engine_path=""
  local job_id=""

  for model in "${models[@]}"; do
    for precision in "${precisions[@]}"; do
      case "$precision" in
        fp32) keypoints_folder="fp32" ;;
        fp16) keypoints_folder="fp16" ;;
        int8) keypoints_folder="int8" ;;
        *)
          fail "Unhandled YOLO precision mapping: $precision"
          ;;
      esac

      keypoints_root="${DATASETS_ROOT}/UPFall_keypoints/${model}/${keypoints_folder}"
      engine_path="${MODELS_ROOT}/ultralytics/${model}-pose/${model}-pose_${precision}.engine"
      job_id="$(make_job_id yolo "$model" "$precision")"

      add_job \
        "$job_id" \
        "yolo" \
        "${model}/${precision}" \
        "$keypoints_root" \
        "$(basename -- "$engine_path")"

      JOB_MODEL_PATH["$job_id"]="$engine_path"
    done
  done
}

add_alphapose_jobs() {
  local job_id=""

  job_id="$(make_job_id alphapose fp32_fp32)"
  add_job \
    "$job_id" \
    "alphapose" \
    "fp32_fp32" \
    "${DATASETS_ROOT}/UPFall_keypoints_alpha/fp32_fp32" \
    "det=$(basename -- "${MODELS_ROOT}/alphapose/yolov3_spp_fp32.engine") fastpose=$(basename -- "${MODELS_ROOT}/alphapose/fastpose_fp32.engine")"
  JOB_DETECTOR_WEIGHTS["$job_id"]="${MODELS_ROOT}/alphapose/yolov3_spp_fp32.engine"
  JOB_FASTPOSE_WEIGHTS["$job_id"]="${MODELS_ROOT}/alphapose/fastpose_fp32.engine"

  job_id="$(make_job_id alphapose fp16_fp16)"
  add_job \
    "$job_id" \
    "alphapose" \
    "fp16_fp16" \
    "${DATASETS_ROOT}/UPFall_keypoints_alpha/fp16_fp16" \
    "det=$(basename -- "${MODELS_ROOT}/alphapose/yolov3_spp_fp16.engine") fastpose=$(basename -- "${MODELS_ROOT}/alphapose/fastpose_fp16.engine")"
  JOB_DETECTOR_WEIGHTS["$job_id"]="${MODELS_ROOT}/alphapose/yolov3_spp_fp16.engine"
  JOB_FASTPOSE_WEIGHTS["$job_id"]="${MODELS_ROOT}/alphapose/fastpose_fp16.engine"

  job_id="$(make_job_id alphapose int8_fp16)"
  add_job \
    "$job_id" \
    "alphapose" \
    "int8_fp16" \
    "${DATASETS_ROOT}/UPFall_keypoints_alpha/int8_fp16" \
    "det=$(basename -- "${MODELS_ROOT}/alphapose/yolov3_spp_int8.engine") fastpose=$(basename -- "${MODELS_ROOT}/alphapose/fastpose_fp16.engine")"
  JOB_DETECTOR_WEIGHTS["$job_id"]="${MODELS_ROOT}/alphapose/yolov3_spp_int8.engine"
  JOB_FASTPOSE_WEIGHTS["$job_id"]="${MODELS_ROOT}/alphapose/fastpose_fp16.engine"
}

add_vitpose_jobs() {
  local job_id=""

  job_id="$(make_job_id vitpose fp32)"
  add_job \
    "$job_id" \
    "vitpose" \
    "fp32" \
    "${DATASETS_ROOT}/UPFall_keypoints_vitpose/fp32_fp32" \
    "det=$(basename -- "${MODELS_ROOT}/vitpose_trt/engines/detector_pekingu_rtdetr_r50vd_coco_o365_fp32.engine") pose=$(basename -- "${MODELS_ROOT}/vitpose_trt/engines/pose_usyd_community_vitpose_base_fp32.engine")"
  JOB_DETECTOR_MODEL["$job_id"]="${MODELS_ROOT}/vitpose_trt/engines/detector_pekingu_rtdetr_r50vd_coco_o365_fp32.engine"
  JOB_POSE_MODEL["$job_id"]="${MODELS_ROOT}/vitpose_trt/engines/pose_usyd_community_vitpose_base_fp32.engine"

  job_id="$(make_job_id vitpose fp16)"
  add_job \
    "$job_id" \
    "vitpose" \
    "fp16" \
    "${DATASETS_ROOT}/UPFall_keypoints_vitpose/fp16" \
    "det=$(basename -- "${MODELS_ROOT}/vitpose_trt/engines/detector_pekingu_rtdetr_r50vd_coco_o365_fp16.engine") pose=$(basename -- "${MODELS_ROOT}/vitpose_trt/engines/pose_usyd_community_vitpose_base_fp16.engine")"
  JOB_DETECTOR_MODEL["$job_id"]="${MODELS_ROOT}/vitpose_trt/engines/detector_pekingu_rtdetr_r50vd_coco_o365_fp16.engine"
  JOB_POSE_MODEL["$job_id"]="${MODELS_ROOT}/vitpose_trt/engines/pose_usyd_community_vitpose_base_fp16.engine"
}

build_command() {
  local job_id="$1"
  local -n cmd_ref="$2"
  local backend="${JOB_BACKEND[$job_id]}"

  cmd_ref=(
    bash "$FIX_BAD_KEYPOINTS_SCRIPT"
    --keypoints-root "${JOB_KEYPOINTS_ROOT[$job_id]}"
    --upfall-root "$UPFALL_ROOT"
  )

  case "$backend" in
    yolo)
      cmd_ref+=(--model-path "${JOB_MODEL_PATH[$job_id]}")
      ;;
    alphapose)
      cmd_ref+=(
        --pose-backend alphapose
        --alphapose-root "$ALPHAPOSE_ROOT"
        --fastpose-weights "${JOB_FASTPOSE_WEIGHTS[$job_id]}"
        --detector-weights "${JOB_DETECTOR_WEIGHTS[$job_id]}"
      )
      ;;
    vitpose)
      cmd_ref+=(
        --pose-backend vitpose
        --detector-model "${JOB_DETECTOR_MODEL[$job_id]}"
        --pose-model "${JOB_POSE_MODEL[$job_id]}"
      )
      ;;
    *)
      fail "Unsupported backend for job ${job_id}: ${backend}"
      ;;
  esac

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
  local backend="${JOB_BACKEND[$job_id]}"

  missing_ref=()

  [[ -f "$FIX_BAD_KEYPOINTS_SCRIPT" ]] || missing_ref+=("$FIX_BAD_KEYPOINTS_SCRIPT (fix_bad_keypoints.sh)")
  [[ -d "${JOB_KEYPOINTS_ROOT[$job_id]}" ]] || missing_ref+=("${JOB_KEYPOINTS_ROOT[$job_id]} (keypoints_root)")
  [[ -d "$UPFALL_ROOT" ]] || missing_ref+=("$UPFALL_ROOT (upfall_root)")

  case "$backend" in
    yolo)
      [[ -f "${JOB_MODEL_PATH[$job_id]}" ]] || missing_ref+=("${JOB_MODEL_PATH[$job_id]} (model_path)")
      ;;
    alphapose)
      [[ -d "$ALPHAPOSE_ROOT" ]] || missing_ref+=("$ALPHAPOSE_ROOT (alphapose_root)")
      [[ -f "${JOB_FASTPOSE_WEIGHTS[$job_id]}" ]] || missing_ref+=("${JOB_FASTPOSE_WEIGHTS[$job_id]} (fastpose_weights)")
      [[ -f "${JOB_DETECTOR_WEIGHTS[$job_id]}" ]] || missing_ref+=("${JOB_DETECTOR_WEIGHTS[$job_id]} (detector_weights)")
      ;;
    vitpose)
      [[ -f "${JOB_DETECTOR_MODEL[$job_id]}" ]] || missing_ref+=("${JOB_DETECTOR_MODEL[$job_id]} (detector_model)")
      [[ -f "${JOB_POSE_MODEL[$job_id]}" ]] || missing_ref+=("${JOB_POSE_MODEL[$job_id]} (pose_model)")
      ;;
    *)
      fail "Unsupported backend for job ${job_id}: ${backend}"
      ;;
  esac
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
  local backend="${JOB_BACKEND[$job_id]}"
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
      printf 'Backend: %s\n' "$backend"
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
  printf '%-10s | %-18s | %-64s | %-56s | %s\n' "backend" "variant" "keypoints_root" "model_info" "status"
  printf '%-10s-+-%-18s-+-%-64s-+-%-56s-+-%s\n' \
    "----------" \
    "------------------" \
    "----------------------------------------------------------------" \
    "--------------------------------------------------------" \
    "--------------------"

  for job_id in "${JOB_ORDER[@]}"; do
    backend="$(trim_field "${JOB_BACKEND[$job_id]}" 10)"
    variant="$(trim_field "${JOB_VARIANT[$job_id]}" 18)"
    keypoints_root="$(trim_field "${JOB_KEYPOINTS_ROOT[$job_id]}" 64)"
    model_info="$(trim_field "${JOB_MODEL_INFO[$job_id]}" 56)"
    status="${JOB_STATUS[$job_id]:-pending}"

    printf '%-10s | %-18s | %-64s | %-56s | %s\n' \
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
SUBJECTS=""
LOCK_SETTINGS=""
DRY_RUN=0
FORCE=0
CONTINUE_ON_ERROR=1
DATASETS_ROOT="${DEFAULT_REPO_ROOT}/Datasets"
MODELS_ROOT="${DEFAULT_REPO_ROOT}/pose_models/quantised"
UPFALL_ROOT=""
ALPHAPOSE_ROOT="${DEFAULT_REPO_ROOT}/fall_models/Prototype/pose_models/AlphaPose"
STATE_DIR="${DEFAULT_REPO_ROOT}/dataset_helpers/.fix_bad_keypoints_state"

declare -a JOB_ORDER=()
declare -A JOB_BACKEND=()
declare -A JOB_VARIANT=()
declare -A JOB_KEYPOINTS_ROOT=()
declare -A JOB_MODEL_INFO=()
declare -A JOB_STATUS=()
declare -A JOB_MESSAGE=()
declare -A JOB_MODEL_PATH=()
declare -A JOB_FASTPOSE_WEIGHTS=()
declare -A JOB_DETECTOR_WEIGHTS=()
declare -A JOB_DETECTOR_MODEL=()
declare -A JOB_POSE_MODEL=()

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
    --alphapose-root)
      require_value "$1" "${2:-}"
      ALPHAPOSE_ROOT="$2"
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
  yolo|alphapose|vitpose|all)
    ;;
  "")
    fail "--arch is required."
    ;;
  *)
    fail "Unsupported --arch: ${ARCH}. Choose from yolo, alphapose, vitpose, all."
    ;;
esac

if [[ -z "$UPFALL_ROOT" ]]; then
  UPFALL_ROOT="${DATASETS_ROOT}/UPFall"
fi

mkdir -p -- "$STATE_DIR" "${STATE_DIR}/logs"

[[ -d "$PROTOTYPE_DIR" ]] || fail "Prototype directory not found: ${PROTOTYPE_DIR}"
[[ -f "$FIX_BAD_KEYPOINTS_SCRIPT" ]] || fail "fix_bad_keypoints.sh not found: ${FIX_BAD_KEYPOINTS_SCRIPT}"

case "$ARCH" in
  yolo)
    add_yolo_jobs
    ;;
  alphapose)
    add_alphapose_jobs
    ;;
  vitpose)
    add_vitpose_jobs
    ;;
  all)
    add_yolo_jobs
    add_alphapose_jobs
    add_vitpose_jobs
    ;;
esac

if (( ${#JOB_ORDER[@]} == 0 )); then
  fail "No jobs were generated for --arch ${ARCH}."
fi

log "Starting bad keypoint batch cleanup"
log "Prototype root: ${PROTOTYPE_DIR}"
log "Arch:           ${ARCH}"
log "Datasets root:  ${DATASETS_ROOT}"
log "Models root:    ${MODELS_ROOT}"
log "UP-Fall root:   ${UPFALL_ROOT}"
log "AlphaPose root: ${ALPHAPOSE_ROOT}"
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
