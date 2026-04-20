#!/usr/bin/env bash

# Benchmark selected resized-input YOLO pose final pipelines.
#
# Requested runs:
#   - yolo11m fp16 imgsz 576 + stgcn
#   - yolo11l fp32 imgsz 512 + cnnlstm
#   - yolo11x fp16 imgsz 448 + cnnlstm
#   - yolo11x fp16 imgsz 448 + stgcn
#   - yolo11x pruned_80 fp16 imgsz 448 + stgcn
#
# Results are stored under:
#   benchmarks/img_downsize/final_pipelines
#
# Intended run location:
#   /home/jetson/.../fall_detection/fall_models/Prototype
#
# This script resolves the Prototype root automatically, so it can be launched
# from elsewhere too.

set -u
set -o pipefail

###############################################################################
# Configuration
###############################################################################

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

BENCH_DIR="${BENCH_DIR:-benchmarks/img_downsize/final_pipelines}"
VIDEO_PATH="${VIDEO_PATH:-../../Datasets/test_vids/activity_all.mp4}"
MODEL_ROOT="${MODEL_ROOT:-../../quantisation/models/img_downsize}"
CLASSIFICATION_ROOT="${CLASSIFICATION_ROOT:-../../web_app/models/classification}"
TEMPORAL_WINDOW_SIZE="${TEMPORAL_WINDOW_SIZE:-64}"
TEMPORAL_WINDOW_STRIDE="${TEMPORAL_WINDOW_STRIDE:-48}"

RUN_SPECS=(
  "mfp16|yolo11m-pose|fp16|576|stgcn|yolo11m-pose|yolo11m-pose_imgsz576_fp16.engine"
  "lfp32|yolo11l-pose|fp32|512|cnnlstm|yolo11l-pose|yolo11l-pose_imgsz512_fp32.engine"
  "xfp16|yolo11x-pose|fp16|448|cnnlstm|yolo11x-pose|yolo11x-pose_imgsz448_fp16.engine"
  "xfp16|yolo11x-pose|fp16|448|stgcn|yolo11x-pose|yolo11x-pose_imgsz448_fp16.engine"
  "xfp16_p80|yolo11x_pruned_80|fp16|448|stgcn|yolo11x-pose|yolo11x_pruned_80_imgsz448_fp16.engine"
)

TOTAL_RUNS="${#RUN_SPECS[@]}"

###############################################################################
# Helpers
###############################################################################

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_success() {
  local msg="$1"
  printf '[%s] %s\n' "$(timestamp)" "$msg" >> "${BENCH_DIR}/successful_runs.log"
}

log_failure() {
  local msg="$1"
  printf '[%s] %s\n' "$(timestamp)" "$msg" >> "${BENCH_DIR}/failed_runs.log"
}

log_skip() {
  local msg="$1"
  printf '[%s] %s\n' "$(timestamp)" "$msg" >> "${BENCH_DIR}/skipped_runs.log"
}

join_cmd() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v out '%s%q ' "$out" "$arg"
  done
  printf '%s' "${out% }"
}

classifier_weight_for_arch_pose() {
  local classifier="$1"
  local pose_checkpoint_tag="$2"

  case "$classifier" in
    cnnlstm) printf '%s/%s/%s/cnnlstm_best.pt' "${CLASSIFICATION_ROOT}" "${classifier}" "${pose_checkpoint_tag}" ;;
    stgcn)   printf '%s/%s/%s/stgcn_best.pt' "${CLASSIFICATION_ROOT}" "${classifier}" "${pose_checkpoint_tag}" ;;
    *)
      return 1
      ;;
  esac
}

pose_weight_path() {
  local engine_name="$1"
  printf '%s/%s' "${MODEL_ROOT}" "${engine_name}"
}

half_flag_for_precision() {
  local precision="$1"

  case "$precision" in
    fp32) printf '%s' "0" ;;
    fp16) printf '%s' "1" ;;
    *)
      return 1
      ;;
  esac
}

benchmark_dest_dir() {
  local pose_model="$1"
  local imgsz="$2"
  local precision="$3"
  local classifier="$4"

  printf '%s/%s/imgsz_%s/%s/%s' "${BENCH_DIR}" "${pose_model}" "${imgsz}" "${precision}" "${classifier}"
}

snapshot_child_dirs() {
  local root_dir="$1"
  find "${root_dir}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort
}

find_new_run_dir() {
  local before_file="$1"
  local after_file="$2"

  local diff_file
  diff_file="$(mktemp)"

  comm -13 "${before_file}" "${after_file}" > "${diff_file}"

  local count
  count="$(grep -c . "${diff_file}" || true)"

  if [[ "${count}" -eq 1 ]]; then
    cat "${diff_file}"
    rm -f "${diff_file}"
    return 0
  fi

  rm -f "${diff_file}"
  return 1
}

require_file_or_log() {
  local path="$1"
  local run_key="$2"
  local pose_model="$3"
  local precision="$4"
  local classifier="$5"
  local description="$6"
  local cmd_str="$7"

  if [[ ! -f "${path}" ]]; then
    printf 'ERROR: %s %s + %s missing %s: %s\n' \
      "${pose_model}" "${precision}" "${classifier}" "${description}" "${path}" >&2
    log_failure \
      "run_key=${run_key} pose_model=${pose_model} version=${precision} classifier=${classifier} status=missing_file missing=${description} path=${path} cmd=\"${cmd_str}\""
    return 1
  fi

  return 0
}

combination_already_done() {
  local dest_dir="$1"
  local classifier="$2"
  local run_dir

  [[ -d "${dest_dir}" ]] || return 1

  while IFS= read -r run_dir; do
    [[ -f "${run_dir}/summary.json" ]] && return 0
  done < <(find "${dest_dir}" -mindepth 1 -maxdepth 1 -type d -name "*__model_${classifier}__*" -print 2>/dev/null)

  return 1
}

run_command_with_heartbeat() {
  local run_label="$1"
  shift

  local interval_raw="${BENCHMARK_HEARTBEAT_S:-60}"
  local interval_s

  case "${interval_raw}" in
    ''|*[!0-9]*)
      interval_s=60
      ;;
    *)
      interval_s="${interval_raw}"
      ;;
  esac

  if [[ "${interval_s}" -le 0 ]]; then
    "$@"
    return $?
  fi

  "$@" &
  local cmd_pid=$!
  local start_ts now elapsed
  start_ts="$(date +%s)"

  trap '
    if [[ -n "${cmd_pid:-}" ]] && kill -0 "${cmd_pid}" 2>/dev/null; then
      printf "\nINTERRUPT: stopping %s (pid=%s)\n" "'"${run_label}"'" "${cmd_pid}" >&2
      kill -INT "${cmd_pid}" 2>/dev/null || true
      wait "${cmd_pid}" 2>/dev/null || true
    fi
    trap - INT TERM
  ' INT TERM

  while kill -0 "${cmd_pid}" 2>/dev/null; do
    sleep "${interval_s}"

    if kill -0 "${cmd_pid}" 2>/dev/null; then
      now="$(date +%s)"
      elapsed=$(( now - start_ts ))
      printf '    ... still running %s elapsed=%02dh:%02dm:%02ds\n' \
        "${run_label}" \
        "$(( elapsed / 3600 ))" \
        "$(( (elapsed % 3600) / 60 ))" \
        "$(( elapsed % 60 ))"
    fi
  done

  wait "${cmd_pid}"
  local rc=$?
  trap - INT TERM
  return "${rc}"
}

build_command() {
  local classifier="$1"
  local cls_weight="$2"
  local pose_weight="$3"
  local imgsz="$4"
  local half_flag="$5"
  local profile_out_dir="$6"

  printf '%s\0' \
    python -m inference.inference_on_video \
    --video "${VIDEO_PATH}" \
    --model "${cls_weight}" \
    --yolo-weights "${pose_weight}" \
    --imgsz "${imgsz}" \
    --arch "${classifier}" \
    --device cuda \
    --half "${half_flag}" \
    --T "${TEMPORAL_WINDOW_SIZE}" \
    --stride "${TEMPORAL_WINDOW_STRIDE}" \
    --max-people 10 \
    --max-det 10 \
    --warmup-frames 5 \
    --warmup-windows 0 \
    --benchmark 1 \
    --profile-out "${profile_out_dir}" \
    --no-display 1
}

run_one_benchmark() {
  local run_idx="$1"
  local spec="$2"

  local run_key pose_model precision imgsz classifier pose_checkpoint_tag engine_name
  IFS='|' read -r run_key pose_model precision imgsz classifier pose_checkpoint_tag engine_name <<< "${spec}"

  local pose_weight
  local cls_weight
  local half_flag
  local dest_dir
  local cmd_str
  local rc=0

  pose_weight="$(pose_weight_path "${engine_name}")"
  cls_weight="$(classifier_weight_for_arch_pose "${classifier}" "${pose_checkpoint_tag}")" || {
    printf 'ERROR: %s %s + %s invalid classifier mapping for checkpoint tag %s\n' \
      "${pose_model}" "${precision}" "${classifier}" "${pose_checkpoint_tag}" >&2
    log_failure \
      "run_key=${run_key} pose_model=${pose_model} version=${precision} classifier=${classifier} status=internal_error reason=invalid_classifier_mapping checkpoint_pose_tag=${pose_checkpoint_tag}"
    return 1
  }
  half_flag="$(half_flag_for_precision "${precision}")" || {
    printf 'ERROR: %s %s + %s invalid precision mapping\n' \
      "${pose_model}" "${precision}" "${classifier}" >&2
    log_failure \
      "run_key=${run_key} pose_model=${pose_model} version=${precision} classifier=${classifier} status=internal_error reason=invalid_precision_mapping"
    return 1
  }

  dest_dir="$(benchmark_dest_dir "${pose_model}" "${imgsz}" "${precision}" "${classifier}")"
  mkdir -p "${dest_dir}"

  if combination_already_done "${dest_dir}" "${classifier}"; then
    printf '[%d/%d] Skipping %s imgsz=%s %s + %s (already benchmarked)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${pose_model}" "${imgsz}" "${precision}" "${classifier}"
    log_skip \
      "run_key=${run_key} pose_model=${pose_model} version=${precision} classifier=${classifier} status=skipped reason=already_benchmarked dest_dir=${dest_dir}"
    return 2
  fi

  local cmd=()
  while IFS= read -r -d '' token; do
    cmd+=("${token}")
  done < <(build_command "${classifier}" "${cls_weight}" "${pose_weight}" "${imgsz}" "${half_flag}" "${dest_dir}")

  cmd_str="$(join_cmd "${cmd[@]}")"

  printf '[%d/%d] Running %s imgsz=%s %s + %s\n' \
    "${run_idx}" "${TOTAL_RUNS}" "${pose_model}" "${imgsz}" "${precision}" "${classifier}"

  require_file_or_log "${VIDEO_PATH}" "${run_key}" "${pose_model}" "${precision}" "${classifier}" "video" "${cmd_str}" || return 1
  require_file_or_log "${pose_weight}" "${run_key}" "${pose_model}" "${precision}" "${classifier}" "pose_weight" "${cmd_str}" || return 1
  require_file_or_log "${cls_weight}" "${run_key}" "${pose_model}" "${precision}" "${classifier}" "classification_weight" "${cmd_str}" || return 1

  local before_file after_file new_run_dir
  before_file="$(mktemp)"
  after_file="$(mktemp)"

  snapshot_child_dirs "${dest_dir}" > "${before_file}"

  run_command_with_heartbeat "${run_key} ${classifier}" "${cmd[@]}"
  rc=$?

  snapshot_child_dirs "${dest_dir}" > "${after_file}"

  if [[ "${rc}" -ne 0 ]]; then
    printf 'ERROR: %s imgsz=%s %s + %s command failed with exit code %s\n' \
      "${pose_model}" "${imgsz}" "${precision}" "${classifier}" "${rc}" >&2
    log_failure \
      "run_key=${run_key} pose_model=${pose_model} version=${precision} classifier=${classifier} status=command_failed exit_code=${rc} cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
    return 1
  fi

  if new_run_dir="$(find_new_run_dir "${before_file}" "${after_file}")"; then
    if [[ -d "${new_run_dir}" ]]; then
      local run_basename
      run_basename="$(basename "${new_run_dir}")"
      log_success \
        "run_key=${run_key} pose_model=${pose_model} version=${precision} classifier=${classifier} status=ok output_dir=${dest_dir}/${run_basename} cmd=\"${cmd_str}\""
      rm -f "${before_file}" "${after_file}"
      return 0
    else
      printf 'ERROR: %s imgsz=%s %s + %s returned a non-directory run artifact: %s\n' \
        "${pose_model}" "${imgsz}" "${precision}" "${classifier}" "${new_run_dir}" >&2
      log_failure \
        "run_key=${run_key} pose_model=${pose_model} version=${precision} classifier=${classifier} status=no_new_directory_found reason=diff_returned_non_directory path=${new_run_dir} cmd=\"${cmd_str}\""
      rm -f "${before_file}" "${after_file}"
      return 1
    fi
  else
    printf 'ERROR: %s imgsz=%s %s + %s finished without a unique new run directory\n' \
      "${pose_model}" "${imgsz}" "${precision}" "${classifier}" >&2
    log_failure \
      "run_key=${run_key} pose_model=${pose_model} version=${precision} classifier=${classifier} status=no_unique_new_directory_found cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
    return 1
  fi
}

###############################################################################
# Main
###############################################################################

cd "${PROJECT_DIR}" || {
  echo "ERROR: Could not cd to ${PROJECT_DIR}" >&2
  exit 1
}

mkdir -p "${BENCH_DIR}"

touch "${BENCH_DIR}/successful_runs.log"
: > "${BENCH_DIR}/failed_runs.log"
: > "${BENCH_DIR}/skipped_runs.log"

for spec in "${RUN_SPECS[@]}"; do
  IFS='|' read -r _run_key pose_model precision imgsz classifier _pose_checkpoint_tag _engine_name <<< "${spec}"
  mkdir -p "$(benchmark_dest_dir "${pose_model}" "${imgsz}" "${precision}" "${classifier}")"
done

echo "Benchmark duration per run: ${BENCHMARK_DURATION_S:-600}s"
echo "Benchmark heartbeat:        ${BENCHMARK_HEARTBEAT_S:-60}s"
echo "Benchmark root:             ${BENCH_DIR}"
echo "Model root:                 ${MODEL_ROOT}"
echo "Classification root:        ${CLASSIFICATION_ROOT}"

run_idx=0
success_count=0
failure_count=0
skip_count=0

for spec in "${RUN_SPECS[@]}"; do
  run_idx=$((run_idx + 1))

  run_one_benchmark "${run_idx}" "${spec}"
  rc=$?

  if [[ "${rc}" -eq 0 ]]; then
    success_count=$((success_count + 1))
  elif [[ "${rc}" -eq 2 ]]; then
    skip_count=$((skip_count + 1))
  else
    failure_count=$((failure_count + 1))
  fi
done

echo
echo "Benchmarking complete."
echo "  New successful runs: ${success_count}"
echo "  Skipped runs:        ${skip_count}"
echo "  Failed runs:         ${failure_count}"
echo "  Success log:         ${BENCH_DIR}/successful_runs.log"
echo "  Skipped log:         ${BENCH_DIR}/skipped_runs.log"
echo "  Failure log:         ${BENCH_DIR}/failed_runs.log"

exit 0
