#!/usr/bin/env bash

# Benchmark selected fully pruned YOLO pose TensorRT engines with CNN-LSTM only.
#
# Run this script from:
#   /home/jetson/NaoiseG/fall_detection/fall_models/Prototype
#
# Or let it cd there automatically below.

set -u
set -o pipefail

###############################################################################
# Configuration
###############################################################################

PROJECT_DIR="/home/jetson/NaoiseG/fall_detection/fall_models/Prototype"
BENCH_DIR="benchmarks"
FULL_PRUNED_MODEL_ROOT="/home/jetson/NaoiseG/fall_detection/pruning/pruned_models/full_pruned"
VIDEO_PATH="../../Datasets/test_vids/sitting.mp4"
CLASSIFIER="cnnlstm"
CNNLSTM_WEIGHT="../../web_app/models/classification/cnnlstm/yolo11l-pose/cnnlstm_best.pt"

FULL_PRUNED_MODELS=(
  "yolo11l_pruned_80"
  "yolo11l_pruned_90"
  "yolo11m_pruned_80"
  "yolo11m_pruned_90"
  "yolo11s_pruned_90"
  "yolo11x_pruned_70"
  "yolo11x_pruned_80"
)

ENGINE_VERSIONS=(
  "fp32"
  "fp16"
  "int8"
)

TOTAL_RUNS=$(( ${#FULL_PRUNED_MODELS[@]} * ${#ENGINE_VERSIONS[@]} ))

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

half_flag_for_version() {
  local version="$1"

  case "$version" in
    fp32) printf '%s' "0" ;;
    fp16|int8) printf '%s' "1" ;;
    *)
      return 1
      ;;
  esac
}

full_pruned_model_dir() {
  local model_name="$1"
  printf '%s/%s' "${FULL_PRUNED_MODEL_ROOT}" "${model_name}"
}

full_pruned_weights_dir() {
  local model_name="$1"
  printf '%s/weights' "$(full_pruned_model_dir "${model_name}")"
}

expected_engine_path_for_version() {
  local model_name="$1"
  local version="$2"
  printf '%s/%s_%s.engine' "$(full_pruned_weights_dir "${model_name}")" "${model_name}" "${version}"
}

resolve_engine_for_version() {
  local model_name="$1"
  local version="$2"
  local weights_dir
  local expected_path
  local matches=()
  local candidate

  weights_dir="$(full_pruned_weights_dir "${model_name}")"
  expected_path="$(expected_engine_path_for_version "${model_name}" "${version}")"

  if [[ -f "${expected_path}" ]]; then
    printf '%s' "${expected_path}"
    return 0
  fi

  shopt -s nullglob
  for candidate in "${weights_dir}"/*_"${version}".engine; do
    [[ -f "${candidate}" ]] || continue
    matches+=("${candidate}")
  done
  shopt -u nullglob

  if [[ "${#matches[@]}" -eq 1 ]]; then
    printf '%s' "${matches[0]}"
    return 0
  fi

  if [[ "${#matches[@]}" -gt 1 ]]; then
    return 2
  fi

  return 1
}

benchmark_dest_dir() {
  local model_name="$1"
  local version="$2"
  printf '%s/pruned_models/full_pruned/%s/%s' "${BENCH_DIR}" "${model_name}" "${version}"
}

snapshot_top_level_dirs() {
  find "${BENCH_DIR}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort
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
  local model_name="$2"
  local version="$3"
  local description="$4"
  local cmd_str="$5"

  if [[ ! -f "${path}" ]]; then
    log_failure \
      "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=missing_file missing=${description} path=${path} cmd=\"${cmd_str}\""
    return 1
  fi

  return 0
}

combination_already_done() {
  local dest_dir="$1"

  [[ -d "${dest_dir}" ]] || return 1

  find "${dest_dir}" -mindepth 1 -maxdepth 1 -type d -name "*__model_${CLASSIFIER}__*" -print -quit 2>/dev/null | grep -q .
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
}

build_command() {
  local version="$1"
  local pose_weight="$2"
  local half_flag

  half_flag="$(half_flag_for_version "${version}")" || return 1

  printf '%s\0' \
    python -m inference.inference_on_video \
    --video "${VIDEO_PATH}" \
    --model "${CNNLSTM_WEIGHT}" \
    --yolo-weights "${pose_weight}" \
    --arch "${CLASSIFIER}" \
    --device cuda \
    --half "${half_flag}" \
    --max-people 10 \
    --max-det 10 \
    --warmup-frames 0 \
    --warmup-windows 0 \
    --benchmark 1 \
    --profile-out "${BENCH_DIR}" \
    --no-display 1
}

run_one_benchmark() {
  local run_idx="$1"
  local model_name="$2"
  local version="$3"

  local model_dir
  local weights_dir
  local pose_weight
  local expected_pose_weight
  local dest_dir
  local cmd_str
  local rc=0

  model_dir="$(full_pruned_model_dir "${model_name}")"
  weights_dir="$(full_pruned_weights_dir "${model_name}")"
  expected_pose_weight="$(expected_engine_path_for_version "${model_name}" "${version}")"
  dest_dir="$(benchmark_dest_dir "${model_name}" "${version}")"

  if [[ ! -d "${model_dir}" ]]; then
    printf '[%d/%d] Skipping %s %s (model directory missing)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${model_name}" "${version}"
    log_skip \
      "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=skipped reason=model_directory_missing path=${model_dir}"
    return 2
  fi

  if [[ ! -d "${weights_dir}" ]]; then
    printf '[%d/%d] Skipping %s %s (weights directory missing)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${model_name}" "${version}"
    log_skip \
      "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=skipped reason=weights_directory_missing path=${weights_dir}"
    return 2
  fi

  pose_weight="$(resolve_engine_for_version "${model_name}" "${version}")"
  rc=$?

  if [[ "${rc}" -eq 1 ]]; then
    printf '[%d/%d] Skipping %s %s (missing engine: %s)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${model_name}" "${version}" "${expected_pose_weight}"
    log_skip \
      "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=skipped reason=missing_engine expected_path=${expected_pose_weight}"
    return 2
  fi

  if [[ "${rc}" -eq 2 ]]; then
    printf '[%d/%d] Skipping %s %s (multiple %s engines found in %s)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${model_name}" "${version}" "${version}" "${weights_dir}"
    log_skip \
      "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=skipped reason=ambiguous_engine_candidates path=${weights_dir}"
    return 2
  fi

  mkdir -p "${dest_dir}"

  if combination_already_done "${dest_dir}"; then
    printf '[%d/%d] Skipping %s %s + %s (already benchmarked)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${model_name}" "${version}" "${CLASSIFIER}"
    log_skip \
      "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=skipped reason=already_benchmarked dest_dir=${dest_dir}"
    return 2
  fi

  local cmd=()
  while IFS= read -r -d '' token; do
    cmd+=("$token")
  done < <(build_command "${version}" "${pose_weight}")

  cmd_str="$(join_cmd "${cmd[@]}")"

  printf '[%d/%d] Running %s %s + %s\n' \
    "${run_idx}" "${TOTAL_RUNS}" "${model_name}" "${version}" "${CLASSIFIER}"

  require_file_or_log "${VIDEO_PATH}" "${model_name}" "${version}" "video" "${cmd_str}" || return 1
  require_file_or_log "${CNNLSTM_WEIGHT}" "${model_name}" "${version}" "classification_weight" "${cmd_str}" || return 1
  require_file_or_log "${pose_weight}" "${model_name}" "${version}" "pose_weight" "${cmd_str}" || return 1

  local before_file after_file new_run_dir
  before_file="$(mktemp)"
  after_file="$(mktemp)"

  snapshot_top_level_dirs > "${before_file}"

  run_command_with_heartbeat "${model_name} ${version} + ${CLASSIFIER}" "${cmd[@]}"
  rc=$?

  snapshot_top_level_dirs > "${after_file}"

  if [[ "${rc}" -ne 0 ]]; then
    log_failure \
      "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=command_failed exit_code=${rc} cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
    return 1
  fi

  if new_run_dir="$(find_new_run_dir "${before_file}" "${after_file}")"; then
    if [[ -d "${new_run_dir}" ]]; then
      local run_basename
      run_basename="$(basename "${new_run_dir}")"

      if mv "${new_run_dir}" "${dest_dir}/"; then
        log_success \
          "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=ok moved_to=${dest_dir}/${run_basename} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 0
      else
        log_failure \
          "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=move_failed source=${new_run_dir} dest=${dest_dir} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 1
      fi
    else
      log_failure \
        "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=no_new_directory_found reason=diff_returned_non_directory path=${new_run_dir} cmd=\"${cmd_str}\""
      rm -f "${before_file}" "${after_file}"
      return 1
    fi
  else
    log_failure \
      "pose_model=${model_name} version=${version} classifier=${CLASSIFIER} status=no_unique_new_directory_found cmd=\"${cmd_str}\""
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

for model_name in "${FULL_PRUNED_MODELS[@]}"; do
  for version in "${ENGINE_VERSIONS[@]}"; do
    mkdir -p "$(benchmark_dest_dir "${model_name}" "${version}")"
  done
done

if [[ ! -d "${FULL_PRUNED_MODEL_ROOT}" ]]; then
  echo "WARNING: FULL_PRUNED_MODEL_ROOT not found: ${FULL_PRUNED_MODEL_ROOT}" >&2
fi

echo "Benchmark duration per run: ${BENCHMARK_DURATION_S:-600}s"
echo "Benchmark heartbeat:        ${BENCHMARK_HEARTBEAT_S:-60}s"
echo "Classifier:                 ${CLASSIFIER}"

run_idx=0
success_count=0
failure_count=0
skip_count=0

for model_name in "${FULL_PRUNED_MODELS[@]}"; do
  for version in "${ENGINE_VERSIONS[@]}"; do
    run_idx=$((run_idx + 1))

    run_one_benchmark "${run_idx}" "${model_name}" "${version}"
    rc=$?

    if [[ "${rc}" -eq 0 ]]; then
      success_count=$((success_count + 1))
    elif [[ "${rc}" -eq 2 ]]; then
      skip_count=$((skip_count + 1))
    else
      failure_count=$((failure_count + 1))
    fi
  done
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
