#!/usr/bin/env bash

# Benchmark all combinations of:
#   5 pruned YOLO pose model sizes x 2 prune variants x 3 classifiers = 30 runs
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
PRUNED_MODEL_ROOT="/home/jetson/NaoiseG/fall_detection/pruning/pruned_models"
VIDEO_PATH="../../Datasets/test_vids/sitting.mp4"
MOTIONBERT_CONFIG="../../web_app/models/classification/MotionBERT/configs/action/MB_ft_UPFall_xsub.yaml"

POSE_MODELS=(
  "yolo11n"
  "yolo11s"
  "yolo11m"
  "yolo11l"
  "yolo11x"
)

VERSIONS=(
  "pruned_80"
  "pruned_90"
)

CLASSIFIERS=(
  "cnnlstm"
  "stgcn"
  "motionbert"
)

# Fixed classification weights for ALL runs
CNNLSTM_WEIGHT="../../web_app/models/classification/cnnlstm/yolo11l-pose/cnnlstm_best.pt"
STGCN_WEIGHT="../../web_app/models/classification/stgcn/yolo11l-pose/stgcn_best.pt"
MOTIONBERT_WEIGHT="../../web_app/models/classification/MotionBERT/yolo11l-pose/checkpoint/action/FT_MB_release_MB_ft_UPFall_xsub/best_epoch.bin"

TOTAL_RUNS=$(( ${#POSE_MODELS[@]} * ${#VERSIONS[@]} * ${#CLASSIFIERS[@]} ))

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
  # Print a shell-escaped command line
  local out=""
  local arg
  for arg in "$@"; do
    printf -v out '%s%q ' "$out" "$arg"
  done
  printf '%s' "${out% }"
}

classifier_weight_for_arch() {
  local classifier="$1"

  case "$classifier" in
    cnnlstm)    printf '%s' "${CNNLSTM_WEIGHT}" ;;
    stgcn)      printf '%s' "${STGCN_WEIGHT}" ;;
    motionbert) printf '%s' "${MOTIONBERT_WEIGHT}" ;;
    *)
      return 1
      ;;
  esac
}

half_flag_for_version() {
  local version="$1"

  case "$version" in
    pruned_80|pruned_90) printf '%s' "0" ;;
    *)
      return 1
      ;;
  esac
}

pruned_model_folder_name_for_variant() {
  local pose_model="$1"
  local version="$2"
  local flops_tag

  case "$version" in
    pruned_80) flops_tag="flops80p" ;;
    pruned_90) flops_tag="flops90p" ;;
    *)
      return 1
      ;;
  esac

  printf 'pruned_pose_%s_pose_%s_trainfrac25p' "${pose_model}" "${flops_tag}"
}

pruned_model_dir_for_variant() {
  local pose_model="$1"
  local version="$2"
  local folder_name

  folder_name="$(pruned_model_folder_name_for_variant "${pose_model}" "${version}")" || return 1
  printf '%s/%s' "${PRUNED_MODEL_ROOT}" "${folder_name}"
}

pruned_model_engine_path_for_dir() {
  local model_dir="$1"
  local folder_name

  folder_name="$(basename "${model_dir}")"
  printf '%s/%s_fp32.engine' "${model_dir}" "${folder_name}"
}

resolve_pruned_pose_weight() {
  local model_dir="$1"
  local engine_path
  local best_path

  engine_path="$(pruned_model_engine_path_for_dir "${model_dir}")"
  best_path="${model_dir}/best.pt"

  if [[ -f "${engine_path}" ]]; then
    printf '%s' "${engine_path}"
    return 0
  fi

  if [[ -f "${best_path}" ]]; then
    printf '%s' "${best_path}"
    return 0
  fi

  return 1
}

benchmark_dest_dir_for_variant() {
  local pose_model="$1"
  local version="$2"

  printf '%s/pruned_models/%s/%s' "${BENCH_DIR}" "${pose_model}" "${version}"
}

snapshot_top_level_dirs() {
  # Print absolute paths of immediate subdirectories inside benchmarks, sorted.
  # This includes both structured destination dirs and run dirs.
  find "${BENCH_DIR}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort
}

find_new_run_dir() {
  # Usage: find_new_run_dir <before_file> <after_file>
  # Returns:
  #   0 + prints directory path if exactly one new top-level dir appeared
  #   1 otherwise
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
  local pose_model="$2"
  local version="$3"
  local classifier="$4"
  local description="$5"
  local cmd_str="$6"

  if [[ ! -f "$path" ]]; then
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=missing_file missing=${description} path=${path} cmd=\"${cmd_str}\""
    return 1
  fi

  return 0
}

combination_already_done() {
  local dest_dir="$1"
  local run_model_tag="$2"

  [[ -d "${dest_dir}" ]] || return 1

  find "${dest_dir}" -mindepth 1 -maxdepth 1 -type d -name "*__model_${run_model_tag}__*" -print -quit 2>/dev/null | grep -q .
}

build_command() {
  local classifier="$1"
  local version="$2"
  local cls_weight="$3"
  local pose_weight="$4"
  local half_flag

  half_flag="$(half_flag_for_version "${version}")" || return 1

  if [[ "$classifier" == "motionbert" ]]; then
    printf '%s\0' \
      python -m inference.infer_motionbert_video \
      --video "${VIDEO_PATH}" \
      --model "${cls_weight}" \
      --config "${MOTIONBERT_CONFIG}" \
      --yolo-weights "${pose_weight}" \
      --device cuda \
      --half "${half_flag}" \
      --max-people 10 \
      --max-det 10 \
      --warmup-frames 0 \
      --warmup-windows 0 \
      --benchmark 1 \
      --profile-out "${BENCH_DIR}" \
      --no-display 1 \
      --out-csv "" \
      --out-pkl ""
  else
    printf '%s\0' \
      python -m inference.inference_on_video \
      --video "${VIDEO_PATH}" \
      --model "${cls_weight}" \
      --yolo-weights "${pose_weight}" \
      --arch "${classifier}" \
      --device cuda \
      --half "${half_flag}" \
      --max-people 10 \
      --max-det 10 \
      --warmup-frames 0 \
      --warmup-windows 0 \
      --benchmark 1 \
      --profile-out "${BENCH_DIR}" \
      --no-display 1
  fi
}

run_one_benchmark() {
  local run_idx="$1"
  local pose_model="$2"
  local version="$3"
  local classifier="$4"

  local pruned_model_dir
  local pruned_engine_path
  local pose_weight
  local cls_weight
  local dest_dir
  local cmd_str
  local rc=0

  pruned_model_dir="$(pruned_model_dir_for_variant "${pose_model}" "${version}")" || {
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=internal_error reason=invalid_pruned_model_mapping"
    return 1
  }

  pruned_engine_path="$(pruned_model_engine_path_for_dir "${pruned_model_dir}")"

  pose_weight="$(resolve_pruned_pose_weight "${pruned_model_dir}")" || {
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=missing_source missing=pruned_pose_weight dir=${pruned_model_dir} expected_engine=${pruned_engine_path} expected_pt=${pruned_model_dir}/best.pt"
    return 1
  }

  cls_weight="$(classifier_weight_for_arch "${classifier}")" || {
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=internal_error reason=invalid_classifier_mapping"
    return 1
  }

  dest_dir="$(benchmark_dest_dir_for_variant "${pose_model}" "${version}")"
  mkdir -p "${dest_dir}"

  if combination_already_done "${dest_dir}" "${classifier}"; then
    printf '[%d/%d] Skipping %s %s + %s (already benchmarked)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${pose_model}" "${version}" "${classifier}"

    log_skip \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=skipped reason=already_benchmarked dest_dir=${dest_dir}"
    return 2
  fi

  local cmd=()
  while IFS= read -r -d '' token; do
    cmd+=("$token")
  done < <(build_command "${classifier}" "${version}" "${cls_weight}" "${pose_weight}")

  cmd_str="$(join_cmd "${cmd[@]}")"

  printf '[%d/%d] Running %s %s + %s\n' "${run_idx}" "${TOTAL_RUNS}" "${pose_model}" "${version}" "${classifier}"

  # Pre-flight checks
  require_file_or_log "${VIDEO_PATH}" "${pose_model}" "${version}" "${classifier}" "video" "${cmd_str}" || return 1
  require_file_or_log "${pose_weight}" "${pose_model}" "${version}" "${classifier}" "pose_weight" "${cmd_str}" || return 1
  require_file_or_log "${cls_weight}" "${pose_model}" "${version}" "${classifier}" "classification_weight" "${cmd_str}" || return 1

  if [[ "${classifier}" == "motionbert" ]]; then
    require_file_or_log "${MOTIONBERT_CONFIG}" "${pose_model}" "${version}" "${classifier}" "motionbert_config" "${cmd_str}" || return 1
  fi

  local before_file after_file new_run_dir
  before_file="$(mktemp)"
  after_file="$(mktemp)"

  snapshot_top_level_dirs > "${before_file}"

  "${cmd[@]}"
  rc=$?

  snapshot_top_level_dirs > "${after_file}"

  if [[ "${rc}" -ne 0 ]]; then
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=command_failed exit_code=${rc} cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
    return 1
  fi

  if new_run_dir="$(find_new_run_dir "${before_file}" "${after_file}")"; then
    if [[ -d "${new_run_dir}" ]]; then
      local run_basename
      run_basename="$(basename "${new_run_dir}")"

      if mv "${new_run_dir}" "${dest_dir}/"; then
        log_success \
          "pose_model=${pose_model} version=${version} classifier=${classifier} status=ok moved_to=${dest_dir}/${run_basename} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 0
      else
        log_failure \
          "pose_model=${pose_model} version=${version} classifier=${classifier} status=move_failed source=${new_run_dir} dest=${dest_dir} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 1
      fi
    else
      log_failure \
        "pose_model=${pose_model} version=${version} classifier=${classifier} status=no_new_directory_found reason=diff_returned_non_directory path=${new_run_dir} cmd=\"${cmd_str}\""
      rm -f "${before_file}" "${after_file}"
      return 1
    fi
  else
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=no_unique_new_directory_found cmd=\"${cmd_str}\""
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

# Keep existing benchmark outputs. Only ensure the directory exists.
mkdir -p "${BENCH_DIR}"

# Logging behavior:
# - successful_runs.log is preserved and appended to
# - failed_runs.log is reset on each invocation
# - skipped_runs.log is reset on each invocation
touch "${BENCH_DIR}/successful_runs.log"
: > "${BENCH_DIR}/failed_runs.log"
: > "${BENCH_DIR}/skipped_runs.log"

# Pre-create expected destination structure
for pose_model in "${POSE_MODELS[@]}"; do
  for version in "${VERSIONS[@]}"; do
    mkdir -p "$(benchmark_dest_dir_for_variant "${pose_model}" "${version}")"
  done
done

run_idx=0
success_count=0
failure_count=0
skip_count=0

for pose_model in "${POSE_MODELS[@]}"; do
  for version in "${VERSIONS[@]}"; do
    for classifier in "${CLASSIFIERS[@]}"; do
      run_idx=$((run_idx + 1))

      run_one_benchmark "${run_idx}" "${pose_model}" "${version}" "${classifier}"
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
