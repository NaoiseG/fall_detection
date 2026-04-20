#!/usr/bin/env bash

# Benchmark resized-input YOLO pose pipelines with CNN-LSTM only.
#
# Workflow:
#   1) export dedicated FP32 TensorRT engines for the requested pose-model/imgsz pairs
#   2) benchmark each exported engine with the CNN-LSTM pipeline
#   3) resume cleanly by reusing existing exports and skipping completed runs
#
# Intended run location:
#   /home/jetson/.../fall_detection/fall_models/Prototype
#
# This script lives in inference/, but resolves the Prototype root automatically.

set -u
set -o pipefail

###############################################################################
# Configuration
###############################################################################

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BENCH_DIR="benchmarks/img_downsize"
VIDEO_PATH="../../Datasets/test_vids/activity_all.mp4"
TEMPORAL_WINDOW_SIZE=64
TEMPORAL_WINDOW_STRIDE=48

COMBINATIONS=(
  "yolo11m-pose:576"
  "yolo11l-pose:512"
  "yolo11x-pose:448"
)

CLASSIFIERS=(
  "cnnlstm"
)

# Fixed classification weight for all runs
CNNLSTM_WEIGHT="../../web_app/models/classification/cnnlstm/yolo11l-pose/cnnlstm_best.pt"

TOTAL_RUNS=$(( ${#COMBINATIONS[@]} * ${#CLASSIFIERS[@]} ))

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

pose_model_dir() {
  local pose_model="$1"
  printf '%s' "../../quantisation/models/ultralytics/${pose_model}"
}

pose_pt_for_combo() {
  local pose_model="$1"
  local base_dir
  base_dir="$(pose_model_dir "${pose_model}")"
  printf '%s/%s.pt' "${base_dir}" "${pose_model}"
}

pose_engine_for_combo() {
  local pose_model="$1"
  local imgsz="$2"
  local base_dir
  base_dir="$(pose_model_dir "${pose_model}")"
  printf '%s/%s_imgsz%s_fp32.engine' "${base_dir}" "${pose_model}" "${imgsz}"
}

combo_dest_dir() {
  local pose_model="$1"
  local imgsz="$2"
  printf '%s/%s/imgsz_%s/fp32' "${BENCH_DIR}" "${pose_model}" "${imgsz}"
}

classifier_weight_for_arch() {
  local classifier="$1"

  case "$classifier" in
    cnnlstm)    printf '%s' "${CNNLSTM_WEIGHT}" ;;
    *)
      return 1
      ;;
  esac
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
  local pose_model="$2"
  local imgsz="$3"
  local classifier="$4"
  local description="$5"
  local cmd_str="$6"
  local phase="$7"

  if [[ ! -f "$path" ]]; then
    log_failure \
      "phase=${phase} pose_model=${pose_model} imgsz=${imgsz} version=fp32 classifier=${classifier} status=missing_file missing=${description} path=${path} cmd=\"${cmd_str}\""
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
  local cls_weight="$2"
  local pose_weight="$3"
  local imgsz="$4"

  printf '%s\0' \
    python -m inference.inference_on_video \
    --video "${VIDEO_PATH}" \
    --model "${cls_weight}" \
    --yolo-weights "${pose_weight}" \
    --imgsz "${imgsz}" \
    --arch "${classifier}" \
    --device cuda \
    --half 0 \
    --T "${TEMPORAL_WINDOW_SIZE}" \
    --stride "${TEMPORAL_WINDOW_STRIDE}" \
    --max-people 10 \
    --max-det 10 \
    --warmup-frames 5 \
    --warmup-windows 0 \
    --benchmark 1 \
    --profile-out "${BENCH_DIR}" \
    --no-display 1
}

export_one_engine() {
  local pose_model="$1"
  local imgsz="$2"

  local pt_weight
  local final_engine
  local model_dir
  local base_name
  local exported_engine
  local onnx_file
  local export_name
  local cmd_str
  local rc=0

  pt_weight="$(pose_pt_for_combo "${pose_model}")"
  final_engine="$(pose_engine_for_combo "${pose_model}" "${imgsz}")"
  model_dir="$(pose_model_dir "${pose_model}")"
  base_name="$(basename "${pt_weight}" .pt)"
  exported_engine="${model_dir}/${base_name}.engine"
  onnx_file="${model_dir}/${base_name}.onnx"
  export_name="${pose_model}_imgsz${imgsz}_fp32_export"

  if [[ -f "${final_engine}" ]]; then
    printf '[export] Reusing %s imgsz=%s -> %s\n' "${pose_model}" "${imgsz}" "${final_engine}"
    log_skip \
      "phase=export pose_model=${pose_model} imgsz=${imgsz} version=fp32 status=skipped reason=engine_exists path=${final_engine}"
    return 2
  fi

  local cmd=(
    python
    -c
    "from ultralytics.cfg import entrypoint; entrypoint()"
    export
    "model=${pt_weight}"
    "format=engine"
    "imgsz=${imgsz}"
    "batch=1"
    "half=False"
    "project=${model_dir}"
    "name=${export_name}"
    "exist_ok=True"
  )
  cmd_str="$(join_cmd "${cmd[@]}")"

  require_file_or_log "${pt_weight}" "${pose_model}" "${imgsz}" "none" "pose_pt" "${cmd_str}" "export" || return 1

  rm -f -- "${exported_engine}" "${onnx_file}"

  printf '[export] Building %s imgsz=%s -> fp32 engine\n' "${pose_model}" "${imgsz}"

  "${cmd[@]}"
  rc=$?

  if [[ "${rc}" -ne 0 ]]; then
    log_failure \
      "phase=export pose_model=${pose_model} imgsz=${imgsz} version=fp32 status=command_failed exit_code=${rc} cmd=\"${cmd_str}\""
    return 1
  fi

  if [[ ! -f "${exported_engine}" ]]; then
    log_failure \
      "phase=export pose_model=${pose_model} imgsz=${imgsz} version=fp32 status=missing_exported_engine expected=${exported_engine} cmd=\"${cmd_str}\""
    return 1
  fi

  if ! mv -f -- "${exported_engine}" "${final_engine}"; then
    log_failure \
      "phase=export pose_model=${pose_model} imgsz=${imgsz} version=fp32 status=rename_failed source=${exported_engine} dest=${final_engine} cmd=\"${cmd_str}\""
    return 1
  fi

  rm -f -- "${onnx_file}"

  log_success \
    "phase=export pose_model=${pose_model} imgsz=${imgsz} version=fp32 status=ok engine=${final_engine} cmd=\"${cmd_str}\""
  return 0
}

run_one_benchmark() {
  local run_idx="$1"
  local pose_model="$2"
  local imgsz="$3"
  local classifier="$4"

  local pose_weight
  local cls_weight
  local dest_dir
  local cmd_str
  local rc=0

  pose_weight="$(pose_engine_for_combo "${pose_model}" "${imgsz}")"
  cls_weight="$(classifier_weight_for_arch "${classifier}")" || {
    log_failure \
      "phase=benchmark pose_model=${pose_model} imgsz=${imgsz} version=fp32 classifier=${classifier} status=internal_error reason=invalid_classifier_mapping"
    return 1
  }

  dest_dir="$(combo_dest_dir "${pose_model}" "${imgsz}")"
  mkdir -p "${dest_dir}"

  if combination_already_done "${dest_dir}" "${classifier}"; then
    printf '[%d/%d] Skipping %s imgsz=%s fp32 + %s (already benchmarked)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${pose_model}" "${imgsz}" "${classifier}"

    log_skip \
      "phase=benchmark pose_model=${pose_model} imgsz=${imgsz} version=fp32 classifier=${classifier} status=skipped reason=already_benchmarked dest_dir=${dest_dir}"
    return 2
  fi

  local cmd=()
  while IFS= read -r -d '' token; do
    cmd+=("$token")
  done < <(build_command "${classifier}" "${cls_weight}" "${pose_weight}" "${imgsz}")

  cmd_str="$(join_cmd "${cmd[@]}")"

  printf '[%d/%d] Running %s imgsz=%s fp32 + %s\n' \
    "${run_idx}" "${TOTAL_RUNS}" "${pose_model}" "${imgsz}" "${classifier}"

  require_file_or_log "${VIDEO_PATH}" "${pose_model}" "${imgsz}" "${classifier}" "video" "${cmd_str}" "benchmark" || return 1
  require_file_or_log "${pose_weight}" "${pose_model}" "${imgsz}" "${classifier}" "pose_weight" "${cmd_str}" "benchmark" || return 1
  require_file_or_log "${cls_weight}" "${pose_model}" "${imgsz}" "${classifier}" "classification_weight" "${cmd_str}" "benchmark" || return 1

  local before_file after_file new_run_dir
  before_file="$(mktemp)"
  after_file="$(mktemp)"

  snapshot_top_level_dirs > "${before_file}"

  "${cmd[@]}"
  rc=$?

  snapshot_top_level_dirs > "${after_file}"

  if [[ "${rc}" -ne 0 ]]; then
    log_failure \
      "phase=benchmark pose_model=${pose_model} imgsz=${imgsz} version=fp32 classifier=${classifier} status=command_failed exit_code=${rc} cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
    return 1
  fi

  if new_run_dir="$(find_new_run_dir "${before_file}" "${after_file}")"; then
    if [[ -d "${new_run_dir}" ]]; then
      local run_basename
      run_basename="$(basename "${new_run_dir}")"

      if mv "${new_run_dir}" "${dest_dir}/"; then
        log_success \
          "phase=benchmark pose_model=${pose_model} imgsz=${imgsz} version=fp32 classifier=${classifier} status=ok moved_to=${dest_dir}/${run_basename} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 0
      else
        log_failure \
          "phase=benchmark pose_model=${pose_model} imgsz=${imgsz} version=fp32 classifier=${classifier} status=move_failed source=${new_run_dir} dest=${dest_dir} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 1
      fi
    else
      log_failure \
        "phase=benchmark pose_model=${pose_model} imgsz=${imgsz} version=fp32 classifier=${classifier} status=no_new_directory_found reason=diff_returned_non_directory path=${new_run_dir} cmd=\"${cmd_str}\""
      rm -f "${before_file}" "${after_file}"
      return 1
    fi
  else
    log_failure \
      "phase=benchmark pose_model=${pose_model} imgsz=${imgsz} version=fp32 classifier=${classifier} status=no_unique_new_directory_found cmd=\"${cmd_str}\""
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

for combo in "${COMBINATIONS[@]}"; do
  IFS=':' read -r pose_model imgsz <<< "${combo}"
  mkdir -p "$(combo_dest_dir "${pose_model}" "${imgsz}")"
done

run_idx=0
success_count=0
failure_count=0
skip_count=0
export_success_count=0
export_failure_count=0
export_skip_count=0

for combo in "${COMBINATIONS[@]}"; do
  IFS=':' read -r pose_model imgsz <<< "${combo}"

  export_one_engine "${pose_model}" "${imgsz}"
  export_rc=$?

  if [[ "${export_rc}" -eq 0 ]]; then
    export_success_count=$((export_success_count + 1))
  elif [[ "${export_rc}" -eq 2 ]]; then
    export_skip_count=$((export_skip_count + 1))
  else
    export_failure_count=$((export_failure_count + 1))
  fi

  for classifier in "${CLASSIFIERS[@]}"; do
    run_idx=$((run_idx + 1))

    if [[ "${export_rc}" -eq 1 ]]; then
      printf '[%d/%d] Skipping %s imgsz=%s fp32 + %s (export failed)\n' \
        "${run_idx}" "${TOTAL_RUNS}" "${pose_model}" "${imgsz}" "${classifier}"
      log_failure \
        "phase=benchmark pose_model=${pose_model} imgsz=${imgsz} version=fp32 classifier=${classifier} status=blocked reason=export_failed"
      failure_count=$((failure_count + 1))
      continue
    fi

    run_one_benchmark "${run_idx}" "${pose_model}" "${imgsz}" "${classifier}"
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
echo "  Exported engines:     ${export_success_count}"
echo "  Reused engines:       ${export_skip_count}"
echo "  Export failures:      ${export_failure_count}"
echo "  New successful runs:  ${success_count}"
echo "  Skipped runs:         ${skip_count}"
echo "  Failed runs:          ${failure_count}"
echo "  Success log:          ${BENCH_DIR}/successful_runs.log"
echo "  Skipped log:          ${BENCH_DIR}/skipped_runs.log"
echo "  Failure log:          ${BENCH_DIR}/failed_runs.log"

exit 0
