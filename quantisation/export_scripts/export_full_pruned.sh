#!/usr/bin/env bash
set -uo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
QUANTISATION_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd -- "${QUANTISATION_DIR}/.." && pwd -P)"
DEFAULT_ROOT="/home/jetson/NaoiseG/fall_detection/pruning/pruned_models/full_pruned"
DEFAULT_CALIB_DIR="${QUANTISATION_DIR}/calibration_dataset_upfall"
DEFAULT_VENV_PYTHON="${QUANTISATION_DIR}/venvs/yolo_export/bin/python"
DEFAULT_PRUNING_VENV_PYTHON="${REPO_ROOT}/pruning/pruning_venv/bin/python"
DEFAULT_NORMALIZE_HELPER="${SCRIPT_DIR}/normalize_pruned_checkpoint.py"
DEFAULT_MODELOPT_ROOT="${REPO_ROOT}/pruning/Model-Optimizer"
DEFAULT_PRUNING_ULTRALYTICS_ROOT="${REPO_ROOT}/pruning/yolov11-prune"
BACKUP_VAL_BASENAME="val.__backup_export_pruned_engines__"

usage() {
  cat <<EOF
Usage: $(basename "$0") [ROOT_DIR] [CALIB_DIR]

Recursively finds pruned checkpoints at:
  ROOT_DIR/*/weights/best.pt

For each checkpoint, the script:
  1) normalizes the pruned checkpoint into an export-ready Ultralytics checkpoint
  2) exports FP32, FP16, and INT8 TensorRT engines
  3) renames the outputs using the model directory name

Example input:
  ${DEFAULT_ROOT}/yolo11l_pruned_80/weights/best.pt

Example outputs:
  ${DEFAULT_ROOT}/yolo11l_pruned_80/weights/yolo11l_pruned_80_fp32.engine
  ${DEFAULT_ROOT}/yolo11l_pruned_80/weights/yolo11l_pruned_80_fp16.engine
  ${DEFAULT_ROOT}/yolo11l_pruned_80/weights/yolo11l_pruned_80_int8.engine
  ${DEFAULT_ROOT}/yolo11l_pruned_80/weights/best_export_ready.pt

Defaults:
  ROOT_DIR  = ${DEFAULT_ROOT}
  CALIB_DIR = ${DEFAULT_CALIB_DIR}

Environment overrides:
  PYTHON_BIN           Python used for export, defaults to the export venv when available
  NORMALIZE_PYTHON     Python used for normalization, defaults to the pruning venv when available
  NORMALIZE_HELPER     Path to normalize_pruned_checkpoint.py
  MODELOPT_ROOT        Path to pruning/Model-Optimizer
  PRUNING_ULTRALYTICS_ROOT
                       Path to pruning/yolov11-prune
EOF
}

cleanup_export_artifacts() {
  local weights_dir="$1"
  local base_name="$2"
  local export_name="$3"

  rm -f -- "${weights_dir}/${base_name}.engine" "${weights_dir}/${base_name}.onnx"

  if [[ -e "${weights_dir}/${export_name}" ]]; then
    rm -rf -- "${weights_dir:?}/${export_name}"
  fi
}

record_failure() {
  local model_name="$1"
  local message="$2"

  FAILURE_MODELS+=("${model_name}")
  FAILURE_MESSAGES+=("${message}")
  printf 'ERROR: %s: %s\n' "${model_name}" "${message}" >&2
}

restore_calibration_labels() {
  if [[ -n "${BACKUP_VAL_DIR:-}" && -d "${BACKUP_VAL_DIR}" ]]; then
    rm -rf -- "${MAIN_VAL_DIR}"
    mv -- "${BACKUP_VAL_DIR}" "${MAIN_VAL_DIR}"
  fi
}

infer_calibration_family() {
  local model_name="$1"

  if [[ "${model_name}" =~ (yolo11[nsmlx]) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi

  return 1
}

prepare_int8_labels() {
  local model_name="$1"
  local model_family=""
  local model_val_dir=""

  if ! model_family="$(infer_calibration_family "${model_name}")"; then
    printf 'Could not infer calibration family from model name: %s\n' "${model_name}" >&2
    return 1
  fi

  model_val_dir="${MAIN_LABELS_DIR}/${model_family}/labels/val"
  if [[ ! -d "${model_val_dir}" ]]; then
    printf 'Model-specific validation labels not found: %s\n' "${model_val_dir}" >&2
    return 1
  fi

  rm -rf -- "${MAIN_VAL_DIR}"
  cp -a -- "${model_val_dir}" "${MAIN_VAL_DIR}"
  rm -f -- "${VAL_CACHE_FILE}"

  return 0
}

normalize_checkpoint() {
  local pt_path="$1"
  local normalized_pt="$2"

  rm -f -- "${normalized_pt}"

  "${NORMALIZE_PYTHON}" "${NORMALIZE_HELPER}" \
    --input "${pt_path}" \
    --output "${normalized_pt}" \
    --modelopt-root "${MODELOPT_ROOT}" \
    --ultralytics-root "${PRUNING_ULTRALYTICS_ROOT}" \
    --overwrite
}

export_engine() {
  local weights_dir="$1"
  local normalized_base="$2"
  local export_name="$3"
  shift 3

  cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${export_name}"

  (
    cd -- "${weights_dir}" &&
    "${PYTHON_BIN}" -c 'from ultralytics.cfg import entrypoint; entrypoint()' export \
      model="./${normalized_base}.pt" \
      format=engine \
      project=. \
      name="${export_name}" \
      exist_ok=True \
      "$@"
  )
}

remove_trt_caches() {
  local weights_dir="$1"

  find "${weights_dir}" -maxdepth 1 -type f \
    \( -iname '*.cache' -o -iname 'calibration.cache' -o -iname 'calib.cache' \) \
    -delete >/dev/null 2>&1 || true
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  printf 'ERROR: Expected zero, one, or two positional arguments.\n\n' >&2
  usage >&2
  exit 2
fi

ROOT_DIR="${1:-${DEFAULT_ROOT}}"
CALIB_DIR="${2:-${DEFAULT_CALIB_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-}"
NORMALIZE_PYTHON="${NORMALIZE_PYTHON:-}"
NORMALIZE_HELPER="${NORMALIZE_HELPER:-${DEFAULT_NORMALIZE_HELPER}}"
MODELOPT_ROOT="${MODELOPT_ROOT:-${DEFAULT_MODELOPT_ROOT}}"
PRUNING_ULTRALYTICS_ROOT="${PRUNING_ULTRALYTICS_ROOT:-${DEFAULT_PRUNING_ULTRALYTICS_ROOT}}"

if [[ ! -d "${ROOT_DIR}" ]]; then
  printf 'ERROR: ROOT_DIR not found: %s\n' "${ROOT_DIR}" >&2
  exit 1
fi

if [[ ! -d "${CALIB_DIR}" ]]; then
  printf 'ERROR: CALIB_DIR not found: %s\n' "${CALIB_DIR}" >&2
  exit 1
fi

ROOT_DIR="$(cd -- "${ROOT_DIR}" && pwd -P)"
CALIB_DIR="$(cd -- "${CALIB_DIR}" && pwd -P)"

DATA_YAML="${CALIB_DIR}/data.yaml"
MAIN_LABELS_DIR="${CALIB_DIR}/labels"
MAIN_VAL_DIR="${MAIN_LABELS_DIR}/val"
VAL_CACHE_FILE="${MAIN_LABELS_DIR}/val.cache"
BACKUP_VAL_DIR="${MAIN_LABELS_DIR}/${BACKUP_VAL_BASENAME}"

if [[ ! -f "${DATA_YAML}" ]]; then
  printf 'ERROR: data.yaml not found at: %s\n' "${DATA_YAML}" >&2
  exit 1
fi

if [[ ! -d "${MAIN_LABELS_DIR}" ]]; then
  printf 'ERROR: labels directory not found at: %s\n' "${MAIN_LABELS_DIR}" >&2
  exit 1
fi

if [[ ! -d "${MAIN_VAL_DIR}" ]]; then
  printf 'ERROR: main val directory not found at: %s\n' "${MAIN_VAL_DIR}" >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "${DEFAULT_VENV_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_VENV_PYTHON}"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  printf 'ERROR: Could not find a usable Python interpreter.\n' >&2
  printf 'Tried active venv, %s, python3, and python.\n' "${DEFAULT_VENV_PYTHON}" >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import ultralytics' >/dev/null 2>&1; then
  printf 'ERROR: Selected Python cannot import ultralytics: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -z "${NORMALIZE_PYTHON}" ]]; then
  if [[ -x "${DEFAULT_PRUNING_VENV_PYTHON}" ]]; then
    NORMALIZE_PYTHON="${DEFAULT_PRUNING_VENV_PYTHON}"
  else
    NORMALIZE_PYTHON="${PYTHON_BIN}"
  fi
fi

if [[ ! -x "${NORMALIZE_PYTHON}" ]]; then
  printf 'ERROR: Could not find a usable normalization Python interpreter: %s\n' "${NORMALIZE_PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${NORMALIZE_HELPER}" ]]; then
  printf 'ERROR: Normalization helper not found: %s\n' "${NORMALIZE_HELPER}" >&2
  exit 1
fi

rm -rf -- "${BACKUP_VAL_DIR}"
cp -a -- "${MAIN_VAL_DIR}" "${BACKUP_VAL_DIR}"
trap restore_calibration_labels EXIT

declare -a CHECKPOINTS=()
declare -a FAILURE_MODELS=()
declare -a FAILURE_MESSAGES=()

while IFS= read -r -d '' pt_path; do
  CHECKPOINTS+=("${pt_path}")
done < <(find "${ROOT_DIR}" -type f -path '*/weights/best.pt' -print0)

TOTAL_MODELS="${#CHECKPOINTS[@]}"
READY_COUNT=0
FAILED_COUNT=0
SKIPPED_COUNT=0

printf 'Root directory: %s\n' "${ROOT_DIR}"
printf 'Calibration dir: %s\n' "${CALIB_DIR}"
printf 'Python: %s\n' "${PYTHON_BIN}"
printf 'Normalize python: %s\n' "${NORMALIZE_PYTHON}"
printf 'Normalize helper: %s\n' "${NORMALIZE_HELPER}"
printf 'Pruned checkpoints found: %d\n' "${TOTAL_MODELS}"

if [[ "${NORMALIZE_PYTHON}" == "${PYTHON_BIN}" ]]; then
  printf 'Warning: normalization is using the export Python. Set NORMALIZE_PYTHON to a ModelOpt-capable env if needed.\n'
fi

if [[ "${TOTAL_MODELS}" -eq 0 ]]; then
  printf 'ERROR: No pruned checkpoints found under %s matching */weights/best.pt\n' "${ROOT_DIR}" >&2
  exit 1
fi

INDEX=0
for pt_path in "${CHECKPOINTS[@]}"; do
  INDEX=$((INDEX + 1))

  weights_dir="$(dirname -- "${pt_path}")"
  model_dir="$(dirname -- "${weights_dir}")"
  model_name="$(basename -- "${model_dir}")"
  normalized_pt="${weights_dir}/best_export_ready.pt"
  normalized_base="best_export_ready"
  fp32_engine="${weights_dir}/${model_name}_fp32.engine"
  fp16_engine="${weights_dir}/${model_name}_fp16.engine"
  int8_engine="${weights_dir}/${model_name}_int8.engine"
  model_failed=0
  missing_exports=0

  printf '\n==> [%d/%d] %s\n' "${INDEX}" "${TOTAL_MODELS}" "${model_name}"
  printf '    Input:  %s\n' "${pt_path}"
  printf '    Output: %s\n' "${weights_dir}"

  for final_engine in "${fp32_engine}" "${fp16_engine}" "${int8_engine}"; do
    if [[ ! -f "${final_engine}" ]]; then
      missing_exports=1
      break
    fi
  done

  if [[ "${missing_exports}" -eq 0 ]]; then
    printf '    Skipping: all target engines already exist.\n'
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    READY_COUNT=$((READY_COUNT + 1))
    continue
  fi

  printf '    Normalized checkpoint: %s\n' "${normalized_pt}"
  if ! normalize_checkpoint "${pt_path}" "${normalized_pt}"; then
    cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_normalize_cleanup"
    rm -f -- "${normalized_pt}"
    record_failure "${model_name}" "checkpoint normalization failed"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if [[ ! -f "${normalized_pt}" ]]; then
    record_failure "${model_name}" "normalized checkpoint missing after normalization: ${normalized_pt}"
    rm -f -- "${normalized_pt}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if [[ -f "${fp32_engine}" ]]; then
    printf '    FP32: exists, skipping.\n'
  else
    printf '    FP32: exporting.\n'
    if ! export_engine "${weights_dir}" "${normalized_base}" "${model_name}_fp32_export"; then
      cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_fp32_export"
      record_failure "${model_name}" "FP32 export failed"
      model_failed=1
    elif [[ ! -f "${weights_dir}/${normalized_base}.engine" ]]; then
      cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_fp32_export"
      record_failure "${model_name}" "FP32 export did not produce ${weights_dir}/${normalized_base}.engine"
      model_failed=1
    elif ! mv -f -- "${weights_dir}/${normalized_base}.engine" "${fp32_engine}"; then
      cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_fp32_export"
      record_failure "${model_name}" "failed to rename FP32 engine to ${fp32_engine}"
      model_failed=1
    else
      cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_fp32_export"
      printf '    FP32: %s\n' "${fp32_engine}"
    fi
  fi

  if [[ -f "${fp16_engine}" ]]; then
    printf '    FP16: exists, skipping.\n'
  else
    printf '    FP16: exporting.\n'
    if ! export_engine "${weights_dir}" "${normalized_base}" "${model_name}_fp16_export" half=True; then
      cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_fp16_export"
      record_failure "${model_name}" "FP16 export failed"
      model_failed=1
    elif [[ ! -f "${weights_dir}/${normalized_base}.engine" ]]; then
      cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_fp16_export"
      record_failure "${model_name}" "FP16 export did not produce ${weights_dir}/${normalized_base}.engine"
      model_failed=1
    elif ! mv -f -- "${weights_dir}/${normalized_base}.engine" "${fp16_engine}"; then
      cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_fp16_export"
      record_failure "${model_name}" "failed to rename FP16 engine to ${fp16_engine}"
      model_failed=1
    else
      cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_fp16_export"
      printf '    FP16: %s\n' "${fp16_engine}"
    fi
  fi

  if [[ -f "${int8_engine}" ]]; then
    printf '    INT8: exists, skipping.\n'
  else
    printf '    INT8: exporting.\n'
    if ! prepare_int8_labels "${model_name}"; then
      record_failure "${model_name}" "INT8 calibration labels could not be prepared"
      model_failed=1
    else
      remove_trt_caches "${weights_dir}"
      if ! export_engine "${weights_dir}" "${normalized_base}" "${model_name}_int8_export" int8=True data="${DATA_YAML}"; then
        cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_int8_export"
        record_failure "${model_name}" "INT8 export failed"
        model_failed=1
      elif [[ ! -f "${weights_dir}/${normalized_base}.engine" ]]; then
        cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_int8_export"
        record_failure "${model_name}" "INT8 export did not produce ${weights_dir}/${normalized_base}.engine"
        model_failed=1
      elif ! mv -f -- "${weights_dir}/${normalized_base}.engine" "${int8_engine}"; then
        cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_int8_export"
        record_failure "${model_name}" "failed to rename INT8 engine to ${int8_engine}"
        model_failed=1
      else
        cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_int8_export"
        printf '    INT8: %s\n' "${int8_engine}"
      fi
    fi
  fi

  cleanup_export_artifacts "${weights_dir}" "${normalized_base}" "${model_name}_cleanup"

  if [[ "${model_failed}" -eq 0 && -f "${fp32_engine}" && -f "${fp16_engine}" && -f "${int8_engine}" ]]; then
    READY_COUNT=$((READY_COUNT + 1))
  else
    FAILED_COUNT=$((FAILED_COUNT + 1))
  fi
done

printf '\nSummary\n'
printf '  total pruned checkpoints found: %d\n' "${TOTAL_MODELS}"
printf '  ready (3 engines present):      %d\n' "${READY_COUNT}"
printf '  skipped (already ready):       %d\n' "${SKIPPED_COUNT}"
printf '  failed or incomplete:          %d\n' "${FAILED_COUNT}"

if [[ "${#FAILURE_MODELS[@]}" -gt 0 ]]; then
  printf '\nFailures\n' >&2
  for i in "${!FAILURE_MODELS[@]}"; do
    printf '  %s\n' "${FAILURE_MODELS[$i]}" >&2
    printf '  %s\n' "${FAILURE_MESSAGES[$i]}" >&2
  done
  exit 1
fi

exit 0
