#!/usr/bin/env bash
set -uo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
QUANTISATION_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd -- "${QUANTISATION_DIR}/.." && pwd -P)"
DEFAULT_ROOT="/home/jetson/NaoiseG/fall_detection/pruning/pruned_models"
DEFAULT_VENV_PYTHON="${QUANTISATION_DIR}/venvs/yolo_export/bin/python"
DEFAULT_PRUNING_VENV_PYTHON="${REPO_ROOT}/pruning/pruning_venv/bin/python"
DEFAULT_NORMALIZE_HELPER="${SCRIPT_DIR}/normalize_pruned_checkpoint.py"
DEFAULT_MODELOPT_ROOT="${REPO_ROOT}/pruning/Model-Optimizer"
DEFAULT_PRUNING_ULTRALYTICS_ROOT="${REPO_ROOT}/pruning/yolov11-prune"

usage() {
  cat <<EOF
Usage: $(basename "$0") [ROOT_DIR]

Exports each immediate pruned-model subdirectory under ROOT_DIR to a TensorRT
FP32 engine using \`yolo export\`.

Default ROOT_DIR:
  ${DEFAULT_ROOT}
EOF
}

cleanup_export_artifacts() {
  local model_dir="$1"
  local base_name="$2"
  local export_name="$3"

  local exported_engine="${model_dir}/${base_name}.engine"
  local onnx_file="${model_dir}/${base_name}.onnx"
  local export_path="${model_dir}/${export_name}"

  rm -f -- "${exported_engine}" "${onnx_file}"

  if [[ -e "${export_path}" ]]; then
    rm -rf -- "${export_path}"
  fi
}

record_failure() {
  local model_dir="$1"
  local message="$2"

  FAILURE_DIRS+=("${model_dir}")
  FAILURE_MESSAGES+=("${message}")
  printf 'ERROR: %s: %s\n' "${model_dir}" "${message}" >&2
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 1 ]]; then
  printf 'ERROR: Expected zero or one positional argument.\n\n' >&2
  usage >&2
  exit 2
fi

ROOT_DIR="${1:-${DEFAULT_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-}"
NORMALIZE_PYTHON="${NORMALIZE_PYTHON:-}"
NORMALIZE_HELPER="${NORMALIZE_HELPER:-${DEFAULT_NORMALIZE_HELPER}}"
MODELOPT_ROOT="${MODELOPT_ROOT:-${DEFAULT_MODELOPT_ROOT}}"
PRUNING_ULTRALYTICS_ROOT="${PRUNING_ULTRALYTICS_ROOT:-${DEFAULT_PRUNING_ULTRALYTICS_ROOT}}"

if [[ ! -d "${ROOT_DIR}" ]]; then
  printf 'ERROR: ROOT_DIR not found: %s\n' "${ROOT_DIR}" >&2
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

ROOT_DIR="$(cd -- "${ROOT_DIR}" && pwd -P)"

declare -a MODEL_DIRS=()
declare -a FAILURE_DIRS=()
declare -a FAILURE_MESSAGES=()

while IFS= read -r -d '' model_dir; do
  MODEL_DIRS+=("${model_dir}")
done < <(find "${ROOT_DIR}" -mindepth 1 -maxdepth 1 -type d -print0)

TOTAL_SEEN="${#MODEL_DIRS[@]}"
EXPORTED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

printf 'Root directory: %s\n' "${ROOT_DIR}"
printf 'Python: %s\n' "${PYTHON_BIN}"
printf 'Normalize python: %s\n' "${NORMALIZE_PYTHON}"
printf 'Normalize helper: %s\n' "${NORMALIZE_HELPER}"
printf 'Model directories seen: %d\n' "${TOTAL_SEEN}"

INDEX=0
for model_dir in "${MODEL_DIRS[@]}"; do
  INDEX=$((INDEX + 1))

  model_name="$(basename -- "${model_dir}")"
  pt_path="${model_dir}/best.pt"
  normalized_pt="${model_dir}/best_export_ready.pt"
  normalized_base="best_export_ready"
  exported_engine="${model_dir}/${normalized_base}.engine"
  final_engine="${model_dir}/${model_name}_fp32.engine"
  export_name="${model_name}_fp32_export"

  printf '\n==> [%d/%d] Exporting %s\n' "${INDEX}" "${TOTAL_SEEN}" "${model_name}"
  printf '    Input:  %s\n' "${pt_path}"
  printf '    Output: %s\n' "${final_engine}"

  if [[ -f "${final_engine}" ]]; then
    printf '    Skipping: final engine already exists.\n'
    SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
    continue
  fi

  if [[ ! -f "${pt_path}" ]]; then
    record_failure "${model_dir}" "best.pt not found"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  printf '    Normalized checkpoint: %s\n' "${normalized_pt}"

  cleanup_export_artifacts "${model_dir}" "${normalized_base}" "${export_name}"
  rm -f -- "${normalized_pt}"

  if ! "${NORMALIZE_PYTHON}" "${NORMALIZE_HELPER}" \
    --input "${pt_path}" \
    --output "${normalized_pt}" \
    --modelopt-root "${MODELOPT_ROOT}" \
    --ultralytics-root "${PRUNING_ULTRALYTICS_ROOT}" \
    --overwrite; then
    cleanup_export_artifacts "${model_dir}" "${normalized_base}" "${export_name}"
    rm -f -- "${normalized_pt}"
    record_failure "${model_dir}" "checkpoint normalization failed"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if [[ ! -f "${normalized_pt}" ]]; then
    cleanup_export_artifacts "${model_dir}" "${normalized_base}" "${export_name}"
    record_failure "${model_dir}" "normalized checkpoint missing after normalization: ${normalized_pt}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if ! (
    cd -- "${model_dir}" &&
    "${PYTHON_BIN}" -c 'from ultralytics.cfg import entrypoint; entrypoint()' export \
      model="./${normalized_base}.pt" \
      format=engine \
      project=. \
      name="${export_name}" \
      exist_ok=True
  ); then
    cleanup_export_artifacts "${model_dir}" "${normalized_base}" "${export_name}"
    rm -f -- "${normalized_pt}"
    record_failure "${model_dir}" "yolo export failed"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if [[ ! -f "${exported_engine}" ]]; then
    cleanup_export_artifacts "${model_dir}" "${normalized_base}" "${export_name}"
    rm -f -- "${normalized_pt}"
    record_failure "${model_dir}" "expected engine not found after export: ${exported_engine}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if ! mv -f -- "${exported_engine}" "${final_engine}"; then
    rm -f -- "${model_dir}/${normalized_base}.onnx"
    if [[ -e "${model_dir}/${export_name}" ]]; then
      rm -rf -- "${model_dir}/${export_name}"
    fi
    rm -f -- "${normalized_pt}"
    record_failure "${model_dir}" "failed to rename engine to ${final_engine}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if [[ ! -f "${final_engine}" ]]; then
    cleanup_export_artifacts "${model_dir}" "${normalized_base}" "${export_name}"
    rm -f -- "${normalized_pt}"
    record_failure "${model_dir}" "final engine missing after rename: ${final_engine}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  cleanup_export_artifacts "${model_dir}" "${normalized_base}" "${export_name}"
  rm -f -- "${normalized_pt}"

  printf '    Final engine: %s\n' "${final_engine}"
  EXPORTED_COUNT=$((EXPORTED_COUNT + 1))
done

printf '\nSummary\n'
printf '  total model directories seen: %d\n' "${TOTAL_SEEN}"
printf '  exported successfully:       %d\n' "${EXPORTED_COUNT}"
printf '  skipped (engine existed):    %d\n' "${SKIPPED_COUNT}"
printf '  failed:                      %d\n' "${FAILED_COUNT}"

if [[ "${FAILED_COUNT}" -gt 0 ]]; then
  printf '\nFailures\n' >&2
  for i in "${!FAILURE_DIRS[@]}"; do
    printf '  %s\n' "${FAILURE_DIRS[$i]}" >&2
    printf '  %s\n' "${FAILURE_MESSAGES[$i]}" >&2
  done
  exit 1
fi

exit 0
