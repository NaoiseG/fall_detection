#!/usr/bin/env bash
set -uo pipefail
IFS=$'\n\t'

DEFAULT_ROOT="/home/jetson/NaoiseG/fall_detection/pruning/pruned_models"

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

if [[ ! -d "${ROOT_DIR}" ]]; then
  printf 'ERROR: ROOT_DIR not found: %s\n' "${ROOT_DIR}" >&2
  exit 1
fi

if ! command -v yolo >/dev/null 2>&1; then
  printf 'ERROR: Could not find `yolo` on PATH.\n' >&2
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
printf 'Model directories seen: %d\n' "${TOTAL_SEEN}"

INDEX=0
for model_dir in "${MODEL_DIRS[@]}"; do
  INDEX=$((INDEX + 1))

  model_name="$(basename -- "${model_dir}")"
  pt_path="${model_dir}/best.pt"
  exported_engine="${model_dir}/best.engine"
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

  cleanup_export_artifacts "${model_dir}" "best" "${export_name}"

  if ! (
    cd -- "${model_dir}" &&
    yolo export \
      model="./best.pt" \
      format=engine \
      project=. \
      name="${export_name}" \
      exist_ok=True
  ); then
    cleanup_export_artifacts "${model_dir}" "best" "${export_name}"
    record_failure "${model_dir}" "yolo export failed"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if [[ ! -f "${exported_engine}" ]]; then
    cleanup_export_artifacts "${model_dir}" "best" "${export_name}"
    record_failure "${model_dir}" "expected engine not found after export: ${exported_engine}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if ! mv -f -- "${exported_engine}" "${final_engine}"; then
    rm -f -- "${model_dir}/best.onnx"
    if [[ -e "${model_dir}/${export_name}" ]]; then
      rm -rf -- "${model_dir}/${export_name}"
    fi
    record_failure "${model_dir}" "failed to rename engine to ${final_engine}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  if [[ ! -f "${final_engine}" ]]; then
    cleanup_export_artifacts "${model_dir}" "best" "${export_name}"
    record_failure "${model_dir}" "final engine missing after rename: ${final_engine}"
    FAILED_COUNT=$((FAILED_COUNT + 1))
    continue
  fi

  cleanup_export_artifacts "${model_dir}" "best" "${export_name}"

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
