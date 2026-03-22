#!/usr/bin/env bash

# Export the two-stage ViTPose stack used by this repo:
#   1. RT-DETR detector
#   2. ViTPose estimator
#
# The script first exports the requested ViTPose stages to ONNX via
# `export_vitpose_onnx.py`, then builds TensorRT engines with `trtexec`.
#
# Notes:
# - This script must run on a machine with NVIDIA TensorRT and `trtexec`.
# - Engine files are hardware/runtime specific. Rebuild them on the target
#   CUDA/TensorRT/GPU stack.
# - The generated engines contain only model forward passes. Hugging Face
#   preprocessing/postprocessing remains outside the engines.
# - INT8 detector export requires a calibration cache generated from a sample
#   image set. This script can generate that cache from data.yaml automatically.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
QUANTISATION_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROTOTYPE_DIR="${REPO_ROOT}/fall_models/Prototype"

DETECTOR_MODEL="PekingU/rtdetr_r50vd_coco_o365"
POSE_MODEL="usyd-community/vitpose-base"
OUTPUT_ROOT="${QUANTISATION_DIR}/models/vitpose_trt"
PYTHON_BIN=""
TRTEXEC_BIN=""
EXPORT_DEVICE="cpu"
ONNX_OPSET=17
MAX_BATCH=8
LOCAL_FILES_ONLY=0
STATIC_BATCH=0
VERBOSE=0
DETECTOR_INT8=0
DETECTOR_ONLY=0
DETECTOR_HEIGHT=""
DETECTOR_WIDTH=""
POSE_HEIGHT=""
POSE_WIDTH=""
DETECTOR_CALIB_DIR="${QUANTISATION_DIR}/calibration_dataset_upfall"
DETECTOR_CALIB_CACHE=""
DETECTOR_CALIB_SPLIT="val"
DETECTOR_CALIB_BATCH_SIZE=8
DETECTOR_CALIB_MAX_IMAGES=256
WORKSPACE_MB=2048
FORCE_DETECTOR_CALIB=0
TRTEXEC_EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Exports RT-DETR and ViTPose to ONNX and then builds TensorRT engines.

Options:
  --detector-model ID        Hugging Face detector model id.
  --pose-model ID            Hugging Face ViTPose model id.
  --output-root PATH         Output root for ONNX, manifests, and engines.
  --python PATH              Python interpreter to use.
  --trtexec PATH             trtexec binary to use.
  --device cpu|cuda          Device used during ONNX export. Default: cpu
  --opset N                  ONNX opset version. Default: 17
  --max-batch N              Max TensorRT optimization-profile batch. Default: 8
  --static-batch             Build fixed-batch engines with batch size 1.
  --detector-int8            Build detector as INT8 instead of fp32/fp16.
  --detector-only            Export/build only the detector stage.
  --detector-height N        Override detector input height.
  --detector-width N         Override detector input width.
  --pose-height N            Override pose input height.
  --pose-width N             Override pose input width.
  --detector-calib-dir DIR   Calibration dataset dir. Default: ./calibration_dataset_upfall
  --detector-calib-cache P   Calibration cache file. Default: <calib-dir>/rtdetr_int8.cache
  --detector-calib-split S   data.yaml split for calibration. Default: val
  --detector-calib-batch N   Calibration batch size. Default: 8
  --detector-calib-max N     Max calibration images. Default: 256
  --force-detector-calib     Regenerate the detector INT8 calibration cache.
  --workspace-mb MB          TensorRT workspace in MiB for INT8 build/cache. Default: 2048
  --local-files-only         Load Hugging Face assets from local cache only.
  --trtexec-arg ARG          Extra argument forwarded to trtexec. Repeatable.
  --verbose                  Pass --verbose to trtexec.
  -h, --help                 Show this help text.

Examples:
  $(basename "$0")
  $(basename "$0") --max-batch 16 --trtexec /usr/src/tensorrt/bin/trtexec
  $(basename "$0") --output-root /data/models/vitpose_trt --local-files-only
  $(basename "$0") --detector-int8 --detector-only
EOF
}

run_cmd() {
  local parts=()
  local arg
  for arg in "$@"; do
    parts+=("$(printf '%q' "$arg")")
  done
  local IFS=' '
  printf '[cmd] %s\n' "${parts[*]}"
  "$@"
}

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
      printf 'ERROR: Python interpreter not executable: %s\n' "${PYTHON_BIN}" >&2
      exit 1
    fi
    return
  fi

  if [[ -x "${PROTOTYPE_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${PROTOTYPE_DIR}/venv/bin/python"
    return
  fi

  if [[ -x "${PROTOTYPE_DIR}/venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="${PROTOTYPE_DIR}/venv/Scripts/python.exe"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return
  fi

  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return
  fi

  printf 'ERROR: Could not find a usable Python interpreter.\n' >&2
  exit 1
}

resolve_trtexec_bin() {
  if [[ -n "${TRTEXEC_BIN}" ]]; then
    if [[ ! -x "${TRTEXEC_BIN}" ]]; then
      printf 'ERROR: trtexec binary not executable: %s\n' "${TRTEXEC_BIN}" >&2
      exit 1
    fi
    return
  fi

  if command -v trtexec >/dev/null 2>&1; then
    TRTEXEC_BIN="$(command -v trtexec)"
    return
  fi

  if [[ -x "/usr/src/tensorrt/bin/trtexec" ]]; then
    TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"
    return
  fi

  printf 'ERROR: Could not find trtexec. Set --trtexec or add it to PATH.\n' >&2
  exit 1
}

build_engine() {
  local stage_slug="$1"
  local onnx_path="$2"
  local input_name="$3"
  local channels="$4"
  local height="$5"
  local width="$6"
  local precision="$7"

  local engine_dir="${OUTPUT_ROOT}/engines"
  mkdir -p -- "${engine_dir}"

  local engine_path="${engine_dir}/${stage_slug}_${precision}.engine"

  local min_batch=1
  local opt_batch=1
  local max_batch="${MAX_BATCH}"

  if [[ "${STATIC_BATCH}" -eq 1 ]]; then
    max_batch=1
  else
    if (( MAX_BATCH >= 4 )); then
      opt_batch=4
    else
      opt_batch="${MAX_BATCH}"
    fi
  fi

  local min_shape="${input_name}:${min_batch}x${channels}x${height}x${width}"
  local opt_shape="${input_name}:${opt_batch}x${channels}x${height}x${width}"
  local max_shape="${input_name}:${max_batch}x${channels}x${height}x${width}"

  local cmd=(
    "${TRTEXEC_BIN}"
    "--onnx=${onnx_path}"
    "--saveEngine=${engine_path}"
    "--skipInference"
    "--minShapes=${min_shape}"
    "--optShapes=${opt_shape}"
    "--maxShapes=${max_shape}"
  )

  if [[ "${precision}" == "fp16" ]]; then
    cmd+=(--fp16)
  fi

  if [[ "${VERBOSE}" -eq 1 ]]; then
    cmd+=(--verbose)
  fi

  if [[ "${#TRTEXEC_EXTRA_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${TRTEXEC_EXTRA_ARGS[@]}")
  fi

  printf '[build] %s -> %s\n' "${stage_slug}" "${engine_path}"
  run_cmd "${cmd[@]}"
}

generate_detector_calib_cache() {
  if [[ -z "${DETECTOR_CALIB_CACHE}" ]]; then
    DETECTOR_CALIB_CACHE="${DETECTOR_CALIB_DIR}/rtdetr_int8.cache"
  fi

  if [[ -f "${DETECTOR_CALIB_CACHE}" && "${FORCE_DETECTOR_CALIB}" -eq 0 ]]; then
    printf '[info] Using existing detector calibration cache: %s\n' "${DETECTOR_CALIB_CACHE}"
    return
  fi

  if [[ ! -d "${DETECTOR_CALIB_DIR}" ]]; then
    printf 'ERROR: Detector calibration dir not found: %s\n' "${DETECTOR_CALIB_DIR}" >&2
    exit 1
  fi

  local data_yaml="${DETECTOR_CALIB_DIR}/data.yaml"
  if [[ ! -f "${data_yaml}" ]]; then
    printf 'ERROR: Detector calibration data.yaml not found: %s\n' "${data_yaml}" >&2
    exit 1
  fi

  mkdir -p -- "$(dirname -- "${DETECTOR_CALIB_CACHE}")"

  local cmd=(
    "${PYTHON_BIN}"
    "${QUANTISATION_DIR}/generate_rtdetr_int8_cache.py"
    "--onnx" "${DETECTOR_ONNX}"
    "--data" "${data_yaml}"
    "--cache" "${DETECTOR_CALIB_CACHE}"
    "--detector-model" "${DETECTOR_MODEL_ID}"
    "--split" "${DETECTOR_CALIB_SPLIT}"
    "--batch-size" "${DETECTOR_CALIB_BATCH_SIZE}"
    "--max-images" "${DETECTOR_CALIB_MAX_IMAGES}"
    "--workspace-mb" "${WORKSPACE_MB}"
    "--input-name" "${DETECTOR_INPUT_NAME}"
    "--input-height" "${DETECTOR_HEIGHT}"
    "--input-width" "${DETECTOR_WIDTH}"
  )

  if [[ "${LOCAL_FILES_ONLY}" -eq 1 ]]; then
    cmd+=(--local-files-only)
  fi

  if [[ "${FORCE_DETECTOR_CALIB}" -eq 1 ]]; then
    cmd+=(--force)
  fi

  if [[ "${VERBOSE}" -eq 1 ]]; then
    cmd+=(--verbose)
  fi

  printf '[calib] Generating RT-DETR INT8 calibration cache: %s\n' "${DETECTOR_CALIB_CACHE}"
  run_cmd "${cmd[@]}"

  if [[ ! -f "${DETECTOR_CALIB_CACHE}" ]]; then
    printf 'ERROR: Detector calibration cache was not written: %s\n' "${DETECTOR_CALIB_CACHE}" >&2
    exit 1
  fi
}

build_detector_int8_engine() {
  local stage_slug="$1"
  local onnx_path="$2"
  local input_name="$3"
  local channels="$4"
  local height="$5"
  local width="$6"

  local engine_dir="${OUTPUT_ROOT}/engines"
  mkdir -p -- "${engine_dir}"

  local engine_path="${engine_dir}/${stage_slug}_int8.engine"

  local min_batch=1
  local opt_batch=1
  local max_batch="${MAX_BATCH}"

  if [[ "${STATIC_BATCH}" -eq 1 ]]; then
    max_batch=1
  else
    if (( MAX_BATCH >= 4 )); then
      opt_batch=4
    else
      opt_batch="${MAX_BATCH}"
    fi
  fi

  local min_shape="${input_name}:${min_batch}x${channels}x${height}x${width}"
  local opt_shape="${input_name}:${opt_batch}x${channels}x${height}x${width}"
  local max_shape="${input_name}:${max_batch}x${channels}x${height}x${width}"

  local cmd=(
    "${TRTEXEC_BIN}"
    "--onnx=${onnx_path}"
    "--saveEngine=${engine_path}"
    "--skipInference"
    "--int8"
    "--calib=${DETECTOR_CALIB_CACHE}"
    "--memPoolSize=workspace:${WORKSPACE_MB}"
    "--minShapes=${min_shape}"
    "--optShapes=${opt_shape}"
    "--maxShapes=${max_shape}"
  )

  if [[ "${VERBOSE}" -eq 1 ]]; then
    cmd+=(--verbose)
  fi

  if [[ "${#TRTEXEC_EXTRA_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${TRTEXEC_EXTRA_ARGS[@]}")
  fi

  printf '[build] %s -> %s\n' "${stage_slug}" "${engine_path}"
  run_cmd "${cmd[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --detector-model)
      DETECTOR_MODEL="$2"
      shift 2
      ;;
    --pose-model)
      POSE_MODEL="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --trtexec)
      TRTEXEC_BIN="$2"
      shift 2
      ;;
    --device)
      EXPORT_DEVICE="$2"
      shift 2
      ;;
    --opset)
      ONNX_OPSET="$2"
      shift 2
      ;;
    --max-batch)
      MAX_BATCH="$2"
      shift 2
      ;;
    --static-batch)
      STATIC_BATCH=1
      shift
      ;;
    --detector-int8)
      DETECTOR_INT8=1
      shift
      ;;
    --detector-only)
      DETECTOR_ONLY=1
      shift
      ;;
    --detector-height)
      DETECTOR_HEIGHT="$2"
      shift 2
      ;;
    --detector-width)
      DETECTOR_WIDTH="$2"
      shift 2
      ;;
    --pose-height)
      POSE_HEIGHT="$2"
      shift 2
      ;;
    --pose-width)
      POSE_WIDTH="$2"
      shift 2
      ;;
    --detector-calib-dir)
      DETECTOR_CALIB_DIR="$2"
      shift 2
      ;;
    --detector-calib-cache)
      DETECTOR_CALIB_CACHE="$2"
      shift 2
      ;;
    --detector-calib-split)
      DETECTOR_CALIB_SPLIT="$2"
      shift 2
      ;;
    --detector-calib-batch)
      DETECTOR_CALIB_BATCH_SIZE="$2"
      shift 2
      ;;
    --detector-calib-max)
      DETECTOR_CALIB_MAX_IMAGES="$2"
      shift 2
      ;;
    --force-detector-calib)
      FORCE_DETECTOR_CALIB=1
      shift
      ;;
    --workspace-mb)
      WORKSPACE_MB="$2"
      shift 2
      ;;
    --local-files-only)
      LOCAL_FILES_ONLY=1
      shift
      ;;
    --trtexec-arg)
      TRTEXEC_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'ERROR: Unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

resolve_python_bin
resolve_trtexec_bin

mkdir -p -- "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd -- "${OUTPUT_ROOT}" && pwd)"

MANIFEST_DIR="${OUTPUT_ROOT}/manifests"
MANIFEST_ENV="${MANIFEST_DIR}/vitpose_export_manifest.env"

if [[ "${DETECTOR_INT8}" -eq 1 ]]; then
  if ! command -v realpath >/dev/null 2>&1; then
    printf 'ERROR: realpath is required for detector INT8 export.\n' >&2
    exit 1
  fi
  DETECTOR_CALIB_DIR="$(realpath "${DETECTOR_CALIB_DIR}")"
  if [[ -n "${DETECTOR_CALIB_CACHE}" ]]; then
    DETECTOR_CALIB_CACHE="$(realpath -m "${DETECTOR_CALIB_CACHE}")"
  fi
fi

EXPORT_CMD=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/export_vitpose_onnx.py"
  "--detector-model" "${DETECTOR_MODEL}"
  "--pose-model" "${POSE_MODEL}"
  "--output-root" "${OUTPUT_ROOT}"
  "--device" "${EXPORT_DEVICE}"
  "--opset" "${ONNX_OPSET}"
)

if [[ "${STATIC_BATCH}" -eq 1 ]]; then
  EXPORT_CMD+=(--static-batch)
fi

if [[ "${DETECTOR_ONLY}" -eq 1 ]]; then
  EXPORT_CMD+=(--detector-only)
fi

if [[ "${LOCAL_FILES_ONLY}" -eq 1 ]]; then
  EXPORT_CMD+=(--local-files-only)
fi

if [[ -n "${DETECTOR_HEIGHT}" || -n "${DETECTOR_WIDTH}" ]]; then
  if [[ -z "${DETECTOR_HEIGHT}" || -z "${DETECTOR_WIDTH}" ]]; then
    printf 'ERROR: Set both --detector-height and --detector-width together.\n' >&2
    exit 1
  fi
  EXPORT_CMD+=(--detector-height "${DETECTOR_HEIGHT}" --detector-width "${DETECTOR_WIDTH}")
fi

if [[ "${DETECTOR_ONLY}" -eq 0 && ( -n "${POSE_HEIGHT}" || -n "${POSE_WIDTH}" ) ]]; then
  if [[ -z "${POSE_HEIGHT}" || -z "${POSE_WIDTH}" ]]; then
    printf 'ERROR: Set both --pose-height and --pose-width together.\n' >&2
    exit 1
  fi
  EXPORT_CMD+=(--pose-height "${POSE_HEIGHT}" --pose-width "${POSE_WIDTH}")
fi

printf '[info] python: %s\n' "${PYTHON_BIN}"
printf '[info] trtexec: %s\n' "${TRTEXEC_BIN}"
printf '[info] output root: %s\n' "${OUTPUT_ROOT}"

run_cmd "${EXPORT_CMD[@]}"

if [[ ! -f "${MANIFEST_ENV}" ]]; then
  printf 'ERROR: Expected manifest env file was not written: %s\n' "${MANIFEST_ENV}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${MANIFEST_ENV}"

if [[ "${DETECTOR_INT8}" -eq 1 ]]; then
  generate_detector_calib_cache
  build_detector_int8_engine "${DETECTOR_SLUG}" "${DETECTOR_ONNX}" "${DETECTOR_INPUT_NAME}" "${DETECTOR_CHANNELS}" "${DETECTOR_HEIGHT}" "${DETECTOR_WIDTH}"
else
  build_engine "${DETECTOR_SLUG}" "${DETECTOR_ONNX}" "${DETECTOR_INPUT_NAME}" "${DETECTOR_CHANNELS}" "${DETECTOR_HEIGHT}" "${DETECTOR_WIDTH}" fp32
  build_engine "${DETECTOR_SLUG}" "${DETECTOR_ONNX}" "${DETECTOR_INPUT_NAME}" "${DETECTOR_CHANNELS}" "${DETECTOR_HEIGHT}" "${DETECTOR_WIDTH}" fp16
fi

if [[ "${POSE_PRESENT:-0}" -eq 1 ]]; then
  if [[ "${DETECTOR_INT8}" -eq 1 ]]; then
    build_engine "${POSE_SLUG}" "${POSE_ONNX}" "${POSE_INPUT_NAME}" "${POSE_CHANNELS}" "${POSE_HEIGHT}" "${POSE_WIDTH}" fp16
    printf '\n[done] Exported detector INT8 and pose FP16 TensorRT engines under:\n'
  else
    build_engine "${POSE_SLUG}" "${POSE_ONNX}" "${POSE_INPUT_NAME}" "${POSE_CHANNELS}" "${POSE_HEIGHT}" "${POSE_WIDTH}" fp32
    build_engine "${POSE_SLUG}" "${POSE_ONNX}" "${POSE_INPUT_NAME}" "${POSE_CHANNELS}" "${POSE_HEIGHT}" "${POSE_WIDTH}" fp16
    printf '\n[done] Exported both ViTPose stages to fp32/fp16 TensorRT engines under:\n'
  fi
else
  if [[ "${DETECTOR_INT8}" -eq 1 ]]; then
    printf '\n[done] Exported detector INT8 TensorRT engine under:\n'
  else
    printf '\n[done] Exported detector fp32/fp16 TensorRT engines under:\n'
  fi
fi
printf '  %s/engines\n' "${OUTPUT_ROOT}"
