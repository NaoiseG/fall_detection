#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build an INT8 TensorRT engine for AlphaPose YOLOv3-SPP.

This script follows the same calibration dataset convention as export_models_int8.sh:
  CALIB_DIR/data.yaml

Important:
- trtexec cannot use data.yaml directly.
- You must provide or pre-generate a TensorRT INT8 calibration cache
  that was created from that same calibration dataset.

Usage:
  bash build_yolov3_spp_int8.sh [options]

Options:
  --onnx PATH              Exact YOLOv3-SPP ONNX path. Required.
  --engine PATH            Output INT8 engine path. Required.
  --calib-dir DIR          Calibration dataset dir. Default: ./calibration_dataset_upfall
  --calib-cache PATH       Calibration cache file. Default: <calib-dir>/yolov3_spp_int8.cache
  --trtexec PATH           trtexec binary. Default: auto-detect
  --workspace-mb MB        Workspace memory pool in MiB. Default: 2048
  --verbose                Add --verbose to trtexec
  --dry-run                Print command without running it

Example:
  bash build_yolov3_spp_int8.sh \
    --onnx onnx/yolov3_spp.onnx \
    --engine engines/yolov3_spp_int8.engine \
    --calib-dir calibration_dataset_upfall
EOF
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

run_cmd() {
    echo
    echo "+ $*"
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    "$@"
}

YOLO_ONNX=""
ENGINE_PATH=""
CALIB_DIR="./calibration_dataset_upfall"
CALIB_CACHE=""
TRTEXEC=""
WORKSPACE_MB="2048"
VERBOSE="0"
DRY_RUN="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx)
            YOLO_ONNX="$2"
            shift 2
            ;;
        --engine)
            ENGINE_PATH="$2"
            shift 2
            ;;
        --calib-dir)
            CALIB_DIR="$2"
            shift 2
            ;;
        --calib-cache)
            CALIB_CACHE="$2"
            shift 2
            ;;
        --trtexec)
            TRTEXEC="$2"
            shift 2
            ;;
        --workspace-mb)
            WORKSPACE_MB="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="1"
            shift
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            ;;
    esac
done

[[ -n "$YOLO_ONNX" ]]   || fail "--onnx is required"
[[ -n "$ENGINE_PATH" ]] || fail "--engine is required"

CALIB_DIR="$(realpath "$CALIB_DIR")"
DATA_YAML="${CALIB_DIR}/data.yaml"

[[ -f "$YOLO_ONNX" ]] || fail "YOLO ONNX not found: $YOLO_ONNX"
[[ -d "$CALIB_DIR" ]] || fail "Calibration dir not found: $CALIB_DIR"
[[ -f "$DATA_YAML" ]] || fail "Calibration data.yaml not found: $DATA_YAML"

if [[ -z "$CALIB_CACHE" ]]; then
    CALIB_CACHE="${CALIB_DIR}/yolov3_spp_int8.cache"
fi

if [[ -z "$TRTEXEC" ]]; then
    TRTEXEC="$(command -v trtexec || true)"
    if [[ -z "$TRTEXEC" && -x /usr/src/tensorrt/bin/trtexec ]]; then
        TRTEXEC=/usr/src/tensorrt/bin/trtexec
    fi
    if [[ -z "$TRTEXEC" && -x /opt/tensorrt/bin/trtexec ]]; then
        TRTEXEC=/opt/tensorrt/bin/trtexec
    fi
fi

[[ -n "$TRTEXEC" ]] || fail "Could not find trtexec. Pass it explicitly with --trtexec"
[[ -x "$TRTEXEC" ]] || fail "trtexec is not executable: $TRTEXEC"

if [[ ! -f "$CALIB_CACHE" ]]; then
    cat >&2 <<EOF
ERROR: Calibration cache not found: $CALIB_CACHE

This script is using the same calibration dataset location as your existing INT8 exports:
  $DATA_YAML

But trtexec cannot calibrate directly from data.yaml.
You need a TensorRT INT8 calibration cache generated from that same dataset first.
EOF
    exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")"

CMD=(
    "$TRTEXEC"
    "--onnx=${YOLO_ONNX}"
    "--saveEngine=${ENGINE_PATH}"
    "--int8"
    "--calib=${CALIB_CACHE}"
    "--memPoolSize=workspace:${WORKSPACE_MB}"
    "--skipInference"
)

if [[ "$VERBOSE" == "1" ]]; then
    CMD+=("--verbose")
fi

echo "Build settings"
echo "  trtexec:       $TRTEXEC"
echo "  yolo onnx:     $YOLO_ONNX"
echo "  engine:        $ENGINE_PATH"
echo "  calib dir:     $CALIB_DIR"
echo "  data yaml:     $DATA_YAML"
echo "  calib cache:   $CALIB_CACHE"
echo "  workspace MiB: $WORKSPACE_MB"

run_cmd "${CMD[@]}"

echo
echo "Done. INT8 engine written to: $ENGINE_PATH"