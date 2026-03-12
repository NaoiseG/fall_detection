#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Build TensorRT engines for AlphaPose FastPose and YOLOv3-SPP using trtexec.

Usage:
  bash build_engines.sh [options]

Options:
  --onnx-dir DIR                Directory containing ONNX files. Used only when
                                an exact ONNX path is not provided.
  --fastpose-onnx PATH          Exact FastPose ONNX path.
  --yolo-onnx PATH              Exact YOLOv3-SPP ONNX path.
  --engine-dir DIR              Output directory for .engine files. Required.
  --precision MODE              fp32 | fp16 | both. Default: both
  --trtexec PATH                trtexec binary. Default: auto-detect.
  --workspace-mb MB             Workspace memory pool in MiB. Default: 2048
  --verbose                     Add --verbose to trtexec.
  --dry-run                     Print commands without running them.

FastPose profile options:
  --fastpose-dynamic            Build FastPose with dynamic batch shapes.
  --fastpose-min-batch N        Default: 1
  --fastpose-opt-batch N        Default: 8
  --fastpose-max-batch N        Default: 32
  --fastpose-height N           Default: 256
  --fastpose-width N            Default: 192
  --fastpose-input-name NAME    Default: images

Engine names generated:
  yolov3_spp_fp32.engine
  yolov3_spp_fp16.engine
  fastpose_fp32.engine
  fastpose_fp16.engine
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

ONNX_DIR=""
FASTPOSE_ONNX=""
YOLO_ONNX=""
ENGINE_DIR=""
PRECISION="both"
TRTEXEC=""
WORKSPACE_MB="2048"
VERBOSE="0"
DRY_RUN="0"
FASTPOSE_DYNAMIC="0"
FASTPOSE_MIN_BATCH="1"
FASTPOSE_OPT_BATCH="8"
FASTPOSE_MAX_BATCH="32"
FASTPOSE_HEIGHT="256"
FASTPOSE_WIDTH="192"
FASTPOSE_INPUT_NAME="images"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --onnx-dir)
            ONNX_DIR="$2"
            shift 2
            ;;
        --fastpose-onnx)
            FASTPOSE_ONNX="$2"
            shift 2
            ;;
        --yolo-onnx)
            YOLO_ONNX="$2"
            shift 2
            ;;
        --engine-dir)
            ENGINE_DIR="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
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
        --fastpose-dynamic)
            FASTPOSE_DYNAMIC="1"
            shift
            ;;
        --fastpose-min-batch)
            FASTPOSE_MIN_BATCH="$2"
            shift 2
            ;;
        --fastpose-opt-batch)
            FASTPOSE_OPT_BATCH="$2"
            shift 2
            ;;
        --fastpose-max-batch)
            FASTPOSE_MAX_BATCH="$2"
            shift 2
            ;;
        --fastpose-height)
            FASTPOSE_HEIGHT="$2"
            shift 2
            ;;
        --fastpose-width)
            FASTPOSE_WIDTH="$2"
            shift 2
            ;;
        --fastpose-input-name)
            FASTPOSE_INPUT_NAME="$2"
            shift 2
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

[[ -n "$ENGINE_DIR" ]] || fail "--engine-dir is required"
mkdir -p "$ENGINE_DIR"

if [[ -z "$FASTPOSE_ONNX" && -n "$ONNX_DIR" ]]; then
    FASTPOSE_ONNX="$ONNX_DIR/fastpose.onnx"
fi
if [[ -z "$YOLO_ONNX" && -n "$ONNX_DIR" ]]; then
    YOLO_ONNX="$ONNX_DIR/yolov3_spp.onnx"
fi

[[ -n "$FASTPOSE_ONNX" ]] || fail "Provide --fastpose-onnx or --onnx-dir"
[[ -n "$YOLO_ONNX" ]] || fail "Provide --yolo-onnx or --onnx-dir"
[[ -f "$FASTPOSE_ONNX" ]] || fail "FastPose ONNX not found: $FASTPOSE_ONNX"
[[ -f "$YOLO_ONNX" ]] || fail "YOLO ONNX not found: $YOLO_ONNX"

case "$PRECISION" in
    fp32|fp16|both) ;;
    *) fail "--precision must be fp32, fp16, or both" ;;
esac

if [[ -z "$TRTEXEC" ]]; then
    TRTEXEC="$(command -v trtexec || true)"
    if [[ -z "$TRTEXEC" && -x /usr/src/tensorrt/bin/trtexec ]]; then
        TRTEXEC=/usr/src/tensorrt/bin/trtexec
    fi
fi
[[ -n "$TRTEXEC" ]] || fail "Could not find trtexec. Pass it explicitly with --trtexec"
[[ -x "$TRTEXEC" ]] || fail "trtexec is not executable: $TRTEXEC"

if (( FASTPOSE_MIN_BATCH < 1 || FASTPOSE_OPT_BATCH < 1 || FASTPOSE_MAX_BATCH < 1 )); then
    fail "FastPose min/opt/max batch values must all be >= 1"
fi
if (( FASTPOSE_MIN_BATCH > FASTPOSE_OPT_BATCH || FASTPOSE_OPT_BATCH > FASTPOSE_MAX_BATCH )); then
    fail "Expected fastpose min_batch <= opt_batch <= max_batch"
fi

COMMON_ARGS=("--memPoolSize=workspace:${WORKSPACE_MB}" "--skipInference")
if [[ "$VERBOSE" == "1" ]]; then
    COMMON_ARGS+=("--verbose")
fi

build_one() {
    local onnx_path="$1"
    local engine_path="$2"
    local mode="$3"
    shift 3
    local extra_args=("$@")

    local cmd=("$TRTEXEC" "--onnx=${onnx_path}" "--saveEngine=${engine_path}")
    cmd+=("${COMMON_ARGS[@]}")
    if [[ "$mode" == "fp16" ]]; then
        cmd+=("--fp16")
    fi
    cmd+=("${extra_args[@]}")
    run_cmd "${cmd[@]}"
}

FASTPOSE_PROFILE_ARGS=()
if [[ "$FASTPOSE_DYNAMIC" == "1" ]]; then
    FASTPOSE_PROFILE_ARGS+=(
        "--minShapes=${FASTPOSE_INPUT_NAME}:${FASTPOSE_MIN_BATCH}x3x${FASTPOSE_HEIGHT}x${FASTPOSE_WIDTH}"
        "--optShapes=${FASTPOSE_INPUT_NAME}:${FASTPOSE_OPT_BATCH}x3x${FASTPOSE_HEIGHT}x${FASTPOSE_WIDTH}"
        "--maxShapes=${FASTPOSE_INPUT_NAME}:${FASTPOSE_MAX_BATCH}x3x${FASTPOSE_HEIGHT}x${FASTPOSE_WIDTH}"
        "--shapes=${FASTPOSE_INPUT_NAME}:${FASTPOSE_OPT_BATCH}x3x${FASTPOSE_HEIGHT}x${FASTPOSE_WIDTH}"
    )
fi

echo "Build settings"
echo "  trtexec:        $TRTEXEC"
echo "  fastpose onnx:  $FASTPOSE_ONNX"
echo "  yolo onnx:      $YOLO_ONNX"
echo "  engine dir:     $ENGINE_DIR"
echo "  precision:      $PRECISION"
echo "  workspace MiB:  $WORKSPACE_MB"
echo "  fastpose dyn:   $FASTPOSE_DYNAMIC"

if [[ "$PRECISION" == "fp32" || "$PRECISION" == "both" ]]; then
    build_one "$YOLO_ONNX" "$ENGINE_DIR/yolov3_spp_fp32.engine" fp32
    build_one "$FASTPOSE_ONNX" "$ENGINE_DIR/fastpose_fp32.engine" fp32 "${FASTPOSE_PROFILE_ARGS[@]}"
fi

if [[ "$PRECISION" == "fp16" || "$PRECISION" == "both" ]]; then
    build_one "$YOLO_ONNX" "$ENGINE_DIR/yolov3_spp_fp16.engine" fp16
    build_one "$FASTPOSE_ONNX" "$ENGINE_DIR/fastpose_fp16.engine" fp16 "${FASTPOSE_PROFILE_ARGS[@]}"
fi

echo
echo "Done. Engines are in: $ENGINE_DIR"
