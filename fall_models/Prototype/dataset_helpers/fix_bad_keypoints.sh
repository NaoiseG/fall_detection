#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage:
  ./fix_bad_keypoints.sh --keypoints-root <path> --upfall-root <path> [--pose-backend yolo|alphapose|vitpose] [backend-args] [options]

Examples (YOLO, default):
  ./fix_bad_keypoints.sh \
    --keypoints-root "$HOME/scratch/keypoints/UPFall_keypoints/yolo11l/base" \
    --upfall-root "$HOME/scratch/UPFall" \
    --model-path "$HOME/models/yolo11l-pose.pt"

Example (AlphaPose):
  ./fix_bad_keypoints.sh \
    --keypoints-root "$HOME/scratch/keypoints/UPFall_keypoints_alpha" \
    --upfall-root "$HOME/scratch/UPFall" \
    --pose-backend alphapose \
    --alphapose-root "$HOME/models/AlphaPose" \
    --fastpose-weights "$HOME/models/fast_res50_256x192.pth" \
    --detector-weights "$HOME/models/yolov3-spp.weights"

Example (ViTPose):
  ./fix_bad_keypoints.sh \
    --keypoints-root "$HOME/scratch/keypoints/UPFall_keypoints_vit" \
    --upfall-root "$HOME/scratch/UPFall" \
    --pose-backend vitpose \
    --detector-model "PekingU/rtdetr_r50vd_coco_o365" \
    --pose-model "usyd-community/vitpose-base"

Required arguments:
  --keypoints-root PATH  Root directory containing the keypoint tree to clean.
  --upfall-root PATH     Root directory of the UP-Fall frames tree used for regeneration.

Backend selection (default: yolo):
  --pose-backend NAME    Pose backend to use: yolo, alphapose, or vitpose.

YOLO backend arguments (required when --pose-backend yolo):
  --model-path PATH      YOLO pose model weights (.pt or .engine).

AlphaPose backend arguments (required when --pose-backend alphapose):
  --alphapose-root PATH      AlphaPose repo root directory.
  --fastpose-weights PATH    FastPose weights (.pth or .engine).
  --detector-weights PATH    YOLOv3-SPP detector weights (.weights or .engine).
  --cfg-path PATH            AlphaPose config YAML (default: configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml).
  --detector-cfg PATH        Detector cfg file (default: detector/yolo/cfg/yolov3-spp.cfg).

ViTPose backend arguments (required when --pose-backend vitpose):
  --detector-model NAME  RT-DETR model ID/path (default: PekingU/rtdetr_r50vd_coco_o365).
  --pose-model NAME      ViTPose model ID/path (default: usyd-community/vitpose-base).

Optional arguments:
  --subjects SPEC        Subject list/range to regenerate, e.g. 1-17 or 1,3,7.
                         Defaults to subjects auto-detected from --keypoints-root,
                         then --upfall-root.
  --lock-settings NAME   Lock preset for regeneration. Defaults to strict_lock.
  --report-dir PATH      Directory for intermediate CSV reports.
  -h, --help             Show this help text.

Environment overrides:
  PROTOTYPE_DIR       Prototype root. Defaults to the parent of this script.
  PYTHON_BIN          Python executable. Defaults to python, then python3.
  REPORT_DIR          Default report directory when --report-dir is omitted.
  CAMERA1_REPORT_PATH Override the Camera1 report path.
  BAD_REPORT_PATH     Override the Camera2 tracking report path.
  EMPTY_REPORT_PATH   Override the Camera2 empty-window report path.

Notes:
  - --keypoints-root should be the directory that directly contains Subject* folders.
  - The cleanup pipeline still targets the UP-Fall Camera1 and Camera2 layout.
EOF
}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

quote_cmd() {
  local parts=()
  local arg
  for arg in "$@"; do
    parts+=("$(printf '%q' "$arg")")
  done
  local IFS=' '
  printf '%s' "${parts[*]}"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

run_cmd() {
  log "CMD: $(quote_cmd "$@")"
  "$@"
}

remove_stale_report() {
  local report_path="$1"
  if [[ -f "$report_path" ]]; then
    rm -f -- "$report_path"
  fi
}

delete_if_report_exists() {
  local report_path="$1"
  shift

  if [[ ! -f "$report_path" ]]; then
    log "No new report at $report_path; skipping delete step."
    return 0
  fi

  run_cmd "$@"
}

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s' "$PYTHON_BIN"
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    printf '%s' "python"
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "python3"
    return 0
  fi

  echo "ERROR: python/python3 not found in PATH." >&2
  exit 1
}

require_value() {
  local flag_name="$1"
  local flag_value="${2:-}"
  if [[ -z "$flag_value" ]]; then
    echo "ERROR: Missing value for ${flag_name}" >&2
    usage >&2
    exit 1
  fi
}

validate_lock_settings() {
  case "$1" in
    strict_lock|default)
      ;;
    *)
      echo "ERROR: Unsupported lock setting: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
}

discover_subjects() {
  local root="$1"
  local -a subject_ids=()
  local subject_dir=""
  local base_name=""
  local subject_id=""
  local joined=""

  [[ -d "$root" ]] || return 1

  while IFS= read -r -d '' subject_dir; do
    base_name="$(basename -- "$subject_dir")"
    if [[ "$base_name" =~ ^Subject([0-9]+)$ ]]; then
      subject_ids+=("${BASH_REMATCH[1]}")
    fi
  done < <(find "$root" -mindepth 1 -maxdepth 1 -type d -name 'Subject*' -print0 2>/dev/null)

  [[ ${#subject_ids[@]} -gt 0 ]] || return 1

  while IFS= read -r subject_id; do
    [[ -n "$subject_id" ]] || continue
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="$subject_id"
  done < <(printf '%s\n' "${subject_ids[@]}" | sort -n -u)

  [[ -n "$joined" ]] || return 1
  printf '%s' "$joined"
}

KEYPOINTS_ROOT=""
UPFALL_ROOT=""
POSE_BACKEND="${POSE_BACKEND:-yolo}"
# yolo
MODEL_PATH=""
# alphapose
ALPHAPOSE_ROOT=""
FASTPOSE_WEIGHTS=""
DETECTOR_WEIGHTS_AP=""
CFG_PATH=""
DETECTOR_CFG=""
# vitpose
DETECTOR_MODEL=""
POSE_MODEL=""
SUBJECTS="${SUBJECTS:-}"
LOCK_SETTINGS="${LOCK_SETTINGS:-strict_lock}"
REPORT_DIR="${REPORT_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keypoints-root)
      require_value "$1" "${2:-}"
      KEYPOINTS_ROOT="$2"
      shift 2
      ;;
    --upfall-root)
      require_value "$1" "${2:-}"
      UPFALL_ROOT="$2"
      shift 2
      ;;
    --pose-backend)
      require_value "$1" "${2:-}"
      POSE_BACKEND="$2"
      shift 2
      ;;
    --model-path)
      require_value "$1" "${2:-}"
      MODEL_PATH="$2"
      shift 2
      ;;
    --alphapose-root)
      require_value "$1" "${2:-}"
      ALPHAPOSE_ROOT="$2"
      shift 2
      ;;
    --fastpose-weights)
      require_value "$1" "${2:-}"
      FASTPOSE_WEIGHTS="$2"
      shift 2
      ;;
    --detector-weights)
      require_value "$1" "${2:-}"
      DETECTOR_WEIGHTS_AP="$2"
      shift 2
      ;;
    --cfg-path)
      require_value "$1" "${2:-}"
      CFG_PATH="$2"
      shift 2
      ;;
    --detector-cfg)
      require_value "$1" "${2:-}"
      DETECTOR_CFG="$2"
      shift 2
      ;;
    --detector-model)
      require_value "$1" "${2:-}"
      DETECTOR_MODEL="$2"
      shift 2
      ;;
    --pose-model)
      require_value "$1" "${2:-}"
      POSE_MODEL="$2"
      shift 2
      ;;
    --subjects)
      require_value "$1" "${2:-}"
      SUBJECTS="$2"
      shift 2
      ;;
    --lock-settings)
      require_value "$1" "${2:-}"
      LOCK_SETTINGS="$2"
      shift 2
      ;;
    --report-dir)
      require_value "$1" "${2:-}"
      REPORT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unrecognized argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$KEYPOINTS_ROOT" || -z "$UPFALL_ROOT" ]]; then
  echo "ERROR: --keypoints-root and --upfall-root are required." >&2
  usage >&2
  exit 1
fi

case "$POSE_BACKEND" in
  yolo)
    if [[ -z "$MODEL_PATH" ]]; then
      echo "ERROR: --model-path is required when --pose-backend is yolo." >&2
      usage >&2
      exit 1
    fi
    ;;
  alphapose)
    if [[ -z "$ALPHAPOSE_ROOT" || -z "$FASTPOSE_WEIGHTS" || -z "$DETECTOR_WEIGHTS_AP" ]]; then
      echo "ERROR: --alphapose-root, --fastpose-weights, and --detector-weights are required when --pose-backend is alphapose." >&2
      usage >&2
      exit 1
    fi
    ;;
  vitpose)
    # detector-model and pose-model have defaults inside the Python script; no hard requirement here.
    ;;
  *)
    echo "ERROR: Unsupported --pose-backend: $POSE_BACKEND. Choose from: yolo, alphapose, vitpose." >&2
    usage >&2
    exit 1
    ;;
esac

validate_lock_settings "$LOCK_SETTINGS"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
export PROTOTYPE_DIR

if [[ ! -d "$PROTOTYPE_DIR" ]]; then
  echo "ERROR: Prototype directory not found: $PROTOTYPE_DIR" >&2
  exit 1
fi

if [[ "$POSE_BACKEND" == "yolo" && ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: Pose model weights not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ "$POSE_BACKEND" == "alphapose" && ! -d "$ALPHAPOSE_ROOT" ]]; then
  echo "ERROR: AlphaPose root not found: $ALPHAPOSE_ROOT" >&2
  exit 1
fi

if [[ ! -d "$KEYPOINTS_ROOT" ]]; then
  echo "ERROR: Keypoints root not found: $KEYPOINTS_ROOT" >&2
  exit 1
fi

if [[ ! -d "$UPFALL_ROOT" ]]; then
  echo "ERROR: UP-Fall root not found: $UPFALL_ROOT" >&2
  exit 1
fi

if [[ -z "$REPORT_DIR" ]]; then
  REPORT_DIR="${KEYPOINTS_ROOT}/_fix_bad_keypoints_reports"
fi

CAMERA1_REPORT_PATH="${CAMERA1_REPORT_PATH:-${REPORT_DIR}/camera1_background_report.csv}"
BAD_REPORT_PATH="${BAD_REPORT_PATH:-${REPORT_DIR}/bad_keypoints_report.csv}"
EMPTY_REPORT_PATH="${EMPTY_REPORT_PATH:-${REPORT_DIR}/empty_windows_report.csv}"
PYTHON_BIN="$(resolve_python_bin)"

mkdir -p -- "$(dirname -- "$CAMERA1_REPORT_PATH")"
mkdir -p -- "$(dirname -- "$BAD_REPORT_PATH")"
mkdir -p -- "$(dirname -- "$EMPTY_REPORT_PATH")"

SUBJECT_SOURCE="cli"
if [[ -z "$SUBJECTS" ]]; then
  if SUBJECTS="$(discover_subjects "$KEYPOINTS_ROOT")"; then
    SUBJECT_SOURCE="keypoints-root"
  elif SUBJECTS="$(discover_subjects "$UPFALL_ROOT")"; then
    SUBJECT_SOURCE="upfall-root"
  else
    echo "ERROR: Could not infer subjects from $KEYPOINTS_ROOT or $UPFALL_ROOT. Pass --subjects explicitly." >&2
    exit 1
  fi
fi

cd "$PROTOTYPE_DIR"

log "Starting bad keypoint cleanup pipeline"
log "Prototype root: ${PROTOTYPE_DIR}"
log "Keypoints root: ${KEYPOINTS_ROOT}"
log "UP-Fall root:   ${UPFALL_ROOT}"
log "Pose backend:   ${POSE_BACKEND}"
case "$POSE_BACKEND" in
  yolo)      log "Model path:     ${MODEL_PATH}" ;;
  alphapose) log "AlphaPose root: ${ALPHAPOSE_ROOT}" ; log "FastPose:       ${FASTPOSE_WEIGHTS}" ;;
  vitpose)   log "Detector model: ${DETECTOR_MODEL:-<default>}" ; log "Pose model:     ${POSE_MODEL:-<default>}" ;;
esac
log "Subjects:       ${SUBJECTS} (${SUBJECT_SOURCE})"
log "Lock settings:  ${LOCK_SETTINGS}"
log "Report dir:     ${REPORT_DIR}"

# Build the base regeneration command for the selected backend.
# Usage: regen_cmd <camera> [extra-flags...]
regen_cmd() {
  local camera="$1"
  shift
  local -a cmd
  case "$POSE_BACKEND" in
    yolo)
      cmd=(
        "$PYTHON_BIN" -m dataset_helpers.get_keypoints_files
        --subjects "$SUBJECTS"
        --camera "$camera"
        --lock-settings "$LOCK_SETTINGS"
        --upfall-root "$UPFALL_ROOT"
        --output-root "$KEYPOINTS_ROOT"
        --model-path "$MODEL_PATH"
      )
      ;;
    alphapose)
      cmd=(
        "$PYTHON_BIN" -m dataset_helpers.get_keypoints_files_alphapose
        --subjects "$SUBJECTS"
        --camera "$camera"
        --lock-settings "$LOCK_SETTINGS"
        --upfall-root "$UPFALL_ROOT"
        --output-root "$KEYPOINTS_ROOT"
        --alphapose-root "$ALPHAPOSE_ROOT"
        --fastpose-weights "$FASTPOSE_WEIGHTS"
        --detector-weights "$DETECTOR_WEIGHTS_AP"
      )
      if [[ -n "$CFG_PATH" ]]; then
        cmd+=(--cfg-path "$CFG_PATH")
      fi
      if [[ -n "$DETECTOR_CFG" ]]; then
        cmd+=(--detector-cfg "$DETECTOR_CFG")
      fi
      ;;
    vitpose)
      cmd=(
        "$PYTHON_BIN" -m dataset_helpers.get_keypoints_files_ViTpose
        --subjects "$SUBJECTS"
        --camera "$camera"
        --lock-settings "$LOCK_SETTINGS"
        --upfall-root "$UPFALL_ROOT"
        --output-root "$KEYPOINTS_ROOT"
      )
      if [[ -n "$DETECTOR_MODEL" ]]; then
        cmd+=(--detector-model "$DETECTOR_MODEL")
      fi
      if [[ -n "$POSE_MODEL" ]]; then
        cmd+=(--pose-model "$POSE_MODEL")
      fi
      ;;
  esac
  run_cmd "${cmd[@]}" "$@"
}

log "Step 1/10: Scan Camera1 for background/reflection switches."
remove_stale_report "$CAMERA1_REPORT_PATH"
run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/check_camera1_background_switches.py" \
  --root "$KEYPOINTS_ROOT" \
  --output "$CAMERA1_REPORT_PATH"

log "Step 2/10: Delete reported Camera1 entries."
delete_if_report_exists \
  "$CAMERA1_REPORT_PATH" \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/delete_reported_entries.py" \
  "$CAMERA1_REPORT_PATH" \
  --root "$KEYPOINTS_ROOT" \
  --execute

log "Step 3/10: Regenerate Camera1 keypoints."
regen_cmd 1

log "Step 4/10: Scan Camera2 for bad keypoint tracks."
remove_stale_report "$BAD_REPORT_PATH"
run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/check_keypoints_tracking.py" \
  --root "$KEYPOINTS_ROOT" \
  --output "$BAD_REPORT_PATH"

log "Step 5/10: Delete reported Camera2 bad keypoint entries."
delete_if_report_exists \
  "$BAD_REPORT_PATH" \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/delete_reported_entries.py" \
  "$BAD_REPORT_PATH" \
  --root "$KEYPOINTS_ROOT" \
  --execute

log "Step 6/10: Regenerate Camera2 keypoints."
regen_cmd 2 --no-suspicious

log "Step 7/10: Scan Camera2 for empty windows."
remove_stale_report "$EMPTY_REPORT_PATH"
run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/check_keypoints_empty_windows.py" \
  --root "$KEYPOINTS_ROOT" \
  --camera-dirs Camera2 \
  --output "$EMPTY_REPORT_PATH"

log "Step 8/10: Delete reported Camera2 empty-window entries."
delete_if_report_exists \
  "$EMPTY_REPORT_PATH" \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/delete_reported_entries.py" \
  "$EMPTY_REPORT_PATH" \
  --root "$KEYPOINTS_ROOT" \
  --execute

log "Step 9/10: Regenerate Camera2 keypoints again with --allow-region2-start."
regen_cmd 2 --no-suspicious --allow-region2-start

log "Step 10/10: Re-scan Camera2 empty windows, delete remaining empties, then fix bad labels."
remove_stale_report "$EMPTY_REPORT_PATH"
run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/check_keypoints_empty_windows.py" \
  --root "$KEYPOINTS_ROOT" \
  --camera-dirs Camera2 \
  --output "$EMPTY_REPORT_PATH"

delete_if_report_exists \
  "$EMPTY_REPORT_PATH" \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/delete_reported_entries.py" \
  "$EMPTY_REPORT_PATH" \
  --root "$KEYPOINTS_ROOT" \
  --execute

run_cmd \
  "$PYTHON_BIN" "$PROTOTYPE_DIR/dataset_helpers/fix_label_20_to_2.py" \
  --root "$KEYPOINTS_ROOT"

log "Pipeline complete."
