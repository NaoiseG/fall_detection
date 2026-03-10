#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./export_models_int8.sh [ROOT_DIR] [CALIB_DIR]
#
# Examples:
#   ./export_models_int8.sh models/ultralytics
#   ./export_models_int8.sh models/ultralytics calibration_dataset_upfall
#
# Result:
#   For each model like:
#     models/ultralytics/yolo11l-pose/yolo11l-pose.pt
#   it creates:
#     models/ultralytics/yolo11l-pose/yolo11l-pose_int8.engine
#
# Notes:
# - Before each export, this script:
#   1) swaps calibration_dataset_upfall/labels/val to the matching model-specific val set
#   2) deletes Ultralytics val.cache
#   3) deletes any existing TensorRT calibration cache files it can find in the model directory
# - After export, it renames:
#     yolo11l-pose.engine -> yolo11l-pose_int8.engine

ROOT_DIR="${1:-.}"
CALIB_DIR="${2:-./calibration_dataset_upfall}"

# Resolve to absolute paths before any cd
ROOT_DIR="$(realpath "$ROOT_DIR")"
CALIB_DIR="$(realpath "$CALIB_DIR")"

DATA_YAML="${CALIB_DIR}/data.yaml"
MAIN_LABELS_DIR="${CALIB_DIR}/labels"
MAIN_VAL_DIR="${MAIN_LABELS_DIR}/val"
VAL_CACHE_FILE="${MAIN_LABELS_DIR}/val.cache"
BACKUP_VAL_DIR="${MAIN_LABELS_DIR}/val.__backup_export_int8__"

cleanup() {
  if [[ -d "$BACKUP_VAL_DIR" ]]; then
    rm -rf "$MAIN_VAL_DIR"
    mv "$BACKUP_VAL_DIR" "$MAIN_VAL_DIR"
  fi
}
trap cleanup EXIT

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 1
fi

if [[ ! -d "$CALIB_DIR" ]]; then
  echo "ERROR: CALIB_DIR not found: $CALIB_DIR" >&2
  exit 1
fi

if [[ ! -f "$DATA_YAML" ]]; then
  echo "ERROR: data.yaml not found at: $DATA_YAML" >&2
  exit 1
fi

if [[ ! -d "$MAIN_LABELS_DIR" ]]; then
  echo "ERROR: labels directory not found at: $MAIN_LABELS_DIR" >&2
  exit 1
fi

if [[ ! -d "$MAIN_VAL_DIR" ]]; then
  echo "ERROR: main val directory not found at: $MAIN_VAL_DIR" >&2
  exit 1
fi

rm -rf "$BACKUP_VAL_DIR"
cp -a "$MAIN_VAL_DIR" "$BACKUP_VAL_DIR"

find "$ROOT_DIR" -type f -name "*.pt" -print0 | while IFS= read -r -d '' pt; do
  dir="$(dirname "$pt")"
  base="$(basename "$pt" .pt)"
  model_family="${base%%-*}"

  # Example:
  # yolo11n-pose.pt -> calibration_dataset_upfall/labels/yolo11n/labels/val
  model_val_dir="${MAIN_LABELS_DIR}/${model_family}/labels/val"

  final_engine="${dir}/${base}_int8.engine"
  exported_engine="${dir}/${base}.engine"
  onnx_file="${dir}/${base}.onnx"

  if [[ ! -d "$model_val_dir" ]]; then
    echo "WARNING: Skipping $pt" >&2
    echo "         Expected model-specific val dir not found: $model_val_dir" >&2
    continue
  fi

  echo "==> Preparing calibration labels for: $pt"

  # Replace shared val dir with model-specific one
  rm -rf "$MAIN_VAL_DIR"
  cp -a "$model_val_dir" "$MAIN_VAL_DIR"

  # Remove Ultralytics dataset cache so swapped labels are re-read
  rm -f "$VAL_CACHE_FILE"

  # Remove previous output files so we only pick up fresh ones
  rm -f "$exported_engine" "$final_engine" "$onnx_file"

  # Remove common TensorRT calibration cache files from this model directory
  find "$dir" -maxdepth 1 -type f \
    \( -iname "*.cache" -o -iname "calibration.cache" -o -iname "calib.cache" \) \
    -print -delete || true

  echo "==> Exporting: $pt"
  (
    cd "$dir"
    yolo export \
      model="./${base}.pt" \
      format=engine \
      project=. \
      int8=True \
      data="$DATA_YAML"
  )

  if [[ ! -f "$exported_engine" ]]; then
    echo "ERROR: Expected engine not found after export: $exported_engine" >&2
    exit 1
  fi

   -f "$exported_engine" "$final_engine"

  echo "==> Final engine: $final_engine"
done

echo "Done."
