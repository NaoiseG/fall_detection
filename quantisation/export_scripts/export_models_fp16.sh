#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./export_all_fp16_engines.sh [ROOT_DIR]
#
# Example:
#   ./export_all_fp16_engines.sh models/ultralytics
#
# Result:
#   For each model like:
#     models/ultralytics/yolo11l-pose/yolo11l-pose.pt
#   it creates:
#     models/ultralytics/yolo11l-pose/yolo11l-pose_fp16.engine
#
# If ROOT_DIR is not provided, it uses the current directory.

ROOT_DIR="${1:-.}"
ROOT_DIR="$(realpath "$ROOT_DIR")"

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "ERROR: ROOT_DIR not found: $ROOT_DIR" >&2
  exit 1
fi

find "$ROOT_DIR" -type f -name "*.pt" -print0 | while IFS= read -r -d '' pt; do
  dir="$(dirname "$pt")"
  base="$(basename "$pt" .pt)"

  exported_engine="${dir}/${base}.engine"
  final_engine="${dir}/${base}_fp16.engine"
  onnx_file="${dir}/${base}.onnx"

  echo "==> Exporting: $pt"

  rm -f "$exported_engine" "$final_engine" "$onnx_file"

  (
    cd "$dir"
    yolo export \
      model="./${base}.pt" \
      format=engine \
      half=True \
      project=. \
      name="${base}_fp16_export" \
      exist_ok=True
  )

  if [[ ! -f "$exported_engine" ]]; then
    echo "ERROR: Expected engine not found after export: $exported_engine" >&2
    exit 1
  fi

  mv -f "$exported_engine" "$final_engine"

  echo "==> Final engine: $final_engine"
done

echo "Done."
