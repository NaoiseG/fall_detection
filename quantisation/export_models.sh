#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./export_all_fp16_engines.sh [ROOT_DIR]
# Example:
#   ./export_all_fp16_engines.sh models/ultralytics
#
# If ROOT_DIR is not provided, it uses the current directory.

ROOT_DIR="${1:-.}"

# Find all .pt files under ROOT_DIR (recursively), and export each to TensorRT FP16 engine
# in the same directory as the .pt file, using Ultralytics' project/name arguments.
find "$ROOT_DIR" -type f -name "*.pt" -print0 | while IFS= read -r -d '' pt; do
  dir="$(dirname "$pt")"
  base="$(basename "$pt" .pt)"

  echo "==> Exporting: $pt"
  (
    cd "$dir"
    # This creates output under: ./${base}_fp16/
    # and the engine will be in that folder (same directory tree as the .pt).
    yolo export model="${base}.pt" format=engine half=True project=. name="${base}_fp16" exist_ok=True
  )
done

echo "Done."
