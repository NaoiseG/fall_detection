#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s' "${PYTHON_BIN}"
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

PYTHON_BIN="$(resolve_python_bin)"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/eval_downsized_models.py" "$@"
