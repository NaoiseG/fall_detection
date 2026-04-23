#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Usage:
  ./dataset_helpers/get_img_downsized_keypoints.sh [options]

Generates and fixes downsized YOLO11 UP-Fall keypoints for:
  models: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
  sizes:  576, 512, 448

Defaults match the scratch layout:
  --upfall-root ../../../scratch/UPFall
  --output-base ../../../scratch/keypoints/downsized_keypoints
  --model-root pose_models/ultralytics

Output layout:
  <output-base>/yolo11n/base_576
  <output-base>/yolo11n/base_512
  <output-base>/yolo11n/base_448
  ...

Options:
  --subjects SPEC     Subject list/range to process (default: 16-17).
  --upfall-root PATH  UP-Fall dataset root (default: ../../../scratch/UPFall).
  --output-base PATH  Base output directory (default: ../../../scratch/keypoints/downsized_keypoints).
  --model-root PATH   Directory containing yolo11*-pose.pt files (default: pose_models/ultralytics).
  --sizes LIST        Comma/space separated image sizes (default: 576,512,448).
  --models LIST       Comma/space separated YOLO suffixes from n,s,m,l,x (default: n,s,m,l,x).
  --force-fix         Re-run the fix pipeline even when a fixed marker exists.
  --dry-run           Print commands without running them or writing markers.
  -h, --help          Show this help text.

Resume behavior:
  - Native generation is safe to rerun because get_keypoints_files skips existing keypoints.npz files.
  - Each model/size writes .fixed.ok only after fix_bad_keypoints.sh completes successfully.
  - Reruns skip model/size pairs whose expected outputs exist and whose .fixed.ok marker exists.
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

join_by_space() {
  local IFS=' '
  printf '%s' "$*"
}

run_cmd() {
  log "CMD: $(quote_cmd "$@")"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
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

parse_subjects() {
  local spec="$1"
  local -a chunks=()
  local -a parsed=()
  local chunk start end subject_id
  local IFS=','

  read -r -a chunks <<< "$spec"
  for chunk in "${chunks[@]}"; do
    chunk="${chunk//[[:space:]]/}"
    [[ -n "$chunk" ]] || continue

    if [[ "$chunk" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start=$((10#${BASH_REMATCH[1]}))
      end=$((10#${BASH_REMATCH[2]}))
      if (( start <= 0 || end <= 0 || end < start )); then
        echo "ERROR: Invalid subject range: $chunk" >&2
        exit 1
      fi
      for ((subject_id = start; subject_id <= end; subject_id++)); do
        parsed+=("$subject_id")
      done
      continue
    fi

    if [[ "$chunk" =~ ^[0-9]+$ ]] && (( 10#$chunk > 0 )); then
      parsed+=("$((10#$chunk))")
      continue
    fi

    echo "ERROR: Invalid subject spec: $chunk" >&2
    exit 1
  done

  if [[ ${#parsed[@]} -eq 0 ]]; then
    echo "ERROR: Subjects cannot be empty." >&2
    exit 1
  fi

  SUBJECT_IDS=()
  while IFS= read -r subject_id; do
    [[ -n "$subject_id" ]] || continue
    SUBJECT_IDS+=("$subject_id")
  done < <(printf '%s\n' "${parsed[@]}" | sort -n -u)
}

parse_sizes() {
  local spec="${1//,/ }"
  local -a parsed=()
  local size
  local IFS=' '

  read -r -a parsed <<< "$spec"
  SIZES=()
  for size in "${parsed[@]}"; do
    [[ -n "$size" ]] || continue
    if [[ ! "$size" =~ ^[0-9]+$ ]] || (( 10#$size <= 0 )); then
      echo "ERROR: Invalid image size: $size" >&2
      exit 1
    fi
    SIZES+=("$((10#$size))")
  done

  if [[ ${#SIZES[@]} -eq 0 ]]; then
    echo "ERROR: Size list cannot be empty." >&2
    exit 1
  fi
}

parse_model_modes() {
  local spec="${1//,/ }"
  local -a parsed=()
  local mode
  local IFS=' '

  read -r -a parsed <<< "$spec"
  MODEL_MODES=()
  for mode in "${parsed[@]}"; do
    [[ -n "$mode" ]] || continue
    if [[ ! "$mode" =~ ^[nsmlx]$ ]]; then
      echo "ERROR: Invalid YOLO model suffix: $mode. Use values from n,s,m,l,x." >&2
      exit 1
    fi
    MODEL_MODES+=("$mode")
  done

  if [[ ${#MODEL_MODES[@]} -eq 0 ]]; then
    echo "ERROR: Model list cannot be empty." >&2
    exit 1
  fi
}

find_expected_npz_paths() {
  local upfall_root="$1"
  local camera="$2"
  local subject_id subject_root frames_dir trial_dir rel_path

  for subject_id in "${SUBJECT_IDS[@]}"; do
    subject_root="${upfall_root}/Subject${subject_id}"
    [[ -d "$subject_root" ]] || continue

    while IFS= read -r -d '' frames_dir; do
      trial_dir="$(dirname -- "$frames_dir")"
      if compgen -G "${trial_dir}/*Features1&0.5.csv" >/dev/null; then
        rel_path="${frames_dir#${upfall_root}/}"
        printf '%s/keypoints.npz\n' "$rel_path"
      fi
    done < <(find "$subject_root" -type d -name "*Camera${camera}" -print0 2>/dev/null)
  done
}

completion_counts() {
  local output_root="$1"
  local expected=0
  local existing=0
  local missing=0
  local rel_path

  while IFS= read -r rel_path; do
    [[ -n "$rel_path" ]] || continue
    expected=$((expected + 1))
    if [[ -f "${output_root}/${rel_path}" ]]; then
      existing=$((existing + 1))
    else
      missing=$((missing + 1))
    fi
  done < <(
    find_expected_npz_paths "$UPFALL_ROOT" 1
    find_expected_npz_paths "$UPFALL_ROOT" 2
  )

  printf '%s\t%s\t%s\n' "$expected" "$existing" "$missing"
}

generate_camera() {
  local camera="$1"
  local lock_settings="$2"
  local model_path="$3"
  local output_root="$4"
  local imgsz="$5"

  run_cmd \
    "$PYTHON_BIN" -m dataset_helpers.get_keypoints_files \
    --subjects "$SUBJECTS" \
    --camera "$camera" \
    --lock-settings "$lock_settings" \
    --upfall-root "$UPFALL_ROOT" \
    --output-root "$output_root" \
    --model-path "$model_path" \
    --imgsz "$imgsz"
}

write_fixed_marker() {
  local marker_path="$1"
  local model_name="$2"
  local imgsz="$3"
  local model_path="$4"

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  {
    printf 'completed_at=%s\n' "$(timestamp)"
    printf 'subjects=%s\n' "$SUBJECTS"
    printf 'model=%s\n' "$model_name"
    printf 'model_path=%s\n' "$model_path"
    printf 'imgsz=%s\n' "$imgsz"
    printf 'upfall_root=%s\n' "$UPFALL_ROOT"
  } > "$marker_path"
}

run_combo() {
  local mode="$1"
  local imgsz="$2"
  local run_idx="$3"
  local total_runs="$4"
  local model_name="yolo11${mode}"
  local model_path="${MODEL_ROOT}/${model_name}-pose.pt"
  local output_root="${OUTPUT_BASE}/${model_name}/base_${imgsz}"
  local marker_path="${output_root}/.fixed.ok"
  local expected existing missing
  local expected_after existing_after missing_after

  log "[$run_idx/$total_runs] ${model_name} imgsz=${imgsz}"
  log "Output root: ${output_root}"

  if [[ ! -f "$model_path" ]]; then
    echo "ERROR: Pose model weights not found: $model_path" >&2
    exit 1
  fi

  read -r expected existing missing <<< "$(completion_counts "$output_root")"
  log "Current outputs: ${existing}/${expected} expected keypoints present (${missing} missing)."

  if [[ "$expected" -gt 0 && "$missing" -eq 0 && -f "$marker_path" && "$FORCE_FIX" == "0" ]]; then
    log "Already generated and fixed; skipping ${model_name} base_${imgsz}."
    return 0
  fi

  if [[ "$DRY_RUN" == "0" ]]; then
    mkdir -p -- "$output_root"
  fi

  if [[ "$expected" -eq 0 || "$missing" -ne 0 ]]; then
    if [[ -f "$marker_path" && "$DRY_RUN" == "0" ]]; then
      rm -f -- "$marker_path"
    fi

    log "Generating missing native keypoints for ${model_name} base_${imgsz}."
    generate_camera 2 strict_lock "$model_path" "$output_root" "$imgsz"
    generate_camera 1 default "$model_path" "$output_root" "$imgsz"
  else
    log "Native keypoint generation already complete; skipping generation commands."
  fi

  read -r expected_after existing_after missing_after <<< "$(completion_counts "$output_root")"
  log "After generation: ${existing_after}/${expected_after} expected keypoints present (${missing_after} missing)."

  if [[ "$DRY_RUN" == "0" && ( "$expected_after" -eq 0 || "$missing_after" -ne 0 ) ]]; then
    echo "ERROR: Generation incomplete for ${model_name} base_${imgsz}." >&2
    exit 1
  fi

  if [[ -f "$marker_path" && "$FORCE_FIX" == "0" ]]; then
    log "Fix marker exists; skipping cleanup for ${model_name} base_${imgsz}."
    return 0
  fi

  log "Running fix_bad_keypoints.sh for ${model_name} base_${imgsz} with imgsz=${imgsz}."
  run_cmd \
    "$BASH_BIN" "$FIX_SCRIPT" \
    --keypoints-root "$output_root" \
    --upfall-root "$UPFALL_ROOT" \
    --pose-backend yolo \
    --model-path "$model_path" \
    --imgsz "$imgsz" \
    --subjects "$SUBJECTS" \
    --camera1-lock-settings default \
    --camera2-lock-settings strict_lock

  write_fixed_marker "$marker_path" "$model_name" "$imgsz" "$model_path"
  log "Finished ${model_name} base_${imgsz}."
}

SUBJECTS="${SUBJECTS:-16-17}"
UPFALL_ROOT="${UPFALL_ROOT:-../../../scratch/UPFall}"
OUTPUT_BASE="${OUTPUT_BASE:-../../../scratch/keypoints/downsized_keypoints}"
MODEL_ROOT="${MODEL_ROOT:-pose_models/ultralytics}"
SIZES_SPEC="${SIZES_SPEC:-576,512,448}"
MODEL_MODES_SPEC="${MODEL_MODES_SPEC:-n,s,m,l,x}"
DRY_RUN=0
FORCE_FIX=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subjects)
      require_value "$1" "${2:-}"
      SUBJECTS="$2"
      shift 2
      ;;
    --upfall-root)
      require_value "$1" "${2:-}"
      UPFALL_ROOT="$2"
      shift 2
      ;;
    --output-base)
      require_value "$1" "${2:-}"
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --model-root)
      require_value "$1" "${2:-}"
      MODEL_ROOT="$2"
      shift 2
      ;;
    --sizes)
      require_value "$1" "${2:-}"
      SIZES_SPEC="$2"
      shift 2
      ;;
    --models|--modes)
      require_value "$1" "${2:-}"
      MODEL_MODES_SPEC="$2"
      shift 2
      ;;
    --force-fix)
      FORCE_FIX=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
FIX_SCRIPT="${SCRIPT_DIR}/fix_bad_keypoints.sh"
PYTHON_BIN="$(resolve_python_bin)"
BASH_BIN="${BASH:-bash}"

parse_subjects "$SUBJECTS"
parse_sizes "$SIZES_SPEC"
parse_model_modes "$MODEL_MODES_SPEC"

cd "$PROTOTYPE_DIR"

UPFALL_ROOT="${UPFALL_ROOT%/}"
OUTPUT_BASE="${OUTPUT_BASE%/}"
MODEL_ROOT="${MODEL_ROOT%/}"

if [[ ! -d "$UPFALL_ROOT" ]]; then
  echo "ERROR: UP-Fall root not found: $UPFALL_ROOT" >&2
  exit 1
fi

if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "ERROR: Model root not found: $MODEL_ROOT" >&2
  exit 1
fi

if [[ ! -f "$FIX_SCRIPT" ]]; then
  echo "ERROR: fix script not found: $FIX_SCRIPT" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "0" ]]; then
  mkdir -p -- "$OUTPUT_BASE"
fi

log "Starting downsized YOLO11 keypoint sweep"
log "Prototype root: ${PROTOTYPE_DIR}"
log "Python:         ${PYTHON_BIN}"
log "Bash:           ${BASH_BIN}"
log "Subjects:       ${SUBJECTS}"
log "UP-Fall root:   ${UPFALL_ROOT}"
log "Model root:     ${MODEL_ROOT}"
log "Output base:    ${OUTPUT_BASE}"
log "Models:         $(join_by_space "${MODEL_MODES[@]}")"
log "Image sizes:    $(join_by_space "${SIZES[@]}")"
log "Dry run:        ${DRY_RUN}"
log "Force fix:      ${FORCE_FIX}"

total_runs=$(( ${#MODEL_MODES[@]} * ${#SIZES[@]} ))
run_idx=0

for mode in "${MODEL_MODES[@]}"; do
  for imgsz in "${SIZES[@]}"; do
    run_idx=$((run_idx + 1))
    run_combo "$mode" "$imgsz" "$run_idx" "$total_runs"
  done
done

log "Downsized YOLO11 keypoint sweep complete."
