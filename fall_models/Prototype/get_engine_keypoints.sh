#!/usr/bin/env bash

# Generate UP-Fall keypoint .npz files for subjects 16-17 using YOLO11 pose
# TensorRT engines, then transfer them to the remote HPC account with rsync.
#
# Resumable behavior:
# - If the remote destination already contains any .npz file, skip that model/precision.
# - Else if the local output root already contains any .npz file, do not regenerate; just transfer it.
# - Else generate for camera 1 and then camera 2 into the same local output root.
# - Only after a successful transfer do we remove that specific local output root.
#
# Save this script inside:
#   ~/NaoiseG/fall_detection/fall_models/Prototype
#
# Then run it from there, or from anywhere after making it executable.
# Relative paths are resolved from the script's own directory.

set -u
set -o pipefail
IFS=$'\n\t'

###############################################################################
# User-configurable settings
###############################################################################

# Set to 1 to preview rsync without actually transferring files.
# In dry-run mode, local cleanup is skipped.
DRY_RUN=0

# Remote SSH target
SSH_TARGET="21376026@sonic.ucd.ie"

# Remote roots
# - REMOTE_ROOT_RSYNC is used in rsync destinations
# - REMOTE_ROOT_HOME is used inside ssh commands on the remote shell
REMOTE_ROOT_RSYNC="~/scratch/keypoints/UPFall_keypoints"
REMOTE_ROOT_HOME='$HOME/scratch/keypoints/UPFall_keypoints'

# Local/project-relative paths
UPFALL_ROOT="../../Datasets/UPFall"
MODELS_ROOT="../../quantisation/models/ultralytics"
DATASET_MODULE="dataset_helpers.get_keypoints_files"

# Model combinations
SIZES=(n s m l x)
PRECISIONS=(fp16 int8)
CAMERAS=(1 2)

###############################################################################
# Script setup
###############################################################################

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || {
  echo "ERROR: Failed to cd to script directory: $SCRIPT_DIR" >&2
  exit 1
}

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p -- "$LOG_DIR"
LOG_FILE="${LOG_DIR}/upfall_keypoints_$(date +%Y%m%d_%H%M%S).log"

# Summary buckets
SKIPPED_REMOTE=()
TRANSFERRED_LOCAL=()
GENERATED_AND_TRANSFERRED=()
SKIPPED_MODEL_MISSING=()
FAILURES=()

###############################################################################
# Helper functions
###############################################################################

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log() {
  local message="$*"
  printf '[%s] %s\n' "$(timestamp)" "$message" | tee -a "$LOG_FILE"
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

build_model_path() {
  local size="$1"
  local precision="$2"
  printf '%s/yolo11%s-pose/yolo11%s-pose_%s.engine' \
    "$MODELS_ROOT" "$size" "$size" "$precision"
}

build_local_output_root() {
  local precision="$1"
  local size="$2"
  printf '../../Datasets/UPFall_keypoints/outputs_npz_%s_%s' \
    "$precision" "$size"
}

build_remote_dir_home() {
  local size="$1"
  local precision="$2"
  printf '%s/yolo11%s/%s' "$REMOTE_ROOT_HOME" "$size" "$precision"
}

build_remote_dest_rsync() {
  local size="$1"
  local precision="$2"
  printf '%s:%s/yolo11%s/%s/' \
    "$SSH_TARGET" "$REMOTE_ROOT_RSYNC" "$size" "$precision"
}

combo_label() {
  local size="$1"
  local precision="$2"
  printf 'yolo11%s/%s' "$size" "$precision"
}

run_and_log() {
  local cmd_str
  cmd_str="$(quote_cmd "$@")"
  log "CMD: $cmd_str"

  "$@" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}

  log "CMD EXIT: $rc"
  return "$rc"
}

remote_has_files() {
  local remote_dir_home="$1"

  # Return codes:
  # 0 => remote has at least one .npz
  # 1 => remote has no .npz (or directory does not exist)
  # 2 => ssh/check failure
  local remote_cmd
  remote_cmd="if [ -d \"$remote_dir_home\" ] && find \"$remote_dir_home\" -type f -name '*.npz' -print -quit 2>/dev/null | grep -q .; then exit 0; else exit 1; fi"

  log "CMD: $(quote_cmd ssh "$SSH_TARGET" "$remote_cmd")"
  ssh "$SSH_TARGET" "$remote_cmd" 2>&1 | tee -a "$LOG_FILE"
  local rc=${PIPESTATUS[0]}
  log "CMD EXIT: $rc"

  case "$rc" in
    0) return 0 ;;
    1) return 1 ;;
    *) return 2 ;;
  esac
}

local_has_files() {
  local local_root="$1"

  # Return codes:
  # 0 => local has at least one .npz
  # 1 => local has no .npz (or directory does not exist)
  # 2 => check failure
  if [[ ! -d "$local_root" ]]; then
    log "Local output root does not exist yet: \"$local_root\""
    return 1
  fi

  log "CMD: find $(printf '%q' "$local_root") -type f -name '*.npz' -print -quit 2>/dev/null | grep -q ."
  if find "$local_root" -type f -name '*.npz' -print -quit 2>/dev/null | grep -q .; then
    log "CMD EXIT: 0"
    return 0
  else
    local rc=$?
    log "CMD EXIT: $rc"
    case "$rc" in
      1) return 1 ;;
      *) return 2 ;;
    esac
  fi
}

run_generation() {
  local camera="$1"
  local model_path="$2"
  local output_root="$3"

  mkdir -p -- "$output_root" || {
    log "ERROR: Failed to create local output root: \"$output_root\""
    return 1
  }

  log "Generation needed for camera ${camera}."
  run_and_log \
    python -m "$DATASET_MODULE" \
      --subjects 16-17 \
      --camera "$camera" \
      --upfall-root "$UPFALL_ROOT" \
      --output-root "$output_root" \
      --model-path "$model_path"
}

sync_to_remote() {
  local local_root="$1"
  local remote_dir_home="$2"
  local remote_dest="$3"

  log "Ensuring remote destination directory exists."
  if ! run_and_log ssh "$SSH_TARGET" "mkdir -p \"$remote_dir_home\""; then
    log "ERROR: Failed to create remote directory."
    return 1
  fi

  local rsync_opts=(-avh --progress --partial)
  if [[ "$DRY_RUN" -eq 1 ]]; then
    rsync_opts+=(--dry-run)
  fi

  log "Starting rsync transfer."
  run_and_log rsync "${rsync_opts[@]}" -- "${local_root}/" "$remote_dest"
}

cleanup_local() {
  local local_root="$1"

  # Safety guard: only remove expected local output roots for this workflow.
  case "$local_root" in
    ../../Datasets/UPFall_keypoints/outputs_npz_*)
      ;;
    *)
      log "ERROR: Refusing to delete unexpected path: \"$local_root\""
      return 1
      ;;
  esac

  if [[ ! -e "$local_root" ]]; then
    log "Local output root already absent, nothing to clean: \"$local_root\""
    return 0
  fi

  log "Cleaning up local output root after successful transfer."
  run_and_log rm -rf -- "$local_root"
}

print_summary_section() {
  local title="$1"
  local array_name="$2"
  declare -n arr_ref="$array_name"

  log "$title: ${#arr_ref[@]}"
  if [[ ${#arr_ref[@]} -eq 0 ]]; then
    log "  (none)"
    return
  fi

  local item
  for item in "${arr_ref[@]}"; do
    log "  - $item"
  done
}

###############################################################################
# Main processing loop
###############################################################################

log "Starting UP-Fall keypoint generation/sync batch."
log "Working directory: \"$PWD\""
log "Log file: \"$LOG_FILE\""
log "Dry-run mode: $DRY_RUN"

for size in "${SIZES[@]}"; do
  for precision in "${PRECISIONS[@]}"; do
    label="$(combo_label "$size" "$precision")"
    model_path="$(build_model_path "$size" "$precision")"
    local_output_root="$(build_local_output_root "$precision" "$size")"
    remote_dir_home="$(build_remote_dir_home "$size" "$precision")"
    remote_dest="$(build_remote_dest_rsync "$size" "$precision")"

    log "=================================================================="
    log "Processing: $label"
    log "Model path: \"$model_path\""
    log "Local output root: \"$local_output_root\""
    log "Remote destination: \"$remote_dest\""

    # 1. Skip if model engine is missing
    if [[ ! -f "$model_path" ]]; then
      log "WARNING: Model engine missing. Skipping $label"
      SKIPPED_MODEL_MISSING+=("$label :: missing \"$model_path\"")
      continue
    fi

    # 2. Skip entire combo if remote already contains any .npz
    log "Checking whether remote files already exist."
    if remote_has_files "$remote_dir_home"; then
      log "Remote .npz files found. Skipping entire combination: $label"
      SKIPPED_REMOTE+=("$label")
      continue
    else
      rc=$?
      if [[ "$rc" -eq 2 ]]; then
        log "ERROR: Remote existence check failed for $label"
        FAILURES+=("$label :: remote existence check failed")
        continue
      fi
      log "No remote .npz files found for $label"
    fi

    # 3. If local files already exist, transfer directly
    log "Checking whether local files already exist."
    if local_has_files "$local_output_root"; then
      log "Local .npz files already exist. Regeneration not needed for $label"

      if sync_to_remote "$local_output_root" "$remote_dir_home" "$remote_dest"; then
        log "Transfer succeeded for existing local files: $label"

        if [[ "$DRY_RUN" -eq 1 ]]; then
          log "Dry-run mode enabled; skipping local cleanup."
          TRANSFERRED_LOCAL+=("$label (dry-run)")
        else
          if cleanup_local "$local_output_root"; then
            log "Local cleanup succeeded: $label"
            TRANSFERRED_LOCAL+=("$label")
          else
            log "ERROR: Transfer succeeded but local cleanup failed: $label"
            FAILURES+=("$label :: transfer succeeded, cleanup failed")
          fi
        fi
      else
        log "ERROR: Transfer failed for existing local files: $label"
        FAILURES+=("$label :: transfer from existing local files failed")
      fi

      continue
    else
      rc=$?
      if [[ "$rc" -eq 2 ]]; then
        log "ERROR: Local existence check failed for $label"
        FAILURES+=("$label :: local existence check failed")
        continue
      fi
      log "No local .npz files found. Generation is required for $label"
    fi

    # 4-5. Generate for camera 1 then camera 2
    generation_failed=0
    for camera in "${CAMERAS[@]}"; do
      if ! run_generation "$camera" "$model_path" "$local_output_root"; then
        log "ERROR: Generation failed for $label on camera $camera"
        FAILURES+=("$label :: generation failed on camera $camera")
        generation_failed=1
        break
      fi
    done

    if [[ "$generation_failed" -ne 0 ]]; then
      log "Keeping any local files that may have been produced for inspection/resume: $label"
      continue
    fi

    # 6. Confirm generation created at least one .npz
    log "Verifying that generation produced local .npz files."
    if ! local_has_files "$local_output_root"; then
      rc=$?
      if [[ "$rc" -eq 2 ]]; then
        log "ERROR: Local verification check failed after generation: $label"
        FAILURES+=("$label :: post-generation local verification failed")
      else
        log "ERROR: Generation completed but no .npz files were found: $label"
        FAILURES+=("$label :: generation completed but no .npz files found")
      fi
      continue
    fi

    # 6-7. Transfer and then clean up on success
    if sync_to_remote "$local_output_root" "$remote_dir_home" "$remote_dest"; then
      log "Transfer succeeded after generation: $label"

      if [[ "$DRY_RUN" -eq 1 ]]; then
        log "Dry-run mode enabled; skipping local cleanup."
        GENERATED_AND_TRANSFERRED+=("$label (dry-run)")
      else
        if cleanup_local "$local_output_root"; then
          log "Local cleanup succeeded after generated transfer: $label"
          GENERATED_AND_TRANSFERRED+=("$label")
        else
          log "ERROR: Transfer succeeded but local cleanup failed: $label"
          FAILURES+=("$label :: generated transfer succeeded, cleanup failed")
        fi
      fi
    else
      log "ERROR: Transfer failed after generation; keeping local files: $label"
      FAILURES+=("$label :: transfer failed after generation")
    fi
  done
done

###############################################################################
# Summary
###############################################################################

log "=================================================================="
log "Batch complete. Summary follows."
print_summary_section "Skipped because remote already exists" SKIPPED_REMOTE
print_summary_section "Transferred from existing local files" TRANSFERRED_LOCAL
print_summary_section "Generated then transferred" GENERATED_AND_TRANSFERRED
print_summary_section "Skipped because model missing" SKIPPED_MODEL_MISSING
print_summary_section "Failures" FAILURES

log "Done."