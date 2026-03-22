#!/bin/bash -l
#SBATCH --job-name=prune_pose_yolo11
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.name@ucdconnect.ie

set -euo pipefail


# ==============================================================================
# USER-EDITABLE SETTINGS
# ------------------------------------------------------------------------------
# These are the settings you will most likely change later:
#   - PYTHON_BIN: your pruning environment's Python interpreter
#   - PRUNE_SCRIPT: location of prune_pose.py
#   - OUT_PROJECT: output directory for pruned runs
#   - GPU_COUNT: number of GPUs to use in parallel on this Slurm allocation
#   - DEFAULT_BATCH: default per-run batch size
#   - MODEL_BATCH_OVERRIDES: optional per-model batch overrides (useful for yolo11x-pose)
# ==============================================================================
PYTHON_BIN="/home/people/21376026/venvs/modelopt/bin/python"
PRUNE_SCRIPT="/home/people/21376026/fall_detection/pruning/prune_pose.py"
DATA_YAML="/home/people/21376026/fall_detection/pruning/coco-pose.yaml"
MODEL_ROOT="/home/people/21376026/scratch/prune_models"
OUT_PROJECT="/home/people/21376026/scratch/pruned_pose_runs"

GPU_COUNT=2
TASK="pose"
IMGSZ=640
DEFAULT_BATCH=16
PATIENCE=20
WORKERS=8
MAX_ITER_DL=20
SEARCH_CKPT="modelopt_fastnas_search_checkpoint.pth"
NAME_PREFIX="pruned_pose"
METRIC="map5095"
FLOPS=("90%" "80%" "70%")

# If --epochs is omitted below, prune_pose.py uses the per-target schedule:
#   90% -> 40 epochs
#   80% -> 60 epochs
#   70% -> 80 epochs
# Set EPOCHS_OVERRIDE to a number to force the same epoch count for every run.
EPOCHS_OVERRIDE=""

# Optional: per-model batch-size overrides.
# This makes it easy to reduce batch size for larger models like yolo11x-pose.
declare -A MODEL_BATCH_OVERRIDES=(
  #["yolo11x-pose"]=8
)

mkdir -p logs "${OUT_PROJECT}"


echo "=== PRECHECK ==="
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V
"${PYTHON_BIN}" - <<'PY'
import platform
print("Machine:", platform.machine())
print("Processor:", platform.processor())
print("Precheck imports...")
import torch
print(" torch:", torch.__version__)
import ultralytics
print(" ultralytics:", ultralytics.__version__)
import modelopt
print(" modelopt: OK")
import modelopt.torch.prune as mtp
print(" modelopt.torch.prune: OK")
print("Precheck passed.")
PY
echo "=== PRECHECK DONE ==="


# ---------- MODEL LIST ----------
# Keep these project-specific and readable.
MODELS=(
  "yolo11n-pose/yolo11n-pose.pt"
  "yolo11s-pose/yolo11s-pose.pt"
  "yolo11m-pose/yolo11m-pose.pt"
  "yolo11l-pose/yolo11l-pose.pt"
  "yolo11x-pose/yolo11x-pose.pt"
)


echo "Job started: $(date)"
echo "Node(s): ${SLURM_NODELIST:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Output project: ${OUT_PROJECT}"
echo "Task: ${TASK}"
echo "Metric: pose mAP50-95 (${METRIC})"
echo "FLOPs targets: ${FLOPS[*]}"

# ---------- Required file checks ----------
[[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python interpreter not executable: ${PYTHON_BIN}"; exit 1; }
[[ -f "${PRUNE_SCRIPT}" ]] || { echo "ERROR: Missing prune script: ${PRUNE_SCRIPT}"; exit 1; }
[[ -f "${DATA_YAML}" ]] || { echo "ERROR: Missing data yaml: ${DATA_YAML}"; exit 1; }
[[ -d "${MODEL_ROOT}" ]] || { echo "ERROR: Missing model root: ${MODEL_ROOT}"; exit 1; }

# ---------- Build absolute model paths and skip missing ----------
MODEL_PATHS=()
for rel_path in "${MODELS[@]}"; do
  abs_path="${MODEL_ROOT}/${rel_path}"
  if [[ -f "${abs_path}" ]]; then
    MODEL_PATHS+=("${abs_path}")
  else
    echo "WARN: Missing model checkpoint, skipping: ${abs_path}"
  fi
done

if (( ${#MODEL_PATHS[@]} == 0 )); then
  echo "ERROR: No valid model files found under MODEL_ROOT=${MODEL_ROOT}"
  exit 1
fi

echo "INFO: Found ${#MODEL_PATHS[@]} model(s):"
printf '  %s\n' "${MODEL_PATHS[@]}"


run_one() {
  local gpu_id="$1"
  local model_path="$2"
  local model_dirname
  local model_tag
  local batch_size

  model_dirname="$(basename "$(dirname "${model_path}")")"
  model_tag="$(basename "${model_path}" .pt)"
  batch_size="${DEFAULT_BATCH}"

  if [[ -n "${MODEL_BATCH_OVERRIDES[${model_dirname}]:-}" ]]; then
    batch_size="${MODEL_BATCH_OVERRIDES[${model_dirname}]}"
  fi

  echo "[$(date)] START model=${model_tag} dir=${model_dirname} gpu=${gpu_id} batch=${batch_size}"

  if [[ -n "${EPOCHS_OVERRIDE}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" "${PRUNE_SCRIPT}" \
      --models "${model_path}" \
      --data "${DATA_YAML}" \
      --task "${TASK}" \
      --imgsz "${IMGSZ}" \
      --batch "${batch_size}" \
      --epochs "${EPOCHS_OVERRIDE}" \
      --patience "${PATIENCE}" \
      --workers "${WORKERS}" \
      --flops "${FLOPS[@]}" \
      --max_iter_data_loader "${MAX_ITER_DL}" \
      --search_ckpt "${SEARCH_CKPT}" \
      --name_prefix "${NAME_PREFIX}" \
      --project "${OUT_PROJECT}" \
      --metric "${METRIC}" \
      --device 0 \
      --cache
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${PYTHON_BIN}" "${PRUNE_SCRIPT}" \
      --models "${model_path}" \
      --data "${DATA_YAML}" \
      --task "${TASK}" \
      --imgsz "${IMGSZ}" \
      --batch "${batch_size}" \
      --patience "${PATIENCE}" \
      --workers "${WORKERS}" \
      --flops "${FLOPS[@]}" \
      --max_iter_data_loader "${MAX_ITER_DL}" \
      --search_ckpt "${SEARCH_CKPT}" \
      --name_prefix "${NAME_PREFIX}" \
      --project "${OUT_PROJECT}" \
      --metric "${METRIC}" \
      --device 0 \
      --cache
  fi

  echo "[$(date)] DONE model=${model_tag} dir=${model_dirname} gpu=${gpu_id}"
}


# ---------- Parallel launcher ----------
# One model per GPU, run in batches, wait after filling all GPU slots.
pids=()
slot=0

for model_path in "${MODEL_PATHS[@]}"; do
  gpu_id=$(( slot % GPU_COUNT ))
  run_one "${gpu_id}" "${model_path}" &
  pids+=("$!")
  slot=$((slot + 1))

  if (( slot % GPU_COUNT == 0 )); then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
  fi
done

# Wait for any leftover background jobs.
for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "All prune jobs completed: $(date)"
