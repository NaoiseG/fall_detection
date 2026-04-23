#!/usr/bin/env bash
set -euo pipefail

# Train/evaluate MotionBERT Lite on the three final base keypoint sources.
# Run this from a GPU job or interactive GPU shell.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTOTYPE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MOTIONBERT_DIR="${PROTOTYPE_DIR}/models/MotionBERT"

PYTHON="${PYTHON:-python}"
KEYPOINTS_ROOT="${KEYPOINTS_ROOT:-${HOME}/scratch/keypoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${HOME}/scratch/final_classification_models/MotionBERT_lite}"
EVAL_ROOT="${EVAL_ROOT:-${HOME}/scratch/evaluations/MotionBERT_lite}"

RUN_NAME="FT_MB_lite_MB_ft_UPFall_xsub"
CONFIG="configs/action/MB_ft_UPFall_xsub_LITE.yaml"
PRETRAIN="checkpoint/pretrain/MB_lite"
TRAIN_PKL="${MOTIONBERT_DIR}/data/action/upfall.pkl"
TEST_PKL="${MOTIONBERT_DIR}/data/action/upfall_test.pkl"
LABEL_MAP="${MOTIONBERT_DIR}/data/action/upfall_label_map.json"

TRAIN_SUBJECTS="${TRAIN_SUBJECTS:-1-12}"
VAL_SUBJECTS="${VAL_SUBJECTS:-13-15}"
TEST_SUBJECTS="${TEST_SUBJECTS:-16-17}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-0}"
# By default, reruns skip pose sources that already have a best checkpoint and
# an evaluation metrics_summary.csv. Use FORCE_RETRAIN=1 and/or FORCE_EVAL=1 to rerun.
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_EVAL="${FORCE_EVAL:-0}"

declare -A NPZ_ROOTS=(
  [yolo11x]="${KEYPOINTS_ROOT}/UPFall_keypoints/yolo11x/base"
  [alphapose]="${KEYPOINTS_ROOT}/UPFall_keypoints_alpha/base"
  [vitpose]="${KEYPOINTS_ROOT}/UPFall_keypoints_vitpose/base"
)

POSES=("$@")
if [[ ${#POSES[@]} -eq 0 ]]; then
  POSES=(yolo11x alphapose vitpose)
fi

check_pkl_cameras() {
  local pkl_path="$1"
  "${PYTHON}" - "${pkl_path}" <<'PY'
import pickle, re, sys
from collections import Counter

pkl_path = sys.argv[1]
cam_re = re.compile(r"(?:^|_)(?:cam|camera)(\d+)(?:_|$)", re.I)
with open(pkl_path, "rb") as f:
    ds = pickle.load(f)

ok = True
for split in ("xsub_train", "xsub_val"):
    counts = Counter()
    for frame_dir in ds.get("split", {}).get(split, []):
        m = cam_re.search(str(frame_dir))
        counts[int(m.group(1)) if m else "unparsed"] += 1
    print(f"{pkl_path} {split}: {dict(sorted(counts.items(), key=lambda kv: str(kv[0])))}", flush=True)
    if counts.get(1, 0) == 0 or counts.get(2, 0) == 0:
        ok = False

if not ok:
    raise SystemExit(f"Expected both Camera1 and Camera2 samples in {pkl_path}")
PY
}

prepare_pkl() {
  local npz_root="$1"
  local val_subjects="$2"
  local out_pkl="$3"

  cd "${PROTOTYPE_DIR}"
  "${PYTHON}" dataset_helpers/prepare_motionbert_dataset.py \
    --outputs-npz-root "${npz_root}" \
    --camera 1 2 \
    --train-subjects "${TRAIN_SUBJECTS}" \
    --val-subjects "${val_subjects}" \
    --out-pkl "${out_pkl}" \
    --out-label-map "${LABEL_MAP}" \
    --label-mode center \
    --win-len 64 \
    --win-step 48

  check_pkl_cameras "${out_pkl}"
}

latest_eval_metrics() {
  local eval_dir="$1"
  if [[ ! -d "${eval_dir}" ]]; then
    return 0
  fi
  find "${eval_dir}" -mindepth 2 -maxdepth 2 -type f -name metrics_summary.csv -print 2>/dev/null | sort | tail -n 1
}

for pose in "${POSES[@]}"; do
  npz_root="${NPZ_ROOTS[${pose}]:-}"
  if [[ -z "${npz_root}" ]]; then
    echo "Unknown pose source '${pose}'. Valid: ${!NPZ_ROOTS[*]}" >&2
    exit 2
  fi
  if [[ ! -d "${npz_root}" ]]; then
    echo "Missing keypoint root: ${npz_root}" >&2
    exit 2
  fi

  ckpt_dir="${OUTPUT_ROOT}/${pose}/${RUN_NAME}"
  eval_dir="${EVAL_ROOT}/${pose}"
  best_ckpt="${ckpt_dir}/best_epoch.bin"
  existing_eval_metrics="$(latest_eval_metrics "${eval_dir}")"

  if [[ -s "${best_ckpt}" && -n "${existing_eval_metrics}" && "${FORCE_RETRAIN}" != "1" && "${FORCE_EVAL}" != "1" ]]; then
    echo
    echo "===== ${pose}: complete, skipping ====="
    echo "Checkpoint: ${best_ckpt}"
    echo "Evaluation: ${existing_eval_metrics}"
    continue
  fi

  if [[ -s "${best_ckpt}" && "${FORCE_RETRAIN}" != "1" ]]; then
    echo
    echo "===== ${pose}: checkpoint exists, skipping training ====="
    echo "Checkpoint: ${best_ckpt}"
  else
    echo
    echo "===== ${pose}: preparing train/val pkl (${TRAIN_SUBJECTS} -> ${VAL_SUBJECTS}) ====="
    prepare_pkl "${npz_root}" "${VAL_SUBJECTS}" "${TRAIN_PKL}"

    echo
    echo "===== ${pose}: training MotionBERT Lite ====="
    if [[ "${FORCE_RETRAIN}" == "1" ]]; then
      rm -rf -- "${ckpt_dir}"
    fi
    mkdir -p "$(dirname "${ckpt_dir}")"
    cd "${MOTIONBERT_DIR}"
    "${PYTHON}" train_action_weighted_balanced.py \
      --config "${CONFIG}" \
      --pretrained "${PRETRAIN}" \
      --checkpoint "${ckpt_dir}" \
      --ckpt-metric composite \
      --ckpt-w 0.7 \
      --ckpt-beta 2.0 \
      --rare-class-boost 1.5 \
      --weighted-sampler 1 \
      --camera 1 2
  fi

  existing_eval_metrics="$(latest_eval_metrics "${eval_dir}")"
  if [[ -n "${existing_eval_metrics}" && "${FORCE_EVAL}" != "1" ]]; then
    echo
    echo "===== ${pose}: evaluation exists, skipping ====="
    echo "Evaluation: ${existing_eval_metrics}"
    continue
  fi

  echo
  echo "===== ${pose}: preparing held-out test pkl (${TEST_SUBJECTS}) ====="
  prepare_pkl "${npz_root}" "${TEST_SUBJECTS}" "${TEST_PKL}"

  echo
  echo "===== ${pose}: evaluating best checkpoint ====="
  mkdir -p "${eval_dir}"
  cd "${MOTIONBERT_DIR}"
  "${PYTHON}" eval_motionbert_action.py \
    --config "${CONFIG}" \
    --checkpoint "${ckpt_dir}/best_epoch.bin" \
    --subjects "${TEST_SUBJECTS}" \
    --camera 1 2 \
    --out-dir "${eval_dir}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${EVAL_NUM_WORKERS}" \
    --device "${DEVICE}" \
    --data-pkl "${TEST_PKL}" \
    --ckpt-metric composite \
    --ckpt-w 0.7 \
    --ckpt-beta 2.0 \
    --fall-class-idx 0
done

echo
echo "Done. Checkpoints: ${OUTPUT_ROOT}"
echo "Evaluations:  ${EVAL_ROOT}"
