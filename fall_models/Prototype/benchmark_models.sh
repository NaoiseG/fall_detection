#!/usr/bin/env bash

# Benchmark all combinations of:
#   5 pose models × 4 versions × 3 classifiers = 60 runs
#
# Run this script from:
#   /home/jetson/NaoiseG/fall_detection/fall_models/Prototype
#
# Or let it cd there automatically below.

set -u
set -o pipefail

###############################################################################
# Configuration
###############################################################################

PROJECT_DIR="/home/jetson/NaoiseG/fall_detection/fall_models/Prototype"
BENCH_DIR="benchmarks"
VIDEO_PATH="../../Datasets/test_vids/activity_all.mp4"
MOTIONBERT_CONFIG="../../web_app/models/classification/MotionBERT/configs/action/MB_ft_UPFall_xsub.yaml"
MODELS_ROOT="../../pose_models/quantised"

POSE_MODELS=(
  "yolo11n-pose"
  "yolo11s-pose"
  "yolo11m-pose"
  "yolo11l-pose"
  "yolo11x-pose"
)

VERSIONS=(
  "base"
  "fp32"
  "fp16"
  "int8"
)

ALPHAPOSE_POSE_MODEL="alphapose"
ALPHAPOSE_ROOT="pose_models/AlphaPose"
ALPHAPOSE_CFG="configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml"
ALPHAPOSE_DETECTOR_CFG="detector/yolo/cfg/yolov3-spp.cfg"

ALPHAPOSE_VERSIONS=(
  "original"
  "fp32_fp32"
  "fp16det_fp32pose"
  "fp16_fp16"
  "int8det_fp16pose"
)

VITPOSE_POSE_MODEL="vitpose"
VITPOSE_VERSIONS=(
  "base"
  "fp32"
  "fp16"
)

CLASSIFIERS=(
  "cnnlstm"
  "stgcn"
  "motionbert"
)

# Fixed classification weights for ALL runs
CNNLSTM_WEIGHT="../../web_app/models/classification/cnnlstm/yolo11l-pose/cnnlstm_best.pt"
STGCN_WEIGHT="../../web_app/models/classification/stgcn/yolo11l-pose/stgcn_best.pt"
MOTIONBERT_WEIGHT="../../web_app/models/classification/MotionBERT/yolo11l-pose/checkpoint/action/FT_MB_release_MB_ft_UPFall_xsub/best_epoch.bin"

TOTAL_RUNS=$(( \
  ${#POSE_MODELS[@]} * ${#VERSIONS[@]} * ${#CLASSIFIERS[@]} + \
  ${#ALPHAPOSE_VERSIONS[@]} * ${#CLASSIFIERS[@]} + \
  ${#VITPOSE_VERSIONS[@]} * ${#CLASSIFIERS[@]} \
))

###############################################################################
# Helpers
###############################################################################

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

log_success() {
  local msg="$1"
  printf '[%s] %s\n' "$(timestamp)" "$msg" >> "${BENCH_DIR}/successful_runs.log"
}

log_failure() {
  local msg="$1"
  printf '[%s] %s\n' "$(timestamp)" "$msg" >> "${BENCH_DIR}/failed_runs.log"
}

log_skip() {
  local msg="$1"
  printf '[%s] %s\n' "$(timestamp)" "$msg" >> "${BENCH_DIR}/skipped_runs.log"
}

join_cmd() {
  # Print a shell-escaped command line
  local out=""
  local arg
  for arg in "$@"; do
    printf -v out '%s%q ' "$out" "$arg"
  done
  printf '%s' "${out% }"
}

pose_weight_for_version() {
  local pose_model="$1"
  local version="$2"
  local base_dir="${MODELS_ROOT}/ultralytics/${pose_model}"

  case "$version" in
    base) printf '%s/%s.pt' "${base_dir}" "${pose_model}" ;;
    fp32) printf '%s/%s_fp32.engine' "${base_dir}" "${pose_model}" ;;
    fp16) printf '%s/%s_fp16.engine' "${base_dir}" "${pose_model}" ;;
    int8) printf '%s/%s_int8.engine' "${base_dir}" "${pose_model}" ;;
    *)
      return 1
      ;;
  esac
}

classifier_weight_for_arch() {
  local classifier="$1"

  case "$classifier" in
    cnnlstm)    printf '%s' "${CNNLSTM_WEIGHT}" ;;
    stgcn)      printf '%s' "${STGCN_WEIGHT}" ;;
    motionbert) printf '%s' "${MOTIONBERT_WEIGHT}" ;;
    *)
      return 1
      ;;
  esac
}

half_flag_for_version() {
  local version="$1"

  case "$version" in
    base|fp32) printf '%s' "0" ;;
    fp16|int8) printf '%s' "1" ;;
    *)
      return 1
      ;;
  esac
}

alphapose_detector_weights_for_version() {
  local version="$1"

  case "$version" in
    original)          printf '%s' "detector/yolo/data/yolov3-spp.weights" ;;
    fp32_fp32)         printf '%s' "${MODELS_ROOT}/alphapose/yolov3_spp_fp32.engine" ;;
    fp16det_fp32pose)  printf '%s' "${MODELS_ROOT}/alphapose/yolov3_spp_fp16.engine" ;;
    fp16_fp16)         printf '%s' "${MODELS_ROOT}/alphapose/yolov3_spp_fp16.engine" ;;
    int8det_fp16pose)  printf '%s' "${MODELS_ROOT}/alphapose/yolov3_spp_int8.engine" ;;
    *)
      return 1
      ;;
  esac
}

alphapose_checkpoint_for_version() {
  local version="$1"

  case "$version" in
    original)          printf '%s' "pretrained_models/fast_res50_256x192.pth" ;;
    fp32_fp32)         printf '%s' "${MODELS_ROOT}/alphapose/fastpose_fp32.engine" ;;
    fp16det_fp32pose)  printf '%s' "${MODELS_ROOT}/alphapose/fastpose_fp32.engine" ;;
    fp16_fp16)         printf '%s' "${MODELS_ROOT}/alphapose/fastpose_fp16.engine" ;;
    int8det_fp16pose)  printf '%s' "${MODELS_ROOT}/alphapose/fastpose_fp16.engine" ;;
    *)
      return 1
      ;;
  esac
}

vitpose_detector_model_for_version() {
  local version="$1"

  case "$version" in
    base) printf '%s' "PekingU/rtdetr_r50vd_coco_o365" ;;
    fp32) printf '%s' "${MODELS_ROOT}/vitpose_trt/engines/detector_pekingu_rtdetr_r50vd_coco_o365_fp32.engine" ;;
    fp16) printf '%s' "${MODELS_ROOT}/vitpose_trt/engines/detector_pekingu_rtdetr_r50vd_coco_o365_fp16.engine" ;;
    *)
      return 1
      ;;
  esac
}

vitpose_pose_model_for_version() {
  local version="$1"

  case "$version" in
    base) printf '%s' "usyd-community/vitpose-base" ;;
    fp32) printf '%s' "${MODELS_ROOT}/vitpose_trt/engines/pose_usyd_community_vitpose_base_fp32.engine" ;;
    fp16) printf '%s' "${MODELS_ROOT}/vitpose_trt/engines/pose_usyd_community_vitpose_base_fp16.engine" ;;
    *)
      return 1
      ;;
  esac
}

resolve_alphapose_check_path() {
  local alphapose_root="$1"
  local path_arg="$2"

  case "$path_arg" in
    /*|./*|../*)
      printf '%s' "${path_arg}"
      ;;
    *)
      if [[ -e "${path_arg}" ]]; then
        printf '%s' "${path_arg}"
      else
        printf '%s/%s' "${alphapose_root%/}" "${path_arg}"
      fi
      ;;
  esac
}

snapshot_top_level_dirs() {
  # Print absolute paths of immediate subdirectories inside benchmarks, sorted.
  # This includes both structured destination dirs and run dirs.
  find "${BENCH_DIR}" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort
}

find_new_run_dir() {
  # Usage: find_new_run_dir <before_file> <after_file>
  # Returns:
  #   0 + prints directory path if exactly one new top-level dir appeared
  #   1 otherwise
  local before_file="$1"
  local after_file="$2"

  local diff_file
  diff_file="$(mktemp)"

  comm -13 "${before_file}" "${after_file}" > "${diff_file}"

  local count
  count="$(grep -c . "${diff_file}" || true)"

  if [[ "${count}" -eq 1 ]]; then
    cat "${diff_file}"
    rm -f "${diff_file}"
    return 0
  fi

  rm -f "${diff_file}"
  return 1
}

require_file_or_log() {
  local path="$1"
  local pose_model="$2"
  local version="$3"
  local classifier="$4"
  local description="$5"
  local cmd_str="$6"

  if [[ ! -f "$path" ]]; then
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=missing_file missing=${description} path=${path} cmd=\"${cmd_str}\""
    return 1
  fi

  return 0
}

require_model_source_or_log() {
  local source="$1"
  local pose_model="$2"
  local version="$3"
  local classifier="$4"
  local description="$5"
  local cmd_str="$6"

  if [[ -e "$source" ]]; then
    return 0
  fi

  case "$source" in
    *.engine|*.pt|*.pth|*.onnx|*.bin|*.ckpt|/*|./*|../*)
      log_failure \
        "pose_model=${pose_model} version=${version} classifier=${classifier} status=missing_source missing=${description} path=${source} cmd=\"${cmd_str}\""
      return 1
      ;;
  esac

  if [[ -z "$source" ]]; then
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=missing_source missing=${description} path=${source} cmd=\"${cmd_str}\""
    return 1
  fi

  return 0
}

combination_already_done() {
  local pose_model="$1"
  local version="$2"
  local run_model_tag="$3"
  local dest_dir="${BENCH_DIR}/${pose_model}/${version}"

  [[ -d "${dest_dir}" ]] || return 1

  find "${dest_dir}" -mindepth 1 -maxdepth 1 -type d -name "*__model_${run_model_tag}__*" -print -quit 2>/dev/null | grep -q .
}

build_command() {
  local classifier="$1"
  local version="$2"
  local cls_weight="$3"
  local pose_weight="$4"
  local half_flag

  half_flag="$(half_flag_for_version "${version}")" || return 1

  if [[ "$classifier" == "motionbert" ]]; then
    printf '%s\0' \
      python -m inference.infer_motionbert_video \
      --video "${VIDEO_PATH}" \
      --model "${cls_weight}" \
      --config "${MOTIONBERT_CONFIG}" \
      --yolo-weights "${pose_weight}" \
      --device cuda \
      --half "${half_flag}" \
      --max-people 10 \
      --max-det 10 \
      --warmup-frames 5 \
      --warmup-windows 0 \
      --benchmark 1 \
      --profile-out "${BENCH_DIR}" \
      --no-display 1 \
      --out-csv "" \
      --out-pkl ""
  else
    printf '%s\0' \
      python -m inference.inference_on_video \
      --video "${VIDEO_PATH}" \
      --model "${cls_weight}" \
      --yolo-weights "${pose_weight}" \
      --arch "${classifier}" \
      --device cuda \
      --half "${half_flag}" \
      --max-people 10 \
      --max-det 10 \
      --warmup-frames 5 \
      --warmup-windows 0 \
      --benchmark 1 \
      --profile-out "${BENCH_DIR}" \
      --no-display 1
  fi
}

build_alphapose_command() {
  local classifier="$1"
  local cls_weight="$2"
  local alphapose_checkpoint="$3"
  local alphapose_detector_weights="$4"

  if [[ "$classifier" == "motionbert" ]]; then
    printf '%s\0' \
      python -m inference.infer_motionbert_video_alphapose \
      --video "${VIDEO_PATH}" \
      --model "${cls_weight}" \
      --config "${MOTIONBERT_CONFIG}" \
      --alphapose-root "${ALPHAPOSE_ROOT}" \
      --alphapose-cfg "${ALPHAPOSE_CFG}" \
      --alphapose-detector-cfg "${ALPHAPOSE_DETECTOR_CFG}" \
      --alphapose-checkpoint "${alphapose_checkpoint}" \
      --alphapose-detector-weights "${alphapose_detector_weights}" \
      --device cuda \
      --max-det 10 \
      --warmup-frames 5 \
      --warmup-windows 0 \
      --benchmark 1 \
      --profile-out "${BENCH_DIR}" \
      --no-display 1 \
      --out-csv "" \
      --out-pkl ""
  else
    printf '%s\0' \
      python -m inference.inference_on_video_alphapose \
      --video "${VIDEO_PATH}" \
      --model "${cls_weight}" \
      --arch "${classifier}" \
      --alphapose-root "${ALPHAPOSE_ROOT}" \
      --alphapose-cfg "${ALPHAPOSE_CFG}" \
      --alphapose-detector-cfg "${ALPHAPOSE_DETECTOR_CFG}" \
      --alphapose-checkpoint "${alphapose_checkpoint}" \
      --alphapose-detector-weights "${alphapose_detector_weights}" \
      --device cuda \
      --max-det 10 \
      --warmup-frames 5 \
      --warmup-windows 0 \
      --benchmark 1 \
      --profile-out "${BENCH_DIR}" \
      --no-display 1
  fi
}

build_vitpose_command() {
  local classifier="$1"
  local cls_weight="$2"
  local vitpose_detector_model="$3"
  local vitpose_pose_model="$4"

  if [[ "$classifier" == "motionbert" ]]; then
    printf '%s\0' \
      python -m inference.infer_motionbert_video_vitpose \
      --video "${VIDEO_PATH}" \
      --model "${cls_weight}" \
      --config "${MOTIONBERT_CONFIG}" \
      --vitpose-detector-model "${vitpose_detector_model}" \
      --vitpose-pose-model "${vitpose_pose_model}" \
      --device cuda \
      --max-det 10 \
      --warmup-frames 5 \
      --warmup-windows 0 \
      --benchmark 1 \
      --profile-out "${BENCH_DIR}" \
      --no-display 1 \
      --out-csv "" \
      --out-pkl ""
  else
    printf '%s\0' \
      python -m inference.inference_on_video_vitpose \
      --video "${VIDEO_PATH}" \
      --model "${cls_weight}" \
      --arch "${classifier}" \
      --vitpose-detector-model "${vitpose_detector_model}" \
      --vitpose-pose-model "${vitpose_pose_model}" \
      --device cuda \
      --max-det 10 \
      --warmup-frames 5 \
      --warmup-windows 0 \
      --benchmark 1 \
      --profile-out "${BENCH_DIR}" \
      --no-display 1
  fi
}

run_one_benchmark() {
  local run_idx="$1"
  local pose_model="$2"
  local version="$3"
  local classifier="$4"

  local pose_weight
  local cls_weight
  local dest_dir
  local cmd_str
  local rc=0

  pose_weight="$(pose_weight_for_version "${pose_model}" "${version}")" || {
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=internal_error reason=invalid_version_mapping"
    return 1
  }

  cls_weight="$(classifier_weight_for_arch "${classifier}")" || {
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=internal_error reason=invalid_classifier_mapping"
    return 1
  }

  dest_dir="${BENCH_DIR}/${pose_model}/${version}"
  mkdir -p "${dest_dir}"

  if combination_already_done "${pose_model}" "${version}" "${classifier}"; then
    printf '[%d/%d] Skipping %s %s + %s (already benchmarked)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${pose_model}" "${version}" "${classifier}"

    log_skip \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=skipped reason=already_benchmarked dest_dir=${dest_dir}"
    return 2
  fi

  local cmd=()
  while IFS= read -r -d '' token; do
    cmd+=("$token")
  done < <(build_command "${classifier}" "${version}" "${cls_weight}" "${pose_weight}")

  cmd_str="$(join_cmd "${cmd[@]}")"

  printf '[%d/%d] Running %s %s + %s\n' "${run_idx}" "${TOTAL_RUNS}" "${pose_model}" "${version}" "${classifier}"

  # Pre-flight checks
  require_file_or_log "${VIDEO_PATH}" "${pose_model}" "${version}" "${classifier}" "video" "${cmd_str}" || return 1
  require_file_or_log "${pose_weight}" "${pose_model}" "${version}" "${classifier}" "pose_weight" "${cmd_str}" || return 1
  require_file_or_log "${cls_weight}" "${pose_model}" "${version}" "${classifier}" "classification_weight" "${cmd_str}" || return 1

  if [[ "${classifier}" == "motionbert" ]]; then
    require_file_or_log "${MOTIONBERT_CONFIG}" "${pose_model}" "${version}" "${classifier}" "motionbert_config" "${cmd_str}" || return 1
  fi

  local before_file after_file new_run_dir
  before_file="$(mktemp)"
  after_file="$(mktemp)"

  snapshot_top_level_dirs > "${before_file}"

  "${cmd[@]}"
  rc=$?

  snapshot_top_level_dirs > "${after_file}"

  if [[ "${rc}" -ne 0 ]]; then
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=command_failed exit_code=${rc} cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
    return 1
  fi

  if new_run_dir="$(find_new_run_dir "${before_file}" "${after_file}")"; then
    if [[ -d "${new_run_dir}" ]]; then
      local run_basename
      run_basename="$(basename "${new_run_dir}")"

      if mv "${new_run_dir}" "${dest_dir}/"; then
        log_success \
          "pose_model=${pose_model} version=${version} classifier=${classifier} status=ok moved_to=${dest_dir}/${run_basename} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 0
      else
        log_failure \
          "pose_model=${pose_model} version=${version} classifier=${classifier} status=move_failed source=${new_run_dir} dest=${dest_dir} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 1
      fi
    else
      log_failure \
        "pose_model=${pose_model} version=${version} classifier=${classifier} status=no_new_directory_found reason=diff_returned_non_directory path=${new_run_dir} cmd=\"${cmd_str}\""
      rm -f "${before_file}" "${after_file}"
      return 1
    fi
  else
    log_failure \
      "pose_model=${pose_model} version=${version} classifier=${classifier} status=no_unique_new_directory_found cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
    return 1
  fi
}

run_one_alphapose_benchmark() {
  local run_idx="$1"
  local version="$2"
  local classifier="$3"

  local alphapose_detector_weights
  local alphapose_checkpoint
  local cls_weight
  local dest_dir
  local cmd_str
  local rc=0
  local detector_cfg_path
  local detector_weights_path
  local checkpoint_path
  local run_model_tag

  alphapose_detector_weights="$(alphapose_detector_weights_for_version "${version}")" || {
    log_failure \
      "pose_model=${ALPHAPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=internal_error reason=invalid_alphapose_detector_mapping"
    return 1
  }

  alphapose_checkpoint="$(alphapose_checkpoint_for_version "${version}")" || {
    log_failure \
      "pose_model=${ALPHAPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=internal_error reason=invalid_alphapose_checkpoint_mapping"
    return 1
  }

  cls_weight="$(classifier_weight_for_arch "${classifier}")" || {
    log_failure \
      "pose_model=${ALPHAPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=internal_error reason=invalid_classifier_mapping"
    return 1
  }

  dest_dir="${BENCH_DIR}/${ALPHAPOSE_POSE_MODEL}/${version}"
  mkdir -p "${dest_dir}"

  run_model_tag="${classifier}_alphapose"
  if combination_already_done "${ALPHAPOSE_POSE_MODEL}" "${version}" "${run_model_tag}"; then
    printf '[%d/%d] Skipping %s %s + %s (already benchmarked)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${ALPHAPOSE_POSE_MODEL}" "${version}" "${classifier}"

    log_skip \
      "pose_model=${ALPHAPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=skipped reason=already_benchmarked dest_dir=${dest_dir}"
    return 2
  fi

  local cmd=()
  while IFS= read -r -d '' token; do
    cmd+=("$token")
  done < <(build_alphapose_command "${classifier}" "${cls_weight}" "${alphapose_checkpoint}" "${alphapose_detector_weights}")

  cmd_str="$(join_cmd "${cmd[@]}")"

  printf '[%d/%d] Running %s %s + %s\n' \
    "${run_idx}" "${TOTAL_RUNS}" "${ALPHAPOSE_POSE_MODEL}" "${version}" "${classifier}"

  detector_cfg_path="$(resolve_alphapose_check_path "${ALPHAPOSE_ROOT}" "${ALPHAPOSE_DETECTOR_CFG}")"
  detector_weights_path="$(resolve_alphapose_check_path "${ALPHAPOSE_ROOT}" "${alphapose_detector_weights}")"
  checkpoint_path="$(resolve_alphapose_check_path "${ALPHAPOSE_ROOT}" "${alphapose_checkpoint}")"

  # Pre-flight checks
  require_file_or_log "${VIDEO_PATH}" "${ALPHAPOSE_POSE_MODEL}" "${version}" "${classifier}" "video" "${cmd_str}" || return 1
  require_file_or_log "${cls_weight}" "${ALPHAPOSE_POSE_MODEL}" "${version}" "${classifier}" "classification_weight" "${cmd_str}" || return 1
  require_file_or_log "${detector_cfg_path}" "${ALPHAPOSE_POSE_MODEL}" "${version}" "${classifier}" "alphapose_detector_cfg" "${cmd_str}" || return 1
  require_file_or_log "${detector_weights_path}" "${ALPHAPOSE_POSE_MODEL}" "${version}" "${classifier}" "alphapose_detector_weights" "${cmd_str}" || return 1
  require_file_or_log "${checkpoint_path}" "${ALPHAPOSE_POSE_MODEL}" "${version}" "${classifier}" "alphapose_fastpose_checkpoint" "${cmd_str}" || return 1

  if [[ "${classifier}" == "motionbert" ]]; then
    require_file_or_log "${MOTIONBERT_CONFIG}" "${ALPHAPOSE_POSE_MODEL}" "${version}" "${classifier}" "motionbert_config" "${cmd_str}" || return 1
  fi

  local before_file after_file new_run_dir
  before_file="$(mktemp)"
  after_file="$(mktemp)"

  snapshot_top_level_dirs > "${before_file}"

  "${cmd[@]}"
  rc=$?

  snapshot_top_level_dirs > "${after_file}"

  if [[ "${rc}" -ne 0 ]]; then
    log_failure \
      "pose_model=${ALPHAPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=command_failed exit_code=${rc} cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
    return 1
  fi

  if new_run_dir="$(find_new_run_dir "${before_file}" "${after_file}")"; then
    if [[ -d "${new_run_dir}" ]]; then
      local run_basename
      run_basename="$(basename "${new_run_dir}")"

      if mv "${new_run_dir}" "${dest_dir}/"; then
        log_success \
          "pose_model=${ALPHAPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=ok moved_to=${dest_dir}/${run_basename} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 0
      else
        log_failure \
          "pose_model=${ALPHAPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=move_failed source=${new_run_dir} dest=${dest_dir} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 1
      fi
    else
      log_failure \
        "pose_model=${ALPHAPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=no_new_directory_found reason=diff_returned_non_directory path=${new_run_dir} cmd=\"${cmd_str}\""
      rm -f "${before_file}" "${after_file}"
      return 1
    fi
  else
    log_failure \
      "pose_model=${ALPHAPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=no_unique_new_directory_found cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
      return 1
  fi
}

run_one_vitpose_benchmark() {
  local run_idx="$1"
  local version="$2"
  local classifier="$3"

  local vitpose_detector_model
  local vitpose_pose_model
  local cls_weight
  local dest_dir
  local cmd_str
  local rc=0
  local run_model_tag

  vitpose_detector_model="$(vitpose_detector_model_for_version "${version}")" || {
    log_failure \
      "pose_model=${VITPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=internal_error reason=invalid_vitpose_detector_mapping"
    return 1
  }

  vitpose_pose_model="$(vitpose_pose_model_for_version "${version}")" || {
    log_failure \
      "pose_model=${VITPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=internal_error reason=invalid_vitpose_pose_mapping"
    return 1
  }

  cls_weight="$(classifier_weight_for_arch "${classifier}")" || {
    log_failure \
      "pose_model=${VITPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=internal_error reason=invalid_classifier_mapping"
    return 1
  }

  dest_dir="${BENCH_DIR}/${VITPOSE_POSE_MODEL}/${version}"
  mkdir -p "${dest_dir}"

  run_model_tag="${classifier}_vitpose"
  if combination_already_done "${VITPOSE_POSE_MODEL}" "${version}" "${run_model_tag}"; then
    printf '[%d/%d] Skipping %s %s + %s (already benchmarked)\n' \
      "${run_idx}" "${TOTAL_RUNS}" "${VITPOSE_POSE_MODEL}" "${version}" "${classifier}"

    log_skip \
      "pose_model=${VITPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=skipped reason=already_benchmarked dest_dir=${dest_dir}"
    return 2
  fi

  local cmd=()
  while IFS= read -r -d '' token; do
    cmd+=("$token")
  done < <(build_vitpose_command "${classifier}" "${cls_weight}" "${vitpose_detector_model}" "${vitpose_pose_model}")

  cmd_str="$(join_cmd "${cmd[@]}")"

  printf '[%d/%d] Running %s %s + %s\n' \
    "${run_idx}" "${TOTAL_RUNS}" "${VITPOSE_POSE_MODEL}" "${version}" "${classifier}"

  # Pre-flight checks
  require_file_or_log "${VIDEO_PATH}" "${VITPOSE_POSE_MODEL}" "${version}" "${classifier}" "video" "${cmd_str}" || return 1
  require_file_or_log "${cls_weight}" "${VITPOSE_POSE_MODEL}" "${version}" "${classifier}" "classification_weight" "${cmd_str}" || return 1
  require_model_source_or_log "${vitpose_detector_model}" "${VITPOSE_POSE_MODEL}" "${version}" "${classifier}" "vitpose_detector_model" "${cmd_str}" || return 1
  require_model_source_or_log "${vitpose_pose_model}" "${VITPOSE_POSE_MODEL}" "${version}" "${classifier}" "vitpose_pose_model" "${cmd_str}" || return 1

  if [[ "${classifier}" == "motionbert" ]]; then
    require_file_or_log "${MOTIONBERT_CONFIG}" "${VITPOSE_POSE_MODEL}" "${version}" "${classifier}" "motionbert_config" "${cmd_str}" || return 1
  fi

  local before_file after_file new_run_dir
  before_file="$(mktemp)"
  after_file="$(mktemp)"

  snapshot_top_level_dirs > "${before_file}"

  "${cmd[@]}"
  rc=$?

  snapshot_top_level_dirs > "${after_file}"

  if [[ "${rc}" -ne 0 ]]; then
    log_failure \
      "pose_model=${VITPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=command_failed exit_code=${rc} cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
    return 1
  fi

  if new_run_dir="$(find_new_run_dir "${before_file}" "${after_file}")"; then
    if [[ -d "${new_run_dir}" ]]; then
      local run_basename
      run_basename="$(basename "${new_run_dir}")"

      if mv "${new_run_dir}" "${dest_dir}/"; then
        log_success \
          "pose_model=${VITPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=ok moved_to=${dest_dir}/${run_basename} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 0
      else
        log_failure \
          "pose_model=${VITPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=move_failed source=${new_run_dir} dest=${dest_dir} cmd=\"${cmd_str}\""
        rm -f "${before_file}" "${after_file}"
        return 1
      fi
    else
      log_failure \
        "pose_model=${VITPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=no_new_directory_found reason=diff_returned_non_directory path=${new_run_dir} cmd=\"${cmd_str}\""
      rm -f "${before_file}" "${after_file}"
      return 1
    fi
  else
    log_failure \
      "pose_model=${VITPOSE_POSE_MODEL} version=${version} classifier=${classifier} status=no_unique_new_directory_found cmd=\"${cmd_str}\""
    rm -f "${before_file}" "${after_file}"
    return 1
  fi
}

###############################################################################
# Main
###############################################################################

cd "${PROJECT_DIR}" || {
  echo "ERROR: Could not cd to ${PROJECT_DIR}" >&2
  exit 1
}

# Keep existing benchmark outputs. Only ensure the directory exists.
mkdir -p "${BENCH_DIR}"

# Logging behavior:
# - successful_runs.log is preserved and appended to
# - failed_runs.log is reset on each invocation
# - skipped_runs.log is reset on each invocation
touch "${BENCH_DIR}/successful_runs.log"
: > "${BENCH_DIR}/failed_runs.log"
: > "${BENCH_DIR}/skipped_runs.log"

# Pre-create expected destination structure
for pose_model in "${POSE_MODELS[@]}"; do
  for version in "${VERSIONS[@]}"; do
    mkdir -p "${BENCH_DIR}/${pose_model}/${version}"
  done
done

for version in "${ALPHAPOSE_VERSIONS[@]}"; do
  mkdir -p "${BENCH_DIR}/${ALPHAPOSE_POSE_MODEL}/${version}"
done

for version in "${VITPOSE_VERSIONS[@]}"; do
  mkdir -p "${BENCH_DIR}/${VITPOSE_POSE_MODEL}/${version}"
done

run_idx=0
success_count=0
failure_count=0
skip_count=0

for pose_model in "${POSE_MODELS[@]}"; do
  for version in "${VERSIONS[@]}"; do
    for classifier in "${CLASSIFIERS[@]}"; do
      run_idx=$((run_idx + 1))

      run_one_benchmark "${run_idx}" "${pose_model}" "${version}" "${classifier}"
      rc=$?

      if [[ "${rc}" -eq 0 ]]; then
        success_count=$((success_count + 1))
      elif [[ "${rc}" -eq 2 ]]; then
        skip_count=$((skip_count + 1))
      else
        failure_count=$((failure_count + 1))
      fi
    done
  done
done

for version in "${ALPHAPOSE_VERSIONS[@]}"; do
  for classifier in "${CLASSIFIERS[@]}"; do
    run_idx=$((run_idx + 1))

    run_one_alphapose_benchmark "${run_idx}" "${version}" "${classifier}"
    rc=$?

    if [[ "${rc}" -eq 0 ]]; then
      success_count=$((success_count + 1))
    elif [[ "${rc}" -eq 2 ]]; then
      skip_count=$((skip_count + 1))
    else
      failure_count=$((failure_count + 1))
    fi
  done
done

for version in "${VITPOSE_VERSIONS[@]}"; do
  for classifier in "${CLASSIFIERS[@]}"; do
    run_idx=$((run_idx + 1))

    run_one_vitpose_benchmark "${run_idx}" "${version}" "${classifier}"
    rc=$?

    if [[ "${rc}" -eq 0 ]]; then
      success_count=$((success_count + 1))
    elif [[ "${rc}" -eq 2 ]]; then
      skip_count=$((skip_count + 1))
    else
      failure_count=$((failure_count + 1))
    fi
  done
done

echo
echo "Benchmarking complete."
echo "  New successful runs: ${success_count}"
echo "  Skipped runs:        ${skip_count}"
echo "  Failed runs:         ${failure_count}"
echo "  Success log:         ${BENCH_DIR}/successful_runs.log"
echo "  Skipped log:         ${BENCH_DIR}/skipped_runs.log"
echo "  Failure log:         ${BENCH_DIR}/failed_runs.log"

exit 0
