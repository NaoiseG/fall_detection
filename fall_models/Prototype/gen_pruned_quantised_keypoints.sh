#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# Assumes this script is run from:
#   /home/jetson/NaoiseG/fall_detection/fall_models/Prototype
#
# Repo root will therefore be:
#   /home/jetson/NaoiseG/fall_detection

REPO_ROOT="$(cd ../.. && pwd)"
POSE_ROOT="${REPO_ROOT}/pose_models/full_pruned"
DATASET_ROOT="${REPO_ROOT}/Datasets/full_pruned"
UPFALL_ROOT="${REPO_ROOT}/Datasets/UPFall"

cd "${REPO_ROOT}"

run_cmd() {
    echo
    echo "Running:"
    printf '  %q' "$@"
    echo
    "$@"
}

processed=0

for weights_dir in "${POSE_ROOT}"/*/weights; do
    [[ -d "${weights_dir}" ]] || continue

    model_dir_name="$(basename "$(dirname "${weights_dir}")")"

    if [[ ! "${model_dir_name}" =~ ^(yolo11[nsmlx])_pruned_([0-9]+)$ ]]; then
        echo "Skipping unrecognized model directory: ${model_dir_name}"
        continue
    fi

    model_name="${BASH_REMATCH[1]}"      # e.g. yolo11s
    prune_level="${BASH_REMATCH[2]}"     # e.g. 80

    for engine_path in "${weights_dir}"/*.engine; do
        [[ -f "${engine_path}" ]] || continue

        engine_file="$(basename "${engine_path}")"

        case "${engine_file}" in
            *_fp32.engine) precision="fp32" ;;
            *_fp16.engine) precision="fp16" ;;
            *_int8.engine) precision="int8" ;;
            *)
                echo "Skipping engine with unknown precision: ${engine_file}"
                continue
                ;;
        esac

        output_root="${DATASET_ROOT}/${model_name}/pruned_${prune_level}/${precision}"
        mkdir -p "${output_root}"

        echo "============================================================"
        echo "Model dir   : ${model_dir_name}"
        echo "Engine      : ${engine_file}"
        echo "Precision   : ${precision}"
        echo "Output root : ${output_root}"
        echo "============================================================"

        run_cmd python -m dataset_helpers.get_keypoints_files \
            --subjects 16-17 \
            --camera 1 \
            --lock-settings default \
            --upfall-root "${UPFALL_ROOT}" \
            --output-root "${output_root}" \
            --model-path "${engine_path}"

        run_cmd python -m dataset_helpers.get_keypoints_files \
            --subjects 16-17 \
            --camera 2 \
            --lock-settings strict_lock \
            --upfall-root "${UPFALL_ROOT}" \
            --output-root "${output_root}" \
            --model-path "${engine_path}"

        processed=$((processed + 1))
    done
done

echo
echo "Done. Processed ${processed} engine files."