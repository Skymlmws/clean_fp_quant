#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b/givens-mxfp-w4a4/experiment-a-per-prompt-seed0}"
RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/vbench_results/wan2.1-t2v-1.3b/experiment-a-per-prompt-seed0-partial-6/methods/givens-mxfp-w4a4-per-prompt-calibration}"
METADATA="${METADATA:-${PROJECT_ROOT}/video_quant_lab/prompts/vbench-official/VBench_full_info.json}"
VBENCH_ROOT="${VBENCH_ROOT:-/home/maoliming/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/home/maoliming/.venv-vbench/bin/python}"

count="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.mp4' -type f -size +0c | wc -l)"
if [[ "${count}" -ne 6 ]]; then
    echo "Expected exactly 6 completed videos, found ${count}" >&2
    exit 1
fi

EVAL_VIDEO_DIR="${RESULT_DIR}/eval_videos"
mkdir -p "${EVAL_VIDEO_DIR}" "${RESULT_DIR}/logs"
for video in "${VIDEO_DIR}"/*.mp4; do
    ln -sfn "${video}" "${EVAL_VIDEO_DIR}/$(basename -- "${video}")"
done

evaluate() {
    local gpu="$1" port="$2" output="$3"
    shift 3
    mkdir -p "${RESULT_DIR}/${output}"
    env CUDA_VISIBLE_DEVICES="${gpu}" MASTER_PORT="${port}" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        PYTHONPATH="${VBENCH_ROOT}" VBENCH_CACHE_DIR="${PROJECT_ROOT}/vbench_cache" \
        "${VBENCH_PYTHON}" "${VBENCH_ROOT}/evaluate.py" \
        --videos_path "${EVAL_VIDEO_DIR}" --output_path "${RESULT_DIR}/${output}" \
        --full_json_dir "${METADATA}" --dimension "$@" \
        --load_ckpt_from_local True --mode vbench_standard
}

evaluate 0 29741 supported_quality_temporal \
    subject_consistency background_consistency temporal_flickering \
    motion_smoothness dynamic_degree \
    >"${RESULT_DIR}/logs/supported_quality_temporal.log" 2>&1 & quality_pid=$!

evaluate 1 29742 supported_scene scene \
    >"${RESULT_DIR}/logs/supported_scene.log" 2>&1 & scene_pid=$!

status=0
wait "${quality_pid}" || status=1
wait "${scene_pid}" || status=1

empty_baseline="${RESULT_DIR}/no_baseline"
mkdir -p "${empty_baseline}"
"${VBENCH_PYTHON}" \
    "${PROJECT_ROOT}/video_quant_lab/baselines/fp_quant/scripts/evaluate/summarize_vbench.py" \
    --result-dir "${RESULT_DIR}" --baseline-result-dir "${empty_baseline}" \
    --video-dir "${VIDEO_DIR}" \
    --method "Givens + MXFP W4A4, per-prompt same-seed calibration" \
    --sample-set "partial subset: first 6 stratified prompts, seed 0" \
    --note "Partial diagnostic result over 6 videos; it is not directly comparable to a 32-video aggregate." \
    --note "Each video used its own prompt and seed-0 BF16 trajectory for Givens calibration."
exit "${status}"
