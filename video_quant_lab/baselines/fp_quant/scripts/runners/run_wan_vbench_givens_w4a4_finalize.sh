#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b/givens-mxfp-w4a4/stratified-32-seed0}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/vbench_results/wan2.1-t2v-1.3b/stratified-32-seed0}"
RESULT_DIR="${RESULT_DIR:-${EXPERIMENT_DIR}/methods/givens-mxfp-w4a4}"
BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-${EXPERIMENT_DIR}/methods/bf16}"
METADATA="${METADATA:-${PROJECT_ROOT}/video_quant_lab/prompts/vbench-official/VBench_full_info.json}"
VBENCH_ROOT="${VBENCH_ROOT:-/home/maoliming/VBench}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/home/maoliming/.venv-vbench/bin/python}"
EXPECTED_VIDEOS="${EXPECTED_VIDEOS:-32}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_QUALITY="${GPU_QUALITY:-3}"
GPU_SEMANTIC="${GPU_SEMANTIC:-6}"
GPU_OBJECT="${GPU_OBJECT:-7}"
METHOD="${METHOD:-Givens + MXFP W4A4, threshold 5}"

mkdir -p "${RESULT_DIR}/logs"
while true; do
    count="$(find "${VIDEO_DIR}" -maxdepth 1 -name '*.mp4' -type f -size +0c | wc -l)"
    printf '%s videos=%s/%s\n' "$(date --iso-8601=seconds)" "${count}" "${EXPECTED_VIDEOS}"
    if [[ "${count}" -ge "${EXPECTED_VIDEOS}" ]]; then break; fi
    sleep "${POLL_SECONDS}"
done

# VBench infers the video suffix from the first directory entry. Generation
# directories also contain plan/result JSON files, so expose a clean MP4-only
# view to avoid VBench accidentally looking for prompt-0.json files.
EVAL_VIDEO_DIR="${RESULT_DIR}/eval_videos"
mkdir -p "${EVAL_VIDEO_DIR}"
for video in "${VIDEO_DIR}"/*.mp4; do
    [[ -s "${video}" ]] || continue
    ln -sfn "${video}" "${EVAL_VIDEO_DIR}/$(basename -- "${video}")"
done
eval_count="$(find "${EVAL_VIDEO_DIR}" -maxdepth 1 -name '*.mp4' -type l | wc -l)"
if [[ "${eval_count}" -lt "${EXPECTED_VIDEOS}" ]]; then
    printf 'evaluation staging incomplete: %s/%s\n' "${eval_count}" "${EXPECTED_VIDEOS}" >&2
    exit 1
fi

evaluate() {
    local gpu="$1" port="$2" output="$3"
    shift 3
    mkdir -p "${RESULT_DIR}/${output}"
    if compgen -G "${RESULT_DIR}/${output}/*_eval_results.json" >/dev/null; then
        printf 'skip completed evaluation shard: %s\n' "${output}"
        return 0
    fi
    env CUDA_VISIBLE_DEVICES="${gpu}" MASTER_PORT="${port}" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        PYTHONPATH="${VBENCH_ROOT}" VBENCH_CACHE_DIR="${PROJECT_ROOT}/vbench_cache" \
        "${VBENCH_PYTHON}" "${VBENCH_ROOT}/evaluate.py" \
        --videos_path "${EVAL_VIDEO_DIR}" --output_path "${RESULT_DIR}/${output}" \
        --full_json_dir "${METADATA}" --dimension "$@" \
        --load_ckpt_from_local True --mode vbench_standard
}

evaluate "${GPU_QUALITY}" 29631 quality_temporal \
    subject_consistency background_consistency temporal_flickering \
    motion_smoothness dynamic_degree aesthetic_quality imaging_quality \
    >"${RESULT_DIR}/logs/quality_temporal.log" 2>&1 & quality_pid=$!

evaluate "${GPU_SEMANTIC}" 29632 semantic_style \
    human_action scene temporal_style appearance_style overall_consistency \
    >"${RESULT_DIR}/logs/semantic_style.log" 2>&1 & semantic_pid=$!

(
    evaluate "${GPU_OBJECT}" 29633 object_class object_class
    evaluate "${GPU_OBJECT}" 29634 multiple_objects multiple_objects
    evaluate "${GPU_OBJECT}" 29635 spatial_relationship spatial_relationship
    evaluate "${GPU_OBJECT}" 29636 color color || true
) >"${RESULT_DIR}/logs/object_spatial.log" 2>&1 & object_pid=$!

status=0
wait "${quality_pid}" || status=1
wait "${semantic_pid}" || status=1
wait "${object_pid}" || status=1

"${VBENCH_PYTHON}" \
    "${PROJECT_ROOT}/video_quant_lab/baselines/fp_quant/scripts/evaluate/summarize_vbench.py" \
    --result-dir "${RESULT_DIR}" --baseline-result-dir "${BASELINE_RESULT_DIR}" \
    --video-dir "${VIDEO_DIR}" --method "${METHOD}"
"${VBENCH_PYTHON}" \
    "${PROJECT_ROOT}/video_quant_lab/baselines/fp_quant/scripts/evaluate/summarize_vbench_comparison.py" \
    --experiment-dir "${EXPERIMENT_DIR}"
exit "${status}"
