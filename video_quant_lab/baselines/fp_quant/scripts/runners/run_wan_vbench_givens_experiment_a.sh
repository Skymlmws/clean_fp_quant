#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${BASELINE_ROOT}/../../.." && pwd)}"
export PYTHONPATH="${BASELINE_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-/home/maoliming/project/.venv/bin/python}"
CHECKPOINT="${CHECKPOINT:-/home/maoliming/project/checkpoints/Wan2.1-T2V-1.3B}"
WAN_REPO="${WAN_REPO:-/home/maoliming/project/wan2.1}"
PROMPT_ROOT="${PROMPT_ROOT:-${PROJECT_ROOT}/video_quant_lab/prompts/vbench-official}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b/givens-mxfp-w4a4/experiment-a-per-prompt-seed0}"
PROMPT_COUNT="${PROMPT_COUNT:-32}"
START_RANK="${START_RANK:-0}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"

mkdir -p "${OUTPUT_DIR}/logs"
read -r -a devices <<< "${GPU_IDS}"
if (( ${#devices[@]} == 0 )); then
    echo "GPU_IDS must contain at least one GPU index" >&2
    exit 2
fi
if (( START_RANK < 0 || START_RANK >= PROMPT_COUNT )); then
    echo "START_RANK must be in [0, PROMPT_COUNT)" >&2
    exit 2
fi

run_worker() {
    local worker_index="$1"
    local device_id="$2"
    local rank
    for ((rank = START_RANK + worker_index; rank < PROMPT_COUNT; rank += ${#devices[@]})); do
        echo "Starting experiment A rank ${rank} on GPU ${device_id}"
        "${PYTHON}" -m scripts.generate.generate_wan_vbench_quant_batch \
            --metadata "${PROMPT_ROOT}/VBench_full_info.json" \
            --augmented-prompts "${PROMPT_ROOT}/all_dimension_aug_wanx_seed42.txt" \
            --checkpoint "${CHECKPOINT}" --wan-repo "${WAN_REPO}" \
            --output-dir "${OUTPUT_DIR}" --prompt-count "${PROMPT_COUNT}" \
            --selection-seed 20260903 --sample-seeds 0 \
            --device-id "${device_id}" --rank "${rank}" --world-size "${PROMPT_COUNT}" \
            --transform-class givens --transform-group-size 32 --outlier-threshold 5 \
            --quant-group-size 32 --weight-observer minmax \
            --calibration-prompt-index "${rank}" \
            >"${OUTPUT_DIR}/logs/rank${rank}-gpu${device_id}.log" 2>&1
        echo "Finished experiment A rank ${rank} on GPU ${device_id}"
    done
}

pids=()
for worker_index in "${!devices[@]}"; do
    run_worker "${worker_index}" "${devices[$worker_index]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"
