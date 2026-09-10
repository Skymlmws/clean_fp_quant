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
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b/givens-mxfp-w4a4/stratified-32-seed0}"
DEVICE_ID="${DEVICE_ID:-3}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
DRY_RUN="${DRY_RUN:-0}"

ARGS=(
    --metadata "${PROMPT_ROOT}/VBench_full_info.json"
    --augmented-prompts "${PROMPT_ROOT}/all_dimension_aug_wanx_seed42.txt"
    --checkpoint "${CHECKPOINT}" --wan-repo "${WAN_REPO}"
    --output-dir "${OUTPUT_DIR}" --prompt-count 32 --selection-seed 20260903
    --sample-seeds 0 --device-id "${DEVICE_ID}" --rank "${RANK}" --world-size "${WORLD_SIZE}"
    --transform-class givens --transform-group-size 32 --outlier-threshold 5 --quant-group-size 32
    --weight-observer minmax --calibration-prompt-index 0
)
if [[ "${DRY_RUN}" == "1" ]]; then ARGS+=(--dry-run); fi
cd "${BASELINE_ROOT}"
exec "${PYTHON}" -m scripts.generate.generate_wan_vbench_quant_batch "${ARGS[@]}"
