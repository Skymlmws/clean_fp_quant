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
PROMPT_ROOT="${PROMPT_ROOT:-${PROJECT_ROOT}/video_quant_lab/prompts/vbench-1.0-mini-0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b/hadamard-mxfp-w4a4/vbench-mini-43-seed0}"
DEVICE_ID="${DEVICE_ID:-3}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
WORKER_INDEX="${WORKER_INDEX:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
DRY_RUN="${DRY_RUN:-0}"

ARGS=(
    --metadata "${PROMPT_ROOT}/VBench_kmeans_info_0.05.json"
    --augmented-prompts "${PROMPT_ROOT}/all_dimension_aug_wanx_seed42_0.05.txt"
    --checkpoint "${CHECKPOINT}" --wan-repo "${WAN_REPO}"
    --output-dir "${OUTPUT_DIR}" --prompt-count 43 --selection-mode all
    --suite-name vbench-1.0-mini-0.05-43
    --sample-seeds 0 --device-id "${DEVICE_ID}" --rank "${RANK}" --world-size "${WORLD_SIZE}"
    --worker-index "${WORKER_INDEX}" --worker-count "${WORKER_COUNT}"
    --transform-class hadamard --transform-group-size 32 --transform-randomize --transform-seed 0
    --quant-group-size 32 --weight-observer minmax --calibration-prompt-index 0
)
if [[ "${DRY_RUN}" == "1" ]]; then ARGS+=(--dry-run); fi
cd "${BASELINE_ROOT}"
exec "${PYTHON}" -m scripts.generate.generate_wan_vbench_quant_batch "${ARGS[@]}"
