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
TRANSFORM_CLASS="${TRANSFORM_CLASS:-identity}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACTIVATION_BITS="${ACTIVATION_BITS:-4}"
QUANT_SCOPE="${QUANT_SCOPE:-all}"
ATTENTION_TRANSFORM_CLASS="${ATTENTION_TRANSFORM_CLASS:-}"
FFN_TRANSFORM_CLASS="${FFN_TRANSFORM_CLASS:-}"
SCOPE_SUFFIX=""
if [[ "${QUANT_SCOPE}" != "all" ]]; then SCOPE_SUFFIX="-${QUANT_SCOPE}"; fi
if [[ -n "${ATTENTION_TRANSFORM_CLASS}" || -n "${FFN_TRANSFORM_CLASS}" ]]; then
    if [[ -z "${ATTENTION_TRANSFORM_CLASS}" || -z "${FFN_TRANSFORM_CLASS}" ]]; then
        echo "Both ATTENTION_TRANSFORM_CLASS and FFN_TRANSFORM_CLASS are required" >&2
        exit 2
    fi
    METHOD_ID="attn-${ATTENTION_TRANSFORM_CLASS}-ffn-${FFN_TRANSFORM_CLASS}-mxfp4-w${WEIGHT_BITS}a${ACTIVATION_BITS}"
else
    METHOD_ID="${TRANSFORM_CLASS}-mxfp4-w${WEIGHT_BITS}a${ACTIVATION_BITS}${SCOPE_SUFFIX}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b/${METHOD_ID}/stratified-32-seed0}"
DEVICE_ID="${DEVICE_ID:-0}"
RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-1}"
WORKER_INDEX="${WORKER_INDEX:-0}"
WORKER_COUNT="${WORKER_COUNT:-1}"
DRY_RUN="${DRY_RUN:-0}"
TRANSFORM_RANDOMIZE="${TRANSFORM_RANDOMIZE:-0}"
TRANSFORM_SEED="${TRANSFORM_SEED:-0}"

ARGS=(
    --metadata "${PROMPT_ROOT}/VBench_full_info.json"
    --augmented-prompts "${PROMPT_ROOT}/all_dimension_aug_wanx_seed42.txt"
    --checkpoint "${CHECKPOINT}" --wan-repo "${WAN_REPO}"
    --output-dir "${OUTPUT_DIR}" --prompt-count 32 --selection-seed 20260903
    --sample-seeds 0 --device-id "${DEVICE_ID}" --rank "${RANK}" --world-size "${WORLD_SIZE}"
    --worker-index "${WORKER_INDEX}" --worker-count "${WORKER_COUNT}"
    --transform-class "${TRANSFORM_CLASS}" --transform-group-size 32 --transform-seed "${TRANSFORM_SEED}"
    --quant-scope "${QUANT_SCOPE}"
    --weight-bits "${WEIGHT_BITS}" --activation-bits "${ACTIVATION_BITS}"
    --quant-group-size 32 --weight-observer minmax --calibration-prompt-index 0
)
if [[ "${TRANSFORM_RANDOMIZE}" == "1" ]]; then ARGS+=(--transform-randomize); fi
if [[ -n "${ATTENTION_TRANSFORM_CLASS}" ]]; then
    ARGS+=(--attention-transform-class "${ATTENTION_TRANSFORM_CLASS}")
fi
if [[ -n "${FFN_TRANSFORM_CLASS}" ]]; then
    ARGS+=(--ffn-transform-class "${FFN_TRANSFORM_CLASS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then ARGS+=(--dry-run); fi
cd "${BASELINE_ROOT}"
exec "${PYTHON}" -m scripts.generate.generate_wan_vbench_quant_batch "${ARGS[@]}"
