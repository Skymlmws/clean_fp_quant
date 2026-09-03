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
DEVICE_ID="${DEVICE_ID:-2}"

PROMPT="${PROMPT:-A small red panda walking in a bamboo forest.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
WIDTH="${WIDTH:-128}"
HEIGHT="${HEIGHT:-128}"
FRAMES="${FRAMES:-5}"
FPS="${FPS:-16}"
STEPS="${STEPS:-4}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
SHIFT="${SHIFT:-5.0}"
SEED="${SEED:-0}"

TRANSFORM_CLASS="${TRANSFORM_CLASS:-givens}"
TRANSFORM_GROUP_SIZE="${TRANSFORM_GROUP_SIZE:-32}"
OUTLIER_THRESHOLD="${OUTLIER_THRESHOLD:-5}"
WEIGHT_BITS="${WEIGHT_BITS:-4}"
ACTIVATION_BITS="${ACTIVATION_BITS:-4}"
QUANT_GROUP_SIZE="${QUANT_GROUP_SIZE:-32}"
WEIGHT_OBSERVER="${WEIGHT_OBSERVER:-minmax}"
REFERENCE_TENSOR="${REFERENCE_TENSOR:-}"
REFERENCE_ONLY="${REFERENCE_ONLY:-0}"

RUN_NAME="${RUN_NAME:-wan1.3b-${TRANSFORM_CLASS}-mxfp4-w${WEIGHT_BITS}a${ACTIVATION_BITS}-${WIDTH}x${HEIGHT}-${FRAMES}f-${STEPS}steps-seed${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/video-quantization-runs/${RUN_NAME}}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python environment not found or not executable: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT}/diffusion_pytorch_model.safetensors" ]]; then
    echo "Wan checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -d "${WAN_REPO}/wan" ]]; then
    echo "Wan repository not found: ${WAN_REPO}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${BASELINE_ROOT}"

REFERENCE_ARGS=()
if [[ -n "${REFERENCE_TENSOR}" ]]; then
    REFERENCE_ARGS=(--reference-tensor "${REFERENCE_TENSOR}")
fi
if [[ "${REFERENCE_ONLY}" == "1" ]]; then
    REFERENCE_ARGS+=(--reference-only)
fi

echo "Running Wan2.1 ${TRANSFORM_CLASS} + MXFP quantization video generation"
echo "GPU: cuda:${DEVICE_ID}"
echo "Output: ${OUTPUT_DIR}"

"${PYTHON}" -m scripts.generate.generate_wan_givens_video \
    --checkpoint "${CHECKPOINT}" \
    --wan-repo "${WAN_REPO}" \
    --device-id "${DEVICE_ID}" \
    --prompt "${PROMPT}" \
    --negative-prompt "${NEGATIVE_PROMPT}" \
    --width "${WIDTH}" \
    --height "${HEIGHT}" \
    --frames "${FRAMES}" \
    --fps "${FPS}" \
    --steps "${STEPS}" \
    --guide-scale "${GUIDE_SCALE}" \
    --shift "${SHIFT}" \
    --seed "${SEED}" \
    --transform-class "${TRANSFORM_CLASS}" \
    --transform-group-size "${TRANSFORM_GROUP_SIZE}" \
    --outlier-threshold "${OUTLIER_THRESHOLD}" \
    --weight-bits "${WEIGHT_BITS}" \
    --activation-bits "${ACTIVATION_BITS}" \
    --quant-group-size "${QUANT_GROUP_SIZE}" \
    --weight-observer "${WEIGHT_OBSERVER}" \
    --output-dir "${OUTPUT_DIR}" \
    "${REFERENCE_ARGS[@]}"
