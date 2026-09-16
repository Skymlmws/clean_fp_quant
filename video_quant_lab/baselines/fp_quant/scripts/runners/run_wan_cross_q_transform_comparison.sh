#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${BASELINE_ROOT}/../../.." && pwd)}"
export PYTHONPATH="${BASELINE_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON="${PYTHON:-/home/maoliming/project/.venv/bin/python}"
DEVICE_ID="${DEVICE_ID:-0}"
PROMPT_ID="${PROMPT_ID:-autumn_station}"
SEED="${SEED:-0}"
TRANSFORM_SEED="${TRANSFORM_SEED:-0}"
SAMPLING_STEPS="${SAMPLING_STEPS:-10,25,40}"
BLOCKS="${BLOCKS:-all}"
RENDER_WORKERS="${RENDER_WORKERS:-4}"
MAX_INFLIGHT_ACTIVATIONS="${MAX_INFLIGHT_ACTIVATIONS:-6}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/activation-visualization/wan-cross-q-transform-comparison/${PROMPT_ID}-seed${SEED}}"

cd "${BASELINE_ROOT}"
exec "${PYTHON}" -m scripts.visualize.compare_wan_cross_q_transforms \
  --device-id "${DEVICE_ID}" \
  --prompt-file "${PROJECT_ROOT}/video_quant_lab/prompts/wan_activation_long_prompts.json" \
  --prompt-id "${PROMPT_ID}" \
  --seed "${SEED}" \
  --transform-seed "${TRANSFORM_SEED}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --blocks "${BLOCKS}" \
  --render-workers "${RENDER_WORKERS}" \
  --max-inflight-activations "${MAX_INFLIGHT_ACTIVATIONS}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
