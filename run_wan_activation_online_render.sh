#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
PYTHON="${PYTHON:-/home/maoliming/project/.venv/bin/python}"
DEVICE_ID="${DEVICE_ID:-2}"
SEED="${SEED:-0}"
SAMPLING_STEPS="${SAMPLING_STEPS:-10,25,40}"
MAX_OUTPUT_GB="${MAX_OUTPUT_GB:-0}"
RENDER_MODE="${RENDER_MODE:-multiprocess}"
RENDER_WORKERS="${RENDER_WORKERS:-4}"
SHARED_MEMORY_DIR="${SHARED_MEMORY_DIR:-/dev/shm}"
MAX_INFLIGHT_ACTIVATIONS="${MAX_INFLIGHT_ACTIVATIONS:-0}"
INFLIGHT_MEMORY_FRACTION="${INFLIGHT_MEMORY_FRACTION:-0.25}"
ISOLATED_GLOBAL_PERCENTILE="${ISOLATED_GLOBAL_PERCENTILE:-99.99}"
ISOLATED_CHANNEL_PERCENTILE="${ISOLATED_CHANNEL_PERCENTILE:-99.0}"
ISOLATED_RATIO="${ISOLATED_RATIO:-5.0}"
ISOLATED_MAX_TOKEN_FRACTION="${ISOLATED_MAX_TOKEN_FRACTION:-0.01}"
MARK_TOP_ISOLATED="${MARK_TOP_ISOLATED:-10}"
ISOLATED_MERGE_TOKEN_GAP="${ISOLATED_MERGE_TOKEN_GAP:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/activation-visualization/wan-activation-long-prompts/wan-activation-480p-seed${SEED}-long-prompt-online-heatmaps}"

cd "${PROJECT_ROOT}"
"${PYTHON}" capture_render_wan_activations_online.py \
  --device-id "${DEVICE_ID}" \
  --seed "${SEED}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --max-output-gb "${MAX_OUTPUT_GB}" \
  --render-mode "${RENDER_MODE}" \
  --render-workers "${RENDER_WORKERS}" \
  --shared-memory-dir "${SHARED_MEMORY_DIR}" \
  --max-inflight-activations "${MAX_INFLIGHT_ACTIVATIONS}" \
  --inflight-memory-fraction "${INFLIGHT_MEMORY_FRACTION}" \
  --isolated-global-percentile "${ISOLATED_GLOBAL_PERCENTILE}" \
  --isolated-channel-percentile "${ISOLATED_CHANNEL_PERCENTILE}" \
  --isolated-ratio "${ISOLATED_RATIO}" \
  --isolated-max-token-fraction "${ISOLATED_MAX_TOKEN_FRACTION}" \
  --mark-top-isolated "${MARK_TOP_ISOLATED}" \
  --isolated-merge-token-gap "${ISOLATED_MERGE_TOKEN_GAP}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
