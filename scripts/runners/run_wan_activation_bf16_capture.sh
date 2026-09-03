#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
PYTHON="${PYTHON:-/home/maoliming/project/.venv/bin/python}"
DEVICE_ID="${DEVICE_ID:-2}"
SEED="${SEED:-0}"
SAMPLING_STEPS="${SAMPLING_STEPS:-10,25,40}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/activation-visualization/wan-activation-short-prompt/wan-activation-480p-seed${SEED}-full-bf16}"

cd "${PROJECT_ROOT}"
"${PYTHON}" -m scripts.profile.capture_wan_activations_bf16 \
  --device-id "${DEVICE_ID}" --seed "${SEED}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --quota-dir "${PROJECT_ROOT}/outputs" --max-output-gb 200 \
  --output-dir "${OUTPUT_DIR}" "$@"
