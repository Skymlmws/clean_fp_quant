#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-/root/diffusers/.venv/bin/python}"
DEVICE_ID="${DEVICE_ID:-0}"
SAMPLING_STEPS="${SAMPLING_STEPS:-25}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/clean_fp_quant/outputs/activations/all-sites-step25-seed0}"
MAX_OUTPUT_GB="${MAX_OUTPUT_GB:-40}"

cd "${PROJECT_ROOT}"
exec "${PYTHON}" -m video_quant_lab.analysis.cli.capture_diffusers_wan_activations_bf16 \
  --model-path "${MODEL_PATH}" \
  --device-id "${DEVICE_ID}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-output-gb "${MAX_OUTPUT_GB}" \
  "$@"
