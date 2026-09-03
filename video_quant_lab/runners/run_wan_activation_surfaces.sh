#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-/home/maoliming/project/.venv/bin/python}"
DEVICE_ID="${DEVICE_ID:-2}"
WIDTH="${WIDTH:-128}"
HEIGHT="${HEIGHT:-128}"
FRAMES="${FRAMES:-5}"
STEPS="${STEPS:-3}"
SEED="${SEED:-0}"
BLOCKS="${BLOCKS:-all}"
SITES="${SITES:-ffn_in}"
CALL_INDEX="${CALL_INDEX:-0}"
CALL_INDICES="${CALL_INDICES:-}"
CAPTURE_ONLY="${CAPTURE_ONLY:-0}"
MAX_OUTPUT_GB="${MAX_OUTPUT_GB:-200}"
MAX_IMAGES="${MAX_IMAGES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/activation-visualization/wan-activation-surfaces-${WIDTH}x${HEIGHT}-${FRAMES}f-${STEPS}steps-seed${SEED}}"
QUOTA_DIR="${QUOTA_DIR:-${PROJECT_ROOT}/outputs}"

cd "${PROJECT_ROOT}"
EXTRA_ARGS=()
if [[ -n "${CALL_INDICES}" ]]; then
  EXTRA_ARGS+=(--call-indices "${CALL_INDICES}")
fi
if [[ "${CAPTURE_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--capture-only)
fi
"${PYTHON}" -m video_quant_lab.analysis.cli.visualize_wan_activation_surfaces \
  --device-id "${DEVICE_ID}" --width "${WIDTH}" --height "${HEIGHT}" \
  --frames "${FRAMES}" --steps "${STEPS}" --seed "${SEED}" \
  --blocks "${BLOCKS}" --sites "${SITES}" --call-index "${CALL_INDEX}" \
  --max-output-gb "${MAX_OUTPUT_GB}" --max-images "${MAX_IMAGES}" \
  --quota-dir "${QUOTA_DIR}" \
  --output-dir "${OUTPUT_DIR}" "${EXTRA_ARGS[@]}" "$@"
