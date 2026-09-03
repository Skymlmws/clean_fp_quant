#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
PYTHON="${PYTHON:-/home/maoliming/project/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
BLOCKS="${BLOCKS:-0}"
SITES="${SITES:-ffn_in}"
LINEARS="${LINEARS:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/profiles/wan-weight-bars}"

cd "${PROJECT_ROOT}"
"${PYTHON}" -m scripts.visualize.visualize_wan_weight_bars \
  --device "${DEVICE}" --blocks "${BLOCKS}" --sites "${SITES}" \
  --linears "${LINEARS}" --output-dir "${OUTPUT_DIR}" "$@"
