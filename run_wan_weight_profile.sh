#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
PYTHON="${PYTHON:-/home/maoliming/project/.venv/bin/python}"
DEVICE_ID="${DEVICE_ID:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/profiles/wan-w16-weight-profile}"

cd "${PROJECT_ROOT}"
"${PYTHON}" visualize_wan_weights.py --device-id "${DEVICE_ID}" --output-dir "${OUTPUT_DIR}" "$@"
