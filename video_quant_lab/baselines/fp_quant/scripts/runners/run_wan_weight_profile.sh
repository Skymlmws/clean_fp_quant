#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${BASELINE_ROOT}/../../.." && pwd)}"
export PYTHONPATH="${BASELINE_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${PYTHON:-/home/maoliming/project/.venv/bin/python}"
DEVICE_ID="${DEVICE_ID:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/profiles/wan-w16-weight-profile}"

cd "${BASELINE_ROOT}"
"${PYTHON}" -m scripts.visualize.visualize_wan_weights --device-id "${DEVICE_ID}" --output-dir "${OUTPUT_DIR}" "$@"
