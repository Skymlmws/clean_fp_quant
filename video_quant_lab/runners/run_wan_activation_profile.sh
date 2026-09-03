#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-/home/maoliming/project/.venv/bin/python}"
DEVICE_ID="${DEVICE_ID:-2}"
WIDTH="${WIDTH:-128}"
HEIGHT="${HEIGHT:-128}"
FRAMES="${FRAMES:-5}"
STEPS="${STEPS:-4}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/profiles/wan-w16a16-profile-${WIDTH}x${HEIGHT}-${FRAMES}f-${STEPS}steps-seed${SEED}}"

cd "${PROJECT_ROOT}"
"${PYTHON}" -m video_quant_lab.analysis.cli.visualize_wan_activations \
  --device-id "${DEVICE_ID}" --width "${WIDTH}" --height "${HEIGHT}" \
  --frames "${FRAMES}" --steps "${STEPS}" --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" "$@"
