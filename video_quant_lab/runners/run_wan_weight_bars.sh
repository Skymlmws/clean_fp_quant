#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-/home/maoliming/project/.venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
BLOCKS="${BLOCKS:-0}"
SITES="${SITES:-ffn_in}"
LINEARS="${LINEARS:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/profiles/wan-weight-bars}"

cd "${PROJECT_ROOT}"
"${PYTHON}" -m video_quant_lab.analysis.cli.visualize_wan_weight_bars \
  --device "${DEVICE}" --blocks "${BLOCKS}" --sites "${SITES}" \
  --linears "${LINEARS}" --output-dir "${OUTPUT_DIR}" "$@"
