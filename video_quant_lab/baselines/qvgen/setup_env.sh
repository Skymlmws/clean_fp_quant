#!/usr/bin/env bash
set -euo pipefail

BASELINE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UV="${UV:-uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

"${UV}" venv --python "${PYTHON_VERSION}" "${BASELINE_DIR}/.venv"
"${UV}" pip install \
  --python "${BASELINE_DIR}/.venv/bin/python" \
  -r "${BASELINE_DIR}/requirements.txt"
