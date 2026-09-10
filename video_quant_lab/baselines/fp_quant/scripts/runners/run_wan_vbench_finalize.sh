#!/usr/bin/env bash
set -euo pipefail

: "${METHOD_ID:?METHOD_ID is required}"
: "${METHOD:?METHOD is required}"
RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
export VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b/${METHOD_ID}/stratified-32-seed0}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/vbench_results/wan2.1-t2v-1.3b/stratified-32-seed0}"
export RESULT_DIR="${RESULT_DIR:-${EXPERIMENT_DIR}/methods/${METHOD_ID}}"
exec "${RUNNER_DIR}/run_wan_vbench_givens_w4a4_finalize.sh"
