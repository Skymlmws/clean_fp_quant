#!/usr/bin/env bash
set -euo pipefail

RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
EXPERIMENT_ID="experiment-a-per-prompt-seed0"

export VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b/givens-mxfp-w4a4/${EXPERIMENT_ID}}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/vbench_results/wan2.1-t2v-1.3b/${EXPERIMENT_ID}}"
export RESULT_DIR="${RESULT_DIR:-${EXPERIMENT_DIR}/methods/givens-mxfp-w4a4-per-prompt-calibration}"
export BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-${PROJECT_ROOT}/vbench_results/wan2.1-t2v-1.3b/stratified-32-seed0/methods/bf16}"
export GPU_QUALITY="${GPU_QUALITY:-0}"
export GPU_SEMANTIC="${GPU_SEMANTIC:-1}"
export GPU_OBJECT="${GPU_OBJECT:-2}"
export METHOD="${METHOD:-Givens + MXFP W4A4, per-prompt same-seed calibration}"

exec "${RUNNER_DIR}/run_wan_vbench_givens_w4a4_finalize.sh"
