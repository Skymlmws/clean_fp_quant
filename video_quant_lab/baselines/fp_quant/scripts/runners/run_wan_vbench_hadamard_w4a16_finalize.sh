#!/usr/bin/env bash
set -euo pipefail

RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
export VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b/hadamard-mxfp4-w4a16/stratified-32-seed0}"
export RESULT_DIR="${RESULT_DIR:-${PROJECT_ROOT}/vbench_results/wan2.1-t2v-1.3b/stratified-32-seed0/methods/hadamard-mxfp4-w4a16}"
export METHOD="${METHOD:-Randomized Hadamard H32 + MXFP4 W4A16}"
export GPU_QUALITY="${GPU_QUALITY:-0}" GPU_SEMANTIC="${GPU_SEMANTIC:-2}" GPU_OBJECT="${GPU_OBJECT:-3}"
exec "${RUNNER_DIR}/run_wan_vbench_givens_w4a4_finalize.sh"
