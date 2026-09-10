#!/usr/bin/env bash
set -euo pipefail

RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TRANSFORM_CLASS=hadamard TRANSFORM_RANDOMIZE=1 TRANSFORM_SEED=0
export WEIGHT_BITS=4 ACTIVATION_BITS=16
exec "${RUNNER_DIR}/run_wan_vbench_mxfp.sh" "$@"
