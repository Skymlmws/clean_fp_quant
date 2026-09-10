#!/usr/bin/env bash
set -euo pipefail

RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TRANSFORM_CLASS=givens WEIGHT_BITS=4 ACTIVATION_BITS=16
exec "${RUNNER_DIR}/run_wan_vbench_mxfp.sh" "$@"
