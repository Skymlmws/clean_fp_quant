#!/usr/bin/env bash
set -euo pipefail

RUNNER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/home/maoliming/FP-Quant}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b/givens-mxfp-w4a4/experiment-a-per-prompt-seed0}"
WAIT_FOR_VIDEOS="${WAIT_FOR_VIDEOS:-12}"
POLL_SECONDS="${POLL_SECONDS:-60}"

export PROJECT_ROOT OUTPUT_DIR
while true; do
    count="$(find "${OUTPUT_DIR}" -maxdepth 1 -name '*.mp4' -type f -size +0c | wc -l)"
    printf '%s waiting_for_current_workers=%s/%s\n' \
        "$(date --iso-8601=seconds)" "${count}" "${WAIT_FOR_VIDEOS}"
    if [[ "${count}" -ge "${WAIT_FOR_VIDEOS}" ]]; then
        break
    fi
    sleep "${POLL_SECONDS}"
done

START_RANK="${START_RANK:-12}" "${RUNNER_DIR}/run_wan_vbench_givens_experiment_a.sh"
"${RUNNER_DIR}/run_wan_vbench_givens_experiment_a_finalize.sh"
