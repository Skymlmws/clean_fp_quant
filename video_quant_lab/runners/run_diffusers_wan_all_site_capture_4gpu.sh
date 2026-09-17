#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON="${PYTHON:-/root/diffusers/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers}"
SAMPLING_STEPS="${SAMPLING_STEPS:-10}"
SEED="${SEED:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/clean_fp_quant/outputs/activations/all-sites-step10-seed0}"
MAX_OUTPUT_GB_PER_SHARD="${MAX_OUTPUT_GB_PER_SHARD:-12}"

block_ranges=("0-7" "8-15" "16-22" "23-29")
mkdir -p "${OUTPUT_DIR}/logs"
cd "${PROJECT_ROOT}"

pids=()
for shard in 0 1 2 3; do
  shard_dir="${OUTPUT_DIR}/shards/shard-${shard}"
  "${PYTHON}" -m video_quant_lab.analysis.cli.capture_diffusers_wan_activations_bf16 \
    --model-path "${MODEL_PATH}" \
    --device-id "${shard}" \
    --seed "${SEED}" \
    --sampling-steps "${SAMPLING_STEPS}" \
    --blocks "${block_ranges[shard]}" \
    --sites all \
    --max-output-gb "${MAX_OUTPUT_GB_PER_SHARD}" \
    --output-dir "${shard_dir}" \
    >"${OUTPUT_DIR}/logs/shard-${shard}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != 0 ]]; then
  echo "At least one capture shard failed; inspect ${OUTPUT_DIR}/logs" >&2
  exit 1
fi

"${PYTHON}" -m video_quant_lab.analysis.cli.consolidate_wan_activation_shards \
  --dataset-dir "${OUTPUT_DIR}" \
  --expected-activations 210
