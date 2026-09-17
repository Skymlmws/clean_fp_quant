#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${BASELINE_ROOT}/../../.." && pwd)}"
PYTHON="${PYTHON:-/root/diffusers/.venv/bin/python}"
ACTIVATION_DIR="${ACTIVATION_DIR:-/root/autodl-tmp/clean_fp_quant/outputs/activations/all-sites-step25-seed0}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/clean_fp_quant/outputs/comparisons/all-sites-step25-seed0}"
NUM_SHARDS="${NUM_SHARDS:-4}"

mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/logs"
cd "${BASELINE_ROOT}"
export PYTHONPATH="${BASELINE_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  device=$((shard % 4))
  "${PYTHON}" -m scripts.visualize.compare_wan_saved_activations \
    --activation-dir "${ACTIVATION_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --device-id "${device}" \
    --shard-index "${shard}" \
    --num-shards "${NUM_SHARDS}" \
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
  echo "At least one comparison shard failed; inspect ${OUTPUT_DIR}/logs" >&2
  exit 1
fi

"${PYTHON}" -m scripts.visualize.compare_wan_saved_activations \
  --activation-dir "${ACTIVATION_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --merge-only
