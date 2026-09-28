#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/clean_fp_quant}"
BASELINE_ROOT="${PROJECT_ROOT}/video_quant_lab/baselines/fp_quant"
PYTHON="${PYTHON:-/root/diffusers/.venv/bin/python}"
MODEL="${MODEL:-/root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/outputs/wan-msfp-real-calibration-long10.json}"
PROMPT_ROOT="${PROJECT_ROOT}/video_quant_lab/prompts/vbench-1.0-mini-0.05"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/vbench/wan2.1-t2v-1.3b-msfp}"
DEVICE="${DEVICE:-cuda:0}"
VARIANTS="${VARIANTS:-mixup signed}"
LIMIT="${LIMIT:-}"

export PYTHONPATH="/root/diffusers/src:${BASELINE_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${BASELINE_ROOT}"

for variant in ${VARIANTS}; do
    args=(
        -m scripts.generate.generate_wan_msfp_vbench_batch
        --model "${MODEL}"
        --variant "${variant}"
        --metadata "${PROMPT_ROOT}/VBench_kmeans_info_0.05.json"
        --augmented-prompts "${PROMPT_ROOT}/all_dimension_aug_wanx_seed42_0.05.txt"
        --output-dir "${OUTPUT_ROOT}/${variant}/vbench-mini-43-seed0"
        --prompt-count 43 --seed 0 --device "${DEVICE}" --min-free-gib 5
    )
    if [[ "${variant}" != "bf16" ]]; then
        args+=(--calibration-config "${CONFIG}")
    fi
    if [[ -n "${LIMIT}" ]]; then
        args+=(--limit "${LIMIT}")
    fi
    "${PYTHON}" "${args[@]}"
done
