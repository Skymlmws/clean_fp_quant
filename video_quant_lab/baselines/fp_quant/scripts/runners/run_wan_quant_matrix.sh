#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${BASELINE_ROOT}/../../.." && pwd)}"
WAN_VIDEO_RUNNER="${BASELINE_ROOT}/scripts/runners/run_wan_givens_video.sh"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
WIDTH="${WIDTH:-832}"
HEIGHT="${HEIGHT:-480}"
FRAMES="${FRAMES:-81}"
FPS="${FPS:-16}"
STEPS="${STEPS:-50}"
SEED="${SEED:-0}"
PROMPT="${PROMPT:-A small red panda walking in a bamboo forest.}"
OUTLIER_THRESHOLD="${OUTLIER_THRESHOLD:-3}"
MATRIX_ROOT="${MATRIX_ROOT:-${PROJECT_ROOT}/outputs/video-quantization-runs/matrix-${WIDTH}x${HEIGHT}-${FRAMES}f-${STEPS}steps-seed${SEED}}"
REFERENCE_TENSOR="${REFERENCE_TENSOR:-${MATRIX_ROOT}/bf16_reference.pt}"

mkdir -p "${MATRIX_ROOT}"
cd "${BASELINE_ROOT}"

common_env() {
    env \
        WIDTH="${WIDTH}" HEIGHT="${HEIGHT}" FRAMES="${FRAMES}" FPS="${FPS}" \
        STEPS="${STEPS}" SEED="${SEED}" PROMPT="${PROMPT}" \
        OUTLIER_THRESHOLD="${OUTLIER_THRESHOLD}" REFERENCE_TENSOR="${REFERENCE_TENSOR}" \
        "$@"
}

if [[ ! -f "${REFERENCE_TENSOR}" ]]; then
    echo "Creating shared BF16 reference on GPU ${GPU_A}"
    common_env DEVICE_ID="${GPU_A}" TRANSFORM_CLASS=identity WEIGHT_BITS=16 \
        ACTIVATION_BITS=16 REFERENCE_ONLY=1 \
        OUTPUT_DIR="${MATRIX_ROOT}/bf16" "${WAN_VIDEO_RUNNER}"
fi

run_arm() {
    local gpu="$1" transform="$2" weight_bits="$3" activation_bits="$4"
    local method="${transform}-w${weight_bits}a${activation_bits}"
    echo "[GPU ${gpu}] ${method}"
    common_env DEVICE_ID="${gpu}" TRANSFORM_CLASS="${transform}" \
        WEIGHT_BITS="${weight_bits}" ACTIVATION_BITS="${activation_bits}" \
        OUTPUT_DIR="${MATRIX_ROOT}/${method}" \
        RUN_NAME="${method}" "${WAN_VIDEO_RUNNER}" \
        >"${MATRIX_ROOT}/${method}.log" 2>&1
}

queue_a() {
    run_arm "${GPU_A}" identity 16 4
    run_arm "${GPU_A}" identity 4 16
    run_arm "${GPU_A}" identity 4 4
    run_arm "${GPU_A}" givens 16 4
    run_arm "${GPU_A}" givens 4 4
}

queue_b() {
    run_arm "${GPU_B}" hadamard 16 4
    run_arm "${GPU_B}" hadamard 4 16
    run_arm "${GPU_B}" hadamard 4 4
    run_arm "${GPU_B}" givens 4 16
}

queue_a & pid_a=$!
queue_b & pid_b=$!
wait "${pid_a}"
wait "${pid_b}"

echo "Matrix complete: ${MATRIX_ROOT}"
