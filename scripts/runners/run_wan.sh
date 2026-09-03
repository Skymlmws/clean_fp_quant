#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

PROJECT_ROOT=${PROJECT_ROOT:-"/home/maoliming/FP-Quant"}
PYTHON=${PYTHON:-"/home/maoliming/project/.venv/bin/python"}
CHECKPOINT=${CHECKPOINT:-"/home/maoliming/project/checkpoints/Wan2.1-T2V-1.3B"}
WAN_REPO=${WAN_REPO:-"/home/maoliming/project/wan2.1"}
DEVICE=${DEVICE:-"cuda:2"}

# Only RTN is currently implemented for Wan.
QUANT_METHOD=${QUANT_METHOD:-"rtn"}
if [[ "${QUANT_METHOD}" != "rtn" ]]; then
    echo "Wan currently supports only QUANT_METHOD=rtn; got '${QUANT_METHOD}'." >&2
    exit 1
fi

# Quantization configuration.
FORMAT=${FORMAT:-"mxfp"}
W_BITS=${W_BITS:-4}
A_BITS=${A_BITS:-4}
QUANT_GROUP_SIZE=${QUANT_GROUP_SIZE:-32}
SCALE_PRECISION=${SCALE_PRECISION:-"e8m0"}
W_OBSERVER=${W_OBSERVER:-"minmax"}

# Transform and calibration configuration.
TRANSFORM_CLASS=${TRANSFORM_CLASS:-"givens"}
TRANSFORM_GROUP_SIZE=${TRANSFORM_GROUP_SIZE:-32}
OUTLIER_THRESHOLD=${OUTLIER_THRESHOLD:-5}
TIMESTEPS=${TIMESTEPS:-"50 250 500 750 950"}
LATENT_FRAMES=${LATENT_FRAMES:-1}
LATENT_HEIGHT=${LATENT_HEIGHT:-8}
LATENT_WIDTH=${LATENT_WIDTH:-8}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32}
SEED=${SEED:-0}

OUTPUT_DIR=${OUTPUT_DIR:-"${PROJECT_ROOT}/outputs"}
RUN_NAME=${RUN_NAME:-"wan1.3b-${FORMAT}-w${W_BITS}a${A_BITS}-${QUANT_METHOD}-${TRANSFORM_CLASS}"}
OUTPUT=${OUTPUT:-"${OUTPUT_DIR}/${RUN_NAME}.json"}

mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON}" -m scripts.generate.quantize_wan \
    --checkpoint "${CHECKPOINT}" \
    --wan-repo "${WAN_REPO}" \
    --device "${DEVICE}" \
    --dtype bfloat16 \
    --format "${FORMAT}" \
    --weight-bits "${W_BITS}" \
    --activation-bits "${A_BITS}" \
    --quant-group-size "${QUANT_GROUP_SIZE}" \
    --scale-precision "${SCALE_PRECISION}" \
    --weight-observer "${W_OBSERVER}" \
    --transform-class "${TRANSFORM_CLASS}" \
    --transform-group-size "${TRANSFORM_GROUP_SIZE}" \
    --outlier-threshold "${OUTLIER_THRESHOLD}" \
    --timesteps ${TIMESTEPS} \
    --latent-frames "${LATENT_FRAMES}" \
    --latent-height "${LATENT_HEIGHT}" \
    --latent-width "${LATENT_WIDTH}" \
    --context-length "${CONTEXT_LENGTH}" \
    --seed "${SEED}" \
    --output "${OUTPUT}"
