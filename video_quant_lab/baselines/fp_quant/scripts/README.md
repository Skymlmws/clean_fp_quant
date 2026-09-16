# Experiment scripts

The reusable implementation and these entry points live together inside the
`fp_quant` baseline directory:

- `generate/`: quantization and video generation
- `visualize/`: FP-Quant-specific visualization that imports its quantizer
- `runners/`: FP-Quant/Givens-specific shell launchers

Method-independent capture, rendering, and artifact tools now live under
`video_quant_lab/analysis/cli`. Existing shell runners remain compatibility
entry points under `video_quant_lab/runners/`.

Run Python entry points from `video_quant_lab/baselines/fp_quant` with module
syntax:

```shell
python -m scripts.generate.quantize_wan --help
```

The shell launchers locate that directory automatically, so they can be called
from the main project root.

`runners/run_wan_vbench_bf16.sh` runs a deterministic 32-prompt BF16 VBench
subset. Wan stays resident across the batch, completed MP4 files are skipped on
resume, and no decoded reference tensors are stored. The sampling parameters
and augmented prompts match the official VBench Wan2.1-T2V-1.3B configuration.

`runners/run_wan_vbench_givens_w4a4.sh` runs the matched Givens + MXFP W4A4
batch. It calibrates one fixed, recorded prompt before quantization, keeps the
quantized Wan pipeline resident, and writes MP4 files with the same names as the
BF16 batch for direct VBench comparison.

`runners/run_wan_vbench_hadamard_w4a4.sh` runs the calibration-free randomized
Hadamard + MXFP W4A4 baseline. It uses deterministic signed H32 rotations and
records the transform seed in the run plan.

`runners/run_wan_vbench_mxfp.sh` is the parameterized MXFP entry point. The
`identity_w4a4`, `identity_w4a16`, and `hadamard_w4a16` launchers are thin,
named configurations over it. Set `DEVICE_ID`, `RANK`, and `WORLD_SIZE` to
shard a matched run; set `DRY_RUN=1` to inspect its plan without loading Wan.
Set `QUANT_SCOPE=attention` to quantize only the self- and cross-attention
Q/K/V/O projections, or `QUANT_SCOPE=ffn` to quantize only `ffn.0` and
`ffn.2`. The default `QUANT_SCOPE=all` preserves the existing behavior and
experiment IDs. For example:

```shell
QUANT_SCOPE=attention TRANSFORM_CLASS=givens WEIGHT_BITS=4 ACTIVATION_BITS=4 \
  ./video_quant_lab/baselines/fp_quant/scripts/runners/run_wan_vbench_mxfp.sh
```

`runners/run_wan_vbench_givens_w4a4_finalize.sh` waits for all 32 generated
videos, evaluates the same VBench dimensions in three GPU shards, and writes a
method report containing BF16 deltas. It then refreshes the shared comparison
at `vbench_results/wan2.1-t2v-1.3b/stratified-32-seed0/comparison.md`. Raw
evaluation outputs are grouped under that experiment's `methods/` directory.

`runners/run_wan_baseline_pipeline.sh` supervises the next Wan baseline matrix
independently of an interactive Codex session. It discovers idle GPUs twice
before use and keeps at least two GPUs outside the pipeline by default. GPUs
already occupied by other workloads count toward those two, so if two or more
are already occupied, the pipeline may use all remaining idle GPUs. It resumes from non-empty
MP4 and VBench JSON outputs, retries failed stages, and refreshes the unified
comparison after evaluation. The configured queue is Identity W16A4,
randomized Hadamard H32 W16A4, and Givens W4A16.

```shell
./video_quant_lab/baselines/fp_quant/scripts/runners/run_wan_baseline_pipeline.sh start
./video_quant_lab/baselines/fp_quant/scripts/runners/run_wan_baseline_pipeline.sh status
./video_quant_lab/baselines/fp_quant/scripts/runners/run_wan_baseline_pipeline.sh logs
./video_quant_lab/baselines/fp_quant/scripts/runners/run_wan_baseline_pipeline.sh stop
```

`runners/run_wan_cross_q_transform_comparison.sh` runs the paired activation
experiment for persistent `cross_q` channel outliers. It first calibrates the
Givens transforms with one BF16 generation, then derives Identity, randomized
signed H32, and calibrated Givens G32 activations from the exact same BF16
inputs. Corresponding heatmaps share a color maximum. Each variant's metadata
also records MXFP4 E2M1/E8M0 fake-quantization MSE, SQNR, maximum error, and
underflow fraction. The default reproduces the existing autumn-station setup at
steps 10, 25, and 40 over all blocks:

```shell
DEVICE_ID=0 \
  ./video_quant_lab/baselines/fp_quant/scripts/runners/run_wan_cross_q_transform_comparison.sh
```

Use `BLOCKS=1 SAMPLING_STEPS=10` for a small targeted run before launching the
full matrix. The result root contains `config.json`, `state.json`, and the
aggregate `comparison.json`; per-variant images and metadata are nested below
each `cross_q` directory.

GPU scheduling can be constrained with `GPU_ALLOWLIST`. The main controls are
`SHARED_GPU_TARGET` (default 2), `MAX_GENERATION_WORKERS` (default 8),
`GPU_MAX_USED_MIB` (default 1000), and `GPU_MAX_UTIL_PERCENT` (default 10).
