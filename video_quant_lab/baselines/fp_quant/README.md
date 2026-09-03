# FP-Quant baseline

- Upstream: `git@github.com:IST-DASLab/FP-Quant.git`
- Pinned revision: `d2e3092f968262c4de5fb050e1aef568a280dadd`
- Original entry point: `model_quant.py`
- Local extensions: Wan2.1 RTN fake quantization and Givens transforms

This directory is the complete tracked FP-Quant baseline used by the project.
It contains the upstream implementation together with this project's Wan and
Givens extensions. Keeping both here makes the baseline runnable without a
second project-level method or integration layer.

No separate source checkout is required: runtime commands use the tracked code
in this directory directly. The upstream base revision above remains recorded
for comparison and provenance.

Create the dedicated environment with:

```shell
./video_quant_lab/baselines/fp_quant/setup_env.sh
```

The setup pins the CUDA 12.6 build of Torch 2.7.1 used by the verified local
runtime and pins the source revision of `fast-hadamard-transform`. Set
`FP_QUANT_PYTHON` only to override this environment, and `FP_QUANT_REPO` only
to override the complete baseline directory.

The upstream entry point initializes Triton before parsing `--help`, so even
the smoke experiment needs CUDA driver access. Wan launchers are under
`scripts/runners/` and write to the main project's `outputs/` directory.
