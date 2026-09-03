# FP-Quant

- Upstream: `git@github.com:IST-DASLab/FP-Quant.git`
- Pinned revision: `d2e3092f968262c4de5fb050e1aef568a280dadd`
- Original entry point: `model_quant.py`

The `source/` directory is a clean checkout of the official implementation.
It does not contain this project's Wan or Givens extensions. Those extensions
remain in the repository root during migration and will receive a separate
baseline identity.

To reproduce the ignored source checkout in a fresh clone of this project:

```shell
git clone git@github.com:IST-DASLab/FP-Quant.git \
  video_quant_lab/baselines/fp_quant/source
git -C video_quant_lab/baselines/fp_quant/source checkout --detach \
  d2e3092f968262c4de5fb050e1aef568a280dadd
```

Create the dedicated environment with:

```shell
./video_quant_lab/baselines/fp_quant/setup_env.sh
```

The setup pins the CUDA 12.6 build of Torch 2.7.1 used by the verified local
runtime and pins the source revision of `fast-hadamard-transform`. Set
`FP_QUANT_PYTHON` only to override this environment, and `FP_QUANT_REPO` only
to override the source checkout location.

The upstream entry point initializes Triton before parsing `--help`, so even
the smoke experiment needs CUDA driver access. It does not load a model or run
quantization.
