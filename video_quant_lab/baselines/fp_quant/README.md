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

## Wan MSFP fake quantization

The Wan RTN path supports a static, search-calibrated MSFP simulator in
addition to the dynamic MXFP/NVFP baselines. It searches a signed FP format
and range for each transformed weight tensor. For every Linear input it also
searches signed FP versus unsigned FP with a data-derived zero point. The
selected parameters remain fixed during generation.

Run the small synthetic-latent calibration smoke test with:

```shell
FORMAT=msfp TRANSFORM_CLASS=identity W_BITS=4 A_BITS=4 \
  ./video_quant_lab/baselines/fp_quant/scripts/runners/run_wan.sh
```

The JSON output includes every layer's selected format, sign mode, maximum
value, zero point, and calibration MSE under `msfp_params`. This is fake
quantization: it simulates W4A4 numerical error but does not pack tensors or
provide a low-bit CUDA speedup.

The default search uses 12 range candidates, 5 zero-point candidates, and a
deterministic sample of at most 4096 values per tensor. The corresponding CLI
controls are `--msfp-maxval-steps`, `--msfp-zero-point-steps`, and
`--msfp-maximum-search-elements`. Increase them for a final experiment after
the end-to-end path has been validated.

For the local Diffusers checkpoint, use the dedicated real-model smoke entry:

```shell
PYTHONPATH=/root/diffusers/src:$PWD/video_quant_lab/baselines/fp_quant \
  /root/diffusers/.venv/bin/python -m scripts.generate.quantize_wan_diffusers \
  --model /root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers \
  --output outputs/wan-msfp-diffusers-smoke.json
```

The local VBench model assets are under `/root/autodl-tmp/vbench`.

Real text-conditioned calibration is available with:

```shell
PYTHONPATH=/root/diffusers/src:$PWD/video_quant_lab/baselines/fp_quant \
  /root/diffusers/.venv/bin/python \
  -m scripts.calibrate.calibrate_wan_msfp_real \
  --prompts video_quant_lab/prompts/wan_activation_long_prompts.json \
  --max-prompts 10 \
  --output outputs/wan-msfp-real-calibration.json
```

Capture indices refer to unique scheduler denoising steps, not raw Transformer
calls. This matters because classifier-free guidance invokes the Transformer
twice at a timestep. Both conditional and unconditional activations are
included at each selected step. A saved configuration can be reused by
`generate_wan_msfp_video --calibration-config <path>` with either the `signed`
or `mixup` variant.

For a controlled signed-FP4 baseline, pass `--sign-mode signed` to
`quantize_wan_diffusers`. Both modes use the same signed weight search and the
same activation format/range search. `mixup` differs only by allowing the 30
GELU-following FFN output projections to consider unsigned FP plus a zero
point. The small paired video entry point is:

```shell
PYTHONPATH=/root/diffusers/src:$PWD/video_quant_lab/baselines/fp_quant \
  /root/diffusers/.venv/bin/python \
  -m scripts.generate.generate_wan_msfp_video \
  --variant mixup --output outputs/msfp-smoke.mp4
```
