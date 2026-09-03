# Video Quantization Lab

This repository is an experiment scaffold for running video-model quantization
baselines, collecting outputs, and applying shared analysis and evaluation.

## Layout

- `video_quant_lab/baselines/`: self-contained baseline implementations
- `video_quant_lab/runners/`: method-independent analysis launchers
- `video_quant_lab/analysis/`: project-owned capture and visualization code
- `video_quant_lab/prompts/`: shared experiment prompts
- `video_quant_lab/tests/`: scaffold and analysis tests
- `outputs/`: generated runs and artifacts
- `vbench_inputs/`, `vbench_results/`, `vbench_cache/`: existing VBench data

FP-Quant, including the local Wan2.1 and Givens extensions, lives entirely in
`video_quant_lab/baselines/fp_quant/`.

## FP-Quant Wan smoke runs

From the project root:

```shell
./video_quant_lab/baselines/fp_quant/scripts/runners/run_wan.sh
./video_quant_lab/baselines/fp_quant/scripts/runners/run_wan_givens_video.sh
```

Both launchers locate the baseline code automatically and keep generated data
under the project-level `outputs/` directory.

## Tests

Run project-owned tests from the project root:

```shell
python -m pytest video_quant_lab/tests
```

Run FP-Quant baseline tests from its directory:

```shell
cd video_quant_lab/baselines/fp_quant
python -m pytest tests
```

See `video_quant_lab/README.md` for the process-level harness and
`video_quant_lab/baselines/fp_quant/README.md` for baseline-specific setup.
