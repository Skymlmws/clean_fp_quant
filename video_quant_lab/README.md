# Video Quantization Lab

This directory is the project workspace. Each baseline keeps its own
implementation and launchers together, while shared analysis stays outside the
baseline directories.

## Responsibilities

- `harness/`: launch a baseline as an external process and record provenance
- `baselines/`: one self-contained implementation and command manifest per baseline
- `experiments/`: pass arguments in each baseline's own CLI format
- `prompts/`: project-owned prompt sets used across baselines
- `analysis/`: project-owned activation analysis has started moving here;
  visualization CLIs and result aggregation will follow incrementally
- `runners/`: convenient shell entry points for project-owned analysis tools
- `tests/`: harness, analysis, rendering, and visualization tests

The harness launches baseline code as an external process. Each baseline may
keep its own environment, dependencies, and command-line interface.

## Run an experiment

First point the manifest at the relevant repositories and environments:

```shell
export FP_QUANT_PYTHON=/path/to/fp-quant-env/bin/python
```

Inspect the exact command without running it:

```shell
python -m video_quant_lab.harness.run \
  --experiment video_quant_lab/experiments/fp_quant_help_smoke.json \
  --dry-run
```

Remove `--dry-run` to execute it. Each run receives an isolated directory under
`outputs/runs/` containing `run.json`, logs, and baseline artifacts.
Use `--run-name NAME` for repeated trials. Existing run directories are never
overwritten.

## Migration boundary

Git commit `d2e3092` is the practical upstream FP-Quant boundary. Code added
after it falls into two groups:

1. method-independent profiling, visualization, artifact handling, evaluation,
   and orchestration; these belong in this workspace;
2. Givens/FP-Quant quantization internals; these now live directly in
   `baselines/fp_quant/` together with the upstream FP-Quant implementation.

Run project-owned and baseline-specific tests separately:

```shell
python -m pytest -q video_quant_lab/tests
cd video_quant_lab/baselines/fp_quant
python -m pytest -q tests
```
