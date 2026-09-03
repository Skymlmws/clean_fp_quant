# Video Quantization Lab

This directory is the method-neutral project workspace. Baseline repositories
are execution targets, not Python dependencies of the harness.

## Responsibilities

- `harness/`: launch a baseline as an external process and record provenance
- `baselines/`: one isolated source checkout and command manifest per baseline
- `experiments/`: pass arguments in each baseline's own CLI format
- `prompts/`: project-owned prompt sets used across baselines
- `analysis/`: project-owned activation analysis has started moving here;
  visualization CLIs and result aggregation will follow incrementally
- `runners/`: convenient shell entry points for project-owned analysis tools
- `tests/`: harness, analysis, rendering, and visualization tests

The harness never imports baseline implementation code. A baseline may use its
own repository, environment, dependency versions, and command-line interface.

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
2. Givens/FP-Quant quantization internals; these belong in the FP-Quant baseline
   fork and should only be invoked through its command manifest.

The existing repository root remains the first baseline workspace during the
migration. It can be replaced by a separate clone after project-owned analysis
code has been extracted.

Run project-owned and baseline-specific tests separately:

```shell
python -m pytest -q video_quant_lab/tests
python -m pytest -q tests
```
