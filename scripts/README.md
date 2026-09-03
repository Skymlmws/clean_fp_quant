# Experiment scripts

The reusable implementation lives under `src/`. This directory contains thin
command-line entry points grouped by workflow:

- `generate/`: quantization and video generation
- `profile/`: activation capture and online profiling
- `visualize/`: rendering and analysis of captured artifacts
- `maintenance/`: artifact migration and lifecycle utilities
- `runners/`: shell launchers with environment-variable defaults

Run Python entry points from the repository root with module syntax:

```shell
python -m scripts.generate.quantize_wan --help
```

This keeps imports stable regardless of the entry point's directory. Shell
launchers change to the repository root before invoking their Python module.
