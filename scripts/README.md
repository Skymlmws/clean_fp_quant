# Experiment scripts

The reusable implementation lives under `src/`. This directory contains thin
command-line entry points grouped by workflow:

- `generate/`: quantization and video generation
- `visualize/`: FP-Quant-specific visualization that imports its quantizer
- `runners/`: FP-Quant/Givens-specific shell launchers

Method-independent capture, rendering, and artifact tools now live under
`video_quant_lab/analysis/cli`. Existing shell runners remain compatibility
entry points under `video_quant_lab/runners/`.

Run Python entry points from the repository root with module syntax:

```shell
python -m scripts.generate.quantize_wan --help
```

This keeps imports stable regardless of the entry point's directory. Shell
launchers change to the repository root before invoking their Python module.
