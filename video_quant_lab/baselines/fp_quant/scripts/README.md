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
