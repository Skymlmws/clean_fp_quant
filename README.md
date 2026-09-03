# Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization 

<a href='https://arxiv.org/pdf/2509.23202'><img src='https://img.shields.io/badge/ArXiv-PDF-red' height="25"></a> &nbsp; 

The official implementation for the paper [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202).

> Project note: the repository root currently remains the FP-Quant baseline
> workspace. Method-neutral orchestration is being extracted into
> `video_quant_lab/`, which launches baseline repositories as external commands
> and does not require them to share implementation code.

This repository contains the code needed to reproduce the results presented in the paper, and it also offers the ability to export quantized models with [QuTLASS](https://github.com/IST-DASLab/qutlass) kernels in the **MXFP4** and **NVFP4 formats**. The exported models can be run either with Hugging Face Transformers or with vLLM.

### Repository structure
---

The repository is structured as follows:

* `model_quant.py` - the original FP-Quant quantization entry point
* `src/` - reusable FP-Quant and Wan quantization implementation
* `scripts/generate/` - Wan quantization and video-generation entry points
* `scripts/profile/` - activation capture and online profiling entry points
* `scripts/visualize/` - artifact rendering and visualization entry points
* `scripts/maintenance/` - artifact migration and cleanup utilities
* `scripts/runners/` - reproducible shell launchers for common experiments
* `tests/` - unit and integration tests

Run Python entry points as modules from the repository root, for example:

```shell
python -m scripts.generate.quantize_wan --help
python -m scripts.visualize.visualize_wan_weights --help
```

The shell launchers may be invoked directly, for example:

```shell
./scripts/runners/run_wan_givens_video.sh
```

### Environment setup
---

**Inference Engines**

FP-Quant has support implemented in:
 - `transformers` with these features:
     - Available in `main` ([Documentation](https://huggingface.co/docs/transformers/main/en/quantization/fp_quant#fp-quant)).
     - RTN on-the-fly quantization.
       ```python
       from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig
       import torch
        
       model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            quantization_config=FPQuantConfig(forward_dtype="mxfp4"),
            device_map="auto",
            dtype=torch.bfloat16,
        )
       model.forward = torch.compile(model.forward, mode="max-autotune", fullgraph=True)
       ```
     - Pseudo-quantization QAT.
 - `vLLM` with these features:
     - Available in [this PR](https://github.com/vllm-project/vllm/pull/24440).
     - Compatible with real quantization models from `FP-Quant` and the `transformers` integration.

### FP-Quant models
---

👉 Check out the quantized MXFP and NVFP models in the [MR-GPTQ](https://huggingface.co/collections/ISTA-DASLab/mr-gptq-68dcde4b1e4b572ded89dbf3) collection on Hugging Face 🤗.  

*Example of quantized model inference with HF*
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, FPQuantConfig
import torch

model_name = "ISTA-DASLab/Llama-3.1-8B-Instruct-MR-GPTQ-nvfp"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
)
prompt = "Explain quantization for neural network in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
    output_tokens = model.generate(**inputs,max_new_tokens=150 )
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(generated_text)
```  
*Example of quantized model inference with vLLM engine*  

```python
from vllm import LLM, SamplingParams

model_name = "ISTA-DASLab/Llama-3.1-8B-Instruct-MR-GPTQ-nvfp"
llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.9)
sampling_params = SamplingParams(
    temperature=0.7,       # creativity
    top_p=0.9,             # nucleus sampling
    max_tokens=150,        # number of new tokens to generate
)
prompt = "Explain quantization for neural networks in simple terms."
outputs = llm.generate([prompt], sampling_params)
print(outputs[0].outputs[0].text)
```
### Quantization
---

**NOTE** - The quantization script is designed to be run on a single GPU.

**NOTE** - Only Llama and Qwen3 models are supported.

Below is an example of the quantization script usage:

```shell
#!/bin/bash
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

MODEL=${MODEL:-"meta-llama/Llama-3.2-1B-Instruct"}
MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
# Data params
NUM_SEQUENCES=${NUM_SEQUENCES:-128}
# Quantization params
FORMAT=${FORMAT:-"nvfp"}
W_BITS=${W_BITS:-4}
A_BITS=${A_BITS:-16}
W_GROUP_SIZE=${W_GROUP_SIZE:-16}
A_GROUP_SIZE=${A_GROUP_SIZE:-16}
GPTQ=${GPTQ:-0}
W_OBSERVER=${W_OBSERVER:-"minmax"}
QUANTIZATION_ORDER=${QUANTIZATION_ORDER:-"default"}
# Save params
EXPORT_QUANTIZATION=${EXPORT_QUANTIZATION:-""}
# Transform params
TRANSFORM_CLASS=${TRANSFORM_CLASS:-"identity"}
HADAMARD_GROUP_SIZE=${HADAMARD_GROUP_SIZE:-128}
# Evaluation params
EVAL_PERPLEXITY=${EVAL_PERPLEXITY:-1}
EVAL_OPENLLM=${EVAL_OPENLLM:-0}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-"auto"}
# Misc params
LOG_WANDB=${LOG_WANDB:-0}
DTYPE=${DTYPE:-"auto"}
CPU_OFFLOAD_ACTIVATIONS=${CPU_OFFLOAD_ACTIVATIONS:-0}

SCRIPT_ARGS=""

if [[ $GPTQ == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --gptq"
fi

if [[ $EVAL_PERPLEXITY == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_perplexity"
fi

if [[ $EVAL_OPENLLM == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --eval_openllm"
fi

if [[ $LOG_WANDB == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --log_wandb"
fi

METHOD_NAME=""
if [[ $GPTQ == 1 ]]; then
    METHOD_NAME="GPTQ"
else
    METHOD_NAME="RTN"
fi

if [[ $CPU_OFFLOAD_MODULES == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --cpu_offload_modules"
fi

if [[ $CPU_OFFLOAD_ACTIVATIONS == 1 ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --cpu_offload_activations"
fi

export WANDB_PROJECT="FP-Quantization-Harness"
export WANDB_NAME=${MODEL}/${FORMAT}-w${W_BITS}-a${A_BITS}-${METHOD_NAME}-${TRANSFORM_CLASS}-transform

if [[ $EXPORT_QUANTIZATION == "realquant" || $EXPORT_QUANTIZATION == "pseudoquant" ]]; then
    SCRIPT_ARGS="${SCRIPT_ARGS} --export_quantized_model ${EXPORT_QUANTIZATION}"
    if [[ $EXPORT_QUANTIZATION == "realquant" ]]; then
        SAVE_DIR=quantized_models
    else
        SAVE_DIR=pseudoquantized_models
    fi
fi

python model_quant.py \
    --model_name_or_path=${MODEL} \
    --format=${FORMAT} \
    --w_bits=${W_BITS} \
    --a_bits=${A_BITS} \
    --w_group_size=${W_GROUP_SIZE} \
    --a_group_size=${A_GROUP_SIZE} \
    --transform_class=${TRANSFORM_CLASS} \
    --w_observer=${W_OBSERVER} \
    --quantization_order=${QUANTIZATION_ORDER} \
    $SCRIPT_ARGS \
    --hadamard_group_size=${HADAMARD_GROUP_SIZE} \
    --dataset_name_or_path=fineweb-edu \
    --num_sequences=${NUM_SEQUENCES} \
    --sequence_length=2048 \
    --dtype=${DTYPE} \
    --lm_eval_batch_size=${LM_EVAL_BATCH_SIZE} \
    --save_path "${SAVE_DIR}/${MODEL_ID}-${FORMAT}-w${W_BITS}-a${A_BITS}-${METHOD_NAME}-${TRANSFORM_CLASS}-transform" \
    --export_quantized_model pseudoquant \
    --cpu_offload_activations \
    --cpu_offload_modules \
    --fuse_global_scale \
    --amp
```

Above:
* `--model_name_or_path` - The model to quantize. (Llama and Qwen3 models are supported)
* `--format` - The quantization format (int, fp, mxfp, nvfp). 
* `--w_bits` - The number of bits to quantize the weights to.
* `--a_bits` - The number of bits to quantize the activations to.
* `--w_group_size` - The number of weights to quantize together.
* `--a_group_size` - The number of activations to quantize together.
* `--init` - Transform initialization.
* `--transform_class` - Transform class. We provide the following options:
    * `identity` - Identity transform
    * `hadamard` - Hadamard transform
    * `givens` - Data-aware block-wise Givens transform for massive outliers
    * `dct` - Discrete cosine transform
    * `dst` - Discrete sine transform
    * `fast_food` - Fast food transform
    * `gsr` - Grouped sequency aligned transform
* `--hadamard_group_size` - Transform group size.
* `--dataset_name_or_path` - Dataset to use for calibration.
* `--sequence_length` - Calibration sequence length.
* `--dtype` - Data type to load the model.
* `--amp` - Whether to use automatic mixed precision.
* `--export_quantized_model` - Whether to export quantized model in `realquant` or `pseudoquant` format. The former allows one to run quantized model with the help of [QuTLASS](https://github.com/IST-DASLab/qutlass) integration, while the latter produces fake quantized model runnable with `triton` kernels.

For evaluation, we provide the following options:

* `--eval_perplexity` - Whether to evaluate perplexity after quantization.
* `--eval_openllm` - Whether to evaluate OpenLLM v1 openllm after quantization.
* `--lm_eval_batch_size` - LM eval batch size to evaluate after quantization.
* `--fuse_global_scale` - Whether to fuse global scale in qkv and gate_up projections as required by `vLLM`.


We note, however, that the evaluation within quantization script is not optimized and may take several days.
The recommended way to evaluate models is to export the quantized model and evaluate it via `vLLM` integration.

*Evaluation*

We evaluate the compressed models on a subset of the tasks from OpenLLM v1 benchmark using the recommended parameters.

Below is an example of the bash evaluation script usage:

```shell
export OMP_NUM_THREADS=8
export VLLM_WORKER_MULTIPROC_METHOD=spawn

NUM_GPUS=$( echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l )
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.8}

MODEL_ID=$( echo $MODEL | awk -F/ '{print $NF}' )
MODEL_ARGS="pretrained=$MODEL,max_model_len=4096,tensor_parallel_size=$NUM_GPUS,dtype=auto,gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},enforce_eager=True"

# Winogrande
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks winogrande \
  --num_fewshot=5 \
  --batch_size auto \
  --output_path lm_eval_results

# Hellaswag
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks hellaswag \
  --num_fewshot=10 \
  --batch_size auto \
  --output_path lm_eval_results

# GSM8k
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks gsm8k_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto \
  --output_path lm_eval_results

# MMLU-CoT 
lm_eval \
  --model vllm \
  --model_args $MODEL_ARGS \
  --tasks mmlu_cot_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto \
  --output_path lm_eval_results
```


### Wan2.1 DiT experimental path

Wan2.1 uses a separate entry point so the diffusion calibration path does not
depend on the LLM model loader. The current implementation performs fake RTN
quantization of the 300 Linear layers inside the 30 DiT blocks.

```shell
python -m scripts.generate.quantize_wan \
  --checkpoint /path/to/Wan2.1-T2V-1.3B \
  --wan-repo /path/to/Wan2.1 \
  --device cuda:0 \
  --transform-class givens \
  --transform-group-size 32 \
  --outlier-threshold 5 \
  --weight-bits 4 \
  --activation-bits 16 \
  --format mxfp \
  --scale-precision e8m0
```

The JSON report includes the number of channel blocks that used Givens versus
the Hadamard fallback. Calibrate the outlier threshold on representative
prompts, latents, and diffusion timesteps before comparing transform quality.

To calibrate Givens during a real BF16 denoising pass and then generate a
matching Givens+MXFP4 W4A4 video with the same prompt and seed:

```shell
python -m scripts.generate.generate_wan_givens_video \
  --checkpoint /path/to/Wan2.1-T2V-1.3B \
  --wan-repo /path/to/Wan2.1 \
  --device-id 0 \
  --prompt "A small red panda walking in a bamboo forest." \
  --width 128 --height 128 --frames 5 --steps 4 \
  --outlier-threshold 5 \
  --output-dir outputs/video-quantization-runs/wan_givens_w4a4_video
```

The first generation supplies real conditional and unconditional activations
from every denoising timestep for calibration. The output directory contains
`bf16.mp4`, `givens_w4a4.mp4`, and `summary.json`.

The same entry point accepts `--transform-class identity|hadamard|givens`,
`--weight-bits 4|16`, and `--activation-bits 4|16`. To run the standard
Identity/Hadamard/Givens comparison matrix on two GPUs with one shared BF16
reference:

```shell
GPU_A=2 GPU_B=3 WIDTH=832 HEIGHT=480 FRAMES=81 FPS=16 STEPS=50 \
  ./run_wan_quant_matrix.sh
```

Before tuning Givens, profile the untouched BF16/W16A16 linear inputs:

```shell
DEVICE_ID=2 WIDTH=832 HEIGHT=480 FRAMES=81 STEPS=50 \
  ./run_wan_activation_profile.sh
```

This records bounded streaming statistics at the seven shared transform sites
in every DiT block. It produces per-call/timestep CSV data, raw channel
statistics, layer-by-site outlier heatmaps, and sorted channel-max curves. No
transform or fake quantizer is inserted during this profiling pass.

Profile all 300 BF16 DiT Linear weights and measure their direct Identity
MXFP4 W4 fake-quantization error with:

```shell
DEVICE_ID=2 ./run_wan_weight_profile.sh
```

The weight report is aligned with the same seven transform sites and includes
raw outlier metrics, per-32-value block dynamic range, W4 relative MSE/SQNR,
and input-channel outlier curves.

Render the actual token-by-channel activation matrix at each selected DiT
block and transform site as a 3D bar chart:

```shell
DEVICE_ID=2 BLOCKS=all SITES=ffn_in STEPS=3 CALL_INDEX=0 \
  ./run_wan_activation_surfaces.sh
```

The horizontal axes are channel and token. Every token-channel pair is one
vertical column whose height is the absolute activation magnitude. Normal
values are blue and values above `--outlier-percentile` (p99.99 by default) are
highlighted in red, following the visual convention of DuQuant Figure 1(a)(b).
The demo defaults to three denoising steps, captures call 0,
and keeps every token and channel. Its compressed NPZ contains the complete
matrix, the full activation range, and original token/channel indices. `SITES=ffn_in` is the shell-script
default so that `BLOCKS=all` produces one full 3D bar chart per Transformer block;
use `SITES=all` when all seven shared Linear-input sites are wanted. Optional
positive `--max-tokens` and `--max-channels` values enable explicit sampling;
a smaller `--z-percentile` can optionally suppress extreme peaks.
The NPZ always keeps the complete matrix. To avoid overplotting in the static
PNG, at most 50,000 evenly distributed blue context bars are shown by default,
while every red outlier is retained from the full matrix. Adjust this display
budget with `--max-background-bars`.
Each activation directory also contains `heatmap.png`, a full token-by-channel
top view of absolute activation magnitude. It uses power-normalized `magma`
colors so moderate structure remains visible beside outliers; tune it with
`--heatmap-percentile` and `--heatmap-gamma`.

Each run is organized by call, block, and site:

```text
outputs/activation-visualization/wan-activation-surfaces-.../
├── config.json
├── manifest.json
└── call_000/
    ├── timestep.json
    ├── block_00/ffn_in/
    │   ├── bars.png
    │   ├── activation.npz
    │   └── metadata.json
    └── block_01/ffn_in/
        └── ...
```

`config.json` contains run settings, `manifest.json` indexes every artifact,
and each `metadata.json` records tensor shapes, axes, value range, and source
Linear. The NPZ stores the activation matrix once plus its token/channel
indices.

Activation rendering is resumable at artifact boundaries. Completed triplets
(`activation.npz`, `bars.png`, `heatmap.png`, and `metadata.json`) are skipped when the same
command is rerun. Generation pauses after 200 GiB by default, or after the
optional `MAX_IMAGES` limit, and records progress in `state.json`. Inspect a
batch and explicitly remove it only after downloading:

```shell
python -m scripts.maintenance.manage_wan_artifacts status outputs/activation-visualization/<batch>
python -m scripts.maintenance.manage_wan_artifacts acknowledge-download outputs/activation-visualization/<batch> \
  --delete --confirmation downloaded
```

The launcher monitors the project-wide `outputs/` directory by default, so the
200 GiB limit includes historical results and all parallel GPU workers. Override
`QUOTA_DIR` only when a narrower quota is intentionally required. Status may be
queried for the complete root, but deletion is restricted to explicit child
batch directories.

Render Linear weight matrices with the same 3D outlier style, using input
channel, output channel, and absolute weight magnitude as the three axes:

```shell
DEVICE=cpu BLOCKS=0 SITES=ffn_in ./run_wan_weight_bars.sh
```

Unlike activation capture, this reads weights directly from the checkpoint and
does not run video generation. Outputs are organized as
`block_00/ffn_in/ffn_0/{bars.png,weight.npz,metadata.json}`. Shared sites such
as `self_qkv` produce a separate directory for each member Linear. Weight plots
default to an evenly spaced 512 by 512 view because FFN matrices contain
millions of values that cannot be resolved in a static image. Zero plot limits
explicitly request the complete matrix; sampling affects only visualization and
never modifies checkpoint weights.

### Citation
---

If you find this project useful, please cite our paper:

```
@misc{egiazarian2025bridginggappromiseperformance,
      title={Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization}, 
      author={Vage Egiazarian and Roberto L. Castro and Denis Kuznedelev and Andrei Panferov and Eldar Kurtic and Shubhra Pandit and Alexandre Marques and Mark Kurtz and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
      year={2025},
      eprint={2509.23202},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.23202}, 
}
```
