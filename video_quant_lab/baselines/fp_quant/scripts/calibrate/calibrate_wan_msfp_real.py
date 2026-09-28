"""Calibrate Wan MSFP from real text-conditioned denoising trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from src.quantization.msfp import MSFPParams, search_msfp_params
from src.utils.wan_utils import WAN_DIFFUSERS_LINEAR_TRANSFORM_GROUPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers"))
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("/root/clean_fp_quant/video_quant_lab/prompts/wan_activation_long_prompts.json"),
    )
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--capture-step-indices", type=int, nargs="+", default=(0, 2, 5, 7, 9))
    parser.add_argument("--samples-per-observation", type=int, default=256)
    parser.add_argument("--maximum-samples-per-layer", type=int, default=4096)
    parser.add_argument("--search-elements", type=int, default=1024)
    parser.add_argument("--maxval-steps", type=int, default=8)
    parser.add_argument("--zero-point-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("outputs/wan-msfp-real-calibration.json"))
    return parser.parse_args()


def load_prompts(path: Path, maximum: int) -> list[dict]:
    payload = json.loads(path.read_text())
    prompts = payload["prompts"] if isinstance(payload, dict) else payload
    if maximum <= 0:
        return prompts
    return prompts[:maximum]


def params_dict(params: MSFPParams) -> dict[str, int | float | bool | str]:
    return {
        "format": params.format_name,
        "exponent_bits": params.exponent_bits,
        "mantissa_bits": params.mantissa_bits,
        "signed": params.signed,
        "maxval": params.maxval,
        "zero_point": params.zero_point,
        "mse": params.mse,
    }


class RealTrajectoryCollector:
    def __init__(
        self,
        transformer: nn.Module,
        capture_indices: set[int],
        samples_per_observation: int,
        maximum_samples_per_layer: int,
    ) -> None:
        self.transformer = transformer
        self.capture_indices = capture_indices
        self.samples_per_observation = samples_per_observation
        self.maximum_samples_per_layer = maximum_samples_per_layer
        self.samples: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.timesteps: list[float] = []
        self.step_index = -1
        self.last_timestep: float | None = None
        self.active = False
        self.handles = []

    def _before_transformer(self, _module, args, kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 1:
            timestep = args[1]
        current_timestep = (
            float(timestep.detach().float().reshape(-1)[0].cpu())
            if timestep is not None else None
        )
        is_new_step = current_timestep != self.last_timestep
        if is_new_step:
            self.step_index += 1
            self.last_timestep = current_timestep
        self.active = self.step_index in self.capture_indices
        if self.active and is_new_step and current_timestep is not None:
            self.timesteps.append(current_timestep)

    def _capture(self, name: str, inputs: tuple[torch.Tensor, ...]) -> None:
        if not self.active:
            return
        existing = sum(part.numel() for part in self.samples[name])
        remaining = self.maximum_samples_per_layer - existing
        if remaining <= 0:
            return
        flat = inputs[0].detach().float().reshape(-1)
        take = min(remaining, self.samples_per_observation, flat.numel())
        stride = max(1, math.ceil(flat.numel() / take))
        self.samples[name].append(flat[::stride][:take].cpu())

    def install(self) -> None:
        self.handles.append(
            self.transformer.register_forward_pre_hook(self._before_transformer, with_kwargs=True)
        )
        for block_index, block in enumerate(self.transformer.blocks):
            modules = dict(block.named_modules())
            for linear_names in WAN_DIFFUSERS_LINEAR_TRANSFORM_GROUPS.values():
                for linear_name in linear_names:
                    name = f"blocks.{block_index}.{linear_name}"
                    module = modules[linear_name]
                    self.handles.append(
                        module.register_forward_pre_hook(
                            lambda _module, inputs, current_name=name: self._capture(current_name, inputs)
                        )
                    )

    def reset_prompt(self) -> None:
        self.step_index = -1
        self.last_timestep = None
        self.active = False

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def merged(self) -> dict[str, torch.Tensor]:
        return {name: torch.cat(parts) for name, parts in self.samples.items()}


def search_config(transformer, samples, args) -> dict:
    layers = {}
    ffn_out_names = set(WAN_DIFFUSERS_LINEAR_TRANSFORM_GROUPS["ffn_out"])
    for block_index, block in enumerate(transformer.blocks):
        modules = dict(block.named_modules())
        for linear_names in WAN_DIFFUSERS_LINEAR_TRANSFORM_GROUPS.values():
            for linear_name in linear_names:
                name = f"blocks.{block_index}.{linear_name}"
                linear = modules[linear_name]
                common = {
                    "bits": 4,
                    "maxval_steps": args.maxval_steps,
                    "maximum_search_elements": args.search_elements,
                }
                weight = search_msfp_params(linear.weight, allow_unsigned=False, **common)
                activation_signed = search_msfp_params(samples[name], allow_unsigned=False, **common)
                activation_msfp = activation_signed
                if linear_name in ffn_out_names:
                    activation_msfp = search_msfp_params(
                        samples[name], allow_unsigned=True,
                        zero_point_steps=args.zero_point_steps, **common,
                    )
                layers[name] = {
                    "weight": params_dict(weight),
                    "activation_signed": params_dict(activation_signed),
                    "activation_msfp": params_dict(activation_msfp),
                    "sample_count": int(samples[name].numel()),
                    "aal": linear_name in ffn_out_names,
                }
    return layers


def main() -> None:
    args = parse_args()
    from diffusers import WanPipeline, WanTransformer3DModel

    prompts = load_prompts(args.prompts, args.max_prompts)
    if not prompts:
        raise ValueError("Calibration prompt set is empty")
    if max(args.capture_step_indices) >= args.steps:
        raise ValueError("capture-step-indices must be smaller than --steps")

    device = torch.device(args.device)
    transformer = WanTransformer3DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device).eval()
    transformer.requires_grad_(False)
    collector = RealTrajectoryCollector(
        transformer,
        set(args.capture_step_indices),
        args.samples_per_observation,
        args.maximum_samples_per_layer,
    )
    collector.install()
    pipe = WanPipeline.from_pretrained(args.model, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload(device=device.index or 0)
    try:
        for prompt_index, prompt in enumerate(prompts):
            collector.reset_prompt()
            generator = torch.Generator(device="cpu").manual_seed(args.seed + prompt_index)
            pipe(
                prompt=prompt["prompt"] if isinstance(prompt, dict) else prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=5.0,
                generator=generator,
                output_type="latent",
            )
    finally:
        collector.remove()

    samples = collector.merged()
    if len(samples) != 300:
        raise RuntimeError(f"Expected samples for 300 Linear layers, got {len(samples)}")
    layers = search_config(transformer, samples, args)
    aal_layers = [value for value in layers.values() if value["aal"]]
    unsigned_aal = sum(not value["activation_msfp"]["signed"] for value in aal_layers)
    signed_mse = sum(value["activation_signed"]["mse"] for value in aal_layers)
    msfp_mse = sum(value["activation_msfp"]["mse"] for value in aal_layers)
    payload = {
        "format_version": 1,
        "method": "wan_msfp_real_trajectory_calibration",
        "model": str(args.model),
        "prompt_file": str(args.prompts),
        "prompt_ids": [prompt.get("id", str(i)) for i, prompt in enumerate(prompts)],
        "seed": args.seed,
        "generation": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "num_inference_steps": args.steps,
            "capture_step_indices": args.capture_step_indices,
            "captured_scheduler_timesteps": collector.timesteps,
        },
        "search": {
            "maxval_steps": args.maxval_steps,
            "zero_point_steps": args.zero_point_steps,
            "maximum_search_elements": args.search_elements,
        },
        "summary": {
            "layer_count": len(layers),
            "aal_layer_count": len(aal_layers),
            "unsigned_aal_count": unsigned_aal,
            "aal_signed_mse": signed_mse,
            "aal_msfp_mse": msfp_mse,
            "aal_mse_reduction_fraction": 1.0 - msfp_mse / signed_mse,
        },
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "layers"}, indent=2))
    print(f"configuration: {args.output}")


if __name__ == "__main__":
    main()
