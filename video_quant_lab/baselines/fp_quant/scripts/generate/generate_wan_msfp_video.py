"""Generate a small paired Wan video with BF16, signed FP4, or MSFP W4A4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.quantization.wan_rtn import apply_wan_msfp_config, wan_rtn_quantization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers"))
    parser.add_argument("--variant", choices=("bf16", "signed", "mixup"), required=True)
    parser.add_argument("--calibration-config", type=Path)
    parser.add_argument("--prompt", default="A red panda walks through a bamboo forest, cinematic lighting.")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-index", type=int, default=0)
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def calibration_batches(device: torch.device, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed + 1000)
    hidden = torch.randn(1, 16, 1, 8, 8, device=device, dtype=torch.bfloat16, generator=generator)
    context = torch.randn(1, 16, 4096, device=device, dtype=torch.bfloat16, generator=generator)
    return [
        ((hidden, torch.tensor([step], device=device), context), {"return_dict": False})
        for step in (50, 250, 500, 750, 950)
    ]


def main() -> None:
    args = parse_args()
    from diffusers import WanPipeline, WanTransformer3DModel
    from diffusers.utils import export_to_video

    if args.prompt_file:
        prompt_lines = [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()]
        args.prompt = prompt_lines[args.prompt_index]

    device = torch.device(args.device)
    transformer = WanTransformer3DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device).eval()
    transformer.requires_grad_(False)
    report = None
    if args.variant != "bf16":
        if args.calibration_config:
            report = apply_wan_msfp_config(
                transformer,
                args.calibration_config,
                device,
                activation_mode="msfp" if args.variant == "mixup" else "signed",
            )
        else:
            report = wan_rtn_quantization(
                transformer,
                calibration_batches(device, args.seed),
                device,
                transform_class="identity",
                weight_bits=4,
                activation_bits=4,
                quant_format="msfp",
                amp_dtype=torch.bfloat16,
                msfp_maxval_steps=8,
                msfp_zero_point_steps=5,
                msfp_maximum_search_elements=1024,
                msfp_allow_unsigned_aal=args.variant == "mixup",
            )

    pipe = WanPipeline.from_pretrained(
        args.model, transformer=transformer, torch_dtype=torch.bfloat16
    )
    if float(pipe.scheduler.config.flow_shift) != 3.0:
        raise ValueError(f"Expected flow_shift=3.0, got {pipe.scheduler.config.flow_shift}")
    pipe.enable_model_cpu_offload(device=device.index or 0)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).frames[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result, str(args.output), fps=args.fps)
    metadata = {
        "variant": args.variant,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "fps": args.fps,
        "sampler": type(pipe.scheduler).__name__,
        "flow_shift": float(pipe.scheduler.config.flow_shift),
        "seed": args.seed,
        "calibration_config": str(args.calibration_config) if args.calibration_config else None,
        "replaced_linears": report.replaced_count if report else 0,
        "activation_unsigned": (
            sum(not value["signed"] for name, value in report.msfp_params.items() if name.endswith(".activation"))
            if report else 0
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"video: {args.output}")


if __name__ == "__main__":
    main()
