"""Run an MSFP smoke test on a local Diffusers Wan2.1 transformer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.quantization.wan_rtn import wan_rtn_quantization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timesteps", type=float, nargs="+", default=(50, 500, 950))
    parser.add_argument("--eval-timestep", type=float, default=625)
    parser.add_argument("--latent-frames", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=8)
    parser.add_argument("--latent-width", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=16)
    parser.add_argument("--search-elements", type=int, default=1024)
    parser.add_argument("--maxval-steps", type=int, default=8)
    parser.add_argument("--zero-point-steps", type=int, default=5)
    parser.add_argument("--sign-mode", choices=("mixup", "signed"), default="mixup")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/wan-msfp-diffusers-smoke.json")
    )
    return parser.parse_args()


def metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    error = reference.float() - candidate.float()
    signal = reference.float().square().sum().double()
    noise = error.square().sum().double()
    return {
        "mse": float(error.square().mean()),
        "sqnr_db": float(10 * torch.log10(signal / noise)),
        "max_abs_error": float(error.abs().max()),
    }


def main() -> None:
    args = parse_args()
    from diffusers import WanTransformer3DModel

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = WanTransformer3DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device).eval()
    model.requires_grad_(False)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    hidden = torch.randn(
        1, 16, args.latent_frames, args.latent_height, args.latent_width,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    context = torch.randn(
        1, args.context_length, 4096,
        device=device, dtype=torch.bfloat16, generator=generator,
    )
    batches = [
        ((hidden, torch.tensor([step], device=device), context), {"return_dict": False})
        for step in args.timesteps
    ]
    eval_hidden = torch.randn(
        hidden.shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    eval_context = torch.randn(
        context.shape, device=device, dtype=torch.bfloat16, generator=generator
    )
    eval_args = (
        eval_hidden,
        torch.tensor([args.eval_timestep], device=device),
        eval_context,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model(*eval_args, return_dict=False)[0]

    report = wan_rtn_quantization(
        model, batches, device,
        transform_class="identity", weight_bits=4, activation_bits=4,
        quant_format="msfp", amp_dtype=torch.bfloat16,
        msfp_maxval_steps=args.maxval_steps,
        msfp_zero_point_steps=args.zero_point_steps,
        msfp_maximum_search_elements=args.search_elements,
        msfp_allow_unsigned_aal=args.sign_mode == "mixup",
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        candidate = model(*eval_args, return_dict=False)[0]

    unsigned = sum(
        not entry["signed"] for name, entry in report.msfp_params.items()
        if name.endswith(".activation")
    )
    summary = {
        "model": str(args.model),
        "sign_mode": args.sign_mode,
        "calibration_timesteps": args.timesteps,
        "eval_timestep": args.eval_timestep,
        "replaced_linears": report.replaced_count,
        "activation_unsigned": unsigned,
        "activation_total": sum(name.endswith(".activation") for name in report.msfp_params),
        "output": metrics(reference, candidate),
        "msfp_params": report.msfp_params,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "msfp_params"}, indent=2))
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
