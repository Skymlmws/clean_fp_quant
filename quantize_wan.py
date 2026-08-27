"""Calibrate and fake-quantize a Wan2.1 DiT with FP-Quant transforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from src.quantization.wan_rtn import WanCalibrationBatch, wan_rtn_quantization
from src.utils.common_utils import fix_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/maoliming/project/checkpoints/Wan2.1-T2V-1.3B"),
    )
    parser.add_argument(
        "--wan-repo", type=Path, default=Path("/home/maoliming/project/wan2.1")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--transform-class", choices=("identity", "hadamard", "givens"), default="givens")
    parser.add_argument("--transform-group-size", type=int, default=32)
    parser.add_argument("--outlier-threshold", type=float, default=50.0)
    parser.add_argument("--weight-bits", type=int, choices=(4, 16), default=4)
    parser.add_argument("--activation-bits", type=int, choices=(4, 16), default=16)
    parser.add_argument("--format", choices=("mxfp", "nvfp", "int"), default="mxfp")
    parser.add_argument("--scale-precision", choices=("e8m0", "e4m3", "fp16"), default="e8m0")
    parser.add_argument("--quant-group-size", type=int, default=32)
    parser.add_argument("--weight-observer", choices=("minmax", "mse"), default="minmax")
    parser.add_argument("--activation-observer", choices=("minmax",), default="minmax")
    parser.add_argument("--timesteps", type=float, nargs="+", default=(50, 250, 500, 750, 950))
    parser.add_argument("--latent-frames", type=int, default=1)
    parser.add_argument("--latent-height", type=int, default=8)
    parser.add_argument("--latent-width", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("wan_givens_summary.json"))
    return parser.parse_args()


def make_inputs(
    args: argparse.Namespace, device: torch.device
) -> tuple[list[WanCalibrationBatch], WanCalibrationBatch]:
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent_shape = (16, args.latent_frames, args.latent_height, args.latent_width)
    context_shape = (args.context_length, 4096)
    seq_len = (
        args.latent_frames
        * (args.latent_height // 2)
        * (args.latent_width // 2)
    )
    batches = []
    for timestep in args.timesteps:
        latent = torch.randn(latent_shape, generator=generator, device=device)
        context = torch.randn(context_shape, generator=generator, device=device)
        input_args = (
            [latent],
            torch.tensor([timestep], device=device, dtype=torch.float32),
            [context],
            seq_len,
        )
        batches.append((input_args, {}))
    return batches, batches[len(batches) // 2]


def output_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float().flatten()
    candidate = candidate.float().flatten()
    error = reference - candidate
    signal = reference.square().sum().double()
    noise = error.square().sum().double()
    return {
        "mse": error.square().mean().item(),
        "sqnr_db": (10 * torch.log10(signal / noise)).item(),
        "max_abs_error": error.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(reference, candidate, dim=0).item(),
    }


def main() -> None:
    args = parse_args()
    fix_seed(args.seed)
    if not args.wan_repo.is_dir():
        raise FileNotFoundError(f"Wan repository not found: {args.wan_repo}")
    if not (args.checkpoint / "diffusion_pytorch_model.safetensors").is_file():
        raise FileNotFoundError(f"Wan DiT checkpoint not found: {args.checkpoint}")
    sys.path.insert(0, str(args.wan_repo))
    from wan.modules.model import WanModel

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    model = WanModel.from_pretrained(args.checkpoint, torch_dtype=dtype).to(device).eval()
    model.requires_grad_(False)
    calibration_batches, evaluation_batch = make_inputs(args, device)

    eval_args, eval_kwargs = evaluation_batch
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
    ):
        reference = model(*eval_args, **eval_kwargs)[0].detach().cpu()

    report = wan_rtn_quantization(
        model,
        calibration_batches,
        device,
        transform_class=args.transform_class,
        transform_group_size=args.transform_group_size,
        outlier_threshold=args.outlier_threshold,
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        quant_format=args.format,
        weight_group_size=args.quant_group_size,
        activation_group_size=args.quant_group_size,
        weight_observer=args.weight_observer,
        activation_observer=args.activation_observer,
        scale_precision=args.scale_precision,
        amp_dtype=dtype,
    )

    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
    ):
        candidate = model(*eval_args, **eval_kwargs)[0].detach().cpu()

    summary = {
        "checkpoint": str(args.checkpoint),
        "transform": args.transform_class,
        "transform_group_size": args.transform_group_size,
        "outlier_threshold": args.outlier_threshold,
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "format": args.format,
        "scale_precision": args.scale_precision,
        "timesteps": args.timesteps,
        "replaced_linears": report.replaced_count,
        "skipped": report.skipped,
        "transform_stats": report.transform_stats,
        "output": output_metrics(reference, candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"summary: {args.output}")


if __name__ == "__main__":
    main()
