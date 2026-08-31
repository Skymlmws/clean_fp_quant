"""Generate matched BF16 and transformed fake-quantized Wan2.1 videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

from src.utils.wan_utils import (
    build_wan_block_transforms,
    finalize_wan_transforms,
    get_wan_transform_stats,
    observe_wan_transforms,
    replace_wan_linears,
)


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
    parser.add_argument("--prompt", default="A small red panda walking in a bamboo forest.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--transform-class", choices=("identity", "hadamard", "givens"), default="givens"
    )
    parser.add_argument("--transform-group-size", type=int, default=32)
    parser.add_argument("--outlier-threshold", type=float, default=5.0)
    parser.add_argument("--weight-bits", type=int, choices=(4, 16), default=4)
    parser.add_argument("--activation-bits", type=int, choices=(4, 16), default=4)
    parser.add_argument("--quant-group-size", type=int, default=32)
    parser.add_argument("--weight-observer", choices=("minmax", "mse"), default="minmax")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/wan_givens_w4a4_video")
    )
    parser.add_argument(
        "--reference-tensor",
        type=Path,
        default=None,
        help="Reusable decoded BF16 tensor. It is created when absent.",
    )
    parser.add_argument("--reference-only", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.frames % 4 != 1:
        raise ValueError("frames must be 4n+1")
    if args.width % 16 or args.height % 16:
        raise ValueError("width and height must be divisible by 16")
    if args.transform_group_size <= 1 or args.transform_group_size & (args.transform_group_size - 1):
        raise ValueError("transform-group-size must be a power of two greater than one")
    if args.quant_group_size != 32:
        raise ValueError("MXFP4 requires quant-group-size=32")
    if not (args.checkpoint / "diffusion_pytorch_model.safetensors").is_file():
        raise FileNotFoundError(f"Wan checkpoint not found: {args.checkpoint}")
    if not args.wan_repo.is_dir():
        raise FileNotFoundError(f"Wan repository not found: {args.wan_repo}")


def generate(pipe, args: argparse.Namespace) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    video = pipe.generate(
        input_prompt=args.prompt,
        size=(args.width, args.height),
        frame_num=args.frames,
        shift=args.shift,
        sample_solver="unipc",
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        n_prompt=args.negative_prompt,
        seed=args.seed,
        offload_model=False,
    )
    return video.detach().cpu(), time.perf_counter() - started


def video_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    candidate = candidate.float()
    error = reference - candidate
    signal_sq = reference.square().sum().double()
    error_sq = error.square().sum().double()
    return {
        "mse": error.square().mean().item(),
        "sqnr_db": (10 * torch.log10(signal_sq / error_sq)).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            reference.flatten(), candidate.flatten(), dim=0
        ).item(),
        "max_abs_error": error.abs().max().item(),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.wan_repo))
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    from wan.utils.utils import cache_video

    device = torch.device(f"cuda:{args.device_id}")
    pipe = WanT2V(
        config=WAN_CONFIGS["t2v-1.3B"],
        checkpoint_dir=str(args.checkpoint),
        device_id=args.device_id,
        t5_cpu=True,
    )

    transform_kwargs = (
        {"outlier_threshold": args.outlier_threshold}
        if args.transform_class == "givens"
        else {}
    )
    block_transforms = build_wan_block_transforms(
        pipe.model,
        args.transform_class,
        args.transform_group_size,
        device,
        **transform_kwargs,
    )

    generated_reference = None
    reference_seconds = 0.0
    if args.transform_class == "givens":
        handles = observe_wan_transforms(pipe.model, block_transforms)
        try:
            # The BF16 pass observes every conditional/unconditional DiT call
            # at every denoising timestep without changing the model output.
            generated_reference, reference_seconds = generate(pipe, args)
        finally:
            for handle in handles:
                handle.remove()
        finalize_wan_transforms(block_transforms)

    if args.reference_tensor is not None and args.reference_tensor.is_file():
        reference = torch.load(args.reference_tensor, map_location="cpu", weights_only=True)
    elif generated_reference is not None:
        reference = generated_reference
    else:
        reference, reference_seconds = generate(pipe, args)

    if args.reference_tensor is not None and not args.reference_tensor.is_file():
        args.reference_tensor.parent.mkdir(parents=True, exist_ok=True)
        torch.save(reference, args.reference_tensor)

    reference_path = args.output_dir / "bf16.mp4"
    cache_video(reference[None], save_file=str(reference_path), fps=args.fps)
    if args.reference_only:
        print(f"reference_tensor: {args.reference_tensor}")
        print(f"reference_video: {reference_path}")
        return

    quantizer_common = {
        "symmetric": True,
        "format": "mxfp",
        "granularity": "group",
        "group_size": args.quant_group_size,
        "scale_precision": "e8m0",
    }
    weight_quantizer_kwargs = (
        {**quantizer_common, "bits": 4, "observer": args.weight_observer}
        if args.weight_bits == 4
        else None
    )
    activation_quantizer_kwargs = (
        {**quantizer_common, "bits": 4, "observer": "minmax"}
        if args.activation_bits == 4
        else None
    )
    report = replace_wan_linears(
        pipe.model,
        block_transforms,
        weight_quantizer_kwargs,
        activation_quantizer_kwargs,
    )
    report.transform_stats = get_wan_transform_stats(block_transforms)
    quantized, quantized_seconds = generate(pipe, args)

    method_name = f"{args.transform_class}_w{args.weight_bits}a{args.activation_bits}"
    quantized_path = args.output_dir / f"{method_name}.mp4"
    cache_video(quantized[None], save_file=str(quantized_path), fps=args.fps)

    summary = {
        "prompt": args.prompt,
        "seed": args.seed,
        "size": [args.width, args.height],
        "frames": args.frames,
        "fps": args.fps,
        "steps": args.steps,
        "transform": args.transform_class,
        "transform_group_size": args.transform_group_size,
        "outlier_threshold": args.outlier_threshold,
        "format": "mxfp",
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "quant_group_size": args.quant_group_size,
        "replaced_linears": report.replaced_count,
        "skipped": report.skipped,
        "transform_stats": report.transform_stats,
        "bf16_seconds": reference_seconds,
        "quantized_seconds": quantized_seconds,
        "decoded_video": video_metrics(reference, quantized),
        "bf16_video": str(reference_path),
        "quantized_video": str(quantized_path),
        "note": "Fake quantization measures numerical behavior, not native FP4 speed.",
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
