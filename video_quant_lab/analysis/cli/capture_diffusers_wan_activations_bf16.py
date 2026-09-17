"""Capture complete Wan Diffusers linear inputs for reusable offline analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from video_quant_lab.analysis.cli.capture_wan_activations_bf16 import write_state
from video_quant_lab.analysis.cli.visualize_wan_activation_surfaces import selected_sites
from video_quant_lab.analysis.wan.sites import DIFFUSERS_WAN_LINEAR_SITES
from video_quant_lab.analysis.wan.wan_activation_disk_capture import (
    QuotaExceeded,
    WanActivationDiskCapture,
)
from video_quant_lab.analysis.wan.wan_activation_surface import parse_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers"),
    )
    parser.add_argument("--prompt", default="A small red panda walking in a bamboo forest.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampling-steps", default="25")
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--blocks", default="all")
    parser.add_argument("--sites", default="all")
    parser.add_argument("--max-output-gb", type=float, default=40.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/clean_fp_quant/outputs/activations/all-sites-step25-seed0"),
    )
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Offload inactive pipeline components so the full pipeline fits on smaller GPUs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames % 4 != 1 or args.width % 16 or args.height % 16:
        raise ValueError("frames must be 4n+1 and width/height divisible by 16")
    if args.max_output_gb <= 0:
        raise ValueError("max-output-gb must be positive")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Diffusers Wan model not found: {args.model_path}")

    from diffusers import WanPipeline

    sampling_steps = parse_indices(args.sampling_steps, args.steps)
    call_indices = [step * 2 for step in sampling_steps]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pipe = WanPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=args.device_id)
    else:
        pipe.to(torch.device(f"cuda:{args.device_id}"))

    blocks = parse_indices(args.blocks, len(pipe.transformer.blocks))
    sites = selected_sites(args.sites)
    expected = len(blocks) * len(sites) * len(call_indices)
    capture = WanActivationDiskCapture(
        pipe.transformer,
        args.output_dir,
        args.output_dir,
        int(args.max_output_gb * 1024**3),
        blocks,
        sites,
        call_indices,
        batch_index=0,
        linear_sites=DIFFUSERS_WAN_LINEAR_SITES,
    )
    config = {
        "schema_version": 1,
        "backend": "diffusers",
        "mode": "complete reusable activation capture stored as BF16",
        "model_path": str(args.model_path),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "size": [args.width, args.height],
        "frames": args.frames,
        "steps": args.steps,
        "sampling_steps": sampling_steps,
        "call_indices": call_indices,
        "branches": ["conditional"],
        "blocks": blocks,
        "sites": sites,
        "linear_sites": {key: list(value) for key, value in DIFFUSERS_WAN_LINEAR_SITES.items()},
        "storage_dtype": "torch.bfloat16",
        "max_output_gb": args.max_output_gb,
        "cpu_offload": args.cpu_offload,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_state(args.output_dir, "running", capture, expected)
    capture.attach()
    try:
        with torch.inference_mode():
            pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guide_scale,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                output_type="latent",
            )
    except QuotaExceeded as error:
        write_state(args.output_dir, "paused_quota", capture, expected, str(error))
        raise
    except BaseException as error:
        write_state(args.output_dir, "interrupted", capture, expected, repr(error))
        raise
    finally:
        capture.remove()

    total_complete = sum(
        1
        for path in args.output_dir.rglob("activation.pt")
        if (path.parent / "metadata.json").exists()
    )
    status = "complete" if total_complete == expected else "incomplete"
    config["text_context_by_call"] = {
        str(call): metadata
        for call, metadata in sorted(capture.text_context_by_call.items())
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_state(args.output_dir, status, capture, expected)
    print(
        json.dumps(
            {
                "status": status,
                "complete": total_complete,
                "expected": expected,
                "bytes_written": capture.bytes_written,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
