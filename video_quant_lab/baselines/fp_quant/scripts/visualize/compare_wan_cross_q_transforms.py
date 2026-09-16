"""Paired online comparison of Identity, randomized H32, and calibrated Givens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn

from src.quantization.quantizer import Quantizer
from src.utils.wan_utils import (
    WanBlockTransforms,
    build_wan_block_transforms,
    finalize_wan_transforms,
    get_wan_transform_stats,
    observe_wan_transforms,
)
from video_quant_lab.analysis.cli.capture_render_wan_activations_online import (
    ONLINE_RENDER_SCHEMA_VERSION,
    WanOnlineActivationRenderer,
)
from video_quant_lab.analysis.wan.wan_activation_surface import parse_indices


VARIANTS = ("identity", "hadamard-h32", "givens-g32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/maoliming/project/checkpoints/Wan2.1-T2V-1.3B"))
    parser.add_argument("--wan-repo", type=Path, default=Path("/home/maoliming/project/wan2.1"))
    parser.add_argument("--prompt-file", type=Path, default=Path("video_quant_lab/prompts/wan_activation_long_prompts.json"))
    parser.add_argument("--prompt-id", default="autumn_station")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampling-steps", default="10,25,40")
    parser.add_argument("--blocks", default="all")
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--transform-seed", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--outlier-threshold", type=float, default=5.0)
    parser.add_argument("--image-width", type=int, default=1800)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument("--heatmap-percentile", type=float, default=100.0)
    parser.add_argument("--heatmap-gamma", type=float, default=1.0)
    parser.add_argument("--plot-kind", choices=("heatmap", "surface", "both"), default="heatmap")
    parser.add_argument("--channel-rms-ratio", type=float, default=5.0)
    parser.add_argument("--mark-top-channels", type=int, default=8)
    parser.add_argument("--isolated-global-percentile", type=float, default=99.99)
    parser.add_argument("--isolated-channel-percentile", type=float, default=99.0)
    parser.add_argument("--isolated-ratio", type=float, default=5.0)
    parser.add_argument("--isolated-max-token-fraction", type=float, default=0.01)
    parser.add_argument("--mark-top-isolated", type=int, default=10)
    parser.add_argument("--isolated-merge-token-gap", type=int, default=1)
    parser.add_argument("--ffn-out-group-size", type=int, default=8)
    parser.add_argument("--max-output-gb", type=float, default=0.0)
    parser.add_argument("--render-mode", choices=("multiprocess", "async", "sync"), default="multiprocess")
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--shared-memory-dir", type=Path, default=Path("/dev/shm"))
    parser.add_argument("--max-inflight-activations", type=int, default=6)
    parser.add_argument("--inflight-memory-fraction", type=float, default=0.25)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/activation-visualization/wan-cross-q-transform-comparison/autumn-station-seed0"),
    )
    return parser.parse_args()


def load_prompt(path: Path, prompt_id: str) -> tuple[str, dict[str, Any]]:
    collection = json.loads(path.read_text())
    matches = [entry for entry in collection.get("prompts", []) if entry.get("id") == prompt_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one prompt {prompt_id!r} in {path}, found {len(matches)}")
    entry = matches[0]
    return entry["prompt"], {key: entry.get(key) for key in ("id", "title", "output_slug", "word_count", "umt5_token_count")}


def cross_q_only(block_transforms: list[WanBlockTransforms]) -> list[WanBlockTransforms]:
    return [WanBlockTransforms({"cross_q": item.transforms["cross_q"]}) for item in block_transforms]


def mxfp4_metrics(value: torch.Tensor, quantizer: Quantizer) -> dict[str, float]:
    scales, zeros = quantizer.get_quantization_params(value)
    reconstructed = quantizer(value, scales, zeros)
    error = value.float() - reconstructed.float()
    signal_energy = value.float().square().sum().double()
    error_energy = error.square().sum().double()
    sqnr = float("inf") if error_energy == 0 else float(10 * torch.log10(signal_energy / error_energy))
    nonzero = value.ne(0)
    underflow = nonzero & reconstructed.eq(0)
    return {
        "mse": float(error.square().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "sqnr_db": sqnr,
        "max_abs_error": float(error.abs().max()),
        "underflow_fraction_of_nonzero": float(underflow.sum() / nonzero.sum()) if nonzero.any() else 0.0,
    }


def full_statistics(value: torch.Tensor) -> dict[str, float | int]:
    magnitudes = value.detach().float().abs().reshape(-1, value.shape[-1])
    channel_rms = magnitudes.square().mean(dim=0).sqrt()
    median_rms = channel_rms.median()
    flattened = magnitudes.flatten()
    maximum_quantile_samples = 1_000_000
    stride = max(1, (flattened.numel() + maximum_quantile_samples - 1) // maximum_quantile_samples)
    quantile_sample = flattened[::stride]
    quantiles = torch.quantile(
        quantile_sample,
        torch.tensor([0.99, 0.999, 0.9999], device=value.device),
    )
    return {
        "max_abs": float(magnitudes.max()),
        "median_abs": float(magnitudes.median()),
        "p99_abs": float(quantiles[0]),
        "p99.9_abs": float(quantiles[1]),
        "p99.99_abs": float(quantiles[2]),
        "quantile_sample_count": quantile_sample.numel(),
        "quantile_sample_stride": stride,
        "max_channel_rms_over_median": float(channel_rms.max() / median_rms) if median_rms else float("inf"),
        "max_rms_channel": int(channel_rms.argmax()),
    }


def summarize(output_dir: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob("step_*/conditional/block_*/cross_q/*/metadata.json")):
        metadata = json.loads(path.read_text())
        samples.append({
            "step": metadata["sampling_step"],
            "block": metadata["block"],
            "variant": metadata["variant"],
            "full_statistics": metadata["full_statistics"],
            "mxfp4": metadata["mxfp4"],
            "persistent_channels_by_frame": [
                [record["channel"] for record in frame["channel_outliers"]]
                for frame in metadata["records"]
            ],
        })
    aggregate = {}
    for variant in VARIANTS:
        selected = [sample for sample in samples if sample["variant"] == variant]
        aggregate[variant] = {
            "sample_count": len(selected),
            "mean_max_channel_rms_over_median": (
                sum(sample["full_statistics"]["max_channel_rms_over_median"] for sample in selected) / len(selected)
                if selected else None
            ),
            "mean_mxfp4_mse": (
                sum(sample["mxfp4"]["mse"] for sample in selected) / len(selected)
                if selected else None
            ),
            "mean_mxfp4_sqnr_db": (
                sum(sample["mxfp4"]["sqnr_db"] for sample in selected) / len(selected)
                if selected else None
            ),
        }
    return {"variants": list(VARIANTS), "aggregate": aggregate, "samples": samples}


def generate(pipe: Any, args: argparse.Namespace, prompt: str) -> torch.Tensor:
    return pipe.generate(
        input_prompt=prompt,
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


def main() -> None:
    args = parse_args()
    if args.frames % 4 != 1 or args.width % 16 or args.height % 16:
        raise ValueError("frames must be 4n+1 and width/height divisible by 16")
    if args.group_size != 32:
        raise ValueError("This paired MXFP4 experiment requires group-size 32")
    prompt, prompt_source = load_prompt(args.prompt_file, args.prompt_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.wan_repo.resolve()))
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V

    device = torch.device(f"cuda:{args.device_id}")
    pipe = WanT2V(
        config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=str(args.checkpoint),
        device_id=args.device_id, t5_cpu=True,
    )
    blocks = parse_indices(args.blocks, len(pipe.model.blocks))
    sampling_steps = parse_indices(args.sampling_steps, args.steps)
    target_calls = [step * 2 for step in sampling_steps]

    givens = cross_q_only(build_wan_block_transforms(
        pipe.model, "givens", args.group_size, device, quant_scope="attention",
        outlier_threshold=args.outlier_threshold,
    ))
    calibration_handles = observe_wan_transforms(pipe.model, givens)
    try:
        calibration_video = generate(pipe, args, prompt)
        del calibration_video
    finally:
        for handle in calibration_handles:
            handle.remove()
    finalize_wan_transforms(givens)

    hadamard = cross_q_only(build_wan_block_transforms(
        pipe.model, "hadamard", args.group_size, device, quant_scope="attention",
        randomize=True, seed=args.transform_seed,
    ))
    quantizer = Quantizer(
        bits=4, symmetric=True, format="mxfp", granularity="group",
        group_size=32, observer="minmax", scale_precision="e8m0",
    )
    renderer = WanOnlineActivationRenderer(pipe.model, args, blocks, ["cross_q"], target_calls)
    renderer.handles.append(pipe.model.register_forward_pre_hook(renderer._model_pre_hook, with_kwargs=True))
    for block_index in blocks:
        module = pipe.model.blocks[block_index].get_submodule("cross_attn.q")
        if not isinstance(module, nn.Linear):
            raise ValueError(f"Expected Linear at blocks.{block_index}.cross_attn.q")

        def hook(_module: nn.Module, inputs: tuple[Any, ...], current_block: int = block_index) -> None:
            if renderer.call_index not in renderer.target_calls:
                return
            source = inputs[0]
            variants = {
                "identity": source,
                "hadamard-h32": hadamard[current_block].transforms["cross_q"](source),
                "givens-g32": givens[current_block].transforms["cross_q"](source),
            }
            shared_color_max = max(float(value.detach().abs().max()) for value in variants.values())
            for name, value in variants.items():
                metadata = {
                    "transform": name,
                    "paired_color_max": shared_color_max,
                    "full_statistics": full_statistics(value),
                    "mxfp4": mxfp4_metrics(value, quantizer),
                }
                renderer._submit(
                    value, current_block, "cross_q", "cross_attn.q",
                    variant=name, color_max_override=shared_color_max,
                    extra_metadata=metadata,
                )

        renderer.handles.append(module.register_forward_pre_hook(hook))

    config = {
        "render_schema_version": ONLINE_RENDER_SCHEMA_VERSION,
        "experiment": "paired cross_q transform comparison",
        "prompt": prompt,
        "prompt_source": {"file": str(args.prompt_file), **prompt_source},
        "seed": args.seed,
        "size": [args.width, args.height],
        "frames": args.frames,
        "steps": args.steps,
        "sampling_steps": sampling_steps,
        "blocks": blocks,
        "variants": list(VARIANTS),
        "transform_group_size": args.group_size,
        "transform_seed": args.transform_seed,
        "hadamard_randomized_signs": True,
        "givens_outlier_threshold": args.outlier_threshold,
        "givens_calibration": "one complete BF16 generation with the same prompt and seed",
        "givens_stats": get_wan_transform_stats(givens),
        "comparison": "all variants derive from the same BF16 activation and share a per-activation color maximum",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    caught: BaseException | None = None
    try:
        video = generate(pipe, args, prompt)
        del video
    except BaseException as error:
        caught = error
    finally:
        renderer.remove()
    try:
        renderer.finish()
    except BaseException as error:
        if caught is None:
            caught = error
    state = {
        "status": "complete" if caught is None else "interrupted",
        "rendered_activations": renderer.rendered_activations,
        "skipped_activations": renderer.skipped_activations,
        "rendered_images": renderer.rendered_images,
        "expected_activations": len(blocks) * len(target_calls) * len(VARIANTS),
        "error": repr(caught) if caught is not None else None,
    }
    (args.output_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    if caught is not None:
        raise caught
    comparison = summarize(args.output_dir)
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
