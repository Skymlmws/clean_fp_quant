"""Capture calibration state and raw Wan activations for one transform site."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn

from scripts.visualize.compare_wan_cross_q_transforms import generate, load_prompt
from src.transforms.transforms import GivensTransform
from src.utils.wan_utils import WAN_LINEAR_TRANSFORM_GROUPS, build_wan_block_transforms, observe_wan_transforms


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
    parser.add_argument("--capture-step", type=int, default=10)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--transform-seed", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument(
        "--site", required=True,
        help="One site or a comma-separated list of sites from WAN_LINEAR_TRANSFORM_GROUPS",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(value, partial)
    partial.replace(path)


def main() -> None:
    args = parse_args()
    sites = tuple(dict.fromkeys(value.strip() for value in args.site.split(",") if value.strip()))
    unknown_sites = sorted(set(sites) - set(WAN_LINEAR_TRANSFORM_GROUPS))
    if not sites or unknown_sites:
        raise ValueError(f"Invalid sites {unknown_sites}; expected values from {tuple(WAN_LINEAR_TRANSFORM_GROUPS)}")
    if not 0 <= args.capture_step < args.steps:
        raise ValueError("capture-step must be in [0, steps)")
    prompt, prompt_source = load_prompt(args.prompt_file, args.prompt_id)
    site_dirs = {
        site: args.output_dir if len(sites) == 1 else args.output_dir / site
        for site in sites
    }
    for site_dir in site_dirs.values():
        site_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.wan_repo.resolve()))
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V

    device = torch.device(f"cuda:{args.device_id}")
    pipe = WanT2V(
        config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=str(args.checkpoint),
        device_id=args.device_id, t5_cpu=True,
    )
    transforms = build_wan_block_transforms(
        pipe.model, "givens", args.group_size, device, quant_scope="all",
        outlier_threshold=float("inf"), seed=args.transform_seed,
    )
    transforms = [type(item)({site: item.transforms[site] for site in sites}) for item in transforms]

    call_index = -1
    target_call = args.capture_step * 2
    captured: dict[str, list[dict[str, Any]]] = {site: [] for site in sites}

    def model_pre_hook(_module: nn.Module, _inputs: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        nonlocal call_index
        call_index += 1

    handles = list(observe_wan_transforms(pipe.model, transforms))
    handles.append(pipe.model.register_forward_pre_hook(model_pre_hook, with_kwargs=True))
    for current_site in sites:
        capture_linear = WAN_LINEAR_TRANSFORM_GROUPS[current_site][0]
        for block_index, block in enumerate(pipe.model.blocks):
            module = block.get_submodule(capture_linear)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"Expected Linear at block {block_index} {capture_linear}")

            def capture_hook(
                _module: nn.Module,
                inputs: tuple[Any, ...],
                hook_block: int = block_index,
                hook_site: str = current_site,
            ) -> None:
                if call_index != target_call:
                    return
                source = inputs[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
                relative = Path("activations") / f"block_{hook_block:02d}.pt"
                atomic_torch_save(source, site_dirs[hook_site] / relative)
                captured[hook_site].append({
                    "block": hook_block,
                    "file": str(relative),
                    "shape": list(source.shape),
                    "dtype": str(source.dtype),
                    "bytes": source.numel() * source.element_size(),
                })

            handles.append(module.register_forward_pre_hook(capture_hook))

    try:
        video = generate(pipe, args, prompt)
        del video
    finally:
        for handle in handles:
            handle.remove()

    summaries = []
    for current_site in sites:
        calibration_blocks = []
        for block_index, item in enumerate(transforms):
            transform = item.transforms[current_site]
            if not isinstance(transform, GivensTransform):
                raise TypeError(f"Expected GivensTransform at block {block_index}")
            calibration_blocks.append({"block": block_index, **transform.calibration_state()})
        site_dir = site_dirs[current_site]
        calibration_path = site_dir / f"{current_site}_calibration.pt"
        atomic_torch_save({
            "schema_version": 1, "site": current_site, "group_size": args.group_size,
            "model_block_count": len(pipe.model.blocks),
            "hidden_size": int(pipe.model.ffn_dim if current_site == "ffn_out" else pipe.model.dim),
            "prompt_id": args.prompt_id, "seed": args.seed,
            "transform_seed": args.transform_seed, "steps": args.steps,
            "blocks": calibration_blocks,
        }, calibration_path)
        site_captured = sorted(captured[current_site], key=lambda item: item["block"])
        manifest = {
            "schema_version": 1,
            "status": "complete" if len(site_captured) == len(pipe.model.blocks) else "incomplete",
            "site": current_site,
            "linear": WAN_LINEAR_TRANSFORM_GROUPS[current_site][0],
            "branch": "conditional", "capture_step": args.capture_step,
            "call_index": target_call, "prompt": prompt,
            "prompt_source": {"file": str(args.prompt_file), **prompt_source},
            "seed": args.seed, "transform_seed": args.transform_seed,
            "size": [args.width, args.height], "frames": args.frames, "steps": args.steps,
            "group_size": args.group_size, "calibration_file": calibration_path.name,
            "captured_block_count": len(site_captured),
            "total_activation_bytes": sum(item["bytes"] for item in site_captured),
            "activations": site_captured,
        }
        (site_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        summaries.append({
            "status": manifest["status"], "site": current_site,
            "captured_block_count": len(site_captured),
            "total_activation_bytes": manifest["total_activation_bytes"],
            "output_dir": str(site_dir),
        })
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
