"""Capture Wan cross_q calibration state and raw activations for offline analysis."""

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
from src.utils.wan_utils import build_wan_block_transforms, observe_wan_transforms


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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(value, partial)
    partial.replace(path)


def main() -> None:
    args = parse_args()
    if not 0 <= args.capture_step < args.steps:
        raise ValueError("capture-step must be in [0, steps)")
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
    transforms = build_wan_block_transforms(
        pipe.model, "givens", args.group_size, device, quant_scope="attention",
        outlier_threshold=float("inf"), seed=args.transform_seed,
    )
    transforms = [type(item)({"cross_q": item.transforms["cross_q"]}) for item in transforms]
    calibration_handles = observe_wan_transforms(pipe.model, transforms)
    try:
        calibration_video = generate(pipe, args, prompt)
        del calibration_video
    finally:
        for handle in calibration_handles:
            handle.remove()

    calibration_blocks = []
    for block_index, item in enumerate(transforms):
        transform = item.transforms["cross_q"]
        if not isinstance(transform, GivensTransform):
            raise TypeError(f"Expected GivensTransform at block {block_index}")
        calibration_blocks.append({"block": block_index, **transform.calibration_state()})
    calibration_path = args.output_dir / "cross_q_calibration.pt"
    atomic_torch_save({
        "schema_version": 1,
        "site": "cross_q",
        "group_size": args.group_size,
        "model_block_count": len(pipe.model.blocks),
        "hidden_size": int(pipe.model.dim),
        "prompt_id": args.prompt_id,
        "seed": args.seed,
        "transform_seed": args.transform_seed,
        "steps": args.steps,
        "blocks": calibration_blocks,
    }, calibration_path)
    del transforms, calibration_blocks

    call_index = -1
    target_call = args.capture_step * 2
    captured: list[dict[str, Any]] = []

    def model_pre_hook(_module: nn.Module, _inputs: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        nonlocal call_index
        call_index += 1

    handles = [pipe.model.register_forward_pre_hook(model_pre_hook, with_kwargs=True)]
    for block_index, block in enumerate(pipe.model.blocks):
        module = block.get_submodule("cross_attn.q")
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Expected Linear at block {block_index} cross_attn.q")

        def capture_hook(
            _module: nn.Module,
            inputs: tuple[Any, ...],
            current_block: int = block_index,
        ) -> None:
            if call_index != target_call:
                return
            source = inputs[0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            relative = Path("activations") / f"block_{current_block:02d}.pt"
            atomic_torch_save(source, args.output_dir / relative)
            captured.append({
                "block": current_block,
                "file": str(relative),
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "bytes": source.numel() * source.element_size(),
            })

        handles.append(module.register_forward_pre_hook(capture_hook))

    try:
        capture_video = generate(pipe, args, prompt)
        del capture_video
    finally:
        for handle in handles:
            handle.remove()

    captured.sort(key=lambda item: item["block"])
    manifest = {
        "schema_version": 1,
        "status": "complete" if len(captured) == len(pipe.model.blocks) else "incomplete",
        "site": "cross_q",
        "linear": "cross_attn.q",
        "branch": "conditional",
        "capture_step": args.capture_step,
        "call_index": target_call,
        "prompt": prompt,
        "prompt_source": {"file": str(args.prompt_file), **prompt_source},
        "seed": args.seed,
        "size": [args.width, args.height],
        "frames": args.frames,
        "steps": args.steps,
        "group_size": args.group_size,
        "calibration_file": calibration_path.name,
        "captured_block_count": len(captured),
        "total_activation_bytes": sum(item["bytes"] for item in captured),
        "activations": captured,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "status": manifest["status"],
        "captured_block_count": len(captured),
        "total_activation_bytes": manifest["total_activation_bytes"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
