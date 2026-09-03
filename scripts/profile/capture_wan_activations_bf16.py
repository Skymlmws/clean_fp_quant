"""Generate one Wan video while streaming complete conditional activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from src.utils.wan_activation_disk_capture import QuotaExceeded, WanActivationDiskCapture
from src.utils.wan_activation_surface import parse_indices
from src.utils.wan_utils import WAN_LINEAR_TRANSFORM_GROUPS
from scripts.visualize.visualize_wan_activation_surfaces import selected_sites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/maoliming/project/checkpoints/Wan2.1-T2V-1.3B"))
    parser.add_argument("--wan-repo", type=Path, default=Path("/home/maoliming/project/wan2.1"))
    parser.add_argument("--prompt", default="A small red panda walking in a bamboo forest.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampling-steps", default="10,25,40")
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--device-id", type=int, default=2)
    parser.add_argument("--blocks", default="all")
    parser.add_argument("--sites", default="all")
    parser.add_argument("--quota-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-output-gb", type=float, default=200.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/activation-visualization/wan-activation-short-prompt/wan-activation-480p-seed0-full-bf16"),
    )
    return parser.parse_args()


def write_state(path: Path, status: str, capture: WanActivationDiskCapture, expected: int, error: str | None = None) -> None:
    state = {
        "status": status,
        "completed_this_run": capture.completed,
        "skipped_existing": capture.skipped,
        "expected": expected,
        "bytes_written_this_run": capture.bytes_written,
        "error": error,
    }
    (path / "state.json").write_text(json.dumps(state, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.frames % 4 != 1 or args.width % 16 or args.height % 16:
        raise ValueError("frames must be 4n+1 and width/height divisible by 16")
    if args.max_output_gb <= 0:
        raise ValueError("max-output-gb must be positive for complete activation capture")
    sampling_steps = parse_indices(args.sampling_steps, args.steps)
    call_indices = [step * 2 for step in sampling_steps]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.quota_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.wan_repo))
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    from wan.utils.utils import cache_video

    pipe = WanT2V(config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=str(args.checkpoint), device_id=args.device_id, t5_cpu=True)
    blocks = parse_indices(args.blocks, len(pipe.model.blocks))
    sites = selected_sites(args.sites)
    expected = len(blocks) * len(sites) * len(call_indices)
    capture = WanActivationDiskCapture(
        pipe.model, args.output_dir, args.quota_dir,
        int(args.max_output_gb * 1024**3), blocks, sites, call_indices,
        batch_index=0,
    )
    config = {
        "mode": "complete activation capture stored as BF16",
        "prompt": args.prompt, "negative_prompt": args.negative_prompt,
        "seed": args.seed, "size": [args.width, args.height], "frames": args.frames,
        "fps": args.fps, "steps": args.steps, "sampling_steps": sampling_steps,
        "call_indices": call_indices, "branches": ["conditional"],
        "blocks": blocks, "sites": sites, "sampling": None,
        "storage_dtype": "torch.bfloat16",
        "quota_dir": str(args.quota_dir), "max_output_gb": args.max_output_gb,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_state(args.output_dir, "running", capture, expected)
    capture.attach()
    try:
        with torch.inference_mode():
            video = pipe.generate(
                input_prompt=args.prompt, size=(args.width, args.height), frame_num=args.frames,
                shift=args.shift, sample_solver="unipc", sampling_steps=args.steps,
                guide_scale=args.guide_scale, n_prompt=args.negative_prompt,
                seed=args.seed, offload_model=False,
            )
        cache_video(video[None], save_file=str(args.output_dir / "video.mp4"), fps=args.fps)
    except QuotaExceeded as error:
        write_state(args.output_dir, "paused_quota", capture, expected, str(error))
        raise
    except BaseException as error:
        write_state(args.output_dir, "interrupted", capture, expected, repr(error))
        raise
    finally:
        capture.remove()
    total_complete = sum(1 for path in args.output_dir.rglob("activation.pt") if (path.parent / "metadata.json").exists())
    status = "complete" if total_complete == expected else "incomplete"
    config["text_context_by_call"] = {
        str(call): metadata for call, metadata in sorted(capture.text_context_by_call.items())
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_state(args.output_dir, status, capture, expected)
    print(json.dumps({"status": status, "complete": total_complete, "expected": expected, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
