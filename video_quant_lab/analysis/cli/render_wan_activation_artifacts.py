"""Render 3D bars and heatmaps from complete BF16 Wan activation files."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

import numpy as np
import torch

from video_quant_lab.analysis.cli.visualize_wan_activation_surfaces import evenly_spaced_indices, render_bars, render_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bar-tokens", type=int, default=128)
    parser.add_argument("--bar-channels", type=int, default=512)
    parser.add_argument("--heatmap-tokens", type=int, default=512)
    parser.add_argument("--heatmap-channels", type=int, default=2048)
    parser.add_argument("--image-width", type=int, default=1400)
    parser.add_argument("--image-height", type=int, default=900)
    return parser.parse_args()


def sampled_view(matrix: torch.Tensor, max_tokens: int, max_channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = matrix.reshape(-1, matrix.shape[-1])
    token_indices = evenly_spaced_indices(matrix.shape[0], max_tokens)
    channel_indices = evenly_spaced_indices(matrix.shape[1], max_channels)
    token_tensor = torch.from_numpy(token_indices)
    channel_tensor = torch.from_numpy(channel_indices)
    view = matrix.index_select(0, token_tensor).index_select(1, channel_tensor).float().numpy()
    return view, token_indices, channel_indices


def render_one(arguments: tuple[str, int, int, int, int, int, int]) -> dict:
    path_text, bar_tokens, bar_channels, heatmap_tokens, heatmap_channels, width, height = arguments
    activation_path = Path(path_text)
    directory = activation_path.parent
    bars_path = directory / "bars.png"
    heatmap_path = directory / "heatmap.png"
    metadata_path = directory / "metadata.json"
    if bars_path.exists() and heatmap_path.exists():
        return {"status": "skipped", "path": path_text}
    metadata = json.loads(metadata_path.read_text())
    activation = torch.load(activation_path, map_location="cpu", weights_only=True)
    title = (
        f"Wan activation | step {metadata['sampling_step']} | {metadata['branch']} | "
        f"block {metadata['block']} | {metadata['site']} | timestep {metadata['timestep']:.5g}"
    )
    render_metadata = metadata.setdefault("render", {})
    if not bars_path.exists():
        values, tokens, channels = sampled_view(activation, bar_tokens, bar_channels)
        temporary = directory / "bars.partial.png"
        stats = render_bars(
            values, tokens, channels, title, temporary, width, height,
            100.0, 99.99, max_background_bars=50000,
        )
        temporary.replace(bars_path)
        render_metadata["bars"] = {
            "file": "bars.png", "view_shape": list(values.shape),
            "token_indices": "evenly_spaced", "channel_indices": "evenly_spaced", **stats,
        }
    if not heatmap_path.exists():
        values, tokens, channels = sampled_view(activation, heatmap_tokens, heatmap_channels)
        temporary = directory / "heatmap.partial.png"
        stats = render_heatmap(values, tokens, channels, title, temporary, width, height, 99.9, 0.45)
        temporary.replace(heatmap_path)
        render_metadata["heatmap"] = {
            "file": "heatmap.png", "view_shape": list(values.shape),
            "token_indices": "evenly_spaced", "channel_indices": "evenly_spaced", **stats,
        }
    metadata["files"].update({"bars": "bars.png", "heatmap": "heatmap.png"})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"status": "rendered", "path": path_text}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    activation_files = sorted(root.rglob("activation.pt"))
    if not activation_files:
        raise ValueError(f"No activation.pt files found under {root}")
    state_path = root / "render_state.json"
    jobs = [
        (str(path), args.bar_tokens, args.bar_channels, args.heatmap_tokens,
         args.heatmap_channels, args.image_width, args.image_height)
        for path in activation_files
    ]
    completed = skipped = failed = 0
    state_path.write_text(json.dumps({"status": "running", "expected": len(jobs)}, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(render_one, job) for job in jobs]
        for future in as_completed(futures):
            try:
                result = future.result()
                if result["status"] == "skipped":
                    skipped += 1
                else:
                    completed += 1
            except BaseException as error:
                failed += 1
                print(f"render failed: {error}", flush=True)
            if (completed + skipped + failed) % 10 == 0:
                state_path.write_text(json.dumps({
                    "status": "running", "expected": len(jobs), "rendered": completed,
                    "skipped": skipped, "failed": failed,
                }, indent=2) + "\n")
    status = "complete" if failed == 0 else "incomplete"
    state = {"status": status, "expected": len(jobs), "rendered": completed, "skipped": skipped, "failed": failed}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
