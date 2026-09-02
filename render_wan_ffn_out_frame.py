"""Render one complete Wan ffn_out latent frame as an overview and channel slices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fp-quant-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import torch

from render_wan_self_qkv_frames import frame_view, video_token_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--branch", default="conditional")
    parser.add_argument("--channels-per-slice", type=int, default=1120)
    parser.add_argument("--overview-group-size", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=1800)
    parser.add_argument("--image-height", type=int, default=1900)
    parser.add_argument("--heatmap-percentile", type=float, default=99.9)
    parser.add_argument("--heatmap-gamma", type=float, default=0.45)
    return parser.parse_args()


def render(
    values: np.ndarray,
    path: Path,
    title: str,
    width: int,
    height: int,
    color_max: float,
    gamma: float,
    channel_start: int,
    channel_end: int,
) -> None:
    figure, axis = plt.subplots(
        figsize=(width / 160, height / 160), dpi=160, facecolor="white"
    )
    image = axis.imshow(
        values,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(channel_start, channel_end, 0, values.shape[0]),
        cmap="magma",
        norm=PowerNorm(gamma=gamma, vmin=0, vmax=color_max, clip=True),
        rasterized=True,
    )
    axis.set_xlabel("Channel")
    axis.set_ylabel("Spatial token (row-major H x W)")
    axis.set_title(title, fontsize=10)
    colorbar = figure.colorbar(
        image, ax=axis, pad=0.02, ticks=np.linspace(0.0, color_max, 7)
    )
    colorbar.set_label("|Activation|")
    figure.tight_layout()
    temporary = path.with_name(path.stem + ".partial.png")
    figure.savefig(temporary, dpi=160, facecolor="white")
    plt.close(figure)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    config = json.loads((input_root / "config.json").read_text())
    grid = video_token_grid(config)
    source_dir = (
        input_root / f"step_{args.step:03d}" / args.branch
        / f"block_{args.block:02d}" / "ffn_out"
    )
    source = source_dir / "activation.pt"
    source_metadata = json.loads((source_dir / "metadata.json").read_text())
    activation = torch.load(source, map_location="cpu", weights_only=True)
    frame = frame_view(activation, args.frame, grid).float().abs().numpy()
    token_count, channel_count = frame.shape

    if channel_count % args.overview_group_size:
        raise ValueError("Channel count must be divisible by overview-group-size")
    color_max = max(float(np.percentile(frame, args.heatmap_percentile)), 1e-12)
    overview = frame.reshape(
        token_count, channel_count // args.overview_group_size, args.overview_group_size
    ).max(axis=2)
    overview_color_max = max(
        float(np.percentile(overview, args.heatmap_percentile)), 1e-12
    )

    destination = (
        output_root / f"step_{args.step:03d}" / args.branch
        / f"block_{args.block:02d}" / "ffn_out" / f"frame_{args.frame:03d}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    render(
        overview,
        destination / "overview.png",
        f"Wan ffn_out overview | step {args.step} | block {args.block} | frame {args.frame}",
        args.image_width,
        args.image_height,
        overview_color_max,
        args.heatmap_gamma,
        0,
        channel_count,
    )

    slices = []
    for start in range(0, channel_count, args.channels_per_slice):
        end = min(start + args.channels_per_slice, channel_count)
        filename = f"channels_{start:04d}_{end - 1:04d}.png"
        render(
            frame[:, start:end],
            destination / filename,
            f"Wan ffn_out | step {args.step} | block {args.block} | frame {args.frame} | channels {start}-{end - 1}",
            args.image_width,
            args.image_height,
            color_max,
            args.heatmap_gamma,
            start,
            end,
        )
        slices.append({"file": filename, "channel_range": [start, end]})

    tokens_per_frame = grid[1] * grid[2]
    metadata = {
        "source_activation": str(source),
        "sampling_step": args.step,
        "timestep": source_metadata["timestep"],
        "branch": args.branch,
        "block": args.block,
        "site": "ffn_out",
        "latent_frame": args.frame,
        "source_token_range": [
            args.frame * tokens_per_frame,
            (args.frame + 1) * tokens_per_frame,
        ],
        "matrix_shape": [token_count, channel_count],
        "spatial_grid": list(grid[1:]),
        "absolute_values": True,
        "overview": {
            "file": "overview.png",
            "shape": list(overview.shape),
            "group_size": args.overview_group_size,
            "reduction": "max(abs(x)) over consecutive channels",
            "color_max": overview_color_max,
        },
        "channel_slices": slices,
        "shared_slice_color": {
            "percentile": args.heatmap_percentile,
            "color_max": color_max,
            "gamma": args.heatmap_gamma,
            "full_frame_abs_max": float(frame.max()),
        },
        "image_size": [args.image_width, args.image_height],
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "images": 1 + len(slices)}, indent=2))


if __name__ == "__main__":
    main()
