"""Render isolated Wan activation outliers from previously stored BF16 tensors."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fp-quant-matplotlib-isolated")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
import torch

from render_wan_self_qkv_frames import frame_view, parse_selection, video_token_grid
from src.utils.wan_activation_outliers import isolated_token_outliers, persistent_channel_outliers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--steps", default="10")
    parser.add_argument("--blocks", default="0,10,20,29")
    parser.add_argument("--frames", default="0,10,20")
    parser.add_argument("--sites", default="self_qkv,self_o,cross_q,cross_o,ffn_in,ffn_out")
    parser.add_argument("--persistent-ratio", type=float, default=5.0)
    parser.add_argument("--max-persistent", type=int, default=8)
    parser.add_argument("--global-percentile", type=float, default=99.99)
    parser.add_argument("--channel-percentile", type=float, default=99.0)
    parser.add_argument("--isolated-ratio", type=float, default=5.0)
    parser.add_argument("--max-token-fraction", type=float, default=0.01)
    parser.add_argument("--max-isolated", type=int, default=10)
    parser.add_argument("--image-width", type=int, default=1900)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def spatial_location(token: int, columns: int) -> dict[str, int]:
    return {"spatial_y": token // columns, "spatial_x": token % columns}


def annotation_text(persistent: list[dict], isolated: list[dict]) -> str:
    lines = ["persistent channels:"]
    lines.extend(
        f"ch {row['channel']}: robust RMS/median {row['robust_rms_over_median']:.2f}"
        for row in persistent
    )
    if not persistent:
        lines.append("none")
    lines.append("isolated points / clusters:")
    lines.extend(
        f"{index}: tok {row['token_start']}-{row['token_end']}, ch {row['channel']}, "
        f"abs {row['abs_value']:.4g}, ratio {row['peak_over_channel_baseline']:.2f}"
        for index, row in enumerate(isolated, 1)
    )
    if not isolated:
        lines.append("none")
    return "\n".join(lines)


def render_marked_heatmap(
    values: np.ndarray,
    title: str,
    output: Path,
    persistent: list[dict],
    isolated: list[dict],
    width: int,
    height: int,
) -> None:
    magnitudes = np.abs(values.astype(np.float32, copy=False))
    color_max = max(float(magnitudes.max()), 1e-12)
    annotation_width = 650
    figure, axis = plt.subplots(
        figsize=((width + annotation_width) / 160, height / 160), dpi=160,
        facecolor="white",
    )
    image = axis.imshow(
        magnitudes, origin="lower", aspect="auto", interpolation="nearest",
        extent=(0, values.shape[1], 0, values.shape[0]), cmap="magma",
        norm=PowerNorm(gamma=1.0, vmin=0, vmax=color_max, clip=True), rasterized=True,
    )
    for row in persistent:
        channel = row["channel"]
        axis.scatter(
            [channel + 0.5], [-0.012], transform=axis.get_xaxis_transform(),
            marker="^", s=22, color="#00e5ff", edgecolors="black",
            linewidths=0.35, clip_on=False, zorder=5,
        )
        axis.text(
            channel + 0.5, -0.032, f"ch {channel}",
            transform=axis.get_xaxis_transform(), ha="center", va="top",
            fontsize=6.5, color="#00e5ff", clip_on=False,
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 1.0},
        )
    for index, row in enumerate(isolated, 1):
        token = row["peak_token"]
        channel = row["channel"]
        axis.scatter(
            [channel + 0.5], [token + 0.5], marker="o", s=42,
            facecolors="none", edgecolors="#39ff14", linewidths=1.0, zorder=6,
        )
        axis.annotate(
            str(index), (channel + 0.5, token + 0.5), xytext=(4, 4),
            textcoords="offset points", fontsize=6.5, color="#39ff14", zorder=7,
        )
    axis.set_xlabel("Channel")
    axis.set_ylabel("Token")
    axis.set_title(title, fontsize=10)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("|Activation|")
    main_fraction = width / (width + annotation_width)
    figure.tight_layout(rect=(0.0, 0.0, main_fraction, 1.0))
    figure.text(
        (width + 20) / (width + annotation_width), 0.94,
        annotation_text(persistent, isolated), ha="left", va="top", fontsize=7.2,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.4"},
    )
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    config = json.loads((input_root / "config.json").read_text())
    grid = video_token_grid(config)
    steps = parse_selection(args.steps)
    blocks = parse_selection(args.blocks)
    frames = parse_selection(args.frames)
    sites = {value.strip() for value in args.sites.split(",") if value.strip()}
    output_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "mode": "offline isolated activation outlier visualization",
        "input_root": str(input_root),
        "steps": sorted(steps) if steps is not None else "all",
        "blocks": sorted(blocks) if blocks is not None else "all",
        "frames": sorted(frames) if frames is not None else "all",
        "sites": sorted(sites),
        "persistent_ratio": args.persistent_ratio,
        "global_percentile": args.global_percentile,
        "channel_percentile": args.channel_percentile,
        "isolated_ratio": args.isolated_ratio,
        "max_token_fraction": args.max_token_fraction,
    }
    (output_root / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    rendered = skipped = 0
    sources = sorted(input_root.glob("step_*/conditional/block_*/*/activation.pt"))
    for source in sources:
        metadata = json.loads((source.parent / "metadata.json").read_text())
        step, block, site = metadata["sampling_step"], metadata["block"], metadata["site"]
        if steps is not None and step not in steps or blocks is not None and block not in blocks or site not in sites:
            continue
        activation = torch.load(source, map_location="cpu", weights_only=True)
        if site == "cross_kv":
            views = [(None, activation.reshape(-1, activation.shape[-1]).float().numpy())]
        else:
            views = [
                (frame, frame_view(activation, frame, grid).float().numpy())
                for frame in range(grid[0]) if frames is None or frame in frames
            ]
        for frame, values in views:
            relative = Path(f"step_{step:03d}") / "conditional" / f"block_{block:02d}" / site
            if frame is not None:
                relative /= f"frame_{frame:03d}"
            destination = output_root / relative
            image_path = destination / "heatmap.png"
            metadata_path = destination / "metadata.json"
            if image_path.exists() and metadata_path.exists() and not args.overwrite:
                skipped += 1
                continue
            destination.mkdir(parents=True, exist_ok=True)
            persistent = persistent_channel_outliers(values, args.persistent_ratio, args.max_persistent)
            isolated = isolated_token_outliers(
                values, {row["channel"] for row in persistent}, args.global_percentile,
                args.channel_percentile, args.isolated_ratio, args.max_token_fraction,
                args.max_isolated,
            )
            if frame is not None:
                for row in isolated:
                    row.update(spatial_location(row["peak_token"], grid[2]))
                    row["latent_frame"] = frame
            title = f"Wan {site} | step {step} | block {block} | " + (
                "text tokens" if frame is None else f"latent frame {frame}/{grid[0] - 1}"
            )
            render_marked_heatmap(values, title, image_path, persistent, isolated, args.image_width, args.image_height)
            record = {
                "source_activation": str(source), "sampling_step": step,
                "block": block, "site": site, "latent_frame": frame,
                "matrix_shape": list(values.shape), "persistent_channels": persistent,
                "isolated_outliers": isolated, "files": {"heatmap": "heatmap.png"},
            }
            metadata_path.write_text(json.dumps(record, indent=2) + "\n")
            rendered += 1
        del activation
    state = {"status": "complete", "rendered": rendered, "skipped": skipped}
    (output_root / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
