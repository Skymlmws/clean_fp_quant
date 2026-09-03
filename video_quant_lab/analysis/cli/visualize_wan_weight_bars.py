"""Render Wan Linear weights as input-output-channel 3D bar charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

from video_quant_lab.analysis.wan.sites import WAN_LINEAR_SITES
from video_quant_lab.analysis.wan.wan_activation_surface import parse_indices
from video_quant_lab.analysis.cli.visualize_wan_activation_surfaces import evenly_spaced_indices, render_bars, selected_sites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/maoliming/project/checkpoints/Wan2.1-T2V-1.3B"))
    parser.add_argument("--wan-repo", type=Path, default=Path("/home/maoliming/project/wan2.1"))
    parser.add_argument("--device", default="cpu", help="Model loading device, such as cpu or cuda:2")
    parser.add_argument("--blocks", default="0", help="all, an index, or inclusive ranges such as 0,5-8")
    parser.add_argument("--sites", default="ffn_in", help="all or comma-separated transform-site names")
    parser.add_argument("--linears", default="all", help="all or comma-separated exact Linear names inside selected sites")
    parser.add_argument("--max-input-channels", type=int, default=512, help="Plot limit; 0 keeps every input channel")
    parser.add_argument("--max-output-channels", type=int, default=512, help="Plot limit; 0 keeps every output channel")
    parser.add_argument("--z-percentile", type=float, default=100.0)
    parser.add_argument("--outlier-percentile", type=float, default=99.9)
    parser.add_argument("--image-width", type=int, default=1400)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/profiles/wan-weight-bars"))
    return parser.parse_args()


def selected_linears(spec: str, sites: list[str]) -> list[tuple[str, str]]:
    available = [
        (site, linear)
        for site in sites
        for linear in WAN_LINEAR_SITES[site]
    ]
    if spec.strip().lower() == "all":
        return available
    requested = {item.strip() for item in spec.split(",") if item.strip()}
    result = [(site, linear) for site, linear in available if linear in requested]
    missing = requested - {linear for _, linear in result}
    if missing:
        raise ValueError(f"Selected Linear names are not present in selected sites: {sorted(missing)}")
    return result


def sample_weight(
    weight: torch.Tensor,
    max_output_channels: int,
    max_input_channels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_indices = evenly_spaced_indices(weight.shape[0], max_output_channels)
    input_indices = evenly_spaced_indices(weight.shape[1], max_input_channels)
    matrix = weight.detach().float().cpu()[output_indices][:, input_indices].numpy()
    return matrix, output_indices, input_indices


def safe_linear_name(name: str) -> str:
    return name.replace(".", "_")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.wan_repo))
    from wan.modules.model import WanModel

    device = torch.device(args.device)
    model = WanModel.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    blocks = parse_indices(args.blocks, len(model.blocks))
    sites = selected_sites(args.sites)
    linears = selected_linears(args.linears, sites)
    records = []
    for block_index in blocks:
        modules = dict(model.blocks[block_index].named_modules())
        for site, linear_name in linears:
            linear = modules.get(linear_name)
            if not isinstance(linear, nn.Linear):
                raise ValueError(f"Expected Linear at blocks.{block_index}.{linear_name}")
            matrix, output_indices, input_indices = sample_weight(
                linear.weight, args.max_output_channels, args.max_input_channels
            )
            relative_dir = Path(f"block_{block_index:02d}") / site / safe_linear_name(linear_name)
            artifact_dir = args.output_dir / relative_dir
            artifact_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                artifact_dir / "weight.npz",
                weight=matrix,
                output_channel_indices=output_indices,
                input_channel_indices=input_indices,
            )
            stats = render_bars(
                matrix, output_indices, input_indices,
                f"Wan weight | block {block_index} | {linear_name}",
                artifact_dir / "bars.png", args.image_width, args.image_height,
                args.z_percentile, args.outlier_percentile,
                x_label="Input channel", y_label="Output channel", z_label="|Weight|",
            )
            metadata = {
                "block": block_index, "site": site, "linear": linear_name,
                "original_shape": list(linear.weight.shape), "matrix_shape": list(matrix.shape),
                "source_dtype": str(linear.weight.dtype), "stored_dtype": str(matrix.dtype),
                "axis": {"x": "input channel", "y": "output channel", "z": "absolute weight magnitude"},
                "files": {"bars": "bars.png", "weight": "weight.npz"}, **stats,
            }
            (artifact_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            records.append({
                **metadata, "directory": relative_dir.as_posix(),
                "bars": (relative_dir / "bars.png").as_posix(),
                "weight": (relative_dir / "weight.npz").as_posix(),
                "metadata": (relative_dir / "metadata.json").as_posix(),
            })

    config = {
        "mode": "native BF16 Wan Linear weights",
        "checkpoint": str(args.checkpoint), "device": str(device),
        "blocks": blocks, "sites": sites, "linears": [name for _, name in linears],
        "plot_limits": {"input_channels": args.max_input_channels, "output_channels": args.max_output_channels},
        "render": {"z_percentile": args.z_percentile, "outlier_percentile": args.outlier_percentile},
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "format_version": 1, "config": "config.json",
        "weight_count": len(records), "weights": records,
    }, indent=2) + "\n")
    print(json.dumps({"weights": len(records), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
