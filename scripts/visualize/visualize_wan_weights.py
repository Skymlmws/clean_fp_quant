"""Profile Wan2.1 BF16 weights and their untransformed MXFP4 error."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

from src.quantization.quantizer import Quantizer
from src.utils.wan_utils import WAN_LINEAR_TRANSFORM_GROUPS
from video_quant_lab.analysis.cli.visualize_wan_activations import channel_plot, heatmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("/home/maoliming/project/checkpoints/Wan2.1-T2V-1.3B"))
    p.add_argument("--wan-repo", type=Path, default=Path("/home/maoliming/project/wan2.1"))
    p.add_argument("--device-id", type=int, default=2)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/profiles/wan_w16_weight_profile"))
    return p.parse_args()


@torch.no_grad()
def analyze_weight(weight: torch.Tensor, quantizer: Quantizer, group_size: int) -> tuple[dict[str, float | int], torch.Tensor, torch.Tensor]:
    w = weight.detach().float()
    absolute = w.abs()
    quantiles = torch.quantile(absolute.flatten(), absolute.new_tensor([0.5, 0.99, 0.999]))
    rms = w.square().mean().sqrt().clamp_min(1e-12)
    blocks = w.reshape(-1, group_size)
    block_rms = blocks.square().mean(dim=1).sqrt().clamp_min(1e-12)
    block_ratio = blocks.abs().amax(dim=1) / block_rms
    scales, zeros = quantizer.get_quantization_params(w)
    reconstructed = quantizer(w, scales, zeros)
    error = w - reconstructed
    noise = error.square().sum().double().clamp_min(1e-30)
    signal = w.square().sum().double().clamp_min(1e-30)
    channel_max = absolute.amax(dim=0).cpu()
    channel_rms = w.square().mean(dim=0).sqrt().cpu()
    metrics: dict[str, float | int] = {
        "out_features": w.shape[0],
        "in_features": w.shape[1],
        "absmax": absolute.max().item(),
        "rms": rms.item(),
        "p50": quantiles[0].item(),
        "p99": quantiles[1].item(),
        "p999": quantiles[2].item(),
        "max_over_rms": (absolute.max() / rms).item(),
        "max_over_p999": (absolute.max() / quantiles[2].clamp_min(1e-12)).item(),
        "block_max_over_rms_mean": block_ratio.mean().item(),
        "block_max_over_rms_p99": torch.quantile(block_ratio, 0.99).item(),
        "w4_mse": error.square().mean().item(),
        "w4_relative_mse": (noise / signal).item(),
        "w4_sqnr_db": (10 * torch.log10(signal / noise)).item(),
        "w4_cosine": torch.nn.functional.cosine_similarity(w.flatten(), reconstructed.flatten(), dim=0).item(),
    }
    return metrics, channel_max, channel_rms


def write_outputs(profile: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(profile, output_dir / "weight_stats.pt")
    rows = profile["linears"]
    with (output_dir / "weight_linears.csv").open("w", newline="") as f:
        serializable = [{k: v for k, v in row.items() if not isinstance(v, torch.Tensor)} for row in rows]
        writer = csv.DictWriter(f, fieldnames=list(serializable[0])); writer.writeheader(); writer.writerows(serializable)

    groups = list(WAN_LINEAR_TRANSFORM_GROUPS)
    layers = max(int(row["layer"]) for row in rows) + 1
    absolute_metrics = ("absmax", "rms", "p50", "p99", "p999")
    metric_names = (*absolute_metrics, "max_over_rms", "max_over_p999", "block_max_over_rms_p99", "w4_sqnr_db", "w4_relative_mse")
    matrices = {metric: np.full((layers, len(groups)), np.nan) for metric in metric_names}
    for layer in range(layers):
        for col, group in enumerate(groups):
            matches = [row for row in rows if row["layer"] == layer and row["group"] == group]
            for metric in metric_names:
                values = [float(row[metric]) for row in matches]
                # Worst member for q/k/v groups; low SQNR is worse, others high are worse.
                matrices[metric][layer, col] = min(values) if metric == "w4_sqnr_db" else max(values)
    labels = [f"block {i}" for i in range(layers)]
    for metric, matrix in matrices.items():
        if metric in absolute_metrics:
            heatmap(
                matrix, labels, groups,
                f"Original BF16 weight: {metric} (cells are absolute values)",
                output_dir / f"heatmap_weight_{metric}.png",
                color_matrix=np.log10(np.maximum(matrix, 1e-12)),
                scale_label="log10 color; cell text is raw",
            )
        else:
            heatmap(matrix, labels, groups, f"Wan weight: {metric}", output_dir / f"heatmap_weight_{metric}.png")

    absolute_rows = []
    for group in groups:
        matches = [row for row in rows if row["group"] == group]
        for metric in absolute_metrics:
            values = np.asarray([float(row[metric]) for row in matches])
            absolute_rows.append({
                "group": group, "metric": metric,
                "minimum": float(values.min()), "median": float(np.median(values)),
                "mean": float(values.mean()), "maximum": float(values.max()),
            })
    with (output_dir / "weight_absolute_ranges.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(absolute_rows[0])); writer.writeheader(); writer.writerows(absolute_rows)

    # Reuse the activation curve renderer's site schema.
    curve_profile = {"sites": {}}
    for layer in range(layers):
        for group in groups:
            values = [row["channel_max"] for row in rows if row["layer"] == layer and row["group"] == group]
            curve_profile["sites"][f"blocks.{layer}.{group}"] = {"channel_max": torch.stack(values).amax(dim=0)}
    channel_plot(curve_profile, output_dir / "sorted_weight_input_channel_absmax.png")
    summary = {
        "mode": "BF16 weights with Identity MXFP4 W4 fake-quant analysis",
        "linears": len(rows), "layers": layers, "group_size": profile["group_size"],
        "artifacts": ["weight_stats.pt", "weight_linears.csv", "weight_absolute_ranges.csv", *[f"heatmap_weight_{m}.png" for m in metric_names], "sorted_weight_input_channel_absmax.png"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.group_size != 32:
        raise ValueError("This MXFP4 analysis requires group-size=32")
    sys.path.insert(0, str(args.wan_repo))
    from wan.modules.model import WanModel
    device = torch.device(f"cuda:{args.device_id}")
    model = WanModel.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16).to(device).eval()
    quantizer = Quantizer(bits=4, symmetric=True, format="mxfp", granularity="group", group_size=32, scale_precision="e8m0", observer="minmax")
    rows = []
    for layer, block in enumerate(model.blocks):
        modules = dict(block.named_modules())
        for group, names in WAN_LINEAR_TRANSFORM_GROUPS.items():
            for linear_name in names:
                linear = modules.get(linear_name)
                if not isinstance(linear, nn.Linear):
                    raise ValueError(f"Expected Linear at blocks.{layer}.{linear_name}")
                metrics, channel_max, channel_rms = analyze_weight(linear.weight, quantizer, args.group_size)
                rows.append({"name": f"blocks.{layer}.{group}.{linear_name}", "layer": layer, "group": group, "linear": linear_name, **metrics, "channel_max": channel_max, "channel_rms": channel_rms})
    profile = {"group_size": args.group_size, "linears": rows}
    write_outputs(profile, args.output_dir)
    print(json.dumps({"mode": "BF16 weight + Identity W4 analysis", "linears": len(rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
