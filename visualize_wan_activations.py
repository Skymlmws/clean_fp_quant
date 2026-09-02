"""Profile and visualize unquantized Wan2.1 (W16A16/BF16) linear inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from src.utils.wan_activation_stats import WanActivationProfiler
from src.utils.wan_utils import WAN_LINEAR_TRANSFORM_GROUPS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("/home/maoliming/project/checkpoints/Wan2.1-T2V-1.3B"))
    p.add_argument("--wan-repo", type=Path, default=Path("/home/maoliming/project/wan2.1"))
    p.add_argument("--prompt", default="A small red panda walking in a bamboo forest.")
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--guide-scale", type=float, default=5.0)
    p.add_argument("--shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device-id", type=int, default=2)
    p.add_argument("--sample-elements", type=int, default=65536)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/wan_w16a16_activation_profile"))
    return p.parse_args()


def color(value: float, low: float, high: float) -> tuple[int, int, int]:
    t = float(np.clip((value - low) / max(high - low, 1e-12), 0, 1))
    stops = ((35, 28, 90), (34, 144, 140), (253, 231, 37))
    pos = t * 2
    a, b, u = (stops[0], stops[1], pos) if pos <= 1 else (stops[1], stops[2], pos - 1)
    return tuple(round(x + (y - x) * u) for x, y in zip(a, b))


def heatmap(
    matrix: np.ndarray,
    rows: list[str],
    cols: list[str],
    title: str,
    path: Path,
    color_matrix: np.ndarray | None = None,
    scale_label: str = "linear",
) -> None:
    cell_w, cell_h, left, top = 86, 22, 125, 55
    image = Image.new("RGB", (left + cell_w * len(cols) + 20, top + cell_h * len(rows) + 45), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default()
    draw.text((12, 12), title, fill="black", font=font)
    colors = matrix if color_matrix is None else color_matrix
    finite = colors[np.isfinite(colors)]
    low, high = (np.percentile(finite, 5), np.percentile(finite, 95)) if finite.size else (0, 1)
    for j, label in enumerate(cols):
        draw.text((left + j * cell_w + 3, top - 18), label, fill="black", font=font)
    for i, label in enumerate(rows):
        draw.text((5, top + i * cell_h + 5), label, fill="black", font=font)
        for j in range(len(cols)):
            v = matrix[i, j]
            color_value = colors[i, j]
            x, y = left + j * cell_w, top + i * cell_h
            draw.rectangle((x, y, x + cell_w - 2, y + cell_h - 2), fill=color(color_value, low, high))
            draw.text((x + 4, y + 5), f"{v:.3g}", fill="white" if color_value < (low + high) / 2 else "black", font=font)
    draw.text((left, top + cell_h * len(rows) + 10), f"color scale ({scale_label}): p5={low:.2f}, p95={high:.2f}", fill="black", font=font)
    image.save(path)


def channel_plot(profile: dict, path: Path) -> None:
    width, height, margin = 1200, 720, 70
    image = Image.new("RGB", (width, height), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default()
    draw.text((15, 12), "Sorted channel absmax (maximum across layers, normalized by median)", fill="black", font=font)
    palette = [(31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),(140,86,75),(227,119,194)]
    groups = list(WAN_LINEAR_TRANSFORM_GROUPS)
    curves = []
    for group in groups:
        arrays = [v["channel_max"].numpy() for k, v in profile["sites"].items() if k.endswith("." + group)]
        values = np.max(np.stack(arrays), axis=0)
        values = np.sort(values / max(np.median(values), 1e-12))
        curves.append(values)
    ymax = max(np.percentile(v, 99.8) for v in curves)
    draw.line((margin, height-margin, width-margin, height-margin), fill="black", width=2)
    draw.line((margin, margin, margin, height-margin), fill="black", width=2)
    for idx, (group, values) in enumerate(zip(groups, curves)):
        sampled = values[np.linspace(0, len(values)-1, min(len(values), 1000)).astype(int)]
        points = []
        for n, v in enumerate(sampled):
            x = margin + n / max(len(sampled)-1, 1) * (width-2*margin)
            y = height-margin - min(v, ymax) / ymax * (height-2*margin)
            points.append((x, y))
        draw.line(points, fill=palette[idx], width=2)
        draw.text((width-250, 45 + idx*18), group, fill=palette[idx], font=font)
    draw.text((width//2-60, height-35), "channel percentile", fill="black", font=font)
    draw.text((8, margin), f"{ymax:.1f}x", fill="black", font=font)
    image.save(path)


def write_outputs(profile: dict, output_dir: Path, metadata: dict) -> None:
    torch.save({"metadata": metadata, **profile}, output_dir / "activation_stats.pt")
    records = []
    groups = list(WAN_LINEAR_TRANSFORM_GROUPS)
    layer_count = len(profile["sites"]) // len(groups)
    metric_names = (
        "absmax", "p99", "p999", "rms",
        "max_over_rms", "max_over_p999", "block_max_over_rms_p99",
    )
    matrices = {
        name: np.full((layer_count, len(groups)), np.nan)
        for name in metric_names
    }
    for key, site in profile["sites"].items():
        layer = int(key.split(".")[1]); group = key.split(".")[2]; col = groups.index(group)
        for call in site["calls"]:
            row = {"site": key, "layer": layer, "group": group, **call}
            records.append(row)
        for metric in matrices:
            matrices[metric][layer, col] = max(call[metric] for call in site["calls"])
    with (output_dir / "activation_calls.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    for metric, matrix in matrices.items():
        if metric in ("absmax", "rms", "p99", "p999"):
            heatmap(
                matrix, [f"block {i}" for i in range(layer_count)], groups,
                f"W16A16 activation: {metric} (cells are absolute values)",
                output_dir / f"heatmap_{metric}.png",
                color_matrix=np.log10(np.maximum(matrix, 1e-12)),
                scale_label="log10 color; cell text is raw",
            )
        else:
            heatmap(matrix, [f"block {i}" for i in range(layer_count)], groups, f"W16A16 activation: {metric}", output_dir / f"heatmap_{metric}.png")
    absmax = matrices["absmax"]
    heatmap(
        absmax,
        [f"block {i}" for i in range(layer_count)],
        groups,
        "W16A16 activation: absmax (log10 color, cells show absolute values)",
        output_dir / "heatmap_absmax_log10.png",
        color_matrix=np.log10(np.maximum(absmax, 1e-12)),
        scale_label="log10",
    )
    channel_plot(profile, output_dir / "sorted_channel_absmax.png")
    absolute_rows = []
    for group in groups:
        calls = [call for key, site in profile["sites"].items() if key.endswith("." + group) for call in site["calls"]]
        for metric in ("absmax", "rms", "p99", "p999"):
            values = np.asarray([float(call[metric]) for call in calls])
            absolute_rows.append({"group": group, "metric": metric, "minimum": float(values.min()), "median": float(np.median(values)), "mean": float(values.mean()), "maximum": float(values.max())})
    with (output_dir / "activation_absolute_ranges.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(absolute_rows[0])); writer.writeheader(); writer.writerows(absolute_rows)
    summary = {**metadata, "model_calls": profile["model_calls"], "sites": len(profile["sites"]), "artifacts": ["activation_stats.pt", "activation_calls.csv", "activation_absolute_ranges.csv", *[f"heatmap_{x}.png" for x in matrices], "heatmap_absmax_log10.png", "sorted_channel_absmax.png"]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if args.frames % 4 != 1 or args.width % 16 or args.height % 16:
        raise ValueError("frames must be 4n+1 and width/height divisible by 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.wan_repo))
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    pipe = WanT2V(config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=str(args.checkpoint), device_id=args.device_id, t5_cpu=True)
    profiler = WanActivationProfiler(pipe.model, args.sample_elements, args.group_size)
    profiler.attach()
    try:
        with torch.inference_mode():
            pipe.generate(input_prompt=args.prompt, size=(args.width,args.height), frame_num=args.frames, shift=args.shift, sample_solver="unipc", sampling_steps=args.steps, guide_scale=args.guide_scale, n_prompt=args.negative_prompt, seed=args.seed, offload_model=False)
    finally:
        profiler.remove()
    metadata = {"mode": "W16A16 (native BF16, no transform, no fake quantization)", "prompt": args.prompt, "seed": args.seed, "size": [args.width,args.height], "frames": args.frames, "steps": args.steps, "guide_scale": args.guide_scale, "shift": args.shift, "group_size": args.group_size}
    write_outputs(profiler.export(), args.output_dir, metadata)
    print(json.dumps({**metadata, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
