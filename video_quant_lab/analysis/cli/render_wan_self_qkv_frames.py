"""Render complete Wan activation heatmaps, splitting video-token sites by frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from video_quant_lab.analysis.cli.visualize_wan_activation_surfaces import plt, render_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--steps", default="all", help="all or comma-separated sampling-step numbers")
    parser.add_argument("--blocks", default="all", help="all or comma-separated block numbers")
    parser.add_argument("--frames", default="all", help="all or comma-separated latent-frame numbers")
    parser.add_argument(
        "--sites", default="self_qkv,self_o,cross_q,cross_kv,cross_o,ffn_in",
        help="Comma-separated activation sites; cross_kv is rendered as one text-token heatmap",
    )
    parser.add_argument("--image-width", type=int, default=1800)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument("--heatmap-percentile", type=float, default=99.9)
    parser.add_argument("--heatmap-gamma", type=float, default=0.45)
    parser.add_argument(
        "--channel-rms-ratio", type=float, default=5.0,
        help="Minimum robust channel RMS divided by the median robust channel RMS",
    )
    parser.add_argument(
        "--mark-top-channels", type=int, default=8,
        help="Maximum persistent bright channels marked on each heatmap; 0 disables",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_selection(spec: str) -> set[int] | None:
    if spec.strip().lower() == "all":
        return None
    return {int(value.strip()) for value in spec.split(",") if value.strip()}


def video_token_grid(config: dict) -> tuple[int, int, int]:
    width, height = config["size"]
    frames = config["frames"]
    if frames % 4 != 1 or height % 16 or width % 16:
        raise ValueError("Expected Wan dimensions: frames=4n+1 and height/width divisible by 16")
    return (frames - 1) // 4 + 1, height // 16, width // 16


def frame_view(activation: torch.Tensor, frame: int, grid: tuple[int, int, int]) -> torch.Tensor:
    temporal, rows, columns = grid
    matrix = activation.reshape(-1, activation.shape[-1])
    expected_tokens = temporal * rows * columns
    if matrix.shape[0] != expected_tokens:
        raise ValueError(f"Expected {expected_tokens} video tokens, found {matrix.shape[0]}")
    if not 0 <= frame < temporal:
        raise IndexError(f"Frame {frame} is outside [0, {temporal})")
    tokens_per_frame = rows * columns
    start = frame * tokens_per_frame
    return matrix[start:start + tokens_per_frame]


def activation_statistics(values: np.ndarray) -> dict[str, float]:
    magnitudes = np.abs(values.astype(np.float32, copy=False))
    maximum = float(magnitudes.max())
    p99, p999, p9999 = (
        float(value) for value in np.percentile(magnitudes, [99.0, 99.9, 99.99])
    )
    channel_rms = np.sqrt(np.mean(np.square(magnitudes), axis=0))
    median_channel_rms = float(np.median(channel_rms))
    return {
        "max": maximum,
        "median": float(np.median(magnitudes)),
        "p99": p99,
        "p99.9": p999,
        "p99.99": p9999,
        "max_over_p99": maximum / p99 if p99 else float("inf"),
        "max_over_p99.9": maximum / p999 if p999 else float("inf"),
        "max_channel_rms_over_median": (
            float(channel_rms.max()) / median_channel_rms
            if median_channel_rms else float("inf")
        ),
    }


def channel_outliers(
    values: np.ndarray,
    channel_indices: np.ndarray,
    minimum_rms_ratio: float,
    maximum_channels: int,
) -> list[dict[str, float | int]]:
    """Rank persistent bright lines by per-channel 95%-winsorized RMS."""
    if minimum_rms_ratio <= 0:
        raise ValueError("channel-rms-ratio must be positive")
    if maximum_channels < 0:
        raise ValueError("mark-top-channels must be non-negative")
    if maximum_channels == 0:
        return []
    magnitudes = np.abs(values.astype(np.float32, copy=False))
    channel_rms = np.sqrt(np.mean(np.square(magnitudes), axis=0))
    channel_p95 = np.percentile(magnitudes, 95.0, axis=0)
    robust_rms = np.sqrt(
        np.mean(np.square(np.minimum(magnitudes, channel_p95[None, :])), axis=0)
    )
    median_rms = float(np.median(robust_rms))
    if median_rms <= 0:
        return []
    ratios = robust_rms / median_rms
    candidates = np.flatnonzero(ratios >= minimum_rms_ratio)
    if candidates.size == 0:
        return []
    order = candidates[np.argsort(ratios[candidates])[::-1]][:maximum_channels]
    return [
        {
            "channel": int(channel_indices[index]),
            "rms": float(channel_rms[index]),
            "robust_rms": float(robust_rms[index]),
            "robust_rms_over_median": float(ratios[index]),
            "max_abs": float(magnitudes[:, index].max()),
        }
        for index in order
    ]


def append_channel_outliers(
    annotation: str, records: list[dict[str, float | int]]
) -> str:
    if not records:
        return annotation + "\nmarked channels: none"
    lines = [annotation, "marked persistent channels:"]
    lines.extend(
        f"ch {record['channel']}: robust RMS/median "
        f"{record['robust_rms_over_median']:.2f}, "
        f"max {record['max_abs']:.4g}"
        for record in records
    )
    return "\n".join(lines)


def statistics_annotation(statistics: dict[str, float]) -> str:
    return (
        f"max: {statistics['max']:.4g}\n"
        f"median: {statistics['median']:.4g}\n"
        f"p99: {statistics['p99']:.4g}\n"
        f"p99.9: {statistics['p99.9']:.4g}\n"
        f"p99.99: {statistics['p99.99']:.4g}\n"
        f"max / p99: {statistics['max_over_p99']:.2f}\n"
        f"max / p99.9: {statistics['max_over_p99.9']:.2f}\n"
        f"max channel RMS / median: {statistics['max_channel_rms_over_median']:.2f}"
    )


def ffn_out_slice_annotation(
    frame_statistics: dict[str, float], slice_statistics: dict[str, float]
) -> str:
    return (
        f"frame max: {frame_statistics['max']:.4g}\n"
        f"slice max: {slice_statistics['max']:.4g}\n"
        f"slice median: {slice_statistics['median']:.4g}\n"
        f"slice p99: {slice_statistics['p99']:.4g}\n"
        f"slice p99.9: {slice_statistics['p99.9']:.4g}\n"
        f"slice p99.99: {slice_statistics['p99.99']:.4g}\n"
        f"slice max / p99: {slice_statistics['max_over_p99']:.2f}\n"
        f"slice max / p99.9: {slice_statistics['max_over_p99.9']:.2f}\n"
        f"slice max channel RMS / median: "
        f"{slice_statistics['max_channel_rms_over_median']:.2f}"
    )


def render_channel_summary(
    values: np.ndarray,
    channel_start: int,
    title: str,
    path: Path,
    width: int,
    height: int,
    shared_y_max: float,
) -> dict:
    magnitudes = np.abs(values.astype(np.float32, copy=False))
    channel_max = magnitudes.max(axis=0)
    channel_p999 = np.percentile(magnitudes, 99.9, axis=0)
    channel_rms = np.sqrt(np.mean(np.square(magnitudes), axis=0))
    channel_median = np.median(magnitudes, axis=0)
    channels = np.arange(channel_start, channel_start + values.shape[1])
    figure, axis = plt.subplots(
        figsize=(width / 160, height / 160), dpi=160, facecolor="white"
    )
    axis.bar(
        channels, channel_max, width=0.9, color="#ef3b2c", alpha=0.72,
        linewidth=0, label="max",
    )
    axis.plot(channels, channel_p999, color="#ff9f1c", linewidth=0.9, label="p99.9")
    axis.plot(channels, channel_rms, color="#277da1", linewidth=0.9, label="RMS")
    axis.plot(channels, channel_median, color="#6c757d", linewidth=0.8, label="median")
    axis.set_xlim(float(channels[0]) - 0.5, float(channels[-1]) + 0.5)
    axis.set_ylim(0, max(shared_y_max * 1.04, 1e-12))
    axis.set_xlabel("Channel")
    axis.set_ylabel("|Activation| statistic over tokens")
    axis.set_title(title, fontsize=10)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper right", ncol=4)
    top_count = min(5, channel_max.size)
    top_local = np.argpartition(channel_max, -top_count)[-top_count:]
    top_local = top_local[np.argsort(channel_max[top_local])[::-1]]
    for local_index in top_local:
        axis.annotate(
            f"ch {int(channels[local_index])}: {channel_max[local_index]:.4g}",
            (channels[local_index], channel_max[local_index]),
            xytext=(0, 5), textcoords="offset points",
            horizontalalignment="center", fontsize=6.5, rotation=90,
        )
    figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)
    return {
        "shared_y_max": float(shared_y_max),
        "series": ["max", "p99.9", "RMS", "median"],
        "top_channels_by_max": [
            {"channel": int(channels[index]), "max": float(channel_max[index])}
            for index in top_local
        ],
    }


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    config = json.loads((input_root / "config.json").read_text())
    grid = video_token_grid(config)
    selected_steps = parse_selection(args.steps)
    selected_blocks = parse_selection(args.blocks)
    selected_frames = parse_selection(args.frames)
    selected_sites = {value.strip() for value in args.sites.split(",") if value.strip()}
    allowed_sites = {"self_qkv", "self_o", "cross_q", "cross_kv", "cross_o", "ffn_in", "ffn_out"}
    if not selected_sites or not selected_sites <= allowed_sites:
        raise ValueError(f"Sites must be selected from {sorted(allowed_sites)}")
    sources = sorted(input_root.glob("step_*/conditional/block_*/*/activation.pt"))
    rendered = skipped = 0

    output_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "mode": "complete activation heatmaps; video-token sites split by latent frame",
        "input_root": str(input_root),
        "sites": sorted(selected_sites),
        "video_token_grid": list(grid),
        "frame_matrix_shape": [grid[1] * grid[2], 1536],
        "axis": {"x": "channel", "y": "spatial token (row-major H x W)"},
        "heatmap_percentile": args.heatmap_percentile,
        "heatmap_gamma": args.heatmap_gamma,
        "channel_marker": {
            "score": (
                "per-channel 95%-winsorized RMS over tokens divided by median "
                "winsorized channel RMS"
            ),
            "minimum_ratio": args.channel_rms_ratio,
            "maximum_channels": args.mark_top_channels,
        },
    }
    (output_root / "config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    for activation_path in sources:
        source_metadata = json.loads((activation_path.parent / "metadata.json").read_text())
        step = int(source_metadata["sampling_step"])
        block = int(source_metadata["block"])
        site = source_metadata["site"]
        if site not in selected_sites:
            continue
        if selected_steps is not None and step not in selected_steps:
            continue
        if selected_blocks is not None and block not in selected_blocks:
            continue
        destination = (
            output_root / f"step_{step:03d}" / source_metadata["branch"]
            / f"block_{block:02d}" / site
        )
        metadata_path = destination / "metadata.json"
        if site == "ffn_out":
            existing_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            frame_records = {
                int(record["latent_frame"]): record
                for record in existing_metadata.get("frames", [])
            }
            activation = torch.load(activation_path, map_location="cpu", weights_only=True)
            channel_count = int(activation.shape[-1])
            shard_count = 8
            if channel_count % shard_count:
                raise ValueError(
                    f"ffn_out channel count {channel_count} is not divisible by {shard_count}"
                )
            shard_width = channel_count // shard_count
            overview_group_size = 8
            if channel_count % overview_group_size:
                raise ValueError(
                    f"ffn_out channel count {channel_count} is not divisible by "
                    f"overview group size {overview_group_size}"
                )
            for frame in range(grid[0]):
                if selected_frames is not None and frame not in selected_frames:
                    continue
                frame_destination = destination / f"frame_{frame:03d}"
                overview_path = frame_destination / "overview.png"
                shard_paths = [
                    frame_destination / f"channels_{start:04d}_{start + shard_width - 1:04d}.png"
                    for start in range(0, channel_count, shard_width)
                ]
                summary_paths = [
                    frame_destination
                    / f"channels_{start:04d}_{start + shard_width - 1:04d}_summary.png"
                    for start in range(0, channel_count, shard_width)
                ]
                expected_paths = [overview_path, *shard_paths, *summary_paths]
                if (
                    all(path.exists() for path in expected_paths)
                    and frame in frame_records
                    and not args.overwrite
                ):
                    skipped += len(expected_paths)
                    continue
                frame_destination.mkdir(parents=True, exist_ok=True)
                values = frame_view(activation, frame, grid).float().numpy()
                frame_stats = activation_statistics(values)
                frame_outlier_channels = channel_outliers(
                    values,
                    np.arange(channel_count, dtype=np.int64),
                    args.channel_rms_ratio,
                    args.mark_top_channels,
                )
                shared_color_max = frame_stats["max"]
                tokens = np.arange(values.shape[0], dtype=np.int64)
                overview = np.abs(values).reshape(
                    values.shape[0], channel_count // overview_group_size, overview_group_size
                ).max(axis=2)
                overview_groups = np.arange(overview.shape[1], dtype=np.int64)
                base_title = (
                    f"Wan ffn_out | sampling step {step} | timestep {source_metadata['timestep']} | "
                    f"block {block} | latent frame {frame}/{grid[0] - 1}"
                )
                temporary = frame_destination / "overview.partial.png"
                overview_render = render_heatmap(
                    overview,
                    tokens,
                    overview_groups,
                    f"{base_title} | max-abs groups of {overview_group_size} channels",
                    temporary,
                    args.image_width,
                    args.image_height,
                    args.heatmap_percentile,
                    args.heatmap_gamma,
                    use_bin_edges=True,
                    annotation_text=append_channel_outliers(
                        statistics_annotation(frame_stats), frame_outlier_channels
                    ),
                    color_max_override=shared_color_max,
                )
                temporary.replace(overview_path)
                rendered += 1
                shard_records = []
                for shard_index, start in enumerate(range(0, channel_count, shard_width)):
                    end = start + shard_width
                    shard_values = values[:, start:end]
                    shard_stats = activation_statistics(shard_values)
                    shard_outlier_channels = [
                        record for record in frame_outlier_channels
                        if start <= record["channel"] < end
                    ]
                    image_path = shard_paths[shard_index]
                    temporary = frame_destination / f"{image_path.stem}.partial.png"
                    shard_render = render_heatmap(
                        shard_values,
                        tokens,
                        np.arange(start, end, dtype=np.int64),
                        f"{base_title} | channels {start}-{end - 1}",
                        temporary,
                        args.image_width,
                        args.image_height,
                        args.heatmap_percentile,
                        args.heatmap_gamma,
                        use_bin_edges=True,
                        annotation_text=append_channel_outliers(
                            ffn_out_slice_annotation(frame_stats, shard_stats),
                            shard_outlier_channels,
                        ),
                        color_max_override=shared_color_max,
                        marked_channels=[
                            record["channel"] for record in shard_outlier_channels
                        ],
                    )
                    temporary.replace(image_path)
                    rendered += 1
                    summary_path = summary_paths[shard_index]
                    temporary = frame_destination / f"{summary_path.stem}.partial.png"
                    summary_render = render_channel_summary(
                        shard_values,
                        start,
                        f"{base_title} | channels {start}-{end - 1} | channel summary",
                        temporary,
                        args.image_width,
                        args.image_height,
                        shared_color_max,
                    )
                    temporary.replace(summary_path)
                    rendered += 1
                    shard_records.append({
                        "shard_index": shard_index,
                        "channel_range": [start, end],
                        "channel_range_semantics": "start inclusive, end exclusive",
                        "matrix_shape": list(shard_values.shape),
                        "files": {
                            "heatmap": image_path.name,
                            "summary": summary_path.name,
                        },
                        "statistics": shard_stats,
                        "channel_outliers": shard_outlier_channels,
                        "render": shard_render,
                        "summary_render": summary_render,
                    })
                frame_records[frame] = {
                    "latent_frame": frame,
                    "source_token_range": [
                        frame * grid[1] * grid[2],
                        (frame + 1) * grid[1] * grid[2],
                    ],
                    "matrix_shape": list(values.shape),
                    "shared_color_max": shared_color_max,
                    "statistics": frame_stats,
                    "channel_outliers": frame_outlier_channels,
                    "overview": {
                        "matrix_shape": list(overview.shape),
                        "source_channel_count": channel_count,
                        "output_channel_groups": int(overview.shape[1]),
                        "channels_per_group": overview_group_size,
                        "aggregation": "max_abs",
                        "files": {"heatmap": f"frame_{frame:03d}/{overview_path.name}"},
                        "render": overview_render,
                    },
                    "shards": shard_records,
                }
            metadata = {
                "source_activation": str(activation_path),
                "sampling_step": step,
                "timestep": source_metadata["timestep"],
                "branch": source_metadata["branch"],
                "block": block,
                "site": site,
                "video_token_grid": list(grid),
                "spatial_grid": list(grid[1:]),
                "source_channel_count": channel_count,
                "shard_count": shard_count,
                "shard_width": shard_width,
                "frame_count": len(frame_records),
                "frames": [frame_records[index] for index in sorted(frame_records)],
            }
            destination.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            continue
        if site == "cross_kv":
            image_path = destination / "heatmap.png"
            if image_path.exists() and metadata_path.exists() and not args.overwrite:
                skipped += 1
                continue
            destination.mkdir(parents=True, exist_ok=True)
            activation = torch.load(activation_path, map_location="cpu", weights_only=True)
            values = activation.reshape(-1, activation.shape[-1]).float().numpy()
            activation_stats = activation_statistics(values)
            tokens = np.arange(values.shape[0], dtype=np.int64)
            channels = np.arange(values.shape[1], dtype=np.int64)
            outlier_channels = channel_outliers(
                values, channels, args.channel_rms_ratio, args.mark_top_channels
            )
            temporary = destination / "heatmap.partial.png"
            stats = render_heatmap(
                values, tokens, channels,
                f"Wan cross_kv | sampling step {step} | timestep {source_metadata['timestep']} | "
                f"block {block} | text tokens",
                temporary, args.image_width, args.image_height,
                args.heatmap_percentile, args.heatmap_gamma, use_bin_edges=True,
                annotation_text=append_channel_outliers(
                    statistics_annotation(activation_stats), outlier_channels
                ),
                marked_channels=[record["channel"] for record in outlier_channels],
            )
            temporary.replace(image_path)
            metadata = {
                "source_activation": str(activation_path), "sampling_step": step,
                "timestep": source_metadata["timestep"], "branch": source_metadata["branch"],
                "block": block, "site": site, "token_kind": "text",
                "matrix_shape": list(values.shape), "complete": True,
                "files": {"heatmap": image_path.name}, "statistics": activation_stats,
                "channel_outliers": outlier_channels, "render": stats,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            rendered += 1
            continue
        existing_metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        frame_records = {
            int(record["latent_frame"]): record
            for record in existing_metadata.get("frames", [])
        }
        legacy_metadata_paths = sorted(destination.glob("frame_*.json"))
        for legacy_path in legacy_metadata_paths:
            legacy = json.loads(legacy_path.read_text())
            frame = int(legacy["latent_frame"])
            frame_records[frame] = {
                "latent_frame": frame,
                "source_token_range": legacy["source_token_range"],
                "matrix_shape": legacy["matrix_shape"],
                "complete_frame": legacy["complete_frame"],
                "files": legacy["files"],
                "render": legacy["render"],
            }
        activation = torch.load(activation_path, map_location="cpu", weights_only=True)
        for frame in range(grid[0]):
            if selected_frames is not None and frame not in selected_frames:
                continue
            image_path = destination / f"frame_{frame:03d}.png"
            if image_path.exists() and frame in frame_records and not args.overwrite:
                skipped += 1
                continue
            destination.mkdir(parents=True, exist_ok=True)
            values = frame_view(activation, frame, grid).float().numpy()
            activation_stats = activation_statistics(values)
            tokens = np.arange(values.shape[0], dtype=np.int64)
            channels = np.arange(values.shape[1], dtype=np.int64)
            outlier_channels = channel_outliers(
                values, channels, args.channel_rms_ratio, args.mark_top_channels
            )
            title = (
                f"Wan {site} | sampling step {step} | timestep {source_metadata['timestep']} | "
                f"block {block} | latent frame {frame}/{grid[0] - 1} | "
                f"spatial grid {grid[1]}x{grid[2]}"
            )
            temporary = destination / "heatmap.partial.png"
            stats = render_heatmap(
                values, tokens, channels, title, temporary,
                args.image_width, args.image_height,
                args.heatmap_percentile, args.heatmap_gamma,
                use_bin_edges=True,
                annotation_text=append_channel_outliers(
                    statistics_annotation(activation_stats), outlier_channels
                ),
                marked_channels=[record["channel"] for record in outlier_channels],
            )
            temporary.replace(image_path)
            frame_records[frame] = {
                "latent_frame": frame,
                "source_token_range": [frame * grid[1] * grid[2], (frame + 1) * grid[1] * grid[2]],
                "matrix_shape": list(values.shape),
                "complete_frame": True,
                "files": {"heatmap": image_path.name},
                "statistics": activation_stats,
                "channel_outliers": outlier_channels,
                "render": stats,
            }
            rendered += 1
        metadata = {
            "source_activation": str(activation_path),
            "sampling_step": step,
            "timestep": source_metadata["timestep"],
            "branch": source_metadata["branch"],
            "block": block,
            "site": site,
            "video_token_grid": list(grid),
            "spatial_grid": list(grid[1:]),
            "frame_count": len(frame_records),
            "frames": [frame_records[index] for index in sorted(frame_records)],
        }
        destination.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        for legacy_path in legacy_metadata_paths:
            legacy_path.unlink()

    state = {"status": "complete", "rendered": rendered, "skipped": skipped}
    (output_root / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
