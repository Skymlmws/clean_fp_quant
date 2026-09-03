"""Render Wan linear-input activations as token-channel 3D bar charts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fp-quant-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import torch

from video_quant_lab.analysis.wan.sites import WAN_LINEAR_SITES
from video_quant_lab.analysis.wan.wan_activation_surface import WanActivationMatrixCapture, parse_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/maoliming/project/checkpoints/Wan2.1-T2V-1.3B"))
    parser.add_argument("--wan-repo", type=Path, default=Path("/home/maoliming/project/wan2.1"))
    parser.add_argument("--prompt", default="A small red panda walking in a bamboo forest.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=2)
    parser.add_argument("--blocks", default="all", help="all, an index, or comma-separated inclusive ranges such as 0,5-8")
    parser.add_argument("--sites", default="all", help="all or comma-separated site names")
    parser.add_argument("--call-index", type=int, default=0, help="Denoising model call to capture")
    parser.add_argument("--call-indices", help="Multiple calls, such as 10,11,50,51; overrides --call-index")
    parser.add_argument("--capture-only", action="store_true", help="Save NPZ/metadata without rendering PNGs")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0, help="Capture limit; 0 keeps every token")
    parser.add_argument("--max-channels", type=int, default=0, help="Capture limit; 0 keeps every channel")
    parser.add_argument("--z-percentile", type=float, default=100.0, help="Symmetric vertical clipping percentile; 100 keeps the full value range")
    parser.add_argument("--outlier-percentile", type=float, default=99.99, help="Absolute-value percentile highlighted in red")
    parser.add_argument("--max-background-bars", type=int, default=50000, help="Maximum blue context bars; red outliers always use the full matrix")
    parser.add_argument("--heatmap-percentile", type=float, default=99.9, help="Heatmap color maximum percentile")
    parser.add_argument("--heatmap-gamma", type=float, default=0.45, help="Power normalization; below 1 reveals smaller values")
    parser.add_argument("--image-width", type=int, default=1400)
    parser.add_argument("--image-height", type=int, default=900)
    parser.add_argument("--max-output-gb", type=float, default=200.0, help="Pause after this output size; 0 disables")
    parser.add_argument("--max-images", type=int, default=0, help="Pause after this many completed images; 0 disables")
    parser.add_argument("--quota-dir", type=Path, help="Shared directory used for size/image limits across workers")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/activation-visualization/wan-activation-surfaces"),
    )
    return parser.parse_args()


def evenly_spaced_indices(length: int, maximum: int) -> np.ndarray:
    if maximum < 0:
        raise ValueError("Sampling limits must be non-negative; zero means unlimited")
    count = length if maximum == 0 else min(length, maximum)
    return np.linspace(0, length - 1, count).round().astype(np.int64)


def downsample(matrix: torch.Tensor, max_tokens: int, max_channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    token_indices = evenly_spaced_indices(matrix.shape[0], max_tokens)
    channel_indices = evenly_spaced_indices(matrix.shape[1], max_channels)
    values = matrix[token_indices][:, channel_indices].numpy()
    return values, token_indices, channel_indices


def render_bars(
    values: np.ndarray,
    token_indices: np.ndarray,
    channel_indices: np.ndarray,
    title: str,
    path: Path,
    width: int,
    height: int,
    z_percentile: float,
    outlier_percentile: float = 99.9,
    x_label: str = "Channel",
    y_label: str = "Token",
    z_label: str = "|Activation|",
    max_background_bars: int = 50000,
) -> dict[str, float]:
    """Render DuQuant-Figure-1-style absolute activation columns."""
    if not 0 < z_percentile <= 100:
        raise ValueError("z-percentile must be in (0, 100]")
    if not 0 < outlier_percentile <= 100:
        raise ValueError("outlier-percentile must be in (0, 100]")
    if max_background_bars < 0:
        raise ValueError("max-background-bars must be non-negative")
    rows, cols = values.shape
    magnitudes = np.abs(values).astype(np.float32, copy=False)
    limit = float(np.percentile(magnitudes, z_percentile))
    limit = max(limit, 1e-12)
    clipped = np.minimum(magnitudes, limit)
    outlier_threshold = float(np.percentile(magnitudes, outlier_percentile))

    channel_grid, token_grid = np.meshgrid(channel_indices, token_indices)
    x = channel_grid.ravel().astype(np.float32)
    y = token_grid.ravel().astype(np.float32)
    z = clipped.ravel()
    outliers = magnitudes.ravel() >= outlier_threshold

    figure = plt.figure(figsize=(width / 160, height / 160), dpi=160, facecolor="white")
    axis = figure.add_subplot(111, projection="3d")

    def add_columns(mask: np.ndarray, color: str, linewidth: float, alpha: float) -> None:
        selected_x, selected_y, selected_z = x[mask], y[mask], z[mask]
        starts = np.column_stack((selected_x, selected_y, np.zeros_like(selected_z)))
        ends = np.column_stack((selected_x, selected_y, selected_z))
        segments = np.stack((starts, ends), axis=1)
        collection = Line3DCollection(segments, colors=color, linewidths=linewidth, alpha=alpha)
        collection.set_rasterized(True)
        axis.add_collection3d(collection)

    background_indices = np.flatnonzero(~outliers)
    if max_background_bars and background_indices.size > max_background_bars:
        positions = np.linspace(0, background_indices.size - 1, max_background_bars).round().astype(np.int64)
        background_indices = background_indices[positions]
    background_mask = np.zeros_like(outliers)
    background_mask[background_indices] = True
    add_columns(background_mask, "#3155c6", 0.28, 0.38)
    add_columns(outliers, "#ef3b2c", 1.0, 0.98)
    axis.set_xlim(float(channel_indices[0]), float(channel_indices[-1]))
    axis.set_ylim(float(token_indices[0]), float(token_indices[-1]))
    axis.set_zlim(0, limit * 1.04)
    axis.set_xlabel(x_label, labelpad=8)
    axis.set_ylabel(y_label, labelpad=8)
    axis.set_zlabel(z_label, labelpad=6)
    axis.set_title(title, pad=8, fontsize=10)
    axis.view_init(elev=23, azim=-67)
    axis.set_box_aspect((1.65, 1.0, 0.82))
    axis.tick_params(labelsize=7, pad=1)
    for pane_axis in (axis.xaxis, axis.yaxis, axis.zaxis):
        pane_axis.pane.set_facecolor((0.97, 0.97, 0.97, 0.22))
        pane_axis.pane.set_edgecolor((0.82, 0.82, 0.82, 0.7))
    axis.grid(True, color="#d8d8d8", linewidth=0.45)
    figure.subplots_adjust(left=0.01, right=0.96, bottom=0.02, top=0.92)
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)
    return {
        "minimum": float(values.min()), "maximum": float(values.max()),
        "max_abs": float(magnitudes.max()), "z_limit": limit,
        "outlier_threshold": outlier_threshold, "outlier_count": int(outliers.sum()),
        "total_values": int(magnitudes.size),
        "rendered_background_count": int(background_mask.sum()),
    }


def render_heatmap(
    values: np.ndarray,
    token_indices: np.ndarray,
    channel_indices: np.ndarray,
    title: str,
    path: Path,
    width: int,
    height: int,
    color_percentile: float = 99.9,
    gamma: float = 0.45,
    use_bin_edges: bool = False,
    annotation_text: str | None = None,
    color_max_override: float | None = None,
    marked_channels: list[int] | None = None,
    channel_marker_labels: dict[int, str] | None = None,
    marked_points: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Render the full absolute activation matrix as a readable 2D heatmap."""
    if not 0 < color_percentile <= 100:
        raise ValueError("heatmap percentile must be in (0, 100]")
    if gamma <= 0:
        raise ValueError("heatmap gamma must be positive")
    magnitudes = np.abs(values).astype(np.float32, copy=False)
    color_max = (
        max(float(color_max_override), 1e-12)
        if color_max_override is not None
        else max(float(np.percentile(magnitudes, color_percentile)), 1e-12)
    )
    if use_bin_edges:
        x_extent = (float(channel_indices[0]), float(channel_indices[-1] + 1))
        y_extent = (float(token_indices[0]), float(token_indices[-1] + 1))
    else:
        x_extent = (float(channel_indices[0]), float(channel_indices[-1]))
        y_extent = (float(token_indices[0]), float(token_indices[-1]))
    marker_labels = channel_marker_labels or {
        channel: f"ch {channel}" for channel in (marked_channels or [])
    }
    annotation_width = (560 if marker_labels else 400) if annotation_text else 0
    canvas_width = width + annotation_width
    figure, axis = plt.subplots(figsize=(canvas_width / 160, height / 160), dpi=160, facecolor="white")
    image = axis.imshow(
        magnitudes,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(*x_extent, *y_extent),
        cmap="magma",
        norm=PowerNorm(gamma=gamma, vmin=0, vmax=color_max, clip=True),
        rasterized=True,
    )
    axis.set_xlabel("Channel")
    axis.set_ylabel("Token")
    if use_bin_edges:
        axis.set_xticks(np.linspace(x_extent[0], x_extent[1], 9))
        axis.set_yticks(np.linspace(y_extent[0], y_extent[1], 9))
    axis.set_title(title, fontsize=10)
    for channel, marker_label in marker_labels.items():
        marker_x = channel + (0.5 if use_bin_edges else 0.0)
        axis.scatter(
            [marker_x],
            [-0.012],
            transform=axis.get_xaxis_transform(),
            marker="^",
            s=22,
            color="#00e5ff",
            edgecolors="black",
            linewidths=0.35,
            clip_on=False,
            zorder=5,
        )
        axis.text(
            marker_x,
            -0.032,
            marker_label,
            transform=axis.get_xaxis_transform(),
            rotation=0,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=6.5,
            color="#00e5ff",
            bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.55, "pad": 1.0},
            clip_on=False,
        )
    for point in marked_points or []:
        marker_x = float(point["channel"]) + (0.5 if use_bin_edges else 0.0)
        marker_y = float(point["token"]) + (0.5 if use_bin_edges else 0.0)
        axis.scatter(
            [marker_x], [marker_y], marker="o", s=42,
            facecolors="none", edgecolors="#39ff14", linewidths=1.0,
            clip_on=True, zorder=6,
        )
        axis.annotate(
            str(point.get("label", "")), (marker_x, marker_y), xytext=(4, 4),
            textcoords="offset points", fontsize=6.5, color="#39ff14",
            clip_on=True, zorder=7,
        )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("|Activation|")
    if annotation_text:
        main_fraction = width / canvas_width
        figure.tight_layout(rect=(0.0, 0.0, main_fraction, 1.0))
        figure.text(
            (width + 20) / canvas_width, 0.92, annotation_text,
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=7.5,
            color="black",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.4"},
        )
    else:
        figure.tight_layout()
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)
    return {
        "heatmap_color_max": color_max,
        "heatmap_percentile": color_percentile,
        "heatmap_gamma": gamma,
        "color_max_override": color_max_override,
        "marked_channels": list(marked_channels or []),
        "channel_marker_labels": marker_labels,
        "marked_points": list(marked_points or []),
    }


def selected_sites(spec: str) -> list[str]:
    available = list(WAN_LINEAR_SITES)
    if spec.strip().lower() == "all":
        return available
    result = [item.strip() for item in spec.split(",") if item.strip()]
    unknown = sorted(set(result) - set(available))
    if unknown:
        raise ValueError(f"Unknown sites {unknown}; choices are {available}")
    if not result:
        raise ValueError("No sites were selected")
    return result


def branch_for_call(call: int) -> str:
    return "conditional" if call % 2 == 0 else "unconditional"


def call_relative_dir(call: int) -> Path:
    return Path(f"step_{call // 2:03d}") / branch_for_call(call)


def artifact_relative_dir(call: int, block: int, site: str) -> Path:
    """Canonical sampling-step/branch location inside a run directory."""
    return call_relative_dir(call) / f"block_{block:02d}" / site


def output_usage(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return sum(item.stat().st_size for item in files), sum(item.name == "bars.png" for item in files)


def main() -> None:
    args = parse_args()
    if args.call_index < 0 or args.batch_index < 0:
        raise ValueError("call-index and batch-index must be non-negative")
    if args.frames % 4 != 1 or args.width % 16 or args.height % 16:
        raise ValueError("frames must be 4n+1 and width/height divisible by 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quota_dir = args.quota_dir or args.output_dir
    quota_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.wan_repo))
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V

    pipe = WanT2V(config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=str(args.checkpoint), device_id=args.device_id, t5_cpu=True)
    blocks = parse_indices(args.blocks, len(pipe.model.blocks))
    sites = selected_sites(args.sites)
    call_indices = parse_indices(args.call_indices, args.steps * 2) if args.call_indices else [args.call_index]
    capture = WanActivationMatrixCapture(
        pipe.model, blocks, sites, args.call_index, call_indices, args.batch_index,
        args.max_tokens, args.max_channels,
    )
    capture.attach()
    try:
        with torch.inference_mode():
            pipe.generate(input_prompt=args.prompt, size=(args.width, args.height), frame_num=args.frames, shift=args.shift, sample_solver="unipc", sampling_steps=args.steps, guide_scale=args.guide_scale, n_prompt=args.negative_prompt, seed=args.seed, offload_model=False)
    finally:
        capture.remove()

    expected = len(blocks) * len(sites) * len(call_indices)
    if len(capture.captures) != expected:
        raise RuntimeError(f"Captured {len(capture.captures)} matrices, expected {expected}; call-index {args.call_index} may not exist")
    records = []
    calls: dict[int, dict] = {}
    paused = False
    for item in capture.captures:
        values = item.matrix.numpy()
        token_indices = item.token_indices.numpy()
        channel_indices = item.channel_indices.numpy()
        relative_dir = artifact_relative_dir(item.call, item.block, item.site)
        artifact_dir = args.output_dir / relative_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        complete_files = [artifact_dir / "activation.npz", artifact_dir / "metadata.json"]
        if not args.capture_only:
            complete_files.extend((artifact_dir / "bars.png", artifact_dir / "heatmap.png"))
        if all(path.exists() for path in complete_files):
            metadata = json.loads((artifact_dir / "metadata.json").read_text())
            record = {
                **metadata, "directory": relative_dir.as_posix(),
                "activation": (relative_dir / "activation.npz").as_posix(),
                "metadata": (relative_dir / "metadata.json").as_posix(),
            }
            if not args.capture_only:
                record.update({
                    "bars": (relative_dir / "bars.png").as_posix(),
                    "heatmap": (relative_dir / "heatmap.png").as_posix(),
                })
            records.append(record)
            calls.setdefault(item.call, {"call": item.call, "timestep": item.timestep, "captures": 0})
            calls[item.call]["captures"] += 1
            continue
        used_bytes, image_count = output_usage(quota_dir)
        size_limit = int(args.max_output_gb * 1024**3) if args.max_output_gb > 0 else 0
        if (size_limit and used_bytes >= size_limit) or (args.max_images and image_count >= args.max_images):
            paused = True
            break
        activation_path = artifact_dir / "activation.npz"
        if not activation_path.exists():
            np.savez_compressed(
                activation_path, activation=values,
                token_indices=token_indices, channel_indices=channel_indices,
            )
        title = f"Wan activation | block {item.block} | {item.site} | call {item.call} | timestep {item.timestep:.5g}"
        metadata_path = artifact_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {
            "block": item.block, "site": item.site, "linear": item.linear,
            "call": item.call, "timestep": item.timestep,
            "original_shape": list(item.original_shape),
            "matrix_shape": list(values.shape),
            "dtype": "float32",
            "axis": {"x": "channel", "y": "token", "z": "absolute activation magnitude"},
        }
        if not args.capture_only and not (artifact_dir / "bars.png").exists():
            metadata.update(render_bars(
                values, token_indices, channel_indices, title,
                artifact_dir / "bars.png", args.image_width, args.image_height,
                args.z_percentile, args.outlier_percentile,
                max_background_bars=args.max_background_bars,
            ))
        if not args.capture_only and not (artifact_dir / "heatmap.png").exists():
            metadata.update(render_heatmap(
                values, token_indices, channel_indices, title,
                artifact_dir / "heatmap.png", args.image_width, args.image_height,
                args.heatmap_percentile, args.heatmap_gamma,
            ))
        metadata["sampling_step"] = item.call // 2
        metadata["branch"] = branch_for_call(item.call)
        metadata["files"] = {"activation": "activation.npz"}
        if not args.capture_only:
            metadata["files"].update({"bars": "bars.png", "heatmap": "heatmap.png"})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        record = {
            **metadata,
            "directory": relative_dir.as_posix(),
            "activation": (relative_dir / "activation.npz").as_posix(),
            "metadata": (relative_dir / "metadata.json").as_posix(),
        }
        if not args.capture_only:
            record.update({
                "bars": (relative_dir / "bars.png").as_posix(),
                "heatmap": (relative_dir / "heatmap.png").as_posix(),
            })
        records.append(record)
        calls.setdefault(item.call, {"call": item.call, "timestep": item.timestep, "captures": 0})
        calls[item.call]["captures"] += 1

    config = {
        "mode": "native BF16 activation matrices, no transform or fake quantization",
        "axis": {"x": "channel", "y": "token", "z": "absolute activation magnitude"},
        "prompt": args.prompt, "seed": args.seed, "size": [args.width, args.height],
        "frames": args.frames, "steps": args.steps, "blocks": blocks, "sites": sites,
        "call_indices": call_indices, "batch_index": args.batch_index,
        "capture_only": args.capture_only,
        "capture_limits": {"tokens": args.max_tokens, "channels": args.max_channels},
        "render": {
            "image_width": args.image_width, "image_height": args.image_height,
            "z_percentile": args.z_percentile,
            "outlier_percentile": args.outlier_percentile,
            "max_background_bars": args.max_background_bars,
            "heatmap_percentile": args.heatmap_percentile,
            "heatmap_gamma": args.heatmap_gamma,
        },
        "quota": {"directory": str(quota_dir), "max_output_gb": args.max_output_gb, "max_images": args.max_images},
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for call, call_metadata in calls.items():
        call_dir = args.output_dir / call_relative_dir(call)
        call_metadata.update({"sampling_step": call // 2, "branch": branch_for_call(call)})
        (call_dir / "timestep.json").write_text(json.dumps(call_metadata, indent=2) + "\n")
    manifest = {
        "format_version": 1,
        "config": "config.json",
        "capture_count": len(records),
        "expected_capture_count": expected,
        "status": "paused" if paused else "complete",
        "calls": list(calls.values()),
        "captures": records,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    state = {
        "status": "paused" if paused else "complete",
        "completed": len(records), "expected": expected,
        "remaining": expected - len(records),
        "resume": "rerun the same command; completed artifacts are skipped",
    }
    (args.output_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({**state, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
