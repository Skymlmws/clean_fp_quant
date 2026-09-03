"""Generate a Wan video and render activation heatmaps without storing activations."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
from typing import Any
import uuid

import numpy as np
import torch
import torch.nn as nn

from scripts.visualize.render_wan_self_qkv_frames import (
    activation_statistics,
    append_channel_outliers,
    channel_outliers,
    frame_view,
    statistics_annotation,
    video_token_grid,
)
from src.utils.wan_activation_disk_capture import branch_for_call
from src.utils.wan_activation_outliers import isolated_token_outliers
from src.utils.wan_activation_surface import parse_indices
from src.utils.wan_utils import WAN_LINEAR_TRANSFORM_GROUPS
from scripts.visualize.visualize_wan_activation_surfaces import render_heatmap, selected_sites


DEFAULT_PROMPT = (
    "A cinematic wide shot follows a young woman in a weathered yellow raincoat walking "
    "through a crowded coastal market at dawn. Fishing boats rock beside the wooden pier "
    "while vendors arrange silver fish, red crabs, green vegetables, woven baskets, glass "
    "bottles, and bright fabric beneath striped canvas awnings. She carries a small blue "
    "suitcase and moves steadily from the foreground toward the center of the market as "
    "people cross naturally in both directions. A brown dog runs past her, several white "
    "gulls circle overhead, and loose paper flutters across the wet stone pavement in the "
    "sea breeze. The camera begins with a slow forward tracking movement at waist height, "
    "then gently rises and pans left to reveal the harbor, distant mountains, and a "
    "lighthouse partially hidden by morning mist. Warm sunlight breaks through gray clouds, "
    "creating long reflections in puddles and soft volumetric rays between the stalls. "
    "Faces, clothing, signs, ropes, nets, and waves remain detailed and temporally "
    "consistent. Natural motion, realistic physics, subtle depth of field, cinematic "
    "composition, rich but balanced colors, no cuts."
)

ONLINE_RENDER_SCHEMA_VERSION = 2


class OutputLimitExceeded(RuntimeError):
    pass


@dataclass
class RenderJob:
    activation: torch.Tensor | None
    source_shape: list[int]
    call_index: int
    timestep: float
    text_context: dict[str, Any] | None
    block: int
    site: str
    linear: str
    shared_memory_path: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("/home/maoliming/project/checkpoints/Wan2.1-T2V-1.3B"))
    parser.add_argument("--wan-repo", type=Path, default=Path("/home/maoliming/project/wan2.1"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prompt-id")
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
    parser.add_argument("--image-width", type=int, default=1800)
    parser.add_argument("--image-height", type=int, default=1200)
    parser.add_argument("--heatmap-percentile", type=float, default=100.0)
    parser.add_argument("--heatmap-gamma", type=float, default=1.0)
    parser.add_argument("--channel-rms-ratio", type=float, default=5.0)
    parser.add_argument("--mark-top-channels", type=int, default=8)
    parser.add_argument("--isolated-global-percentile", type=float, default=99.99)
    parser.add_argument("--isolated-channel-percentile", type=float, default=99.0)
    parser.add_argument("--isolated-ratio", type=float, default=5.0)
    parser.add_argument("--isolated-max-token-fraction", type=float, default=0.01)
    parser.add_argument("--mark-top-isolated", type=int, default=10)
    parser.add_argument("--isolated-merge-token-gap", type=int, default=1)
    parser.add_argument("--ffn-out-group-size", type=int, default=8)
    parser.add_argument(
        "--max-output-gb", type=float, default=0.0,
        help="Maximum completed output size in GiB; 0 disables the limit",
    )
    parser.add_argument(
        "--render-mode", choices=("multiprocess", "async", "sync"), default="multiprocess",
        help="multiprocess uses independent Matplotlib workers; async uses one thread",
    )
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--shared-memory-dir", type=Path, default=Path("/dev/shm"))
    parser.add_argument(
        "--max-inflight-activations", type=int, default=0,
        help="Maximum CPU activations including the one rendering; 0 sizes from available RAM",
    )
    parser.add_argument(
        "--inflight-memory-fraction", type=float, default=0.25,
        help="Fraction of currently available RAM usable by auto-sized activation buffering",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/activation-visualization/wan-activation-long-prompts/wan-activation-long-prompt-online-heatmaps"),
    )
    return parser.parse_args()


def resolve_prompt(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.prompt_file is None:
        if args.prompt_id is not None:
            raise ValueError("--prompt-id requires --prompt-file")
        return None
    collection = json.loads(args.prompt_file.read_text())
    entries = collection.get("prompts", [])
    if not entries:
        raise ValueError(f"No prompts found in {args.prompt_file}")
    if args.prompt_id is None:
        if len(entries) != 1:
            choices = [entry.get("id") for entry in entries]
            raise ValueError(f"--prompt-id is required; choices are {choices}")
        selected = entries[0]
    else:
        matches = [entry for entry in entries if entry.get("id") == args.prompt_id]
        if len(matches) != 1:
            choices = [entry.get("id") for entry in entries]
            raise ValueError(f"Unknown prompt id {args.prompt_id!r}; choices are {choices}")
        selected = matches[0]
    args.prompt = selected["prompt"]
    return {
        "file": str(args.prompt_file),
        "id": selected.get("id"),
        "title": selected.get("title"),
        "output_slug": selected.get("output_slug"),
        "word_count": selected.get("word_count"),
        "umt5_token_count": selected.get("umt5_token_count"),
    }


def available_memory_bytes() -> int:
    fields = {}
    with Path("/proc/meminfo").open() as handle:
        for line in handle:
            key, value = line.split(":", 1)
            fields[key] = int(value.strip().split()[0]) * 1024
    if "MemAvailable" in fields:
        return fields["MemAvailable"]
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


class WanOnlineActivationRenderer:
    """Render selected Linear inputs without storing complete activations."""

    def __init__(self, model: nn.Module, args: argparse.Namespace, blocks: list[int], sites: list[str], calls: list[int]) -> None:
        self.model = model
        self.args = args
        self.blocks = blocks
        self.sites = sites
        self.target_calls = set(calls)
        self.call_index = -1
        self.timestep = float("nan")
        self.text_context_by_call: dict[int, dict[str, Any]] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.rendered_activations = 0
        self.skipped_activations = 0
        self.rendered_images = 0
        self.bytes_at_start = self._directory_size()
        self.bytes_written = 0
        self.max_output_bytes = int(args.max_output_gb * 1024**3) if args.max_output_gb else 0
        self.grid = video_token_grid({"size": [args.width, args.height], "frames": args.frames})
        self.render_mode = args.render_mode
        if args.render_workers <= 0:
            raise ValueError("render-workers must be positive")
        expected_activations = len(blocks) * len(sites) * len(calls)
        if args.max_inflight_activations < 0:
            raise ValueError("max-inflight-activations must be non-negative")
        if not 0 < args.inflight_memory_fraction <= 1:
            raise ValueError("inflight-memory-fraction must be in (0, 1]")
        if self.render_mode == "sync":
            self.max_inflight_activations = 1
        elif args.max_inflight_activations:
            self.max_inflight_activations = args.max_inflight_activations
        else:
            video_tokens = self.grid[0] * self.grid[1] * self.grid[2]
            modules = dict(model.named_modules())
            maximum_channels = max(
                modules[f"blocks.{block}.{WAN_LINEAR_TRANSFORM_GROUPS[site][0]}"].in_features
                for block in blocks for site in sites
            )
            worst_activation_bytes = video_tokens * maximum_channels * 2
            memory_budget = int(available_memory_bytes() * args.inflight_memory_fraction)
            memory_limited_count = max(2, memory_budget // worst_activation_bytes)
            self.max_inflight_activations = min(expected_activations, memory_limited_count)
        queue_capacity = max(1, self.max_inflight_activations - 1)
        self._queue: queue.Queue[RenderJob | None] = queue.Queue(maxsize=queue_capacity)
        self._inflight = threading.Semaphore(self.max_inflight_activations)
        self._worker_error: BaseException | None = None
        self._worker_error_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._futures: set[Future] = set()
        self._executor: ProcessPoolExecutor | None = None
        self._worker: threading.Thread | None = None
        if self.render_mode == "async":
            self._worker = threading.Thread(
                target=self._worker_loop, name="wan-activation-renderer", daemon=False
            )
            self._worker.start()
        elif self.render_mode == "multiprocess":
            self._executor = ProcessPoolExecutor(
                max_workers=args.render_workers,
                mp_context=mp.get_context("spawn"),
            )

    def _directory_size(self) -> int:
        return sum(path.stat().st_size for path in self.args.output_dir.rglob("*") if path.is_file())

    def _check_limit(self) -> None:
        if self.max_output_bytes and self.bytes_at_start + self.bytes_written > self.max_output_bytes:
            raise OutputLimitExceeded(
                f"Online render output exceeded {self.args.max_output_gb:.2f} GiB"
            )

    def _model_pre_hook(self, _module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.call_index += 1
        timestep = inputs[1] if len(inputs) > 1 else kwargs.get("t")
        if isinstance(timestep, torch.Tensor) and timestep.numel():
            self.timestep = timestep.detach().float().flatten()[0].item()
        context = inputs[2] if len(inputs) > 2 else kwargs.get("context")
        if isinstance(context, (list, tuple)) and context and isinstance(context[0], torch.Tensor):
            effective = int(context[0].shape[0])
            padded = int(getattr(self.model, "text_len", effective))
            self.text_context_by_call[self.call_index] = {
                "effective_token_count": effective,
                "padded_token_count": padded,
                "token_mask": [True] * effective + [False] * (padded - effective),
            }

    def _destination(self, call_index: int, block: int, site: str) -> Path:
        return (
            self.args.output_dir / f"step_{call_index // 2:03d}"
            / branch_for_call(call_index) / f"block_{block:02d}" / site
        )

    def _is_complete(self, destination: Path) -> bool:
        metadata_path = destination / "metadata.json"
        if not metadata_path.exists():
            return False
        metadata = json.loads(metadata_path.read_text())
        return (
            int(metadata.get("render_schema_version", 1)) >= ONLINE_RENDER_SCHEMA_VERSION
            and bool(metadata.get("complete"))
            and all(
            (destination / relative).exists() for relative in metadata.get("image_files", [])
            )
        )

    def _set_worker_error(self, error: BaseException) -> None:
        with self._worker_error_lock:
            if self._worker_error is None:
                self._worker_error = error

    def _raise_worker_error(self) -> None:
        with self._worker_error_lock:
            error = self._worker_error
        if error is not None:
            raise RuntimeError("Online activation renderer failed") from error

    def _worker_loop(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                if self._worker_error is None:
                    self._render_job(job)
            except BaseException as error:
                self._set_worker_error(error)
            finally:
                if job is not None:
                    self._inflight.release()
                self._queue.task_done()

    def _submit(self, value: torch.Tensor, block: int, site: str, linear: str) -> None:
        call_index = self.call_index
        destination = self._destination(call_index, block, site)
        if self._is_complete(destination):
            self.skipped_activations += 1
            return
        self._raise_worker_error()
        if self.render_mode == "multiprocess":
            self._inflight.acquire()
            shared_memory_path: Path | None = None
            try:
                self._raise_worker_error()
                self.args.shared_memory_dir.mkdir(parents=True, exist_ok=True)
                shared_memory_path = self.args.shared_memory_dir / (
                    f"fp-quant-wan-{os.getpid()}-{uuid.uuid4().hex}.bf16"
                )
                activation = torch.from_file(
                    str(shared_memory_path),
                    shared=True,
                    size=value.numel(),
                    dtype=torch.bfloat16,
                ).reshape(tuple(value.shape))
                activation.copy_(value.detach())
                job = RenderJob(
                    activation=None,
                    source_shape=list(value.shape),
                    call_index=call_index,
                    timestep=self.timestep,
                    text_context=self.text_context_by_call.get(call_index),
                    block=block,
                    site=site,
                    linear=linear,
                    shared_memory_path=str(shared_memory_path),
                )
                assert self._executor is not None
                future = self._executor.submit(render_job_in_process, job, self.args, self.grid)
                with self._counter_lock:
                    self._futures.add(future)
                future.add_done_callback(
                    lambda completed, path=shared_memory_path: self._process_done(completed, path)
                )
            except BaseException:
                if shared_memory_path is not None:
                    shared_memory_path.unlink(missing_ok=True)
                self._inflight.release()
                raise
        elif self.render_mode == "async":
            self._inflight.acquire()
            try:
                self._raise_worker_error()
                activation = value.detach().to(device="cpu", dtype=torch.bfloat16)
                job = RenderJob(
                    activation=activation,
                    source_shape=list(value.shape),
                    call_index=call_index,
                    timestep=self.timestep,
                    text_context=self.text_context_by_call.get(call_index),
                    block=block,
                    site=site,
                    linear=linear,
                )
                self._queue.put(job)
            except BaseException:
                self._inflight.release()
                raise
        else:
            self._render_job(RenderJob(
                activation=value.detach().to(device="cpu", dtype=torch.bfloat16),
                source_shape=list(value.shape),
                call_index=call_index,
                timestep=self.timestep,
                text_context=self.text_context_by_call.get(call_index),
                block=block,
                site=site,
                linear=linear,
            ))

    def _process_done(self, future: Future, shared_memory_path: Path | None = None) -> None:
        try:
            result = future.result()
            with self._counter_lock:
                self.rendered_activations += int(result["rendered_activations"])
                self.rendered_images += int(result["rendered_images"])
                self.bytes_written += int(result["bytes_written"])
        except BaseException as error:
            self._set_worker_error(error)
        finally:
            if shared_memory_path is not None:
                shared_memory_path.unlink(missing_ok=True)
            with self._counter_lock:
                self._futures.discard(future)
            self._inflight.release()

    def _render_job(self, job: RenderJob) -> None:
        destination = self._destination(job.call_index, job.block, job.site)
        self._check_limit()
        destination.mkdir(parents=True, exist_ok=True)
        if job.activation is None:
            raise ValueError("Render job activation was not attached")
        activation = job.activation
        activation = activation.reshape(-1, activation.shape[-1])
        records: list[dict[str, Any]] = []
        image_files: list[str] = []

        if job.site == "cross_kv":
            views = [(None, activation.float().numpy())]
        else:
            views = (
                (frame, frame_view(activation, frame, self.grid).float().numpy())
                for frame in range(self.grid[0])
            )

        for frame, values in views:
            original_channel_count = values.shape[1]
            channels = np.arange(original_channel_count, dtype=np.int64)
            detection_values = values
            if job.site == "cross_kv" and job.text_context is not None:
                effective_tokens = int(job.text_context.get("effective_token_count", values.shape[0]))
                detection_values = values[:effective_tokens]
            outliers = channel_outliers(
                detection_values, channels,
                self.args.channel_rms_ratio, self.args.mark_top_channels,
            )
            isolated = isolated_token_outliers(
                detection_values,
                {int(record["channel"]) for record in outliers},
                self.args.isolated_global_percentile,
                self.args.isolated_channel_percentile,
                self.args.isolated_ratio,
                self.args.isolated_max_token_fraction,
                self.args.mark_top_isolated,
                self.args.isolated_merge_token_gap,
            )
            display_values = values
            marker_labels: dict[int, str] | None = None
            display_channels = channels
            aggregation: dict[str, Any] | None = None
            if job.site == "ffn_out":
                group = self.args.ffn_out_group_size
                if group <= 0 or original_channel_count % group:
                    raise ValueError("ffn-out-group-size must positively divide ffn_out channels")
                display_values = np.abs(values).reshape(values.shape[0], -1, group).max(axis=2)
                display_channels = np.arange(display_values.shape[1], dtype=np.int64)
                marker_labels = {}
                for record in outliers:
                    position = int(record["channel"]) // group
                    label = f"ch {record['channel']}"
                    marker_labels[position] = (
                        f"{marker_labels[position]}, {label}" if position in marker_labels else label
                    )
                aggregation = {
                    "kind": "max_abs_channel_groups",
                    "channels_per_group": group,
                    "source_channel_count": original_channel_count,
                }
            marked_points = [
                {
                    "token": int(record["peak_token"]),
                    "channel": (
                        int(record["channel"]) // self.args.ffn_out_group_size
                        if job.site == "ffn_out" else int(record["channel"])
                    ),
                    "label": str(index),
                    "source_channel": int(record["channel"]),
                }
                for index, record in enumerate(isolated, 1)
            ]
            isolated_lines = [
                f"{index}: tok {record['token_start']}-{record['token_end']}, "
                f"ch {record['channel']}, peak/baseline "
                f"{record['peak_over_channel_baseline']:.2f}"
                for index, record in enumerate(isolated, 1)
            ]
            stats = activation_statistics(values)
            annotation = append_channel_outliers(statistics_annotation(stats), outliers)
            annotation += "\nisolated token outliers:"
            annotation += "\n" + ("\n".join(isolated_lines) if isolated_lines else "none")
            tokens = np.arange(values.shape[0], dtype=np.int64)
            filename = "heatmap.png" if frame is None else f"frame_{frame:03d}.png"
            image_path = destination / filename
            temporary = destination / f"{filename}.partial.png"
            frame_text = "text tokens" if frame is None else f"latent frame {frame}/{self.grid[0] - 1}"
            render = render_heatmap(
                display_values,
                tokens,
                display_channels,
                f"Wan {job.site} | step {job.call_index // 2} | "
                f"block {job.block} | {frame_text}",
                temporary,
                self.args.image_width,
                self.args.image_height,
                self.args.heatmap_percentile,
                self.args.heatmap_gamma,
                use_bin_edges=True,
                annotation_text=annotation,
                marked_channels=[int(record["channel"]) for record in outliers] if marker_labels is None else None,
                channel_marker_labels=marker_labels,
                marked_points=marked_points,
            )
            temporary.replace(image_path)
            self.bytes_written += image_path.stat().st_size
            self.rendered_images += 1
            image_files.append(filename)
            records.append({
                "latent_frame": frame,
                "matrix_shape": list(values.shape),
                "display_shape": list(display_values.shape),
                "statistics": stats,
                "channel_outliers": outliers,
                "isolated_outliers": isolated,
                "aggregation": aggregation,
                "file": filename,
                "render": render,
            })
            self._check_limit()
        del activation

        metadata = {
            "render_schema_version": ONLINE_RENDER_SCHEMA_VERSION,
            "sampling_step": job.call_index // 2,
            "timestep": job.timestep,
            "branch": branch_for_call(job.call_index),
            "call_index": job.call_index,
            "block": job.block,
            "site": job.site,
            "linear": job.linear,
            "source_shape": job.source_shape,
            "text_context": job.text_context,
            "activation_stored": False,
            "online_render": True,
            "records": records,
            "image_files": image_files,
            "complete": True,
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.rendered_activations += 1

    def attach(self) -> None:
        self.handles.append(self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        for block_index in self.blocks:
            modules = dict(self.model.blocks[block_index].named_modules())
            for site in self.sites:
                linear_name = WAN_LINEAR_TRANSFORM_GROUPS[site][0]
                module = modules.get(linear_name)
                if not isinstance(module, nn.Linear):
                    raise ValueError(f"Expected Linear at blocks.{block_index}.{linear_name}")

                def hook(
                    _module: nn.Module,
                    inputs: tuple[Any, ...],
                    current_block: int = block_index,
                    current_site: str = site,
                    current_linear: str = linear_name,
                ) -> None:
                    if self.call_index in self.target_calls:
                        self._submit(inputs[0], current_block, current_site, current_linear)

                self.handles.append(module.register_forward_pre_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def finish(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join()
            self._worker = None
        self._raise_worker_error()


def render_job_in_process(
    job: RenderJob, args: argparse.Namespace, grid: tuple[int, int, int]
) -> dict[str, int]:
    """Render one shared-memory activation inside an isolated Matplotlib process."""
    renderer = WanOnlineActivationRenderer.__new__(WanOnlineActivationRenderer)
    renderer.args = args
    renderer.grid = grid
    renderer.max_output_bytes = 0
    renderer.bytes_at_start = 0
    renderer.bytes_written = 0
    renderer.rendered_images = 0
    renderer.rendered_activations = 0
    if job.shared_memory_path is not None:
        job.activation = torch.from_file(
            job.shared_memory_path,
            shared=True,
            size=int(np.prod(job.source_shape)),
            dtype=torch.bfloat16,
        ).reshape(job.source_shape)
    renderer._render_job(job)
    return {
        "rendered_activations": renderer.rendered_activations,
        "rendered_images": renderer.rendered_images,
        "bytes_written": renderer.bytes_written,
    }


def main() -> None:
    args = parse_args()
    prompt_source = resolve_prompt(args)
    if args.frames % 4 != 1 or args.width % 16 or args.height % 16:
        raise ValueError("frames must be 4n+1 and width/height divisible by 16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampling_steps = parse_indices(args.sampling_steps, args.steps)
    call_indices = [step * 2 for step in sampling_steps]
    sys.path.insert(0, str(args.wan_repo))
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    from wan.utils.utils import cache_video

    pipe = WanT2V(config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=str(args.checkpoint), device_id=args.device_id, t5_cpu=True)
    blocks = parse_indices(args.blocks, len(pipe.model.blocks))
    sites = selected_sites(args.sites)
    renderer = WanOnlineActivationRenderer(pipe.model, args, blocks, sites, call_indices)
    config = {
        "render_schema_version": ONLINE_RENDER_SCHEMA_VERSION,
        "mode": "online activation heatmaps; complete activations are not stored",
        "prompt": args.prompt,
        "prompt_source": prompt_source,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "size": [args.width, args.height],
        "frames": args.frames,
        "fps": args.fps,
        "steps": args.steps,
        "sampling_steps": sampling_steps,
        "call_indices": call_indices,
        "blocks": blocks,
        "sites": sites,
        "render_mode": args.render_mode,
        "async_queue_capacity": (
            renderer.max_inflight_activations - 1
            if args.render_mode in ("multiprocess", "async") else 0
        ),
        "maximum_inflight_activations": renderer.max_inflight_activations,
        "render_workers": args.render_workers,
        "shared_memory_dir": str(args.shared_memory_dir),
        "inflight_memory_fraction": args.inflight_memory_fraction,
        "ffn_out_group_size": args.ffn_out_group_size,
        "isolated_outliers": {
            "global_percentile": args.isolated_global_percentile,
            "channel_percentile": args.isolated_channel_percentile,
            "minimum_channel_ratio": args.isolated_ratio,
            "maximum_token_fraction": args.isolated_max_token_fraction,
            "maximum_entities": args.mark_top_isolated,
            "merge_token_gap": args.isolated_merge_token_gap,
            "cross_kv_ignores_padding": True,
        },
        "max_output_gb": args.max_output_gb,
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    renderer.attach()
    caught_error: BaseException | None = None
    video = None
    try:
        with torch.inference_mode():
            video = pipe.generate(
                input_prompt=args.prompt,
                size=(args.width, args.height),
                frame_num=args.frames,
                shift=args.shift,
                sample_solver="unipc",
                sampling_steps=args.steps,
                guide_scale=args.guide_scale,
                n_prompt=args.negative_prompt,
                seed=args.seed,
                offload_model=False,
            )
    except BaseException as caught:
        caught_error = caught
    finally:
        renderer.remove()
    if caught_error is None and video is not None:
        try:
            cache_video(video[None], save_file=str(args.output_dir / "video.mp4"), fps=args.fps)
        except BaseException as caught:
            caught_error = caught
    del video
    renderer.model = None
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        renderer.finish()
    except BaseException as caught:
        if caught_error is None:
            caught_error = caught

    if isinstance(caught_error, OutputLimitExceeded) or isinstance(
        getattr(caught_error, "__cause__", None), OutputLimitExceeded
    ):
        status = "paused_output_limit"
    elif caught_error is not None:
        status = "interrupted"
    else:
        status = "complete"
    error = repr(caught_error) if caught_error is not None else None
    try:
        config["text_context_by_call"] = {
            str(call): value for call, value in sorted(renderer.text_context_by_call.items())
        }
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        state = {
            "status": status,
            "rendered_activations": renderer.rendered_activations,
            "skipped_activations": renderer.skipped_activations,
            "rendered_images": renderer.rendered_images,
            "expected_activations": len(blocks) * len(sites) * len(call_indices),
            "error": error,
        }
        (args.output_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    finally:
        if caught_error is not None:
            raise caught_error
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
