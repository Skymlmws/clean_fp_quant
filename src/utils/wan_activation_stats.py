"""Streaming W16A16 activation statistics for Wan Transformer linears."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .wan_utils import WAN_LINEAR_TRANSFORM_GROUPS


@dataclass
class ActivationSiteStats:
    channel_max: torch.Tensor | None = None
    channel_sumsq: torch.Tensor | None = None
    rows: int = 0
    calls: list[dict[str, float | int]] = field(default_factory=list)

    @torch.no_grad()
    def update(
        self,
        value: torch.Tensor,
        call_index: int,
        timestep: float,
        sample_elements: int,
        group_size: int,
    ) -> None:
        x = value.detach().reshape(-1, value.shape[-1]).float()
        row_count, channels = x.shape
        abs_x = x.abs()
        current_max = abs_x.amax(dim=0).cpu()
        current_sumsq = x.square().sum(dim=0).double().cpu()
        self.channel_max = current_max if self.channel_max is None else torch.maximum(self.channel_max, current_max)
        self.channel_sumsq = current_sumsq if self.channel_sumsq is None else self.channel_sumsq + current_sumsq
        self.rows += row_count

        # Quantiles over a deterministic, evenly spaced sample keep profiling bounded.
        flat = abs_x.flatten()
        if flat.numel() > sample_elements:
            index = torch.linspace(0, flat.numel() - 1, sample_elements, device=flat.device).long()
            flat = flat[index]
        quantiles = torch.quantile(flat, flat.new_tensor([0.5, 0.99, 0.999]))
        rms = x.square().mean().sqrt()

        usable = channels - channels % group_size
        if usable:
            blocks = x[:, :usable].reshape(-1, group_size)
            if blocks.shape[0] > max(1, sample_elements // group_size):
                index = torch.linspace(
                    0, blocks.shape[0] - 1, max(1, sample_elements // group_size), device=x.device
                ).long()
                blocks = blocks[index]
            block_rms = blocks.square().mean(dim=1).sqrt().clamp_min(1e-12)
            block_ratio = blocks.abs().amax(dim=1) / block_rms
            block_ratio_mean = block_ratio.mean()
            block_ratio_p99 = torch.quantile(block_ratio, 0.99)
        else:
            block_ratio_mean = x.new_tensor(float("nan"))
            block_ratio_p99 = x.new_tensor(float("nan"))

        self.calls.append({
            "call": call_index,
            "timestep": timestep,
            "rows": row_count,
            "absmax": abs_x.max().item(),
            "rms": rms.item(),
            "max_over_rms": (abs_x.max() / rms.clamp_min(1e-12)).item(),
            "p50": quantiles[0].item(),
            "p99": quantiles[1].item(),
            "p999": quantiles[2].item(),
            "max_over_p999": (abs_x.max() / quantiles[2].clamp_min(1e-12)).item(),
            "block_max_over_rms_mean": block_ratio_mean.item(),
            "block_max_over_rms_p99": block_ratio_p99.item(),
        })

    def export(self) -> dict[str, Any]:
        assert self.channel_max is not None and self.channel_sumsq is not None
        channel_rms = (self.channel_sumsq / self.rows).sqrt().float()
        return {
            "rows": self.rows,
            "channel_max": self.channel_max,
            "channel_rms": channel_rms,
            "calls": self.calls,
        }


class WanActivationProfiler:
    """Hooks the seven shared Wan linear input sites without modifying activations."""

    def __init__(self, model: nn.Module, sample_elements: int = 65536, group_size: int = 32):
        self.model = model
        self.sample_elements = sample_elements
        self.group_size = group_size
        self.stats: dict[str, ActivationSiteStats] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.call_index = -1
        self.timestep = float("nan")

    def _model_pre_hook(
        self, _module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self.call_index += 1
        timestep = inputs[1] if len(inputs) > 1 else kwargs.get("t")
        if isinstance(timestep, torch.Tensor) and timestep.numel():
            self.timestep = timestep.detach().float().flatten()[0].item()

    def attach(self) -> None:
        if self.handles:
            raise RuntimeError("Profiler is already attached")
        self.handles.append(
            self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
        for block_idx, block in enumerate(self.model.blocks):
            modules = dict(block.named_modules())
            for group_name, linear_names in WAN_LINEAR_TRANSFORM_GROUPS.items():
                # q/k/v inputs are identical within each shared transform group.
                linear_name = linear_names[0]
                module = modules.get(linear_name)
                if not isinstance(module, nn.Linear):
                    raise ValueError(f"Expected Linear at blocks.{block_idx}.{linear_name}")
                key = f"blocks.{block_idx}.{group_name}"
                self.stats[key] = ActivationSiteStats()

                def hook(_module, inputs, current_key=key):
                    self.stats[current_key].update(
                        inputs[0], self.call_index, self.timestep,
                        self.sample_elements, self.group_size,
                    )

                self.handles.append(module.register_forward_pre_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def export(self) -> dict[str, Any]:
        return {
            "group_size": self.group_size,
            "sample_elements": self.sample_elements,
            "model_calls": self.call_index + 1,
            "sites": {key: value.export() for key, value in self.stats.items()},
        }
