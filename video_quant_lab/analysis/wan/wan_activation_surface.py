"""Capture token-by-channel activation matrices from Wan Transformer linears."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn

from .sites import WAN_LINEAR_SITES


def parse_indices(spec: str, size: int) -> list[int]:
    """Parse ``all``, comma-separated indices, and inclusive ranges."""
    if spec.strip().lower() == "all":
        return list(range(size))
    result: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            values: Iterable[int] = range(int(start_text), int(end_text) + 1)
        else:
            values = (int(part),)
        for value in values:
            if not 0 <= value < size:
                raise ValueError(f"Index {value} is outside [0, {size})")
            if value not in result:
                result.append(value)
    if not result:
        raise ValueError("No indices were selected")
    return result


def activation_matrix(value: torch.Tensor, batch_index: int) -> torch.Tensor:
    """Return a CPU float32 [token, channel] matrix from a linear input."""
    if value.ndim < 2:
        raise ValueError(f"Expected a tensor with at least 2 dimensions, got {tuple(value.shape)}")
    selected = value.detach()
    if selected.ndim > 2:
        if not 0 <= batch_index < selected.shape[0]:
            raise ValueError(
                f"batch-index {batch_index} is outside [0, {selected.shape[0]})"
            )
        selected = selected[batch_index]
    return selected.reshape(-1, selected.shape[-1]).float().cpu()


def sampled_activation_matrix(
    value: torch.Tensor,
    batch_index: int,
    max_tokens: int,
    max_channels: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a linear input on-device before copying it to CPU."""
    if max_tokens < 0 or max_channels < 0:
        raise ValueError("Sampling limits must be non-negative; zero means unlimited")
    if value.ndim < 2:
        raise ValueError(f"Expected a tensor with at least 2 dimensions, got {tuple(value.shape)}")
    selected = value.detach()
    if selected.ndim > 2:
        if not 0 <= batch_index < selected.shape[0]:
            raise ValueError(
                f"batch-index {batch_index} is outside [0, {selected.shape[0]})"
            )
        selected = selected[batch_index]
    matrix = selected.reshape(-1, selected.shape[-1])
    token_count = matrix.shape[0] if max_tokens == 0 else min(matrix.shape[0], max_tokens)
    channel_count = matrix.shape[1] if max_channels == 0 else min(matrix.shape[1], max_channels)
    token_indices = torch.linspace(
        0, matrix.shape[0] - 1, token_count, device=matrix.device
    ).round().long()
    channel_indices = torch.linspace(
        0, matrix.shape[1] - 1, channel_count, device=matrix.device
    ).round().long()
    sampled = matrix.index_select(0, token_indices).index_select(1, channel_indices)
    return sampled.float().cpu(), token_indices.cpu(), channel_indices.cpu()


@dataclass
class CapturedActivation:
    block: int
    site: str
    linear: str
    call: int
    timestep: float
    original_shape: tuple[int, ...]
    matrix: torch.Tensor
    token_indices: torch.Tensor
    channel_indices: torch.Tensor


class WanActivationMatrixCapture:
    """Capture one model call at selected Wan block/site linear inputs."""

    def __init__(
        self,
        model: nn.Module,
        blocks: Iterable[int],
        sites: Iterable[str],
        call_index: int = 0,
        call_indices: Iterable[int] | None = None,
        batch_index: int = 0,
        max_tokens: int = 0,
        max_channels: int = 0,
    ) -> None:
        self.model = model
        self.blocks = list(blocks)
        self.sites = list(sites)
        self.target_calls = set(call_indices if call_indices is not None else (call_index,))
        self.batch_index = batch_index
        self.max_tokens = max_tokens
        self.max_channels = max_channels
        self.call_index = -1
        self.timestep = float("nan")
        self.captures: list[CapturedActivation] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def _model_pre_hook(
        self, _module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self.call_index += 1
        timestep = inputs[1] if len(inputs) > 1 else kwargs.get("t")
        if isinstance(timestep, torch.Tensor) and timestep.numel():
            self.timestep = timestep.detach().float().flatten()[0].item()

    def attach(self) -> None:
        if self.handles:
            raise RuntimeError("Capture is already attached")
        self.handles.append(
            self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True)
        )
        for block_index in self.blocks:
            block = self.model.blocks[block_index]
            modules = dict(block.named_modules())
            for site in self.sites:
                if site not in WAN_LINEAR_SITES:
                    raise ValueError(f"Unknown activation site: {site}")
                # Members of a shared group receive the same input; capture it once.
                linear_name = WAN_LINEAR_SITES[site][0]
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
                    if self.call_index not in self.target_calls:
                        return
                    value = inputs[0]
                    matrix, token_indices, channel_indices = sampled_activation_matrix(
                        value, self.batch_index, self.max_tokens, self.max_channels
                    )
                    self.captures.append(
                        CapturedActivation(
                            block=current_block,
                            site=current_site,
                            linear=current_linear,
                            call=self.call_index,
                            timestep=self.timestep,
                            original_shape=tuple(value.shape),
                            matrix=matrix,
                            token_indices=token_indices,
                            channel_indices=channel_indices,
                        )
                    )

                self.handles.append(module.register_forward_pre_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
