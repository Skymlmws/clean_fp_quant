"""Stream complete Wan activation matrices to disk in their source dtype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from .sites import WAN_LINEAR_SITES


class QuotaExceeded(RuntimeError):
    pass


def branch_for_call(call: int) -> str:
    return "conditional" if call % 2 == 0 else "unconditional"


def artifact_dir(root: Path, call: int, block: int, site: str) -> Path:
    return root / f"step_{call // 2:03d}" / branch_for_call(call) / f"block_{block:02d}" / site


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class WanActivationDiskCapture:
    """Synchronously save selected complete Linear inputs at forward-hook time."""

    def __init__(
        self,
        model: nn.Module,
        output_dir: Path,
        quota_dir: Path,
        max_output_bytes: int,
        blocks: Iterable[int],
        sites: Iterable[str],
        call_indices: Iterable[int],
        batch_index: int = 0,
    ) -> None:
        self.model = model
        self.output_dir = output_dir
        self.quota_dir = quota_dir
        self.max_output_bytes = max_output_bytes
        self.blocks = list(blocks)
        self.sites = list(sites)
        self.target_calls = set(call_indices)
        self.batch_index = batch_index
        self.call_index = -1
        self.timestep = float("nan")
        self.effective_text_token_count: int | None = None
        self.text_token_mask: list[bool] | None = None
        self.text_context_by_call: dict[int, dict[str, Any]] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.completed = 0
        self.skipped = 0
        self.bytes_at_start = directory_size(quota_dir)
        self.bytes_written = 0

    def _model_pre_hook(
        self, _module: nn.Module, inputs: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self.call_index += 1
        timestep = inputs[1] if len(inputs) > 1 else kwargs.get("t")
        if isinstance(timestep, torch.Tensor) and timestep.numel():
            self.timestep = timestep.detach().float().flatten()[0].item()
        context = inputs[2] if len(inputs) > 2 else kwargs.get("context")
        if isinstance(context, (list, tuple)) and context:
            if not 0 <= self.batch_index < len(context):
                raise IndexError(
                    f"batch-index {self.batch_index} is outside text context batch "
                    f"[0, {len(context)})"
                )
            selected_context = context[self.batch_index]
            if isinstance(selected_context, torch.Tensor) and selected_context.ndim >= 1:
                effective_count = int(selected_context.shape[0])
                padded_count = int(getattr(self.model, "text_len", effective_count))
                if effective_count > padded_count:
                    raise ValueError(
                        f"Effective text length {effective_count} exceeds padded length "
                        f"{padded_count}"
                    )
                self.effective_text_token_count = effective_count
                self.text_token_mask = [True] * effective_count + [False] * (
                    padded_count - effective_count
                )
                self.text_context_by_call[self.call_index] = {
                    "effective_token_count": effective_count,
                    "padded_token_count": padded_count,
                    "token_mask": self.text_token_mask,
                    "mask_semantics": "true marks an effective UMT5 token; false marks right padding",
                }

    def _save(self, value: torch.Tensor, block: int, site: str, linear: str) -> None:
        destination = artifact_dir(self.output_dir, self.call_index, block, site)
        activation_path = destination / "activation.pt"
        metadata_path = destination / "metadata.json"
        if activation_path.exists() and metadata_path.exists():
            self.skipped += 1
            return
        required_bytes = value.numel() * torch.tensor([], dtype=torch.bfloat16).element_size()
        projected = self.bytes_at_start + self.bytes_written + required_bytes
        if self.max_output_bytes and projected > self.max_output_bytes:
            raise QuotaExceeded(
                f"Writing blocks.{block}.{site} would exceed quota: "
                f"{projected / 1024**3:.2f} GiB > {self.max_output_bytes / 1024**3:.2f} GiB"
            )
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / "activation.pt.partial"
        if temporary.exists():
            temporary.unlink()
        source_dtype = value.dtype
        cpu_value = value.detach().to(device="cpu", dtype=torch.bfloat16)
        torch.save(cpu_value, temporary)
        temporary.replace(activation_path)
        actual_bytes = activation_path.stat().st_size
        metadata = {
            "sampling_step": self.call_index // 2,
            "timestep": self.timestep,
            "branch": branch_for_call(self.call_index),
            "call_index": self.call_index,
            "block": block,
            "site": site,
            "linear": linear,
            "shape": list(cpu_value.shape),
            "source_dtype": str(source_dtype),
            "stored_dtype": str(cpu_value.dtype),
            "numel": cpu_value.numel(),
            "element_size": cpu_value.element_size(),
            "bytes": actual_bytes,
            "complete": True,
            "files": {"activation": "activation.pt"},
        }
        if self.effective_text_token_count is not None and self.text_token_mask is not None:
            metadata["text_context"] = {
                "effective_token_count": self.effective_text_token_count,
                "padded_token_count": len(self.text_token_mask),
                "token_mask": self.text_token_mask,
                "mask_semantics": (
                    "true marks an effective UMT5 token; false marks right padding"
                ),
            }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        self.bytes_written += actual_bytes
        self.completed += 1

    def attach(self) -> None:
        if self.handles:
            raise RuntimeError("Capture is already attached")
        self.handles.append(self.model.register_forward_pre_hook(self._model_pre_hook, with_kwargs=True))
        for block_index in self.blocks:
            modules = dict(self.model.blocks[block_index].named_modules())
            for site in self.sites:
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
                    if self.call_index in self.target_calls:
                        self._save(inputs[0], current_block, current_site, current_linear)

                self.handles.append(module.register_forward_pre_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
