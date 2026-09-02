"""Stream complete Wan activation matrices to disk in their source dtype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from .wan_utils import WAN_LINEAR_TRANSFORM_GROUPS


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
    ) -> None:
        self.model = model
        self.output_dir = output_dir
        self.quota_dir = quota_dir
        self.max_output_bytes = max_output_bytes
        self.blocks = list(blocks)
        self.sites = list(sites)
        self.target_calls = set(call_indices)
        self.call_index = -1
        self.timestep = float("nan")
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
                        self._save(inputs[0], current_block, current_site, current_linear)

                self.handles.append(module.register_forward_pre_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
