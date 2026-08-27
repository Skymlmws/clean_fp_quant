"""Wan2.1 DiT adapters for FP-Quant transforms and fake RTN quantization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantization.quantizer import Quantizer
from ..transforms.transforms import BaseTransform, GivensTransform, build_transform


WAN_LINEAR_TRANSFORM_GROUPS = {
    "self_qkv": ("self_attn.q", "self_attn.k", "self_attn.v"),
    "self_o": ("self_attn.o",),
    "cross_q": ("cross_attn.q",),
    "cross_kv": ("cross_attn.k", "cross_attn.v"),
    "cross_o": ("cross_attn.o",),
    "ffn_in": ("ffn.0",),
    "ffn_out": ("ffn.2",),
}


@dataclass
class WanBlockTransforms:
    transforms: dict[str, BaseTransform]
    linears: dict[str, BaseTransform] = field(init=False)

    def __post_init__(self) -> None:
        self.linears = {
            linear_name: self.transforms[group_name]
            for group_name, linear_names in WAN_LINEAR_TRANSFORM_GROUPS.items()
            for linear_name in linear_names
        }


@dataclass
class WanQuantizationReport:
    replaced: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    transform_stats: dict[str, int | float] = field(default_factory=dict)

    @property
    def replaced_count(self) -> int:
        return len(self.replaced)


class WanRTNLinear(nn.Linear):
    """Wan-compatible Linear with a fixed input transform and fake quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        transform: BaseTransform,
        weight_quantizer: Quantizer | None,
        activation_quantizer: Quantizer | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.input_transform = transform
        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        transform: BaseTransform,
        weight_quantizer_kwargs: dict[str, Any] | None,
        activation_quantizer_kwargs: dict[str, Any] | None,
    ) -> "WanRTNLinear":
        weight_quantizer = Quantizer(**weight_quantizer_kwargs) if weight_quantizer_kwargs else None
        activation_quantizer = (
            Quantizer(**activation_quantizer_kwargs) if activation_quantizer_kwargs else None
        )
        quantized = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            transform,
            weight_quantizer,
            activation_quantizer,
            linear.weight.device,
            linear.weight.dtype,
        )

        weight = transform(linear.weight, inv_t=True)
        if weight_quantizer is not None:
            scales, zeros = weight_quantizer.get_quantization_params(weight)
            weight = weight_quantizer(weight, scales, zeros)
        quantized.weight.copy_(weight)
        if linear.bias is not None:
            quantized.bias.copy_(linear.bias)
        quantized.requires_grad_(False)
        quantized.train(linear.training)
        return quantized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_transform(x)
        if self.activation_quantizer is not None:
            scales, zeros = self.activation_quantizer.get_quantization_params(x)
            x = self.activation_quantizer(x, scales, zeros)
        return F.linear(x, self.weight, self.bias)


def build_wan_block_transforms(
    model: nn.Module,
    transform_class: str,
    group_size: int,
    device: torch.device,
    **transform_kwargs: Any,
) -> list[WanBlockTransforms]:
    if not hasattr(model, "blocks") or not hasattr(model, "dim") or not hasattr(model, "ffn_dim"):
        raise ValueError("Expected a WanModel-like module with blocks, dim, and ffn_dim")

    result = []
    for _ in model.blocks:
        transforms = {}
        for name in WAN_LINEAR_TRANSFORM_GROUPS:
            size = model.ffn_dim if name == "ffn_out" else model.dim
            transforms[name] = build_transform(
                transform_class,
                size=size,
                group_size=group_size,
                device=device,
                **transform_kwargs,
            )
        result.append(WanBlockTransforms(transforms))
    return result


def observe_wan_transforms(
    model: nn.Module,
    block_transforms: list[WanBlockTransforms],
) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach hooks that collect Givens statistics without changing model outputs."""
    handles = []
    for block, transform_set in zip(model.blocks, block_transforms):
        modules = dict(block.named_modules())
        observed_ids = set()
        for linear_name, transform in transform_set.linears.items():
            if not isinstance(transform, GivensTransform) or id(transform) in observed_ids:
                continue
            module = modules.get(linear_name)
            if not isinstance(module, nn.Linear):
                raise ValueError(f"Expected Wan Linear at {linear_name}, got {type(module).__name__}")

            def observe_input(_module, inputs, current_transform=transform):
                current_transform.observe(inputs[0])

            handles.append(module.register_forward_pre_hook(observe_input))
            observed_ids.add(id(transform))
    return handles


def finalize_wan_transforms(block_transforms: list[WanBlockTransforms]) -> None:
    for transform_set in block_transforms:
        for transform in transform_set.transforms.values():
            if isinstance(transform, GivensTransform):
                transform.finalize_calibration()


def get_wan_transform_stats(
    block_transforms: list[WanBlockTransforms],
) -> dict[str, int | float]:
    givens_transforms = [
        transform
        for transform_set in block_transforms
        for transform in transform_set.transforms.values()
        if isinstance(transform, GivensTransform)
    ]
    if not givens_transforms:
        return {}
    return {
        "givens_blocks": sum(transform.givens_blocks for transform in givens_transforms),
        "hadamard_blocks": sum(transform.hadamard_blocks for transform in givens_transforms),
        "observed_abs_max": max(transform.observed_abs_max for transform in givens_transforms),
    }


def replace_wan_linears(
    model: nn.Module,
    block_transforms: list[WanBlockTransforms],
    weight_quantizer_kwargs: dict[str, Any] | None,
    activation_quantizer_kwargs: dict[str, Any] | None,
) -> WanQuantizationReport:
    report = WanQuantizationReport()
    for block_idx, (block, transform_set) in enumerate(zip(model.blocks, block_transforms)):
        for linear_name, transform in transform_set.linears.items():
            parent_name, child_name = linear_name.rsplit(".", 1)
            parent = block.get_submodule(parent_name)
            linear = getattr(parent, child_name)
            qualified_name = f"blocks.{block_idx}.{linear_name}"
            if isinstance(linear, WanRTNLinear):
                report.skipped[qualified_name] = "already quantized"
                continue
            if not isinstance(linear, nn.Linear):
                report.skipped[qualified_name] = f"expected Linear, got {type(linear).__name__}"
                continue
            setattr(
                parent,
                child_name,
                WanRTNLinear.from_linear(
                    linear,
                    transform,
                    weight_quantizer_kwargs,
                    activation_quantizer_kwargs,
                ),
            )
            report.replaced.append(qualified_name)
    return report
