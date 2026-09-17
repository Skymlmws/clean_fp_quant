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

WAN_QUANT_SCOPES = {
    "all": tuple(WAN_LINEAR_TRANSFORM_GROUPS),
    "attention": tuple(
        name for name in WAN_LINEAR_TRANSFORM_GROUPS if not name.startswith("ffn_")
    ),
    "ffn": tuple(name for name in WAN_LINEAR_TRANSFORM_GROUPS if name.startswith("ffn_")),
}


@dataclass
class WanBlockTransforms:
    transforms: dict[str, BaseTransform]
    linears: dict[str, BaseTransform] = field(init=False)

    def __post_init__(self) -> None:
        self.linears = {
            linear_name: self.transforms[group_name]
            for group_name, linear_names in WAN_LINEAR_TRANSFORM_GROUPS.items()
            if group_name in self.transforms
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


@torch.no_grad()
def rotation_validation_metrics(
    source: torch.Tensor,
    transformed: torch.Tensor,
    transform: nn.Module,
    weight: torch.Tensor,
    *,
    maximum_tokens: int = 256,
    maximum_outputs: int = 128,
) -> dict[str, float | int]:
    """Measure norm preservation and Linear equivalence for a fixed rotation."""
    source_flat = source.detach().reshape(-1, source.shape[-1])
    transformed_flat = transformed.detach().reshape(-1, transformed.shape[-1])
    source_norm = source_flat.float().norm(dim=-1)
    transformed_norm = transformed_flat.float().norm(dim=-1)
    norm_error = (transformed_norm - source_norm).abs()
    relative_norm_error = norm_error / source_norm.clamp_min(1e-12)

    stride = max(1, (source_flat.shape[0] + maximum_tokens - 1) // maximum_tokens)
    sampled_source = source_flat[::stride][:maximum_tokens]
    sampled_transformed = transformed_flat[::stride][:maximum_tokens]
    sampled_weight = weight.detach()[:maximum_outputs]
    transformed_weight = transform(sampled_weight, inv_t=True)
    reference_output = F.linear(sampled_source, sampled_weight).float()
    transformed_output = F.linear(sampled_transformed, transformed_weight).float()
    output_error = transformed_output - reference_output
    reference_energy = reference_output.square().sum().double()
    error_energy = output_error.square().sum().double()

    return {
        "token_count": source_flat.shape[0],
        "sampled_token_count": sampled_source.shape[0],
        "sampled_output_count": sampled_weight.shape[0],
        "max_token_l2_abs_error": float(norm_error.max()),
        "mean_token_l2_abs_error": float(norm_error.mean()),
        "max_token_l2_relative_error": float(relative_norm_error.max()),
        "mean_token_l2_relative_error": float(relative_norm_error.mean()),
        "max_linear_abs_error": float(output_error.abs().max()),
        "linear_rmse": float(output_error.square().mean().sqrt()),
        "linear_relative_l2_error": (
            float(torch.sqrt(error_energy / reference_energy))
            if reference_energy > 0 else 0.0
        ),
    }


def build_wan_block_transforms(
    model: nn.Module,
    transform_class: str,
    group_size: int,
    device: torch.device,
    quant_scope: str = "all",
    **transform_kwargs: Any,
) -> list[WanBlockTransforms]:
    if not hasattr(model, "blocks") or not hasattr(model, "dim") or not hasattr(model, "ffn_dim"):
        raise ValueError("Expected a WanModel-like module with blocks, dim, and ffn_dim")
    if quant_scope not in WAN_QUANT_SCOPES:
        choices = ", ".join(WAN_QUANT_SCOPES)
        raise ValueError(f"Unknown Wan quantization scope {quant_scope!r}; expected one of: {choices}")

    result = []
    base_seed = transform_kwargs.pop("seed", None)
    selected_groups = set(WAN_QUANT_SCOPES[quant_scope])
    for block_idx, _ in enumerate(model.blocks):
        transforms = {}
        for transform_idx, name in enumerate(WAN_LINEAR_TRANSFORM_GROUPS):
            if name not in selected_groups:
                continue
            size = model.ffn_dim if name == "ffn_out" else model.dim
            current_kwargs = dict(transform_kwargs)
            if base_seed is not None:
                current_kwargs["seed"] = (
                    base_seed + block_idx * len(WAN_LINEAR_TRANSFORM_GROUPS) + transform_idx
                )
            transforms[name] = build_transform(
                transform_class,
                size=size,
                group_size=group_size,
                device=device,
                **current_kwargs,
            )
        result.append(WanBlockTransforms(transforms))
    return result


def build_wan_mixed_block_transforms(
    model: nn.Module,
    attention_transform_class: str,
    ffn_transform_class: str,
    group_size: int,
    device: torch.device,
    *,
    outlier_threshold: float = 50.0,
    hadamard_randomize: bool = False,
    seed: int = 0,
) -> list[WanBlockTransforms]:
    """Build independently selected transforms for Attention and FFN linears."""
    per_scope = []
    for scope, transform_class in (
        ("attention", attention_transform_class),
        ("ffn", ffn_transform_class),
    ):
        kwargs: dict[str, Any] = {}
        if transform_class == "givens":
            kwargs.update(outlier_threshold=outlier_threshold, seed=seed)
        elif transform_class == "hadamard":
            kwargs.update(randomize=hadamard_randomize, seed=seed)
        per_scope.append(
            build_wan_block_transforms(
                model,
                transform_class,
                group_size,
                device,
                quant_scope=scope,
                **kwargs,
            )
        )
    return [
        WanBlockTransforms({**attention.transforms, **ffn.transforms})
        for attention, ffn in zip(*per_scope)
    ]


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
