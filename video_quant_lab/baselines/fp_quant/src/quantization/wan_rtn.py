"""RTN fake-quantization flow specialized for the Wan2.1 DiT backbone."""

from __future__ import annotations

from collections.abc import Iterable
import math
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils.common_utils import to
from ..utils.wan_utils import (
    WanQuantizationReport,
    build_wan_block_transforms,
    build_wan_mixed_block_transforms,
    finalize_wan_transforms,
    get_wan_transform_stats,
    observe_wan_transforms,
    replace_wan_linears,
)
from .msfp import MSFPParams, search_msfp_params


WanCalibrationBatch = tuple[tuple[Any, ...], dict[str, Any]]


def _msfp_params_from_dict(payload: dict[str, Any]) -> MSFPParams:
    return MSFPParams(
        exponent_bits=int(payload["exponent_bits"]),
        mantissa_bits=int(payload["mantissa_bits"]),
        signed=bool(payload["signed"]),
        maxval=float(payload["maxval"]),
        zero_point=float(payload.get("zero_point", 0.0)),
        mse=float(payload.get("mse", math.inf)),
    )


@torch.no_grad()
def apply_wan_msfp_config(
    model: nn.Module,
    config: str | Path | dict[str, Any],
    device: torch.device,
    *,
    activation_mode: str = "msfp",
) -> WanQuantizationReport:
    """Apply a saved real-trajectory MSFP configuration without recalibration."""
    if activation_mode not in {"signed", "msfp"}:
        raise ValueError("activation_mode must be 'signed' or 'msfp'")
    if not isinstance(config, dict):
        config = json.loads(Path(config).read_text())
    layer_payloads = config.get("layers", {})
    block_transforms = build_wan_block_transforms(
        model, "identity", group_size=32, device=device, quant_scope="all"
    )
    parameters = {}
    expected_names = {
        f"blocks.{block_index}.{linear_name}"
        for block_index, transform_set in enumerate(block_transforms)
        for linear_name in transform_set.linears
    }
    missing = expected_names - set(layer_payloads)
    if missing:
        preview = ", ".join(sorted(missing)[:3])
        raise ValueError(f"MSFP configuration is missing {len(missing)} layers: {preview}")
    activation_key = f"activation_{activation_mode}"
    for name in expected_names:
        payload = layer_payloads[name]
        parameters[name] = (
            _msfp_params_from_dict(payload["weight"]),
            _msfp_params_from_dict(payload[activation_key]),
        )
    return replace_wan_linears(model, block_transforms, None, None, parameters)


def _collect_msfp_activation_samples(
    model: nn.Module,
    block_transforms,
    calibration_batches: Iterable[WanCalibrationBatch],
    device: torch.device,
    amp_dtype: torch.dtype,
    maximum_elements: int,
) -> dict[str, torch.Tensor]:
    samples: dict[str, list[torch.Tensor]] = {}
    handles = []
    batch_count = len(calibration_batches)  # The public entry point materializes this iterable.
    per_call_elements = math.ceil(maximum_elements / max(batch_count, 1))
    for block_idx, (block, transform_set) in enumerate(zip(model.blocks, block_transforms)):
        modules = dict(block.named_modules())
        for linear_name, transform in transform_set.linears.items():
            qualified_name = f"blocks.{block_idx}.{linear_name}"
            module = modules[linear_name]

            def capture(_module, inputs, name=qualified_name, current_transform=transform):
                flat = current_transform(inputs[0]).detach().float().reshape(-1)
                remaining = maximum_elements - sum(part.numel() for part in samples.get(name, ()))
                if remaining <= 0:
                    return
                take = min(remaining, per_call_elements)
                stride = max(1, (flat.numel() + take - 1) // take)
                samples.setdefault(name, []).append(flat[::stride][:take].cpu())

            handles.append(module.register_forward_pre_hook(capture))
    count = 0
    try:
        for input_args, input_kwargs in calibration_batches:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                model(*to(input_args, device=device), **to(input_kwargs, device=device))
            count += 1
    finally:
        for handle in handles:
            handle.remove()
    if count == 0:
        raise ValueError("MSFP activation calibration requires at least one Wan calibration batch")
    return {name: torch.cat(parts) for name, parts in samples.items()}


def _search_wan_msfp_parameters(
    model: nn.Module,
    block_transforms,
    activation_samples: dict[str, torch.Tensor],
    *,
    weight_bits: int,
    activation_bits: int,
    maxval_steps: int,
    zero_point_steps: int,
    maximum_search_elements: int,
    allow_unsigned_aal: bool,
) -> dict[str, tuple[MSFPParams | None, MSFPParams | None]]:
    result = {}
    for block_idx, (block, transform_set) in enumerate(zip(model.blocks, block_transforms)):
        modules = dict(block.named_modules())
        for linear_name, transform in transform_set.linears.items():
            name = f"blocks.{block_idx}.{linear_name}"
            linear = modules[linear_name]
            # MSFP reserves unsigned+zero-point candidates for AALs. In Wan,
            # the second FFN projection consumes GELU output; attention and
            # FFN-input projections consume normalized, signed activations.
            allow_unsigned = (
                allow_unsigned_aal
                and linear_name in transform_set.linear_groups["ffn_out"]
            )
            weight_params = None
            activation_params = None
            if weight_bits < 16:
                weight_params = search_msfp_params(
                    transform(linear.weight, inv_t=True), bits=weight_bits,
                    allow_unsigned=False, maxval_steps=maxval_steps,
                    maximum_search_elements=maximum_search_elements,
                )
            if activation_bits < 16:
                activation_params = search_msfp_params(
                    activation_samples[name], bits=activation_bits,
                    allow_unsigned=allow_unsigned, maxval_steps=maxval_steps,
                    zero_point_steps=zero_point_steps,
                    maximum_search_elements=maximum_search_elements,
                )
            result[name] = (weight_params, activation_params)
    return result


def _quantizer_kwargs(
    bits: int,
    quant_format: str,
    granularity: str,
    observer: str,
    group_size: int | None,
    scale_precision: str,
) -> dict[str, Any] | None:
    if bits >= 16:
        return None
    return {
        "bits": bits,
        "symmetric": True,
        "format": quant_format,
        "granularity": granularity,
        "observer": observer,
        "group_size": group_size,
        "scale_precision": scale_precision,
    }


@torch.no_grad()
def wan_rtn_quantization(
    model: nn.Module,
    calibration_batches: Iterable[WanCalibrationBatch],
    device: torch.device,
    *,
    transform_class: str = "givens",
    transform_group_size: int = 32,
    transform_randomize: bool = False,
    transform_seed: int = 0,
    quant_scope: str = "all",
    attention_transform_class: str | None = None,
    ffn_transform_class: str | None = None,
    outlier_threshold: float = 50.0,
    weight_bits: int = 4,
    activation_bits: int = 16,
    quant_format: str = "mxfp",
    weight_granularity: str = "group",
    activation_granularity: str = "group",
    weight_group_size: int | None = 32,
    activation_group_size: int | None = 32,
    weight_observer: str = "minmax",
    activation_observer: str = "minmax",
    scale_precision: str = "e8m0",
    amp_dtype: torch.dtype = torch.bfloat16,
    msfp_maxval_steps: int = 12,
    msfp_zero_point_steps: int = 5,
    msfp_maximum_search_elements: int = 4096,
    msfp_allow_unsigned_aal: bool = True,
) -> WanQuantizationReport:
    """Calibrate input transforms and replace the 300 DiT Linear layers.

    This is a fake-quant path. Transformed weights are quantized/dequantized once.
    MXFP/NVFP activations use dynamic parameters, while MSFP activations use the
    fixed per-layer parameters selected from calibration data.
    """
    calibration_batches = list(calibration_batches)
    mixed = attention_transform_class is not None or ffn_transform_class is not None
    if mixed:
        if quant_scope != "all":
            raise ValueError("Mixed Attention/FFN transforms require quant_scope='all'")
        if attention_transform_class is None or ffn_transform_class is None:
            raise ValueError("Both attention_transform_class and ffn_transform_class are required")
        block_transforms = build_wan_mixed_block_transforms(
            model,
            attention_transform_class,
            ffn_transform_class,
            transform_group_size,
            device,
            outlier_threshold=outlier_threshold,
            hadamard_randomize=transform_randomize,
            seed=transform_seed,
        )
    else:
        transform_kwargs = {}
        if transform_class == "givens":
            transform_kwargs.update(outlier_threshold=outlier_threshold, seed=transform_seed)
        elif transform_class == "hadamard":
            transform_kwargs.update(randomize=transform_randomize, seed=transform_seed)
        block_transforms = build_wan_block_transforms(
            model,
            transform_class,
            transform_group_size,
            device,
            quant_scope=quant_scope,
            **transform_kwargs,
        )

    has_givens = transform_class == "givens" if not mixed else (
        attention_transform_class == "givens" or ffn_transform_class == "givens"
    )
    if has_givens:
        handles = observe_wan_transforms(model, block_transforms)
        sample_count = 0
        try:
            for input_args, input_kwargs in calibration_batches:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
                    model(*to(input_args, device=device), **to(input_kwargs, device=device))
                sample_count += 1
        finally:
            for handle in handles:
                handle.remove()
        if sample_count == 0:
            raise ValueError("Givens calibration requires at least one Wan calibration batch")
        finalize_wan_transforms(block_transforms)

    msfp_parameters = None
    if quant_format == "msfp":
        activation_samples = {}
        if activation_bits < 16:
            activation_samples = _collect_msfp_activation_samples(
                model, block_transforms, calibration_batches, device, amp_dtype,
                msfp_maximum_search_elements,
            )
        msfp_parameters = _search_wan_msfp_parameters(
            model, block_transforms, activation_samples,
            weight_bits=weight_bits, activation_bits=activation_bits,
            maxval_steps=msfp_maxval_steps, zero_point_steps=msfp_zero_point_steps,
            maximum_search_elements=msfp_maximum_search_elements,
            allow_unsigned_aal=msfp_allow_unsigned_aal,
        )

    weight_quantizer_kwargs = None if quant_format == "msfp" else _quantizer_kwargs(
        weight_bits,
        quant_format,
        weight_granularity,
        weight_observer,
        weight_group_size if weight_granularity == "group" else None,
        scale_precision,
    )
    activation_quantizer_kwargs = None if quant_format == "msfp" else _quantizer_kwargs(
        activation_bits,
        quant_format,
        activation_granularity,
        activation_observer,
        activation_group_size if activation_granularity == "group" else None,
        scale_precision,
    )
    report = replace_wan_linears(
        model,
        block_transforms,
        weight_quantizer_kwargs,
        activation_quantizer_kwargs,
        msfp_parameters,
    )
    if has_givens:
        report.transform_stats = get_wan_transform_stats(block_transforms)
    return report
