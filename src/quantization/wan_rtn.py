"""RTN fake-quantization flow specialized for the Wan2.1 DiT backbone."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from ..utils.common_utils import to
from ..utils.wan_utils import (
    WanQuantizationReport,
    build_wan_block_transforms,
    finalize_wan_transforms,
    get_wan_transform_stats,
    observe_wan_transforms,
    replace_wan_linears,
)


WanCalibrationBatch = tuple[tuple[Any, ...], dict[str, Any]]


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
) -> WanQuantizationReport:
    """Calibrate input transforms and replace the 300 DiT Linear layers.

    This is a fake-quant path: transformed weights are quantized/dequantized once,
    while activations are dynamically fake-quantized on each forward when requested.
    """
    transform_kwargs = {}
    if transform_class == "givens":
        transform_kwargs["outlier_threshold"] = outlier_threshold
    block_transforms = build_wan_block_transforms(
        model,
        transform_class,
        transform_group_size,
        device,
        **transform_kwargs,
    )

    if transform_class == "givens":
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

    weight_quantizer_kwargs = _quantizer_kwargs(
        weight_bits,
        quant_format,
        weight_granularity,
        weight_observer,
        weight_group_size if weight_granularity == "group" else None,
        scale_precision,
    )
    activation_quantizer_kwargs = _quantizer_kwargs(
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
    )
    if transform_class == "givens":
        report.transform_stats = get_wan_transform_stats(block_transforms)
    return report
