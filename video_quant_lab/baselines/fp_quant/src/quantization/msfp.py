"""Static Mixup-Sign floating-point fake quantization.

This module implements the search-based PTQ stage described by MSFP.  It uses
explicit FP codebooks, so it is intentionally a numerical simulator rather
than a packed 4-bit kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class MSFPParams:
    exponent_bits: int
    mantissa_bits: int
    signed: bool
    maxval: float
    zero_point: float = 0.0
    mse: float = math.inf

    @property
    def format_name(self) -> str:
        return f"E{self.exponent_bits}M{self.mantissa_bits}"


def _positive_codebook(exponent_bits: int, mantissa_bits: int) -> torch.Tensor:
    """Return a normalized positive FP codebook including exact zero.

    Bias is represented by scaling the largest finite code to ``maxval``.
    E0 formats are treated as an evenly spaced mantissa-only codebook.
    """
    if exponent_bits == 0:
        count = 1 << mantissa_bits
        return torch.linspace(0.0, 1.0, count, dtype=torch.float64)
    values = []
    for exponent in range(1 << exponent_bits):
        for mantissa in range(1 << mantissa_bits):
            values.append((2.0**exponent) * (1.0 + mantissa / (2.0**mantissa_bits)))
    # Reserve the smallest encoding for exact zero. Signed formats therefore
    # have one redundant negative-zero encoding, as real FP formats do.
    values = sorted(set(values))
    result = torch.tensor([0.0, *values[1:]], dtype=torch.float64)
    return result / result[-1]


def msfp_codebook(params: MSFPParams, *, device: torch.device | None = None) -> torch.Tensor:
    positive = _positive_codebook(params.exponent_bits, params.mantissa_bits)
    if params.signed:
        levels = torch.cat((-positive[1:].flip(0), positive)) * params.maxval
    else:
        levels = positive * (params.maxval - params.zero_point) + params.zero_point
    return levels.to(device=device, dtype=torch.float32)


def fake_quantize_msfp(x: torch.Tensor, params: MSFPParams) -> torch.Tensor:
    levels = msfp_codebook(params, device=x.device)
    source = x.float().contiguous()
    indices = torch.searchsorted(levels, source)
    upper_index = indices.clamp(max=levels.numel() - 1)
    lower_index = (indices - 1).clamp(min=0)
    lower = levels[lower_index]
    upper = levels[upper_index]
    nearest = torch.where((source - lower).abs() <= (upper - source).abs(), lower, upper)
    return nearest.to(dtype=x.dtype)


def _sample_for_search(x: torch.Tensor, maximum_elements: int) -> torch.Tensor:
    flat = x.detach().float().reshape(-1)
    if flat.numel() <= maximum_elements:
        return flat
    stride = math.ceil(flat.numel() / maximum_elements)
    return flat[::stride][:maximum_elements]


def search_msfp_params(
    x: torch.Tensor,
    *,
    bits: int = 4,
    allow_unsigned: bool,
    maxval_steps: int = 25,
    zero_point_steps: int = 7,
    maximum_search_elements: int = 65536,
) -> MSFPParams:
    """Choose format, range, sign mode and zero point by reconstruction MSE."""
    if bits < 2:
        raise ValueError("MSFP requires at least two bits")
    if maxval_steps < 2:
        raise ValueError("maxval_steps must be at least two")
    sample = _sample_for_search(x, maximum_search_elements)
    if sample.numel() == 0:
        raise ValueError("Cannot calibrate MSFP from an empty tensor")
    observed_max = float(sample.max())
    observed_abs_max = float(sample.abs().max())
    if observed_abs_max == 0:
        return MSFPParams(bits - 1, 0, True, 1.0, mse=0.0)

    best: MSFPParams | None = None
    signed_formats = [(e, bits - 1 - e) for e in range(bits)]
    signed_maxvals = torch.linspace(0.8, 2.0, maxval_steps) * observed_abs_max
    for exponent_bits, mantissa_bits in signed_formats:
        for maxval in signed_maxvals.tolist():
            candidate = MSFPParams(exponent_bits, mantissa_bits, True, maxval)
            error = float((fake_quantize_msfp(sample, candidate).float() - sample).square().mean())
            if best is None or error < best.mse:
                best = MSFPParams(exponent_bits, mantissa_bits, True, maxval, mse=error)

    if allow_unsigned:
        observed_min = min(float(sample.min()), 0.0)
        zero_points = torch.linspace(observed_min, 0.0, zero_point_steps).tolist()
        upper = max(observed_max, 0.0)
        unsigned_maxvals = torch.linspace(0.05, 1.0, maxval_steps) * max(
            upper - observed_min, torch.finfo(torch.float32).eps
        )
        for exponent_bits in range(bits + 1):
            mantissa_bits = bits - exponent_bits
            for zero_point in zero_points:
                for span in unsigned_maxvals.tolist():
                    maxval = zero_point + span
                    if maxval <= zero_point:
                        continue
                    candidate = MSFPParams(
                        exponent_bits, mantissa_bits, False, maxval, zero_point
                    )
                    error = float(
                        (fake_quantize_msfp(sample, candidate).float() - sample).square().mean()
                    )
                    if best is None or error < best.mse:
                        best = MSFPParams(
                            exponent_bits, mantissa_bits, False, maxval, zero_point, error
                        )
    assert best is not None
    return best


class StaticMSFPQuantizer:
    """A fixed-parameter fake quantizer selected during calibration."""

    def __init__(self, params: MSFPParams) -> None:
        self.params = params

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quantize_msfp(x, self.params)
