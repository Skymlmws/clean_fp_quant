"""Pure detection utilities for structured Wan activation outliers."""

from __future__ import annotations

import numpy as np


def persistent_channel_outliers(
    values: np.ndarray,
    minimum_ratio: float = 5.0,
    maximum_channels: int = 8,
    winsor_percentile: float = 95.0,
) -> list[dict]:
    """Find channels that remain unusually large across many tokens."""
    if not 0 < winsor_percentile <= 100:
        raise ValueError("winsor_percentile must be in (0, 100]")
    if minimum_ratio <= 0 or maximum_channels < 0:
        raise ValueError("minimum_ratio must be positive and maximum_channels non-negative")
    if maximum_channels == 0:
        return []
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("values must be a non-empty 2D token-by-channel matrix")
    magnitudes = np.abs(array)
    limits = np.percentile(magnitudes, winsor_percentile, axis=0)
    robust_rms = np.sqrt(np.mean(np.square(np.minimum(magnitudes, limits[None, :])), axis=0))
    median = float(np.median(robust_rms))
    if median <= 0:
        return []
    ratios = robust_rms / median
    candidates = np.flatnonzero(ratios >= minimum_ratio)
    order = candidates[np.argsort(ratios[candidates])[::-1]][:maximum_channels]
    raw_rms = np.sqrt(np.mean(np.square(magnitudes), axis=0))
    return [
        {
            "channel": int(channel),
            "robust_rms": float(robust_rms[channel]),
            "robust_rms_over_median": float(ratios[channel]),
            "rms": float(raw_rms[channel]),
            "max_abs": float(magnitudes[:, channel].max()),
        }
        for channel in order
    ]


def isolated_token_outliers(
    values: np.ndarray,
    persistent_channels: set[int] | None = None,
    global_percentile: float = 99.99,
    channel_percentile: float = 99.0,
    minimum_channel_ratio: float = 5.0,
    maximum_token_fraction: float = 0.01,
    maximum_entities: int = 10,
    merge_token_gap: int = 1,
) -> list[dict]:
    """Find isolated points and short same-channel token clusters."""
    if not 0 < global_percentile <= 100 or not 0 < channel_percentile <= 100:
        raise ValueError("percentiles must be in (0, 100]")
    if minimum_channel_ratio <= 0:
        raise ValueError("minimum_channel_ratio must be positive")
    if not 0 < maximum_token_fraction <= 1:
        raise ValueError("maximum_token_fraction must be in (0, 1]")
    if maximum_entities < 0 or merge_token_gap < 0:
        raise ValueError("maximum_entities and merge_token_gap must be non-negative")
    if maximum_entities == 0:
        return []

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("values must be a non-empty 2D token-by-channel matrix")
    magnitudes = np.abs(array)
    token_count, channel_count = magnitudes.shape
    global_threshold = float(np.percentile(magnitudes, global_percentile))
    channel_baselines = np.percentile(magnitudes, channel_percentile, axis=0)
    positive_baselines = channel_baselines[channel_baselines > 0]
    fallback = float(np.median(positive_baselines)) if positive_baselines.size else 1e-12
    denominators = np.maximum(channel_baselines, max(fallback * 1e-6, 1e-12))
    ratios = magnitudes / denominators[None, :]
    candidate_mask = (magnitudes >= global_threshold) & (ratios >= minimum_channel_ratio)
    for channel in persistent_channels or set():
        if 0 <= channel < channel_count:
            candidate_mask[:, channel] = False

    maximum_tokens = max(1, int(np.floor(token_count * maximum_token_fraction)))
    entities: list[dict] = []
    for channel in np.flatnonzero(candidate_mask.any(axis=0)):
        tokens = np.flatnonzero(candidate_mask[:, channel])
        if tokens.size > maximum_tokens:
            continue
        groups: list[list[int]] = []
        for token in tokens.tolist():
            if not groups or token - groups[-1][-1] > merge_token_gap:
                groups.append([token])
            else:
                groups[-1].append(token)
        for group in groups:
            group_array = np.asarray(group, dtype=np.int64)
            local_peak = int(group_array[np.argmax(magnitudes[group_array, channel])])
            entities.append({
                "kind": "point" if len(group) == 1 else "token_cluster",
                "channel": int(channel),
                "token_start": int(group[0]),
                "token_end": int(group[-1]),
                "token_count": len(group),
                "peak_token": local_peak,
                "value": float(array[local_peak, channel]),
                "abs_value": float(magnitudes[local_peak, channel]),
                "channel_baseline_percentile": channel_percentile,
                "channel_baseline": float(channel_baselines[channel]),
                "peak_over_channel_baseline": float(ratios[local_peak, channel]),
                "global_threshold_percentile": global_percentile,
                "global_threshold": global_threshold,
            })
    entities.sort(key=lambda record: record["peak_over_channel_baseline"], reverse=True)
    return entities[:maximum_entities]
