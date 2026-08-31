import torch

from src.utils.wan_activation_stats import ActivationSiteStats


def test_activation_site_stats_tracks_channels_and_calls():
    stats = ActivationSiteStats()
    x = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    stats.update(x, call_index=3, timestep=42.0, sample_elements=64, group_size=32)
    exported = stats.export()
    assert exported["rows"] == 2
    assert exported["channel_max"].shape == (32,)
    assert exported["channel_rms"].shape == (32,)
    assert exported["calls"][0]["call"] == 3
    assert exported["calls"][0]["timestep"] == 42.0
    assert exported["calls"][0]["max_over_rms"] > 1
