import numpy as np
import pytest

from video_quant_lab.analysis.wan.wan_activation_outliers import isolated_token_outliers, persistent_channel_outliers


def test_distinguishes_persistent_channel_from_isolated_point():
    values = np.ones((1000, 8), dtype=np.float32)
    values[:, 3] = 12.0
    values[417, 6] = 100.0
    persistent = persistent_channel_outliers(values, minimum_ratio=5.0)
    assert [row["channel"] for row in persistent] == [3]
    isolated = isolated_token_outliers(
        values, {3}, global_percentile=99.9, minimum_channel_ratio=5.0,
    )
    assert len(isolated) == 1
    assert isolated[0]["kind"] == "point"
    assert isolated[0]["peak_token"] == 417
    assert isolated[0]["channel"] == 6


def test_merges_adjacent_tokens_into_a_cluster():
    values = np.ones((1000, 4), dtype=np.float32)
    values[100:103, 2] = [20.0, 30.0, 25.0]
    isolated = isolated_token_outliers(
        values, global_percentile=99.7, minimum_channel_ratio=5.0,
    )
    assert len(isolated) == 1
    assert isolated[0]["kind"] == "token_cluster"
    assert isolated[0]["token_start"] == 100
    assert isolated[0]["token_end"] == 102
    assert isolated[0]["peak_token"] == 101


def test_rejects_empty_or_non_matrix_inputs():
    with pytest.raises(ValueError, match="non-empty 2D"):
        isolated_token_outliers(np.empty((0, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty 2D"):
        persistent_channel_outliers(np.ones(4, dtype=np.float32))
