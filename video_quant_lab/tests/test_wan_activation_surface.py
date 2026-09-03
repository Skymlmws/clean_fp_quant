from argparse import Namespace
import threading

import numpy as np
import torch
import torch.nn as nn

from video_quant_lab.analysis.wan.wan_activation_disk_capture import WanActivationDiskCapture
from video_quant_lab.analysis.cli.capture_render_wan_activations_online import WanOnlineActivationRenderer
from video_quant_lab.analysis.wan.wan_activation_surface import activation_matrix, parse_indices, sampled_activation_matrix
from video_quant_lab.analysis.cli.render_wan_self_qkv_frames import channel_outliers
from video_quant_lab.analysis.cli.visualize_wan_activation_surfaces import artifact_relative_dir, branch_for_call, call_relative_dir, downsample, output_usage, render_bars, render_heatmap, selected_sites


def test_activation_matrix_selects_batch_and_flattens_tokens():
    value = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    result = activation_matrix(value, batch_index=1)
    assert result.shape == (3, 4)
    assert torch.equal(result, value[1])
    sampled, tokens, channels = sampled_activation_matrix(value, 1, 2, 3)
    assert sampled.shape == (2, 3)
    assert tokens.tolist() == [0, 2]
    assert channels.tolist() == [0, 2, 3]


def test_parse_indices_and_sites():
    assert parse_indices("0,2-4,2", 6) == [0, 2, 3, 4]
    assert "ffn_in" in selected_sites("all")
    assert selected_sites("self_qkv,ffn_out") == ["self_qkv", "ffn_out"]
    assert artifact_relative_dir(2, 7, "ffn_in").as_posix() == "step_001/conditional/block_07/ffn_in"
    assert call_relative_dir(3).as_posix() == "step_001/unconditional"
    assert branch_for_call(3) == "unconditional"


def test_downsample_preserves_endpoints_and_bar_renderer_writes_png(tmp_path):
    matrix = torch.linspace(-2, 3, 20 * 30).reshape(20, 30)
    values, tokens, channels = downsample(matrix, 8, 10)
    assert values.shape == (8, 10)
    assert (tokens[[0, -1]] == np.array([0, 19])).all()
    assert (channels[[0, -1]] == np.array([0, 29])).all()
    output = tmp_path / "bars.png"
    stats = render_bars(values, tokens, channels, "test", output, 640, 480, 99.5)
    assert output.exists()
    assert stats["minimum"] < 0 < stats["maximum"]


def test_zero_sampling_limits_keep_the_complete_matrix():
    matrix = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    sampled, tokens, channels = sampled_activation_matrix(matrix, 0, 0, 0)
    assert torch.equal(sampled, matrix)
    assert tokens.tolist() == list(range(5))
    assert channels.tolist() == list(range(7))
    values, token_array, channel_array = downsample(matrix, 0, 0)
    assert values.shape == (5, 7)
    assert token_array.tolist() == list(range(5))
    assert channel_array.tolist() == list(range(7))


def test_output_usage_counts_completed_bar_images(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "bars.png").write_bytes(b"123")
    (tmp_path / "state.json").write_bytes(b"{}")
    assert output_usage(tmp_path) == (5, 1)


def test_heatmap_renderer_writes_full_matrix_view(tmp_path):
    values = np.arange(35, dtype=np.float32).reshape(5, 7)
    output = tmp_path / "heatmap.png"
    stats = render_heatmap(values, np.arange(5), np.arange(7), "test", output, 640, 480)
    assert output.exists()
    assert stats["heatmap_color_max"] > 0


def test_disk_capture_records_effective_text_tokens(tmp_path):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.text_len = 8
            self.blocks = nn.ModuleList()

    model = Model()
    capture = WanActivationDiskCapture(
        model, tmp_path / "output", tmp_path, 0, [], [], [0]
    )
    context = [torch.zeros(3, 16)]
    capture._model_pre_hook(model, ([], torch.tensor([1.0]), context, 0), {})
    assert capture.effective_text_token_count == 3
    assert capture.text_token_mask == [True, True, True, False, False, False, False, False]
    assert capture.text_context_by_call[0]["padded_token_count"] == 8


def test_channel_outliers_rank_persistent_bright_lines():
    values = np.ones((20, 5), dtype=np.float32)
    values[:, 3] = 12.0
    values[0, 1] = 100.0
    records = channel_outliers(values, np.arange(5), 5.0, 8)
    assert [record["channel"] for record in records] == [3]
    assert records[0]["robust_rms_over_median"] == 12.0


def test_online_async_renderer_bounds_inflight_activations(tmp_path):
    args = Namespace(
        output_dir=tmp_path,
        max_output_gb=1.0,
        render_mode="async",
        render_workers=1,
        shared_memory_dir=tmp_path,
        max_inflight_activations=2,
        inflight_memory_fraction=0.25,
        width=16,
        height=16,
        frames=5,
    )
    model = nn.Module()
    renderer = WanOnlineActivationRenderer(model, args, [], [], [0])
    renderer.call_index = 0
    started = threading.Event()
    release = threading.Event()
    third_submitted = threading.Event()

    def hold_job(_job):
        started.set()
        assert release.wait(timeout=2)

    renderer._render_job = hold_job
    value = torch.ones(1, 2, 4)
    renderer._submit(value, 0, "self_qkv", "self_attn.q")
    assert started.wait(timeout=2)
    renderer._submit(value, 1, "self_qkv", "self_attn.q")

    def submit_third():
        renderer._submit(value, 2, "self_qkv", "self_attn.q")
        third_submitted.set()

    submitter = threading.Thread(target=submit_third)
    submitter.start()
    assert not third_submitted.wait(timeout=0.1)
    release.set()
    submitter.join(timeout=2)
    assert third_submitted.is_set()
    renderer.finish()
