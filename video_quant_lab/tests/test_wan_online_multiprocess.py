from argparse import Namespace
import json

import torch
import torch.nn as nn

from video_quant_lab.analysis.cli.capture_render_wan_activations_online import WanOnlineActivationRenderer


def test_multiprocess_renderer_uses_shared_activation_and_writes_results(tmp_path):
    args = Namespace(
        output_dir=tmp_path,
        max_output_gb=0.0,
        render_mode="multiprocess",
        render_workers=2,
        shared_memory_dir=tmp_path / "shared",
        max_inflight_activations=4,
        inflight_memory_fraction=0.25,
        width=16,
        height=16,
        frames=5,
        channel_rms_ratio=5.0,
        mark_top_channels=8,
        isolated_global_percentile=99.99,
        isolated_channel_percentile=99.0,
        isolated_ratio=5.0,
        isolated_max_token_fraction=0.01,
        mark_top_isolated=10,
        isolated_merge_token_gap=1,
        ffn_out_group_size=2,
        image_width=480,
        image_height=360,
        heatmap_percentile=100.0,
        heatmap_gamma=1.0,
        plot_kind="heatmap",
    )
    renderer = WanOnlineActivationRenderer(nn.Module(), args, [], [], [0])
    renderer.call_index = 0
    renderer.timestep = 1.0
    value = torch.ones(1, 2, 4)
    value[:, :, 3] = 12
    renderer._submit(value, 0, "self_qkv", "self_attn.q")
    renderer._submit(value, 0, "self_o", "self_attn.o")
    text_value = torch.ones(1, 1000, 4)
    text_value[:, :, 3] = 12
    text_value[:, 417, 2] = 100
    renderer.text_context_by_call[0] = {
        "effective_token_count": 500,
        "padded_token_count": 1000,
        "token_mask": [True] * 500 + [False] * 500,
    }
    renderer._submit(text_value, 0, "cross_kv", "cross_attn.k")
    renderer.finish()

    for site in ("self_qkv", "self_o"):
        root = tmp_path / "step_000" / "conditional" / "block_00" / site
        metadata = json.loads((root / "metadata.json").read_text())
        assert metadata["render_schema_version"] == 4
        assert metadata["complete"] is True
        assert metadata["activation_stored"] is False
        assert len(list(root.glob("frame_*.png"))) == 2
    cross_root = tmp_path / "step_000" / "conditional" / "block_00" / "cross_kv"
    cross_metadata = json.loads((cross_root / "metadata.json").read_text())
    isolated = cross_metadata["records"][0]["isolated_outliers"]
    assert isolated[0]["peak_token"] == 417
    assert isolated[0]["channel"] == 2
    assert renderer.rendered_activations == 3
    assert renderer.rendered_images == 5


def test_multiprocess_renderer_writes_reference_surface(tmp_path):
    args = Namespace(
        output_dir=tmp_path,
        max_output_gb=0.0,
        render_mode="multiprocess",
        render_workers=1,
        shared_memory_dir=tmp_path / "shared",
        max_inflight_activations=1,
        inflight_memory_fraction=0.25,
        width=16,
        height=16,
        frames=5,
        channel_rms_ratio=5.0,
        mark_top_channels=0,
        isolated_global_percentile=99.99,
        isolated_channel_percentile=99.0,
        isolated_ratio=5.0,
        isolated_max_token_fraction=0.01,
        mark_top_isolated=0,
        isolated_merge_token_gap=1,
        ffn_out_group_size=2,
        image_width=480,
        image_height=360,
        heatmap_percentile=100.0,
        heatmap_gamma=1.0,
        plot_kind="surface",
    )
    renderer = WanOnlineActivationRenderer(nn.Module(), args, [], [], [0])
    renderer.call_index = 0
    renderer._submit(torch.arange(8).reshape(1, 2, 4), 0, "self_qkv", "self_attn.q")
    renderer.finish()

    root = tmp_path / "step_000" / "conditional" / "block_00" / "self_qkv"
    metadata = json.loads((root / "metadata.json").read_text())
    assert len(list(root.glob("surface_frame_*.png"))) == 2
    assert metadata["plot_kind"] == "surface"
    assert metadata["records"][0]["renders"]["surface"]["cmap"] == "viridis"
    assert metadata["records"][0]["renders"]["surface"]["coordinate_range"] == {
        "channel": [0, 3], "token": [0, 0]
    }


def test_renderer_writes_paired_variant_metadata_and_fixed_color_scale(tmp_path):
    args = Namespace(
        output_dir=tmp_path,
        max_output_gb=0.0,
        render_mode="sync",
        render_workers=1,
        shared_memory_dir=tmp_path,
        max_inflight_activations=1,
        inflight_memory_fraction=0.25,
        width=16,
        height=16,
        frames=5,
        channel_rms_ratio=5.0,
        mark_top_channels=0,
        isolated_global_percentile=99.99,
        isolated_channel_percentile=99.0,
        isolated_ratio=5.0,
        isolated_max_token_fraction=0.01,
        mark_top_isolated=0,
        isolated_merge_token_gap=1,
        ffn_out_group_size=2,
        image_width=480,
        image_height=360,
        heatmap_percentile=100.0,
        heatmap_gamma=1.0,
        plot_kind="heatmap",
    )
    renderer = WanOnlineActivationRenderer(nn.Module(), args, [], [], [0])
    renderer.call_index = 0
    renderer._submit(
        torch.arange(8).reshape(1, 2, 4),
        1,
        "cross_q",
        "cross_attn.q",
        variant="hadamard-h32",
        color_max_override=25.0,
        extra_metadata={"mxfp4": {"mse": 0.125}},
    )
    renderer.finish()

    root = (
        tmp_path / "step_000" / "conditional" / "block_01" / "cross_q"
        / "hadamard-h32"
    )
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["variant"] == "hadamard-h32"
    assert metadata["mxfp4"]["mse"] == 0.125
    assert metadata["records"][0]["renders"]["heatmap"]["heatmap_color_max"] == 25.0
