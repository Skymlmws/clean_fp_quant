import torch

from render_wan_self_qkv_frames import frame_view, video_token_grid


def test_video_token_grid_for_wan_480p():
    assert video_token_grid({"size": [832, 480], "frames": 81}) == (21, 30, 52)


def test_frame_view_selects_contiguous_spatial_tokens():
    activation = torch.arange(2 * 2 * 3 * 4).reshape(1, 12, 4)
    result = frame_view(activation, 1, (2, 2, 3))
    assert result.shape == (6, 4)
    assert torch.equal(result, activation[0, 6:12])
