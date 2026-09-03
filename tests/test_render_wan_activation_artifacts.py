import torch

from scripts.visualize.render_wan_activation_artifacts import sampled_view


def test_sampled_view_limits_only_the_render_view():
    matrix = torch.arange(20 * 30, dtype=torch.bfloat16).reshape(1, 20, 30)
    view, tokens, channels = sampled_view(matrix, 8, 10)
    assert view.shape == (8, 10)
    assert tokens[[0, -1]].tolist() == [0, 19]
    assert channels[[0, -1]].tolist() == [0, 29]
