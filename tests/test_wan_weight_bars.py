import torch

from scripts.visualize.visualize_wan_weight_bars import safe_linear_name, sample_weight, selected_linears


def test_sample_weight_keeps_complete_matrix_when_limits_are_zero():
    weight = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    matrix, outputs, inputs = sample_weight(weight, 0, 0)
    assert matrix.shape == (5, 7)
    assert outputs.tolist() == list(range(5))
    assert inputs.tolist() == list(range(7))


def test_selected_linears_expands_shared_site_members():
    assert selected_linears("all", ["self_qkv"]) == [
        ("self_qkv", "self_attn.q"),
        ("self_qkv", "self_attn.k"),
        ("self_qkv", "self_attn.v"),
    ]
    assert safe_linear_name("self_attn.q") == "self_attn_q"
