import torch
import torch.nn as nn
import pytest

from src.quantization.wan_rtn import wan_rtn_quantization
from src.utils.wan_utils import WanRTNLinear


class ToyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class ToyWanBlock(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.self_attn = ToyAttention(dim)
        self.cross_attn = ToyAttention(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, x, context):
        self_value = self.self_attn.q(x) + self.self_attn.k(x) + self.self_attn.v(x)
        x = x + self.self_attn.o(self_value)
        cross_value = (
            self.cross_attn.q(x)
            + self.cross_attn.k(context)
            + self.cross_attn.v(context)
        )
        x = x + self.cross_attn.o(cross_value)
        return x + self.ffn(x)


class ToyWanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 8
        self.ffn_dim = 16
        self.blocks = nn.ModuleList([ToyWanBlock(self.dim, self.ffn_dim)])

    def forward(self, x, context):
        for block in self.blocks:
            x = block(x, context)
        return x


def test_wan_rtn_replaces_ten_linears_and_preserves_transform_equivalence():
    model = ToyWanModel().eval()
    x = torch.randn(2, 4, 8)
    context = torch.randn(2, 4, 8)
    x[0, 0, 2] = 100
    reference = model(x, context)

    report = wan_rtn_quantization(
        model,
        [((x, context), {})],
        torch.device("cpu"),
        transform_class="givens",
        transform_group_size=4,
        outlier_threshold=20,
        weight_bits=16,
        activation_bits=16,
    )
    candidate = model(x, context)

    assert report.replaced_count == 10
    assert sum(isinstance(module, WanRTNLinear) for module in model.modules()) == 10
    torch.testing.assert_close(candidate, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("transform_class", ["identity", "hadamard"])
def test_wan_rtn_non_calibrated_transforms(transform_class):
    model = ToyWanModel().eval()
    x = torch.randn(2, 4, 8)
    context = torch.randn(2, 4, 8)
    reference = model(x, context)

    report = wan_rtn_quantization(
        model,
        [],
        torch.device("cpu"),
        transform_class=transform_class,
        transform_group_size=4,
        weight_bits=16,
        activation_bits=16,
    )

    assert report.replaced_count == 10
    torch.testing.assert_close(model(x, context), reference, atol=2e-5, rtol=2e-5)
