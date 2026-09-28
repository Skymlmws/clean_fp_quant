import torch

from src.quantization.msfp import (
    MSFPParams,
    StaticMSFPQuantizer,
    fake_quantize_msfp,
    msfp_codebook,
    search_msfp_params,
)


def test_codebooks_respect_bit_budget_and_include_zero():
    signed = msfp_codebook(MSFPParams(2, 1, True, 6.0))
    unsigned = msfp_codebook(MSFPParams(2, 2, False, 6.0, -0.2))

    assert signed.numel() <= 2**4
    assert unsigned.numel() == 2**4
    assert 0.0 in signed
    assert unsigned[0].item() == torch.tensor(-0.2, dtype=torch.float32).item()


def test_fake_quantization_uses_fixed_codebook():
    params = MSFPParams(2, 1, True, 6.0)
    x = torch.tensor([-20.0, -1.1, 0.0, 1.1, 20.0])
    quantized = fake_quantize_msfp(x, params)

    assert quantized.min() == -6.0
    assert quantized.max() == 6.0
    assert quantized[2] == 0.0
    torch.testing.assert_close(StaticMSFPQuantizer(params)(x), quantized)


def test_search_selects_unsigned_for_gelu_like_distribution():
    torch.manual_seed(0)
    x = torch.cat((torch.linspace(-0.17, 0.0, 128), torch.rand(4096) * 8.0))
    params = search_msfp_params(
        x,
        bits=4,
        allow_unsigned=True,
        maxval_steps=12,
        zero_point_steps=5,
        maximum_search_elements=4096,
    )

    assert not params.signed
    assert params.zero_point <= 0
    assert params.mse < float((x - x.mean()).square().mean())


def test_weight_search_remains_signed():
    x = torch.randn(2048)
    params = search_msfp_params(x, bits=4, allow_unsigned=False, maxval_steps=8)
    assert params.signed
    assert params.exponent_bits + params.mantissa_bits + 1 == 4
