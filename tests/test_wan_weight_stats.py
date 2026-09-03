import torch

from src.quantization.quantizer import Quantizer
from scripts.visualize.visualize_wan_weights import analyze_weight


def test_analyze_weight_reports_mxfp4_error_and_channels():
    quantizer = Quantizer(bits=4, symmetric=True, format="mxfp", granularity="group", group_size=32, scale_precision="e8m0", observer="minmax")
    weight = torch.linspace(-3, 3, 64).reshape(2, 32)
    metrics, channel_max, channel_rms = analyze_weight(weight, quantizer, 32)
    assert metrics["in_features"] == 32
    assert metrics["out_features"] == 2
    assert metrics["w4_mse"] > 0
    assert metrics["w4_sqnr_db"] > 0
    assert channel_max.shape == (32,)
    assert channel_rms.shape == (32,)
