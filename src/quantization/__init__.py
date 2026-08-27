"""Quantization entry points, imported lazily to avoid model-specific dependencies."""

__all__ = ("rtn_quantization", "gptq_quantization", "wan_rtn_quantization")


def __getattr__(name):
    if name == "rtn_quantization":
        from .rtn import rtn_quantization

        return rtn_quantization
    if name == "gptq_quantization":
        from .gptq import gptq_quantization

        return gptq_quantization
    if name == "wan_rtn_quantization":
        from .wan_rtn import wan_rtn_quantization

        return wan_rtn_quantization
    raise AttributeError(name)
