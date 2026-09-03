"""Stable names of Wan linear sites used by capture and visualization tools."""

WAN_LINEAR_SITES = {
    "self_qkv": ("self_attn.q", "self_attn.k", "self_attn.v"),
    "self_o": ("self_attn.o",),
    "cross_q": ("cross_attn.q",),
    "cross_kv": ("cross_attn.k", "cross_attn.v"),
    "cross_o": ("cross_attn.o",),
    "ffn_in": ("ffn.0",),
    "ffn_out": ("ffn.2",),
}
