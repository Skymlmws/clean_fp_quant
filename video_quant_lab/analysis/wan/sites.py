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


# Hugging Face Diffusers uses different module names for the same seven Wan
# transform groups. Keeping the semantic site names stable lets captures be
# consumed by the same offline analysis regardless of which Wan frontend made
# them.
DIFFUSERS_WAN_LINEAR_SITES = {
    "self_qkv": ("attn1.to_q", "attn1.to_k", "attn1.to_v"),
    "self_o": ("attn1.to_out.0",),
    "cross_q": ("attn2.to_q",),
    "cross_kv": ("attn2.to_k", "attn2.to_v"),
    "cross_o": ("attn2.to_out.0",),
    "ffn_in": ("ffn.net.0.proj",),
    "ffn_out": ("ffn.net.2",),
}
