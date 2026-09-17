from pathlib import Path

import torch
import torch.nn as nn

from video_quant_lab.analysis.wan.sites import DIFFUSERS_WAN_LINEAR_SITES
from video_quant_lab.analysis.wan.wan_activation_disk_capture import (
    WanActivationDiskCapture,
    artifact_dir,
    branch_for_call,
)


def test_disk_capture_paths_use_sampling_step_and_branch():
    root = Path("outputs/run")
    assert artifact_dir(root, 20, 3, "ffn_in") == root / "step_010/conditional/block_03/ffn_in"
    assert artifact_dir(root, 21, 3, "ffn_in") == root / "step_010/unconditional/block_03/ffn_in"
    assert branch_for_call(20) == "conditional"


def test_disk_capture_accepts_diffusers_site_paths(tmp_path):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn1 = nn.Module()
            self.attn1.to_q = nn.Linear(4, 4)

    model = nn.Module()
    model.blocks = nn.ModuleList([Block()])
    capture = WanActivationDiskCapture(
        model,
        tmp_path,
        tmp_path,
        1024**2,
        [0],
        ["self_qkv"],
        [0],
        linear_sites=DIFFUSERS_WAN_LINEAR_SITES,
    )
    capture.attach()
    capture.call_index = 0
    model.blocks[0].attn1.to_q(torch.ones(1, 2, 4))
    capture.remove()
    metadata = tmp_path / "step_000/conditional/block_00/self_qkv/metadata.json"
    assert metadata.exists()
