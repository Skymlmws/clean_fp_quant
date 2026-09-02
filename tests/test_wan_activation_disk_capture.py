from pathlib import Path

import torch

from src.utils.wan_activation_disk_capture import artifact_dir, branch_for_call


def test_disk_capture_paths_use_sampling_step_and_branch():
    root = Path("outputs/run")
    assert artifact_dir(root, 20, 3, "ffn_in") == root / "step_010/conditional/block_03/ffn_in"
    assert artifact_dir(root, 21, 3, "ffn_in") == root / "step_010/unconditional/block_03/ffn_in"
    assert branch_for_call(20) == "conditional"
