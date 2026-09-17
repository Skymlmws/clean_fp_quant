import json
from pathlib import Path

from video_quant_lab.analysis.cli.consolidate_wan_activation_shards import consolidate


def _write_shard(root: Path, shard: int, blocks: list[int]) -> None:
    shard_dir = root / "shards" / f"shard-{shard}"
    count = 0
    for block in blocks:
        site = shard_dir / "step_010" / "conditional" / f"block_{block:02d}" / "cross_q"
        site.mkdir(parents=True)
        (site / "activation.pt").write_bytes(b"activation")
        count += 1
    config = {"blocks": blocks, "max_output_gb": 1.0, "sampling_steps": [10]}
    state = {
        "status": "complete",
        "completed_this_run": count,
        "skipped_existing": 0,
        "expected": count,
        "bytes_written_this_run": count * 10,
        "error": None,
    }
    (shard_dir / "config.json").write_text(json.dumps(config))
    (shard_dir / "state.json").write_text(json.dumps(state))


def test_consolidate_flattens_disjoint_shards_and_merges_state(tmp_path):
    _write_shard(tmp_path, 0, [0, 1])
    _write_shard(tmp_path, 1, [2])

    result = consolidate(tmp_path, expected_activations=3)

    assert result == {
        "status": "complete",
        "activations": 3,
        "blocks": [0, 1, 2],
        "shards": 2,
    }
    assert not (tmp_path / "shards").exists()
    assert (tmp_path / "step_010/conditional/block_00/cross_q/activation.pt").is_file()
    assert (tmp_path / "step_010/conditional/block_02/cross_q/activation.pt").is_file()
    assert json.loads((tmp_path / "state.json").read_text())["expected"] == 3
    assert json.loads((tmp_path / "config.json").read_text())["blocks"] == [0, 1, 2]


def test_consolidate_is_idempotent_for_flat_dataset(tmp_path):
    _write_shard(tmp_path, 0, [0])
    consolidate(tmp_path, expected_activations=1)

    assert consolidate(tmp_path, expected_activations=1) == {
        "status": "already_flat",
        "activations": 1,
    }
