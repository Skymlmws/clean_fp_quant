"""Flatten completed multi-GPU Wan activation shards into one dataset layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-activations", type=int, default=210)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def consolidate(dataset_dir: Path, expected_activations: int = 210) -> dict[str, Any]:
    shard_root = dataset_dir / "shards"
    shard_dirs = sorted(path for path in shard_root.glob("shard-*") if path.is_dir())
    existing = list(dataset_dir.glob("step_*/conditional/block_*/**/activation.pt"))
    if not shard_dirs:
        if len(existing) != expected_activations:
            raise ValueError(
                f"No shards found and flat dataset has {len(existing)} activations; "
                f"expected {expected_activations}"
            )
        return {"status": "already_flat", "activations": len(existing)}

    shard_records = []
    seen_blocks: set[int] = set()
    total_files = 0
    for shard_dir in shard_dirs:
        config_path = shard_dir / "config.json"
        state_path = shard_dir / "state.json"
        if not config_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(f"Missing config/state in {shard_dir}")
        config = _load(config_path)
        state = _load(state_path)
        if state.get("status") != "complete":
            raise ValueError(f"Shard is not complete: {shard_dir}")
        blocks = {int(block) for block in config["blocks"]}
        overlap = seen_blocks & blocks
        if overlap:
            raise ValueError(f"Overlapping blocks in {shard_dir}: {sorted(overlap)}")
        seen_blocks.update(blocks)
        activation_files = list(shard_dir.glob("step_*/conditional/block_*/**/activation.pt"))
        if len(activation_files) != int(state["expected"]):
            raise ValueError(
                f"{shard_dir}: found {len(activation_files)} activations, "
                f"state expects {state['expected']}"
            )
        total_files += len(activation_files)
        shard_records.append(
            {
                "name": shard_dir.name,
                "blocks": sorted(blocks),
                "state": state,
            }
        )

    if total_files != expected_activations:
        raise ValueError(f"Found {total_files} sharded activations; expected {expected_activations}")

    block_dirs = [
        block_dir
        for shard_dir in shard_dirs
        for block_dir in shard_dir.glob("step_*/conditional/block_*")
    ]
    for block_dir in block_dirs:
        step_name = block_dir.parents[1].name
        destination = dataset_dir / step_name / "conditional" / block_dir.name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")

    for block_dir in block_dirs:
        step_name = block_dir.parents[1].name
        destination_parent = dataset_dir / step_name / "conditional"
        destination_parent.mkdir(parents=True, exist_ok=True)
        block_dir.rename(destination_parent / block_dir.name)

    first_config = _load(shard_dirs[0] / "config.json")
    first_config["blocks"] = sorted(seen_blocks)
    first_config["capture_topology"] = {
        "mode": "multi_gpu_block_shards_consolidated",
        "shards": shard_records,
    }
    first_config["max_output_gb"] = sum(
        float(_load(shard_dir / "config.json")["max_output_gb"])
        for shard_dir in shard_dirs
    )
    merged_state = {
        "status": "complete",
        "completed_this_run": sum(int(item["state"]["completed_this_run"]) for item in shard_records),
        "skipped_existing": sum(int(item["state"]["skipped_existing"]) for item in shard_records),
        "expected": expected_activations,
        "bytes_written_this_run": sum(int(item["state"]["bytes_written_this_run"]) for item in shard_records),
        "error": None,
        "capture_shards": len(shard_records),
    }
    (dataset_dir / "config.json").write_text(json.dumps(first_config, indent=2) + "\n")
    (dataset_dir / "state.json").write_text(json.dumps(merged_state, indent=2) + "\n")

    for shard_dir in shard_dirs:
        for path in sorted(shard_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        shard_dir.rmdir()
    shard_root.rmdir()

    flattened = list(dataset_dir.glob("step_*/conditional/block_*/**/activation.pt"))
    if len(flattened) != expected_activations:
        raise RuntimeError(
            f"Consolidation produced {len(flattened)} activations; expected {expected_activations}"
        )
    return {
        "status": "complete",
        "activations": len(flattened),
        "blocks": sorted(seen_blocks),
        "shards": len(shard_records),
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(consolidate(args.dataset_dir, args.expected_activations), indent=2))


if __name__ == "__main__":
    main()
