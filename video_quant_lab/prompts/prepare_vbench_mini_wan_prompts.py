#!/usr/bin/env python3
"""Align VBench mini metadata with the official Wan augmented prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROMPTS_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mini-metadata",
        type=Path,
        default=PROMPTS_ROOT / "vbench-1.0-mini-0.05/VBench_kmeans_info_0.05.json",
    )
    parser.add_argument(
        "--official-metadata",
        type=Path,
        default=PROMPTS_ROOT / "vbench-official/VBench_full_info.json",
    )
    parser.add_argument(
        "--official-augmented-prompts",
        type=Path,
        default=PROMPTS_ROOT / "vbench-official/all_dimension_aug_wanx_seed42.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROMPTS_ROOT
        / "vbench-1.0-mini-0.05/all_dimension_aug_wanx_seed42_0.05.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mini = json.loads(args.mini_metadata.read_text())
    official = json.loads(args.official_metadata.read_text())
    augmented = args.official_augmented_prompts.read_text().splitlines()
    if len(official) != len(augmented):
        raise ValueError(
            f"Official metadata/prompts must align: {len(official)} != {len(augmented)}"
        )

    indices: dict[str, list[int]] = {}
    for index, record in enumerate(official):
        indices.setdefault(record["prompt_en"], []).append(index)

    selected: list[str] = []
    for record in mini:
        matches = indices.get(record["prompt_en"], [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected one official match for {record['prompt_en']!r}, got {len(matches)}"
            )
        selected.append(augmented[matches[0]])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n")
    print(f"wrote {len(selected)} aligned Wan prompts to {args.output}")


if __name__ == "__main__":
    main()
