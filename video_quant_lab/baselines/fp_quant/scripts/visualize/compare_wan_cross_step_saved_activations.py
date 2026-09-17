"""Calibrate transforms on one saved Wan dataset and evaluate on another."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.quantization.quantizer import Quantizer
from src.transforms.transforms import GivensTransform, HadamardTransform
from src.utils.wan_utils import WAN_LINEAR_TRANSFORM_GROUPS
from scripts.visualize.compare_wan_saved_activations import (
    SITE_ORDER,
    VARIANTS,
    discover_groups,
    full_statistics,
    merge,
    mxfp4_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--outlier-threshold", type=float, default=5.0)
    parser.add_argument("--transform-seed", type=int, default=0)
    parser.add_argument("--quant-chunk-rows", type=int, default=2048)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def write_part(
    path: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    status: str,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "cross-step calibration and evaluation",
                "status": status,
                "calibration_dir": str(args.calibration_dir),
                "evaluation_dir": str(args.evaluation_dir),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "variants": list(VARIANTS),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


@torch.no_grad()
def compare(args: argparse.Namespace) -> None:
    calibration = dict(discover_groups(args.calibration_dir))
    evaluation_groups = discover_groups(args.evaluation_dir)
    missing = [key for key, _ in evaluation_groups if key not in calibration]
    if missing:
        raise ValueError(f"Missing calibration groups: {missing}")
    selected = [
        group for index, group in enumerate(evaluation_groups)
        if index % args.num_shards == args.shard_index
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    part_path = args.output_dir / f"part-{args.shard_index}.json"
    records: list[dict[str, Any]] = []
    if part_path.exists():
        records = list(json.loads(part_path.read_text()).get("records", []))
    completed = {
        (record["relative_path"], record["variant"])
        for record in records
    }

    device = torch.device(f"cuda:{args.device_id}")
    quantizer = Quantizer(
        bits=4,
        symmetric=True,
        format="mxfp",
        granularity="group",
        group_size=args.group_size,
        observer="minmax",
        scale_precision="e8m0",
    )
    for group_index, ((block, site), evaluation_paths) in enumerate(selected, start=1):
        calibration_paths = calibration[(block, site)]
        metadata = json.loads((calibration_paths[0].parent / "metadata.json").read_text())
        channels = int(metadata["shape"][-1])
        givens = GivensTransform(
            size=channels,
            group_size=args.group_size,
            outlier_threshold=args.outlier_threshold,
            device=device,
        )
        for path in calibration_paths:
            value = torch.load(path, map_location="cpu", weights_only=True).to(device)
            givens.observe(value)
            del value
        givens.finalize_calibration()
        site_seed = args.transform_seed + block * len(SITE_ORDER) + SITE_ORDER.index(site)
        hadamard = HadamardTransform(
            group_size=args.group_size,
            randomize=True,
            seed=site_seed,
        ).to(device)

        for path in evaluation_paths:
            relative = str(path.relative_to(args.evaluation_dir))
            source = torch.load(path, map_location="cpu", weights_only=True).to(device)
            variants = {
                "identity": source,
                "hadamard-h32": hadamard(source),
                "givens-g32": givens(source),
            }
            source_metadata = json.loads((path.parent / "metadata.json").read_text())
            for variant, value in variants.items():
                if (relative, variant) in completed:
                    continue
                records.append(
                    {
                        "relative_path": relative,
                        "calibration_paths": [
                            str(item.relative_to(args.calibration_dir))
                            for item in calibration_paths
                        ],
                        "sampling_step": source_metadata["sampling_step"],
                        "branch": source_metadata["branch"],
                        "block": block,
                        "site": site,
                        "shape": source_metadata["shape"],
                        "variant": variant,
                        "full_statistics": full_statistics(value),
                        "mxfp4": mxfp4_metrics(value, quantizer, args.quant_chunk_rows),
                        "transform": {
                            "group_size": args.group_size,
                            "seed": site_seed if variant == "hadamard-h32" else None,
                            "givens_blocks": givens.givens_blocks if variant == "givens-g32" else None,
                            "hadamard_blocks": givens.hadamard_blocks if variant == "givens-g32" else None,
                            "observed_abs_max": givens.observed_abs_max if variant == "givens-g32" else None,
                        },
                    }
                )
            del variants, source
            torch.cuda.empty_cache()
        write_part(part_path, args, records, "running")
        print(
            f"shard {args.shard_index}: {group_index}/{len(selected)} "
            f"blocks.{block}.{site}",
            flush=True,
        )
    write_part(part_path, args, records, "complete")


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge(args.output_dir)
    else:
        compare(args)


if __name__ == "__main__":
    main()
