"""Offline Identity/Hadamard/Givens comparison over saved Wan activations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch

from src.quantization.quantizer import Quantizer
from src.transforms.transforms import GivensTransform, HadamardTransform
from src.utils.wan_utils import WAN_LINEAR_TRANSFORM_GROUPS


VARIANTS = ("identity", "hadamard-h32", "givens-g32")
SITE_ORDER = tuple(WAN_LINEAR_TRANSFORM_GROUPS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=Path, required=True)
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


def discover_groups(root: Path) -> list[tuple[tuple[int, str], list[Path]]]:
    groups: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for path in root.rglob("activation.pt"):
        metadata_path = path.parent / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text())
        groups[(int(metadata["block"]), str(metadata["site"]))].append(path)
    return sorted(
        ((key, sorted(paths)) for key, paths in groups.items()),
        key=lambda item: (item[0][0], SITE_ORDER.index(item[0][1])),
    )


@torch.no_grad()
def full_statistics(value: torch.Tensor) -> dict[str, float | int]:
    matrix = value.reshape(-1, value.shape[-1]).float()
    channel_rms = matrix.square().mean(dim=0).sqrt()
    median_rms = channel_rms.median()
    flattened = matrix.abs().flatten()
    maximum_samples = 1_000_000
    stride = max(1, (flattened.numel() + maximum_samples - 1) // maximum_samples)
    sample = flattened[::stride]
    quantiles = torch.quantile(
        sample,
        torch.tensor([0.5, 0.99, 0.999, 0.9999], device=value.device),
    )
    return {
        "max_abs": float(flattened.max()),
        "median_abs": float(quantiles[0]),
        "p99_abs": float(quantiles[1]),
        "p99.9_abs": float(quantiles[2]),
        "p99.99_abs": float(quantiles[3]),
        "quantile_sample_count": sample.numel(),
        "quantile_sample_stride": stride,
        "max_channel_rms_over_median": (
            float(channel_rms.max() / median_rms) if median_rms else float("inf")
        ),
        "max_rms_channel": int(channel_rms.argmax()),
    }


@torch.no_grad()
def mxfp4_metrics(
    value: torch.Tensor,
    quantizer: Quantizer,
    chunk_rows: int,
) -> dict[str, float]:
    matrix = value.reshape(-1, value.shape[-1])
    error_sum = 0.0
    signal_sum = 0.0
    maximum_error = 0.0
    underflow_count = 0
    nonzero_count = 0
    total = 0
    for chunk in matrix.split(chunk_rows):
        scales, zeros = quantizer.get_quantization_params(chunk)
        reconstructed = quantizer(chunk, scales, zeros)
        error = chunk.float() - reconstructed.float()
        error_sum += float(error.square().sum().double())
        signal_sum += float(chunk.float().square().sum().double())
        maximum_error = max(maximum_error, float(error.abs().max()))
        nonzero = chunk.ne(0)
        underflow_count += int((nonzero & reconstructed.eq(0)).sum())
        nonzero_count += int(nonzero.sum())
        total += chunk.numel()
    mse = error_sum / total
    sqnr = float("inf") if error_sum == 0 else 10.0 * torch.log10(
        torch.tensor(signal_sum / error_sum, dtype=torch.float64)
    ).item()
    return {
        "mse": mse,
        "rmse": mse**0.5,
        "sqnr_db": sqnr,
        "max_abs_error": maximum_error,
        "underflow_fraction_of_nonzero": (
            underflow_count / nonzero_count if nonzero_count else 0.0
        ),
    }


def write_part(
    path: Path,
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    status: str = "running",
) -> None:
    payload = {
        "schema_version": 1,
        "status": status,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "variants": list(VARIANTS),
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def compare(args: argparse.Namespace) -> None:
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    groups = discover_groups(args.activation_dir)
    selected = [
        group for index, group in enumerate(groups)
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
    for group_index, ((block, site), paths) in enumerate(selected, start=1):
        metadata = json.loads((paths[0].parent / "metadata.json").read_text())
        channels = int(metadata["shape"][-1])
        givens = GivensTransform(
            size=channels,
            group_size=args.group_size,
            outlier_threshold=args.outlier_threshold,
            device=device,
        )
        for path in paths:
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

        for path in paths:
            relative = str(path.relative_to(args.activation_dir))
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
        write_part(part_path, args, records)
        print(
            f"shard {args.shard_index}: {group_index}/{len(selected)} "
            f"blocks.{block}.{site}",
            flush=True,
        )

    write_part(part_path, args, records, status="complete")


def merge(output_dir: Path) -> None:
    parts = [json.loads(path.read_text()) for path in sorted(output_dir.glob("part-*.json"))]
    if not parts:
        raise FileNotFoundError(f"No part files found in {output_dir}")
    incomplete = [part["shard_index"] for part in parts if part.get("status") != "complete"]
    records = [record for part in parts for record in part.get("records", [])]
    aggregate: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for site in SITE_ORDER:
        aggregate[site] = {}
        for variant in VARIANTS:
            selected = [
                record for record in records
                if record["site"] == site and record["variant"] == variant
            ]
            aggregate[site][variant] = {
                "sample_count": len(selected),
                "mean_max_channel_rms_over_median": (
                    sum(record["full_statistics"]["max_channel_rms_over_median"] for record in selected) / len(selected)
                    if selected else None
                ),
                "mean_mxfp4_mse": (
                    sum(record["mxfp4"]["mse"] for record in selected) / len(selected)
                    if selected else None
                ),
                "mean_mxfp4_sqnr_db": (
                    sum(record["mxfp4"]["sqnr_db"] for record in selected) / len(selected)
                    if selected else None
                ),
            }
    result = {
        "schema_version": 1,
        "status": "complete" if not incomplete else "incomplete",
        "incomplete_shards": incomplete,
        "variants": list(VARIANTS),
        "record_count": len(records),
        "aggregate": aggregate,
        "records": records,
    }
    (output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "record_count", "incomplete_shards")}, indent=2))


def main() -> None:
    args = parse_args()
    if args.merge_only:
        merge(args.output_dir)
    else:
        compare(args)


if __name__ == "__main__":
    main()
