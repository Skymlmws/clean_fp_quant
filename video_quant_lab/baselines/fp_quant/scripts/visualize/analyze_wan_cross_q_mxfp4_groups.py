"""Compare Hadamard and Givens at the individual MXFP4-group level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.quantization.quantizer import Quantizer
from src.transforms.transforms import GivensTransform, HadamardTransform
from src.utils.wan_utils import WAN_LINEAR_TRANSFORM_GROUPS


PERCENTILES = (0.0, 0.5, 0.9, 0.99, 0.999, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="3,5,8,12,16,24,40")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--transform-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", default="", help="Comma-separated block indices; empty means all blocks")
    parser.add_argument("--index-only", action="store_true", help="Build index.md from existing block JSON files")
    return parser.parse_args()


def distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten()
    quantiles = torch.quantile(values, torch.tensor(PERCENTILES, device=values.device))
    return {
        "mean": float(values.mean()),
        "p0": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "p99_9": float(quantiles[4]),
        "max": float(quantiles[5]),
    }


def group_tensors(value: torch.Tensor, quantizer: Quantizer) -> dict[str, torch.Tensor]:
    groups = value.float().reshape(-1, 32)
    scales, zeros = quantizer.get_quantization_params(value)
    reconstructed = quantizer(value, scales, zeros).float().reshape(-1, 32)
    error_sq = (groups - reconstructed).square()
    energy = groups.square().sum(dim=-1)
    mse = error_sq.mean(dim=-1)
    relative_mse = error_sq.sum(dim=-1) / energy.clamp_min(1e-12)
    nonzero = groups.ne(0)
    underflow = (nonzero & reconstructed.eq(0)).sum(dim=-1) / nonzero.sum(dim=-1).clamp_min(1)
    rms = groups.square().mean(dim=-1).sqrt()
    max_abs = groups.abs().amax(dim=-1)
    return {
        "scale": scales.float().reshape(-1),
        "max_abs_over_rms": max_abs / rms.clamp_min(1e-12),
        "mse": mse,
        "relative_mse": relative_mse,
        "underflow_fraction": underflow.float(),
    }


def summarize_groups(groups: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {name: distribution(values) for name, values in groups.items()}


def paired_comparison(
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    source_outlier_score: torch.Tensor,
) -> dict[str, Any]:
    candidate_error = candidate["mse"]
    baseline_error = baseline["mse"]
    tolerance = torch.maximum(candidate_error, baseline_error).clamp_min(1e-12) * 1e-6
    delta = candidate_error - baseline_error
    boundaries = torch.quantile(
        source_outlier_score.float(),
        torch.tensor((0.9, 0.99), device=source_outlier_score.device),
    )
    masks = {
        "all": torch.ones_like(source_outlier_score, dtype=torch.bool),
        "normal_bottom_90pct": source_outlier_score < boundaries[0],
        "elevated_p90_to_p99": (source_outlier_score >= boundaries[0]) & (source_outlier_score < boundaries[1]),
        "extreme_top_1pct": source_outlier_score >= boundaries[1],
    }
    buckets = {}
    for name, mask in masks.items():
        selected_delta = delta[mask]
        selected_candidate = candidate_error[mask]
        selected_baseline = baseline_error[mask]
        selected_tolerance = tolerance[mask]
        buckets[name] = {
            "groups": int(mask.sum()),
            "givens_win_fraction": float((selected_delta < -selected_tolerance).float().mean()),
            "hadamard_win_fraction": float((selected_delta > selected_tolerance).float().mean()),
            "tie_fraction": float((selected_delta.abs() <= selected_tolerance).float().mean()),
            "mean_givens_mse": float(selected_candidate.mean()),
            "mean_hadamard_mse": float(selected_baseline.mean()),
            "mean_mse_delta": float(selected_delta.mean()),
            "mean_mse_ratio": float(selected_candidate.mean() / selected_baseline.mean().clamp_min(1e-12)),
        }
    return {
        "source_outlier_score_p90": float(boundaries[0]),
        "source_outlier_score_p99": float(boundaries[1]),
        "buckets": buckets,
    }


def fmt(value: float) -> str:
    return f"{value:.6g}"


def distribution_table(methods: dict[str, dict[str, Any]], metric: str) -> list[str]:
    lines = [
        f"### {metric}", "",
        "| Method | Mean | p50 | p90 | p99 | p99.9 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in methods.items():
        item = result[metric]
        lines.append(
            f"| {method} | {fmt(item['mean'])} | {fmt(item['p50'])} | {fmt(item['p90'])} | "
            f"{fmt(item['p99'])} | {fmt(item['p99_9'])} | {fmt(item['max'])} |"
        )
    lines.append("")
    return lines


def render_block(block: dict[str, Any], thresholds: list[float]) -> str:
    methods = {"Hadamard": block["hadamard"]["distributions"]}
    methods.update({
        f"Givens t={threshold:g}": block["givens"][str(threshold)]["distributions"]
        for threshold in thresholds
    })
    lines = [
        f"# Block {block['block']:02d}：cross_q MXFP4 逐量化组分析", "",
        f"量化组数：{block['quantization_groups']:,}。每组是一个 token 的连续 32 个 channels。", "",
        "`scale` 是实际 E8M0 scale；`max_abs_over_rms` 描述组内尖峰；"
        "`mse` 和 `relative_mse` 越低越好；`underflow_fraction` 是非零值量化为零的比例。", "",
        "## 指标分布", "",
    ]
    for metric in ("scale", "max_abs_over_rms", "mse", "relative_mse", "underflow_fraction"):
        lines.extend(distribution_table(methods, metric))
    lines.extend([
        "## Givens 与 Hadamard 逐组配对", "",
        "分桶依据是旋转前同一量化组的 `max_abs / RMS`：bottom 90% 为普通组，p90–p99 为较强异常组，top 1% 为极端组。", "",
    ])
    for threshold in thresholds:
        item = block["givens"][str(threshold)]
        comparison = item["versus_hadamard"]
        lines.extend([
            f"### Threshold = {threshold:g}", "",
            f"路由：Givens transform groups = {item['givens_transform_groups']}，"
            f"Hadamard transform groups = {item['hadamard_transform_groups']}。", "",
            "| Source bucket | Groups | Givens win | Hadamard win | Tie | Givens MSE | Hadamard MSE | MSE ratio G/H |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for bucket_name, bucket in comparison["buckets"].items():
            lines.append(
                f"| {bucket_name} | {bucket['groups']:,} | {bucket['givens_win_fraction']:.2%} | "
                f"{bucket['hadamard_win_fraction']:.2%} | {bucket['tie_fraction']:.2%} | "
                f"{fmt(bucket['mean_givens_mse'])} | {fmt(bucket['mean_hadamard_mse'])} | "
                f"{fmt(bucket['mean_mse_ratio'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_index(blocks: list[dict[str, Any]], thresholds: list[float]) -> str:
    lines = [
        "# Wan cross_q MXFP4 逐量化组分析", "",
        "数据范围：step 10、conditional 分支、全部 30 个 Transformer blocks。", "",
        "每个 block 使用独立文档；分析完全基于保存的 BF16 激活，没有重新运行 Wan。", "",
        "## Block reports", "",
    ]
    for block in blocks:
        lines.append(f"- [Block {block['block']:02d}](block_{block['block']:02d}.md)")
    lines.extend([
        "", "## 全层平均逐组指标", "",
        "这些数值按每个 block 的量化组数加权；当前各 block 的组数相同。", "",
        "| Method | Mean scale | Mean max_abs / RMS | Mean relative MSE | Mean underflow |",
        "|---|---:|---:|---:|---:|",
    ])
    method_items = [("Hadamard", None)] + [
        (f"Givens t={threshold:g}", str(threshold)) for threshold in thresholds
    ]
    for method, threshold_key in method_items:
        weighted = {}
        total_groups = sum(block["quantization_groups"] for block in blocks)
        for metric in ("scale", "max_abs_over_rms", "relative_mse", "underflow_fraction"):
            weighted[metric] = sum(
                block["quantization_groups"] * (
                    block["hadamard"]["distributions"][metric]["mean"]
                    if threshold_key is None else
                    block["givens"][threshold_key]["distributions"][metric]["mean"]
                )
                for block in blocks
            ) / total_groups
        lines.append(
            f"| {method} | {fmt(weighted['scale'])} | {fmt(weighted['max_abs_over_rms'])} | "
            f"{fmt(weighted['relative_mse'])} | {fmt(weighted['underflow_fraction'])} |"
        )
    lines.extend(["", "## 全层逐组配对摘要", ""])
    for threshold in thresholds:
        totals = {name: 0 for name in ("groups", "givens_wins", "hadamard_wins", "ties")}
        givens_error_sum = 0.0
        hadamard_error_sum = 0.0
        for block in blocks:
            bucket = block["givens"][str(threshold)]["versus_hadamard"]["buckets"]["all"]
            count = bucket["groups"]
            totals["groups"] += count
            totals["givens_wins"] += round(bucket["givens_win_fraction"] * count)
            totals["hadamard_wins"] += round(bucket["hadamard_win_fraction"] * count)
            totals["ties"] += round(bucket["tie_fraction"] * count)
            givens_error_sum += bucket["mean_givens_mse"] * count
            hadamard_error_sum += bucket["mean_hadamard_mse"] * count
        lines.extend([
            f"### Threshold = {threshold:g}", "",
            f"- Givens win：{totals['givens_wins'] / totals['groups']:.2%}",
            f"- Hadamard win：{totals['hadamard_wins'] / totals['groups']:.2%}",
            f"- Tie：{totals['ties'] / totals['groups']:.2%}",
            f"- Global group-element MSE ratio G/H：{givens_error_sum / hadamard_error_sum:.6f}", "",
            "| Source bucket | Groups | Givens win | Hadamard win | Tie | MSE ratio G/H |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for bucket_name in ("normal_bottom_90pct", "elevated_p90_to_p99", "extreme_top_1pct"):
            bucket_items = [
                block["givens"][str(threshold)]["versus_hadamard"]["buckets"][bucket_name]
                for block in blocks
            ]
            bucket_groups = sum(item["groups"] for item in bucket_items)
            weighted_value = lambda key: sum(
                item[key] * item["groups"] for item in bucket_items
            ) / bucket_groups
            bucket_givens_mse = sum(
                item["mean_givens_mse"] * item["groups"] for item in bucket_items
            )
            bucket_hadamard_mse = sum(
                item["mean_hadamard_mse"] * item["groups"] for item in bucket_items
            )
            lines.append(
                f"| {bucket_name} | {bucket_groups:,} | "
                f"{weighted_value('givens_win_fraction'):.2%} | "
                f"{weighted_value('hadamard_win_fraction'):.2%} | "
                f"{weighted_value('tie_fraction'):.2%} | "
                f"{bucket_givens_mse / bucket_hadamard_mse:.6f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    thresholds = sorted({float(value) for value in args.thresholds.split(",") if value.strip()})
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        blocks = [json.loads(path.read_text()) for path in sorted(args.output_dir.glob("block_*.json"))]
        if not blocks:
            raise FileNotFoundError(f"No block JSON files found in {args.output_dir}")
        (args.output_dir / "index.md").write_text(render_index(blocks, thresholds))
        print(args.output_dir / "index.md")
        return
    artifact = torch.load(args.data_dir / manifest["calibration_file"], map_location="cpu", weights_only=True)
    calibration_by_block = {int(item["block"]): item for item in artifact["blocks"]}
    device = torch.device(f"cuda:{args.device_id}")
    group_size = int(artifact["group_size"])
    hidden_size = int(artifact["hidden_size"])
    transform_group_count = len(WAN_LINEAR_TRANSFORM_GROUPS)
    cross_q_index = tuple(WAN_LINEAR_TRANSFORM_GROUPS).index("cross_q")
    quantizer = Quantizer(
        bits=4, symmetric=True, format="mxfp", granularity="group",
        group_size=32, observer="minmax", scale_precision="e8m0",
    )
    selected_blocks = (
        {int(value) for value in args.blocks.split(",") if value.strip()}
        if args.blocks else None
    )
    blocks = []
    for activation_info in manifest["activations"]:
        block_index = int(activation_info["block"])
        if selected_blocks is not None and block_index not in selected_blocks:
            continue
        source_cpu = torch.load(args.data_dir / activation_info["file"], map_location="cpu", weights_only=True, mmap=True)
        source = source_cpu.to(device)
        source_groups = source.float().reshape(-1, 32)
        source_score = source_groups.abs().amax(dim=-1) / source_groups.square().mean(dim=-1).sqrt().clamp_min(1e-12)
        seed = args.transform_seed + block_index * transform_group_count + cross_q_index
        hadamard = HadamardTransform(group_size=group_size, randomize=True, seed=seed).to(device)
        hadamard_groups = group_tensors(hadamard(source), quantizer)
        block_result: dict[str, Any] = {
            "block": block_index,
            "quantization_groups": int(source_groups.shape[0]),
            "hadamard": {"distributions": summarize_groups(hadamard_groups)},
            "givens": {},
        }
        calibration = calibration_by_block[block_index]
        for threshold in thresholds:
            transform = GivensTransform(
                size=hidden_size, group_size=group_size, outlier_threshold=threshold,
                fallback_randomize=True, seed=seed, device=device,
            ).to(device)
            transform.load_calibration_state(calibration)
            transform.finalize_calibration()
            candidate_groups = group_tensors(transform(source), quantizer)
            block_result["givens"][str(threshold)] = {
                "givens_transform_groups": transform.givens_blocks,
                "hadamard_transform_groups": transform.hadamard_blocks,
                "distributions": summarize_groups(candidate_groups),
                "versus_hadamard": paired_comparison(candidate_groups, hadamard_groups, source_score),
            }
            del transform, candidate_groups
        blocks.append(block_result)
        (args.output_dir / f"block_{block_index:02d}.md").write_text(render_block(block_result, thresholds))
        (args.output_dir / f"block_{block_index:02d}.json").write_text(json.dumps(block_result, indent=2) + "\n")
        del source, source_cpu, source_groups, source_score, hadamard, hadamard_groups
        torch.cuda.empty_cache()
        print(json.dumps({"completed_block": block_index, "total_blocks": len(manifest["activations"])}), flush=True)
    if selected_blocks is None:
        (args.output_dir / "index.md").write_text(render_index(blocks, thresholds))
        print(args.output_dir / "index.md")


if __name__ == "__main__":
    main()
