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
    parser.add_argument(
        "--render-only", action="store_true",
        help="Re-render block Markdown files and index.md from existing block JSON files",
    )
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


def transform_channel_group_comparison(
    candidate: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    givens_mask: torch.Tensor,
) -> dict[str, Any]:
    """Aggregate MXFP4 errors over all tokens for each fixed 32-channel group."""
    channel_groups = int(givens_mask.numel())
    candidate_mse = candidate["mse"].reshape(-1, channel_groups)
    baseline_mse = baseline["mse"].reshape(-1, channel_groups)
    candidate_mean = candidate_mse.mean(dim=0)
    baseline_mean = baseline_mse.mean(dim=0)
    tolerance = torch.maximum(candidate_mean, baseline_mean).clamp_min(1e-12) * 1e-6
    delta = candidate_mean - baseline_mean
    routed_indices = givens_mask.nonzero(as_tuple=False).flatten()
    groups = []
    for index_tensor in routed_indices:
        index = int(index_tensor)
        token_delta = candidate_mse[:, index] - baseline_mse[:, index]
        token_tolerance = torch.maximum(
            candidate_mse[:, index], baseline_mse[:, index]
        ).clamp_min(1e-12) * 1e-6
        groups.append({
            "channel_group": index,
            "channel_start": index * 32,
            "channel_end": (index + 1) * 32 - 1,
            "givens_mse": float(candidate_mean[index]),
            "hadamard_mse": float(baseline_mean[index]),
            "mse_ratio": float(candidate_mean[index] / baseline_mean[index].clamp_min(1e-12)),
            "token_givens_win_fraction": float((token_delta < -token_tolerance).float().mean()),
            "token_hadamard_win_fraction": float((token_delta > token_tolerance).float().mean()),
        })
    if not groups:
        return {
            "routed_groups": 0, "givens_win_groups": 0,
            "hadamard_win_groups": 0, "tie_groups": 0,
            "routed_mse_ratio": None, "groups": [],
        }
    routed_delta = delta[routed_indices]
    routed_tolerance = tolerance[routed_indices]
    return {
        "routed_groups": len(groups),
        "givens_win_groups": int((routed_delta < -routed_tolerance).sum()),
        "hadamard_win_groups": int((routed_delta > routed_tolerance).sum()),
        "tie_groups": int((routed_delta.abs() <= routed_tolerance).sum()),
        "routed_mse_ratio": float(
            candidate_mean[routed_indices].sum() / baseline_mean[routed_indices].sum().clamp_min(1e-12)
        ),
        "groups": groups,
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
        f"# Block {block['block']:02d}：{block['site']} MXFP4 逐量化组分析", "",
        f"量化组数：{block['quantization_groups']:,}。每组是一个 token 的连续 32 个 channels。", "",
        "`scale` 是实际 E8M0 scale；`max_abs_over_rms` 描述组内尖峰；"
        "`mse` 和 `relative_mse` 越低越好；`underflow_fraction` 是非零值量化为零的比例。", "",
        "## 指标与表头说明", "",
        "- `scale`：该 token × 32-channel MXFP4 quantization group 实际使用的 E8M0 scale。",
        "- `max_abs_over_rms`：组内最大绝对值 ÷ 组 RMS；越小表示尖峰越弱。",
        "- `mse`：组内 MXFP4 fake quantization 后的元素均方误差；`relative_mse` 为组内误差能量 ÷ 原始信号能量。两者越低越好。",
        "- `underflow_fraction`：原本非零、量化后变为零的元素比例；越低越好。",
        "- `MSE ratio G/H`：Givens MSE ÷ Hadamard MSE；小于 1 表示 Givens 更好。",
        "- `Token G win` / `Token H win`：在固定的 transform channel group 内，分别表示 token × 32-channel quantization group 中哪方 MSE 更低的比例。",
        "- `Group` 与 `Channels`：共享同一旋转矩阵的 32-channel transform group 编号及通道范围；其 MSE 已汇总该组的全部 token。",
        "",
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
        channel_comparison = item["transform_channel_groups"]
        ratio = channel_comparison["routed_mse_ratio"]
        lines.extend([
            "#### 逐 transform channel group 净收益", "",
            f"仅统计实际路由到 Givens 的 groups：共 {channel_comparison['routed_groups']} 组，"
            f"Givens 净胜 {channel_comparison['givens_win_groups']} 组，"
            f"Hadamard 净胜 {channel_comparison['hadamard_win_groups']} 组，"
            f"持平 {channel_comparison['tie_groups']} 组，"
            f"汇总 MSE ratio G/H = {fmt(ratio) if ratio is not None else 'n/a'}。", "",
        ])
        if channel_comparison["groups"]:
            lines.extend([
                "| Group | Channels | Givens MSE | Hadamard MSE | MSE ratio G/H | Token G win | Token H win |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for group in channel_comparison["groups"]:
                lines.append(
                    f"| {group['channel_group']} | {group['channel_start']}–{group['channel_end']} | "
                    f"{fmt(group['givens_mse'])} | {fmt(group['hadamard_mse'])} | "
                    f"{fmt(group['mse_ratio'])} | {group['token_givens_win_fraction']:.2%} | "
                    f"{group['token_hadamard_win_fraction']:.2%} |"
                )
            lines.append("")
    return "\n".join(lines)


def render_index(blocks: list[dict[str, Any]], thresholds: list[float]) -> str:
    lines = [
        f"# Wan {blocks[0]['site']} MXFP4 逐量化组分析", "",
        "数据范围：step 10、conditional 分支、全部 30 个 Transformer blocks。", "",
        "每个 block 使用独立文档；分析完全基于保存的 BF16 激活，没有重新运行 Wan。", "",
        "## 指标与表头说明", "",
        "- `MXFP4 quantization group`：一个 token 的连续 32 个 channels；该组 32 个 FP4 值共享一个 E8M0 scale。",
        "- `transform channel group`：所有 token 在同一段连续 32 个 channels 上组成的矩阵；该组所有 token 共用一个 Givens 或 Hadamard 旋转矩阵。",
        "- `Mean scale`：各 MXFP4 quantization group 实际使用的 E8M0 scale 的平均值。scale 小不必然更好，需结合误差和 underflow 一并判断。",
        "- `Mean max_abs / RMS`：每个 MXFP4 quantization group 的最大绝对值除以该组 RMS，再取平均。越小表示组内尖峰越弱、共享 scale 越容易适配。",
        "- `Mean relative MSE`：每组误差能量 / 原始信号能量，再取平均。越低越好。",
        "- `Mean underflow`：原本非零、fake quantization 后变为零的元素占比，再对组取平均。越低越好。",
        "- `Givens win` / `Hadamard win` / `Tie`：针对相同的 MXFP4 quantization group，比对 Givens 混合方案与纯 Hadamard 的组 MSE；前两者分别表示哪方更低，`Tie` 表示数值差在容差内。",
        "- `MSE ratio G/H`：Givens MSE ÷ Hadamard MSE。小于 1 表示 Givens 更好，大于 1 表示 Hadamard 更好。",
        "- `Source bucket`：按旋转前该 quantization group 的 `max_abs / RMS` 分桶：bottom 90% 为普通组，p90–p99 为较强异常组，top 1% 为极端组。",
        "- `Transform channel group 汇总`：只统计实际路由到 Givens 的 32-channel transform groups；`净胜`表示将该 group 的全部 token 误差汇总后，哪种旋转的总 MSE 更低。",
        "",
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
        channel_items = [
            block["givens"][str(threshold)]["transform_channel_groups"]
            for block in blocks
        ]
        routed = sum(item["routed_groups"] for item in channel_items)
        givens_wins = sum(item["givens_win_groups"] for item in channel_items)
        hadamard_wins = sum(item["hadamard_win_groups"] for item in channel_items)
        ties = sum(item["tie_groups"] for item in channel_items)
        routed_givens_mse = sum(
            group["givens_mse"] for item in channel_items for group in item["groups"]
        )
        routed_hadamard_mse = sum(
            group["hadamard_mse"] for item in channel_items for group in item["groups"]
        )
        lines.extend([
            "#### Transform channel group 汇总", "",
            f"实际 Givens groups：{routed}；Givens 净胜：{givens_wins}；"
            f"Hadamard 净胜：{hadamard_wins}；持平：{ties}；"
            f"routed group MSE ratio G/H："
            f"{routed_givens_mse / routed_hadamard_mse:.6f}" if routed else
            "实际 Givens groups：0；无 routed group 可比较。",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    thresholds = sorted({float(value) for value in args.thresholds.split(",") if value.strip()})
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.index_only or args.render_only:
        blocks = [json.loads(path.read_text()) for path in sorted(args.output_dir.glob("block_*.json"))]
        if not blocks:
            raise FileNotFoundError(f"No block JSON files found in {args.output_dir}")
        if args.render_only:
            for block in blocks:
                (args.output_dir / f"block_{block['block']:02d}.md").write_text(render_block(block, thresholds))
        (args.output_dir / "index.md").write_text(render_index(blocks, thresholds))
        print(args.output_dir / "index.md")
        return
    artifact = torch.load(args.data_dir / manifest["calibration_file"], map_location="cpu", weights_only=True)
    calibration_by_block = {int(item["block"]): item for item in artifact["blocks"]}
    device = torch.device(f"cuda:{args.device_id}")
    group_size = int(artifact["group_size"])
    hidden_size = int(artifact["hidden_size"])
    transform_group_count = len(WAN_LINEAR_TRANSFORM_GROUPS)
    site = str(manifest["site"])
    site_index = tuple(WAN_LINEAR_TRANSFORM_GROUPS).index(site)
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
        seed = args.transform_seed + block_index * transform_group_count + site_index
        hadamard = HadamardTransform(group_size=group_size, randomize=True, seed=seed).to(device)
        hadamard_groups = group_tensors(hadamard(source), quantizer)
        block_result: dict[str, Any] = {
            "block": block_index,
            "site": site,
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
                "transform_channel_groups": transform_channel_group_comparison(
                    candidate_groups, hadamard_groups, transform.givens_mask
                ),
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
