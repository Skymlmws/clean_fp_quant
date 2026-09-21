"""Build one comparison report from per-site Wan offline analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", action="append", required=True, help="SITE=DATA_DIR")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def weighted_bucket(blocks: list[dict], threshold: float, bucket_name: str) -> dict[str, float]:
    items = [
        block["givens"][str(threshold)]["versus_hadamard"]["buckets"][bucket_name]
        for block in blocks
    ]
    groups = sum(item["groups"] for item in items)
    givens_mse = sum(item["mean_givens_mse"] * item["groups"] for item in items)
    hadamard_mse = sum(item["mean_hadamard_mse"] * item["groups"] for item in items)
    return {
        "groups": groups,
        "givens_win": sum(item["givens_win_fraction"] * item["groups"] for item in items) / groups,
        "hadamard_win": sum(item["hadamard_win_fraction"] * item["groups"] for item in items) / groups,
        "mse_ratio": givens_mse / hadamard_mse,
    }


def main() -> None:
    args = parse_args()
    site_dirs = {}
    for value in args.site:
        site, raw_path = value.split("=", 1)
        site_dirs[site] = Path(raw_path)
    results = []
    for site, data_dir in site_dirs.items():
        block_metrics = json.loads((data_dir / "analysis/block-metrics/block_metrics.json").read_text())
        group_blocks = [
            json.loads(path.read_text())
            for path in sorted((data_dir / "analysis/mxfp4-groups").glob("block_*.json"))
        ]
        baseline = next(item for item in block_metrics["aggregate"] if "Hadamard baseline" in item["method"])
        for threshold in block_metrics["thresholds"]:
            aggregate = next(
                item for item in block_metrics["aggregate"]
                if item["method"] == f"Givens threshold={threshold:g}"
            )
            all_groups = weighted_bucket(group_blocks, threshold, "all")
            extreme = weighted_bucket(group_blocks, threshold, "extreme_top_1pct")
            channel_items = [
                block["givens"][str(threshold)]["transform_channel_groups"]
                for block in group_blocks
            ]
            routed_groups = sum(item["routed_groups"] for item in channel_items)
            routed_givens_mse = sum(
                group["givens_mse"] for item in channel_items for group in item["groups"]
            )
            routed_hadamard_mse = sum(
                group["hadamard_mse"] for item in channel_items for group in item["groups"]
            )
            results.append({
                "site": site,
                "threshold": threshold,
                "givens_groups_per_block": aggregate["mean_givens_groups_per_block"],
                "hadamard_mse": baseline["mean_mxfp4_mse"],
                "block_mse_ratio": aggregate["mean_mxfp4_mse"] / baseline["mean_mxfp4_mse"],
                "sqnr_delta_db": aggregate["mean_mxfp4_sqnr_db"] - baseline["mean_mxfp4_sqnr_db"],
                "all_group_mse_ratio": all_groups["mse_ratio"],
                "extreme_mse_ratio": extreme["mse_ratio"],
                "extreme_givens_win": extreme["givens_win"],
                "extreme_hadamard_win": extreme["hadamard_win"],
                "routed_groups": routed_groups,
                "channel_givens_wins": sum(item["givens_win_groups"] for item in channel_items),
                "channel_hadamard_wins": sum(item["hadamard_win_groups"] for item in channel_items),
                "routed_channel_mse_ratio": (
                    routed_givens_mse / routed_hadamard_mse if routed_groups else None
                ),
            })
    lines = [
        "# Wan 全量化点位：Givens 与 Hadamard 离线对比", "",
        "范围：autumn_station、seed 0、step 10、conditional、全部 30 个 Transformer blocks。", "",
        "`MSE ratio G/H` 小于 1 表示 Givens 更好，大于 1 表示 Hadamard 更好；"
        "`SQNR delta` 大于 0 表示 Givens 更好。极端组是按旋转前单个 MXFP4 组的 `max_abs/RMS` 定义的 top 1%。", "",
        "## 指标与表头说明", "",
        "- `Site`：量化点位；例如 `self_qkv` 表示 self-attention 的 Q/K/V 共享输入旋转，`ffn_out` 表示 FFN 第二个 Linear 的输入旋转。",
        "- `Threshold`：校准代表向量在一个 32-channel group 内的 `max_abs` 触发阈值。大于该阈值时该 transform channel group 路由到 Givens，否则使用 Hadamard。",
        "- `Routed channel groups`：30 个 blocks 合计实际路由到 Givens 的 32-channel transform groups 数。它不是 MXFP4 quantization group 数；后者是逐 token 的。",
        "- `Channel G wins` / `Channel H wins`：在 routed channel groups 中，分别统计把该 group 的所有 token 的 MXFP4 误差汇总后，Givens 或 Hadamard 总 MSE 更低的 group 数。",
        "- `Routed channel MSE G/H`：只在 routed channel groups 上计算的总 MSE 比值，即 Givens ÷ Hadamard。小于 1 表示 Givens 在被选中的 groups 上整体更好；`n/a` 表示没有 group 被路由到 Givens。",
        "- `Block MSE G/H`：该方案在 30 个 blocks 的平均 MXFP4 MSE ÷ 纯 Hadamard 的对应平均 MSE。小于 1 表示该方案整体更好。",
        "- `SQNR delta (dB)`：该方案的平均 SQNR − 纯 Hadamard 的平均 SQNR。大于 0 表示该方案整体更好。",
        "- `Extreme MSE G/H`：原始 `max_abs / RMS` 位于 top 1% 的 MXFP4 quantization groups 上，Givens MSE ÷ Hadamard MSE。小于 1 表示 Givens 更适合极端组。",
        "- `Hadamard MXFP4 MSE`：纯 Hadamard baseline 的 30-block 平均元素级 MXFP4 MSE，用于了解各 site 的绝对误差尺度；跨 site 比较时更应优先看比值而非绝对 MSE。",
        "",
        "## Hadamard baseline", "",
        "| Site | Hadamard MXFP4 MSE |", "|---|---:|",
    ]
    for site in site_dirs:
        item = next(row for row in results if row["site"] == site)
        lines.append(f"| {site} | {item['hadamard_mse']:.6g} |")
    lines.extend([
        "", "## 全部 threshold", "",
        "| Site | Threshold | Routed channel groups | Channel G wins | Channel H wins | Routed channel MSE G/H | Block MSE G/H | SQNR delta (dB) | Extreme MSE G/H |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in results:
        if row["routed_channel_mse_ratio"] is None:
            prefix = f"| {row['site']} | {row['threshold']:g} | 0 | 0 | 0 | n/a | "
        else:
            prefix = (
                f"| {row['site']} | {row['threshold']:g} | {row['routed_groups']} | "
                f"{row['channel_givens_wins']} | {row['channel_hadamard_wins']} | "
                f"{row['routed_channel_mse_ratio']:.4f} | "
            )
        lines.append(prefix + (
            f"{row['block_mse_ratio']:.4f} | {row['sqnr_delta_db']:.4f} | "
            f"{row['extreme_mse_ratio']:.4f} |"
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
