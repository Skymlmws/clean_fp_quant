"""Compute per-block transform and MXFP4 metrics from saved cross_q activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from scripts.visualize.compare_wan_cross_q_transforms import full_statistics, mxfp4_metrics
from src.quantization.quantizer import Quantizer
from src.transforms.transforms import GivensTransform, HadamardTransform
from src.utils.wan_utils import WAN_LINEAR_TRANSFORM_GROUPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="3,5,8,12,16,24,40")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--transform-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--render-json", type=Path,
        help="Re-render block_metrics.md from an existing result JSON without GPU computation",
    )
    parser.add_argument("--site", help="Site name fallback when re-rendering legacy result JSON")
    return parser.parse_args()


def token_norm_metrics(source: torch.Tensor, transformed: torch.Tensor) -> dict[str, float]:
    source_norm = source.float().reshape(-1, source.shape[-1]).norm(dim=-1)
    transformed_norm = transformed.float().reshape(-1, transformed.shape[-1]).norm(dim=-1)
    error = (transformed_norm - source_norm).abs()
    relative = error / source_norm.clamp_min(1e-12)
    return {
        "max_token_l2_relative_error": float(relative.max()),
        "mean_token_l2_relative_error": float(relative.mean()),
    }


def metrics(
    source: torch.Tensor,
    transformed: torch.Tensor,
    quantizer: Quantizer,
) -> dict[str, Any]:
    return {
        "statistics": full_statistics(transformed),
        "mxfp4": mxfp4_metrics(transformed, quantizer),
        "rotation_validation": token_norm_metrics(source, transformed),
    }


def render_report(result: dict[str, Any]) -> str:
    thresholds = result["thresholds"]
    lines = [
        f"# Wan {result['site']} 全层离线阈值指标",
        "",
        "数据范围：step 10、conditional 分支、全部 30 个 Transformer blocks。",
        f"所有指标由保存的 BF16 {result['site']} 输入激活离线计算，没有重新运行 Wan。",
        "",
        "## 指标与表头说明",
        "",
        "- `Method`：旋转方案。`Givens threshold=t` 表示校准代表向量的组内 `max_abs > t` 时采用 Givens，否则采用 Hadamard；`Randomized fast Hadamard baseline` 为全部 groups 使用 Hadamard。",
        "- `Mean Givens groups/block`：30 个 blocks 中，每层实际采用 Givens 的 32-channel transform groups 的平均数。`self_qkv` 等 1536 维点位每层共有 48 组；`ffn_out` 每层共有 280 组。",
        "- `Max abs`：旋转后该 block 激活矩阵中所有元素绝对值的最大值；`Mean Max abs` 是其在 30 个 blocks 上的平均。越小通常越有利于共享 scale，但不能单独决定量化误差。",
        "- `Max channel RMS / median`：先对每个 channel 跨全部 token 计算 RMS，再以其中最大 RMS 除以中位 RMS。越接近 1，表示 channel 间能量越均匀。",
        "- `MXFP4 MSE`：旋转后激活经 MXFP4 E2M1、每 32 个 channel 共用 E8M0 scale 的 fake quantization 后，原值与重建值的元素均方误差。越低越好。",
        "- `SQNR (dB)`：10 × log10(信号能量 / 量化误差能量)。越高越好。",
        "- `Max token L2 relative error`：旋转前后每个 token 的 32/1536 维向量 L2 范数相对差中的最大值。这是正交旋转的数值正确性检查，不是量化质量指标；越接近 0 越好。",
        "- `Block`：Transformer block 编号，从 0 到 29。逐层表中的数值只对应这一个 block，不是全层平均。",
        "",
        "## 全层平均指标",
        "",
        "| Method | Mean Givens groups/block | Mean Max abs | Mean Max channel RMS / median | Mean MXFP4 MSE | Mean SQNR (dB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in result["aggregate"]:
        lines.append(
            f"| {item['method']} | {item['mean_givens_groups_per_block']:.4f} | "
            f"{item['mean_max_abs']:.6f} | {item['mean_max_channel_rms_over_median']:.6f} | "
            f"{item['mean_mxfp4_mse']:.6f} | {item['mean_mxfp4_sqnr_db']:.6f} |"
        )
    lines.extend(["", "## 逐层结果", ""])
    for threshold in thresholds:
        lines.extend([
            f"### Threshold = {threshold:g}",
            "",
            "| Block | Givens groups | Max abs | Max channel RMS / median | MXFP4 MSE | SQNR (dB) | Max token L2 relative error |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for block in result["blocks"]:
            item = block["givens"][str(threshold)]
            lines.append(
                f"| {block['block']} | {item['givens_groups']} | "
                f"{item['statistics']['max_abs']:.6f} | "
                f"{item['statistics']['max_channel_rms_over_median']:.6f} | "
                f"{item['mxfp4']['mse']:.6f} | {item['mxfp4']['sqnr_db']:.6f} | "
                f"{item['rotation_validation']['max_token_l2_relative_error']:.6f} |"
            )
        lines.append("")
    lines.extend([
        "## Hadamard baseline 逐层结果",
        "",
        "| Block | Max abs | Max channel RMS / median | MXFP4 MSE | SQNR (dB) | Max token L2 relative error |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for block in result["blocks"]:
        item = block["hadamard"]
        lines.append(
            f"| {block['block']} | {item['statistics']['max_abs']:.6f} | "
            f"{item['statistics']['max_channel_rms_over_median']:.6f} | "
            f"{item['mxfp4']['mse']:.6f} | {item['mxfp4']['sqnr_db']:.6f} | "
            f"{item['rotation_validation']['max_token_l2_relative_error']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.render_json is not None:
        result = json.loads(args.render_json.read_text())
        if "site" not in result:
            if args.site is None:
                raise ValueError("Legacy result JSON has no site; pass --site when using --render-json")
            result["site"] = args.site
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "block_metrics.md").write_text(render_report(result))
        print(args.output_dir / "block_metrics.md")
        return
    thresholds = sorted({float(value) for value in args.thresholds.split(",") if value.strip()})
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    artifact = torch.load(
        args.data_dir / manifest["calibration_file"], map_location="cpu", weights_only=True
    )
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
    blocks = []
    for activation_info in manifest["activations"]:
        block_index = int(activation_info["block"])
        source_cpu = torch.load(
            args.data_dir / activation_info["file"], map_location="cpu", weights_only=True, mmap=True
        )
        source = source_cpu.to(device)
        seed = args.transform_seed + block_index * transform_group_count + site_index
        hadamard = HadamardTransform(group_size=group_size, randomize=True, seed=seed).to(device)
        transformed_h = hadamard(source)
        block_result = {
            "block": block_index,
            "hadamard": metrics(source, transformed_h, quantizer),
            "givens": {},
        }
        del transformed_h, hadamard
        calibration = calibration_by_block[block_index]
        for threshold in thresholds:
            transform = GivensTransform(
                size=hidden_size,
                group_size=group_size,
                outlier_threshold=threshold,
                fallback_randomize=True,
                seed=seed,
                device=device,
            ).to(device)
            transform.load_calibration_state(calibration)
            transform.finalize_calibration()
            transformed = transform(source)
            block_result["givens"][str(threshold)] = {
                "givens_groups": transform.givens_blocks,
                "hadamard_groups": transform.hadamard_blocks,
                **metrics(source, transformed, quantizer),
            }
            del transformed, transform
        blocks.append(block_result)
        del source, source_cpu
        torch.cuda.empty_cache()
        print(json.dumps({"completed_block": block_index, "total_blocks": len(manifest["activations"])}))

    aggregate = []
    for threshold in thresholds:
        selected = [block["givens"][str(threshold)] for block in blocks]
        aggregate.append({
            "method": f"Givens threshold={threshold:g}",
            "mean_givens_groups_per_block": sum(item["givens_groups"] for item in selected) / len(selected),
            "mean_max_abs": sum(item["statistics"]["max_abs"] for item in selected) / len(selected),
            "mean_max_channel_rms_over_median": sum(item["statistics"]["max_channel_rms_over_median"] for item in selected) / len(selected),
            "mean_mxfp4_mse": sum(item["mxfp4"]["mse"] for item in selected) / len(selected),
            "mean_mxfp4_sqnr_db": sum(item["mxfp4"]["sqnr_db"] for item in selected) / len(selected),
        })
    selected = [block["hadamard"] for block in blocks]
    aggregate.append({
        "method": "Randomized fast Hadamard baseline",
        "mean_givens_groups_per_block": 0.0,
        "mean_max_abs": sum(item["statistics"]["max_abs"] for item in selected) / len(selected),
        "mean_max_channel_rms_over_median": sum(item["statistics"]["max_channel_rms_over_median"] for item in selected) / len(selected),
        "mean_mxfp4_mse": sum(item["mxfp4"]["mse"] for item in selected) / len(selected),
        "mean_mxfp4_sqnr_db": sum(item["mxfp4"]["sqnr_db"] for item in selected) / len(selected),
    })
    result = {
        "data_dir": str(args.data_dir.resolve()),
        "site": site,
        "device": str(device),
        "thresholds": thresholds,
        "aggregate": aggregate,
        "blocks": blocks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "block_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "block_metrics.md").write_text(render_report(result))
    print(args.output_dir / "block_metrics.md")


if __name__ == "__main__":
    main()
