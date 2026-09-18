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
        "# Wan cross_q 全层离线阈值指标",
        "",
        "数据范围：step 10、conditional 分支、全部 30 个 Transformer blocks。",
        "所有指标由保存的 BF16 cross_q 输入激活离线计算，没有重新运行 Wan。",
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
    cross_q_index = tuple(WAN_LINEAR_TRANSFORM_GROUPS).index("cross_q")
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
        seed = args.transform_seed + block_index * transform_group_count + cross_q_index
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
