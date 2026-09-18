"""Analyze saved Wan cross_q Givens calibration without rerunning the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--thresholds", default="3,5,8,12,16,24,40")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def render_markdown(result: dict) -> str:
    lines = [
        "# Wan cross_q 离线路由分析",
        "",
        "数据来自一次完整 BF16 校准；无需重新运行模型即可计算任意 threshold 的路由。",
        "",
        "## 全模型统计",
        "",
        "| Threshold | Givens groups | Hadamard groups |",
        "|---:|---:|---:|",
    ]
    for item in result["threshold_summary"]:
        lines.append(
            f"| {item['threshold']:g} | {item['givens_groups']} | {item['hadamard_groups']} |"
        )
    lines.extend([
        "",
        "## 逐层 Givens group 数量",
        "",
        "| Block | " + " | ".join(f"t={value:g}" for value in result["thresholds"]) + " |",
        "|---:|" + "---:|" * len(result["thresholds"]),
    ])
    for block in result["blocks"]:
        lines.append(
            f"| {block['block']} | "
            + " | ".join(str(block["givens_groups"][str(value)]) for value in result["thresholds"])
            + " |"
        )
    lines.extend(["", "## 相邻阈值之间切换为 Hadamard 的 groups", ""])
    for transition in result["transitions"]:
        lines.append(
            f"### {transition['lower']:g} → {transition['upper']:g}："
            f"{transition['switched_group_count']} groups"
        )
        lines.append("")
        if not transition["groups"]:
            lines.append("无。")
        else:
            for group in transition["groups"]:
                lines.append(
                    f"- block {group['block']}, group {group['group']}, "
                    f"channels {group['channel_start']}–{group['channel_end']}, "
                    f"calibration max = {group['calibration_max']:.6f}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    thresholds = sorted({float(value) for value in args.thresholds.split(",") if value.strip()})
    artifact = torch.load(args.calibration, map_location="cpu", weights_only=True)
    group_size = int(artifact["group_size"])
    blocks = []
    for block in artifact["blocks"]:
        maxima = block["group_max_abs"].float()
        blocks.append({
            "block": int(block["block"]),
            "group_max_abs": maxima.tolist(),
            "givens_groups": {str(value): int((maxima > value).sum()) for value in thresholds},
        })
    total_groups = sum(len(block["group_max_abs"]) for block in blocks)
    threshold_summary = []
    for threshold in thresholds:
        givens = sum(block["givens_groups"][str(threshold)] for block in blocks)
        threshold_summary.append({
            "threshold": threshold,
            "givens_groups": givens,
            "hadamard_groups": total_groups - givens,
        })
    transitions = []
    for lower, upper in zip(thresholds, thresholds[1:]):
        groups = []
        for block in blocks:
            for group_index, maximum in enumerate(block["group_max_abs"]):
                if lower < maximum <= upper:
                    groups.append({
                        "block": block["block"],
                        "group": group_index,
                        "channel_start": group_index * group_size,
                        "channel_end": (group_index + 1) * group_size - 1,
                        "calibration_max": maximum,
                    })
        transitions.append({
            "lower": lower,
            "upper": upper,
            "switched_group_count": len(groups),
            "groups": groups,
        })
    result = {
        "calibration": str(args.calibration.resolve()),
        "group_size": group_size,
        "thresholds": thresholds,
        "total_groups": total_groups,
        "threshold_summary": threshold_summary,
        "blocks": blocks,
        "transitions": transitions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "route_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "route_analysis.md").write_text(render_markdown(result))
    print(args.output_dir / "route_analysis.md")


if __name__ == "__main__":
    main()
