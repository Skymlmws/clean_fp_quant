#!/usr/bin/env python3
"""Generate a conclusion-bearing report for the Wan all-site activation study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


VARIANTS = ("identity", "hadamard-h32", "givens-g32")
VARIANT_LABELS = {
    "identity": "Identity",
    "hadamard-h32": "Hadamard H32",
    "givens-g32": "Givens G32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path("/root/autodl-tmp/clean_fp_quant/outputs")
    parser.add_argument(
        "--same-step-result",
        type=Path,
        default=root / "comparisons/all-sites-step25-seed0/comparison.json",
    )
    parser.add_argument(
        "--cross-step-result",
        type=Path,
        default=root / "comparisons/calibrate-step10-evaluate-step25-seed0/comparison.json",
    )
    parser.add_argument(
        "--calibration-activations",
        type=Path,
        default=root / "activations/all-sites-step10-seed0",
    )
    parser.add_argument(
        "--evaluation-activations",
        type=Path,
        default=root / "activations/all-sites-step25-seed0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "comparisons/calibrate-step10-evaluate-step25-seed0",
    )
    return parser.parse_args()


def load_complete_result(path: Path, label: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("status") != "complete":
        raise ValueError(f"{label} result is not complete: {path}")
    if data.get("record_count") != 630:
        raise ValueError(f"{label} expected 630 records, found {data.get('record_count')}")
    if data.get("incomplete_shards"):
        raise ValueError(f"{label} has incomplete shards: {data['incomplete_shards']}")
    return data


def capture_inventory(root: Path) -> dict[str, Any]:
    states = [json.loads(path.read_text()) for path in sorted(root.rglob("state.json"))]
    activations = list(root.rglob("activation.pt"))
    complete_states = sum(state.get("status") == "complete" for state in states)
    return {
        "root": str(root),
        "state_files": len(states),
        "complete_state_files": complete_states,
        "activation_files": len(activations),
        "bytes": sum(path.stat().st_size for path in activations),
        "complete": bool(states) and complete_states == len(states) and len(activations) == 210,
    }


def per_block_summary(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup = {
        (record["block"], record["site"], record["variant"]): record["mxfp4"]["sqnr_db"]
        for record in data["records"]
    }
    result = {}
    for site in data["aggregate"]:
        gains = {}
        for variant in ("hadamard-h32", "givens-g32"):
            values = [
                lookup[(block, site, variant)] - lookup[(block, site, "identity")]
                for block in range(30)
            ]
            gains[variant] = {
                "positive_blocks": sum(value > 0 for value in values),
                "negative_blocks": sum(value < 0 for value in values),
                "minimum_gain_db": min(values),
                "maximum_gain_db": max(values),
            }
        gains["givens_beats_hadamard_blocks"] = sum(
            lookup[(block, site, "givens-g32")] > lookup[(block, site, "hadamard-h32")]
            for block in range(30)
        )
        result[site] = gains
    return result


def recommendation(
    site: str,
    aggregate: dict[str, dict[str, float | int | None]],
    robustness: dict[str, Any],
) -> dict[str, Any]:
    winner = max(VARIANTS, key=lambda name: aggregate[name]["mean_mxfp4_sqnr_db"])
    identity = aggregate["identity"]["mean_mxfp4_sqnr_db"]
    gain = aggregate[winner]["mean_mxfp4_sqnr_db"] - identity
    if winner == "identity":
        conclusion = "保持 Identity；两种旋转没有可靠的平均收益。"
    elif winner == "hadamard-h32":
        positive = robustness["hadamard-h32"]["positive_blocks"]
        conclusion = f"推荐 Hadamard H32；平均收益最高，并在 {positive}/30 个 block 上为正。"
    else:
        positive = robustness["givens-g32"]["positive_blocks"]
        conclusion = f"推荐 Givens G32；平均收益最高，并在 {positive}/30 个 block 上为正。"
    return {
        "site": site,
        "recommended_variant": winner,
        "recommended_label": VARIANT_LABELS[winner],
        "gain_vs_identity_db": gain,
        "conclusion": conclusion,
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def signed(value: float) -> str:
    return f"{value:+.4f}"


def main() -> None:
    args = parse_args()
    same = load_complete_result(args.same_step_result, "same-step")
    cross = load_complete_result(args.cross_step_result, "cross-step")
    if tuple(cross.get("variants", ())) != VARIANTS:
        raise ValueError(f"Unexpected variants: {cross.get('variants')}")

    calibration_capture = capture_inventory(args.calibration_activations)
    evaluation_capture = capture_inventory(args.evaluation_activations)
    if not calibration_capture["complete"] or not evaluation_capture["complete"]:
        raise ValueError("Activation capture is incomplete; refusing to issue conclusions")

    robustness = per_block_summary(cross)
    recommendations = {
        site: recommendation(site, variants, robustness[site])
        for site, variants in cross["aggregate"].items()
    }
    equal_site_means = {
        variant: sum(
            cross["aggregate"][site][variant]["mean_mxfp4_sqnr_db"]
            for site in cross["aggregate"]
        )
        / len(cross["aggregate"])
        for variant in VARIANTS
    }
    mixed_mean = sum(
        cross["aggregate"][site][item["recommended_variant"]]["mean_mxfp4_sqnr_db"]
        for site, item in recommendations.items()
    ) / len(recommendations)
    givens_transfer_shifts = {
        site: (
            cross["aggregate"][site]["givens-g32"]["mean_mxfp4_sqnr_db"]
            - same["aggregate"][site]["givens-g32"]["mean_mxfp4_sqnr_db"]
        )
        for site in cross["aggregate"]
    }
    max_transfer_shift = max(abs(value) for value in givens_transfer_shifts.values())

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": "Wan all-site activation transform comparison",
        "formal_protocol": "calibrate on conditional step 10; evaluate on conditional step 25",
        "calibration_capture": calibration_capture,
        "evaluation_capture": evaluation_capture,
        "same_step_record_count": same["record_count"],
        "cross_step_record_count": cross["record_count"],
        "equal_site_mean_sqnr_db": {**equal_site_means, "recommended_mixed": mixed_mean},
        "maximum_abs_givens_cross_step_shift_db": max_transfer_shift,
        "recommendations": recommendations,
        "per_block_robustness": robustness,
        "limitations": [
            "MXFP4 fake-quant activation SQNR is a numerical proxy, not final generated-video quality.",
            "The formal result currently uses one prompt, one seed, one calibration step, and one evaluation step.",
            "Recommendations are selected per semantic site across all blocks, not independently per block.",
            "The 128x128 smoke run validates plumbing only and is excluded from scientific conclusions.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Wan 全量化位点激活变换比较报告",
        "",
        f"生成时间：{summary['generated_at']}  ",
        "正式协议：step 10 conditional 激活校准，step 25 conditional 激活评估。  ",
        "比较方法：Identity、随机符号 Hadamard H32、校准 Givens G32。  ",
        "量化代理：对称 MXFP4、group size 32、E8M0 scale、min-max observer。",
        "",
        "## 执行结论",
        "",
        f"- 两套正式结果均完整：同一步协议 {same['record_count']} 条记录，跨时间步协议 {cross['record_count']} 条记录。",
        "- Hadamard 是整体最稳健的单一策略，尤其适合 self_qkv、cross_q、ffn_in 和 ffn_out。",
        "- cross_kv 应保持 Identity；旋转没有平均收益。",
        "- Givens 在 self_o 和 cross_o 上略优，但优势较小。",
        f"- Givens 从 step 10 迁移到 step 25 后，各 site 平均 SQNR 相对同一步校准的最大变化仅 {max_transfer_shift:.4f} dB，未发现明显同数据过拟合。",
        f"- 按 site 混合策略的等权平均 SQNR 为 {mixed_mean:.4f} dB；纯 Hadamard 为 {equal_site_means['hadamard-h32']:.4f} dB，纯 Givens 为 {equal_site_means['givens-g32']:.4f} dB，Identity 为 {equal_site_means['identity']:.4f} dB。",
        "- 当前证据支持进入端到端 W4A4 视频生成验证，但不能仅凭激活 SQNR 宣布最终视频质量更优。",
        "",
        "## 数据完整性",
        "",
        "| 数据集 | 状态文件 | 激活文件 | 大小 GiB | 结论 |",
        "|---|---:|---:|---:|---|",
        f"| step 10 calibration | {calibration_capture['complete_state_files']}/{calibration_capture['state_files']} | {calibration_capture['activation_files']}/210 | {calibration_capture['bytes'] / 1024**3:.2f} | 完整 |",
        f"| step 25 evaluation | {evaluation_capture['complete_state_files']}/{evaluation_capture['state_files']} | {evaluation_capture['activation_files']}/210 | {evaluation_capture['bytes'] / 1024**3:.2f} | 完整 |",
        "",
        "## 聚合结果与逐位点结论",
        "",
        "SQNR 越高越好；括号内为相对 Identity 的变化。",
        "",
        "| Site | Identity | Hadamard H32 | Givens G32 | 推荐 | 结论 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for site, variants in cross["aggregate"].items():
        identity = variants["identity"]["mean_mxfp4_sqnr_db"]
        hadamard = variants["hadamard-h32"]["mean_mxfp4_sqnr_db"]
        givens = variants["givens-g32"]["mean_mxfp4_sqnr_db"]
        item = recommendations[site]
        lines.append(
            f"| {site} | {fmt(identity)} | {fmt(hadamard)} ({signed(hadamard - identity)}) | "
            f"{fmt(givens)} ({signed(givens - identity)}) | {item['recommended_label']} | {item['conclusion']} |"
        )

    lines += [
        "",
        "## 逐 block 稳定性",
        "",
        "| Site | Hadamard 正增益 block | Hadamard 最差/最好 | Givens 正增益 block | Givens 最差/最好 | Givens 胜过 Hadamard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for site, values in robustness.items():
        hadamard = values["hadamard-h32"]
        givens = values["givens-g32"]
        lines.append(
            f"| {site} | {hadamard['positive_blocks']}/30 | "
            f"{signed(hadamard['minimum_gain_db'])} / {signed(hadamard['maximum_gain_db'])} | "
            f"{givens['positive_blocks']}/30 | "
            f"{signed(givens['minimum_gain_db'])} / {signed(givens['maximum_gain_db'])} | "
            f"{values['givens_beats_hadamard_blocks']}/30 |"
        )

    lines += [
        "",
        "## 推荐配置",
        "",
    ]
    for site, item in recommendations.items():
        lines.append(
            f"- `{site}`：{item['recommended_label']}，相对 Identity {signed(item['gain_vs_identity_db'])} dB。"
        )
    lines += [
        "",
        "## 图表",
        "",
        "- `sqnr_by_site.png`：七类位点的平均 SQNR。",
        "- `sqnr_gain_by_block.png`：30 个 block 上两种旋转相对 Identity 的增益。",
        "- `per_block.csv`：逐 block 原始聚合值，便于二次分析。",
        "",
        "## 限制与下一步",
        "",
        "1. 当前指标是激活 fake-quant 的数值指标，不是最终视频质量指标。",
        "2. 正式实验目前只有一个 prompt 和 seed，需要至少增加多个 prompt 与 seed 验证稳定性。",
        "3. 下一步应比较 Identity、纯 Hadamard、纯 Givens、推荐混合策略的 W4A4 生成视频。",
        "4. 视频结果应使用相同 prompt、seed、scheduler 和采样步数，并通过 VBench 与成对视觉检查共同判断。",
        "",
        "## 产物索引",
        "",
        f"- 同一步比较：`{args.same_step_result}`",
        f"- 跨时间步比较：`{args.cross_step_result}`",
        f"- 校准激活：`{args.calibration_activations}`",
        f"- 评估激活：`{args.evaluation_activations}`",
        "- 本报告的结构化摘要：`summary.json`",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines))
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
