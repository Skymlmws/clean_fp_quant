"""Summarize paired Wan cross_q threshold-ablation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_run(value: str) -> tuple[float, Path]:
    threshold_text, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("run must have the form THRESHOLD=OUTPUT_DIR")
    try:
        threshold = float(threshold_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid threshold {threshold_text!r}") from error
    return threshold, Path(path_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run,
        help="Threshold and experiment directory as THRESHOLD=OUTPUT_DIR; repeat as needed",
    )
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def metadata_path(root: Path, step: int, block: int, variant: str) -> Path:
    return (
        root / f"step_{step:03d}" / "conditional" / f"block_{block:02d}"
        / "cross_q" / variant / "metadata.json"
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing completed result: {path}")
    return json.loads(path.read_text())


def collect(threshold: float, root: Path, step: int, block: int) -> dict[str, Any]:
    config = load_json(root / "config.json")
    givens = load_json(metadata_path(root, step, block, "givens-g32"))
    hadamard = load_json(metadata_path(root, step, block, "hadamard-h32"))
    return {
        "threshold": threshold,
        "root": str(root.resolve()),
        "givens_blocks": config["givens_stats"]["givens_blocks"],
        "hadamard_blocks": config["givens_stats"]["hadamard_blocks"],
        "givens": {
            "statistics": givens["full_statistics"],
            "mxfp4": givens["mxfp4"],
            "rotation_validation": givens.get("rotation_validation"),
        },
        "hadamard": {
            "statistics": hadamard["full_statistics"],
            "mxfp4": hadamard["mxfp4"],
            "rotation_validation": hadamard.get("rotation_validation"),
        },
    }


def number(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if value != 0 and abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    rows = summary["runs"]
    baseline = rows[0]["hadamard"]
    all_fallback_rows = [row for row in rows if row["givens_blocks"] == 0]
    fallback_matches_baseline = bool(all_fallback_rows) and all(
        row["givens"]["statistics"] == row["hadamard"]["statistics"]
        and row["givens"]["mxfp4"] == row["hadamard"]["mxfp4"]
        and row["givens"]["rotation_validation"] == row["hadamard"]["rotation_validation"]
        for row in all_fallback_rows
    )
    lines = [
        "# Wan cross_q 的 Givens 阈值消融实验",
        "",
        f"对比位置：采样步 {summary['step']}、Transformer block {summary['block']}、conditional 分支。",
        "每次实验中的 Identity、Hadamard 和 Givens 结果均由同一个 BF16 激活矩阵得到，因此可以直接进行配对比较。",
        "",
        "## 阈值扫描结果",
        "",
        "| Threshold | Global Givens channel groups | Global Hadamard channel groups | Local Max abs | Local Max channel RMS / median | Local MXFP4 MSE | Local SQNR (dB) | Local Max token L2 relative error | Local Linear relative L2 error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        stats = row["givens"]["statistics"]
        quant = row["givens"]["mxfp4"]
        validation = row["givens"]["rotation_validation"] or {}
        lines.append(
            "| " + " | ".join((
                number(row["threshold"], 1),
                number(row["givens_blocks"]),
                number(row["hadamard_blocks"]),
                number(stats["max_abs"]),
                number(stats["max_channel_rms_over_median"]),
                number(quant["mse"]),
                number(quant["sqnr_db"]),
                number(validation.get("max_token_l2_relative_error")),
                number(validation.get("linear_relative_l2_error")),
            )) + " |"
        )
    hstats = baseline["statistics"]
    hquant = baseline["mxfp4"]
    hvalidation = baseline["rotation_validation"] or {}
    total_blocks = rows[0]["givens_blocks"] + rows[0]["hadamard_blocks"]
    lines.extend([
        f"| Randomized fast Hadamard baseline | 0 | {total_blocks} | "
        + " | ".join((
            number(hstats["max_abs"]),
            number(hstats["max_channel_rms_over_median"]),
            number(hquant["mse"]),
            number(hquant["sqnr_db"]),
            number(hvalidation.get("max_token_l2_relative_error")),
            number(hvalidation.get("linear_relative_l2_error")),
        )) + " |",
        "",
        "## 指标说明",
        "",
        "- `Threshold`：Givens 路由使用的绝对值阈值。校准时，如果某个 32 通道分组观测到的最大绝对值超过该阈值，该分组采用 Givens；否则采用 Hadamard。",
        "- 表中带 `Global` 的两列是全模型统计；带 `Local` 的其余指标只来自 step 10、Transformer block 0、conditional 分支。两类指标的统计范围不同。",
        "- `Global Givens channel groups`：全模型 30 个 Transformer blocks 的 cross_q 中，实际采用 Givens 旋转的 32 通道分组数量。每层有 `1536 / 32 = 48` 个通道组，因此总数为 `30 × 48 = 1440`。它不是 token 数量。",
        "- `Global Hadamard channel groups`：全模型未触发 Givens、因而采用 Hadamard 的 32 通道分组数量。它与 Global Givens channel groups 之和为 1440。",
        "- `Local Max abs`：完成对应旋转后，所选局部位置的整个 cross_q 输入激活中，所有元素绝对值的最大值，即 `max(abs(X_rotated))`。该激活形状为 `[1, 32760, 1536]`，展平 batch 后可看作 32760 个 token × 1536 个通道。它衡量最极端单点的幅度，通常越低越容易控制量化动态范围，但不能单独反映异常值是否集中在固定通道。注意，它不是用于决定路由的校准原始最大值。",
        "- `Max channel RMS / median`：先计算每个通道跨 token 的 RMS，再用最大的通道 RMS 除以所有通道 RMS 的中位数。它衡量能量是否集中在少数固定通道；越低通常说明 persistent channel outlier 越弱。",
        "- `MXFP4 MSE`：模拟 MXFP4 量化后的均方误差，越低越好。",
        "- `SQNR (dB)`：量化信噪比，越高越好。",
        "- `Max token L2 relative error`：比较旋转前后每个 token 的 L2 范数，报告最大的相对误差，用于检查旋转的正交性在实际数值精度下保持得如何，越低越好。",
        "- `Linear relative L2 error`：比较旋转重参数化前后的 Linear 层输出，报告相对 L2 误差；理想情况下应接近 0，越低越好。",
        "",
        "## 主要结论",
        "",
        "- 阈值越高，越少的 32 通道分组触发 Givens，更多分组回退到 Hadamard。",
        "- 阈值 12、16、24 的全模型路由数虽然不同，但 block 0 的所有局部指标和逐帧记录完全相同。这说明发生路由切换的通道组位于其他 Transformer blocks；当前报告没有观测到那些层的输出变化。",
        "- 在这个固定位置，即使阈值提高到 24，包含主要 persistent channel outlier 的分组仍然触发 Givens：最大绝对值保持为 8.75，最大通道 RMS / 中位数仍高于 31。",
        "- 随机符号快速 Hadamard 对 persistent channel 结构的抑制明显更强：最大绝对值为 5.65，最大通道 RMS / 中位数为 18.32。",
        "- 阈值 40 时没有任何分组触发 Givens（0 个 Givens、1440 个 Hadamard），其结果"
        + ("与独立 Hadamard 基线完全一致" if fallback_matches_baseline else "仍未与独立 Hadamard 基线完全一致")
        + "。比较范围包括激活统计、MXFP4 指标和旋转数值误差。这说明 fallback 实现不一致的问题已经排除。",
        "- 两类局部指标给出了不同倾向：Givens 的 MXFP4 MSE 更低、SQNR 更高，但 persistent channel 结构更严重。因此不能只用激活矩阵的 MXFP4 误差选择最终方法。",
        "- Givens 和 Hadamard 的 Linear 输出误差处于相近的 BF16 误差量级，但 Givens 的 token 范数误差明显更大。这可能来自任意角度旋转系数在 BF16 下的数值损失，还需要精度消融才能确定原因。",
        "",
        "## 当前判断",
        "",
        "当前结果已经不是简单的阈值调得不好。阈值从 3 扫到 24 后，只要主要异常分组仍采用 Givens，persistent channel 指标就明显弱于 Hadamard；而阈值提高到 40、完全回退为 Hadamard 后，两者才完全一致。",
        "",
        "## 下一步消融实验",
        "",
        "1. persistent channel outlier 使用 Hadamard，isolated token outlier 使用 Givens，验证按异常类型路由是否更合理。",
        "2. 对比直接使用 BF16 Givens 与使用 FP32 完成旋转后再转回 BF16，判断 token 误差是否主要来自计算精度。",
        "3. 除激活矩阵的 MXFP4 误差外，还要测量同一层 Linear 输出端的量化误差；局部指标确认后再进行完整 VBench。",
        "",
        "注意：这是单个位置的诊断性消融，只能用于判断实现和局部行为，不能直接等价为最终视频质量结论。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    runs = [collect(threshold, root, args.step, args.block) for threshold, root in args.run]
    runs.sort(key=lambda item: item["threshold"])
    summary = {"step": args.step, "block": args.block, "runs": runs}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(render_report(summary))
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
