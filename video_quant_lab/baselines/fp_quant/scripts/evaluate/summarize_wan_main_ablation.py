#!/usr/bin/env python3
"""Summarize the 2x2 Attention/FFN Givens main ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_vbench import DIMENSIONS, collect, label


METHODS = {
    "A": "identity-mxfp4-w4a4",
    "B": "attn-givens-ffn-identity-mxfp4-w4a4",
    "C": "attn-identity-ffn-givens-mxfp4-w4a4",
    "D": "givens-mxfp-w4a4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.experiment_dir / "methods"
    scores = {key: collect(root / method_id) for key, method_id in METHODS.items()}
    rows = []
    for dimension in DIMENSIONS:
        values = {
            key: result[0] if (result := scores[key].get(dimension)) else None
            for key in METHODS
        }
        effects = {"attention": None, "ffn": None, "both": None, "interaction": None}
        if all(values[key] is not None for key in METHODS):
            effects = {
                "attention": values["B"] - values["A"],
                "ffn": values["C"] - values["A"],
                "both": values["D"] - values["A"],
                "interaction": values["D"] - values["B"] - values["C"] + values["A"],
            }
        rows.append({"dimension": dimension, **values, **effects})

    def effect_summary(name: str) -> dict[str, float | int | None]:
        values = [row[name] for row in rows if row[name] is not None]
        return {
            "evaluated_dimensions": len(values),
            "positive_dimensions": sum(value > 0 for value in values),
            "negative_dimensions": sum(value < 0 for value in values),
            "unchanged_dimensions": sum(value == 0 for value in values),
            "unweighted_mean": sum(values) / len(values) if values else None,
        }

    summaries = {
        name: effect_summary(name) for name in ("attention", "ffn", "both", "interaction")
    }
    payload = {"schema_version": 1, "methods": METHODS, "effects": summaries, "dimensions": rows}
    lines = [
        "# Wan MXFP4 W4A4 main 2x2 ablation",
        "",
        "A: Attention Identity, FFN Identity  ",
        "B: Attention Givens, FFN Identity  ",
        "C: Attention Identity, FFN Givens  ",
        "D: Attention Givens, FFN Givens",
        "",
        "| Dimension | A | B | C | D | Attention B-A | FFN C-A | Both D-A | Interaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fmt = lambda value: "N/A" if value is None else f"{value:.6f}"
    signed = lambda value: "N/A" if value is None else f"{value:+.6f}"
    for row in rows:
        lines.append(
            f"| {label(row['dimension'])} | {fmt(row['A'])} | {fmt(row['B'])} | "
            f"{fmt(row['C'])} | {fmt(row['D'])} | {signed(row['attention'])} | "
            f"{signed(row['ffn'])} | {signed(row['both'])} | {signed(row['interaction'])} |"
        )
    lines += ["", "## Diagnostic summaries", ""]
    for name, summary in summaries.items():
        mean = summary["unweighted_mean"]
        lines.append(
            f"- {name}: {summary['positive_dimensions']} positive, "
            f"{summary['negative_dimensions']} negative, {summary['unchanged_dimensions']} unchanged; "
            f"unweighted mean {'N/A' if mean is None else f'{mean:+.6f}'}"
        )
    lines += [
        "",
        "The unweighted means are diagnostic summaries, not official VBench aggregates. "
        "Per-dimension effects are the primary evidence.",
        "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
