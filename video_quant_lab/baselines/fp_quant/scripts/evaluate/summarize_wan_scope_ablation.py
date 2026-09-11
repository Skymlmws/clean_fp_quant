#!/usr/bin/env python3
"""Summarize paired Identity/Givens VBench scope ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_vbench import DIMENSIONS, collect, label


PAIRS = {
    "attention": (
        "identity-mxfp4-w4a4-attention",
        "givens-mxfp4-w4a4-attention",
    ),
    "ffn": (
        "identity-mxfp4-w4a4-ffn",
        "givens-mxfp4-w4a4-ffn",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_root = args.experiment_dir / "methods"
    bf16 = collect(method_root / "bf16")
    methods = {
        method_id: collect(method_root / method_id)
        for pair in PAIRS.values()
        for method_id in pair
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "comparison": "Givens minus Identity at the same quantization scope",
        "scopes": {},
    }
    lines = [
        "# Wan MXFP4 W4A4 module-scope ablation",
        "",
        "Paired comparison: Givens minus Identity under the same quantized module scope.",
        "Positive VBench deltas are better.",
    ]
    for scope, (identity_id, givens_id) in PAIRS.items():
        identity = methods[identity_id]
        givens = methods[givens_id]
        rows = []
        deltas = []
        lines += [
            "",
            f"## {scope.title()}",
            "",
            "| Dimension | BF16 | Identity | Givens | Givens - Identity |",
            "|---|---:|---:|---:|---:|",
        ]
        for dimension in DIMENSIONS:
            reference = bf16.get(dimension)
            baseline = identity.get(dimension)
            optimized = givens.get(dimension)
            delta = optimized[0] - baseline[0] if baseline and optimized else None
            if delta is not None:
                deltas.append(delta)
            rows.append({
                "dimension": dimension,
                "bf16": reference[0] if reference else None,
                "identity": baseline[0] if baseline else None,
                "givens": optimized[0] if optimized else None,
                "givens_minus_identity": delta,
            })
            fmt = lambda value: "N/A" if value is None else f"{value:.6f}"
            lines.append(
                f"| {label(dimension)} | {fmt(rows[-1]['bf16'])} | "
                f"{fmt(rows[-1]['identity'])} | {fmt(rows[-1]['givens'])} | "
                f"{('N/A' if delta is None else f'{delta:+.6f}')} |"
            )
        positive = sum(value > 0 for value in deltas)
        negative = sum(value < 0 for value in deltas)
        unchanged = len(deltas) - positive - negative
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        scope_summary = {
            "identity_method": identity_id,
            "givens_method": givens_id,
            "evaluated_dimensions": len(deltas),
            "positive_dimensions": positive,
            "negative_dimensions": negative,
            "unchanged_dimensions": unchanged,
            "unweighted_mean_delta": mean_delta,
            "dimensions": rows,
        }
        payload["scopes"][scope] = scope_summary
        mean_text = "N/A" if mean_delta is None else f"{mean_delta:+.6f}"
        lines += [
            "",
            f"Summary: {positive} positive, {negative} negative, {unchanged} unchanged; "
            f"unweighted mean delta {mean_text}.",
        ]
    lines += [
        "",
        "## Interpretation note",
        "",
        "The unweighted mean is a diagnostic summary, not an official VBench aggregate. "
        "The per-dimension paired deltas are the primary ablation evidence.",
        "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
