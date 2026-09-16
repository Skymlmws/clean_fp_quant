"""Build one comparison report for every method in a VBench experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_vbench import DIMENSIONS, collect, label


FOCUSED_VIEWS = (
    ("Givens methods", lambda method: "givens" in method["id"]),
    ("W4A4 methods", lambda method: "w4a4" in method["id"]),
    ("Attention-only methods", lambda method: method["id"].endswith("-attention")),
    ("FFN-only methods", lambda method: method["id"].endswith("-ffn")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def render_table(
    methods: list[dict],
    scores: dict[str, dict[str, tuple[float, int]]],
    reference: dict,
) -> list[str]:
    """Render scores for a method subset, always including the reference."""
    selected = [reference] + [
        method for method in methods if method["id"] != reference["id"]
    ]
    reference_scores = scores[reference["id"]]
    header = ["Dimension", *(method["label"] for method in selected)]
    separator = ["---", *("---:" for _ in selected)]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(separator) + "|",
    ]

    for dimension in DIMENSIONS:
        row = [label(dimension)]
        ref_value = reference_scores.get(dimension)
        for method in selected:
            current = scores[method["id"]].get(dimension)
            if method["id"] == reference["id"]:
                row.append(f"{current[0]:.6f}" if current else "N/A")
            elif not current:
                row.append("N/A")
            else:
                delta = current[0] - ref_value[0] if ref_value else None
                delta_text = f"{delta:+.6f}" if delta is not None else "N/A"
                row.append(f"{current[0]:.6f} ({delta_text})")
        lines.append("| " + " | ".join(row) + " |")
    return lines


def main() -> None:
    args = parse_args()
    manifest_path = args.experiment_dir / "experiment.json"
    manifest = json.loads(manifest_path.read_text())
    configured_methods = manifest["methods"]
    references = [method for method in configured_methods if method.get("reference")]
    if len(references) != 1:
        raise ValueError("experiment.json must define exactly one reference method")

    scores = {
        method["id"]: collect(args.experiment_dir / "methods" / method["id"])
        for method in configured_methods
    }
    reference = references[0]
    methods = [
        method for method in configured_methods
        if method.get("reference") or scores[method["id"]]
    ]
    unavailable_methods = [
        method for method in configured_methods if method not in methods
    ]

    lines = [
        f"# {manifest['model']} — VBench comparison",
        "",
        f"Protocol: {manifest['protocol']}  ",
        f"Prompt count: {manifest['prompt_count']}  ",
        f"Sample seed: {manifest['sample_seed']}",
        "",
        "## Full comparison",
        "",
    ]
    lines += render_table(methods, scores, reference)

    lines += [
        "",
        "## Focused comparisons",
        "",
        "Each view retains BF16 as the reference. Values in parentheses are deltas against BF16.",
    ]
    for title, predicate in FOCUSED_VIEWS:
        selected = [method for method in methods if predicate(method)]
        if not selected:
            continue
        lines += ["", f"### {title}", ""]
        lines += render_table(selected, scores, reference)

    lines += ["", "## Methods", ""]
    for method in methods:
        lines.append(f"- {method['label']}: `{method['id']}`")
    if unavailable_methods:
        lines += ["", "Configured methods without evaluation results are omitted:"]
        for method in unavailable_methods:
            lines.append(f"- {method['label']}: `{method['id']}`")
    lines += [
        "",
        "## Notes",
        "",
        "- This is a matched comparison over the same prompt subset and sample seed.",
        "- Values in parentheses are deltas against BF16; positive is higher.",
        "- N/A means VBench returned no valid samples for that dimension.",
        "- Raw JSON outputs and logs are stored under each directory in `methods/`.",
        "",
    ]
    (args.experiment_dir / "comparison.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
