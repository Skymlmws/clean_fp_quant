"""Build one comparison report for every method in a VBench experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_vbench import DIMENSIONS, collect, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.experiment_dir / "experiment.json"
    manifest = json.loads(manifest_path.read_text())
    methods = manifest["methods"]
    references = [method for method in methods if method.get("reference")]
    if len(references) != 1:
        raise ValueError("experiment.json must define exactly one reference method")

    scores = {
        method["id"]: collect(args.experiment_dir / "methods" / method["id"])
        for method in methods
    }
    reference = references[0]
    reference_scores = scores[reference["id"]]

    header = ["Dimension"]
    separator = ["---"]
    for method in methods:
        header.append(method["label"])
        separator.append("---:")

    lines = [
        f"# {manifest['model']} — VBench comparison",
        "",
        f"Protocol: {manifest['protocol']}  ",
        f"Prompt count: {manifest['prompt_count']}  ",
        f"Sample seed: {manifest['sample_seed']}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(separator) + "|",
    ]
    for dimension in DIMENSIONS:
        row = [label(dimension)]
        ref_value = reference_scores.get(dimension)
        for method in methods:
            current = scores[method["id"]].get(dimension)
            if method.get("reference"):
                row.append(f"{current[0]:.6f}" if current else "N/A")
                continue

            if not current:
                row.append("N/A")
                continue

            delta = current[0] - ref_value[0] if ref_value else None
            delta_text = f"{delta:+.6f}" if delta is not None else "N/A"
            row.append(f"{current[0]:.6f} ({delta_text})")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Methods", ""]
    for method in methods:
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
