"""Merge sharded VBench outputs into a Markdown comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DIMENSIONS = (
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality",
    "imaging_quality", "object_class", "multiple_objects", "human_action",
    "color", "spatial_relationship", "scene", "temporal_style",
    "appearance_style", "overall_consistency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--baseline-result-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    return parser.parse_args()


def collect(root: Path) -> dict[str, tuple[float, int]]:
    values: dict[str, tuple[float, int]] = {}
    files = sorted(root.glob("**/*_eval_results.json"), key=lambda path: path.stat().st_mtime)
    for path in files:
        payload = json.loads(path.read_text())
        for dimension, value in payload.items():
            if dimension in DIMENSIONS and isinstance(value, list) and len(value) >= 2:
                values[dimension] = (float(value[0]), len(value[1]))
    return values


def label(name: str) -> str:
    return name.replace("_", " ").title()


def main() -> None:
    args = parse_args()
    scores = collect(args.result_dir)
    baseline = collect(args.baseline_result_dir)
    video_count = sum(1 for path in args.video_dir.glob("*.mp4") if path.stat().st_size > 0)
    lines = [
        "# VBench evaluation report",
        "",
        f"Method: {args.method}  ",
        "Model: Wan2.1-T2V-1.3B  ",
        "Sample set: stratified 32 prompts, seed 0  ",
        "Evaluation mode: VBench standard  ",
        f"Generated videos: {video_count}",
        "",
        "## Aggregate scores",
        "",
        "| Dimension | Method | BF16 | Delta | Evaluated videos |",
        "|---|---:|---:|---:|---:|",
    ]
    for dimension in DIMENSIONS:
        current = scores.get(dimension)
        reference = baseline.get(dimension)
        if current is None:
            lines.append(f"| {label(dimension)} | N/A | " +
                         (f"{reference[0]:.6f}" if reference else "N/A") + " | N/A | 0 |")
            continue
        delta = current[0] - reference[0] if reference else None
        lines.append(
            f"| {label(dimension)} | {current[0]:.6f} | "
            f"{reference[0]:.6f} | {delta:+.6f} | {current[1]} |"
            if reference else
            f"| {label(dimension)} | {current[0]:.6f} | N/A | N/A | {current[1]} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Scores cover the matched 32-prompt stratified subset, not the complete VBench benchmark.",
        "- N/A means that VBench did not return a valid aggregate for that dimension.",
        "- Detailed per-video values are stored in the sibling evaluation JSON files.",
        "",
    ]
    args.result_dir.mkdir(parents=True, exist_ok=True)
    (args.result_dir / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
