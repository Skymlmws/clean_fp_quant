"""CLI for planning and running an experiment in an external baseline repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import build_plan, execute, plan_record
from .schema import BaselineManifest, Experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path("video_quant_lab/baselines"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument(
        "--run-name",
        help="Output directory name; defaults to the experiment name",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = Experiment.load(args.experiment)
    manifest_path = args.baselines_dir / experiment.baseline / "manifest.json"
    manifest = BaselineManifest.load(manifest_path)
    plan = build_plan(experiment, manifest, args.output_root, run_name=args.run_name)
    if args.dry_run:
        print(json.dumps(plan_record(plan), indent=2))
        return
    return_code = execute(plan)
    print(f"run: {plan.run_dir}")
    print(f"status: {'complete' if return_code == 0 else 'failed'}")
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
