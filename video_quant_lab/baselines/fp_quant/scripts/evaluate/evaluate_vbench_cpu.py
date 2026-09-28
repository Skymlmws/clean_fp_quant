#!/usr/bin/env python3
"""Run selected VBench dimensions on CPU without initializing NCCL."""

import argparse
from datetime import datetime

import torch
from vbench import VBench


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--dimension", nargs="+", required=True)
    args = parser.parse_args()

    evaluator = VBench(torch.device("cpu"), args.metadata, args.output_path)
    evaluator.evaluate(
        videos_path=args.videos_path,
        name=f"results_{datetime.now():%Y-%m-%d-%H:%M:%S}",
        dimension_list=args.dimension,
        local=True,
        mode="vbench_standard",
    )


if __name__ == "__main__":
    main()
