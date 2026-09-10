"""Generate a resumable quantized VBench subset with one resident Wan pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from scripts.generate.generate_wan_vbench_batch import (
    OFFICIAL_NEGATIVE_PROMPT,
    load_records,
    safe_filename,
    select_stratified,
    write_json,
)
from src.utils.wan_utils import (
    build_wan_block_transforms,
    finalize_wan_transforms,
    get_wan_transform_stats,
    observe_wan_transforms,
    replace_wan_linears,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--augmented-prompts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=32)
    parser.add_argument("--selection-seed", type=int, default=20260903)
    parser.add_argument("--sample-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--transform-class", choices=("identity", "hadamard", "givens"), default="givens"
    )
    parser.add_argument("--transform-group-size", type=int, default=32)
    parser.add_argument("--transform-randomize", action="store_true")
    parser.add_argument("--transform-seed", type=int, default=0)
    parser.add_argument("--outlier-threshold", type=float, default=5.0)
    parser.add_argument("--quant-group-size", type=int, default=32)
    parser.add_argument("--weight-bits", type=int, choices=(4, 16), default=4)
    parser.add_argument("--activation-bits", type=int, choices=(4, 16), default=4)
    parser.add_argument("--weight-observer", choices=("minmax", "mse"), default="minmax")
    parser.add_argument("--calibration-prompt-index", type=int, default=0)
    parser.add_argument("--min-free-gib", type=float, default=200.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def generate(pipe, prompt: str, seed: int) -> torch.Tensor:
    return pipe.generate(
        input_prompt=prompt,
        size=(832, 480),
        frame_num=81,
        shift=3.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=6.0,
        n_prompt=OFFICIAL_NEGATIVE_PROMPT,
        seed=seed,
        offload_model=False,
    )


def main() -> None:
    args = parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world-size)")
    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    if args.quant_group_size != 32:
        raise ValueError("MXFP4 requires quant-group-size=32")
    if args.transform_group_size <= 1 or args.transform_group_size & (args.transform_group_size - 1):
        raise ValueError("transform-group-size must be a power of two greater than one")
    if args.transform_randomize and args.transform_class != "hadamard":
        raise ValueError("transform-randomize is supported only with transform-class=hadamard")

    selected = select_stratified(
        load_records(args.metadata, args.augmented_prompts),
        args.prompt_count,
        args.selection_seed,
    )
    if not 0 <= args.calibration_prompt_index < len(selected):
        raise ValueError("calibration-prompt-index must identify a selected prompt")
    tasks = []
    for prompt_index, record in enumerate(selected):
        for sample_index, seed in enumerate(args.sample_seeds):
            tasks.append({
                "task_index": len(tasks),
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "seed": seed,
                **record,
                "filename": safe_filename(record["prompt"], sample_index),
            })
    rank_tasks = [task for task in tasks if task["task_index"] % args.world_size == args.rank]
    assigned = rank_tasks[args.worker_index::args.worker_count]
    calibration = selected[args.calibration_prompt_index]
    method = f"{args.transform_class}-mxfp4-w{args.weight_bits}a{args.activation_bits}"
    quantization = {
        "transform": args.transform_class,
        "transform_group_size": args.transform_group_size,
        "transform_randomize": args.transform_randomize,
        "transform_seed": args.transform_seed,
        "format": "mxfp",
        "weight_bits": args.weight_bits,
        "activation_bits": args.activation_bits,
        "quant_group_size": args.quant_group_size,
        "weight_observer": args.weight_observer,
    }
    if args.transform_class == "givens":
        quantization.update({
            "outlier_threshold": args.outlier_threshold,
            "calibration_prompt_index": args.calibration_prompt_index,
            "calibration_source_index": calibration["source_index"],
            "calibration_prompt": calibration["prompt"],
        })
    plan = {
        "schema_version": 1,
        "suite": f"vbench-stratified-{args.prompt_count}",
        "method": method,
        "selection_seed": args.selection_seed,
        "sample_seeds": args.sample_seeds,
        "quantization": quantization,
        "official_wan_config": {
            "size": [832, 480], "frames": 81, "fps": 16,
            "steps": 50, "sampler": "unipc", "guide_scale": 6.0,
            "flow_shift": 3.0, "prompt_source": str(args.augmented_prompts.resolve()),
            "negative_prompt": OFFICIAL_NEGATIVE_PROMPT,
        },
        "rank": args.rank, "world_size": args.world_size,
        "worker_index": args.worker_index, "worker_count": args.worker_count,
        "tasks": assigned,
    }
    if args.dry_run:
        print(json.dumps({**plan, "tasks": assigned[:3], "task_count": len(assigned)}, ensure_ascii=False, indent=2))
        return
    if not (args.checkpoint / "diffusion_pytorch_model.safetensors").is_file():
        raise FileNotFoundError(f"Wan checkpoint not found: {args.checkpoint}")
    if not (args.wan_repo / "wan").is_dir():
        raise FileNotFoundError(f"Wan repository not found: {args.wan_repo}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_suffix = (
        f"-worker{args.worker_index}of{args.worker_count}" if args.worker_count > 1 else ""
    )
    write_json(args.output_dir / f"plan-rank{args.rank}{worker_suffix}.json", plan)
    sys.path.insert(0, str(args.wan_repo.resolve()))
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    from wan.utils.utils import cache_video

    device = torch.device(f"cuda:{args.device_id}")
    pipe = WanT2V(
        config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=str(args.checkpoint),
        device_id=args.device_id, t5_cpu=True,
    )
    transform_kwargs = {}
    if args.transform_class == "givens":
        transform_kwargs["outlier_threshold"] = args.outlier_threshold
    elif args.transform_class == "hadamard":
        transform_kwargs.update(randomize=args.transform_randomize, seed=args.transform_seed)
    block_transforms = build_wan_block_transforms(
        pipe.model, args.transform_class, args.transform_group_size, device, **transform_kwargs
    )
    if args.transform_class == "givens":
        handles = observe_wan_transforms(pipe.model, block_transforms)
        try:
            calibration_video = generate(pipe, calibration["augmented_prompt"], args.sample_seeds[0])
            del calibration_video
        finally:
            for handle in handles:
                handle.remove()
        finalize_wan_transforms(block_transforms)

    quantizer_common = {
        "symmetric": True, "format": "mxfp", "granularity": "group",
        "group_size": args.quant_group_size, "scale_precision": "e8m0",
    }
    weight_quantizer_kwargs = (
        {**quantizer_common, "bits": args.weight_bits, "observer": args.weight_observer}
        if args.weight_bits < 16 else None
    )
    activation_quantizer_kwargs = (
        {**quantizer_common, "bits": args.activation_bits, "observer": "minmax"}
        if args.activation_bits < 16 else None
    )
    report = replace_wan_linears(
        pipe.model,
        block_transforms,
        weight_quantizer_kwargs,
        activation_quantizer_kwargs,
    )
    plan["quantization"]["replaced_linears"] = report.replaced_count
    plan["quantization"]["skipped"] = report.skipped
    if args.transform_class == "givens":
        plan["quantization"]["transform_stats"] = get_wan_transform_stats(block_transforms)
    write_json(args.output_dir / f"plan-rank{args.rank}{worker_suffix}.json", plan)

    results = []
    result_path = args.output_dir / f"results-rank{args.rank}{worker_suffix}.json"
    for task in assigned:
        output = args.output_dir / task["filename"]
        if output.is_file() and output.stat().st_size > 0:
            results.append({**task, "status": "skipped", "path": str(output)})
            write_json(result_path, results)
            continue
        free_gib = shutil.disk_usage(args.output_dir).free / 1024**3
        if free_gib < args.min_free_gib:
            raise RuntimeError(f"Only {free_gib:.1f} GiB free; stopping before disk pressure")
        started = time.perf_counter()
        video = generate(pipe, task["augmented_prompt"], task["seed"])
        temporary = output.with_name(output.stem + ".partial.mp4")
        saved = cache_video(video[None], save_file=str(temporary), fps=16)
        del video
        if saved is None or not temporary.is_file():
            raise RuntimeError(f"Failed to encode {output}")
        temporary.replace(output)
        results.append({
            **task, "status": "complete", "path": str(output),
            "seconds": time.perf_counter() - started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        })
        write_json(result_path, results)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
