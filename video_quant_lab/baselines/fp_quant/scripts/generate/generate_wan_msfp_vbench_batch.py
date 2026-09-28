"""Resumable VBench-mini generation for BF16, signed FP4, and MSFP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import torch

from scripts.generate.generate_wan_vbench_batch import (
    OFFICIAL_NEGATIVE_PROMPT,
    load_records,
    safe_filename,
    select_records,
    write_json,
)
from src.quantization.wan_rtn import apply_wan_msfp_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers"))
    parser.add_argument("--variant", choices=("bf16", "signed", "mixup"), required=True)
    parser.add_argument("--calibration-config", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--augmented-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=43)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.variant != "bf16" and args.calibration_config is None:
        raise ValueError("Quantized variants require --calibration-config")
    records = select_records(
        load_records(args.metadata, args.augmented_prompts),
        args.prompt_count, 0, "all",
    )
    tasks = []
    for prompt_index, record in enumerate(records):
        tasks.append({
            "task_index": prompt_index,
            "prompt_index": prompt_index,
            "sample_index": 0,
            "seed": args.seed,
            **record,
            "filename": safe_filename(record["prompt"], 0),
        })
    stop = len(tasks) if args.limit is None else args.start_index + args.limit
    assigned = tasks[args.start_index:stop]
    plan = {
        "schema_version": 1,
        "suite": "vbench-1.0-mini-0.05-43",
        "method": args.variant,
        "model": str(args.model),
        "calibration_config": str(args.calibration_config) if args.calibration_config else None,
        "generation": {
            "size": [832, 480], "frames": 81, "fps": 16,
            "steps": 50, "sampler": "UniPCMultistepScheduler",
            "guidance_scale": 6.0, "flow_shift": 3.0, "seed": args.seed,
            "negative_prompt": OFFICIAL_NEGATIVE_PROMPT,
        },
        "tasks": assigned,
    }
    if args.dry_run:
        print(json.dumps({**plan, "tasks": assigned[:3], "task_count": len(assigned)}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "plan.json", plan)

    from diffusers import WanPipeline, WanTransformer3DModel
    from diffusers.utils import export_to_video

    device = torch.device(args.device)
    transformer = WanTransformer3DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device).eval()
    transformer.requires_grad_(False)
    if args.variant != "bf16":
        report = apply_wan_msfp_config(
            transformer, args.calibration_config, device,
            activation_mode="msfp" if args.variant == "mixup" else "signed",
        )
        if report.replaced_count != 300:
            raise RuntimeError(f"Expected 300 replaced linears, got {report.replaced_count}")
    pipe = WanPipeline.from_pretrained(args.model, transformer=transformer, torch_dtype=torch.bfloat16)
    if type(pipe.scheduler).__name__ != "UniPCMultistepScheduler":
        raise ValueError(f"Expected UniPC scheduler, got {type(pipe.scheduler).__name__}")
    if float(pipe.scheduler.config.flow_shift) != 3.0:
        raise ValueError(f"Expected flow_shift=3.0, got {pipe.scheduler.config.flow_shift}")
    pipe.enable_model_cpu_offload(device=device.index or 0)

    result_path = args.output_dir / "results.json"
    results = json.loads(result_path.read_text()) if result_path.is_file() else []
    known = {entry["task_index"]: entry for entry in results}
    for task in assigned:
        output = args.output_dir / task["filename"]
        if output.is_file() and output.stat().st_size > 0:
            known[task["task_index"]] = {**task, "status": "skipped", "path": str(output)}
            write_json(result_path, [known[key] for key in sorted(known)])
            continue
        free_gib = shutil.disk_usage(args.output_dir).free / 1024**3
        if free_gib < args.min_free_gib:
            raise RuntimeError(f"Only {free_gib:.1f} GiB free; stopping before disk pressure")
        started = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(task["seed"])
        frames = pipe(
            prompt=task["augmented_prompt"],
            negative_prompt=OFFICIAL_NEGATIVE_PROMPT,
            height=480, width=832, num_frames=81,
            num_inference_steps=50, guidance_scale=6.0,
            generator=generator,
        ).frames[0]
        temporary = output.with_name(output.stem + ".partial.mp4")
        export_to_video(frames, str(temporary), fps=16)
        temporary.replace(output)
        known[task["task_index"]] = {
            **task, "status": "complete", "path": str(output),
            "seconds": time.perf_counter() - started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(result_path, [known[key] for key in sorted(known)])
        del frames
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
