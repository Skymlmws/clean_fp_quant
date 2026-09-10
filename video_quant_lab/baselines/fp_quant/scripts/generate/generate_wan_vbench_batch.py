"""Generate a resumable VBench subset with one resident Wan pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any


DIMENSIONS = (
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality",
    "imaging_quality", "object_class", "multiple_objects", "human_action",
    "color", "spatial_relationship", "scene", "temporal_style",
    "appearance_style", "overall_consistency",
)

OFFICIAL_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
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
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--min-free-gib", type=float, default=200.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_records(metadata: Path, augmented_prompts: Path) -> list[dict[str, Any]]:
    records = json.loads(metadata.read_text())
    prompts = augmented_prompts.read_text().splitlines()
    if not isinstance(records, list) or len(records) != len(prompts):
        raise ValueError(
            f"Official metadata/prompts must align: {len(records)} != {len(prompts)}"
        )
    merged = []
    for index, (record, augmented) in enumerate(zip(records, prompts, strict=True)):
        dimensions = record.get("dimension")
        if not isinstance(record.get("prompt_en"), str) or not isinstance(dimensions, list):
            raise ValueError(f"Invalid VBench metadata record {index}")
        merged.append({
            "source_index": index,
            "prompt": record["prompt_en"],
            "augmented_prompt": augmented,
            "dimensions": dimensions,
            "auxiliary_info": record.get("auxiliary_info"),
        })
    return merged


def select_stratified(
    records: list[dict[str, Any]], count: int, selection_seed: int
) -> list[dict[str, Any]]:
    if count < 2 * len(DIMENSIONS) or count > len(records):
        raise ValueError(f"prompt-count must be between {2 * len(DIMENSIONS)} and {len(records)}")
    rng = random.Random(selection_seed)
    selected: list[int] = []
    used: set[int] = set()
    for dimension in DIMENSIONS:
        pool = [i for i, record in enumerate(records) if dimension in record["dimensions"]]
        rng.shuffle(pool)
        chosen = [i for i in pool if i not in used][:2]
        if len(chosen) != 2:
            raise ValueError(f"Could not select two unique prompts for {dimension}")
        selected.extend(chosen)
        used.update(chosen)
    remaining = [i for i in range(len(records)) if i not in used]
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    return [records[index] for index in selected]


def safe_filename(prompt: str, sample_index: int) -> str:
    name = prompt.replace("/", "_").replace("\0", "").strip()
    suffix = f"-{sample_index}.mp4"
    while len((name + suffix).encode()) > 240:
        name = name[:-1]
    return name + suffix


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world-size)")
    selected = select_stratified(
        load_records(args.metadata, args.augmented_prompts),
        args.prompt_count,
        args.selection_seed,
    )
    tasks = []
    for prompt_index, record in enumerate(selected):
        for sample_index, seed in enumerate(args.sample_seeds):
            task = {
                "task_index": len(tasks), "prompt_index": prompt_index,
                "sample_index": sample_index, "seed": seed, **record,
                "filename": safe_filename(record["prompt"], sample_index),
            }
            tasks.append(task)
    assigned = [task for task in tasks if task["task_index"] % args.world_size == args.rank]
    plan = {
        "schema_version": 1,
        "suite": f"vbench-stratified-{args.prompt_count}",
        "method": "bf16",
        "selection_seed": args.selection_seed,
        "sample_seeds": args.sample_seeds,
        "official_wan_config": {
            "size": [832, 480], "frames": 81, "fps": 16,
            "steps": 50, "sampler": "unipc", "guide_scale": 6.0,
            "flow_shift": 3.0, "prompt_source": str(args.augmented_prompts.resolve()),
            "negative_prompt": OFFICIAL_NEGATIVE_PROMPT,
        },
        "rank": args.rank, "world_size": args.world_size, "tasks": assigned,
    }
    if args.dry_run:
        print(json.dumps({**plan, "tasks": assigned[:3], "task_count": len(assigned)}, ensure_ascii=False, indent=2))
        return
    if not (args.checkpoint / "diffusion_pytorch_model.safetensors").is_file():
        raise FileNotFoundError(f"Wan checkpoint not found: {args.checkpoint}")
    if not (args.wan_repo / "wan").is_dir():
        raise FileNotFoundError(f"Wan repository not found: {args.wan_repo}")

    import torch
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"plan-rank{args.rank}.json", plan)
    sys.path.insert(0, str(args.wan_repo.resolve()))
    from wan.configs import WAN_CONFIGS
    from wan.text2video import WanT2V
    from wan.utils.utils import cache_video

    pipe = WanT2V(
        config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=str(args.checkpoint),
        device_id=args.device_id, t5_cpu=True,
    )
    results: list[dict[str, Any]] = []
    result_path = args.output_dir / f"results-rank{args.rank}.json"
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
        video = pipe.generate(
            input_prompt=task["augmented_prompt"], size=(832, 480), frame_num=81,
            shift=3.0, sample_solver="unipc", sampling_steps=50, guide_scale=6.0,
            n_prompt=OFFICIAL_NEGATIVE_PROMPT, seed=task["seed"], offload_model=False,
        )
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
