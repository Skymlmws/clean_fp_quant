#!/usr/bin/env python3
"""Evaluate Wan signed-FP4 and MSFP output error on held-out real trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import torch

from scripts.generate.generate_wan_vbench_batch import OFFICIAL_NEGATIVE_PROMPT
from src.quantization.wan_rtn import apply_wan_msfp_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/root/autodl-tmp/Wan2.1-T2V-1.3B-Diffusers"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--calibration-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=10)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--capture-step-indices", type=int, nargs="+", default=(0, 2, 5, 7, 9))
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def clone_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, tuple):
        return tuple(clone_cpu(item) for item in value)
    if isinstance(value, list):
        return [clone_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_cpu(item) for key, item in value.items()}
    return value


def move_to(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_to(item, device) for item in value)
    if isinstance(value, list):
        return [move_to(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move_to(item, device) for key, item in value.items()}
    return value


def output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    raise TypeError(f"Unsupported transformer output type: {type(output)}")


class TrajectoryCollector:
    def __init__(self, transformer: torch.nn.Module, capture_indices: set[int]) -> None:
        self.transformer = transformer
        self.capture_indices = capture_indices
        self.records: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.prompt_index = -1
        self.step_index = -1
        self.last_timestep: float | None = None
        self.call_index = 0
        self.pending: dict[str, Any] | None = None

    def reset_prompt(self, prompt_index: int) -> None:
        self.prompt_index = prompt_index
        self.step_index = -1
        self.last_timestep = None
        self.call_index = 0
        self.pending = None

    def before(self, _module, args, kwargs) -> None:
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 1:
            timestep = args[1]
        timestep_value = float(timestep.detach().float().reshape(-1)[0].cpu())
        if timestep_value != self.last_timestep:
            self.step_index += 1
            self.last_timestep = timestep_value
            self.call_index = 0
        if self.step_index in self.capture_indices:
            self.pending = {
                "prompt_index": self.prompt_index,
                "step_index": self.step_index,
                "call_index": self.call_index,
                "timestep": timestep_value,
                "args": clone_cpu(args),
                "kwargs": clone_cpu(kwargs),
            }
        else:
            self.pending = None
        self.call_index += 1

    def after(self, _module, _args, _kwargs, output) -> None:
        if self.pending is None:
            return
        self.pending["reference"] = clone_cpu(output_tensor(output))
        self.records.append(self.pending)
        self.pending = None

    def install(self) -> None:
        self.handles.append(self.transformer.register_forward_pre_hook(self.before, with_kwargs=True))
        self.handles.append(self.transformer.register_forward_hook(self.after, with_kwargs=True))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def empty_cuda() -> None:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def load_prompts(plan_path: Path, maximum: int) -> list[str]:
    tasks = json.loads(plan_path.read_text())["tasks"][:maximum]
    return [task.get("augmented_prompt", task["prompt"]) for task in tasks]


def collect_records(args: argparse.Namespace, prompts: list[str]) -> list[dict[str, Any]]:
    from diffusers import WanPipeline, WanTransformer3DModel

    device = torch.device(args.device)
    transformer = WanTransformer3DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device).eval()
    transformer.requires_grad_(False)
    collector = TrajectoryCollector(transformer, set(args.capture_step_indices))
    collector.install()
    pipe = WanPipeline.from_pretrained(args.model, transformer=transformer, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload(device=device.index or 0)
    try:
        for prompt_index, prompt in enumerate(prompts):
            collector.reset_prompt(prompt_index)
            generator = torch.Generator(device="cpu").manual_seed(args.seed)
            pipe(
                prompt=prompt,
                negative_prompt=OFFICIAL_NEGATIVE_PROMPT,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                output_type="latent",
            )
            print(f"captured prompt {prompt_index + 1}/{len(prompts)}: {len(collector.records)} records", flush=True)
    finally:
        collector.remove()
    records = collector.records
    del pipe, transformer
    empty_cuda()
    return records


def metric_parts(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | int]:
    error = reference.float() - candidate.float()
    return {
        "numel": error.numel(),
        "error_sum": float(error.double().square().sum()),
        "signal_sum": float(reference.double().square().sum()),
        "max_abs_error": float(error.abs().max()),
    }


def finalize(parts: list[dict[str, float | int]]) -> dict[str, float | int]:
    numel = sum(int(part["numel"]) for part in parts)
    error_sum = sum(float(part["error_sum"]) for part in parts)
    signal_sum = sum(float(part["signal_sum"]) for part in parts)
    return {
        "samples": len(parts),
        "numel": numel,
        "mse": error_sum / numel,
        "sqnr_db": 10.0 * math.log10(signal_sum / error_sum),
        "max_abs_error": max(float(part["max_abs_error"]) for part in parts),
    }


def evaluate_variant(args: argparse.Namespace, records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    from diffusers import WanTransformer3DModel

    device = torch.device(args.device)
    model = WanTransformer3DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device).eval()
    model.requires_grad_(False)
    report = apply_wan_msfp_config(model, args.calibration_config, device, activation_mode=mode)
    if report.replaced_count != 300:
        raise RuntimeError(f"Expected 300 quantized Linear layers, got {report.replaced_count}")
    results = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index, record in enumerate(records):
            call_args = move_to(record["args"], device)
            call_kwargs = move_to(record["kwargs"], device)
            candidate = output_tensor(model(*call_args, **call_kwargs)).cpu()
            parts = metric_parts(record["reference"], candidate)
            results.append({
                "prompt_index": record["prompt_index"],
                "step_index": record["step_index"],
                "call_index": record["call_index"],
                "timestep": record["timestep"],
                **parts,
                "mse": float(parts["error_sum"]) / int(parts["numel"]),
                "sqnr_db": 10.0 * math.log10(float(parts["signal_sum"]) / float(parts["error_sum"])),
            })
            print(f"{mode} sample {index + 1}/{len(records)}", flush=True)
    del model
    empty_cuda()
    return results


def main() -> None:
    args = parse_args()
    if max(args.capture_step_indices) >= args.steps:
        raise ValueError("capture step indices must be smaller than steps")
    prompts = load_prompts(args.plan, args.max_prompts)
    records = collect_records(args, prompts)
    expected = len(prompts) * len(args.capture_step_indices) * 2
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} CFG records, captured {len(records)}")

    signed = evaluate_variant(args, records, "signed")
    msfp = evaluate_variant(args, records, "msfp")
    signed_summary = finalize(signed)
    msfp_summary = finalize(msfp)
    paired = []
    prompt_parts: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"signed": [], "msfp": []})
    for signed_item, msfp_item in zip(signed, msfp, strict=True):
        improvement = 1.0 - msfp_item["mse"] / signed_item["mse"]
        paired.append({
            "prompt_index": signed_item["prompt_index"],
            "step_index": signed_item["step_index"],
            "call_index": signed_item["call_index"],
            "timestep": signed_item["timestep"],
            "signed": signed_item,
            "msfp": msfp_item,
            "msfp_mse_improvement_fraction": improvement,
        })
        prompt_parts[int(signed_item["prompt_index"])]["signed"].append(signed_item)
        prompt_parts[int(signed_item["prompt_index"])]["msfp"].append(msfp_item)
    per_prompt = []
    for prompt_index in range(len(prompts)):
        signed_prompt = finalize(prompt_parts[prompt_index]["signed"])
        msfp_prompt = finalize(prompt_parts[prompt_index]["msfp"])
        per_prompt.append({
            "prompt_index": prompt_index,
            "prompt": prompts[prompt_index],
            "signed": signed_prompt,
            "msfp": msfp_prompt,
            "msfp_mse_improvement_fraction": 1.0 - msfp_prompt["mse"] / signed_prompt["mse"],
        })
    payload = {
        "schema_version": 1,
        "model": str(args.model),
        "plan": str(args.plan),
        "calibration_config": str(args.calibration_config),
        "held_out_prompt_count": len(prompts),
        "trajectory": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "steps": args.steps,
            "capture_step_indices": args.capture_step_indices,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "cfg_calls_per_step": 2,
            "paired_samples": len(records),
        },
        "summary": {
            "signed": signed_summary,
            "msfp": msfp_summary,
            "msfp_mse_improvement_fraction": 1.0 - msfp_summary["mse"] / signed_summary["mse"],
            "msfp_sqnr_improvement_db": msfp_summary["sqnr_db"] - signed_summary["sqnr_db"],
            "msfp_wins": sum(item["msfp"]["mse"] < item["signed"]["mse"] for item in paired),
            "paired_samples": len(paired),
        },
        "per_prompt": per_prompt,
        "paired_samples": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))
    print(f"result: {args.output}")


if __name__ == "__main__":
    main()
