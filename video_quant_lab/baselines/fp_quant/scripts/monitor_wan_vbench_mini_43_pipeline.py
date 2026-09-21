#!/usr/bin/env python3
"""Run the complete Wan VBench-mini-43 generation and evaluation matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from monitor_wan_baseline_pipeline import (
    atomic_json,
    evaluated_dimensions,
    evaluation_complete,
    gpu_snapshot,
    now,
    schedulable_count,
    stable_available_gpus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNNER_DIR = PROJECT_ROOT / "video_quant_lab/baselines/fp_quant/scripts/runners"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/vbench/wan2.1-t2v-1.3b"
EXPERIMENT_DIR = PROJECT_ROOT / "vbench_results/wan2.1-t2v-1.3b/vbench-mini-43-seed0"
EXPERIMENT_TEMPLATE = PROJECT_ROOT / "video_quant_lab/experiments/wan_vbench_mini_43.json"
RUNTIME_DIR = EXPERIMENT_DIR / "full-matrix"
STATE_PATH = RUNTIME_DIR / "state.json"
EXPECTED_VIDEOS = 43


def task(
    method_id: str,
    label: str,
    runner: str,
    *,
    environment: dict[str, str] | None = None,
    rank_sharded: bool = False,
) -> dict[str, Any]:
    return {
        "id": method_id,
        "label": label,
        "runner": runner,
        "environment": environment or {},
        "rank_sharded": rank_sharded,
    }


TASKS = (
    task("bf16", "BF16", "run_wan_vbench_bf16.sh", rank_sharded=True),
    task("identity-mxfp4-w4a16", "Identity + MXFP4 W4A16", "run_wan_vbench_identity_w4a16.sh"),
    task(
        "hadamard-mxfp4-w4a16",
        "Randomized Hadamard H32 + MXFP4 W4A16",
        "run_wan_vbench_hadamard_w4a16.sh",
    ),
    task("givens-mxfp4-w4a16", "Givens + MXFP4 W4A16", "run_wan_vbench_givens_w4a16.sh"),
    task("identity-mxfp4-w16a4", "Identity + MXFP4 W16A4", "run_wan_vbench_identity_w16a4.sh"),
    task(
        "hadamard-mxfp4-w16a4",
        "Randomized Hadamard H32 + MXFP4 W16A4",
        "run_wan_vbench_hadamard_w16a4.sh",
    ),
    task("identity-mxfp4-w4a4", "Identity + MXFP4 W4A4", "run_wan_vbench_identity_w4a4.sh"),
    task(
        "hadamard-mxfp-w4a4",
        "Randomized Hadamard H32 + MXFP4 W4A4",
        "run_wan_vbench_hadamard_w4a4.sh",
    ),
    task("givens-mxfp-w4a4", "Givens + MXFP4 W4A4", "run_wan_vbench_givens_w4a4.sh"),
    task(
        "identity-mxfp4-w4a4-attention",
        "Identity + MXFP4 W4A4, Attention-only",
        "run_wan_vbench_mxfp.sh",
        environment={
            "TRANSFORM_CLASS": "identity", "QUANT_SCOPE": "attention",
            "WEIGHT_BITS": "4", "ACTIVATION_BITS": "4",
        },
    ),
    task(
        "identity-mxfp4-w4a4-ffn",
        "Identity + MXFP4 W4A4, FFN-only",
        "run_wan_vbench_mxfp.sh",
        environment={
            "TRANSFORM_CLASS": "identity", "QUANT_SCOPE": "ffn",
            "WEIGHT_BITS": "4", "ACTIVATION_BITS": "4",
        },
    ),
    task(
        "givens-mxfp4-w4a4-attention",
        "Givens + MXFP4 W4A4, Attention-only",
        "run_wan_vbench_mxfp.sh",
        environment={
            "TRANSFORM_CLASS": "givens", "QUANT_SCOPE": "attention",
            "WEIGHT_BITS": "4", "ACTIVATION_BITS": "4",
        },
    ),
    task(
        "givens-mxfp4-w4a4-ffn",
        "Givens + MXFP4 W4A4, FFN-only",
        "run_wan_vbench_mxfp.sh",
        environment={
            "TRANSFORM_CLASS": "givens", "QUANT_SCOPE": "ffn",
            "WEIGHT_BITS": "4", "ACTIVATION_BITS": "4",
        },
    ),
    task(
        "attn-givens-ffn-identity-mxfp4-w4a4",
        "Attention Givens + FFN Identity MXFP4 W4A4",
        "run_wan_vbench_mxfp.sh",
        environment={
            "ATTENTION_TRANSFORM_CLASS": "givens",
            "FFN_TRANSFORM_CLASS": "identity",
            "QUANT_SCOPE": "all", "WEIGHT_BITS": "4", "ACTIVATION_BITS": "4",
        },
    ),
    task(
        "attn-identity-ffn-givens-mxfp4-w4a4",
        "Attention Identity + FFN Givens MXFP4 W4A4",
        "run_wan_vbench_mxfp.sh",
        environment={
            "ATTENTION_TRANSFORM_CLASS": "identity",
            "FFN_TRANSFORM_CLASS": "givens",
            "QUANT_SCOPE": "all", "WEIGHT_BITS": "4", "ACTIVATION_BITS": "4",
        },
    ),
)


def ensure_manifest() -> None:
    destination = EXPERIMENT_DIR / "experiment.json"
    if destination.exists():
        return
    value = json.loads(EXPERIMENT_TEMPLATE.read_text())
    atomic_json(destination, value)


def video_count(method_id: str) -> int:
    directory = OUTPUT_ROOT / method_id / "vbench-mini-43-seed0"
    return sum(path.stat().st_size > 0 for path in directory.glob("*.mp4"))


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "schema_version": 1,
        "protocol": "vbench-1.0-mini-0.05-43",
        "status": "new",
        "created_at": now(),
        "updated_at": now(),
        "active_pids": [],
        "tasks": {item["id"]: {"stage": "pending", "retries": 0} for item in TASKS},
    }


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    atomic_json(STATE_PATH, state)


def mark_manifest_complete(method_id: str) -> None:
    path = EXPERIMENT_DIR / "experiment.json"
    manifest = json.loads(path.read_text())
    for method in manifest["methods"]:
        if method["id"] == method_id:
            method["status"] = "complete"
            atomic_json(path, manifest)
            return
    raise KeyError(f"method is absent from experiment manifest: {method_id}")


class Supervisor:
    def __init__(self) -> None:
        self.state = load_state()
        self.children: list[subprocess.Popen[Any]] = []
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

    def request_stop(self, _signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.state["status"] = "stopping"
        save_state(self.state)
        for child in self.children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

    def run_group(self, commands: list[tuple[list[str], dict[str, str], Path]]) -> bool:
        handles = []
        self.children = []
        try:
            for command, environment, log_path in commands:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("a")
                handles.append(handle)
                child = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env={**os.environ, **environment},
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self.children.append(child)
            self.state["active_pids"] = [child.pid for child in self.children]
            save_state(self.state)
            while any(child.poll() is None for child in self.children):
                if self.stop_requested:
                    break
                time.sleep(5)
            return not self.stop_requested and all(child.wait() == 0 for child in self.children)
        finally:
            self.state["active_pids"] = []
            save_state(self.state)
            for handle in handles:
                handle.close()
            self.children = []

    def generate(self, item: dict[str, Any], poll_seconds: int, shared_target: int) -> bool:
        if video_count(item["id"]) >= EXPECTED_VIDEOS:
            return True
        maximum = int(os.environ.get("MAX_GENERATION_WORKERS", "8"))
        while not self.stop_requested:
            try:
                pool, idle = gpu_snapshot()
            except RuntimeError as error:
                print(f"{now()} {error}; retrying", flush=True)
                time.sleep(poll_seconds)
                continue
            worker_count = min(maximum, schedulable_count(len(pool), len(idle), shared_target))
            if worker_count < 1:
                stable_available_gpus(1, shared_target, poll_seconds)
                continue
            gpus = stable_available_gpus(worker_count, shared_target, poll_seconds)
            commands = []
            for worker_index, gpu in enumerate(gpus):
                environment = {
                    **item["environment"],
                    "DEVICE_ID": str(gpu),
                }
                if item["rank_sharded"]:
                    environment.update(RANK=str(worker_index), WORLD_SIZE=str(worker_count))
                else:
                    environment.update(
                        WORKER_INDEX=str(worker_index), WORKER_COUNT=str(worker_count)
                    )
                log = RUNTIME_DIR / "logs" / item["id"] / f"generate-worker{worker_index}.log"
                commands.append(([str(RUNNER_DIR / item["runner"])], environment, log))
            print(f"{now()} generation {item['id']} on GPUs {gpus}", flush=True)
            succeeded = self.run_group(commands)
            self.state["tasks"][item["id"]]["videos"] = video_count(item["id"])
            save_state(self.state)
            if video_count(item["id"]) >= EXPECTED_VIDEOS:
                return True
            if not succeeded:
                return False
        return False

    def evaluate(self, item: dict[str, Any], poll_seconds: int, shared_target: int) -> bool:
        if evaluation_complete(item["id"]):
            return True
        gpus = stable_available_gpus(3, shared_target, poll_seconds)
        environment = {
            "METHOD_ID": item["id"],
            "METHOD": item["label"],
            "GPU_QUALITY": str(gpus[0]),
            "GPU_SEMANTIC": str(gpus[1]),
            "GPU_OBJECT": str(gpus[2]),
        }
        log = RUNTIME_DIR / "logs" / item["id"] / "finalize.log"
        print(f"{now()} evaluation {item['id']} on GPUs {gpus}", flush=True)
        return self.run_group(
            [([str(RUNNER_DIR / "run_wan_vbench_finalize.sh")], environment, log)]
        ) and evaluation_complete(item["id"])

    def run(self) -> int:
        ensure_manifest()
        poll_seconds = int(os.environ.get("GPU_POLL_SECONDS", "60"))
        shared_target = int(os.environ.get("SHARED_GPU_TARGET", "2"))
        maximum_retries = int(os.environ.get("MAX_STAGE_RETRIES", "3"))
        retry_delay = int(os.environ.get("RETRY_DELAY_SECONDS", "120"))
        self.state.update(status="running", supervisor_pid=os.getpid())
        save_state(self.state)
        for item in TASKS:
            task_state = self.state["tasks"].setdefault(
                item["id"], {"stage": "pending", "retries": 0}
            )
            for stage, operation in (("generating", self.generate), ("evaluating", self.evaluate)):
                complete = (
                    video_count(item["id"]) >= EXPECTED_VIDEOS
                    if stage == "generating"
                    else evaluation_complete(item["id"])
                )
                while not complete and not self.stop_requested:
                    task_state["stage"] = stage
                    save_state(self.state)
                    if operation(item, poll_seconds, shared_target):
                        complete = True
                        break
                    task_state["retries"] = task_state.get("retries", 0) + 1
                    task_state["last_error"] = f"{stage} attempt failed at {now()}"
                    save_state(self.state)
                    if task_state["retries"] >= maximum_retries:
                        self.state["status"] = "failed"
                        save_state(self.state)
                        return 1
                    time.sleep(retry_delay)
            if self.stop_requested:
                self.state["status"] = "stopped"
                save_state(self.state)
                return 130
            task_state.update(
                stage="complete",
                videos=video_count(item["id"]),
                dimensions=len(evaluated_dimensions(item["id"])),
                completed_at=now(),
            )
            mark_manifest_complete(item["id"])
            save_state(self.state)
        self.state["status"] = "complete"
        save_state(self.state)
        return 0


def print_status() -> None:
    state = load_state()
    print(f"pipeline: {state['status']} (updated {state['updated_at']})")
    for item in TASKS:
        current = state["tasks"].get(item["id"], {})
        print(
            f"{item['id']}: stage={current.get('stage', 'pending')}, "
            f"videos={video_count(item['id'])}/{EXPECTED_VIDEOS}, "
            f"dimensions={len(evaluated_dimensions(item['id']))}/15, "
            f"retries={current.get('retries', 0)}"
        )


def print_plan() -> None:
    print(f"protocol=vbench-1.0-mini-0.05-43 methods={len(TASKS)} prompts=43")
    for index, item in enumerate(TASKS, 1):
        print(f"{index:02d}. {item['id']} — {item['label']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "status", "plan"))
    args = parser.parse_args()
    if args.command == "status":
        print_status()
        return 0
    if args.command == "plan":
        print_plan()
        return 0
    return Supervisor().run()


if __name__ == "__main__":
    sys.exit(main())
