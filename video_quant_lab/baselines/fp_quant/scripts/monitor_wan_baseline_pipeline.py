#!/usr/bin/env python3
"""Persistent, resumable Wan baseline generation and VBench supervisor."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNNER_DIR = PROJECT_ROOT / "video_quant_lab/baselines/fp_quant/scripts/runners"
EXPERIMENT_DIR = PROJECT_ROOT / "vbench_results/wan2.1-t2v-1.3b/stratified-32-seed0"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/vbench/wan2.1-t2v-1.3b"
RUNTIME_DIR = EXPERIMENT_DIR / "pipeline"
STATE_PATH = RUNTIME_DIR / "state.json"
EXPECTED_VIDEOS = 32
REQUIRED_DIMENSIONS = {
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "human_action", "spatial_relationship",
    "scene", "temporal_style", "appearance_style", "overall_consistency",
}
TASKS = (
    {
        "id": "identity-mxfp4-w16a4",
        "label": "Identity + MXFP4 W16A4",
        "runner": "run_wan_vbench_identity_w16a4.sh",
    },
    {
        "id": "hadamard-mxfp4-w16a4",
        "label": "Randomized Hadamard H32 + MXFP4 W16A4",
        "runner": "run_wan_vbench_hadamard_w16a4.sh",
    },
    {
        "id": "givens-mxfp4-w4a16",
        "label": "Givens + MXFP4 W4A16",
        "runner": "run_wan_vbench_givens_w4a16.sh",
    },
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "schema_version": 1,
        "status": "new",
        "created_at": now(),
        "updated_at": now(),
        "active_pids": [],
        "tasks": {task["id"]: {"stage": "pending", "retries": 0} for task in TASKS},
    }


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    atomic_json(STATE_PATH, state)


def video_count(method_id: str) -> int:
    directory = OUTPUT_ROOT / method_id / "stratified-32-seed0"
    return sum(path.stat().st_size > 0 for path in directory.glob("*.mp4"))


def evaluated_dimensions(method_id: str) -> set[str]:
    result_dir = EXPERIMENT_DIR / "methods" / method_id
    found: set[str] = set()
    for path in result_dir.rglob("*_eval_results.json"):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            found.update(set(value) & REQUIRED_DIMENSIONS)
    return found


def evaluation_complete(method_id: str) -> bool:
    report = EXPERIMENT_DIR / "methods" / method_id / "report.md"
    return report.is_file() and REQUIRED_DIMENSIONS <= evaluated_dimensions(method_id)


def mark_manifest_complete(method_id: str) -> None:
    path = EXPERIMENT_DIR / "experiment.json"
    manifest = json.loads(path.read_text())
    for method in manifest["methods"]:
        if method["id"] == method_id:
            method["status"] = "complete"
            atomic_json(path, manifest)
            return
    raise KeyError(f"method is absent from experiment.json: {method_id}")


def parse_allowlist() -> set[int] | None:
    raw = os.environ.get("GPU_ALLOWLIST") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw:
        return None
    try:
        return {int(item.strip()) for item in raw.split(",") if item.strip()}
    except ValueError as error:
        raise ValueError("GPU_ALLOWLIST must contain comma-separated physical GPU IDs") from error


def gpu_snapshot() -> tuple[list[int], list[int]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=int(os.environ.get("NVIDIA_SMI_TIMEOUT_SECONDS", "15")),
        )
    except (subprocess.SubprocessError, OSError) as error:
        raise RuntimeError(f"GPU discovery failed: {error}") from error
    allowlist = parse_allowlist()
    maximum_used = int(os.environ.get("GPU_MAX_USED_MIB", "1000"))
    maximum_util = int(os.environ.get("GPU_MAX_UTIL_PERCENT", "10"))
    pool = []
    idle = []
    for line in output.splitlines():
        index, used, utilization = (int(part.strip()) for part in line.split(","))
        if allowlist is not None and index not in allowlist:
            continue
        pool.append(index)
        if used <= maximum_used and utilization <= maximum_util:
            idle.append(index)
    return pool, idle


def schedulable_count(pool_count: int, idle_count: int, shared_gpu_target: int) -> int:
    """Return idle GPUs usable while keeping the target outside this pipeline."""
    occupied_count = pool_count - idle_count
    additionally_reserved = max(0, shared_gpu_target - occupied_count)
    return max(0, idle_count - additionally_reserved)


def idle_gpus() -> list[int]:
    return gpu_snapshot()[1]


def stable_available_gpus(required: int, shared_gpu_target: int, poll_seconds: int) -> list[int]:
    while True:
        try:
            first_pool, first = gpu_snapshot()
        except RuntimeError as error:
            print(f"{now()} {error}; retrying", flush=True)
            time.sleep(poll_seconds)
            continue
        usable = schedulable_count(len(first_pool), len(first), shared_gpu_target)
        if usable >= required:
            time.sleep(min(5, poll_seconds))
            try:
                second_pool, second = gpu_snapshot()
            except RuntimeError as error:
                print(f"{now()} {error}; retrying", flush=True)
                time.sleep(poll_seconds)
                continue
            stable = [gpu for gpu in first if gpu in second]
            stable_usable = schedulable_count(
                len(second_pool), len(stable), shared_gpu_target
            )
            if stable_usable >= required:
                return stable[:required]
        print(
            f"{now()} waiting for GPUs: idle={first}, required={required}, "
            f"shared_gpu_target={shared_gpu_target}",
            flush=True,
        )
        time.sleep(poll_seconds)


class Supervisor:
    def __init__(self) -> None:
        self.state = load_state()
        self.children: list[subprocess.Popen[Any]] = []
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

    def request_stop(self, signum: int, _frame: Any) -> None:
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

    def generate(self, task: dict[str, str], poll_seconds: int, shared_gpu_target: int) -> bool:
        if video_count(task["id"]) >= EXPECTED_VIDEOS:
            return True
        max_workers = int(os.environ.get("MAX_GENERATION_WORKERS", "8"))
        while not self.stop_requested:
            try:
                pool, idle = gpu_snapshot()
            except RuntimeError as error:
                print(f"{now()} {error}; retrying", flush=True)
                time.sleep(poll_seconds)
                continue
            worker_count = min(
                max_workers,
                schedulable_count(len(pool), len(idle), shared_gpu_target),
            )
            if worker_count < 1:
                stable_available_gpus(1, shared_gpu_target, poll_seconds)
                continue
            gpus = stable_available_gpus(worker_count, shared_gpu_target, poll_seconds)
            commands = []
            for worker_index, gpu in enumerate(gpus):
                log = RUNTIME_DIR / "logs" / task["id"] / f"generate-worker{worker_index}.log"
                environment = {
                    "DEVICE_ID": str(gpu),
                    "WORKER_INDEX": str(worker_index),
                    "WORKER_COUNT": str(worker_count),
                }
                commands.append(([str(RUNNER_DIR / task["runner"])], environment, log))
            print(f"{now()} generation {task['id']} on GPUs {gpus}", flush=True)
            succeeded = self.run_group(commands)
            count = video_count(task["id"])
            self.state["tasks"][task["id"]]["videos"] = count
            save_state(self.state)
            if count >= EXPECTED_VIDEOS:
                return True
            if not succeeded:
                return False
        return False

    def evaluate(self, task: dict[str, str], poll_seconds: int, shared_gpu_target: int) -> bool:
        if evaluation_complete(task["id"]):
            return True
        gpus = stable_available_gpus(3, shared_gpu_target, poll_seconds)
        log = RUNTIME_DIR / "logs" / task["id"] / "finalize.log"
        environment = {
            "METHOD_ID": task["id"],
            "METHOD": task["label"],
            "GPU_QUALITY": str(gpus[0]),
            "GPU_SEMANTIC": str(gpus[1]),
            "GPU_OBJECT": str(gpus[2]),
        }
        print(f"{now()} evaluation {task['id']} on GPUs {gpus}", flush=True)
        succeeded = self.run_group(
            [([str(RUNNER_DIR / "run_wan_vbench_finalize.sh")], environment, log)]
        )
        return succeeded and evaluation_complete(task["id"])

    def run(self) -> int:
        poll_seconds = int(os.environ.get("GPU_POLL_SECONDS", "60"))
        shared_gpu_target = int(os.environ.get("SHARED_GPU_TARGET", "2"))
        max_retries = int(os.environ.get("MAX_STAGE_RETRIES", "3"))
        retry_delay = int(os.environ.get("RETRY_DELAY_SECONDS", "120"))
        self.state["status"] = "running"
        self.state["supervisor_pid"] = os.getpid()
        save_state(self.state)
        for task in TASKS:
            task_state = self.state["tasks"].setdefault(task["id"], {})
            for stage, operation in (("generating", self.generate), ("evaluating", self.evaluate)):
                complete = (
                    video_count(task["id"]) >= EXPECTED_VIDEOS
                    if stage == "generating"
                    else evaluation_complete(task["id"])
                )
                while not complete and not self.stop_requested:
                    task_state["stage"] = stage
                    save_state(self.state)
                    if operation(task, poll_seconds, shared_gpu_target):
                        complete = True
                        break
                    task_state["retries"] = task_state.get("retries", 0) + 1
                    task_state["last_error"] = f"{stage} attempt failed at {now()}"
                    save_state(self.state)
                    if task_state["retries"] >= max_retries:
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
                videos=video_count(task["id"]),
                dimensions=len(evaluated_dimensions(task["id"])),
                completed_at=now(),
            )
            mark_manifest_complete(task["id"])
            save_state(self.state)
        self.state["status"] = "complete"
        save_state(self.state)
        return 0


def print_status() -> None:
    state = load_state()
    print(f"pipeline: {state['status']} (updated {state['updated_at']})")
    for task in TASKS:
        task_state = state["tasks"].get(task["id"], {})
        print(
            f"{task['id']}: stage={task_state.get('stage', 'pending')}, "
            f"videos={video_count(task['id'])}/{EXPECTED_VIDEOS}, "
            f"dimensions={len(evaluated_dimensions(task['id']))}/15, "
            f"retries={task_state.get('retries', 0)}"
        )
    if state.get("active_pids"):
        print("active pids:", ", ".join(map(str, state["active_pids"])))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "status"))
    args = parser.parse_args()
    if args.command == "status":
        print_status()
        return 0
    return Supervisor().run()


if __name__ == "__main__":
    sys.exit(main())
