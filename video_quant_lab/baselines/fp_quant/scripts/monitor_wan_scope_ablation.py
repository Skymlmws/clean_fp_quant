#!/usr/bin/env python3
"""Wait for Wan scope-ablation videos, run VBench, and summarize results."""

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
    now,
    stable_available_gpus,
)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
RUNNER = PROJECT_ROOT / "video_quant_lab/baselines/fp_quant/scripts/runners/run_wan_vbench_finalize.sh"
SUMMARIZER = PROJECT_ROOT / "video_quant_lab/baselines/fp_quant/scripts/evaluate/summarize_wan_scope_ablation.py"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/vbench/wan2.1-t2v-1.3b"
EXPERIMENT_DIR = PROJECT_ROOT / "vbench_results/wan2.1-t2v-1.3b/stratified-32-seed0"
RUNTIME_DIR = EXPERIMENT_DIR / "scope-ablation"
STATE_PATH = RUNTIME_DIR / "state.json"
EXPECTED_VIDEOS = 32
TASKS = (
    ("identity-mxfp4-w4a4-attention", "Identity + MXFP4 W4A4, Attention-only"),
    ("identity-mxfp4-w4a4-ffn", "Identity + MXFP4 W4A4, FFN-only"),
    ("givens-mxfp4-w4a4-attention", "Givens + MXFP4 W4A4, Attention-only"),
    ("givens-mxfp4-w4a4-ffn", "Givens + MXFP4 W4A4, FFN-only"),
)


def video_count(method_id: str) -> int:
    root = OUTPUT_ROOT / method_id / "stratified-32-seed0"
    return sum(path.stat().st_size > 0 for path in root.glob("*.mp4"))


def initial_state() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "new",
        "created_at": now(),
        "updated_at": now(),
        "active_pid": None,
        "tasks": {
            method_id: {"stage": "generating", "videos": video_count(method_id)}
            for method_id, _ in TASKS
        },
    }


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return initial_state()


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    atomic_json(STATE_PATH, state)


def ensure_manifest_methods() -> None:
    path = EXPERIMENT_DIR / "experiment.json"
    manifest = json.loads(path.read_text())
    existing = {method["id"] for method in manifest["methods"]}
    for method_id, label in TASKS:
        if method_id not in existing:
            manifest["methods"].append({"id": method_id, "label": label})
    atomic_json(path, manifest)


class Supervisor:
    def __init__(self) -> None:
        self.state = load_state()
        self.child: subprocess.Popen[Any] | None = None
        self.stop_requested = False
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

    def request_stop(self, _signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.state["status"] = "stopping"
        save_state(self.state)
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid, signal.SIGTERM)

    def wait_for_videos(self, poll_seconds: int, stale_seconds: int) -> bool:
        last_total = -1
        last_progress = time.monotonic()
        while not self.stop_requested:
            counts = {method_id: video_count(method_id) for method_id, _ in TASKS}
            total = sum(counts.values())
            for method_id, count in counts.items():
                task = self.state["tasks"].setdefault(method_id, {})
                task.update(stage="generated" if count >= EXPECTED_VIDEOS else "generating", videos=count)
            self.state["status"] = "waiting-for-videos"
            save_state(self.state)
            print(
                f"{now()} " + ", ".join(
                    f"{method_id}={count}/{EXPECTED_VIDEOS}" for method_id, count in counts.items()
                ),
                flush=True,
            )
            if all(count >= EXPECTED_VIDEOS for count in counts.values()):
                return True
            if total > last_total:
                last_total = total
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress >= stale_seconds:
                self.state["status"] = "stalled"
                self.state["error"] = f"No new completed video for {stale_seconds} seconds"
                save_state(self.state)
                return False
            time.sleep(poll_seconds)
        return False

    def evaluate(self, method_id: str, label: str, poll_seconds: int, shared_target: int) -> bool:
        if evaluation_complete(method_id):
            return True
        gpus = stable_available_gpus(3, shared_target, poll_seconds)
        log = RUNTIME_DIR / "logs" / method_id / "finalize.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = {
            **os.environ,
            "METHOD_ID": method_id,
            "METHOD": label,
            "GPU_QUALITY": str(gpus[0]),
            "GPU_SEMANTIC": str(gpus[1]),
            "GPU_OBJECT": str(gpus[2]),
        }
        print(f"{now()} evaluating {method_id} on GPUs {gpus}", flush=True)
        with log.open("a") as handle:
            self.child = subprocess.Popen(
                [str(RUNNER)],
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.state["active_pid"] = self.child.pid
            save_state(self.state)
            status = self.child.wait()
        self.child = None
        self.state["active_pid"] = None
        save_state(self.state)
        return status == 0 and evaluation_complete(method_id)

    def run(self) -> int:
        poll_seconds = int(os.environ.get("SCOPE_POLL_SECONDS", "60"))
        stale_seconds = int(os.environ.get("SCOPE_STALE_SECONDS", "2400"))
        shared_target = int(os.environ.get("SHARED_GPU_TARGET", "2"))
        ensure_manifest_methods()
        self.state["status"] = "running"
        self.state["supervisor_pid"] = os.getpid()
        save_state(self.state)
        if not self.wait_for_videos(poll_seconds, stale_seconds):
            return 130 if self.stop_requested else 1
        for method_id, label in TASKS:
            if self.stop_requested:
                self.state["status"] = "stopped"
                save_state(self.state)
                return 130
            task = self.state["tasks"][method_id]
            task["stage"] = "evaluating"
            save_state(self.state)
            if not self.evaluate(method_id, label, poll_seconds, shared_target):
                task["stage"] = "evaluation-failed"
                self.state["status"] = "failed"
                save_state(self.state)
                return 1
            task.update(stage="complete", dimensions=len(evaluated_dimensions(method_id)))
            save_state(self.state)
        subprocess.run(
            [
                sys.executable,
                str(SUMMARIZER),
                "--experiment-dir", str(EXPERIMENT_DIR),
                "--output-dir", str(RUNTIME_DIR),
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
        self.state["status"] = "complete"
        self.state["report"] = str(RUNTIME_DIR / "report.md")
        save_state(self.state)
        return 0


def print_status() -> None:
    state = load_state()
    print(f"scope ablation: {state['status']} (updated {state['updated_at']})")
    for method_id, _ in TASKS:
        task = state.get("tasks", {}).get(method_id, {})
        print(
            f"{method_id}: stage={task.get('stage', 'pending')}, "
            f"videos={video_count(method_id)}/{EXPECTED_VIDEOS}, "
            f"dimensions={len(evaluated_dimensions(method_id))}"
        )
    if state.get("active_pid"):
        print(f"active evaluation pid: {state['active_pid']}")


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
