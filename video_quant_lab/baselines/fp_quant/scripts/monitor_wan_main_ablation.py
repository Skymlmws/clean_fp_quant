#!/usr/bin/env python3
"""Monitor mixed-transform runs, evaluate them, and summarize the main ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from monitor_wan_baseline_pipeline import atomic_json, evaluation_complete, now, stable_available_gpus


PROJECT_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/vbench/wan2.1-t2v-1.3b"
EXPERIMENT_DIR = PROJECT_ROOT / "vbench_results/wan2.1-t2v-1.3b/stratified-32-seed0"
RUNTIME_DIR = EXPERIMENT_DIR / "main-ablation"
STATE_PATH = RUNTIME_DIR / "state.json"
FINALIZER = PROJECT_ROOT / "video_quant_lab/baselines/fp_quant/scripts/runners/run_wan_vbench_finalize.sh"
SUMMARIZER = PROJECT_ROOT / "video_quant_lab/baselines/fp_quant/scripts/evaluate/summarize_wan_main_ablation.py"
EXPECTED = 32
TASKS = (
    ("attn-givens-ffn-identity-mxfp4-w4a4", "Attention Givens + FFN Identity MXFP4 W4A4"),
    ("attn-identity-ffn-givens-mxfp4-w4a4", "Attention Identity + FFN Givens MXFP4 W4A4"),
)


def count(method_id: str) -> int:
    root = OUTPUT_ROOT / method_id / "stratified-32-seed0"
    return sum(path.stat().st_size > 0 for path in root.glob("*.mp4"))


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"schema_version": 1, "status": "new", "created_at": now(), "tasks": {}}


def save(state: dict) -> None:
    state["updated_at"] = now()
    atomic_json(STATE_PATH, state)


def ensure_manifest() -> None:
    path = EXPERIMENT_DIR / "experiment.json"
    manifest = json.loads(path.read_text())
    existing = {method["id"] for method in manifest["methods"]}
    for method_id, label in TASKS:
        if method_id not in existing:
            manifest["methods"].append({"id": method_id, "label": label})
    atomic_json(path, manifest)


def run() -> int:
    poll = int(os.environ.get("MAIN_ABLATION_POLL_SECONDS", "60"))
    stale_limit = int(os.environ.get("MAIN_ABLATION_STALE_SECONDS", "2400"))
    shared_target = int(os.environ.get("SHARED_GPU_TARGET", "2"))
    state = load_state()
    state.update(status="waiting-for-videos", supervisor_pid=os.getpid())
    ensure_manifest()
    last_total = -1
    last_progress = time.monotonic()
    while True:
        counts = {method_id: count(method_id) for method_id, _ in TASKS}
        for method_id, value in counts.items():
            state["tasks"].setdefault(method_id, {}).update(
                stage="generated" if value >= EXPECTED else "generating", videos=value
            )
        save(state)
        print(f"{now()} " + ", ".join(f"{key}={value}/{EXPECTED}" for key, value in counts.items()), flush=True)
        if all(value >= EXPECTED for value in counts.values()):
            break
        total = sum(counts.values())
        if total > last_total:
            last_total, last_progress = total, time.monotonic()
        elif time.monotonic() - last_progress >= stale_limit:
            state.update(status="stalled", error=f"No new video for {stale_limit} seconds")
            save(state)
            return 1
        time.sleep(poll)

    for method_id, label in TASKS:
        task = state["tasks"][method_id]
        if evaluation_complete(method_id):
            task["stage"] = "complete"
            save(state)
            continue
        task["stage"] = "evaluating"
        save(state)
        gpus = stable_available_gpus(3, shared_target, poll)
        log = RUNTIME_DIR / "logs" / method_id / "finalize.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = {
            **os.environ, "METHOD_ID": method_id, "METHOD": label,
            "GPU_QUALITY": str(gpus[0]), "GPU_SEMANTIC": str(gpus[1]),
            "GPU_OBJECT": str(gpus[2]),
        }
        print(f"{now()} evaluating {method_id} on GPUs {gpus}", flush=True)
        with log.open("a") as handle:
            status = subprocess.run(
                [str(FINALIZER)], cwd=PROJECT_ROOT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT,
            ).returncode
        if status or not evaluation_complete(method_id):
            task["stage"] = "evaluation-failed"
            state["status"] = "failed"
            save(state)
            return 1
        task["stage"] = "complete"
        save(state)
    subprocess.run(
        [sys.executable, str(SUMMARIZER), "--experiment-dir", str(EXPERIMENT_DIR),
         "--output-dir", str(RUNTIME_DIR)],
        cwd=PROJECT_ROOT, check=True,
    )
    state.update(status="complete", report=str(RUNTIME_DIR / "report.md"))
    save(state)
    return 0


def status() -> None:
    state = load_state()
    print(f"main ablation: {state['status']} (updated {state.get('updated_at', 'never')})")
    for method_id, _ in TASKS:
        task = state.get("tasks", {}).get(method_id, {})
        print(f"{method_id}: stage={task.get('stage', 'pending')}, videos={count(method_id)}/{EXPECTED}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "status"))
    args = parser.parse_args()
    sys.exit(run() if args.command == "run" else (status() or 0))
