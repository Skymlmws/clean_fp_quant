"""Run an external baseline without importing or modifying its implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .schema import BaselineManifest, Experiment


@dataclass(frozen=True)
class RunPlan:
    experiment: Experiment
    manifest: BaselineManifest
    repository: Path
    run_dir: Path
    artifact_dir: Path
    command: tuple[str, ...]


def _expand(value: str, variables: dict[str, str]) -> str:
    expanded = os.path.expandvars(value)
    try:
        return expanded.format_map(variables)
    except KeyError as error:
        raise ValueError(f"Unknown command placeholder: {error.args[0]}") from error


def build_plan(
    experiment: Experiment,
    manifest: BaselineManifest,
    output_root: Path,
    run_name: str | None = None,
) -> RunPlan:
    if experiment.baseline != manifest.name:
        raise ValueError(
            f"Experiment requests {experiment.baseline!r}, manifest is {manifest.name!r}"
        )
    repository_value = os.environ.get(manifest.repository_env, manifest.repository_default)
    if not repository_value:
        raise ValueError(
            f"Set {manifest.repository_env} to the baseline repository directory"
        )
    repository = Path(os.path.expandvars(repository_value)).expanduser().resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"Baseline repository not found: {repository}")

    run_dir = (output_root / (run_name or experiment.name)).resolve()
    artifact_dir = run_dir / "artifacts"
    python = os.environ.get(manifest.python_env) or manifest.python_default or sys.executable
    python_path = Path(os.path.expandvars(python)).expanduser()
    if not python_path.is_absolute():
        python_path = Path.cwd() / python_path
    # Keep the configured path rather than resolving symlinks. Virtualenv Python
    # executables are commonly symlinks to the system interpreter, but invoking
    # the symlink is what activates that environment's package search path.
    python = str(python_path.absolute())
    variables = {
        "python": python,
        "repository": str(repository),
        "run_dir": str(run_dir),
        "artifact_dir": str(artifact_dir),
    }
    command = tuple(
        _expand(value, variables)
        for value in (*manifest.command, *experiment.arguments, *manifest.output_arguments)
    )
    return RunPlan(experiment, manifest, repository, run_dir, artifact_dir, command)


def _git_commit(repository: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def plan_record(plan: RunPlan) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": asdict(plan.experiment),
        "baseline": plan.manifest.name,
        "source_url": plan.manifest.source_url,
        "source_revision": plan.manifest.source_revision,
        "repository": str(plan.repository),
        "repository_commit": _git_commit(plan.repository),
        "run_dir": str(plan.run_dir),
        "artifact_dir": str(plan.artifact_dir),
        "command": list(plan.command),
    }


def execute(plan: RunPlan) -> int:
    if plan.run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {plan.run_dir}")
    plan.artifact_dir.mkdir(parents=True)
    record = plan_record(plan)
    record["started_at"] = datetime.now(timezone.utc).isoformat()
    record["status"] = "running"
    record_path = plan.run_dir / "run.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    environment = os.environ.copy()
    environment.update(plan.experiment.environment)
    started = time.perf_counter()
    with (plan.run_dir / "stdout.log").open("w") as stdout, (
        plan.run_dir / "stderr.log"
    ).open("w") as stderr:
        result = subprocess.run(
            plan.command,
            cwd=plan.repository,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    record["duration_seconds"] = time.perf_counter() - started
    record["return_code"] = result.returncode
    record["status"] = "complete" if result.returncode == 0 else "failed"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return result.returncode
