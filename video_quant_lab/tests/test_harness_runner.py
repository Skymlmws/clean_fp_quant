import json
from pathlib import Path

import pytest

from video_quant_lab.harness.runner import build_plan, execute
from video_quant_lab.harness.schema import BaselineManifest, Experiment


def test_build_plan_keeps_baseline_as_an_external_command(tmp_path, monkeypatch):
    repository = tmp_path / "upstream"
    repository.mkdir()
    monkeypatch.setenv("TEST_BASELINE_REPO", str(repository))
    monkeypatch.setenv("TEST_BASELINE_PYTHON", "/env/bin/python")
    manifest = BaselineManifest(
        name="original_method",
        repository_env="TEST_BASELINE_REPO",
        repository_default=None,
        python_env="TEST_BASELINE_PYTHON",
        command=("{python}", "upstream.py"),
        output_arguments=("--output", "{artifact_dir}"),
    )
    experiment = Experiment("trial", "original_method", ("--seed", "7"), {})

    plan = build_plan(experiment, manifest, tmp_path / "runs")

    assert plan.repository == repository
    assert plan.command == (
        "/env/bin/python",
        "upstream.py",
        "--seed",
        "7",
        "--output",
        str(plan.artifact_dir),
    )


def test_execute_records_command_logs_and_status(tmp_path):
    repository = tmp_path / "upstream"
    repository.mkdir()
    manifest = BaselineManifest(
        name="fixture",
        repository_env="UNUSED_REPOSITORY_ENV",
        repository_default=str(repository),
        python_env="UNUSED_PYTHON_ENV",
        command=("{python}", "-c", "print('baseline output')"),
        output_arguments=(),
    )
    experiment = Experiment("successful-run", "fixture", (), {})
    plan = build_plan(experiment, manifest, tmp_path / "runs")

    assert execute(plan) == 0
    record = json.loads((plan.run_dir / "run.json").read_text())
    assert record["status"] == "complete"
    assert record["return_code"] == 0
    assert (plan.run_dir / "stdout.log").read_text() == "baseline output\n"
    assert (plan.run_dir / "stderr.log").read_text() == ""


def test_execute_refuses_to_overwrite_an_existing_run(tmp_path):
    repository = tmp_path / "upstream"
    repository.mkdir()
    manifest = BaselineManifest(
        "fixture", "UNUSED_REPOSITORY_ENV", str(repository),
        "UNUSED_PYTHON_ENV", ("{python}", "-c", "pass"), (),
    )
    plan = build_plan(Experiment("existing", "fixture", (), {}), manifest, tmp_path)
    plan.run_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError, match="already exists"):
        execute(plan)


def test_run_name_allows_repeated_trials(tmp_path):
    repository = tmp_path / "upstream"
    repository.mkdir()
    manifest = BaselineManifest(
        "fixture", "UNUSED_REPOSITORY_ENV", str(repository),
        "UNUSED_PYTHON_ENV", ("{python}", "-c", "pass"), (),
    )
    experiment = Experiment("experiment-name", "fixture", (), {})

    plan = build_plan(experiment, manifest, tmp_path, run_name="trial-seed-7")

    assert plan.run_dir == (tmp_path / "trial-seed-7").resolve()
