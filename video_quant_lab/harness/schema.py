"""Configuration schema for baseline commands and experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


def _require_string(data: dict[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: {key!r} must be a non-empty string")
    return value


def _string_list(value: Any, key: str, source: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{source}: {key!r} must be a list of strings")
    return tuple(value)


@dataclass(frozen=True)
class BaselineManifest:
    name: str
    repository_env: str
    repository_default: str | None
    python_env: str
    command: tuple[str, ...]
    output_arguments: tuple[str, ...]
    python_default: str | None = None
    source_url: str | None = None
    source_revision: str | None = None

    @classmethod
    def load(cls, path: Path) -> "BaselineManifest":
        data = json.loads(path.read_text())
        repository = data.get("repository")
        python = data.get("python")
        source = data.get("source", {})
        if not isinstance(repository, dict) or not isinstance(python, dict):
            raise ValueError(f"{path}: repository and python must be objects")
        if not isinstance(source, dict):
            raise ValueError(f"{path}: source must be an object")
        default = repository.get("default")
        if default is not None and not isinstance(default, str):
            raise ValueError(f"{path}: repository.default must be a string or null")
        return cls(
            name=_require_string(data, "name", path),
            repository_env=_require_string(repository, "env", path),
            repository_default=default,
            python_env=_require_string(python, "env", path),
            command=_string_list(data.get("command"), "command", path),
            output_arguments=_string_list(
                data.get("output_arguments", []), "output_arguments", path
            ),
            python_default=python.get("default"),
            source_url=source.get("url"),
            source_revision=source.get("revision"),
        )


@dataclass(frozen=True)
class Experiment:
    name: str
    baseline: str
    arguments: tuple[str, ...]
    environment: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "Experiment":
        data = json.loads(path.read_text())
        environment = data.get("environment", {})
        if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()
        ):
            raise ValueError(f"{path}: environment must map strings to strings")
        return cls(
            name=_require_string(data, "name", path),
            baseline=_require_string(data, "baseline", path),
            arguments=_string_list(data.get("arguments", []), "arguments", path),
            environment=environment,
        )
