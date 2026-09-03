"""Migrate call_NNN activation artifacts to step_NNN/{conditional,unconditional}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def replace_paths(value, replacements: dict[str, str]):
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [replace_paths(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_paths(item, replacements) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    outputs = (Path(__file__).resolve().parents[3] / "outputs").resolve()
    if outputs not in root.parents:
        raise ValueError(f"Root must be a child of {outputs}")

    call_dirs = sorted(path for path in root.rglob("call_[0-9][0-9][0-9]") if path.is_dir())
    replacements: dict[str, str] = {}
    moved = 0
    for call_dir in call_dirs:
        call = int(call_dir.name.split("_")[1])
        relative_old = call_dir.relative_to(root)
        branch = "conditional" if call % 2 == 0 else "unconditional"
        relative_new = relative_old.parent / f"step_{call // 2:03d}" / branch
        target = root / relative_new
        target.mkdir(parents=True, exist_ok=True)
        replacements[relative_old.as_posix()] = relative_new.as_posix()
        for child in list(call_dir.iterdir()):
            destination = target / child.name
            if destination.exists():
                raise FileExistsError(f"Migration target already exists: {destination}")
            shutil.move(str(child), str(destination))
            moved += 1
        call_dir.rmdir()

    for json_path in root.rglob("*.json"):
        data = json.loads(json_path.read_text())
        data = replace_paths(data, replacements)
        if json_path.name == "timestep.json" and json_path.parent.parent.name.startswith("step_"):
            data["sampling_step"] = int(json_path.parent.parent.name.split("_")[1])
            data["branch"] = json_path.parent.name
        json_path.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({"root": str(root), "call_directories": len(call_dirs), "entries_moved": moved}, indent=2))


if __name__ == "__main__":
    main()
