"""Inspect and explicitly remove downloaded Wan visualization batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def usage(path: Path) -> tuple[int, int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return sum(item.stat().st_size for item in files), len(files), sum(item.name == "bars.png" for item in files)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("path", type=Path)
    acknowledge = subparsers.add_parser("acknowledge-download")
    acknowledge.add_argument("path", type=Path)
    acknowledge.add_argument("--delete", action="store_true", required=True)
    acknowledge.add_argument("--confirmation", required=True, choices=("downloaded",))
    args = parser.parse_args()

    path = args.path.resolve()
    outputs_root = (Path(__file__).resolve().parent / "outputs").resolve()
    if path != outputs_root and outputs_root not in path.parents:
        raise ValueError(f"Path must be {outputs_root} or one of its children")
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Batch directory does not exist: {path}")
    size, files, images = usage(path)
    if args.command == "status":
        state_path = path / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "unknown"}
        print(json.dumps({"path": str(path), "bytes": size, "gb": size / 1024**3, "files": files, "images": images, **state}, indent=2))
        return
    if path == outputs_root:
        raise ValueError("Refusing to delete the outputs root; select one downloaded batch directory")
    shutil.rmtree(path)
    print(json.dumps({"deleted": str(path), "freed_bytes": size, "recoverable": False}, indent=2))


if __name__ == "__main__":
    main()
