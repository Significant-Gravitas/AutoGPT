#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


SHARDS = {
    "data": ["backend/data"],
    "copilot": ["backend/copilot"],
    "util-executor": ["backend/util", "backend/executor"],
}


def shard_args(shard: str) -> list[str]:
    if shard in SHARDS:
        return ["-c", "pyproject.toml", *SHARDS[shard]]
    if shard != "remainder":
        raise ValueError(f"Unknown backend test shard: {shard}")
    return [
        "-c",
        "pyproject.toml",
        ".",
        "../../.github/scripts",
        *(f"--ignore={path}" for paths in SHARDS.values() for path in paths),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", choices=[*SHARDS, "remainder"])
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    if args.shard == "data":
        os.environ["E2E_REDIS_CLUSTER_RESTART"] = "1"
    return subprocess.run(
        ["poetry", "run", "pytest", *shard_args(args.shard), *pytest_args],
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
