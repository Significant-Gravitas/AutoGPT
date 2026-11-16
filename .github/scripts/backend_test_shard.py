#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


# A shard runs its paths minus any other shard's paths nested inside them, so
# "copilot" runs backend/copilot without the folders "copilot-sdk" takes.
SHARDS = {
    "data": ["backend/data"],
    "copilot": ["backend/copilot"],
    "copilot-sdk": [
        "backend/copilot/sdk",
        "backend/copilot/billing_e2e",
        "backend/copilot/graphiti",
        "backend/copilot/dream",
    ],
    "util-executor": ["backend/util", "backend/executor"],
}


def _is_within(path: str, parent: str) -> bool:
    return path.startswith(f"{parent}/")


def shard_args(shard: str) -> list[str]:
    if shard in SHARDS:
        nested = [
            path
            for other, paths in SHARDS.items()
            if other != shard
            for path in paths
            if any(_is_within(path, own) for own in SHARDS[shard])
        ]
        return [
            "-c",
            "pyproject.toml",
            *SHARDS[shard],
            *(f"--ignore={path}" for path in nested),
        ]
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
