#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import PurePosixPath


SHARDS = {
    "data": ["backend/data"],
    "copilot": ["backend/copilot"],
    "copilot-memory": ["backend/copilot/dream", "backend/copilot/graphiti"],
    "copilot-billing": ["backend/copilot/billing_e2e"],
    "billing-threads": [
        "backend/copilot/billing_e2e/thread_billing_test.py",
        "backend/executor/billing_e2e",
    ],
    "util-executor": ["backend/util", "backend/executor"],
    "api": ["backend/api"],
}


def shard_args(shard: str) -> list[str]:
    if shard in SHARDS:
        paths = SHARDS[shard]
        # Nested groups own their subtrees, including individually split files.
        ignored = [
            path
            for other, group in SHARDS.items()
            if other != shard
            for path in group
            if any(PurePosixPath(path).is_relative_to(parent) for parent in paths)
        ]
    elif shard == "remainder":
        paths = [".", "../../.github/scripts"]
        ignored = [path for group in SHARDS.values() for path in group]
    else:
        raise ValueError(f"Unknown backend test shard: {shard}")
    return [
        "-c",
        "pyproject.toml",
        *paths,
        *(f"--ignore={path}" for path in ignored),
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
