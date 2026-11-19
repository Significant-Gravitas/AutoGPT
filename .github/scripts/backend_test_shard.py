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
    # In the order they ran inside "copilot": a graphiti test fails on what
    # the sdk tests leave for the garbage collector.
    "copilot-sdk": [
        "backend/copilot/billing_e2e",
        "backend/copilot/dream",
        "backend/copilot/graphiti",
        "backend/copilot/sdk",
    ],
    "util-executor": ["backend/util", "backend/executor"],
}

# The docker-backed e2e tests a shard turns on; elsewhere they skip unless asked.
SHARD_E2E_FLAGS = {
    "data": "E2E_REDIS_CLUSTER_RESTART",
    "util-executor": "E2E_CLAMD_CONTAINER",
}

# No test in "copilot-sdk" starts the database manager. Its conftests answer
# database manager calls with a fake (backend/copilot/database_manager_fixtures.py);
# this is the backstop for a call that slips past it, so it fails after a few
# tries instead of retrying until the job's time limit.
SHARDS_WITHOUT_DATABASE_MANAGER = {"copilot-sdk"}


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
    if flag := SHARD_E2E_FLAGS.get(args.shard):
        os.environ[flag] = "1"
    if args.shard in SHARDS_WITHOUT_DATABASE_MANAGER:
        os.environ["PYRO_CLIENT_COMM_RETRY"] = "3"
    return subprocess.run(
        ["poetry", "run", "pytest", *shard_args(args.shard), *pytest_args],
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
