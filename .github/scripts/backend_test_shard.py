#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

SHARDS = {
    "data": ["backend/data"],
    "copilot": ["backend/copilot"],
    "memory-experts": [
        "backend/copilot/dream",
        "backend/copilot/graphiti",
        "backend/api/features/experts",
    ],
    "sdk-billing": ["backend/copilot/sdk", "backend/copilot/billing_e2e"],
    "billing-threads": [
        "backend/copilot/billing_e2e/thread_billing_test.py",
        "backend/executor/billing_e2e",
    ],
    "util-executor": ["backend/util", "backend/executor"],
    "api": ["backend/api"],
}
PARTITIONS = {shard: 4 for shard in (*SHARDS, "remainder")}
PARTITIONS["billing-threads"] = 3


def report_shards(partitioned: bool = False) -> list[str]:
    """Enumerate every required report identity independently of uploaded files."""
    return [
        f"{shard}-{part}" if partitioned else shard
        for shard, count in PARTITIONS.items()
        for part in range(1, (count if partitioned else 1) + 1)
    ]


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
    parser.add_argument("--layout", choices=["paths", "weighted"], default="paths")
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("shard", choices=[*SHARDS, "remainder"])
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 <= args.partition <= PARTITIONS[args.shard]:
        parser.error("partition is outside the configured shard range")
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    if args.shard == "data":
        os.environ["E2E_REDIS_CLUSTER_RESTART"] = "1"
    os.environ["BACKEND_TEST_SHARD"] = args.shard
    os.environ["BACKEND_TEST_LAYOUT"] = args.layout
    os.environ["BACKEND_TEST_PARTITION"] = str(args.partition)
    scripts = str(Path(__file__).resolve().parent)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        part for part in (scripts, os.environ.get("PYTHONPATH", "")) if part
    )
    selected = (
        shard_args(args.shard)
        if args.layout == "paths"
        else ["-c", "pyproject.toml", ".", "../../.github/scripts"]
    )
    os.environ["BACKEND_TEST_LAUNCH_TIME"] = str(time.perf_counter())
    return subprocess.run(
        [
            "poetry",
            "run",
            *( ["python", str(Path(__file__).with_name("preload_prisma_pytest.py"))]
                if os.environ.get("BENCHMARK_PRELOAD_PRISMA") == "1" else ["pytest"] ),
            "-p",
            "backend_shard_plugin",
            *selected,
            *pytest_args,
        ],
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
