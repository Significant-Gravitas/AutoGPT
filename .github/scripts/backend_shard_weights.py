import fnmatch
import json
import math
import os
from pathlib import Path, PurePosixPath

from backend_test_shard import SHARDS

DEFAULT_FILE_SECONDS = 1.0


def path_shard(path: str) -> str:
    """Keep each file in its existing most-specific fixture group."""
    candidates = [
        (len(parent), shard)
        for shard, paths in SHARDS.items()
        for parent in paths
        if PurePosixPath(path).is_relative_to(parent)
    ]
    return max(candidates)[1] if candidates else "remainder"


def partition_files(
    files: list[str], weights: dict[str, float], count: int
) -> dict[str, int]:
    """Balance whole files, including new files, without sharing process state."""
    if count < 1:
        raise ValueError("A shard needs at least one partition")
    loads = dict.fromkeys(range(1, count + 1), 0.0)
    assignments = {}
    for path in sorted(
        set(files), key=lambda p: (-weights.get(p, DEFAULT_FILE_SECONDS), p)
    ):
        owner = min(loads, key=lambda part: (loads[part], part))
        assignments[path] = owner
        loads[owner] += max(weights.get(path, DEFAULT_FILE_SECONDS), 0.001)
    return assignments


def assign_files(
    files: list[str],
    weights: dict[str, float],
    shards: tuple[str, ...] = (*SHARDS, "remainder"),
) -> dict[str, str]:
    workers = [name for name in shards if name != "data"]
    if not workers or "data" not in shards or len(set(shards)) != len(shards):
        raise ValueError("Shards need data and distinct balancing workers")
    loads = dict.fromkeys(workers, 0.0)
    assignments = {}
    for path in sorted(
        set(files), key=lambda p: (-weights.get(p, DEFAULT_FILE_SECONDS), p)
    ):
        if PurePosixPath(path).is_relative_to("backend/data"):
            assignments[path] = "data"
            continue
        owner = min(workers, key=lambda name: loads[name])
        assignments[path] = owner
        loads[owner] += max(weights.get(path, DEFAULT_FILE_SECONDS), 0.001)
    return assignments


def load_weights(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("version") != 1
        or not isinstance(data.get("seconds"), dict)
    ):
        raise ValueError("Expected version 1 test durations")
    weights = data["seconds"]
    if any(
        not isinstance(name, str)
        or type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        for name, value in weights.items()
    ):
        raise ValueError("Test durations must be finite, nonnegative numbers")
    return weights


def discover_files(root: Path, patterns: list[str], excluded: list[str]) -> list[str]:
    files = set()
    for start in (root, root / "../../.github/scripts"):
        for directory, dirs, names in os.walk(start):
            dirs[:] = [
                name
                for name in dirs
                if not any(fnmatch.fnmatch(name, p) for p in excluded)
            ]
            for name in names:
                if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                    files.add(relative_file(Path(directory) / name, root))
    return sorted(files)


def relative_file(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def owner_for(path: str, assignments: dict[str, str]) -> str:
    if PurePosixPath(path).is_relative_to("backend/data"):
        return "data"
    return assignments.get(path, "remainder")
