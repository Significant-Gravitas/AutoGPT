import json
import os
import platform
import subprocess
import time
from pathlib import Path

import pytest
from backend_shard_weights import (
    assign_files,
    discover_files,
    load_weights,
    owner_for,
    partition_files,
    path_shard,
    relative_file,
)
from backend_test_shard import PARTITIONS, SHARDS


def pytest_configure(config: pytest.Config) -> None:
    shard = os.environ["BACKEND_TEST_SHARD"]
    layout = os.environ["BACKEND_TEST_LAYOUT"]
    if shard not in (*SHARDS, "remainder") or layout not in ("paths", "weighted"):
        raise pytest.UsageError("Invalid backend shard or layout")
    config.pluginmanager.register(
        ShardMetrics(config, shard, layout), "backend-shard-metrics"
    )


class ShardMetrics:
    def __init__(self, config: pytest.Config, shard: str, layout: str):
        self.root = config.rootpath
        self.shard = shard
        self.layout = layout
        self.started = time.perf_counter()
        self.nodeids: list[str] = []
        self.item_files: dict[str, str] = {}
        self.files: dict[str, dict[str, float]] = {}
        self.assignments: dict[str, str] = {}
        self.partition = int(os.environ.get("BACKEND_TEST_PARTITION", "0"))
        if not 0 <= self.partition <= PARTITIONS[shard]:
            raise pytest.UsageError("Invalid backend partition")
        self.partitions: dict[str, int] = {}
        if layout == "weighted" or self.partition:
            weights = load_weights(
                Path(
                    os.environ.get(
                        "BACKEND_TEST_WEIGHTS",
                        Path(__file__).with_name("backend-test-durations.json"),
                    )
                )
            )
            files = discover_files(
                self.root, config.getini("python_files"), config.getini("norecursedirs")
            )
            if layout == "weighted":
                self.assignments = assign_files(files, weights)
            if self.partition:
                selected = [
                    path
                    for path in files
                    if (
                        owner_for(path, self.assignments)
                        if layout == "weighted"
                        else path_shard(path)
                    )
                    == shard
                ]
                self.partitions = partition_files(selected, weights, PARTITIONS[shard])

    def pytest_ignore_collect(self, collection_path: Path) -> bool | None:
        if collection_path.is_dir():
            return None
        path = relative_file(collection_path, self.root)
        if self.partition and path in self.partitions:
            return True if self.partitions[path] != self.partition else None
        if self.layout == "weighted":
            return True if owner_for(path, self.assignments) != self.shard else None
        return None

    @pytest.hookimpl(wrapper=True)
    def pytest_make_collect_report(self, collector: pytest.Collector):
        started = time.perf_counter()
        report = yield
        if isinstance(collector, pytest.Module):
            path = relative_file(collector.path, self.root)
            self.file_metrics(path)["collection_seconds"] += (
                time.perf_counter() - started
            )
        return report

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        if self.layout == "weighted":
            selected, deselected = [], []
            for item in items:
                path = relative_file(item.path, self.root)
                target = (
                    selected
                    if owner_for(path, self.assignments) == self.shard
                    else deselected
                )
                target.append(item)
            items[:] = selected
            config.hook.pytest_deselected(items=deselected)
        if self.partition:
            unexpected = [
                item.nodeid
                for item in items
                if relative_file(item.path, self.root) not in self.partitions
            ]
            if unexpected:
                raise pytest.UsageError(
                    f"Tests escaped partition selection: {unexpected}"
                )
            selected, deselected = [], []
            for item in items:
                target = (
                    selected
                    if self.partitions[relative_file(item.path, self.root)]
                    == self.partition
                    else deselected
                )
                target.append(item)
            items[:] = selected
            config.hook.pytest_deselected(items=deselected)
        self.nodeids = [item.nodeid for item in items]
        for item in items:
            path = relative_file(item.path, self.root)
            self.item_files[item.nodeid] = path
            self.file_metrics(path)["cases"] += 1

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        path = self.item_files.get(report.nodeid)
        if path is not None:
            self.file_metrics(path)["test_seconds"] += report.duration

    def pytest_sessionfinish(self, exitstatus: int) -> None:
        report = {
            "version": 1,
            "layout": self.layout,
            "shard": self.shard,
            "partition": self.partition,
            "exit_code": int(exitstatus),
            "pytest_seconds": time.perf_counter() - self.started,
            "source_sha": source_sha(self.root),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model(),
            "runner_image": os.environ.get("ImageVersion"),
            "runner_os": os.environ.get("ImageOS"),
            "nodeids": self.nodeids,
            "files": self.files,
        }
        path = Path(
            os.environ.get("BACKEND_TEST_METRICS", f"shard-metrics-{self.shard}.json")
        )
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def file_metrics(self, path: str) -> dict[str, float]:
        return self.files.setdefault(
            path, {"collection_seconds": 0.0, "test_seconds": 0.0, "cases": 0}
        )


def source_sha(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.is_file():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor()
