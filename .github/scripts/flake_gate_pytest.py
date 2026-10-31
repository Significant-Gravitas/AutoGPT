"""pytest plugin for the flake gate (.github/scripts/flake_gate.py).

Load it with ``-p flake_gate_pytest`` (this directory on PYTHONPATH):

``--flake-gate-seed=N``
    Run the tests in an order shuffled by N: the files, then the classes in
    each file, then the tests in each class. Classes and files stay together,
    so their fixtures are set up once, as in a normal run. Before each test,
    ``random`` is seeded from N and the test's id, so a run with the same seed
    repeats both the order and the random values.
``--flake-gate-report=PATH``
    Append each test's outcome to PATH as JSON lines, flushed as they happen:
    a run killed part-way still says which test it was in.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("flake-gate")
    group.addoption(
        "--flake-gate-seed",
        type=int,
        default=None,
        help="shuffle the test order with this seed",
    )
    group.addoption(
        "--flake-gate-report",
        default=None,
        help="append each test's outcome to this file as JSON lines",
    )


def pytest_configure(config: pytest.Config) -> None:
    path = config.getoption("flake_gate_report")
    if path:
        config.pluginmanager.register(_Recorder(path), "flake_gate_recorder")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    seed = config.getoption("flake_gate_seed")
    if seed is not None:
        items[:] = shuffled(items, seed)


def pytest_runtest_setup(item: pytest.Item) -> None:
    seed = item.config.getoption("flake_gate_seed")
    if seed is not None:
        random.seed(f"{seed}:{item.nodeid}")


def shuffled(items: list[Any], seed: int) -> list[Any]:
    """``items`` in an order shuffled by ``seed``, keeping each file's and
    each class's tests together. Works on anything with a ``nodeid``."""
    files: dict[str, dict[str, list[Any]]] = {}
    for item in items:
        # A parametrize id can itself contain "::".
        parts = item.nodeid.split("[", 1)[0].split("::")
        classes = files.setdefault(parts[0], {})
        classes.setdefault("::".join(parts[:-1]), []).append(item)

    rng = random.Random(seed)
    order: list[Any] = []
    file_groups = list(files.values())
    rng.shuffle(file_groups)
    for classes in file_groups:
        groups = list(classes.values())
        rng.shuffle(groups)
        for tests in groups:
            rng.shuffle(tests)
            order.extend(tests)
    return order


class _Recorder:
    def __init__(self, path: str):
        self._out = open(path, "a", encoding="utf-8")

    def _write(self, **record: Any) -> None:
        self._out.write(json.dumps(record) + "\n")
        self._out.flush()

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self._write(
                nodeid=report.nodeid,
                phase="collect",
                outcome="failed",
                message=report.longreprtext,
            )

    def pytest_runtest_logstart(
        self, nodeid: str, location: tuple[str, int | None, str]
    ) -> None:
        line = location[1]
        self._write(
            nodeid=nodeid, phase="start", line=None if line is None else line + 1
        )

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call" and report.passed:
            return
        self._write(
            nodeid=report.nodeid,
            phase=report.when,
            outcome=report.outcome,
            duration=report.duration,
            message=report.longreprtext if report.failed else "",
        )

    def pytest_unconfigure(self) -> None:
        self._out.close()
