"""Tests bound elapsed time in CPU time, or in wall time only generously.

Motivation (#16920): a CI runner can stall a test's thread for a second or
more (a full garbage collection, a busy neighbour), so an assert like
``time.monotonic() - started < 1.0`` fails while the code under test is fine.
On 2026-11-18 a 1.3 s stall failed ``copilot/executor/processor_test.py`` on a
dev push and in the merge queue (#16911), and
``integrations/creds_manager_lock_test.py`` had failed a merge group the same
way (#15687).

The rule reads every test module of the backend and autogpt_libs and fails an
``assert`` that bounds a wall-clock elapsed time from above by less than
``MIN_WALL_CLOCK_BOUND_SECONDS``; ``architecture_wall_clock_scan_test.py``
says which forms it reads. CPU clocks (``time.process_time``,
``time.thread_time``) pass: they stand still while the runner has the thread
descheduled.
"""

import ast
import functools
import pathlib

import pytest

import backend.util.architecture_wall_clock_scan_test as scan
from backend.util.architecture_test import BACKEND_ROOT

MIN_WALL_CLOCK_BOUND_SECONDS = 5.0

_PLATFORM_ROOT = BACKEND_ROOT.parents[1]
# Each tree of test modules, with the package its top directory is.
_SCAN_ROOTS = (
    (BACKEND_ROOT, "backend"),
    (BACKEND_ROOT.parent / "test", "test"),
    (_PLATFORM_ROOT / "autogpt_libs" / "autogpt_libs", "autogpt_libs"),
)


def test_tests_bound_wall_clock_time_only_generously():
    tight = [
        f"  {rel}:{line} bounds wall-clock time by {bound:g} s"
        for rel, line, bound, reason in sorted(_wall_clock_bounds())
        if _too_tight(bound, reason)
    ]
    assert not tight, (
        "These asserts bound elapsed wall-clock time below "
        f"{MIN_WALL_CLOCK_BOUND_SECONDS:g} s. A CI runner can stall a test's "
        "thread for seconds, so they fail while the code is fine (#16920).\n"
        "- Work that must stay cheap: time it with time.thread_time() (or "
        "time.process_time() when it runs on other threads), or count the "
        "work.\n"
        "- A wait that must end at its timeout: keep wall time with a bound "
        "several times the timeout, as a named constant like the file's "
        "_HANG_SECONDS = 10, or assert the order of events instead.\n"
        "- If wall time under that really is the property, say why in a "
        "comment on the assert or right above it: `# wall-clock: <reason>`.\n"
        + "\n".join(tight)
    )


def test_the_scan_reads_test_modules_and_only_those(tmp_path: pathlib.Path):
    # A scan that finds nothing proves nothing; this one must find these two.
    source = "import time\nstarted = time.monotonic()\n"
    source += "assert time.monotonic() - started < 1\n"
    package = tmp_path / "pkg"
    package.mkdir()
    for name in ("checks_test.py", "test_checks.py", "checks.py"):
        (package / name).write_text(source)
    found = _scan(((package, "pkg"),), tmp_path)
    assert [(rel, bound) for rel, _, bound, _ in found] == [
        ("pkg/checks_test.py", 1.0),
        ("pkg/test_checks.py", 1.0),
    ]


@pytest.mark.parametrize(
    "source, expected",
    [
        ("assert time.monotonic() - started < 1.0", [1.0]),
        ("assert time.perf_counter() - started <= 0.5", [0.5]),
        ("assert time.time() - started < 2", [2.0]),
        ("assert time.monotonic_ns() - started < 500_000_000", [0.5]),
        ("assert 1 > time.monotonic() - started", [1.0]),
        ("assert 0.05 <= time.monotonic() - started <= 0.35", [0.35]),
        ("assert ok and time.monotonic() - started < 1", [1.0]),
        ("assert round(time.monotonic() - started, 2) < 1", [1.0]),
        ("assert round(time.monotonic() - started, 1) == 0.1", [0.1]),
        ("assert 0.1 == time.monotonic() - started", [0.1]),
        ("assert (time.monotonic() - started) * 1000 < 500", [0.5]),
        ("assert 1000 * (time.monotonic() - started) < 500", [0.5]),
        ("assert (time.monotonic_ns() - started) / 1e6 < 500", [0.5]),
        ("assert (time.monotonic() - started) * 0 < 1", []),
        ("assert (time.monotonic() - started) * budget() < 1", []),
        ("ms = (time.perf_counter() - started) * 1000\nassert ms < 500", [0.5]),
        ("assert not (time.monotonic() - started < 1)", []),
        ("assert not (time.monotonic() - started > 1)", [1.0]),
        ("assert not time.monotonic() - started == 1", []),
        ("assert (time.monotonic() - started < 1) is True", [1.0]),
        ("assert (time.monotonic() - started < 1) is False", []),
        ("assert not (0 <= time.monotonic() - started <= 1)", []),
        ("assert ok or time.monotonic() - started < 1", []),
        ("assert not (ok or time.monotonic() - started > 1)", [1.0]),
        ("observed = time.monotonic()\nassert observed < time.monotonic() + 1", []),
        ("assert time.monotonic() < time.monotonic() + 1", []),
        (
            "elapsed = time.monotonic() - started\n"
            "elapsed = time.thread_time() - started\n"
            "assert elapsed < 1",
            [],
        ),
        (
            "finished = time.monotonic_ns()\nfinished = time.monotonic()\n"
            "assert finished - started < 30",
            [],
        ),
        (
            "deadline = time.monotonic() + 1\ndeadline = time.monotonic() + 30\n"
            "assert time.monotonic() < deadline",
            [],
        ),
        (
            "LIMIT = 1\npresets = [LIMIT for LIMIT in (10, 20)]\n"
            "assert time.monotonic() - started < LIMIT",
            [1.0],
        ),
        (
            "from asyncio import get_running_loop as current\n"
            "assert current().time() - started < 1",
            [1.0],
        ),
        ("elapsed = time.perf_counter() - started\nassert elapsed < 1", [1.0]),
        ("now = time.monotonic()\nassert now - started < 1", [1.0]),
        ("deadline = time.monotonic() + 1\nassert time.monotonic() < deadline", [1.0]),
        ("assert reset > time.time() - 1", [1.0]),
        ("assert int(time.time()) - started < 2", [2.0]),
        ("started = time.monotonic()\nassert time.monotonic() < started + 1", [1.0]),
        ("assert time.monotonic() < deadline + 1", []),
        ("assert loop.time() - started < 1", [1.0]),
        ("assert self._loop.time() - started < 1", [1.0]),
        ("assert asyncio.get_running_loop().time() - started < 1", [1.0]),
        ("from time import perf_counter as pc\nassert pc() - started < 1", [1.0]),
        ("PROMPT = 0.5\nassert time.monotonic() - started < PROMPT", [0.5]),
        ("assert time.monotonic() - started < 0.4 + 0.6", [1.0]),
        ("assert time.monotonic() - started < 1.0  # wall-clock: why", []),
        ("# wall-clock: why\n# more\nassert time.monotonic() - started < 1", []),
        ("assert time.monotonic() - started < 1.0  # wall-clock:", [1.0]),
        ("assert time.monotonic() - started < 1  # TODO: a wall-clock: why", [1.0]),
        ("assert time.monotonic() - started < 1  # noqa  # wall-clock: why", []),
        ('assert time.monotonic() - started < 1, "# wall-clock: no"', [1.0]),
        ("assert time.monotonic() - started < 5", []),
        ("assert time.monotonic() - started > 0.09", []),
        ("assert time.thread_time() - started < 1", []),
        ("assert time.process_time() - started < 1", []),
        ("assert time.monotonic() - started < budget()", []),
        ("assert time.monotonic() < deadline", []),
        ("assert time.monotonic() - started < 1 / 0", []),
        ("assert dt.time() - started < 1", []),
        ("assert time.monotonic() - started != 1", []),
        ("assert time.monotonic() - started < -(-1)", [1.0]),
        ("assert time.monotonic() - started < +budget()", []),
        ("assert time.monotonic() - started < budget() + 1", []),
        ("assert time.monotonic() - started < self.LIMIT", []),
        ("from math import pi\nassert time.monotonic() - started < pi", []),
        ("A = B\nB = A\nassert time.monotonic() - started < A", []),
        ("assert finished - started < 1", []),
        ("assert compute(1) - started < 1", []),
        ("elapsed = finished - started\nassert elapsed < 1", []),
        ("elapsed = time.monotonic() - started\nassert elapsed < budget()", []),
        ("deadline = 1 + time.monotonic()\nassert time.monotonic() < deadline", [1.0]),
        ("deadline = start + 1\nassert time.monotonic() < deadline", []),
        ("deadline = compute() + 1\nassert time.monotonic() < deadline", []),
        ("assert time.monotonic() < deadline * 2", []),
        (
            "from asyncio import get_running_loop\n"
            "assert get_running_loop().time() - started < 1",
            [1.0],
        ),
        ("PROMPT = 1\nPROMPT = 2\nassert time.monotonic() - started < PROMPT", []),
        (
            "LIMIT = 0.5\nclass Other:\n    LIMIT = 10\n"
            "assert time.monotonic() - started < LIMIT",
            [0.5],
        ),
        ("def test(budget=0.5):\n    assert time.monotonic() - started < budget", []),
        (
            "def test():\n"
            "    budget = 0.5\n"
            "    async def inner():\n"
            "        assert time.monotonic() - started < budget\n",
            [0.5],
        ),
        (
            "def test():\n"
            "    for budget in (0.5, 0.6):\n"
            "        assert time.monotonic() - started < budget\n",
            [],
        ),
        (
            "from backend.fake.limits import BOUND\n"
            "assert time.monotonic() - started < BOUND",
            [1.0],
        ),
        (
            "from backend.fake import limits\n"
            "assert time.monotonic() - started < 2 * limits.BOUND",
            [2.0],
        ),
        (
            "from backend.fake import limits\n"
            "assert time.monotonic() - started < limits.ELSEWHERE",
            [3.0],
        ),
        (
            "from backend.fake import limits\n"
            "assert time.monotonic() - started < limits.MISSING",
            [],
        ),
        (
            "import backend.fake.limits\n"
            "assert time.monotonic() - started < backend.fake.limits.BOUND",
            [1.0],
        ),
    ],
)
def test_the_scan_reads_each_form(
    source: str, expected: list[float], monkeypatch: pytest.MonkeyPatch
):
    # Synthetic modules, so a change to real code cannot move this test.
    fake = {
        "backend.fake.limits": (
            "from backend.fake.other import OTHER as ELSEWHERE\nBOUND = 1.0\n"
        ),
        "backend.fake.other": "OTHER = 3.0\n",
    }
    monkeypatch.setattr(
        scan,
        "_source_tree",
        lambda module: (
            (ast.parse(fake[module]), "backend.fake") if module in fake else None
        ),
    )
    source = "import asyncio\nimport time\n" + source
    bounds = scan.wall_clock_bounds(source, "backend.util")
    tight = [bound for _, bound, reason in bounds if _too_tight(bound, reason)]
    assert tight == pytest.approx(expected)


@pytest.mark.parametrize(
    "module, found",
    [
        ("backend.util.architecture_test", True),
        ("autogpt_libs.auth", True),
        ("autogpt_libs.auth.config", True),
        ("autogpt_libs.no_such_module", False),
        ("math", False),
    ],
)
def test_constants_are_read_from_both_packages_sources(module: str, found: bool):
    assert (scan._source_tree(module) is not None) is found


def _too_tight(bound: float, reason: bool) -> bool:
    return bound < MIN_WALL_CLOCK_BOUND_SECONDS and not reason


@functools.cache
def _wall_clock_bounds() -> list[tuple[str, int, float, bool]]:
    return _scan(_SCAN_ROOTS, _PLATFORM_ROOT)


def _scan(
    roots: tuple[tuple[pathlib.Path, str], ...], base: pathlib.Path
) -> list[tuple[str, int, float, bool]]:
    """Each assert of the test modules under *roots* that bounds a
    wall-clock elapsed time from above by a known number of seconds, and
    whether it says why."""
    found: list[tuple[str, int, float, bool]] = []
    for root, package in roots:
        assert root.is_dir(), f"{root} moved; point _SCAN_ROOTS at it"
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or not _is_test_module(path):
                continue
            source = path.read_text(encoding="utf-8")
            if "time" not in source:
                continue
            module_package = ".".join([package, *path.relative_to(root).parts[:-1]])
            rel = path.relative_to(base).as_posix()
            found += [
                (rel, line, bound, reason)
                for line, bound, reason in scan.wall_clock_bounds(
                    source, module_package
                )
            ]
    return found


def _is_test_module(path: pathlib.Path) -> bool:
    return path.name.endswith("_test.py") or path.name.startswith("test_")
