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
``MIN_WALL_CLOCK_BOUND_SECONDS``. It reads the bound written out
(``clock() - started < 1.0``), through a variable (``elapsed = clock() -
started``) and as a deadline (``deadline = clock() + 1.0``, then ``clock() <
deadline``). A wall clock is ``time.monotonic``, ``time.perf_counter`` or
``time.time`` (or their ``_ns`` forms), or an event loop's ``time()``. CPU
clocks (``time.process_time``, ``time.thread_time``) pass: they stand still
while the runner has the thread descheduled. A bound the scan cannot work out
from literals and constants, such as a value measured at run time, is not
checked.
"""

import ast
import functools
import operator
import pathlib
import re
import sys
from collections.abc import Iterator

import pytest

from backend.util.architecture_test import BACKEND_ROOT, _import_bindings, _module_tree

MIN_WALL_CLOCK_BOUND_SECONDS = 5.0

_PLATFORM_ROOT = BACKEND_ROOT.parents[1]
# Each tree of test modules, with the package its top directory is.
_SCAN_ROOTS = (
    (BACKEND_ROOT, "backend"),
    (BACKEND_ROOT.parent / "test", "test"),
    (_PLATFORM_ROOT / "autogpt_libs" / "autogpt_libs", "autogpt_libs"),
)
# Seconds per unit of each wall clock in the `time` module.
_WALL_CLOCKS = {
    "monotonic": 1.0,
    "perf_counter": 1.0,
    "time": 1.0,
    "monotonic_ns": 1e-9,
    "perf_counter_ns": 1e-9,
    "time_ns": 1e-9,
}
_LOOP_GETTERS = frozenset({"get_running_loop", "get_event_loop"})
# Calls that keep the value they are given as a time span.
_SPAN_PRESERVING_CALLS = frozenset({"abs", "float", "int", "round"})
_ARITHMETIC = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_REASON = re.compile(r"#\s*wall-clock:\s*\S")
# A bound that is a runtime value, or a name bound by anything but one
# assignment (a parameter, a loop variable), resolves to nothing.
_UNKNOWN = ast.Constant(value=None)


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
        "- If wall time under that really is the property, say why on the "
        "assert or the comment line above it: `# wall-clock: <reason>`.\n"
        + "\n".join(tight)
    )


def test_the_scan_sees_the_suites_wall_clock_bounds():
    # The bounds the suite keeps, generous and annotated ones included; a
    # scan that finds none proves nothing.
    assert len(_wall_clock_bounds()) >= 20


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
        ("assert time.monotonic() - started < 5", []),
        ("assert time.monotonic() - started > 0.09", []),
        ("assert time.thread_time() - started < 1", []),
        ("assert time.process_time() - started < 1", []),
        ("assert time.monotonic() - started < budget()", []),
        ("assert time.monotonic() < deadline", []),
        ("assert time.monotonic() - started < 1 / 0", []),
        ("assert dt.time() - started < 1", []),
        ("assert time.monotonic() - started == 1", []),
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
        sys.modules[__name__],
        "_module_tree",
        lambda module: (
            (ast.parse(fake[module]), "backend.fake") if module in fake else None
        ),
    )
    source = "import asyncio\nimport time\n" + source
    bounds = _bounds_in(ast.parse(source), source.splitlines(), "backend.util")
    tight = [bound for _, bound, reason in bounds if _too_tight(bound, reason)]
    assert tight == pytest.approx(expected)


def _too_tight(bound: float, reason: bool) -> bool:
    return bound < MIN_WALL_CLOCK_BOUND_SECONDS and not reason


@functools.cache
def _wall_clock_bounds() -> list[tuple[str, int, float, bool]]:
    """Each assert of the test modules that bounds a wall-clock elapsed time
    from above by a known number of seconds, and whether it says why."""
    found: list[tuple[str, int, float, bool]] = []
    for root, package in _SCAN_ROOTS:
        assert root.is_dir(), f"{root} moved; point _SCAN_ROOTS at it"
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts or not _is_test_module(path):
                continue
            source = path.read_text(encoding="utf-8")
            if "time" not in source:
                continue
            rel = path.relative_to(root)
            module_package = ".".join([package, *rel.parts[:-1]])
            tree = ast.parse(source, filename=str(path))
            found += [
                (path.relative_to(_PLATFORM_ROOT).as_posix(), line, bound, reason)
                for line, bound, reason in _bounds_in(
                    tree, source.splitlines(), module_package
                )
            ]
    return found


def _is_test_module(path: pathlib.Path) -> bool:
    return path.name.endswith("_test.py") or path.name.startswith("test_")


def _bounds_in(
    tree: ast.Module, lines: list[str], package: str
) -> list[tuple[int, float, bool]]:
    found: list[tuple[int, float, bool]] = []
    for node, scope in _asserts(tree, _Scope(tree, package, None)):
        bounds = [
            bound
            for small, big in _upper_bound_pairs(node.test)
            if (bound := _bound(small, big, scope)) is not None
        ]
        if bounds:
            found.append((node.lineno, min(bounds), _gives_a_reason(node, lines)))
    return found


class _Scope:
    """The names one function body, or the module, assigns and imports;
    a name it does not bind is looked up in the scope around it."""

    def __init__(self, node: ast.AST, package: str, outer: "_Scope | None") -> None:
        nodes = list(_own_nodes(node))
        self.outer = outer
        self.package = package
        self.imports = _import_bindings(nodes, package)
        self.values: dict[str, list[ast.expr]] = {}
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for argument in ast.walk(node.args):
                if isinstance(argument, ast.arg):
                    self.values.setdefault(argument.arg, []).append(_UNKNOWN)
        assigned: set[int] = set()
        for child in nodes:
            for target, value in _assignments(child):
                assigned.add(id(target))
                self.values.setdefault(target.id, []).append(value)
        for child in nodes:
            if (
                isinstance(child, ast.Name)
                and isinstance(child.ctx, ast.Store)
                and id(child) not in assigned
            ):
                self.values.setdefault(child.id, []).append(_UNKNOWN)

    def lookup(self, name: str) -> tuple[list[ast.expr], str | None, "_Scope"]:
        """What binds `name`: the values assigned to it, or the dotted name it
        was imported from, and the scope that binds it."""
        scope: _Scope | None = self
        while scope is not None:
            if name in scope.values:
                return scope.values[name], None, scope
            if name in scope.imports:
                return [], scope.imports[name], scope
            scope = scope.outer
        return [], None, self


def _own_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Every node inside `node` that is not inside a function nested in it."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield from _own_nodes(child)


def _assignments(node: ast.AST) -> list[tuple[ast.Name, ast.expr]]:
    if isinstance(node, ast.Assign):
        return [(t, node.value) for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
        if isinstance(node.target, ast.Name) and node.value is not None:
            # `x += 1` gives x a second value, so it resolves to nothing.
            return [(node.target, node.value)]
    return []


def _asserts(node: ast.AST, scope: _Scope) -> Iterator[tuple[ast.Assert, _Scope]]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield from _asserts(child, _Scope(child, scope.package, scope))
            continue
        if isinstance(child, ast.Assert):
            yield child, scope
        yield from _asserts(child, scope)


def _upper_bound_pairs(test: ast.expr) -> Iterator[tuple[ast.expr, ast.expr]]:
    """Each `small < big` the assert checks, whichever way it is written."""
    for node in ast.walk(test):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for left, op, right in zip(operands, node.ops, operands[1:]):
            if isinstance(op, (ast.Lt, ast.LtE)):
                yield left, right
            elif isinstance(op, (ast.Gt, ast.GtE)):
                yield right, left


def _bound(small: ast.expr, big: ast.expr, scope: _Scope) -> float | None:
    """The seconds `small < big` caps a wall-clock elapsed time at, if any."""
    scale = _elapsed_scale(small, scope)
    if scale is not None:
        seconds = _seconds(big, scope)
        if (
            seconds is None
            and isinstance(small, ast.BinOp)
            and _clock_scale(small.left, scope) is not None
        ):
            # `clock() - 1 < started` is `clock() - started < 1`.
            seconds = _seconds(small.right, scope)
        return None if seconds is None else seconds * scale
    scale = _clock_scale(small, scope)
    offset = None if scale is None else _deadline_offset(big, scope)
    return None if scale is None or offset is None else offset * scale


def _elapsed_scale(node: ast.expr, scope: _Scope) -> float | None:
    """Seconds per unit if `node` is a wall clock read minus something."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _SPAN_PRESERVING_CALLS
        and node.args
    ):
        return _elapsed_scale(node.args[0], scope)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return _clock_scale(node.left, scope)
    if isinstance(node, ast.Name):
        values, _, owner = scope.lookup(node.id)
        for value in values:
            if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Sub):
                scale = _clock_scale(value.left, owner)
                if scale is not None:
                    return scale
    return None


def _clock_scale(node: ast.expr, scope: _Scope) -> float | None:
    """Seconds per unit if `node` reads a wall clock, itself or through a
    name it was assigned to."""
    if isinstance(node, ast.Name):
        values, _, owner = scope.lookup(node.id)
        reads = [_clock_call_scale(value, owner) for value in values]
        return reads[0] if reads and None not in reads else None
    return _clock_call_scale(node, scope)


def _clock_call_scale(node: ast.expr, scope: _Scope) -> float | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id in _SPAN_PRESERVING_CALLS and node.args:
        return _clock_call_scale(node.args[0], scope)
    if node.args or node.keywords:
        return None
    if isinstance(func, ast.Attribute) and func.attr == "time":
        if _is_event_loop(func.value):
            return 1.0
    module, _, name = (_dotted_origin(func, scope) or "").rpartition(".")
    return _WALL_CLOCKS.get(name) if module == "time" else None


def _is_event_loop(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr in _LOOP_GETTERS
        return isinstance(func, ast.Name) and func.id in _LOOP_GETTERS
    if isinstance(node, ast.Attribute):
        return node.attr.lower().endswith("loop")
    return isinstance(node, ast.Name) and node.id.lower().endswith("loop")


def _dotted_origin(node: ast.expr, scope: _Scope) -> str | None:
    """`time.monotonic` for `time.monotonic` or an alias imported for it."""
    if isinstance(node, ast.Name):
        return scope.lookup(node.id)[1]
    if isinstance(node, ast.Attribute):
        base = _dotted_origin(node.value, scope)
        return f"{base}.{node.attr}" if base else None
    return None


def _deadline_offset(node: ast.expr, scope: _Scope) -> float | None:
    """The `bound` of `clock() + bound`, or of a name assigned it, in the
    clock's units."""
    if isinstance(node, ast.Name):
        values, _, owner = scope.lookup(node.id)
        offsets = [_clock_plus(value, owner) for value in values]
        known = [offset for offset in offsets if offset is not None]
        return min(known) if known and len(known) == len(offsets) else None
    return _clock_plus(node, scope)


def _clock_plus(node: ast.expr, scope: _Scope) -> float | None:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return None
    if _clock_scale(node.left, scope) is not None:
        return _seconds(node.right, scope)
    if _clock_scale(node.right, scope) is not None:
        return _seconds(node.left, scope)
    return None


def _seconds(node: ast.expr, scope: _Scope, depth: int = 0) -> float | None:
    """`node` as a number, if literals and constants make it one."""
    if depth > 10:
        return None
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _seconds(node.operand, scope, depth + 1)
        if operand is None:
            return None
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        left = _seconds(node.left, scope, depth + 1)
        right = _seconds(node.right, scope, depth + 1)
        if left is None or right is None:
            return None
        try:
            return _ARITHMETIC[type(node.op)](left, right)
        except ZeroDivisionError:
            return None
    if isinstance(node, ast.Name):
        values, origin, owner = scope.lookup(node.id)
        if origin is not None:
            return _imported_seconds(origin, depth + 1)
        return _seconds(values[0], owner, depth + 1) if len(values) == 1 else None
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        origin = scope.lookup(node.value.id)[1]
        if origin is not None:
            return _imported_seconds(f"{origin}.{node.attr}", depth + 1)
    return None


def _imported_seconds(dotted: str, depth: int) -> float | None:
    """A module-level constant of a backend module, by its dotted name."""
    module, _, name = dotted.rpartition(".")
    found = _module_tree(module)
    if found is None:
        return None
    tree, package = found
    return _seconds(
        ast.Name(id=name, ctx=ast.Load()), _Scope(tree, package, None), depth
    )


def _gives_a_reason(node: ast.Assert, lines: list[str]) -> bool:
    """A `# wall-clock: <reason>` comment on the assert or right above it."""
    first = node.lineno - 1
    while first > 0 and lines[first - 1].lstrip().startswith("#"):
        first -= 1
    return any(_REASON.search(line) for line in lines[first : node.end_lineno])
