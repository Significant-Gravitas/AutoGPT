"""Reads the elapsed wall-clock bounds a test module's asserts check, for the
rule in ``architecture_wall_clock_bounds_test.py``.

An assert bounds elapsed wall-clock time from above when it requires, by
``<``, ``<=``, ``==`` or their mirrors (through ``and``, ``not`` and
``is True``/``is False``):

- a wall clock read minus something (``clock() - started``), written out or
  through a variable, scaled by a constant (``* 1000`` for milliseconds) or
  kept by ``round``, ``abs``, ``int`` or ``float``; or
- a fresh clock read below a deadline set from an earlier one
  (``clock() < deadline`` where ``deadline = clock() + 1``, or
  ``clock() < started + 1``).

A name assigned more than once counts only when its assignments agree.

A wall clock is ``time.monotonic``, ``time.perf_counter`` or ``time.time``
(or their ``_ns`` forms), or an event loop's ``time()``. Bounds are worked out
from literals and constants, including module constants imported from the
backend or autogpt_libs; anything else reads as unknown and is not reported.
"""

import ast
import functools
import io
import operator
import re
import tokenize
from collections.abc import Iterator

from backend.util.architecture_test import BACKEND_ROOT, _import_bindings, _module_tree

_LIBS_ROOT = BACKEND_ROOT.parents[1] / "autogpt_libs" / "autogpt_libs"
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
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
# The value of a name bound by anything but one assignment (a parameter, a
# loop variable): it resolves to nothing.
_UNKNOWN = ast.Constant(value=None)


def wall_clock_bounds(source: str, package: str) -> list[tuple[int, float, bool]]:
    """Each assert in *source* that bounds a wall-clock elapsed time from
    above by a known number of seconds: its line, its tightest bound, and
    whether a ``# wall-clock: <reason>`` comment covers it."""
    tree = ast.parse(source)
    lines = source.splitlines()
    reasons = _reason_lines(source)
    found: list[tuple[int, float, bool]] = []
    for node, scope in _asserts(tree, _Scope(tree, package, None)):
        bounds = [
            bound
            for small, big in _upper_bound_pairs(node.test, negated=False)
            if (bound := _bound(small, big, scope)) is not None
        ]
        if bounds:
            reason = _gives_a_reason(node, lines, reasons)
            found.append((node.lineno, min(bounds), reason))
    return found


def _reason_lines(source: str) -> set[int]:
    """Lines with a ``# wall-clock: <reason>`` comment, a comment token and
    not text inside a string."""
    return {
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and _REASON.search(token.string)
    }


def _gives_a_reason(node: ast.Assert, lines: list[str], reasons: set[int]) -> bool:
    """A reason on one of the assert's lines or in the comments right above."""
    first = node.lineno
    while first > 1 and lines[first - 2].lstrip().startswith("#"):
        first -= 1
    last = node.end_lineno or node.lineno
    return any(line in reasons for line in range(first, last + 1))


class _Scope:
    """The names one function body, or the module, assigns and imports;
    a name it does not bind is looked up in the scope around it."""

    def __init__(self, node: ast.AST, package: str, outer: "_Scope | None") -> None:
        nodes = list(_own_nodes(node))
        self.outer = outer
        self.package = package
        self.imports = _import_bindings(nodes, package)
        self.values: dict[str, list[ast.expr]] = {}
        if isinstance(node, _SCOPES):
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
        """What binds *name*: the values assigned to it, or the dotted name it
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
    """Every node inside *node* outside the functions, classes and
    comprehensions nested in it, whose names the code around them can't see."""
    for child in ast.iter_child_nodes(node):
        yield child
        if not isinstance(child, (*_SCOPES, ast.ClassDef, *_COMPREHENSIONS)):
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
        if isinstance(child, _SCOPES):
            yield from _asserts(child, _Scope(child, scope.package, scope))
            continue
        if isinstance(child, ast.Assert):
            yield child, scope
        yield from _asserts(child, scope)


def _upper_bound_pairs(
    node: ast.expr, negated: bool
) -> Iterator[tuple[ast.expr, ast.expr]]:
    """Each ``small <= big`` the assert requires, whichever way it is written:
    every part of an ``and`` (of an ``or`` under a ``not``), and a comparison
    flipped by ``not`` or ``is False``."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        yield from _upper_bound_pairs(node.operand, not negated)
    elif isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.Or if negated else ast.And):
            for value in node.values:
                yield from _upper_bound_pairs(value, negated)
    elif isinstance(node, ast.Compare):
        yield from _compare_pairs(node, negated)


def _compare_pairs(
    node: ast.Compare, negated: bool
) -> Iterator[tuple[ast.expr, ast.expr]]:
    truth = node.comparators[0]
    if (
        len(node.ops) == 1
        and isinstance(node.ops[0], (ast.Is, ast.IsNot, ast.Eq, ast.NotEq))
        and isinstance(truth, ast.Constant)
        and isinstance(truth.value, bool)
    ):
        # `(x < 1) is False` requires the opposite of `x < 1`.
        flip = (not truth.value) != isinstance(node.ops[0], (ast.IsNot, ast.NotEq))
        yield from _upper_bound_pairs(node.left, negated != flip)
        return
    if negated and len(node.ops) > 1:
        return  # `not (a < b < c)` requires neither link on its own
    operands = [node.left, *node.comparators]
    for left, op, right in zip(operands, node.ops, operands[1:]):
        if isinstance(op, (ast.Lt, ast.LtE)):
            yield (right, left) if negated else (left, right)
        elif isinstance(op, (ast.Gt, ast.GtE)):
            yield (left, right) if negated else (right, left)
        elif isinstance(op, ast.Eq) and not negated:
            yield left, right
            yield right, left


def _bound(small: ast.expr, big: ast.expr, scope: _Scope) -> float | None:
    """The seconds ``small <= big`` caps a wall-clock elapsed time at, if any."""
    scale = _elapsed_scale(small, scope)
    if scale is not None:
        seconds = _seconds(big, scope)
        if (
            seconds is None
            and isinstance(small, ast.BinOp)
            and isinstance(small.op, ast.Sub)
            and _clock_scale(small.left, scope) is not None
        ):
            # `clock() - 1 < started` is `clock() - started < 1`.
            seconds = _seconds(small.right, scope)
        return None if seconds is None else seconds * scale
    scale = _clock_call_scale(small, scope)
    offset = None if scale is None else _deadline_offset(big, scope)
    return None if scale is None or offset is None else offset * scale


def _elapsed_scale(node: ast.expr, scope: _Scope, names: bool = True) -> float | None:
    """Seconds per unit if *node* is a wall clock read minus something,
    written out or (when *names*) through a variable assigned it."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _SPAN_PRESERVING_CALLS
        and node.args
    ):
        return _elapsed_scale(node.args[0], scope, names)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return _clock_scale(node.left, scope)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        return _scaled_elapsed_scale(node, scope, names)
    if names and isinstance(node, ast.Name):
        values, _, owner = scope.lookup(node.id)
        return _agreed(_elapsed_scale(value, owner, names=False) for value in values)
    return None


def _agreed(found: Iterator[float | None]) -> float | None:
    """The one value every assignment of a name gives, else nothing."""
    values = set(found)
    return values.pop() if len(values) == 1 else None


def _scaled_elapsed_scale(node: ast.BinOp, scope: _Scope, names: bool) -> float | None:
    """An elapsed time multiplied or divided by a positive constant, such as
    ``(clock() - started) * 1000`` in milliseconds."""
    pairs = [(node.left, node.right)]
    if isinstance(node.op, ast.Mult):
        pairs.append((node.right, node.left))
    for elapsed, factor in pairs:
        scale = _elapsed_scale(elapsed, scope, names)
        times = _seconds(factor, scope)
        if scale is not None and times is not None and times > 0:
            return scale / times if isinstance(node.op, ast.Mult) else scale * times
    return None


def _clock_scale(node: ast.expr, scope: _Scope) -> float | None:
    """Seconds per unit if *node* reads a wall clock, itself or through a
    name it was assigned to."""
    if isinstance(node, ast.Name):
        values, _, owner = scope.lookup(node.id)
        return _agreed(_clock_call_scale(value, owner) for value in values)
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
        if _is_event_loop(func.value, scope):
            return 1.0
    module, _, name = (_dotted_origin(func, scope) or "").rpartition(".")
    return _WALL_CLOCKS.get(name) if module == "time" else None


def _is_event_loop(node: ast.expr, scope: _Scope) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr in _LOOP_GETTERS
        if not isinstance(func, ast.Name):
            return False
        origin = scope.lookup(func.id)[1] or func.id
        return origin.rpartition(".")[2] in _LOOP_GETTERS
    if isinstance(node, ast.Attribute):
        return node.attr.lower().endswith("loop")
    return isinstance(node, ast.Name) and node.id.lower().endswith("loop")


def _dotted_origin(node: ast.expr, scope: _Scope) -> str | None:
    """``time.monotonic`` for ``time.monotonic`` or an alias imported for it."""
    if isinstance(node, ast.Name):
        return scope.lookup(node.id)[1]
    if isinstance(node, ast.Attribute):
        base = _dotted_origin(node.value, scope)
        return f"{base}.{node.attr}" if base else None
    return None


def _deadline_offset(node: ast.expr, scope: _Scope) -> float | None:
    """The ``bound`` of a deadline set from an earlier clock read, in the
    clock's units: a name assigned ``clock() + bound``, or ``started +
    bound`` for a name holding a read."""
    if isinstance(node, ast.Name):
        values, _, owner = scope.lookup(node.id)
        return _agreed(_earlier_read_plus(value, owner, True) for value in values)
    return _earlier_read_plus(node, scope, False)


def _earlier_read_plus(node: ast.expr, scope: _Scope, assigned: bool) -> float | None:
    """``read + bound``, where the read happened before the assert's own: a
    name holding one, or, in a deadline's assignment, a clock call."""
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return None
    for read, bound in ((node.left, node.right), (node.right, node.left)):
        earlier = isinstance(read, ast.Name) and _clock_scale(read, scope)
        if earlier or (assigned and _clock_call_scale(read, scope) is not None):
            return _seconds(bound, scope)
    return None


def _seconds(node: ast.expr, scope: _Scope, depth: int = 0) -> float | None:
    """*node* as a number, if literals and constants make it one."""
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
    if isinstance(node, ast.Attribute):
        origin = _dotted_origin(node, scope)
        if origin is not None:
            return _imported_seconds(origin, depth + 1)
    return None


def _imported_seconds(dotted: str, depth: int) -> float | None:
    """A module-level constant of a backend or autogpt_libs module."""
    module, _, name = dotted.rpartition(".")
    found = _source_tree(module)
    if found is None:
        return None
    tree, package = found
    scope = _Scope(tree, package, None)
    return _seconds(ast.Name(id=name, ctx=ast.Load()), scope, depth)


@functools.cache
def _source_tree(module: str) -> tuple[ast.Module, str] | None:
    """The parsed module, read from source and never imported, and the
    package its relative imports use."""
    if module != "autogpt_libs" and not module.startswith("autogpt_libs."):
        return _module_tree(module)
    base = _LIBS_ROOT.joinpath(*module.split(".")[1:])
    for path, package in (
        (base / "__init__.py", module),
        (base.with_suffix(".py"), module.rpartition(".")[0]),
    ):
        if path.is_file():
            return ast.parse(path.read_text(encoding="utf-8")), package
    return None
