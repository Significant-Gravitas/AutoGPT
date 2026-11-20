"""Reads the caps a test module puts on its own waits, for the rule in
``architecture_hang_cap_test.py``.

A cap is ``asyncio.wait_for(aw, N)``, ``asyncio.timeout(N)`` or
``asyncio.timeout_at(<clock read> + N)``, aliases included. When one runs out
it raises ``TimeoutError`` at the test's own await, so it caps work the test
expects to finish unless the test handles that error there:

- inside ``pytest.raises`` of it, with no ``match`` (the cap's own error has
  no message to match);
- inside ``contextlib.suppress`` of it; or
- in the body of a ``try`` whose ``except`` takes it and neither raises,
  asserts nor calls ``pytest.fail``.

A handler takes it when it names ``TimeoutError``, an alias, or a type
``TimeoutError`` is one of, directly or in a tuple. It applies to the code
written inside it, not to a function defined there.

``N`` is worked out from literals and constants, as the wall-clock rule works
out its bounds (``architecture_wall_clock_scan_test.py``), or from a
parameter's default when ``N`` is a parameter nothing else assigns. Anything
else reads as unknown and is not reported.
"""

import ast
import io
import re
import tokenize
from collections.abc import Iterator

import backend.util.architecture_wall_clock_scan_test as scan

_CAPS = {
    "asyncio.wait_for": "wait_for",
    "asyncio.tasks.wait_for": "wait_for",
    "asyncio.timeout": "timeout",
    "asyncio.timeouts.timeout": "timeout",
    "asyncio.timeout_at": "timeout_at",
    "asyncio.timeouts.timeout_at": "timeout_at",
}
# Where each cap takes its seconds: the position, and the keyword.
_SECONDS_ARGUMENT = {"wait_for": (1, "timeout"), "timeout": (0, "delay")}
# TimeoutError, its aliases and the types it is one of: catching any of them
# catches a cap's expiry.
_TIMEOUT_TYPES = frozenset(
    {
        "TimeoutError",
        "asyncio.TimeoutError",
        "asyncio.exceptions.TimeoutError",
        "concurrent.futures.TimeoutError",
        "socket.timeout",
        "OSError",
        "EnvironmentError",
        "IOError",
        "Exception",
        "BaseException",
    }
)
_REASON = re.compile(r"#\s*hang-cap:\s*\S")
# A module without both can't reach a cap, so it is not parsed: the name is in
# the call, or in the import that aliases it.
_CAP_NAMES = re.compile(r"\b(?:wait_for|timeout|timeout_at)\b")
_Function = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
_COMPOUND_STATEMENTS = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.With,
    ast.AsyncWith,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.Try,
    ast.TryStar,
    ast.Match,
)


def hang_caps(source: str, package: str) -> list[tuple[int, float, bool]]:
    """Each cap in *source* on work its test expects to finish, whose
    seconds are known: its line, its seconds, and whether a ``# hang-cap:
    <reason>`` comment covers it."""
    if "asyncio" not in source or not _CAP_NAMES.search(source):
        return []
    tree = ast.parse(source)
    reader = _Reader(source, package)
    reader.walk(tree, scan._Scope(tree, package, None), handled=False, stmt=None)
    return reader.caps


class _Reader:
    def __init__(self, source: str, package: str) -> None:
        self.source = source
        self.lines = source.splitlines()
        # Tokenized only once a cap needs it: most modules have none.
        self.reasons: set[int] | None = None
        self.package = package
        # The function each scope is, to read its parameters' defaults.
        self.functions: dict[int, _Function] = {}
        self.caps: list[tuple[int, float, bool]] = []

    def walk(
        self,
        node: ast.AST,
        scope: scan._Scope,
        handled: bool,
        stmt: ast.stmt | None,
    ) -> None:
        """Read *node*; *handled* says whether a handler around it takes a
        cap's expiry, and *stmt* is the simple statement it is part of."""
        if isinstance(node, ast.stmt) and not isinstance(node, _COMPOUND_STATEMENTS):
            stmt = node
        if isinstance(node, ast.Call) and not handled:
            self.read_call(node, scope, stmt)
        if isinstance(node, _FUNCTIONS):
            scope = scan._Scope(node, self.package, scope)
            self.functions[id(scope)] = node
            handled = False
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                self.walk(item, scope, handled, stmt)
            inside = handled or any(
                _handles_expiry(item.context_expr, scope) for item in node.items
            )
            for child in node.body:
                self.walk(child, scope, inside, stmt)
            return
        if isinstance(node, (ast.Try, ast.TryStar)):
            caught = any(_takes_expiry(handler, scope) for handler in node.handlers)
            for child in node.body:
                self.walk(child, scope, handled or caught, stmt)
            for child in (*node.handlers, *node.orelse, *node.finalbody):
                self.walk(child, scope, handled, stmt)
            return
        for child in ast.iter_child_nodes(node):
            self.walk(child, scope, handled, stmt)

    def read_call(
        self, call: ast.Call, scope: scan._Scope, stmt: ast.stmt | None
    ) -> None:
        kind = _CAPS.get(scan._dotted_origin(call.func, scope) or "")
        seconds = None if kind is None else self.cap_seconds(kind, call, scope)
        if seconds is not None:
            reason = self.gives_a_reason(stmt or call)
            self.caps.append((call.lineno, seconds, reason))

    def cap_seconds(
        self, kind: str, call: ast.Call, scope: scan._Scope
    ) -> float | None:
        if kind == "timeout_at":
            when = _argument(call, 0, "when")
            if when is None:
                return None
            if isinstance(when, ast.Name):
                return scan._deadline_offset(when, scope)
            # A clock read in the argument is taken as the cap is set.
            return scan._earlier_read_plus(when, scope, True)
        index, keyword = _SECONDS_ARGUMENT[kind]
        value = _argument(call, index, keyword)
        if value is None:
            return None
        seconds = scan._seconds(value, scope)
        if seconds is None and isinstance(value, ast.Name):
            return self.default_seconds(value.id, scope)
        return seconds

    def default_seconds(self, name: str, scope: scan._Scope) -> float | None:
        """The default of the parameter *name*, when nothing else binds it."""
        values, origin, owner = scope.lookup(name)
        function = self.functions.get(id(owner))
        if function is None or origin is not None or len(values) != 1:
            return None
        default = _parameter_defaults(function).get(name)
        if default is None or owner.outer is None:
            return None
        return scan._seconds(default, owner.outer)

    def gives_a_reason(self, node: ast.stmt | ast.expr) -> bool:
        """A reason on one of *node*'s lines or in the comments right above."""
        if self.reasons is None:
            self.reasons = _reason_lines(self.source)
        first = node.lineno
        while first > 1 and self.lines[first - 2].lstrip().startswith("#"):
            first -= 1
        last = node.end_lineno or node.lineno
        return any(line in self.reasons for line in range(first, last + 1))


def _argument(call: ast.Call, index: int, keyword: str) -> ast.expr | None:
    if len(call.args) > index:
        return call.args[index]
    return _keyword(call, keyword)


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _parameter_defaults(function: _Function) -> dict[str, ast.expr]:
    arguments = function.args
    positional = [*arguments.posonlyargs, *arguments.args]
    with_defaults = positional[len(positional) - len(arguments.defaults) :]
    defaults = {arg.arg: value for arg, value in zip(with_defaults, arguments.defaults)}
    for arg, value in zip(arguments.kwonlyargs, arguments.kw_defaults):
        if value is not None:
            defaults[arg.arg] = value
    return defaults


def _handles_expiry(expr: ast.expr, scope: scan._Scope) -> bool:
    """Whether ``with`` *expr* takes a cap's ``TimeoutError``."""
    if not isinstance(expr, ast.Call):
        return False
    origin = scan._dotted_origin(expr.func, scope)
    if origin == "contextlib.suppress":
        return any(_is_timeout_type(arg, scope) for arg in expr.args)
    if origin != "pytest.raises":
        return False
    match = _keyword(expr, "match")
    if match is not None and not (
        isinstance(match, ast.Constant) and match.value is None
    ):
        return False
    expected = _argument(expr, 0, "expected_exception")
    return expected is not None and _is_timeout_type(expected, scope)


def _takes_expiry(handler: ast.ExceptHandler, scope: scan._Scope) -> bool:
    """Whether *handler* takes a cap's ``TimeoutError`` and lets the test go
    on: it neither raises, asserts nor calls ``pytest.fail``."""
    if handler.type is not None and not _is_timeout_type(handler.type, scope):
        return False
    return not any(_fails(node, scope) for node in _own_nodes(handler.body))


def _fails(node: ast.AST, scope: scan._Scope) -> bool:
    if isinstance(node, (ast.Raise, ast.Assert)):
        return True
    return (
        isinstance(node, ast.Call)
        and scan._dotted_origin(node.func, scope) == "pytest.fail"
    )


def _own_nodes(body: list[ast.stmt]) -> Iterator[ast.AST]:
    """Every node of *body* outside the functions defined in it."""
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _FUNCTIONS):
            stack.extend(ast.iter_child_nodes(node))


def _is_timeout_type(node: ast.expr, scope: scan._Scope) -> bool:
    if isinstance(node, (ast.Tuple, ast.List)):
        return any(_is_timeout_type(element, scope) for element in node.elts)
    if isinstance(node, ast.Name):
        values, origin, _ = scope.lookup(node.id)
        # A name the code assigns is not read.
        return not values and (origin or node.id) in _TIMEOUT_TYPES
    return (scan._dotted_origin(node, scope) or "") in _TIMEOUT_TYPES


def _reason_lines(source: str) -> set[int]:
    """Lines with a ``# hang-cap: <reason>`` comment, a comment token and not
    text inside a string."""
    return {
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and _REASON.search(token.string)
    }
