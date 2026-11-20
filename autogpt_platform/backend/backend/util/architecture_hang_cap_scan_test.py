"""Reads the caps a test module puts on its own waits, for the rule in
``architecture_hang_cap_test.py``.

A cap is ``asyncio.wait_for(aw, N)``, ``asyncio.timeout(N)`` or
``asyncio.timeout_at(<clock read> + N)``, aliases included. When one runs out
it raises ``TimeoutError`` at the test's own await, so it caps work the test
expects to finish unless the test handles that error there:

- inside ``pytest.raises`` of it, with no ``match`` or ``check`` (the cap's
  own error has no message, and a check can refuse it);
- inside ``contextlib.suppress`` of it; or
- in the body of a ``try`` whose first handler to take it names it and
  neither raises, asserts nor calls ``pytest.fail``.

Only ``TimeoutError`` and its aliases (``asyncio.TimeoutError``,
``builtins.TimeoutError`` and the rest), directly or in a tuple, handle a
cap. A broad handler (a type ``TimeoutError`` is one of, or a bare
``except``) takes the expiry too, but a test that swallows every error still
has later asserts to fail on a slow runner, so it handles nothing; as the
first handler to take the expiry, it also keeps a later ``except
TimeoutError`` from handling it. A handler applies to the code written inside
it, not to a function defined there.

A ``# hang-cap: <reason>`` comment counts on any line of the cap's simple
statement, or of the header it is part of (a multi-line ``async with`` up to
its colon), or in the comments right above either.

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
# TimeoutError and its aliases: a handler of one handles a cap's expiry.
_TIMEOUT_TYPES = frozenset(
    {
        "TimeoutError",
        "asyncio.TimeoutError",
        "asyncio.exceptions.TimeoutError",
        "concurrent.futures.TimeoutError",
        "socket.timeout",
    }
)
# The types TimeoutError is one of: a handler of one swallows the expiry.
_TIMEOUT_BASES = frozenset(
    {"OSError", "EnvironmentError", "IOError", "Exception", "BaseException"}
)
_HANDLES, _SWALLOWS = "handles", "swallows"
_REASON = re.compile(r"#\s*hang-cap:\s*\S")
# A module without both can't reach a cap, so it is not parsed: the name is in
# the call, or in the import that aliases it.
_CAP_NAMES = re.compile(r"\b(?:wait_for|timeout|timeout_at)\b")
_Function = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
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
# Compound statements whose header is an expression after the keyword, up to
# the colon before the body.
_HEADED = (ast.With, ast.AsyncWith, ast.For, ast.AsyncFor, ast.While, ast.If)
_Headed = ast.With | ast.AsyncWith | ast.For | ast.AsyncFor | ast.While | ast.If
# A compound statement's fields that are not its header.
_BODY_FIELDS = frozenset({"body", "orelse", "finalbody", "handlers", "cases"})
_Span = tuple[int, int]


def hang_caps(source: str, package: str) -> list[tuple[int, float, bool]]:
    """Each cap in *source* on work its test expects to finish, whose
    seconds are known: its line, its seconds, and whether a ``# hang-cap:
    <reason>`` comment covers it."""
    if "asyncio" not in source or not _CAP_NAMES.search(source):
        return []
    tree = ast.parse(source)
    reader = _Reader(source, package)
    reader.walk(tree, scan._Scope(tree, package, None), handled=False, span=None)
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
        self, node: ast.AST, scope: scan._Scope, handled: bool, span: _Span | None
    ) -> None:
        """Read *node*: *handled* says whether a handler around it takes a
        cap's expiry, and *span* holds the lines a cap's reason may be on."""
        if isinstance(node, _COMPOUND_STATEMENTS):
            self.walk_compound(node, scope, handled)
            return
        if isinstance(node, ast.stmt):
            span = (node.lineno, node.end_lineno or node.lineno)
        if isinstance(node, ast.Call) and not handled:
            self.read_call(node, scope, span)
        if isinstance(node, ast.Lambda):
            # Its defaults are worked out where it is written, its body when
            # it is called, outside any handler around the definition.
            self.walk(node.args, scope, handled, span)
            self.walk(node.body, self.enter(node, scope), False, span)
            return
        for child in ast.iter_child_nodes(node):
            self.walk(child, scope, handled, span)

    def walk_compound(self, node: ast.stmt, scope: scan._Scope, handled: bool) -> None:
        header = [
            child
            for name, value in ast.iter_fields(node)
            if name not in _BODY_FIELDS
            for child in _nodes(value)
        ]
        # The header (with its decorators and defaults) runs in the scope
        # around the statement.
        span = self.header_span(node, header) if isinstance(node, _HEADED) else None
        for child in header:
            self.walk(child, scope, handled, span)
        if isinstance(node, (ast.Try, ast.TryStar)):
            caught = handled or _takes_expiry(node.handlers, scope)
            for child in node.body:
                self.walk(child, scope, caught, None)
            for child in (*node.handlers, *node.orelse, *node.finalbody):
                self.walk(child, scope, handled, None)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scope, handled = self.enter(node, scope), False
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            handled = handled or any(
                _handles_expiry(item.context_expr, scope) for item in node.items
            )
        for name, value in ast.iter_fields(node):
            if name in _BODY_FIELDS:
                for child in _nodes(value):
                    self.walk(child, scope, handled, None)

    def enter(self, function: _Function, scope: scan._Scope) -> scan._Scope:
        inner = scan._Scope(function, self.package, scope)
        self.functions[id(inner)] = function
        return inner

    def header_span(self, node: _Headed, header: list[ast.AST]) -> _Span:
        """From the statement's first line to its header's last: the last
        line of code before its body, and no earlier than where the header's
        own expressions end."""
        last = node.body[0].lineno - 1
        while last > node.lineno and _is_comment_or_blank(self.lines[last - 1]):
            last -= 1
        ends = [
            part.end_lineno or part.lineno
            for child in header
            for part in ast.walk(child)
            if isinstance(part, ast.expr)
        ]
        return node.lineno, max([last, node.lineno, *ends])

    def read_call(self, call: ast.Call, scope: scan._Scope, span: _Span | None) -> None:
        kind = _CAPS.get(scan._dotted_origin(call.func, scope) or "")
        seconds = None if kind is None else self.cap_seconds(kind, call, scope)
        if seconds is not None:
            own = (call.lineno, call.end_lineno or call.lineno)
            self.caps.append((call.lineno, seconds, self.gives_a_reason(span or own)))

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

    def gives_a_reason(self, span: _Span) -> bool:
        """A reason on one of the span's lines or in the comments right above."""
        if self.reasons is None:
            self.reasons = _reason_lines(self.source)
        first, last = span
        while first > 1 and self.lines[first - 2].lstrip().startswith("#"):
            first -= 1
        return any(line in self.reasons for line in range(first, last + 1))


def _nodes(value: object) -> Iterator[ast.AST]:
    """The nodes a field of a node holds: one, a list of them, or none."""
    if isinstance(value, ast.AST):
        yield value
    elif isinstance(value, list):
        yield from (item for item in value if isinstance(item, ast.AST))


def _is_comment_or_blank(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.startswith("#")


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
    """Whether ``with`` *expr* handles a cap's ``TimeoutError``."""
    if not isinstance(expr, ast.Call):
        return False
    origin = scan._dotted_origin(expr.func, scope)
    if origin == "contextlib.suppress":
        return any(_catches(arg, scope) == _HANDLES for arg in expr.args)
    if origin != "pytest.raises":
        return False
    if any(_is_set(_keyword(expr, name)) for name in ("match", "check")):
        return False
    expected = _argument(expr, 0, "expected_exception")
    return expected is not None and _catches(expected, scope) == _HANDLES


def _is_set(value: ast.expr | None) -> bool:
    return value is not None and not (
        isinstance(value, ast.Constant) and value.value is None
    )


def _takes_expiry(handlers: list[ast.ExceptHandler], scope: scan._Scope) -> bool:
    """Whether the first of *handlers* to take a cap's ``TimeoutError`` (the
    one Python runs) handles it, and lets the test go on: it neither raises,
    asserts nor calls ``pytest.fail``."""
    for handler in handlers:
        kind = _catches(handler.type, scope)
        if kind is not None:
            return kind == _HANDLES and not any(
                _fails(node, scope) for node in _own_nodes(handler.body)
            )
    return False


def _catches(node: ast.expr | None, scope: scan._Scope) -> str | None:
    """How a handler of *node* (``None`` for a bare ``except``) takes a cap's
    ``TimeoutError``: it handles it, swallows it, or doesn't take it."""
    if node is None:
        return _SWALLOWS
    if isinstance(node, (ast.Tuple, ast.List)):
        kinds = {_catches(element, scope) for element in node.elts}
        return next((kind for kind in (_HANDLES, _SWALLOWS) if kind in kinds), None)
    origin = _exception_origin(node, scope)
    if origin in _TIMEOUT_TYPES:
        return _HANDLES
    return _SWALLOWS if origin in _TIMEOUT_BASES else None


def _exception_origin(node: ast.expr, scope: scan._Scope) -> str | None:
    """The type a handler names, ``builtins.`` dropped, or ``None`` for a
    name the code assigns, which is not read."""
    if isinstance(node, ast.Name):
        values, origin, _ = scope.lookup(node.id)
        dotted = None if values else origin or node.id
    else:
        dotted = scan._dotted_origin(node, scope)
    return dotted.removeprefix("builtins.") if dotted else None


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
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))


def _reason_lines(source: str) -> set[int]:
    """Lines with a ``# hang-cap: <reason>`` comment, a comment token and not
    text inside a string."""
    return {
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and _REASON.search(token.string)
    }
