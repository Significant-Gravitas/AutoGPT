import ast

import pytest

from backend.util.architecture_test import (
    BACKEND_ROOT,
    _annotation_names,
    _decorator_name,
    _iter_backend_py_files,
)

# ---------------------------------------------------------------------------
# Rule: the API reads a validation error only through the bounded helper
# ---------------------------------------------------------------------------
#
# Motivation: `bounded_validation_errors` is the one place a pydantic or
# FastAPI validation error becomes response or log content, so its limits on
# count and length hold everywhere. A log line takes `loggable_validation_errors`
# or `describe_validation_error`, which keep the caller's own data out of it.
# Scoped to `backend/api`, where the validated value comes from the request.

_VALIDATION_ERROR_HELPER = "api/utils/validation_errors.py"
_HELPER_MODULE = "backend.api.utils.validation_errors"
_BOUNDED_READERS = frozenset(
    {
        "bounded_validation_errors",
        "loggable_validation_errors",
        "describe_validation_error",
    }
)
# pydantic's and FastAPI's own. Our `GraphValidationError` carries a body we
# build, and is not one of these.
_VALIDATION_ERROR_NAMES = frozenset({"ValidationError", "RequestValidationError"})
# A pydantic `ValidationError` is a `ValueError`, so these catch one too.
_BROAD_EXCEPTION_NAMES = frozenset({"ValueError", "Exception", "BaseException"})
_VALIDATE_CALLS = frozenset(
    {"model_validate", "model_validate_json", "validate_python", "validate_json"}
)
# Keeps what the caller sent in a location or a message, for the response.
_RESPONSE_READER = "bounded_validation_errors"
_LOG_METHODS = frozenset(
    {"debug", "info", "warning", "error", "exception", "critical", "log"}
)
# Validates the skills catalog's expert files, never a request.
_NOT_REQUEST_VALIDATION = frozenset({"api/features/experts/roster.py"})


def test_api_reads_validation_errors_only_through_the_bounded_helper():
    offenders: list[str] = []
    for py in _iter_backend_py_files():
        rel = py.relative_to(BACKEND_ROOT).as_posix()
        if not rel.startswith("api/") or rel.endswith("_test.py"):
            continue
        if rel == _VALIDATION_ERROR_HELPER or rel in _NOT_REQUEST_VALIDATION:
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        offenders += [
            f"  backend/{rel}:{lineno} {reason}"
            for lineno, reason in _validation_error_reads(tree)
        ]

    assert not offenders, (
        "A validation error's detail is read only by "
        f"`backend/{_VALIDATION_ERROR_HELPER}`, which caps how many errors it "
        "reports and how long each string is. Pass the exception to "
        "`bounded_validation_errors(exc)`, or `loggable_validation_errors(exc)` "
        "for a log line, instead of calling `.errors()` on it or rendering it, "
        "or its traceback, into a log line:\n" + "\n".join(sorted(offenders))
    )


def _validation_error_reads(tree: ast.Module) -> list[tuple[int, str]]:
    """Every `.errors()` call, and every other use of a caught ValidationError."""
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    readers = _reader_calls(tree, parents)
    # A local the response's rendering was assigned to, by its scope and name.
    rendered_locals = {
        (id(_scope_of(node, parents)), target.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and readers.get(id(node.value)) == _RESPONSE_READER
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    found = [
        (node.lineno, "calls `.errors()`")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "errors"
    ]
    found += [
        (call.lineno, f"logs `{_RESPONSE_READER}`, the response's rendering")
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in _LOG_METHODS
        for argument in [*call.args, *(keyword.value for keyword in call.keywords)]
        for inner in ast.walk(argument)
        if readers.get(id(inner)) == _RESPONSE_READER
        or (
            isinstance(inner, ast.Name)
            and isinstance(inner.ctx, ast.Load)
            and (id(_scope_of(inner, parents)), inner.id) in rendered_locals
        )
    ]
    for handler in _handlers_seeing_a_validation_error(tree):
        if handler.name:
            found += [
                (use.lineno, f"renders the caught `{handler.name}`")
                for use in _unbounded_uses(handler, handler.name, readers)
            ]
        if _catches_a_validation_error(handler.type):
            found += [
                (call.lineno, "logs the caught error's traceback")
                for call in ast.walk(handler)
                if isinstance(call, ast.Call) and _logs_a_traceback(call)
            ]
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = function.args
        found += [
            (use.lineno, f"renders the parameter `{argument.arg}`")
            for argument in [
                *arguments.posonlyargs,
                *arguments.args,
                *arguments.kwonlyargs,
            ]
            if _annotation_names(argument.annotation) & _VALIDATION_ERROR_NAMES
            for use in _unbounded_uses(function, argument.arg, readers)
        ]
    return found


def _handlers_seeing_a_validation_error(tree: ast.Module):
    """Handlers naming a validation error, and broad ones around a validate call
    that no earlier handler of the same `try` catches it in."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Try, ast.TryStar)):
            continue
        validates = any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr in _VALIDATE_CALLS
            for statement in node.body
            for call in ast.walk(statement)
        )
        for handler in node.handlers:
            if _catches_a_validation_error(handler.type):
                yield handler
                break
            if validates and _catches_broadly(handler.type):
                yield handler


def _catches_a_validation_error(caught: ast.expr | None) -> bool:
    names = caught.elts if isinstance(caught, ast.Tuple) else [caught]
    return any(
        _decorator_name(name) in _VALIDATION_ERROR_NAMES
        for name in names
        if name is not None
    )


def _catches_broadly(caught: ast.expr | None) -> bool:
    names = caught.elts if isinstance(caught, ast.Tuple) else [caught]
    return any(
        name is None or _decorator_name(name) in _BROAD_EXCEPTION_NAMES
        for name in names
    )


def _logs_a_traceback(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute) and call.func.attr == "exception"
    ) or any(keyword.arg == "exc_info" for keyword in call.keywords)


_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _reader_calls(tree: ast.Module, parents: dict[ast.AST, ast.AST]) -> dict[int, str]:
    """The calls, by identity, of a name that resolves to one of the helper's
    readers, imported from the helper in the call's own function, an enclosing
    one or the module and bound to nothing else there, with the reader's name.
    As in Python, a class body's names are seen only from the body itself."""
    # Each scope's names, with the reader each is bound to, or None for any
    # other binding, which then decides the name in that scope.
    bindings: dict[int, dict[str, str | None]] = {}
    for node in ast.walk(tree):
        for name, reader in _bound_names(node):
            table = bindings.setdefault(id(_scope_of(node, parents)), {})
            table[name] = reader if table.get(name, reader) == reader else None
    calls: dict[int, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        scope: ast.AST | None = _scope_of(node, parents)
        own = True
        while scope is not None:
            table = (
                bindings.get(id(scope), {})
                if own or not isinstance(scope, ast.ClassDef)
                else {}
            )
            if node.func.id in table:
                if reader := table[node.func.id]:
                    calls[id(node)] = reader
                break
            own = False
            scope = _scope_of(scope, parents) if scope in parents else None
    return calls


def _bound_names(node: ast.AST) -> list[tuple[str, str | None]]:
    """The names `node` binds in its scope, each with the helper's reader it is
    imported as, if it is one."""
    if isinstance(node, ast.ImportFrom):
        return [
            (
                alias.asname or alias.name,
                (
                    alias.name
                    if node.module == _HELPER_MODULE and alias.name in _BOUNDED_READERS
                    else None
                ),
            )
            for alias in node.names
        ]
    if isinstance(node, ast.Import):
        return [
            ((alias.asname or alias.name).split(".")[0], None) for alias in node.names
        ]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [(node.name, None)]
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        return [(node.id, None)]
    if isinstance(node, ast.arg):
        return [(node.arg, None)]
    if isinstance(node, ast.ExceptHandler) and node.name:
        return [(node.name, None)]
    return []


def _scope_of(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST:
    """The function, lambda or module whose names `node` binds or reads."""
    scope = parents[node]
    while not isinstance(scope, _SCOPES):
        scope = parents[scope]
    return scope


def _unbounded_uses(
    scope: ast.AST, name: str, readers: dict[int, str]
) -> list[ast.Name]:
    """Loads of `name` other than a reader call's argument/`raise … from name`."""
    bounded = {
        id(argument)
        for node in ast.walk(scope)
        if isinstance(node, ast.Call) and id(node) in readers
        for argument in [
            *node.args,
            *(keyword.value for keyword in node.keywords if keyword.arg == "exc"),
        ]
    } | {
        id(node.cause)
        for node in ast.walk(scope)
        if isinstance(node, ast.Raise) and node.cause is not None
    }
    return [
        node
        for node in ast.walk(scope)
        if isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        and id(node) not in bounded
    ]


@pytest.mark.parametrize(
    "source, flagged",
    [
        (
            "from backend.api.utils.validation_errors import describe_validation_error\n"
            "try:\n    pass\nexcept ValidationError as exc:\n"
            "    log(describe_validation_error(exc))\n",
            False,
        ),
        # A method of the same name elsewhere renders what it likes.
        (
            "try:\n    pass\nexcept ValidationError as exc:\n"
            "    log(legacy.describe_validation_error(exc))\n",
            True,
        ),
        (
            "def describe_validation_error(e):\n    return str(e)\n"
            "try:\n    pass\nexcept ValidationError as exc:\n"
            "    log(describe_validation_error(exc))\n",
            True,
        ),
        # Imported only in a sibling function, so not where the call is.
        (
            "def other():\n"
            "    from backend.api.utils.validation_errors import describe_validation_error\n"
            "def report():\n"
            "    try:\n        pass\n    except ValidationError as exc:\n"
            "        log(describe_validation_error(exc))\n",
            True,
        ),
        # Imported in the function that calls it.
        (
            "def report():\n"
            "    from backend.api.utils.validation_errors import describe_validation_error\n"
            "    try:\n        pass\n    except ValidationError as exc:\n"
            "        log(describe_validation_error(exc))\n",
            False,
        ),
        # A sibling imports it, and the module's own is the one called here.
        (
            "def other():\n"
            "    from backend.api.utils.validation_errors import describe_validation_error\n"
            "def describe_validation_error(e):\n    return str(e)\n"
            "def report():\n"
            "    try:\n        pass\n    except ValidationError as exc:\n"
            "        log(describe_validation_error(exc))\n",
            True,
        ),
        # The response's rendering keeps what the caller sent, so not in a log.
        (
            "from backend.api.utils.validation_errors import bounded_validation_errors\n"
            "try:\n    pass\nexcept ValidationError as exc:\n"
            "    logger.error('Invalid: %s', bounded_validation_errors(exc))\n",
            True,
        ),
        (
            "from backend.api.utils.validation_errors import bounded_validation_errors\n"
            "try:\n    pass\nexcept ValidationError as exc:\n"
            "    logger.warning(f'Invalid: {bounded_validation_errors(exc)}')\n",
            True,
        ),
        (
            "from backend.api.utils.validation_errors import loggable_validation_errors\n"
            "try:\n    pass\nexcept ValidationError as exc:\n"
            "    logger.error('Invalid: %s', loggable_validation_errors(exc))\n",
            False,
        ),
        (
            "from backend.api.utils.validation_errors import bounded_validation_errors\n"
            "try:\n    pass\nexcept ValidationError as exc:\n"
            "    raise HTTPException(422, detail=bounded_validation_errors(exc))\n",
            False,
        ),
        # The response's rendering logged through a local.
        (
            "from backend.api.utils.validation_errors import bounded_validation_errors\n"
            "try:\n    pass\nexcept ValidationError as exc:\n"
            "    detail = bounded_validation_errors(exc)\n"
            "    logger.error('Invalid: %s', detail)\n",
            True,
        ),
        (
            "from backend.api.utils.validation_errors import bounded_validation_errors\n"
            "try:\n    pass\nexcept ValidationError as exc:\n"
            "    detail = bounded_validation_errors(exc)\n"
            "    raise HTTPException(422, detail=detail)\n",
            False,
        ),
        # A reader called by keyword.
        (
            "from backend.api.utils.validation_errors import loggable_validation_errors\n"
            "try:\n    pass\nexcept ValidationError as exc:\n"
            "    logger.error('Invalid: %s', loggable_validation_errors(exc=exc))\n",
            False,
        ),
        # A method of the same name binds only in its class.
        (
            "from backend.api.utils.validation_errors import describe_validation_error\n"
            "class Report:\n"
            "    def describe_validation_error(self, e):\n        return str(e)\n"
            "def report():\n"
            "    try:\n        pass\n    except ValidationError as exc:\n"
            "        log(describe_validation_error(exc))\n",
            False,
        ),
    ],
)
def test_only_the_helpers_own_readers_bound_a_caught_error(source: str, flagged: bool):
    assert bool(_validation_error_reads(ast.parse(source))) is flagged
