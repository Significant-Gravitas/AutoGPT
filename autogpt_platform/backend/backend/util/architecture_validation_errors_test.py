import ast

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
# count and length hold everywhere. Scoped to `backend/api`, where the
# validated value comes from the request.

_VALIDATION_ERROR_HELPER = "api/utils/validation_errors.py"
# pydantic's and FastAPI's own. Our `GraphValidationError` carries a body we
# build, and is not one of these.
_VALIDATION_ERROR_NAMES = frozenset({"ValidationError", "RequestValidationError"})
# A pydantic `ValidationError` is a `ValueError`, so these catch one too.
_BROAD_EXCEPTION_NAMES = frozenset({"ValueError", "Exception", "BaseException"})
_VALIDATE_CALLS = frozenset(
    {"model_validate", "model_validate_json", "validate_python", "validate_json"}
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
        "`bounded_validation_errors(exc)` instead of calling `.errors()` on it "
        "or rendering it, or its traceback, into a log line:\n"
        + "\n".join(sorted(offenders))
    )


def _validation_error_reads(tree: ast.Module) -> list[tuple[int, str]]:
    """Every `.errors()` call, and every other use of a caught ValidationError."""
    found = [
        (node.lineno, "calls `.errors()`")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "errors"
    ]
    for handler in _handlers_seeing_a_validation_error(tree):
        if handler.name:
            found += [
                (use.lineno, f"renders the caught `{handler.name}`")
                for use in _unbounded_uses(handler, handler.name)
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
            for use in _unbounded_uses(function, argument.arg)
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


def _unbounded_uses(scope: ast.AST, name: str) -> list[ast.Name]:
    """Loads of `name` other than `bounded_validation_errors(name)`/`raise … from name`."""
    bounded = {
        id(argument)
        for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and _decorator_name(node.func) == "bounded_validation_errors"
        for argument in node.args
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
