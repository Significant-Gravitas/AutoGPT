import ast
import collections
import functools

from backend.util.architecture_test import (
    BACKEND_ROOT,
    _argument,
    _iter_backend_py_files,
)

# ---------------------------------------------------------------------------
# Rule: a response's Content-Type comes from a literal or a constant
# ---------------------------------------------------------------------------
#
# Serve stored bytes through ``workspace.routes.create_file_download_response``,
# or list the function below with the reason its types are safe.

_RESPONSE_CLASSES = frozenset({"Response", "StreamingResponse", "FileResponse"})
# `media_type` is the fourth positional parameter of all three classes.
_MEDIA_TYPE_POSITION = 3

# "path function" -> why the type it sends is safe. Only shrinks.
_DATA_TYPED_RESPONSE_SITES = {
    "api/features/workspace/routes.py _create_streaming_response": (
        "attachment or inline disposition, with `sandbox` and `nosniff`"
    ),
    "api/features/store/routes.py get_store_media": (
        "type from local_media's raster image and video allowlist"
    ),
    "api/features/store/routes.py get_private_submission_media": (
        "type from local_media's raster image and video allowlist"
    ),
}


def test_response_content_type_is_not_read_from_data():
    offenders = [
        f"  backend/{site} (line {lineno})"
        for site, lineno in _data_typed_responses()
        if site not in _DATA_TYPED_RESPONSE_SITES
    ]
    assert not offenders, (
        "These responses take their Content-Type from data. Use a literal or a "
        "module constant, serve stored bytes through "
        "`create_file_download_response`, or add the function to "
        "`_DATA_TYPED_RESPONSE_SITES` with the reason its types are safe:\n"
        + "\n".join(sorted(offenders))
    )


def test_data_typed_response_sites_are_not_stale():
    found = {site for site, _ in _data_typed_responses()}
    stale = sorted(set(_DATA_TYPED_RESPONSE_SITES) - found)
    assert not stale, (
        "Remove these `_DATA_TYPED_RESPONSE_SITES` entries; they no longer "
        "build a data-typed response:\n  " + "\n  ".join(stale)
    )


@functools.cache
def _data_typed_responses() -> tuple[tuple[str, int], ...]:
    sites: list[tuple[str, int]] = []
    for py in _iter_backend_py_files():
        if py.name.endswith("_test.py") or py.name == "conftest.py":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        rel = py.relative_to(BACKEND_ROOT).as_posix()
        classes = _RESPONSE_CLASSES | _response_class_aliases(tree)
        for function, nodes in _nodes_by_function(tree):
            scope = _assignments(nodes)
            sites.extend(
                (f"{rel} {function}", node.lineno)
                for node in nodes
                if _is_data_typed_site(node, classes, scope)
            )
    return tuple(sites)


def _response_class_aliases(tree: ast.AST) -> set[str]:
    return {
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name in _RESPONSE_CLASSES and alias.asname
    }


def _nodes_by_function(tree: ast.AST):
    scopes: list[tuple[str, ast.AST]] = [("<module>", tree)]
    while scopes:
        function, scope = scopes.pop()
        nodes: list[ast.AST] = []
        pending = list(ast.iter_child_nodes(scope))
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scopes.append((node.name, node))
                continue
            nodes.append(node)
            pending.extend(ast.iter_child_nodes(node))
        yield function, nodes


# (name -> values assigned to it, name -> values set as its Content-Type key)
_Scope = tuple[dict[str, list[ast.expr]], dict[str, list[ast.expr]]]


def _assignments(nodes: list[ast.AST]) -> _Scope:
    assigned: dict[str, list[ast.expr]] = collections.defaultdict(list)
    content_types: dict[str, list[ast.expr]] = collections.defaultdict(list)
    for node in nodes:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                assigned[target.id].append(value)
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and _is_content_type(target.slice)
            ):
                content_types[target.value.id].append(value)
    return assigned, content_types


def _is_data_typed_site(node: ast.AST, classes: set[str], scope: _Scope) -> bool:
    if isinstance(node, ast.Call):
        return _response_type_is_data(node, classes, scope)
    # `response.headers["Content-Type"] = ...` after construction
    return (
        isinstance(node, ast.Assign)
        and _is_data(node.value)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr == "headers"
            and _is_content_type(target.slice)
            for target in node.targets
        )
    )


def _response_type_is_data(call: ast.Call, classes: set[str], scope: _Scope) -> bool:
    callee = call.func
    name = callee.attr if isinstance(callee, ast.Attribute) else None
    if isinstance(callee, ast.Name):
        name = callee.id
    if name not in classes:
        return False
    if _is_data(_argument(call, _MEDIA_TYPE_POSITION, "media_type")):
        return True
    return _headers_type_is_data(_argument(call, 2, "headers"), scope)


def _headers_type_is_data(
    headers: ast.expr | None, scope: _Scope, seen: frozenset[str] = frozenset()
) -> bool:
    if isinstance(headers, ast.Dict):
        return any(
            (
                _headers_type_is_data(value, scope, seen)
                if key is None
                else _is_content_type(key) and _is_data(value)
            )
            for key, value in zip(headers.keys, headers.values)
        )
    if not isinstance(headers, ast.Name) or headers.id in seen:
        return False
    assigned, content_types = scope
    seen = seen | {headers.id}
    return any(_is_data(v) for v in content_types.get(headers.id, ())) or any(
        _headers_type_is_data(v, scope, seen) for v in assigned.get(headers.id, ())
    )


def _is_content_type(expr: ast.expr) -> bool:
    return isinstance(expr, ast.Constant) and str(expr.value).lower() == "content-type"


def _is_data(expr: ast.expr | None) -> bool:
    if expr is None or isinstance(expr, ast.Constant):
        return False
    return not (isinstance(expr, ast.Name) and expr.id.isupper())
