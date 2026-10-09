"""Fingerprint reference syntax without line numbers or import-alias spelling."""

import ast
from collections.abc import Callable
from hashlib import sha256


def reference_identity(
    node: ast.AST, resolve_name: Callable[[ast.expr], set[str]]
) -> str:
    return sha256(_syntax(node, resolve_name).encode("utf-8")).hexdigest()


def _syntax(value: object, resolve_name: Callable[[ast.expr], set[str]]) -> str:
    if isinstance(value, ast.Name) and (targets := resolve_name(value)):
        value = ast.Name(id="|".join(sorted(targets)), ctx=value.ctx)
    if isinstance(value, ast.AST):
        fields = ",".join(
            f"{name}={_syntax(child, resolve_name)}"
            for name, child in ast.iter_fields(value)
        )
        return f"{type(value).__name__}({fields})"
    if isinstance(value, list):
        return "[" + ",".join(_syntax(child, resolve_name) for child in value) + "]"
    return repr(value)
