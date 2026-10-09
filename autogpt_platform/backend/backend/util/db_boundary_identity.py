"""Fingerprint reference syntax without line numbers or import-alias spelling."""

import ast
from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256


def reference_identity(
    node: ast.AST, resolve_name: Callable[[ast.expr], set[str]]
) -> str:
    canonical = _CanonicalNames(resolve_name).visit(deepcopy(node))
    syntax = _syntax(canonical)
    return sha256(syntax.encode("utf-8")).hexdigest()


def _syntax(value: object) -> str:
    if isinstance(value, ast.AST):
        fields = ",".join(
            f"{name}={_syntax(child)}" for name, child in ast.iter_fields(value)
        )
        return f"{type(value).__name__}({fields})"
    if isinstance(value, list):
        return "[" + ",".join(_syntax(child) for child in value) + "]"
    return repr(value)


class _CanonicalNames(ast.NodeTransformer):
    def __init__(self, resolve_name: Callable[[ast.expr], set[str]]):
        self.resolve_name = resolve_name

    def visit_Name(self, node: ast.Name) -> ast.Name:
        targets = self.resolve_name(node)
        if targets:
            return ast.Name(id="|".join(sorted(targets)), ctx=node.ctx)
        return node
