#!/usr/bin/env python3
"""
Generate a lightweight stub for prisma/types.py that collapses all exported
symbols to Any. This prevents Pyright from spending time/budget on Prisma's
query DSL types while keeping runtime behavior unchanged.

Usage:
    poetry run gen-prisma-stub

This script automatically finds the prisma package location and generates
the types.pyi stub file in the same directory as types.py.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Iterable, Set


def _iter_assigned_names(target: ast.expr) -> Iterable[str]:
    """Extract names from assignment targets (handles tuple unpacking)."""
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _iter_assigned_names(elt)


def _is_private(name: str) -> bool:
    """Check if a name is private (starts with _ but not __)."""
    return name.startswith("_") and not name.startswith("__")


def _is_safe_type_alias(node: ast.Assign) -> bool:
    """Check if an assignment is a safe type alias that shouldn't be stubbed.

    Safe types are:
    - Literal types (don't cause type budget issues)
    - Simple type references (SortMode, SortOrder, etc.)
    - TypeVar definitions
    """
    if not node.value:
        return False

    # Check if it's a Subscript (like Literal[...], Union[...], TypeVar[...])
    if isinstance(node.value, ast.Subscript):
        # Get the base type name
        if isinstance(node.value.value, ast.Name):
            base_name = node.value.value.id
            # Literal types are safe
            if base_name == "Literal":
                return True
            # TypeVar is safe
            if base_name == "TypeVar":
                return True
        elif isinstance(node.value.value, ast.Attribute):
            # Handle typing_extensions.Literal etc.
            if node.value.value.attr == "Literal":
                return True

    # Check if it's a simple Name reference (like SortMode = _types.SortMode)
    if isinstance(node.value, ast.Attribute):
        return True

    # Check if it's a Call (like TypeVar(...))
    if isinstance(node.value, ast.Call):
        if isinstance(node.value.func, ast.Name):
            if node.value.func.id == "TypeVar":
                return True

    return False


def collect_top_level_symbols(
    tree: ast.Module, source_lines: list[str]
) -> tuple[Set[str], Set[str], list[str], Set[str]]:
    """Collect all top-level symbols from an AST module.

    Returns:
        Tuple of (class_names, function_names, safe_variable_sources, unsafe_variable_names)
        safe_variable_sources contains the actual source code lines for safe variables
    """
    classes: Set[str] = set()
    functions: Set[str] = set()
    safe_variable_sources: list[str] = []
    unsafe_variables: Set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not _is_private(node.name):
                classes.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_private(node.name):
                functions.add(node.name)
        elif isinstance(node, ast.Assign):
            is_safe = _is_safe_type_alias(node)
            names = []
            for t in node.targets:
                for n in _iter_assigned_names(t):
                    if not _is_private(n):
                        names.append(n)
            if names:
                if is_safe:
                    # Extract the source code for this assignment
                    start_line = node.lineno - 1  # 0-indexed
                    end_line = node.end_lineno if node.end_lineno else node.lineno
                    source = "\n".join(source_lines[start_line:end_line])
                    safe_variable_sources.append(source)
                else:
                    unsafe_variables.update(names)
        elif isinstance(node, ast.AnnAssign) and node.target:
            # Annotated assignments are always stubbed
            for n in _iter_assigned_names(node.target):
                if not _is_private(n):
                    unsafe_variables.add(n)

    return classes, functions, safe_variable_sources, unsafe_variables


def find_prisma_types_path() -> Path:
    """Find the prisma types.py file in the installed package."""
    spec = importlib.util.find_spec("prisma")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not find prisma package. Is it installed?")

    prisma_dir = Path(spec.origin).parent
    types_path = prisma_dir / "types.py"

    if not types_path.exists():
        raise RuntimeError(f"prisma/types.py not found at {types_path}")

    return types_path


def generate_stub(src_path: Path, stub_path: Path) -> int:
    """Generate the .pyi stub file from the source types.py."""
    code = src_path.read_text(encoding="utf-8", errors="ignore")
    source_lines = code.splitlines()
    tree = ast.parse(code, filename=str(src_path))
    classes, functions, safe_variable_sources, unsafe_variables = (
        collect_top_level_symbols(tree, source_lines)
    )

    header = """\
# -*- coding: utf-8 -*-
# Auto-generated stub file - DO NOT EDIT
# Generated by gen_prisma_types_stub.py
#
# This stub intentionally collapses complex Prisma query DSL types to Any.
# Prisma's generated types can explode Pyright's type inference budgets
# on large schemas. We collapse them to Any so the rest of the codebase
# can remain strongly typed while keeping runtime behavior unchanged.
#
# Safe types (Literal, TypeVar, simple references) are preserved from the
# original types.py to maintain proper type checking where possible.

from __future__ import annotations
from typing import Any
from typing_extensions import Literal

# Re-export commonly used typing constructs that may be imported from this module
from typing import TYPE_CHECKING, TypeVar, Generic, Union, Optional, List, Dict

# Base type alias for stubbed Prisma types - allows any dict structure
_PrismaDict = dict[str, Any]

"""

    lines = [header]

    # Include safe variable definitions (Literal types, TypeVars, etc.)
    lines.append("# Safe type definitions preserved from original types.py")
    for source in safe_variable_sources:
        lines.append(source)
    lines.append("")

    # Stub all classes and unsafe variables uniformly as dict[str, Any] aliases
    # This allows:
    # 1. Use in type annotations: x: SomeType
    # 2. Constructor calls: SomeType(...)
    # 3. Dict literal assignments: x: SomeType = {...}
    lines.append(
        "# Stubbed types (collapsed to dict[str, Any] to prevent type budget exhaustion)"
    )
    all_stubbed = sorted(classes | unsafe_variables)
    for name in all_stubbed:
        lines.append(f"{name} = _PrismaDict")

    lines.append("")

    # Stub functions
    for name in sorted(functions):
        lines.append(f"def {name}(*args: Any, **kwargs: Any) -> Any: ...")

    lines.append("")

    stub_path.write_text("\n".join(lines), encoding="utf-8")
    return (
        len(classes)
        + len(functions)
        + len(safe_variable_sources)
        + len(unsafe_variables)
    )


def main() -> None:
    """Main entry point."""
    try:
        types_path = find_prisma_types_path()
        stub_path = types_path.with_suffix(".pyi")

        print(f"Found prisma types.py at: {types_path}")
        print(f"Generating stub at: {stub_path}")

        num_symbols = generate_stub(types_path, stub_path)
        print(f"Generated {stub_path.name} with {num_symbols} Any-typed symbols")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
