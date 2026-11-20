"""AgentGraph rows are created, or given another owner or ID, only by
backend/data/graph.py, which checks each ID's owner first."""

import ast
import re
from collections.abc import Iterable

from backend.util.architecture_test import BACKEND_ROOT, _iter_backend_py_files

_AGENT_GRAPH_MODEL_NAMES = frozenset({"AgentGraph", "agentgraph"})
_ROW_CREATING_METHODS = frozenset({"create", "create_many", "upsert"})
_ROW_UPDATING_METHODS = frozenset({"update", "update_many"})
_OWNERSHIP_FIELDS = frozenset({"id", "userId", "User"})
_AGENT_GRAPH_RELATIONS = frozenset({"AgentGraph", "AgentGraphs"})
_NESTED_CREATING_KEYS = frozenset(
    {"create", "create_many", "connect_or_create", "upsert"}
)

_RAW_AGENT_GRAPH_INSERT = re.compile(
    r'INSERT\s+INTO\s+\S*?"?AgentGraph\b', re.IGNORECASE
)
_RAW_AGENT_GRAPH_UPDATE = re.compile(
    r'UPDATE\s+\S*?"?AgentGraph\b"?(?:\s+\w+)?\s+SET\s+(.*?)(?:\bFROM\b|\bWHERE\b|$)',
    re.IGNORECASE | re.DOTALL,
)

# The one writer. Tests and fixtures seed rows directly on purpose.
_AGENT_GRAPH_WRITERS = frozenset({"data/graph.py"})


def test_agent_graph_rows_are_created_only_by_the_graph_module():
    offenders: list[str] = []
    for py in _iter_backend_py_files():
        rel = py.relative_to(BACKEND_ROOT).as_posix()
        if rel in _AGENT_GRAPH_WRITERS or py.name.endswith(("_test.py", "conftest.py")):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        offenders += [
            f"backend/{rel}:{lineno} {what}"
            for lineno, what in _agent_graph_row_writes(tree)
        ]

    assert not offenders, (
        "AgentGraph rows must be created by backend/data/graph.py only; route the "
        "write through create_graph, fork_graph or copy_graph.\n\n"
        "Offenders:\n  " + "\n  ".join(offenders)
    )


def _agent_graph_row_writes(tree: ast.AST) -> Iterable[tuple[int, str]]:
    """Every write that creates an AgentGraph row or sets its ID or owner,
    through Prisma, a nested create on another model, or raw SQL. An update
    whose data is not a literal counts."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Dict, ast.Call)) and _creates_agent_graph_through(
            node
        ):
            yield node.lineno, "nested create"
        if isinstance(node, (ast.Constant, ast.JoinedStr)):
            sql = _string_text(node)
            if sql and _is_raw_agent_graph_write(sql):
                yield node.lineno, "raw SQL"
            continue
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if not _receiver_names(node.func.value) & _AGENT_GRAPH_MODEL_NAMES:
            continue
        if method in _ROW_CREATING_METHODS or (
            method in _ROW_UPDATING_METHODS and _may_set_ownership(node)
        ):
            yield node.lineno, f".{method}()"


def _creates_agent_graph_through(node: ast.Dict | ast.Call) -> bool:
    """A dict or typed create input on another model that inserts AgentGraph rows."""
    return any(
        name in _AGENT_GRAPH_RELATIONS
        and {field for field, _ in _fields(value)} & _NESTED_CREATING_KEYS
        for name, value in _fields(node)
    )


def _fields(node: ast.expr) -> list[tuple[str | None, ast.expr]]:
    """A dict literal's or a typed input's ``field: value`` pairs."""
    if isinstance(node, ast.Dict):
        return [
            (key.value if isinstance(key, ast.Constant) else None, value)
            for key, value in zip(node.keys, node.values)
        ]
    if isinstance(node, ast.Call):
        return [(kw.arg, kw.value) for kw in node.keywords]
    return []


def _may_set_ownership(call: ast.Call) -> bool:
    data = next((k.value for k in call.keywords if k.arg == "data"), None)
    if not isinstance(data, ast.Dict):
        return True
    return any(
        not isinstance(key, ast.Constant) or key.value in _OWNERSHIP_FIELDS
        for key in data.keys
    )


def _is_raw_agent_graph_write(sql: str) -> bool:
    if _RAW_AGENT_GRAPH_INSERT.search(sql):
        return True
    return any(
        assignment.split("=", 1)[0].strip().split(".")[-1].strip('"')
        in _OWNERSHIP_FIELDS
        for match in _RAW_AGENT_GRAPH_UPDATE.finditer(sql)
        for assignment in match.group(1).split(",")
    )


def _string_text(node: ast.Constant | ast.JoinedStr) -> str | None:
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    return "".join(
        part.value if isinstance(part, ast.Constant) else "x" for part in node.values
    )


def _receiver_names(node: ast.expr) -> set[str]:
    """Every name in the chain a method is called on, e.g. ``a.B.c(x)``."""
    names: set[str] = set()
    while True:
        if isinstance(node, ast.Call):
            node = node.func
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
            node = node.value
        elif isinstance(node, ast.Name):
            names.add(node.id)
            return names
        else:
            return names
