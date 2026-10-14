"""Find imports and calls that can query Prisma outside the database layer.

The call graph is deliberately conservative: a helper that can reach Prisma is
database access even if today's caller happens to own a connection. Only the
connection-aware accessors and DatabaseManager clients stop that propagation.
"""

import json
from collections import Counter
from functools import cache, partial
from pathlib import Path

from backend.util.db_boundary_ast import collect_references, resolve_alias
from backend.util.db_boundary_policy import (
    CONNECTION_OWNERS,
    RAW_DATABASE_ACCESS,
    is_connection_owner_dispatch,
    is_database_implementation,
    is_gateway,
)

GUIDANCE = (
    "Database access must use backend.data.db_accessors or DatabaseManager RPC.\n"
    "Move queries into database implementations and shrink legacy exceptions when fixing them.\n"
    "See 'Database access boundary' in autogpt_platform/backend/AGENTS.md."
)


def main() -> int:
    failures = check_database_boundary(Path(__file__).resolve().parents[1])
    if failures:
        print(GUIDANCE)
        print("\n".join(failures))
        return 1
    print("Database access boundary passed.")
    return 0


def check_database_boundary(root: Path) -> list[str]:
    references = find_database_references(read_sources(root))
    baseline_path = root / "util" / "database_boundary_legacy.json"
    baseline: dict[str, list[str]] = json.loads(
        baseline_path.read_text(encoding="utf-8")
    )
    return [
        failure
        for key in sorted(references.keys() | baseline.keys())
        for failure in _compare_legacy_references(
            key, references.get(key, []), baseline.get(key, [])
        )
    ]


def _compare_legacy_references(
    key: str, references: list[tuple[int, str]], allowed: list[str]
) -> list[str]:
    path, detail = key.split("::", 1)
    locations = ",".join(map(str, sorted({line for line, _ in references})))
    if len(references) > len(allowed):
        return [
            f"backend/{path}:{locations} {detail} "
            f"({len(references)} references; {len(allowed)} legacy)"
        ]
    if len(references) < len(allowed):
        return [
            f"Remove stale legacy database exception: {key} "
            f"({len(allowed)} -> {len(references)})"
        ]
    if Counter(identity for _, identity in references) != Counter(allowed):
        return [
            f"Changed legacy database reference: backend/{path}:{locations} {detail}. "
            "Route replacement queries through the database gateway."
        ]
    return []


def read_sources(root: Path) -> dict[str, str]:
    return {
        "backend."
        + ".".join(path.relative_to(root).with_suffix("").parts): path.read_text(
            encoding="utf-8"
        )
        for path in root.rglob("*.py")
        if not (
            path.name.endswith("_test.py")
            or path.name.startswith("test_")
            or path.name in {"conftest.py", "_test_data.py"}
            or {"test", "tests", "__pycache__"}.intersection(
                path.relative_to(root).parts
            )
        )
    }


def find_violations(sources: dict[str, str]) -> dict[str, list[int]]:
    return {
        key: [line for line, _ in references]
        for key, references in find_database_references(sources).items()
    }


def find_database_references(
    sources: dict[str, str],
) -> dict[str, list[tuple[int, str]]]:
    references, aliases, callables, rpc_clients = collect_references(sources)
    resolve = cache(partial(resolve_alias, aliases=aliases))
    resolved = {
        scope: [
            (resolved_target, line, identity)
            for target, line, identity in targets
            for resolved_target in resolve(target)
        ]
        for scope, targets in references.items()
    }
    unsafe = _query_callables(resolved, callables, rpc_clients)
    return _module_violations(resolved, sources, unsafe, callables, rpc_clients)


def _query_callables(
    references: dict[str, list[tuple[str, int, str]]],
    callables: set[str],
    rpc_clients: set[str],
) -> set[str]:
    unsafe = set(RAW_DATABASE_ACCESS)
    while True:
        discovered = {
            scope
            for scope in callables - unsafe
            if not is_gateway(scope)
            and scope not in rpc_clients
            and any(
                _unsafe_target(target, unsafe, callables, rpc_clients)
                for target, _, _ in references[scope]
            )
        }
        if not discovered:
            break
        unsafe.update(discovered)
    return unsafe


def _module_violations(
    references: dict[str, list[tuple[str, int, str]]],
    sources: dict[str, str],
    unsafe: set[str],
    callables: set[str],
    rpc_clients: set[str],
) -> dict[str, list[tuple[int, str]]]:
    violations: dict[str, list[tuple[int, str]]] = {}
    for scope, targets in references.items():
        parts = scope.split(".")
        module = next(
            name
            for name in (".".join(parts[:end]) for end in range(len(parts), 0, -1))
            if name in sources
        )
        if is_database_implementation(module) or module in CONNECTION_OWNERS:
            continue
        path = module.removeprefix("backend.").replace(".", "/") + ".py"
        owner = scope.removeprefix(module + ".") if scope != module else "<module>"
        for target, line, identity in targets:
            if is_connection_owner_dispatch(module, target):
                continue
            if scope in rpc_clients and not _unsafe_target(
                target, set(RAW_DATABASE_ACCESS), callables, set()
            ):
                continue
            query = _unsafe_target(target, unsafe, callables, rpc_clients)
            if query:
                key = f"{path}::{owner} -> {query}"
                violations.setdefault(key, []).append((line, identity))
    return violations


def _unsafe_target(
    target: str, unsafe: set[str], callables: set[str], rpc_clients: set[str]
) -> str | None:
    if (
        is_gateway(target)
        or target
        in {
            "backend.data.db.is_connected",
            "prisma.Prisma.is_connected",
            "prisma.Client.is_connected",
            "prisma.client.Prisma.is_connected",
            "prisma.client.Client.is_connected",
            ".prisma.is_connected",
        }
        or any(
            target == client or target.startswith(client + ".")
            for client in rpc_clients
        )
    ):
        return None
    if target in unsafe:
        return target
    if target in callables:
        return None
    if target.rsplit(".", 1)[-1] in {"cache_delete", "cache_clear"}:
        return None
    parts = target.split(".")
    for length in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:length])
        if prefix in unsafe:
            return prefix
    return None


if __name__ == "__main__":
    raise SystemExit(main())
