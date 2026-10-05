"""Find imports and calls that can query Prisma outside the database layer.

The call graph is deliberately conservative: a helper that can reach Prisma is
database access even if today's caller happens to own a connection. Only the
connection-aware accessors and DatabaseManager clients stop that propagation.
"""

import json
from pathlib import Path

from backend.util.db_boundary_ast import collect_references, resolve_alias
from backend.util.db_boundary_policy import (
    RAW_DATABASE_ACCESS,
    is_database_implementation,
    is_gateway,
)


def main() -> int:
    failures = check_database_boundary(Path(__file__).resolve().parents[1])
    if failures:
        print(
            "Database access must use backend.data.db_accessors or DatabaseManager RPC."
        )
        print(
            "Move queries into database implementations and shrink legacy exceptions when fixing them."
        )
        print("\n".join(failures))
        return 1
    print("Database access boundary passed.")
    return 0


def check_database_boundary(root: Path) -> list[str]:
    violations = find_violations(read_sources(root))
    baseline_path = root / "util" / "database_boundary_legacy.json"
    baseline: dict[str, int] = json.loads(baseline_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for key in sorted(violations.keys() | baseline.keys()):
        lines = violations.get(key, [])
        allowed = baseline.get(key, 0)
        if len(lines) > allowed:
            path, detail = key.split("::", 1)
            failures.append(
                f"backend/{path}:{lines[0]} {detail} ({len(lines)} references; {allowed} legacy)"
            )
        elif len(lines) < allowed:
            failures.append(
                f"Remove stale legacy database exception: {key} ({allowed} -> {len(lines)})"
            )
    return failures


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
    references, aliases, callables, rpc_clients = collect_references(sources)
    resolved = {
        scope: [
            (resolved_target, line)
            for target, line in targets
            for resolved_target in resolve_alias(target, aliases)
        ]
        for scope, targets in references.items()
    }
    unsafe = _query_callables(resolved, callables, rpc_clients)
    return _module_violations(resolved, sources, unsafe, rpc_clients)


def _query_callables(
    references: dict[str, list[tuple[str, int]]],
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
                _unsafe_target(target, unsafe, rpc_clients)
                for target, _ in references[scope]
            )
        }
        if not discovered:
            break
        unsafe.update(discovered)
    return unsafe


def _module_violations(
    references: dict[str, list[tuple[str, int]]],
    sources: dict[str, str],
    unsafe: set[str],
    rpc_clients: set[str],
) -> dict[str, list[int]]:
    violations: dict[str, list[int]] = {}
    modules = sorted(sources, key=len, reverse=True)
    for scope, targets in references.items():
        module = next(
            name for name in modules if scope == name or scope.startswith(name + ".")
        )
        if is_database_implementation(module):
            continue
        path = module.removeprefix("backend.").replace(".", "/") + ".py"
        owner = scope.removeprefix(module + ".") if scope != module else "<module>"
        for target, line in targets:
            if scope in rpc_clients and not _unsafe_target(
                target, set(RAW_DATABASE_ACCESS), set()
            ):
                continue
            query = _unsafe_target(target, unsafe, rpc_clients)
            if query:
                key = f"{path}::{owner} -> {query}"
                violations.setdefault(key, []).append(line)
    return violations


def _unsafe_target(target: str, unsafe: set[str], rpc_clients: set[str]) -> str | None:
    if (
        is_gateway(target)
        or target.rsplit(".", 1)[-1] in {"cache_delete", "cache_clear", "is_connected"}
        or any(
            target == client or target.startswith(client + ".")
            for client in rpc_clients
        )
    ):
        return None
    if target in unsafe:
        return target
    parts = target.split(".")
    for length in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:length])
        if prefix in unsafe:
            return prefix
    return None


if __name__ == "__main__":
    raise SystemExit(main())
