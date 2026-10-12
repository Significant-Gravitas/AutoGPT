#!/usr/bin/env python3
"""Fail when an async network client is shared by the whole backend process.

An async client (AsyncOpenAI, AsyncAnthropic, httpx.AsyncClient,
aiohttp.ClientSession, ...) keeps connections that belong to the event loop
that opened them, and the backend runs one event loop per worker thread. A
client shared across those loops sends a call on another loop's connection:
reading the reply fails after the request went out, the SDK retries, and the
provider receives and bills the call twice while the platform records it once.

A client becomes process-wide when it is:
- assigned at module or class level: ``_client = AsyncOpenAI(...)``;
- assigned to a ``global`` inside a function;
- returned from a function cached for the process (``@cached``,
  ``@functools.cache``, ``@functools.lru_cache``).

It can't see a client kept at runtime on a long-lived object
(``self._session = aiohttp.ClientSession()`` on a singleton) or in a dict keyed
by ``id(loop)``; review those by hand.

Usage: check_loop_bound_clients.py [PATH ...]  (default: the backend packages)
"""

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS = [
    REPO_ROOT / "autogpt_platform" / "backend" / "backend",
    REPO_ROOT / "autogpt_platform" / "autogpt_libs" / "autogpt_libs",
]

# Constructors of clients whose connections belong to one event loop.
LOOP_BOUND_CLIENTS = frozenset(
    {
        "AsyncAnthropic",
        "AsyncAnthropicBedrock",
        "AsyncAnthropicVertex",
        "AsyncAzureOpenAI",
        "AsyncClient",  # httpx, ollama
        "AsyncGroq",
        "AsyncOpenAI",
        "AsyncRabbitMQ",
        "AsyncRedisExecutionEventBus",
        "ClientSession",  # aiohttp
        "DefaultAsyncHttpxClient",
    }
)
PROCESS_CACHES = frozenset({"cached", "cache", "lru_cache"})

# "<path relative to the repo>::<name>" -> why one event loop uses it for the
# life of the process. Keep this empty unless that is really true.
ALLOWED: dict[str, str] = {}

HOW_TO_FIX = """\
Each of these clients is shared by every event loop in the process. Give each
loop its own instead:
  - decorate the factory with @per_loop_cached from backend.util.per_loop_cache
    (one client per running loop, closed when that loop shuts down), and pass
    http_client=DefaultAsyncHttpxClient(event_hooks=same_loop_hooks(...)) so a
    client that still reaches another loop fails before it sends; or
  - build the client where it is used, e.g.
    `async with httpx.AsyncClient() as client:`.
If one event loop really does use the client for the life of the process, add
its "<path>::<name>" key to ALLOWED in .github/scripts/check_loop_bound_clients.py
with the reason."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, default=DEFAULT_PATHS)
    args = parser.parse_args(argv)
    offenders = [
        offender
        for path in python_files(args.paths)
        for offender in find_offenders(path)
        if offender.key not in ALLOWED
    ]
    if not offenders:
        return 0
    for offender in offenders:
        print(f"{offender.location}: {offender.what}", file=sys.stderr)
    print(f"\n{HOW_TO_FIX}", file=sys.stderr)
    return 1


class Offender:
    def __init__(self, path: Path, line: int, name: str, what: str) -> None:
        relative = _relative(path)
        self.key = f"{relative}::{name}"
        self.location = f"{relative}:{line}"
        self.what = what


def python_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        files.extend(
            file
            for file in sorted(path.rglob("*.py"))
            if "__pycache__" not in file.parts and not _is_test(file)
        )
    return files


def find_offenders(path: Path) -> list[Offender]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    aliases = _import_aliases(tree)

    def client_built_by(node: ast.expr | None) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        name = _resolve(_called_name(node.func), aliases)
        return name if name in LOOP_BOUND_CLIENTS else None

    offenders: list[Offender] = []
    for statement in _import_time_statements(tree):
        for target, value in _assignments(statement):
            client = client_built_by(value)
            if client:
                offenders.append(
                    Offender(
                        path,
                        statement.lineno,
                        target,
                        f"{target} = {client}(...) is built once for the process",
                    )
                )
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        offenders.extend(_global_clients(path, function, client_built_by))
        cache = _process_cache(function)
        if cache is None:
            continue
        client = next(
            (
                built
                for built in (
                    *(
                        _resolve(name, aliases)
                        for name in _annotation_names(function.returns)
                    ),
                    *(
                        client_built_by(node.value)
                        for node in ast.walk(function)
                        if isinstance(node, ast.Return)
                    ),
                )
                if built in LOOP_BOUND_CLIENTS
            ),
            None,
        )
        if client:
            offenders.append(
                Offender(
                    path,
                    function.lineno,
                    function.name,
                    f"@{cache} {function.name}() keeps one {client} for the process",
                )
            )
    return offenders


def _global_clients(path, function, client_built_by) -> list[Offender]:
    declared = {
        name
        for node in ast.walk(function)
        if isinstance(node, ast.Global)
        for name in node.names
    }
    if not declared:
        return []
    return [
        Offender(
            path,
            node.lineno,
            target,
            f"{function.name}() stores {client}(...) in the global {target}",
        )
        for node in ast.walk(function)
        for target, value in _assignments(node)
        if target in declared and (client := client_built_by(value))
    ]


def _import_time_statements(node: ast.AST):
    """Statements that run when the module is imported: module and class
    bodies, including their ``if``/``try``/``with`` blocks, not functions."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, ast.stmt):
            yield child
        yield from _import_time_statements(child)


def _assignments(node: ast.AST) -> list[tuple[str, ast.expr | None]]:
    if isinstance(node, ast.Assign):
        return [(t.id, node.value) for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [(node.target.id, node.value)]
    return []


def _process_cache(function: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    for decorator in function.decorator_list:
        name = _called_name(
            decorator.func if isinstance(decorator, ast.Call) else decorator
        )
        if name in PROCESS_CACHES:
            return name
    return None


def _called_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    return {
        alias.asname: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname
    }


def _resolve(name: str | None, aliases: dict[str, str]) -> str | None:
    return aliases.get(name, name) if name else None


def _annotation_names(annotation: ast.expr | None) -> set[str]:
    if annotation is None:
        return set()
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return set()
    return {
        name
        for node in ast.walk(annotation)
        if (name := _called_name(node) if isinstance(node, ast.expr) else None)
    }


def _is_test(path: Path) -> bool:
    return (
        path.name == "conftest.py"
        or path.name.startswith("test_")
        or path.stem.endswith("_test")
        or "tests" in path.parts
    )


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


if __name__ == "__main__":
    sys.exit(main())
