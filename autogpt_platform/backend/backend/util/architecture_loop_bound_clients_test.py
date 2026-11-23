import ast
import textwrap
from collections.abc import Callable, Iterator

import pytest

from backend.util.architecture_test import (
    BACKEND_ROOT,
    _annotation_names,
    _decorator_name,
    _is_test_file,
    _iter_backend_py_files,
)

# ---------------------------------------------------------------------------
# Rule: no async network client is shared by the whole process
# ---------------------------------------------------------------------------
#
# Motivation: an async client (AsyncOpenAI, AsyncAnthropic, httpx.AsyncClient,
# aiohttp.ClientSession, ...) keeps connections that belong to the event loop
# that opened them, and the executors run one event loop per worker thread. A
# client shared by those loops sends a call on another loop's connection:
# reading the reply fails after the request went out, the SDK retries, and the
# provider receives and bills the call twice while the platform records it
# once.
#
# A client is shared by the process when it is built at import (module or
# class level), stored in a `global` by a function, or returned from a
# process-wide cache (`@cached`, `@functools.cache`, `@lru_cache`). The scan
# can't see a client kept on a long-lived object (`self._session = ...` on a
# singleton) or in a dict keyed by `id(loop)`; review those by hand.

_LOOP_BOUND_CLIENTS = frozenset(
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
_PROCESS_CACHES = frozenset({"cached", "cache", "lru_cache"})


def test_no_async_client_is_shared_by_the_process():
    offenders = [
        f"  backend/{rel.as_posix()}:{line} {what}"
        for py in _iter_backend_py_files()
        if not _is_test_file(rel := py.relative_to(BACKEND_ROOT))
        for line, what in _process_wide_clients(
            ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        )
    ]
    assert not offenders, (
        "These async clients are shared by every event loop in the process, so "
        "a call that lands on another loop's connection reaches the provider "
        "twice. Decorate the factory with `@per_loop_cached` from "
        "`backend.util.per_loop_cache` and give the client "
        "`http_client=DefaultAsyncHttpxClient(event_hooks=same_loop_hooks(...))`, "
        "or build the client where it is used "
        "(`async with httpx.AsyncClient() as client:`):\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "source, found",
    [
        pytest.param(
            """
            from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI

            _main_client = None

            def _get_main_client():
                global _main_client
                if _main_client is None:
                    _main_client = LangfuseAsyncOpenAI(api_key="key")
                return _main_client
            """,
            ["_get_main_client() stores AsyncOpenAI(...) in the global _main_client"],
            id="a global set by its getter",
        ),
        pytest.param(
            """
            @cached(ttl_seconds=3600)
            def get_openai_client(*, prefer_openrouter=False) -> "AsyncOpenAI | None":
                from openai import AsyncOpenAI
                return AsyncOpenAI(api_key="key")
            """,
            ["@cached get_openai_client() keeps one AsyncOpenAI for the process"],
            id="a factory cached for the process",
        ),
        pytest.param(
            """
            @functools.cache
            def session():
                return aiohttp.ClientSession()

            @functools.lru_cache(maxsize=1)
            async def http():
                return httpx.AsyncClient()
            """,
            [
                "@cache session() keeps one ClientSession for the process",
                "@lru_cache http() keeps one AsyncClient for the process",
            ],
            id="functools caches",
        ),
        pytest.param(
            """
            CLIENT = AsyncAnthropic()

            class Api:
                http: httpx.AsyncClient = httpx.AsyncClient()

            try:
                SESSION = httpx.AsyncClient()
            except ImportError:
                SESSION = None
            """,
            [
                "CLIENT = AsyncAnthropic(...) is built once for the process",
                "http = AsyncClient(...) is built once for the process",
                "SESSION = AsyncClient(...) is built once for the process",
            ],
            id="clients built at import",
        ),
        pytest.param(
            """
            @per_loop_cached
            def get_client() -> AsyncOpenAI:
                return AsyncOpenAI(
                    http_client=DefaultAsyncHttpxClient(
                        event_hooks=same_loop_hooks("get_client()")
                    ),
                )

            async def fetch(url):
                async with httpx.AsyncClient() as client:
                    return await client.get(url)

            class Adapter:
                def __init__(self):
                    self._http = httpx.AsyncClient()
            """,
            [],
            id="per loop, per call and per object pass",
        ),
    ],
)
def test_the_scan_finds_each_shared_client(source: str, found: list[str]):
    tree = ast.parse(textwrap.dedent(source))
    assert [what for _, what in _process_wide_clients(tree)] == found


def _process_wide_clients(tree: ast.Module) -> list[tuple[int, str]]:
    aliases = {
        alias.asname: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.asname
    }

    def client_built_by(node: ast.expr | None) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        name = _decorator_name(node.func)
        name = aliases.get(name, name) if name else None
        return name if name in _LOOP_BOUND_CLIENTS else None

    found = [
        (statement.lineno, f"{target} = {client}(...) is built once for the process")
        for statement in _import_time_statements(tree)
        for target, value in _assignments(statement)
        if (client := client_built_by(value))
    ]
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found += _global_clients(function, client_built_by)
            found += _cached_clients(function, aliases, client_built_by)
    return found


def _import_time_statements(node: ast.AST) -> Iterator[ast.stmt]:
    """Statements run at import: module and class bodies, including their
    ``if``/``try``/``with`` blocks, not functions."""
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


def _global_clients(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    client_built_by: Callable[[ast.expr | None], str | None],
) -> list[tuple[int, str]]:
    declared = {
        name
        for node in ast.walk(function)
        if isinstance(node, ast.Global)
        for name in node.names
    }
    return [
        (node.lineno, f"{function.name}() stores {client}(...) in the global {target}")
        for node in ast.walk(function)
        for target, value in _assignments(node)
        if target in declared and (client := client_built_by(value))
    ]


def _cached_clients(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: dict[str, str],
    client_built_by: Callable[[ast.expr | None], str | None],
) -> list[tuple[int, str]]:
    cache = next(
        (
            name
            for decorator in function.decorator_list
            if (name := _decorator_name(decorator)) in _PROCESS_CACHES
        ),
        None,
    )
    if cache is None:
        return []
    returned = [
        *(aliases.get(name, name) for name in _annotation_names(function.returns)),
        *(
            client_built_by(node.value)
            for node in ast.walk(function)
            if isinstance(node, ast.Return)
        ),
    ]
    client = next((name for name in returned if name in _LOOP_BOUND_CLIENTS), None)
    if client is None:
        return []
    return [
        (
            function.lineno,
            f"@{cache} {function.name}() keeps one {client} for the process",
        )
    ]
