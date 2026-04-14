"""
Architectural tests for the backend package.

Each rule here exists to prevent a *class* of bug, not to police style.
When adding a rule, document the incident or failure mode that motivated
it so future maintainers know whether the rule still earns its keep.
"""

import ast
import pathlib

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Rule: no process-wide @cached(...) around event-loop-bound async clients
# ---------------------------------------------------------------------------
#
# Motivation: `backend.util.cache.cached` stores its result in a process-wide
# dict for ttl_seconds. Async clients (AsyncOpenAI, httpx.AsyncClient,
# AsyncRabbitMQ, supabase AClient, ...) wrap connection pools whose internal
# asyncio primitives lazily bind to the first event loop that uses them. The
# executor runs two long-lived loops on separate threads; once the cache is
# populated from loop A, any subsequent call from loop B raises
# `RuntimeError: ... bound to a different event loop`, surfaced as an opaque
# `APIConnectionError: Connection error.` and poisons the cache for a full
# TTL window.
#
# Use `per_loop_cached` (keyed on id(running loop)) or construct per-call.

LOOP_BOUND_TYPES = frozenset(
    {
        "AsyncOpenAI",
        "LangfuseAsyncOpenAI",
        "AsyncClient",  # httpx, openai internal
        "AsyncRabbitMQ",
        "AClient",  # supabase async
        "AsyncRedisExecutionEventBus",
    }
)

# Pre-existing offenders tracked for future cleanup. Exclude from this test
# so the rule can still catch NEW violations without blocking unrelated PRs.
_KNOWN_OFFENDERS = frozenset(
    {
        "util/clients.py get_async_supabase",
        "util/clients.py get_openai_client",
    }
)


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _annotation_names(annotation: ast.expr | None) -> set[str]:
    if annotation is None:
        return set()
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            parsed = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return set()
        return _annotation_names(parsed)
    names: set[str] = set()
    for child in ast.walk(annotation):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


def _iter_backend_py_files():
    for path in BACKEND_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        yield path


def test_known_offenders_use_posix_separators():
    """_KNOWN_OFFENDERS must use forward slashes since the comparison key
    is built from pathlib.Path.relative_to() which uses OS-native separators.
    On Windows this would be backslashes, causing false positives.

    Ensure the key construction normalises to forward slashes.
    """
    for entry in _KNOWN_OFFENDERS:
        path_part = entry.split()[0]
        assert "\\" not in path_part, (
            f"_KNOWN_OFFENDERS entry uses backslash: {entry!r}. "
            "Use forward slashes — the test should normalise Path separators."
        )


def test_no_process_cached_loop_bound_clients():
    offenders: list[str] = []
    for py in _iter_backend_py_files():
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorators = {_decorator_name(d) for d in node.decorator_list}
            if "cached" not in decorators:
                continue
            bound = _annotation_names(node.returns) & LOOP_BOUND_TYPES
            if bound:
                rel = py.relative_to(BACKEND_ROOT)
                key = f"{rel.as_posix()} {node.name}"
                if key in _KNOWN_OFFENDERS:
                    continue
                offenders.append(
                    f"{rel}:{node.lineno} {node.name}() -> {sorted(bound)}"
                )

    assert not offenders, (
        "Process-wide @cached(...) must not wrap functions returning event-"
        "loop-bound async clients. These objects lazily bind their connection "
        "pool to the first event loop that uses them; caching them across "
        "loops poisons the cache and surfaces as opaque connection errors.\n\n"
        "Offenders:\n  " + "\n  ".join(offenders) + "\n\n"
        "Fix: construct the client per-call, or introduce a per-loop factory "
        "keyed on id(asyncio.get_running_loop()). See "
        "backend/util/clients.py::get_openai_client for context."
    )
