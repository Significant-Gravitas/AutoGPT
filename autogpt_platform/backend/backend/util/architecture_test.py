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
        # Inner helper extracted from ``get_openai_client`` so the local-
        # transport branch shares one cached client across both
        # ``prefer_openrouter`` arms (no semantic change vs the parent
        # offender — same loop-binding caveat applies, will be migrated
        # together when ``get_openai_client`` is moved to ``per_loop_cached``).
        "util/clients.py _get_local_openai_client",
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


# ---------------------------------------------------------------------------
# Rule: backend code uses pydantic, not stdlib dataclasses
# ---------------------------------------------------------------------------
#
# Motivation: pydantic ``BaseModel`` gives us validation, (de)serialization,
# and JSON-schema generation for free, and the whole backend relies on that
# consistency across the API, the executor, and blocks. A plain ``@dataclass``
# silently bypasses all of it. Shared, framework-agnostic helpers living under
# a ``lib``/``libs`` directory are exempt. Pydantic dataclasses
# (``pydantic.dataclasses``) are a different module and are intentionally
# allowed.

# Path segments whose subtrees are exempt: shared helpers under a `lib`/`libs`
# folder may use stdlib dataclasses; everything else must not.
_DATACLASS_EXEMPT_DIR_NAMES = frozenset({"lib", "libs"})

# Pre-existing offenders (present in `dev` as of merge-base 9822fe4b) tracked
# for future migration to pydantic. Excluded so the rule catches NEW violations
# without blocking unrelated PRs. Paths are posix-relative to the backend
# package root. Burn this list down; do not add to it.
_DATACLASS_KNOWN_OFFENDERS = frozenset(
    {
        "api/features/builder/db.py",
        # ``search/`` siblings to the original ``store/`` offenders — landed
        # in dev's global-search rollout (#13217) after this allowlist was
        # last snapshotted. Tracked here so the rule still catches genuinely
        # new violations; migrate alongside the ``store/`` originals.
        "api/features/search/content_handlers.py",
        "api/features/search/hybrid_search.py",
        "api/features/store/content_handlers.py",
        "api/features/store/hybrid_search.py",
        "blocks/codex.py",
        "blocks/mcp/client.py",
        "copilot/baseline/service.py",
        "copilot/bot/adapters/base.py",
        "copilot/bot/bot_backend.py",
        "copilot/bot/handler.py",
        "copilot/sdk/compaction.py",
        "copilot/sdk/file_ref.py",
        "copilot/sdk/service.py",
        "copilot/sdk/service_test.py",
        "copilot/sdk/session_waiter.py",
        "copilot/stream_registry.py",
        # Self-distilled skills registry added by dev's copilot-skills PR
        # series; predates this allowlist refresh, same migration plan.
        "copilot/tools/skills.py",
        "copilot/tools/helpers.py",
        "copilot/transcript.py",
        "util/cache.py",
        "util/prompt.py",
        "util/sandbox_files.py",
        "util/tool_call_loop.py",
    }
)


def _dataclass_imports(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return (lineno, description) for every stdlib ``dataclasses`` import."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return []
    offenses: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "dataclasses" or alias.name.startswith("dataclasses."):
                    offenses.append((node.lineno, "imports the `dataclasses` module"))
        elif isinstance(node, ast.ImportFrom):
            # `from dataclasses import ...`; `from pydantic.dataclasses import ...`
            # has a different module and is intentionally allowed.
            if node.module == "dataclasses":
                names = ", ".join(alias.name for alias in node.names)
                offenses.append((node.lineno, f"imports `{names}` from `dataclasses`"))
    return offenses


def test_backend_uses_pydantic_not_dataclasses():
    offenders: list[str] = []
    for py in _iter_backend_py_files():
        rel = py.relative_to(BACKEND_ROOT)
        if set(rel.parts) & _DATACLASS_EXEMPT_DIR_NAMES:
            continue
        if rel.as_posix() in _DATACLASS_KNOWN_OFFENDERS:
            continue
        for lineno, reason in _dataclass_imports(py):
            offenders.append(f"  backend/{rel}:{lineno} {reason}")

    assert not offenders, (
        "Backend code must use a pydantic `BaseModel` instead of stdlib "
        "`dataclasses` (only shared `lib`/`libs` helpers are exempt). "
        "Replace these with `pydantic.BaseModel`:\n" + "\n".join(sorted(offenders))
    )


def test_known_offenders_use_posix_separators():
    """_KNOWN_OFFENDERS must use forward slashes since the comparison key
    is built from pathlib.Path.relative_to() which uses OS-native separators.
    On Windows this would be backslashes, causing false positives.

    Ensure the key construction normalises to forward slashes.
    """
    for entry in _KNOWN_OFFENDERS | _DATACLASS_KNOWN_OFFENDERS:
        path_part = entry.split()[0]
        assert "\\" not in path_part, (
            f"known-offenders entry uses backslash: {entry!r}. "
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
