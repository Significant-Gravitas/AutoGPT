"""
Architectural tests for the backend package.

Each rule here exists to prevent a *class* of bug, not to police style.
When adding a rule, document the incident or failure mode that motivated
it so future maintainers know whether the rule still earns its keep.
"""

import ast
import pathlib

BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]


# Event-loop-bound async clients (AsyncOpenAI, httpx.AsyncClient, ...) must not
# be shared by the whole process; .github/scripts/check_loop_bound_clients.py
# enforces that in the backend lint job.


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
        "copilot/baseline/service.py",
        "copilot/bot/adapters/base.py",
        "copilot/bot/bot_backend.py",
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
    """_DATACLASS_KNOWN_OFFENDERS must use forward slashes since the comparison
    key is built from pathlib.Path.relative_to() which uses OS-native
    separators. On Windows this would be backslashes, causing false positives.

    Ensure the key construction normalises to forward slashes.
    """
    for entry in _DATACLASS_KNOWN_OFFENDERS:
        path_part = entry.split()[0]
        assert "\\" not in path_part, (
            f"known-offenders entry uses backslash: {entry!r}. "
            "Use forward slashes — the test should normalise Path separators."
        )
