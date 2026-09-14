"""Every learning data function reached through the accessor seams must be
exposed on the DatabaseManager RPC client, or the Prisma-less copilot
executor and scheduler would fail at runtime while unit tests (which patch
the seams) stay green."""

from __future__ import annotations

import re
from pathlib import Path

from backend.data import (
    skill_learning,
    skill_publication,
    skill_reviews,
    skill_use,
    skill_versions,
)
from backend.data.db_manager import DatabaseManagerAsyncClient

_CALL_RE = re.compile(
    r"(skill_learning_db|skill_reviews_db|skill_versions_db|skill_publication_db"
    r"|skill_use_db)\(\)\s*\.\s*(\w+)"
)
_ROOTS = (
    Path(__file__).parent,
    Path(__file__).parent.parent / "tools" / "skills.py",
    Path(__file__).parent.parent.parent / "api" / "features" / "skill_learning",
)


def _accessor_calls() -> set[str]:
    names: set[str] = set()
    for root in _ROOTS:
        files = [root] if root.is_file() else root.glob("*.py")
        for path in files:
            if path.name.endswith("_test.py") or path.name.startswith("_fake"):
                continue
            names.update(m.group(2) for m in _CALL_RE.finditer(path.read_text()))
    return names


def test_every_accessor_call_is_exposed_on_the_rpc_client():
    names = _accessor_calls()
    assert names, "expected at least one accessor call"
    for name in sorted(names):
        assert any(
            hasattr(module, name)
            for module in (
                skill_learning,
                skill_reviews,
                skill_versions,
                skill_publication,
                skill_use,
            )
        ), name
        # ``db_accessors`` hands Prisma-less processes (copilot executor,
        # scheduler) the async client; the sync client is not on these paths.
        assert hasattr(DatabaseManagerAsyncClient, name), f"async client lacks {name}"
