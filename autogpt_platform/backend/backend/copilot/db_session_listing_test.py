"""Tests for copilot.db session LIST paths — dream-session exclusion.

The dream pass creates ChatSession rows with ``metadata.kind == "dream"``.
Those must be hidden from the user-facing LIST paths (chat sidebar,
pagination count, /search/global title search) while staying fetchable
by id. The whole risk is SQL NULL semantics: most sessions predate the
``kind`` metadata key, so ``metadata->>'kind'`` is NULL for them and a
naive ``<>`` comparison would silently hide every normal chat.

Mock-based tests pin the exact WHERE construction; the integration tests
at the bottom prove the NULL semantics against the real Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from prisma.models import ChatSession as PrismaChatSession
from prisma.types import ChatSessionCreateInput

from backend.copilot.db import (
    create_chat_session,
    get_chat_session_metadata,
    get_user_chat_sessions,
    get_user_session_count,
    update_chat_session_title,
)
from backend.copilot.model import ChatSessionMetadata
from backend.util.json import SafeJson

_NULL_SAFE_DREAM_FILTER = "(metadata->>'kind' IS DISTINCT FROM 'dream')"
_RAW_QUERY_TARGET = "backend.copilot.db.db.query_raw_with_schema"


def _make_prisma_session(session_id: str = "sess-1") -> PrismaChatSession:
    now = datetime.now(UTC)
    return PrismaChatSession(
        id=session_id,
        createdAt=now,
        updatedAt=now,
        userId="user-1",
        title="hello",
        credentials="{}",
        successfulAgentRuns="{}",
        successfulAgentSchedules="{}",
        metadata="{}",
        totalPromptTokens=0,
        totalCompletionTokens=0,
        chatStatus="idle",
        isShared=False,
        shareToken=None,
        sharedAt=None,
        autoShareExecutions=False,
    )


# ---------- WHERE construction (mocked raw query) ----------


@pytest.mark.asyncio
async def test_list_query_excludes_dreams_with_null_safe_operator():
    """The list WHERE clause uses IS DISTINCT FROM, never a bare inequality.

    ``NULL <> 'dream'`` evaluates to NULL for legacy rows without a
    ``kind`` key, which would hide every normal chat from the sidebar.
    """
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        result = await get_user_chat_sessions("u1")

    assert result == []
    query = raw.call_args.args[0]
    assert _NULL_SAFE_DREAM_FILTER in query
    assert "<>" not in query
    assert "!=" not in query
    assert "ILIKE" not in query
    assert raw.call_args.args[1:] == ("u1", 50, 0)
    assert raw.call_args.kwargs["model"] is PrismaChatSession


@pytest.mark.asyncio
async def test_list_query_maps_raw_rows_to_chat_session_info():
    raw = AsyncMock(return_value=[_make_prisma_session("sess-42")])
    with patch(_RAW_QUERY_TARGET, raw):
        result = await get_user_chat_sessions("user-1")

    assert len(result) == 1
    assert result[0].session_id == "sess-42"
    assert result[0].user_id == "user-1"
    assert result[0].metadata.kind == "normal"


@pytest.mark.asyncio
async def test_title_search_escapes_like_wildcards_and_keeps_dream_filter():
    """User-supplied search text must match literally (no %/_ wildcards)."""
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        await get_user_chat_sessions(
            "u1", limit=10, offset=5, title_contains="50%_done\\"
        )

    query = raw.call_args.args[0]
    assert _NULL_SAFE_DREAM_FILTER in query
    assert '"title" ILIKE $2' in query
    assert "LIMIT $3 OFFSET $4" in query
    assert raw.call_args.args[1:] == ("u1", "%50\\%\\_done\\\\%", 10, 5)


@pytest.mark.asyncio
async def test_count_query_uses_same_dream_exclusion_as_list():
    """Sidebar pagination count must stay consistent with the visible list."""
    raw = AsyncMock(return_value=[{"count": 7}])
    with patch(_RAW_QUERY_TARGET, raw):
        count = await get_user_session_count("u1")

    assert count == 7
    query = raw.call_args.args[0]
    assert _NULL_SAFE_DREAM_FILTER in query
    assert raw.call_args.args[1:] == ("u1",)


@pytest.mark.asyncio
async def test_count_returns_zero_when_query_yields_no_rows():
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        assert await get_user_session_count("u1") == 0


# ---------- NULL semantics against the real database ----------


async def _create_legacy_session_without_kind_key(user_id: str) -> str:
    """Insert a pre-dream-PR row whose metadata JSON has no ``kind`` key."""
    legacy_id = str(uuid4())
    await PrismaChatSession.prisma().create(
        data=ChatSessionCreateInput(
            id=legacy_id,
            userId=user_id,
            credentials=SafeJson({}),
            successfulAgentRuns=SafeJson({}),
            successfulAgentSchedules=SafeJson({}),
            metadata=SafeJson({"dry_run": False}),
        )
    )
    return legacy_id


@pytest.mark.asyncio(loop_scope="session")
async def test_dream_sessions_hidden_from_list_but_fetchable_by_id(
    setup_test_user, test_user_id
):
    """Legacy rows (no ``kind`` key) stay listed; dream rows are hidden
    from the list yet still resolvable by session_id (Memory Visualizer
    and dream flows read them directly)."""
    normal = await create_chat_session(str(uuid4()), test_user_id)
    legacy_id = await _create_legacy_session_without_kind_key(test_user_id)
    dream = await create_chat_session(
        str(uuid4()),
        test_user_id,
        metadata=ChatSessionMetadata(kind="dream", dream_pass_id="pass-1"),
    )

    listed_ids = {
        s.session_id for s in await get_user_chat_sessions(test_user_id, limit=10_000)
    }
    assert normal.session_id in listed_ids
    assert legacy_id in listed_ids
    assert dream.session_id not in listed_ids

    fetched = await get_chat_session_metadata(dream.session_id)
    assert fetched is not None
    assert fetched.metadata.kind == "dream"
    assert fetched.metadata.dream_pass_id == "pass-1"


@pytest.mark.asyncio(loop_scope="session")
async def test_session_count_stays_consistent_with_visible_list(
    setup_test_user, test_user_id
):
    """Creating a dream session must not bump the pagination total."""
    count_before = await get_user_session_count(test_user_id)

    await create_chat_session(str(uuid4()), test_user_id)
    await create_chat_session(
        str(uuid4()),
        test_user_id,
        metadata=ChatSessionMetadata(kind="dream", dream_pass_id="pass-2"),
    )

    count_after = await get_user_session_count(test_user_id)
    assert count_after == count_before + 1

    listed = await get_user_chat_sessions(test_user_id, limit=10_000)
    assert count_after == len(listed)


@pytest.mark.asyncio(loop_scope="session")
async def test_title_search_matches_literally_and_skips_dream_sessions(
    setup_test_user, test_user_id
):
    """ILIKE search is case-insensitive, treats ``_`` literally, and never
    surfaces dream sessions even on an exact title match."""
    marker = f"Ndl{uuid4().hex[:8]}"

    normal = await create_chat_session(str(uuid4()), test_user_id)
    assert await update_chat_session_title(
        normal.session_id, test_user_id, f"{marker}_alpha plan"
    )
    decoy = await create_chat_session(str(uuid4()), test_user_id)
    assert await update_chat_session_title(
        decoy.session_id, test_user_id, f"{marker}Xalpha plan"
    )
    dream = await create_chat_session(
        str(uuid4()),
        test_user_id,
        metadata=ChatSessionMetadata(kind="dream", dream_pass_id="pass-3"),
    )
    assert await update_chat_session_title(
        dream.session_id, test_user_id, f"{marker}_alpha dream"
    )

    results = await get_user_chat_sessions(
        test_user_id, title_contains=f"{marker}_ALPHA"
    )
    result_ids = {s.session_id for s in results}
    assert normal.session_id in result_ids
    assert decoy.session_id not in result_ids
    assert dream.session_id not in result_ids
