"""Tests for copilot.db's grouped session read, behind the sidebar's Recent chats.

Every expert the user has chats with, Otto included, gets a group carrying its
real total and its newest few chats. The mocked tests pin the SQL the read
builds and how it assembles the groups; the integration test runs both
queries against the real Postgres and checks every group against paging that
expert's chats with the plain list.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from prisma.enums import ResourceVisibility
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import Expert
from prisma.types import ChatSessionCreateInput

from backend.copilot.db import (
    get_user_chat_session_groups,
    get_user_chat_sessions,
    get_user_session_count,
)
from backend.data.user import get_or_create_user_with_status
from backend.util.json import SafeJson

_RAW_QUERY_TARGET = "backend.copilot.db.db.query_raw_with_schema"
_NULL_SAFE_DREAM_FILTER = "(metadata->>'kind' IS DISTINCT FROM 'dream')"
_ORG_SCOPE = (
    '("organizationId" = $2 OR "organizationId" IS NULL OR "expertId" IS NOT NULL)'
)
_NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


def _row(session_id: str, expert_id: str | None, minutes_ago: int):
    at = _NOW - timedelta(minutes=minutes_ago)
    return PrismaChatSession(
        id=session_id,
        createdAt=at,
        updatedAt=at,
        userId="u1",
        title=session_id,
        credentials="{}",
        successfulAgentRuns="{}",
        successfulAgentSchedules="{}",
        metadata="{}",
        totalPromptTokens=0,
        totalCompletionTokens=0,
        chatStatus="idle",
        isPinned=False,
        isShared=False,
        shareToken=None,
        sharedAt=None,
        autoShareExecutions=False,
        visibility=ResourceVisibility.PRIVATE,
        expertId=expert_id,
    )


def _raw_reads(totals: list[dict], rows: list[PrismaChatSession]) -> AsyncMock:
    """Answers the totals read and the rows read, told apart by the model the
    rows read hydrates into."""

    async def answer(query: str, *args, model=None, **kwargs):
        return rows if model is PrismaChatSession else totals

    return AsyncMock(side_effect=answer)


# ---------- SQL construction and assembly (mocked raw query) ----------


@pytest.mark.asyncio
async def test_both_reads_apply_the_same_visibility_rules_as_the_list():
    raw = _raw_reads([], [])
    with patch(_RAW_QUERY_TARGET, raw):
        await get_user_chat_session_groups(
            "u1", 3, organization_id="org-1", pinned=False
        )

    totals_call, rows_call = sorted(
        raw.await_args_list, key=lambda call: "model" in call.kwargs
    )
    for call in (totals_call, rows_call):
        query = call.args[0]
        assert _NULL_SAFE_DREAM_FILTER in query
        assert _ORG_SCOPE in query
        assert 'AND NOT "isPinned"' in query
    assert 'GROUP BY "expertId"' in totals_call.args[0]
    assert totals_call.args[1:] == ("u1", "org-1")
    assert (
        'PARTITION BY "expertId" ORDER BY "updatedAt" DESC, "id" DESC'
        in rows_call.args[0]
    )
    assert '"rank" <= $3' in rows_call.args[0]
    assert rows_call.args[1:] == ("u1", "org-1", 3)


@pytest.mark.asyncio
async def test_groups_keep_real_totals_with_newest_chats_and_most_active_first():
    rows = [
        _row("otto-old", None, minutes_ago=50),
        _row("maria-new", "maria", minutes_ago=1),
        _row("otto-new", None, minutes_ago=5),
        _row("maria-old", "maria", minutes_ago=30),
    ]
    totals = [{"expertId": None, "total": 29}, {"expertId": "maria", "total": 17}]
    with patch(_RAW_QUERY_TARGET, _raw_reads(totals, rows)):
        groups = await get_user_chat_session_groups("u1", 2)

    assert [(group.expert_id, group.total) for group in groups] == [
        ("maria", 17),
        (None, 29),
    ]
    assert [s.session_id for s in groups[0].sessions] == ["maria-new", "maria-old"]
    assert [s.session_id for s in groups[1].sessions] == ["otto-new", "otto-old"]


@pytest.mark.asyncio
async def test_a_group_emptied_between_the_two_reads_is_dropped():
    totals = [{"expertId": None, "total": 1}, {"expertId": "clip", "total": 1}]
    rows = [_row("otto-1", None, minutes_ago=1)]
    with patch(_RAW_QUERY_TARGET, _raw_reads(totals, rows)):
        groups = await get_user_chat_session_groups("u1", 3)

    assert [group.expert_id for group in groups] == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pinned, condition", [(True, 'AND "isPinned"'), (False, 'AND NOT "isPinned"')]
)
async def test_list_and_count_can_keep_only_pinned_or_only_unpinned(pinned, condition):
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        await get_user_chat_sessions("u1", pinned=pinned)
        await get_user_session_count("u1", pinned=pinned)

    assert [condition in call.args[0] for call in raw.await_args_list] == [
        True,
        True,
    ]


@pytest.mark.asyncio
async def test_list_and_count_keep_pinned_and_unpinned_by_default():
    raw = AsyncMock(return_value=[])
    with patch(_RAW_QUERY_TARGET, raw):
        await get_user_chat_sessions("u1")
        await get_user_session_count("u1")

    for call in raw.await_args_list:
        assert 'AND "isPinned"' not in call.args[0]
        assert 'NOT "isPinned"' not in call.args[0]


# ---------- Against the real database ----------


async def _session(
    user_id: str,
    expert_id: str | None,
    *,
    minutes_ago: int,
    pinned: bool = False,
    dream: bool = False,
) -> str:
    session_id = str(uuid4())
    at = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    await PrismaChatSession.prisma().create(
        data=ChatSessionCreateInput(
            id=session_id,
            userId=user_id,
            expertId=expert_id,
            isPinned=pinned,
            createdAt=at,
            updatedAt=at,
            credentials=SafeJson({}),
            successfulAgentRuns=SafeJson({}),
            successfulAgentSchedules=SafeJson({}),
            metadata=SafeJson({"kind": "dream"} if dream else {}),
        )
    )
    return session_id


@pytest.mark.asyncio(loop_scope="session")
async def test_each_group_matches_paging_through_that_experts_chats(server):
    """A fresh user, so other tests' sessions can't move the counts."""
    user_id = str(uuid4())
    await get_or_create_user_with_status(
        {"sub": user_id, "email": f"{user_id}@example.com"}
    )
    expert = await Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": "Maria",
            "role": "Marketing",
            "identity": "Grows the brand.",
        }
    )
    otto_ids = [
        await _session(user_id, None, minutes_ago=minutes)
        for minutes in (10, 20, 30, 40)
    ]
    await _session(user_id, None, minutes_ago=50, pinned=True)
    # The newest row of all, but a dream pass: it must neither count nor
    # make Otto's group the most recently active.
    await _session(user_id, None, minutes_ago=1, dream=True)
    maria_ids = [
        await _session(user_id, expert.id, minutes_ago=minutes) for minutes in (5, 15)
    ]

    groups = await get_user_chat_session_groups(user_id, 3)
    assert [(group.expert_id, group.total) for group in groups] == [
        (expert.id, 2),
        (None, 5),
    ]
    assert [s.session_id for s in groups[0].sessions] == maria_ids
    assert [s.session_id for s in groups[1].sessions] == otto_ids[:3]

    unpinned = await get_user_chat_session_groups(user_id, 3, pinned=False)
    assert [(group.expert_id, group.total) for group in unpinned] == [
        (expert.id, 2),
        (None, 4),
    ]
    for group in unpinned:
        if group.expert_id is None:
            page = await get_user_chat_sessions(
                user_id, 3, 0, autopilot_only=True, pinned_first=False, pinned=False
            )
            total = await get_user_session_count(
                user_id, autopilot_only=True, pinned=False
            )
        else:
            page = await get_user_chat_sessions(
                user_id,
                3,
                0,
                expert_id=group.expert_id,
                pinned_first=False,
                pinned=False,
            )
            total = await get_user_session_count(
                user_id, expert_id=group.expert_id, pinned=False
            )
        assert [s.session_id for s in page] == [s.session_id for s in group.sessions]
        assert total == group.total
