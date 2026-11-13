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
            "u1", 3, 30, organization_id="org-1", pinned=False
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
    assert 'ROW_NUMBER() OVER (ORDER BY "updatedAt" DESC, "id" DESC)' in (
        rows_call.args[0]
    )
    assert '"groupRank" <= $3 OR "overallRank" <= $4' in rows_call.args[0]
    assert rows_call.args[1:] == ("u1", "org-1", 3, 30)


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
        groups = await get_user_chat_session_groups("u1", 2, 30)

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
        groups = await get_user_chat_session_groups("u1", 3, 30)

    assert [group.expert_id for group in groups] == [None]


@pytest.mark.asyncio
async def test_the_newest_chats_past_each_minimum_fill_the_budget():
    """Each group keeps its one newest chat, and the one slot left goes to
    the newest of the rest: Otto's second chat, not Maria's older one."""
    rows = [
        _row("otto-1", None, minutes_ago=1),
        _row("otto-2", None, minutes_ago=2),
        _row("otto-3", None, minutes_ago=3),
        _row("maria-1", "maria", minutes_ago=10),
        _row("maria-2", "maria", minutes_ago=20),
    ]
    totals = [{"expertId": None, "total": 3}, {"expertId": "maria", "total": 2}]
    with patch(_RAW_QUERY_TARGET, _raw_reads(totals, rows)):
        groups = await get_user_chat_session_groups("u1", 1, 3)

    assert [
        (group.expert_id, [s.session_id for s in group.sessions]) for group in groups
    ] == [(None, ["otto-1", "otto-2"]), ("maria", ["maria-1"])]


@pytest.mark.asyncio
async def test_every_group_keeps_its_minimum_past_the_budget():
    rows = [
        _row("otto-1", None, minutes_ago=1),
        _row("otto-2", None, minutes_ago=2),
        _row("maria-1", "maria", minutes_ago=3),
        _row("maria-2", "maria", minutes_ago=4),
    ]
    totals = [{"expertId": None, "total": 2}, {"expertId": "maria", "total": 2}]
    with patch(_RAW_QUERY_TARGET, _raw_reads(totals, rows)):
        groups = await get_user_chat_session_groups("u1", 2, 1)

    assert [len(group.sessions) for group in groups] == [2, 2]


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
    """A fresh user, so other tests' sessions can't move the counts. The
    sessions and the expert it makes are removed again afterwards."""
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
    try:
        await _check_groups_against_paging(user_id, expert.id)
    finally:
        await PrismaChatSession.prisma().delete_many(where={"userId": user_id})
        await Expert.prisma().delete(where={"id": expert.id})


async def _check_groups_against_paging(user_id: str, expert_id: str) -> None:
    otto_ids = [
        await _session(user_id, None, minutes_ago=minutes)
        for minutes in (10, 20, 30, 40)
    ]
    await _session(user_id, None, minutes_ago=50, pinned=True)
    # The newest row of all, but a dream pass: it must neither count nor
    # make Otto's group the most recently active.
    await _session(user_id, None, minutes_ago=1, dream=True)
    maria_ids = [
        await _session(user_id, expert_id, minutes_ago=minutes) for minutes in (5, 15)
    ]

    # One each, then one more for the newest of the rest: Maria's second.
    groups = await get_user_chat_session_groups(user_id, 1, 3)
    assert [
        (group.expert_id, group.total, [s.session_id for s in group.sessions])
        for group in groups
    ] == [(expert_id, 2, maria_ids), (None, 5, otto_ids[:1])]

    unpinned = await get_user_chat_session_groups(user_id, 3, 30, pinned=False)
    assert [
        (group.expert_id, group.total, [s.session_id for s in group.sessions])
        for group in unpinned
    ] == [(expert_id, 2, maria_ids), (None, 4, otto_ids)]
    for group in unpinned:
        listed = len(group.sessions)
        if group.expert_id is None:
            page = await get_user_chat_sessions(
                user_id,
                listed,
                0,
                autopilot_only=True,
                pinned_first=False,
                pinned=False,
            )
            total = await get_user_session_count(
                user_id, autopilot_only=True, pinned=False
            )
        else:
            page = await get_user_chat_sessions(
                user_id,
                listed,
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
