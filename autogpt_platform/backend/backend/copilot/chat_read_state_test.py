"""Tests for the sidebar's per-chat markers kept on the server (SECRT-3542).

A chat wants its owner when a question or an approval waits on them
(``needs_you``), or when a turn ended since they last opened it (``error`` if
that turn failed, ``unread`` otherwise). The unit tests pin the rule; the
integration test runs a user's chats through it against the real Postgres:
turns end, a question and an approval arrive, and the owner marks chats read.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import Expert, PendingHumanReview
from prisma.types import ChatSessionCreateInput

from backend.copilot.db import (
    get_chats_awaiting_review,
    get_user_chat_session_groups,
    get_user_chat_sessions,
    mark_sessions_read,
    set_session_pending_question,
    update_chat_session_status,
)
from backend.copilot.model import ChatSessionInfo, ChatSessionMetadata, PendingQuestion
from backend.data.user import get_or_create_user_with_status
from backend.util.json import SafeJson

_NOW = datetime(2026, 11, 17, 12, 0, tzinfo=UTC)


def _info(
    *,
    ended_minutes_ago: int | None = None,
    failed: bool | None = None,
    read_minutes_ago: int | None = None,
    metadata: ChatSessionMetadata | None = None,
) -> ChatSessionInfo:
    def ago(minutes: int | None) -> datetime | None:
        return None if minutes is None else _NOW - timedelta(minutes=minutes)

    return ChatSessionInfo(
        session_id="s1",
        user_id="u1",
        usage=[],
        started_at=_NOW - timedelta(days=1),
        updated_at=_NOW,
        metadata=metadata or ChatSessionMetadata(),
        last_turn_ended_at=ago(ended_minutes_ago),
        last_turn_failed=failed,
        last_read_at=ago(read_minutes_ago),
    )


_QUESTION = PendingQuestion(text="Which list?", asked_at=_NOW)


# ---------- The rule ----------


def test_a_chat_whose_turns_ended_before_this_shipped_counts_as_read():
    assert _info().attention(awaiting_review=False) is None


def test_a_turn_that_ended_after_the_last_visit_is_unread():
    chat = _info(ended_minutes_ago=5, failed=False, read_minutes_ago=10)
    assert chat.attention(awaiting_review=False) == "unread"


def test_a_never_opened_chat_with_an_ended_turn_is_unread():
    chat = _info(ended_minutes_ago=5, failed=False)
    assert chat.attention(awaiting_review=False) == "unread"


def test_opening_the_chat_after_the_turn_ended_clears_it():
    chat = _info(ended_minutes_ago=10, failed=True, read_minutes_ago=5)
    assert chat.attention(awaiting_review=False) is None


def test_an_unread_failed_turn_is_an_error():
    chat = _info(ended_minutes_ago=5, failed=True, read_minutes_ago=10)
    assert chat.attention(awaiting_review=False) == "error"


def test_a_waiting_approval_outranks_an_unread_error_and_outlasts_a_visit():
    assert _info(ended_minutes_ago=5, failed=True).attention(True) == "needs_you"
    read = _info(ended_minutes_ago=10, read_minutes_ago=5)
    assert read.attention(awaiting_review=True) == "needs_you"


def test_an_open_question_needs_the_owner_even_once_read():
    chat = _info(
        ended_minutes_ago=10,
        read_minutes_ago=5,
        metadata=ChatSessionMetadata(pending_question=_QUESTION),
    )
    assert chat.attention(awaiting_review=False) == "needs_you"


def test_a_delegated_sub_chats_question_is_its_callers_to_report():
    delegated = ChatSessionMetadata(
        pending_question=_QUESTION, delegated_by_session_id="caller"
    )
    assert _info(metadata=delegated).attention(awaiting_review=False) is None

    handed_off = ChatSessionMetadata(
        pending_question=_QUESTION,
        delegated_by_session_id="caller",
        handed_off_from_expert_id="maria",
    )
    assert _info(metadata=handed_off).attention(awaiting_review=False) == ("needs_you")


# ---------- SQL construction (mocked) ----------


@pytest.mark.asyncio
async def test_marking_read_touches_only_unread_chats_the_list_shows():
    execute = AsyncMock(return_value=2)
    with patch("backend.copilot.db.db.execute_raw_with_schema", execute):
        marked = await mark_sessions_read("u1", "org-1", expert_id="maria")

    assert marked == 2
    query, *params = execute.await_args.args
    assert query.startswith('UPDATE {schema_prefix}"ChatSession" SET "lastReadAt"')
    # Raw SQL leaves updatedAt alone, so the sidebar's order doesn't move.
    assert '"updatedAt"' not in query
    assert '"archivedAt" IS NULL' in query
    assert '"expertId" = $3' in query
    assert '"lastReadAt" < "lastTurnEndedAt"' in query
    *scope, read_at = params
    assert scope == ["u1", "org-1", "maria"]
    # Never below the stored turn end, whatever this process's clock says.
    assert 'GREATEST($4, "lastTurnEndedAt")' in query
    assert isinstance(read_at, datetime)


@pytest.mark.asyncio
async def test_marking_one_chat_read_is_scoped_to_its_owner():
    execute = AsyncMock(return_value=1)
    with patch("backend.copilot.db.db.execute_raw_with_schema", execute):
        await mark_sessions_read("u1", session_id="chat-1")

    query, *params = execute.await_args.args
    assert '"userId" = $1' in query
    assert '"id" = $2' in query
    assert params[:-1] == ["u1", "chat-1"]


@pytest.mark.asyncio
async def test_group_counts_ride_on_the_totals_read():
    async def answer(query: str, *args, model=None, **kwargs):
        if model is not None:
            return []
        return [{"expertId": None, "total": 4, "attention": 3, "unread": 2}]

    raw = AsyncMock(side_effect=answer)
    with patch("backend.copilot.db.db.query_raw_with_schema", raw):
        await get_user_chat_session_groups("u1", 3, 30)

    totals_query = next(
        call.args[0] for call in raw.await_args_list if "model" not in call.kwargs
    )
    assert 'AS "attention"' in totals_query
    assert 'AS "unread"' in totals_query
    assert '"PendingHumanReview"' in totals_query


# ---------- Against the real database ----------


async def _chat(user_id: str, expert_id: str | None, minutes_ago: int) -> str:
    session_id = str(uuid4())
    at = datetime.now(UTC) - timedelta(minutes=minutes_ago)
    await PrismaChatSession.prisma().create(
        data=ChatSessionCreateInput(
            id=session_id,
            userId=user_id,
            expertId=expert_id,
            createdAt=at,
            updatedAt=at,
            credentials=SafeJson({}),
            successfulAgentRuns=SafeJson({}),
            successfulAgentSchedules=SafeJson({}),
            metadata=SafeJson({}),
        )
    )
    return session_id


async def _end_turn(session_id: str, *, failed: bool) -> None:
    await update_chat_session_status(
        session_id=session_id, status="idle", turn_failed=failed
    )


async def _markers(user_id: str) -> dict[str, str | None]:
    awaiting = await get_chats_awaiting_review(user_id)
    chats = await get_user_chat_sessions(user_id, 50, 0, pinned_first=False)
    return {
        chat.session_id: chat.attention(chat.session_id in awaiting) for chat in chats
    }


async def _order(user_id: str) -> list[str]:
    return [s.session_id for s in await get_user_chat_sessions(user_id)]


async def _counts(user_id: str) -> dict[str | None, tuple[int, int, int]]:
    groups = await get_user_chat_session_groups(user_id, 3, 30)
    return {
        group.expert_id: (group.total, group.attention_count, group.unread_count)
        for group in groups
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_markers_follow_turns_questions_approvals_and_visits(server):
    """A fresh user, so no other test's chats move the counts; everything it
    makes is removed again afterwards."""
    user_id = str(uuid4())
    await get_or_create_user_with_status(
        {"sub": user_id, "email": f"{user_id}@example.com"}
    )
    try:
        expert = await Expert.prisma().create(
            data={
                "ownerUserId": user_id,
                "name": "Maria",
                "role": "Marketing",
                "identity": "Grows the brand.",
            }
        )
        await _run_markers_scenario(user_id, expert.id)
    finally:
        await PendingHumanReview.prisma().delete_many(where={"userId": user_id})
        await PrismaChatSession.prisma().delete_many(where={"userId": user_id})
        await Expert.prisma().delete_many(where={"ownerUserId": user_id})


async def _run_markers_scenario(user_id: str, maria: str) -> None:
    summary = await _chat(user_id, None, minutes_ago=30)
    draft = await _chat(user_id, None, minutes_ago=40)
    campaign = await _chat(user_id, maria, minutes_ago=10)
    launch = await _chat(user_id, maria, minutes_ago=20)
    newsletter = await _chat(user_id, maria, minutes_ago=25)

    # Every chat here predates the markers: nothing is flagged.
    assert set((await _markers(user_id)).values()) == {None}

    await _end_turn(summary, failed=False)
    await _end_turn(launch, failed=True)
    await _end_turn(newsletter, failed=False)
    await set_session_pending_question(campaign, user_id, "Which list?", _NOW)
    await PendingHumanReview.prisma().create(
        data={
            "nodeExecId": f"review-{uuid4()}",
            "userId": user_id,
            "chatSessionId": draft,
            "payload": SafeJson({"action": "send"}),
        }
    )

    assert await _markers(user_id) == {
        summary: "unread",
        draft: "needs_you",
        campaign: "needs_you",
        launch: "error",
        newsletter: "unread",
    }
    # (total, attention, unread) per group.
    assert await _counts(user_id) == {None: (2, 2, 1), maria: (3, 3, 2)}
    # Ending a turn moves a chat up the sidebar; reading one must not.
    order_before_reads = await _order(user_id)

    # Marking Maria's chats read clears her unread and failed chats, not the
    # question still waiting on the owner, and leaves Otto's alone.
    assert await mark_sessions_read(user_id, expert_id=maria) == 2
    assert await _markers(user_id) == {
        summary: "unread",
        draft: "needs_you",
        campaign: "needs_you",
        launch: None,
        newsletter: None,
    }
    assert await _counts(user_id) == {None: (2, 2, 1), maria: (3, 1, 0)}

    # Opening one chat clears it on every device; a second visit changes
    # nothing.
    assert await mark_sessions_read(user_id, session_id=summary) == 1
    assert await mark_sessions_read(user_id, session_id=summary) == 0
    assert (await _markers(user_id))[summary] is None
    assert await _order(user_id) == order_before_reads

    # A new turn ending makes a read chat unread again.
    await _end_turn(newsletter, failed=False)
    assert (await _markers(user_id))[newsletter] == "unread"

    # Another user can't mark these chats read.
    assert await mark_sessions_read(str(uuid4()), session_id=newsletter) == 0
    assert (await _markers(user_id))[newsletter] == "unread"
