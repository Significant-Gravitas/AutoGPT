"""Stopping a turn stops the threads it handed work to."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import delegation_cancel
from backend.copilot.model import ChatMessage, ChatSessionMetadata

TURN_START = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def _child(session_id: str, parent: str | None, *, handed_off: bool = False):
    child = MagicMock(session_id=session_id)
    child.metadata = ChatSessionMetadata(
        delegated_by_session_id=parent,
        handed_off_from_expert_id="expert-a" if handed_off else None,
    )
    return child


def _sent(sender: str | None, *, minutes_after_start: int) -> ChatMessage:
    return ChatMessage(
        role="user",
        content="Draft the PRD",
        metadata={"from_session_id": sender} if sender else None,
        created_at=TURN_START + timedelta(minutes=minutes_after_start),
    )


@pytest.fixture
def seams(monkeypatch):
    running = [
        _child("sub-running", "parent"),
        _child("other-chat", None),
        _child("someone-elses-sub", "another-parent"),
        _child("handed-off", "parent", handed_off=True),
    ]
    queued = [_child("sub-queued", "parent")]
    db = MagicMock()

    async def by_status(*, user_id: str, status: str):
        return {"running": running, "queued": queued}[status]

    # The message that started each child's current turn.
    sent = {
        "sub-running": _sent("parent", minutes_after_start=1),
        "sub-queued": _sent("parent", minutes_after_start=2),
        "handed-off": _sent("parent", minutes_after_start=1),
        "earlier-turn": _sent("parent", minutes_after_start=-30),
        "user-typing-in-thread": _sent(None, minutes_after_start=3),
    }
    running += [
        _child("earlier-turn", "parent"),
        _child("user-typing-in-thread", "parent"),
    ]

    async def latest_user_message(session_id: str):
        return sent.get(session_id)

    db.list_chat_sessions_by_status = AsyncMock(side_effect=by_status)
    db.get_latest_user_message_in_session = AsyncMock(side_effect=latest_user_message)
    monkeypatch.setattr(delegation_cancel, "chat_db", lambda: db)
    dequeue = AsyncMock(
        side_effect=lambda *, user_id, session_id: session_id == "sub-queued"
    )
    cancel = AsyncMock()
    monkeypatch.setattr(delegation_cancel, "cancel_queued_turn", dequeue)
    monkeypatch.setattr(delegation_cancel, "enqueue_cancel_task", cancel)
    return dequeue, cancel


@pytest.mark.asyncio
async def test_stopping_a_turn_stops_its_working_and_queued_children(seams):
    dequeue, cancel = seams

    stopped = await delegation_cancel.cancel_delegated_children(
        "parent", "alice", TURN_START
    )

    assert sorted(stopped) == ["sub-queued", "sub-running"]
    cancel.assert_awaited_once_with("sub-running")
    assert {c.kwargs["session_id"] for c in dequeue.await_args_list} == {
        "sub-queued",
        "sub-running",
    }


@pytest.mark.asyncio
async def test_a_handed_off_thread_is_not_the_parents_to_stop(seams):
    """A hand-off transferred the task for good; nobody is waiting on it."""
    _, cancel = seams

    stopped = await delegation_cancel.cancel_delegated_children(
        "parent", "alice", TURN_START
    )

    assert "handed-off" not in stopped
    assert all(c.args[0] != "handed-off" for c in cancel.await_args_list)


@pytest.mark.asyncio
async def test_an_anonymous_turn_has_no_children_to_find(seams):
    dequeue, cancel = seams

    assert (
        await delegation_cancel.cancel_delegated_children("parent", None, TURN_START)
        == []
    )
    dequeue.assert_not_awaited()
    cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_the_threads_this_turn_set_working_stop(seams):
    """A hand-off an earlier turn left working, and a turn the user started
    in the expert's own thread, are not this turn's to stop."""
    _, cancel = seams

    stopped = await delegation_cancel.cancel_delegated_children(
        "parent", "alice", TURN_START
    )

    assert "earlier-turn" not in stopped
    assert "user-typing-in-thread" not in stopped
    assert all(
        c.args[0] not in {"earlier-turn", "user-typing-in-thread"}
        for c in cancel.await_args_list
    )
