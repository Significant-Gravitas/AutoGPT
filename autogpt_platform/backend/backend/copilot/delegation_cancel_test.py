"""Stopping a turn stops the threads it handed work to."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import delegation_cancel
from backend.copilot.model import ChatSessionMetadata


def _child(session_id: str, parent: str | None, *, handed_off: bool = False):
    child = MagicMock(session_id=session_id)
    child.metadata = ChatSessionMetadata(
        delegated_by_session_id=parent,
        handed_off_from_expert_id="expert-a" if handed_off else None,
    )
    return child


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

    db.list_chat_sessions_by_status = AsyncMock(side_effect=by_status)
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

    stopped = await delegation_cancel.cancel_delegated_children("parent", "alice")

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

    stopped = await delegation_cancel.cancel_delegated_children("parent", "alice")

    assert "handed-off" not in stopped
    assert all(c.args[0] != "handed-off" for c in cancel.await_args_list)


@pytest.mark.asyncio
async def test_an_anonymous_turn_has_no_children_to_find(seams):
    dequeue, cancel = seams

    assert await delegation_cancel.cancel_delegated_children("parent", None) == []
    dequeue.assert_not_awaited()
    cancel.assert_not_awaited()
