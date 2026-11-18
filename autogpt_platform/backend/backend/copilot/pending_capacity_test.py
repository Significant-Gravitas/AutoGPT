from unittest.mock import AsyncMock

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.pending_message_helpers import QueuePendingMessageResponse
from backend.copilot.sdk import session_waiter
from backend.copilot.tools import message_session
from backend.copilot.tools.models import ErrorResponse
from backend.data.redis_helpers import RedisListCapacityExceeded


@pytest.mark.parametrize("scheduled", [False, True])
async def test_message_session_refuses_full_queue_without_waking_target(
    monkeypatch: pytest.MonkeyPatch, scheduled: bool
) -> None:
    caller = ChatSession.new("owner", dry_run=False)
    target = ChatSession.new("owner", dry_run=False)
    inactive = QueuePendingMessageResponse(
        buffer_length=0, max_buffer_length=10, turn_in_flight=False
    )
    monkeypatch.setattr(message_session, "take_session_message_slot", lambda: None)
    monkeypatch.setattr(
        message_session, "get_chat_session_metadata", AsyncMock(return_value=target)
    )
    monkeypatch.setattr(
        message_session, "get_chat_session_status", AsyncMock(return_value="queued")
    )
    monkeypatch.setattr(
        message_session,
        "queue_user_message",
        AsyncMock(
            side_effect=([inactive] if scheduled else [])
            + [RedisListCapacityExceeded()]
        ),
    )
    tool = message_session.MessageSessionTool()
    wake = AsyncMock()
    monkeypatch.setattr(tool, "_wake", wake)

    result = await tool._execute(
        "owner", caller, session_id=target.session_id, message="Keep this"
    )

    assert isinstance(result, ErrorResponse)
    assert "not sent" in result.message
    assert "try again" in result.message
    wake.assert_not_awaited()


@pytest.mark.parametrize("timeout", [0, 30])
async def test_busy_session_refuses_full_queue_without_waiting_or_dispatching(
    monkeypatch: pytest.MonkeyPatch, timeout: int
) -> None:
    session = ChatSession.new("owner", dry_run=False)
    monkeypatch.setattr(
        session_waiter, "get_chat_session", AsyncMock(return_value=session)
    )
    monkeypatch.setattr(
        session_waiter, "is_turn_in_flight", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        session_waiter,
        "queue_user_message",
        AsyncMock(side_effect=RedisListCapacityExceeded()),
    )
    wait = AsyncMock()
    dispatch = AsyncMock()
    monkeypatch.setattr(session_waiter, "wait_for_session_result", wait)
    monkeypatch.setattr(session_waiter, "schedule_turn", dispatch)

    outcome, result = await session_waiter.run_copilot_turn_via_queue(
        session_id=session.session_id,
        user_id="owner",
        message="Keep this",
        timeout=timeout,
        tool_call_id="sub:parent",
        tool_name="run_sub_session",
    )

    assert outcome == "refused"
    assert not result.queued
    assert result.refusal and "not sent" in result.refusal
    assert "try again" in result.refusal
    wait.assert_not_awaited()
    dispatch.assert_not_awaited()
