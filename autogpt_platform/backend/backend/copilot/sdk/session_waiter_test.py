"""Tests for the shared queue primitive in ``session_waiter``.

Focuses on the queue-on-busy fallback: when ``is_turn_in_flight`` is
true, ``run_copilot_turn_via_queue`` must push into the pending buffer
and return ``("queued", SessionResult(queued=True, ...))`` WITHOUT
touching the stream registry or RabbitMQ.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.sdk.session_waiter import SessionResult, run_copilot_turn_via_queue


@pytest.mark.asyncio
async def test_queue_branch_skips_registry_and_enqueue():
    """Busy session → no registry session, no enqueue, queued result."""
    queue_mock = AsyncMock(
        return_value=type(
            "QR",
            (),
            {
                "buffer_length": 4,
                "max_buffer_length": 10,
                "turn_in_flight": True,
            },
        )()
    )
    create_session = AsyncMock()
    enqueue = AsyncMock()

    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.queue_user_message",
            new=queue_mock,
        ),
        patch(
            "backend.copilot.sdk.session_waiter.stream_registry.create_session",
            new=create_session,
        ),
        patch(
            "backend.copilot.sdk.session_waiter.enqueue_copilot_turn",
            new=enqueue,
        ),
    ):
        outcome, result = await run_copilot_turn_via_queue(
            session_id="sess-busy",
            user_id="u1",
            message="follow-up",
            timeout=0.1,
            tool_call_id="sub:parent",
            tool_name="run_sub_session",
        )

    assert outcome == "queued"
    assert isinstance(result, SessionResult)
    assert result.queued is True
    assert result.pending_buffer_length == 4
    # Short-circuit: shared primitive must NOT touch registry or RabbitMQ.
    create_session.assert_not_awaited()
    enqueue.assert_not_awaited()
    queue_mock.assert_awaited_once_with(session_id="sess-busy", message="follow-up")


@pytest.mark.asyncio
async def test_idle_session_enqueues_normally():
    """Idle session → registry session created, enqueued, drain waits."""
    create_session = AsyncMock()
    enqueue = AsyncMock()
    wait_result = AsyncMock(return_value=("completed", SessionResult()))

    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.stream_registry.create_session",
            new=create_session,
        ),
        patch(
            "backend.copilot.sdk.session_waiter.enqueue_copilot_turn",
            new=enqueue,
        ),
        patch(
            "backend.copilot.sdk.session_waiter.wait_for_session_result",
            new=wait_result,
        ),
    ):
        outcome, result = await run_copilot_turn_via_queue(
            session_id="sess-idle",
            user_id="u1",
            message="kick off",
            timeout=0.1,
            tool_call_id="autopilot_block",
            tool_name="autopilot_block",
        )

    assert outcome == "completed"
    assert result.queued is False
    create_session.assert_awaited_once()
    enqueue.assert_awaited_once()
