"""Tests for the shared queue primitive in ``session_waiter``.

Focuses on the queue-on-busy fallback:

* ``timeout == 0`` — push into the buffer and return immediately with
  ``("queued", SessionResult(queued=True, ...))``; skip registry +
  RabbitMQ entirely.
* ``timeout > 0`` — push into the buffer, then subscribe to the
  in-flight turn's stream and return its aggregated result (with
  ``queued=True`` annotation) so callers get the same shape as a
  fresh dispatch.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.sdk.session_waiter import SessionResult, run_copilot_turn_via_queue

_QR = type(
    "QR",
    (),
    {"buffer_length": 4, "max_buffer_length": 10, "turn_in_flight": True},
)


@pytest.mark.asyncio
async def test_queue_branch_timeout_zero_returns_immediately():
    """Busy + timeout=0 → no registry, no enqueue, no wait, queued result."""
    queue_mock = AsyncMock(return_value=_QR())
    create_session = AsyncMock()
    enqueue = AsyncMock()
    wait_result = AsyncMock()

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
            "backend.copilot.executor.utils.enqueue_copilot_turn",
            new=enqueue,
        ),
        patch(
            "backend.copilot.sdk.session_waiter.wait_for_session_result",
            new=wait_result,
        ),
    ):
        outcome, result = await run_copilot_turn_via_queue(
            session_id="sess-busy",
            user_id="u1",
            message="follow-up",
            timeout=0,
            tool_call_id="sub:parent",
            tool_name="run_sub_session",
        )

    assert outcome == "queued"
    assert isinstance(result, SessionResult)
    assert result.queued is True
    assert result.pending_buffer_length == 4
    create_session.assert_not_awaited()
    enqueue.assert_not_awaited()
    wait_result.assert_not_awaited()
    queue_mock.assert_awaited_once_with(session_id="sess-busy", message="follow-up")


@pytest.mark.asyncio
async def test_queue_branch_positive_timeout_rides_inflight_turn():
    """Busy + timeout>0 → push buffer, subscribe to in-flight turn, return
    its aggregated result with ``queued=True`` annotation."""
    queue_mock = AsyncMock(return_value=_QR())
    create_session = AsyncMock()
    enqueue = AsyncMock()
    observed = SessionResult()
    observed.response_text = "final answer from in-flight turn"
    wait_result = AsyncMock(return_value=("completed", observed))

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
            "backend.copilot.executor.utils.enqueue_copilot_turn",
            new=enqueue,
        ),
        patch(
            "backend.copilot.sdk.session_waiter.wait_for_session_result",
            new=wait_result,
        ),
    ):
        outcome, result = await run_copilot_turn_via_queue(
            session_id="sess-busy",
            user_id="u1",
            message="follow-up",
            timeout=30.0,
            tool_call_id="autopilot_block",
            tool_name="autopilot_block",
        )

    # We rode on the existing turn — its outcome + aggregate propagate up.
    assert outcome == "completed"
    assert result.response_text == "final answer from in-flight turn"
    # Marker so callers can tell we didn't start a fresh turn.
    assert result.queued is True
    assert result.pending_buffer_length == 4
    # Still no new registry entry / no new RabbitMQ job — that was the point.
    create_session.assert_not_awaited()
    enqueue.assert_not_awaited()
    # Subscribed to the session stream (not a new turn_id).
    wait_result.assert_awaited_once()
    assert wait_result.await_args.kwargs["session_id"] == "sess-busy"


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
            "backend.copilot.executor.utils.enqueue_copilot_turn",
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


@pytest.mark.asyncio
async def test_idle_session_concurrent_turn_cap_returns_rejected_outcome():
    """Slot-cap rejection in ``schedule_turn`` surfaces as the dedicated
    ``rejected_concurrent_turn_cap`` outcome (not generic ``failed``) so
    callers can render an actionable message instead of pointing at an
    empty transcript."""
    from backend.copilot.active_turns import ConcurrentTurnLimitError

    create_session = AsyncMock()
    enqueue = AsyncMock()
    wait_result = AsyncMock()

    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.executor.utils.acquire_turn_slot",
            side_effect=ConcurrentTurnLimitError(),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.stream_registry.create_session",
            new=create_session,
        ),
        patch(
            "backend.copilot.executor.utils.enqueue_copilot_turn",
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

    assert outcome == "rejected_concurrent_turn_cap"
    assert isinstance(result, SessionResult)
    create_session.assert_not_awaited()
    enqueue.assert_not_awaited()
    wait_result.assert_not_awaited()
