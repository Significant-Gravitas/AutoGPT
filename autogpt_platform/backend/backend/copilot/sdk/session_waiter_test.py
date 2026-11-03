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

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import active_turns
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.executor.utils import TurnPublishUnconfirmed
from backend.copilot.response_model import (
    StreamFinish,
    StreamStart,
    StreamTextDelta,
    StreamTurnPaused,
)
from backend.copilot.sdk.session_waiter import (
    SessionResult,
    TurnNotDispatched,
    run_copilot_turn_via_queue,
    wait_for_session_result,
)
from backend.copilot.turn_pause import TurnPause

_QR = type(
    "QR",
    (),
    {"buffer_length": 4, "max_buffer_length": 10, "turn_in_flight": True},
)


@pytest.fixture(autouse=True)
def mock_session_lookup():
    session = MagicMock()
    session.metadata.llm_auth_provider = "platform"
    session.metadata.llm_credential_id = None
    session.metadata.delegated_by_session_id = None
    with patch(
        "backend.copilot.sdk.session_waiter.get_chat_session",
        new=AsyncMock(return_value=session),
    ) as lookup:
        yield lookup


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
    queue_mock.assert_awaited_once_with(
        session_id="sess-busy", message="follow-up", metadata=None
    )


@pytest.mark.asyncio
async def test_queue_branch_keeps_the_callers_message_metadata():
    """A spawn tool's sender provenance must survive the in-flight fallback:
    the pending message is what becomes the persisted user row."""
    queue_mock = AsyncMock(return_value=_QR())
    provenance = {"from_session_id": "sess-parent", "from_expert_id": "expert-a"}

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
            "backend.copilot.sdk.session_waiter.wait_for_session_result",
            new=AsyncMock(),
        ),
    ):
        outcome, _ = await run_copilot_turn_via_queue(
            session_id="sess-busy",
            user_id="u1",
            message="follow-up",
            timeout=0,
            tool_call_id="sub:parent",
            tool_name="run_sub_session",
            message_metadata=provenance,
        )

    assert outcome == "queued"
    assert queue_mock.await_args.kwargs["metadata"] == provenance


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
    idle_db = MagicMock()
    # Session is idle → CAS idle → running succeeds; running count is 1
    # after the flip (this caller is the only running session).
    idle_db.admit_chat_session_turn = AsyncMock(return_value="admitted")
    idle_db.update_chat_session_status = AsyncMock(return_value=True)
    idle_db.list_chat_sessions_by_status = AsyncMock(return_value=[])
    idle_db.get_chat_session_status = AsyncMock(return_value="idle")

    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.count_inflight_turns",
            new=AsyncMock(return_value=0),
        ),
        patch.object(active_turns, "chat_db", return_value=idle_db),
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


@pytest.mark.parametrize(
    "delegated_by, inflight, expected",
    [
        # A sub-session's turn with four running waits for a slot instead of
        # taking the fifth.
        ("parent-session", 4, "queued_for_slot"),
        # An AutoPilotBlock turn keeps the inflight cap...
        (None, 4, "completed"),
        # ...which still refuses at its number.
        (None, 15, "rejected_concurrent_turn_cap"),
    ],
)
@pytest.mark.asyncio
async def test_a_delegated_turn_leaves_the_users_last_slot_free(
    mock_session_lookup, delegated_by: str | None, inflight: int, expected: str
):
    mock_session_lookup.return_value.metadata.delegated_by_session_id = delegated_by
    db = MagicMock()
    # Four already running: the admit decides by the capacity it is given.
    db.admit_chat_session_turn = AsyncMock(
        side_effect=lambda **kw: "full" if 4 >= kw["capacity"] else "admitted"
    )
    db.get_chat_session_status = AsyncMock(return_value="idle")
    queued = MagicMock()
    queued.get_chat_session_status = AsyncMock(return_value="queued")

    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.turn_queue.count_inflight_turns",
            new=AsyncMock(return_value=inflight),
        ),
        patch.object(active_turns, "chat_db", return_value=db),
        patch.object(active_turns, "get_running_turn_limit", return_value=5),
        patch(
            "backend.copilot.executor.utils.get_inflight_turn_limit", return_value=15
        ),
        patch("backend.copilot.sdk.session_waiter.stream_registry.create_session"),
        patch("backend.copilot.executor.utils.enqueue_copilot_turn"),
        patch(
            "backend.copilot.sdk.session_waiter.wait_for_session_result",
            new=AsyncMock(return_value=("completed", SessionResult())),
        ),
        patch("backend.copilot.sdk.session_waiter.queue_spawned_turn"),
        patch("backend.copilot.sdk.session_waiter.chat_db", return_value=queued),
    ):
        outcome, _ = await run_copilot_turn_via_queue(
            session_id="sess-sub",
            user_id="u1",
            message="kick off",
            timeout=0.1,
            tool_call_id="sub:parent-session",
            tool_name="run_sub_session",
        )

    assert outcome == expected


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
            "backend.copilot.turn_queue.count_inflight_turns",
            new=AsyncMock(return_value=0),
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


@pytest.mark.asyncio
async def test_in_flight_session_with_allow_queue_false_is_refused():
    """``allow_queue=False`` must refuse an in-flight target rather than
    append to its buffer — a spawn's prompt would otherwise execute inside
    another turn's envelope and permissions. Covered directly, because the
    only other coverage is through mocked callers, so deleting the guard
    would leave those green."""
    queue_message = AsyncMock()

    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.queue_user_message",
            new=queue_message,
        ),
        patch(
            "backend.copilot.sdk.session_waiter.chat_db",
            return_value=_status_db("running"),
        ),
    ):
        outcome, result = await run_copilot_turn_via_queue(
            session_id="sess-busy",
            user_id="u1",
            message="do the thing",
            timeout=5,
            tool_call_id="autopilot_block",
            tool_name="autopilot_block",
            allow_queue=False,
        )

    assert outcome == "refused"
    assert result.refusal, "a refusal must carry a message the model can act on"
    queue_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_queued_session_with_allow_queue_false_is_refused():
    """A session that already has a task waiting for a slot takes no second
    one: its queued row is the one the promotion replays."""
    scheduled = AsyncMock()

    with (
        patch(
            "backend.copilot.sdk.session_waiter.chat_db",
            return_value=_status_db("queued"),
        ),
        patch("backend.copilot.sdk.session_waiter.schedule_turn", new=scheduled),
    ):
        outcome, result = await run_copilot_turn_via_queue(
            session_id="sess-waiting",
            user_id="u1",
            message="do the thing",
            timeout=5,
            tool_call_id="sub:parent",
            tool_name="run_sub_session",
            allow_queue=False,
        )

    assert outcome == "refused"
    assert result.refusal
    scheduled.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_tree_refusal_becomes_the_refused_outcome():
    """A ledger refusal (budget or node cap) reaches the caller as
    ``refused`` carrying the ledger's own message, not as ``failed``."""
    from backend.copilot.tree import TreeRefusal

    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.schedule_turn",
            new=AsyncMock(side_effect=TreeRefusal("This task's tree has closed.")),
        ),
    ):
        outcome, result = await run_copilot_turn_via_queue(
            session_id="sess-idle",
            user_id="u1",
            message="spawn",
            timeout=5,
            tool_call_id="sub:parent",
            tool_name="run_sub_session",
        )

    assert outcome == "refused"
    assert result.refusal == "This task's tree has closed."


async def _wait_on(*events):
    queue: asyncio.Queue = asyncio.Queue()
    for event in events:
        queue.put_nowait(("0-0", event))
    with (
        patch(
            "backend.copilot.sdk.session_waiter.stream_registry.subscribe_to_session",
            new=AsyncMock(return_value=queue),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.stream_registry.unsubscribe_from_session",
            new=AsyncMock(),
        ),
    ):
        return await wait_for_session_result(
            session_id="sess-1", user_id="u1", timeout=5
        )


@pytest.mark.asyncio
async def test_a_turn_refused_at_pickup_is_refused_not_completed():
    """The executor ends a turn the allowance can't cover as a not-started
    pause and the stream completes; reported as done, the caller would take
    the empty answer for the task's result."""
    pause = TurnPause(
        reason="weekly",
        resets_at=datetime(2026, 10, 19, tzinfo=UTC),
        checkpoint_saved=False,
        started=False,
    )
    outcome, result = await _wait_on(
        StreamStart(messageId="msg-1"),
        StreamTurnPaused(pause=pause.as_part()),
        StreamFinish(),
    )

    assert outcome == "refused"
    assert result.pause == pause
    assert result.response_text == ""
    assert result.refusal == (
        "Not started: out of this week's budget. "
        "The account's budget resets 2026-10-19 00:00 UTC."
    )


@pytest.mark.asyncio
async def test_a_refusal_with_nothing_to_wait_for_says_so():
    pause = TurnPause(reason="trial", checkpoint_saved=False, started=False)
    outcome, result = await _wait_on(
        StreamTurnPaused(pause=pause.as_part()), StreamFinish()
    )
    assert outcome == "refused"
    assert result.refusal == (
        "Not started: the trial allowance is used up. "
        "The account's budget doesn't reset on its own."
    )


@pytest.mark.asyncio
async def test_a_turn_that_paused_after_running_still_completes():
    pause = TurnPause(reason="daily", checkpoint_saved=True, started=True)
    outcome, result = await _wait_on(
        StreamTextDelta(id="t-1", delta="Where I got to: half the inbox."),
        StreamTurnPaused(pause=pause.as_part()),
        StreamFinish(),
    )
    assert outcome == "completed"
    assert result.pause == pause
    assert result.refusal == ""


@pytest.mark.asyncio
async def test_a_turn_that_ran_and_could_not_continue_still_completes():
    """Out of budget before queued messages, the engine ends a turn that ran
    with a not-started pause after its output; that turn's answer stands."""
    pause = TurnPause(reason="weekly", checkpoint_saved=False, started=False)
    outcome, result = await _wait_on(
        StreamTextDelta(id="t-1", delta="Here is the summary."),
        StreamTurnPaused(pause=pause.as_part()),
        StreamFinish(),
    )
    assert outcome == "completed"
    assert result.response_text == "Here is the summary."
    assert result.refusal == ""


def _status_db(status: str) -> MagicMock:
    db = MagicMock()
    db.get_chat_session_status = AsyncMock(return_value=status)
    return db


@pytest.mark.asyncio
@pytest.mark.parametrize("impersonated", [False, True])
async def test_a_child_turn_carries_its_parents_impersonation(impersonated: bool):
    """A sub-session started inside an impersonated turn is the admin's work
    too, so its LLM analytics are marked the same way."""
    scheduled = AsyncMock()

    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_impersonated",
            return_value=impersonated,
        ),
        patch("backend.copilot.sdk.session_waiter.schedule_turn", new=scheduled),
        patch(
            "backend.copilot.sdk.session_waiter.wait_for_session_result",
            new=AsyncMock(return_value=("completed", SessionResult())),
        ),
    ):
        await run_copilot_turn_via_queue(
            session_id="sess-sub",
            user_id="u1",
            message="spawn",
            timeout=5,
            tool_call_id="sub:parent",
            tool_name="run_sub_session",
        )

    assert scheduled.await_args.kwargs["impersonated"] is impersonated


@pytest.mark.asyncio
async def test_a_queued_child_turn_carries_its_parents_impersonation(
    mock_session_lookup,
):
    """A delegated child the running cap sends to the queue keeps the flag
    for when it is promoted."""
    mock_session_lookup.return_value.metadata.delegated_by_session_id = "parent"
    queued = AsyncMock()

    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_impersonated",
            return_value=True,
        ),
        patch(
            "backend.copilot.sdk.session_waiter.schedule_turn",
            new=AsyncMock(side_effect=active_turns.ConcurrentTurnLimitError()),
        ),
        patch("backend.copilot.sdk.session_waiter.queue_spawned_turn", new=queued),
        patch(
            "backend.copilot.sdk.session_waiter.wait_for_queued_session",
            new=AsyncMock(return_value=("completed", SessionResult())),
        ),
    ):
        await run_copilot_turn_via_queue(
            session_id="sess-sub",
            user_id="u1",
            message="spawn",
            timeout=5,
            tool_call_id="sub:parent",
            tool_name="run_sub_session",
        )

    assert queued.await_args.kwargs["impersonated"] is True


@pytest.mark.asyncio
async def test_a_dispatched_turn_with_no_wait_returns_without_reading_its_stream():
    """``timeout == 0`` after a dispatch has nothing to wait for. Reading the
    stream anyway is pure risk: the turn is already queued, so a Redis error
    there would read as "not started" and a caller retrying on it would run
    the same task twice (SECRT-3138 review)."""
    scheduled = AsyncMock()
    stream_read = AsyncMock(side_effect=RuntimeError("redis subscribe failed"))
    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch("backend.copilot.sdk.session_waiter.schedule_turn", new=scheduled),
        patch(
            "backend.copilot.sdk.session_waiter.wait_for_session_result",
            new=stream_read,
        ),
    ):
        outcome, result = await run_copilot_turn_via_queue(
            session_id="sess-fire-and-forget",
            user_id="u1",
            message="own the Printify pipeline",
            timeout=0,
            tool_call_id="delegate:parent",
            tool_name="delegate_to_expert",
        )

    assert outcome == "running"
    assert result == SessionResult()
    scheduled.assert_awaited_once()
    stream_read.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_turn_queued_for_a_slot_with_no_wait_returns_without_reading_it(
    mock_session_lookup,
):
    """Once queue_spawned_turn has stored the turn it will run when a slot
    frees, so with nothing to wait for a status read can only fail, and a
    failure there would read as "not queued" (SECRT-3138 review)."""
    mock_session_lookup.return_value.metadata.delegated_by_session_id = "parent"
    queued = AsyncMock()
    status_read = AsyncMock(side_effect=RuntimeError("db blip"))
    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.schedule_turn",
            new=AsyncMock(side_effect=ConcurrentTurnLimitError("full")),
        ),
        patch("backend.copilot.sdk.session_waiter.queue_spawned_turn", new=queued),
        patch(
            "backend.copilot.sdk.session_waiter.wait_for_queued_session",
            new=status_read,
        ),
    ):
        outcome, result = await run_copilot_turn_via_queue(
            session_id="sess-slot",
            user_id="u1",
            message="own the Printify pipeline",
            timeout=0,
            tool_call_id="delegate:parent",
            tool_name="delegate_to_expert",
        )

    assert outcome == "queued_for_slot"
    assert result.queued is True
    queued.assert_awaited_once()
    status_read.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failure_before_the_dispatch_is_typed_as_not_dispatched():
    """Callers may remove a thread or send again only when the turn provably
    never reached the queue, so that case is typed; a publish that raised
    stays TurnPublishUnconfirmed (Sentry on #16101)."""
    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.schedule_turn",
            new=AsyncMock(side_effect=ConnectionError("redis down")),
        ),
        pytest.raises(TurnNotDispatched) as raised,
    ):
        await run_copilot_turn_via_queue(
            session_id="s",
            user_id="u1",
            message="m",
            timeout=0,
            tool_call_id="delegate:p",
            tool_name="delegate_to_expert",
        )
    assert isinstance(raised.value.__cause__, ConnectionError)
    # Callers that report str(exc) (the AutoPilot block) keep the reason.
    assert "redis down" in str(raised.value)


@pytest.mark.asyncio
async def test_a_failed_status_check_is_typed_as_not_dispatched():
    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(side_effect=ConnectionError("redis down")),
        ),
        pytest.raises(TurnNotDispatched, match="redis down"),
    ):
        await run_copilot_turn_via_queue(
            session_id="s",
            user_id="u1",
            message="m",
            timeout=0,
            tool_call_id="delegate:p",
            tool_name="delegate_to_expert",
        )


@pytest.mark.asyncio
async def test_a_publish_that_raised_is_not_called_undispatched():
    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.schedule_turn",
            new=AsyncMock(side_effect=TurnPublishUnconfirmed("confirm lost")),
        ),
        pytest.raises(TurnPublishUnconfirmed),
    ):
        await run_copilot_turn_via_queue(
            session_id="s",
            user_id="u1",
            message="m",
            timeout=0,
            tool_call_id="delegate:p",
            tool_name="delegate_to_expert",
        )


async def _queue_that_raised(chat):
    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.schedule_turn",
            new=AsyncMock(side_effect=ConcurrentTurnLimitError("full")),
        ),
        patch(
            "backend.copilot.sdk.session_waiter.queue_spawned_turn",
            new=AsyncMock(side_effect=ConnectionError("redis down")),
        ),
        patch("backend.copilot.sdk.session_waiter.chat_db", return_value=chat),
    ):
        await run_copilot_turn_via_queue(
            session_id="sess-slot",
            user_id="u1",
            message="own the Printify pipeline",
            timeout=0,
            tool_call_id="delegate:parent",
            tool_name="delegate_to_expert",
        )


def _chat(*, status="idle", next_sequence=0, read_error=None):
    chat = MagicMock()
    chat.get_chat_session_status = AsyncMock(return_value=status)
    chat.get_next_sequence = AsyncMock(
        return_value=next_sequence, side_effect=read_error
    )
    return chat


@pytest.mark.asyncio
async def test_a_wait_for_a_slot_that_stored_nothing_is_not_dispatched(
    mock_session_lookup,
):
    """The queue is the session's rows, so a queue attempt that raised and
    left a fresh thread with no message, not queued, holds nothing that can
    run (Sentry on #16101)."""
    mock_session_lookup.return_value.metadata.delegated_by_session_id = "parent"
    with pytest.raises(TurnNotDispatched) as raised:
        await _queue_that_raised(_chat())
    assert isinstance(raised.value.__cause__, ConnectionError)
    # Callers that report str(exc) (the AutoPilot block) keep the reason.
    assert "redis down" in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat",
    [
        pytest.param(_chat(next_sequence=1), id="a-message-was-stored"),
        pytest.param(_chat(status="queued"), id="the-session-is-queued"),
        pytest.param(_chat(read_error=RuntimeError("db")), id="the-check-failed"),
    ],
)
async def test_a_wait_for_a_slot_that_may_have_stored_the_turn_stays_untyped(
    mock_session_lookup, chat
):
    mock_session_lookup.return_value.metadata.delegated_by_session_id = "parent"
    with pytest.raises(ConnectionError):
        await _queue_that_raised(chat)
