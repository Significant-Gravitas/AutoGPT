"""Tests for the shared queue primitive in ``session_waiter``."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import active_turns
from backend.copilot.sdk.session_waiter import SessionResult, run_copilot_turn_via_queue


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


@pytest.fixture(autouse=True)
def idle_target():
    """The target session is idle unless a test says otherwise."""
    with patch(
        "backend.copilot.sdk.session_waiter.chat_db",
        return_value=_status_db("idle"),
    ):
        yield


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
    # Idle when the waiter first looks; queued once the turn waits for a slot.
    statuses = iter(["idle"])
    queued.get_chat_session_status = AsyncMock(
        side_effect=lambda *_a, **_k: next(statuses, "queued")
    )

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
async def test_an_in_flight_session_is_refused():
    """An in-flight target is refused, never joined — a spawn's prompt
    would otherwise execute inside another turn's envelope and
    permissions. Covered directly, because the only other coverage is
    through mocked callers, so deleting the guard would leave those green."""
    scheduled = AsyncMock()

    with (
        patch(
            "backend.copilot.sdk.session_waiter.is_turn_in_flight",
            new=AsyncMock(return_value=True),
        ),
        patch("backend.copilot.sdk.session_waiter.schedule_turn", new=scheduled),
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
        )

    assert outcome == "refused"
    assert result.refusal, "a refusal must carry a message the model can act on"
    scheduled.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_queued_session_is_refused():
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


def _status_db(status: str) -> MagicMock:
    db = MagicMock()
    db.get_chat_session_status = AsyncMock(return_value=status)
    return db
