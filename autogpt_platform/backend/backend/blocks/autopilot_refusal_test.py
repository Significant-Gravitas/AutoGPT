"""AutoPilotBlock must treat a refused turn as terminal.

``allow_queue=False`` (autopilot.py) made ``refused`` reachable for this block:
``run_copilot_turn_via_queue`` returns it when the target session already has a
turn in flight, and when the tree ledger refuses the spawn. Without a branch for
it the block falls through and renders an empty ``SessionResult`` as a
successful turn, dropping the refusal.

The executor can refuse a turn too, when the account is out of AutoPilot
budget as it picks the turn up. That turn completes as a not-started pause, so
the block has to read the pause to fail instead of returning an empty answer.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.autopilot import (
    AutoPilotBlock,
    AutoPilotOutOfBudgetError,
    AutoPilotTransport,
)
from backend.copilot.sdk.session_waiter import SessionResult
from backend.copilot.turn_pause import TurnPause
from backend.data.execution import ExecutionContext


@pytest.mark.asyncio
async def test_a_refused_turn_raises_instead_of_returning_an_empty_success():
    refusal = "That session already has a turn in flight."

    with patch(
        "backend.copilot.sdk.session_waiter.run_copilot_turn_via_queue",
        new=AsyncMock(return_value=("refused", SessionResult(refusal=refusal))),
    ):
        with pytest.raises(RuntimeError, match="already has a turn in flight"):
            await AutoPilotBlock().execute_copilot(
                prompt="do the thing",
                system_context="",
                session_id="sess-busy",
                max_recursion_depth=3,
                user_id="u1",
            )


@pytest.mark.asyncio
async def test_a_refusal_with_no_message_still_raises():
    """``SessionResult.refusal`` is optional, so the fallback must not turn an
    empty refusal back into a silent success."""
    with patch(
        "backend.copilot.sdk.session_waiter.run_copilot_turn_via_queue",
        new=AsyncMock(return_value=("refused", SessionResult())),
    ):
        with pytest.raises(RuntimeError, match="refused"):
            await AutoPilotBlock().execute_copilot(
                prompt="do the thing",
                system_context="",
                session_id="sess-busy",
                max_recursion_depth=3,
                user_id="u1",
            )


_OUT_OF_TODAY = TurnPause(
    reason="daily",
    resets_at=datetime(2026, 10, 12, tzinfo=UTC),
    checkpoint_saved=False,
    started=False,
)


async def _execute():
    return await AutoPilotBlock().execute_copilot(
        prompt="do the thing",
        system_context="",
        session_id="sess-1",
        max_recursion_depth=3,
        user_id="u1",
    )


def _completed(result: SessionResult):
    return patch(
        "backend.copilot.sdk.session_waiter.run_copilot_turn_via_queue",
        new=AsyncMock(return_value=("completed", result)),
    )


@pytest.mark.asyncio
async def test_a_turn_refused_for_budget_fails_with_its_reset():
    with _completed(SessionResult(pause=_OUT_OF_TODAY)):
        with pytest.raises(AutoPilotOutOfBudgetError) as raised:
            await _execute()
    assert str(raised.value) == "Out of AutoPilot budget until 2026-10-12 00:00 UTC."


@pytest.mark.asyncio
async def test_a_spent_trial_says_its_allowance_is_used_up():
    pause = TurnPause(reason="trial", checkpoint_saved=False, started=False)
    with _completed(SessionResult(pause=pause)):
        with pytest.raises(AutoPilotOutOfBudgetError, match="trial allowance"):
            await _execute()


@pytest.mark.asyncio
async def test_a_turn_that_paused_after_running_returns_its_checkpoint():
    landed = _OUT_OF_TODAY.model_copy(update={"started": True})
    checkpoint = "Where I got to: half the inbox."
    with _completed(SessionResult(response_text=checkpoint, pause=landed)):
        response, *_ = await _execute()
    assert response == checkpoint


@pytest.mark.asyncio
async def test_a_budget_refusal_reaches_the_graph_without_a_recovery_run():
    """Re-enqueueing it for recovery would only be refused again."""
    block = AutoPilotBlock()
    recovery = AsyncMock()
    context = ExecutionContext(
        user_id="user-1",
        graph_id="graph-1",
        graph_exec_id="graph-exec-1",
        node_id="node-1",
        node_exec_id="node-exec-1",
    )

    with (
        patch.object(block, "create_session", AsyncMock(return_value="sess-1")),
        _completed(SessionResult(pause=_OUT_OF_TODAY)),
        patch("backend.blocks.autopilot._enqueue_for_recovery", recovery),
    ):
        outputs = [
            item
            async for item in block.run(
                AutoPilotBlock.Input(
                    prompt="go", transport=AutoPilotTransport.PLATFORM
                ),
                execution_context=context,
            )
        ]

    assert outputs == [
        ("session_id", "sess-1"),
        ("error", "Out of AutoPilot budget until 2026-10-12 00:00 UTC."),
    ]
    recovery.assert_not_awaited()
