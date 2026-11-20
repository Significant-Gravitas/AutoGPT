"""AutoPilotBlock must treat a refused turn as terminal.

``allow_queue=False`` (autopilot.py) made ``refused`` reachable for this block:
``run_copilot_turn_via_queue`` returns it when the target session already has a
turn in flight, and when the tree ledger refuses the spawn. Without a branch for
it the block falls through and renders an empty ``SessionResult`` as a
successful turn, dropping the refusal.

The executor can refuse a turn too, when the account is out of AutoPilot
budget as it picks the turn up and its credit balance can't pay for the turn
either. The waiter reports that as ``refused`` with the not-started pause,
which the block turns into an out-of-budget error that skips recovery.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.autopilot import (
    AutoPilotBlock,
    AutoPilotOutOfBudgetError,
    AutoPilotTransport,
    _enqueue_for_recovery,
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
    return _waited("completed", result)


def _refused_for_budget(pause: TurnPause):
    return _waited("refused", SessionResult(refusal=pause.summary, pause=pause))


def _waited(outcome: str, result: SessionResult):
    return patch(
        "backend.copilot.sdk.session_waiter.run_copilot_turn_via_queue",
        new=AsyncMock(return_value=(outcome, result)),
    )


_OUT_OF_TODAY_MESSAGE = (
    "Out of budget: your plan's allowance is used up, and "
    "your credit balance doesn't cover a run (it needs at least $0.50). Add "
    "credits at https://platform.agpt.co/profile/credits to run it on credits. "
    "The allowance resets 2026-10-12 00:00 UTC."
)


@pytest.fixture
def frontend(mocker):
    settings = MagicMock()
    settings.config.frontend_base_url = "https://platform.agpt.co/"
    mocker.patch("backend.blocks.autopilot.Settings", return_value=settings)
    mocker.patch(
        "backend.blocks.autopilot.credits_can_pay_block_turns", return_value=True
    )


@pytest.mark.asyncio
async def test_a_turn_refused_for_budget_says_why_how_to_top_up_and_when_it_resets(
    frontend,
):
    """The executor only refuses a block turn the credit balance can't cover."""
    with _refused_for_budget(_OUT_OF_TODAY):
        with pytest.raises(AutoPilotOutOfBudgetError) as raised:
            await _execute()
    assert str(raised.value) == _OUT_OF_TODAY_MESSAGE


@pytest.mark.asyncio
async def test_a_spent_trial_says_its_allowance_is_used_up(frontend):
    pause = TurnPause(reason="trial", checkpoint_saved=False, started=False)
    with _refused_for_budget(pause):
        with pytest.raises(AutoPilotOutOfBudgetError) as raised:
            await _execute()
    assert str(raised.value) == (
        "Out of budget: the trial's allowance is used up, and "
        "your credit balance doesn't cover a run (it needs at least $0.50). Add "
        "credits at https://platform.agpt.co/profile/credits to run it on credits."
    )


@pytest.mark.asyncio
async def test_where_credits_cant_pay_the_refusal_doesnt_send_the_user_to_top_up(
    frontend, mocker
):
    # Credits off, or no engine to cap a credit-billed turn: credits never pay.
    mocker.patch(
        "backend.blocks.autopilot.credits_can_pay_block_turns", return_value=False
    )
    with _refused_for_budget(_OUT_OF_TODAY):
        with pytest.raises(AutoPilotOutOfBudgetError) as raised:
            await _execute()
    assert str(raised.value) == (
        "Out of budget: your plan's allowance is used up. "
        "The allowance resets 2026-10-12 00:00 UTC."
    )


@pytest.mark.asyncio
async def test_a_recovery_run_is_still_the_blocks_turn():
    """So it may run on the credit balance as the first run could."""
    session = MagicMock()
    session.metadata.llm_auth_provider = "platform"
    session.metadata.llm_credential_id = None
    session.organization_id = "org-1"
    session.team_id = "team-1"
    enqueue = AsyncMock()

    with (
        patch(
            "backend.copilot.model.get_chat_session", AsyncMock(return_value=session)
        ),
        patch("backend.copilot.executor.utils.enqueue_copilot_turn", enqueue),
    ):
        await _enqueue_for_recovery("sess-1", "user-1", "do the thing", dry_run=False)

    enqueue.assert_awaited_once()
    assert enqueue.await_args.kwargs["tool_name"] == "autopilot_block"
    # Billed to the same wallet: the organization's in an organization's run.
    kwargs = enqueue.await_args.kwargs
    assert (kwargs["organization_id"], kwargs["team_id"]) == ("org-1", "team-1")


@pytest.mark.asyncio
async def test_a_turn_that_paused_after_running_returns_its_checkpoint():
    landed = _OUT_OF_TODAY.model_copy(update={"started": True})
    checkpoint = "Where I got to: half the inbox."
    with _completed(SessionResult(response_text=checkpoint, pause=landed)):
        response, *_ = await _execute()
    assert response == checkpoint


@pytest.mark.asyncio
async def test_a_budget_refusal_reaches_the_graph_without_a_recovery_run(frontend):
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
        _refused_for_budget(_OUT_OF_TODAY),
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

    assert outputs == [("session_id", "sess-1"), ("error", _OUT_OF_TODAY_MESSAGE)]
    recovery.assert_not_awaited()
