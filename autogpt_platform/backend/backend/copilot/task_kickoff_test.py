"""Unit tests for the task kickoff — no DB, every outbound edge stubbed."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.task_kickoff import start_task_in_new_session

_MODULE = "backend.copilot.task_kickoff"


def _patches(*, claimed: bool = True, session_id: str = "sess-new"):
    client = MagicMock()
    client.claim_task_for_session = AsyncMock(return_value=claimed)
    return client, {
        "create": patch(
            f"{_MODULE}.create_chat_session",
            AsyncMock(return_value=MagicMock(session_id=session_id)),
        ),
        "client": patch(
            f"{_MODULE}.get_database_manager_async_client", return_value=client
        ),
        "schedule": patch(f"{_MODULE}.schedule_chat_turn", AsyncMock()),
    }


@pytest.mark.asyncio
async def test_kickoff_claims_the_task_then_dispatches_a_turn():
    client, patches = _patches()
    with patches["create"] as create, patches["client"], patches[
        "schedule"
    ] as schedule:
        session_id = await start_task_in_new_session(
            "user-1", task_id="task-1", title="Map your client accounts", expert_id="e1"
        )

    assert session_id == "sess-new"
    assert create.call_args.kwargs["expert_id"] == "e1"
    assert create.call_args.kwargs["delegated_task_id"] == "task-1"
    client.claim_task_for_session.assert_awaited_once_with(
        "user-1", "task-1", "sess-new"
    )
    assert schedule.call_args.kwargs["session_id"] == "sess-new"
    assert "task-1" in schedule.call_args.kwargs["message"]


@pytest.mark.asyncio
async def test_kickoff_losing_the_claim_dispatches_nothing():
    """Two kickoffs can race the same task (a hire and the overseer sweep).
    Only the one that wins the QUEUED→WORKING flip gets to run a turn."""
    _, patches = _patches(claimed=False)
    with patches["create"], patches["client"], patches["schedule"] as schedule:
        session_id = await start_task_in_new_session(
            "user-1", task_id="task-1", title="Map your client accounts", expert_id="e1"
        )

    assert session_id is None
    schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_kickoff_reports_none_when_the_session_cannot_be_opened():
    client, patches = _patches()
    with patch(
        f"{_MODULE}.create_chat_session", AsyncMock(side_effect=RuntimeError("boom"))
    ), patches["client"], patches["schedule"] as schedule:
        session_id = await start_task_in_new_session(
            "user-1", task_id="task-1", title="Map your client accounts", expert_id="e1"
        )

    assert session_id is None
    client.claim_task_for_session.assert_not_awaited()
    schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_kickoff_keeps_the_claim_when_the_dispatch_fails():
    """The claim already bound the task to a session, so the overseer can
    retry into it — losing that binding would strand the task again."""
    client, patches = _patches()
    with patches["create"], patches["client"], patch(
        f"{_MODULE}.schedule_chat_turn", AsyncMock(side_effect=RuntimeError("queue"))
    ):
        session_id = await start_task_in_new_session(
            "user-1", task_id="task-1", title="Map your client accounts", expert_id="e1"
        )

    assert session_id == "sess-new"
    client.claim_task_for_session.assert_awaited_once()
