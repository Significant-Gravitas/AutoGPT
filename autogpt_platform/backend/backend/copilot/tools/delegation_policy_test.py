"""The user's delegation settings, as delegate_to_expert applies them."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.delegation_settings import DelegationSettings
from backend.copilot.sdk.session_waiter import SessionOutcome, SessionResult

from .delegate_to_expert import DelegateToExpertTool
from .get_sub_session_result import GetSubSessionResultTool
from .models import ErrorResponse


@pytest.fixture
def settings_db(monkeypatch):
    db = MagicMock()
    db.get_delegation_settings = AsyncMock(
        return_value=DelegationSettings(
            mode="unsupervised", per_delegation_cap_usd=2.0, daily_budget_usd=10.0
        )
    )
    db.get_delegation_spend_since = AsyncMock(return_value=0)
    db.get_session_costs = AsyncMock(return_value={})
    monkeypatch.setattr(
        "backend.copilot.tools.delegation_policy.delegation_db", lambda: db
    )
    monkeypatch.setattr(
        "backend.copilot.tools.sub_session_facts.delegation_db", lambda: db
    )
    return db


@pytest.fixture
def cancel(monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(
        "backend.copilot.tools.delegation_policy.enqueue_cancel_task", mock
    )
    return mock


@pytest.fixture
def delegate(monkeypatch):
    target = MagicMock(id="expert-b", role="PM", avatar_url=None, color="")
    target.name = "Bea"
    target.is_archived = False
    target.schedules_paused_at = None
    create = AsyncMock(return_value=MagicMock(session_id="inner-1"))
    turn = AsyncMock(return_value=("running", SessionResult()))
    for name, value in {
        "resolve_target_expert": AsyncMock(return_value=target),
        "chain_refusal": AsyncMock(return_value=None),
        "create_chat_session": create,
        "run_copilot_turn_via_queue": turn,
        "list_sub_workspace_files": AsyncMock(return_value=[]),
        "build_spawn_state_note": AsyncMock(return_value=""),
        "experts_db": lambda: MagicMock(get_expert=AsyncMock(return_value=None)),
    }.items():
        monkeypatch.setattr(f"backend.copilot.tools.delegate_to_expert.{name}", value)
    return create, turn


def _parent(expert_id: str | None) -> MagicMock:
    sess = MagicMock(session_id="parent", user_id="alice", dry_run=False)
    sess.expert_id = expert_id
    sess.metadata.llm_auth_provider = "platform"
    sess.metadata.llm_credential_id = None
    sess.metadata.origin = "interactive"
    sess.metadata.autopilot_mode = "ask_first"
    return sess


async def _delegate(parent: MagicMock):
    return await DelegateToExpertTool()._execute(
        user_id="alice", session=parent, expert_id="expert-b", prompt="go"
    )


@pytest.mark.asyncio
async def test_a_spent_daily_budget_refuses_the_hand_off(settings_db, delegate):
    create, turn = delegate
    settings_db.get_delegation_spend_since.return_value = 10_000_000

    r = await _delegate(_parent(None))

    assert isinstance(r, ErrorResponse)
    assert "$10.00" in r.message and "budget" in r.message
    create.assert_not_awaited()
    turn.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_budget_refuses_rather_than_overspends(
    settings_db, delegate
):
    create, _ = delegate
    settings_db.get_delegation_spend_since.side_effect = RuntimeError("down")

    r = await _delegate(_parent(None))

    assert isinstance(r, ErrorResponse)
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_ottos_hand_off_runs_in_the_settings_mode_under_its_cap(
    settings_db, delegate
):
    create, _ = delegate

    await _delegate(_parent(None))

    kwargs = create.await_args.kwargs
    assert kwargs["autopilot_mode"] == "unsupervised"
    assert kwargs["delegation_cap_usd"] == 2.0


@pytest.mark.asyncio
async def test_an_experts_hand_off_keeps_the_experts_own_mode(settings_db, delegate):
    create, _ = delegate

    await _delegate(_parent("expert-a"))

    assert create.await_args.kwargs["autopilot_mode"] == "ask_first"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["running", "queued"])
async def test_a_working_delegation_past_its_cap_is_stopped(
    settings_db, delegate, cancel, outcome: SessionOutcome
):
    _, turn = delegate
    turn.return_value = (outcome, SessionResult())
    settings_db.get_session_costs.return_value = {"inner-1": 2_500_000}

    r = await _delegate(_parent(None))

    assert (r.status, r.error) == ("error", "cap reached")
    assert "$2.00" in r.message
    cancel.assert_awaited_once_with("inner-1")


@pytest.mark.asyncio
async def test_a_finished_delegation_past_its_cap_keeps_its_result(
    settings_db, delegate, cancel
):
    _, turn = delegate
    turn.return_value = ("completed", SessionResult(response_text="done"))
    settings_db.get_session_costs.return_value = {"inner-1": 2_500_000}

    r = await _delegate(_parent(None))

    assert r.status == "completed"
    cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_poll_stops_a_running_thread_past_its_stored_cap(
    monkeypatch, settings_db, cancel
):
    sub = MagicMock(user_id="alice", expert_id="expert-b", messages=[])
    sub.metadata.delegated_by_session_id = "parent"
    sub.metadata.handed_off_from_expert_id = None
    sub.metadata.pending_question = None
    sub.metadata.delegation_cap_usd = 1.0
    settings_db.get_session_costs.return_value = {"inner-1": 1_000_000}
    for target, value in {
        "get_chat_session": AsyncMock(return_value=sub),
        "stream_registry.get_session": AsyncMock(
            return_value=MagicMock(status="running")
        ),
        "wait_for_session_result": AsyncMock(return_value=("running", SessionResult())),
        "_delegated_expert_info": AsyncMock(return_value=None),
    }.items():
        monkeypatch.setattr(
            f"backend.copilot.tools.get_sub_session_result.{target}", value
        )

    r = await GetSubSessionResultTool()._execute(
        user_id="alice", session=_parent(None), sub_session_id="inner-1"
    )

    assert (r.status, r.error) == ("error", "cap reached")
    cancel.assert_awaited_once_with("inner-1")
