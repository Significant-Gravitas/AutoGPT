"""The user's delegation settings, as delegate_to_expert applies them."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.delegation_settings import DelegationSettings
from backend.copilot.model import ChatSessionInfo, ChatSessionMetadata, PendingQuestion
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
    db.get_expert_hired_at = AsyncMock(
        return_value=datetime.now(UTC) - timedelta(days=30)
    )
    monkeypatch.setattr(
        "backend.copilot.tools.delegation_policy.delegation_db", lambda: db
    )
    monkeypatch.setattr(
        "backend.copilot.tools.sub_session_facts.delegation_db", lambda: db
    )
    return db


def _no_over_cap_question() -> DelegationSettings:
    return DelegationSettings(
        mode="unsupervised", per_delegation_cap_usd=2.0, ask_before_over_cap=False
    )


@pytest.fixture
def parked(monkeypatch):
    chat = MagicMock()
    chat.set_session_pending_question = AsyncMock()
    monkeypatch.setattr("backend.copilot.delegation_cap.chat_db", lambda: chat)
    return chat.set_session_pending_question


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
    settings_db, delegate, cancel, parked, outcome: SessionOutcome
):
    """With ``ask_before_over_cap`` off: stop it and say the cap was reached."""
    _, turn = delegate
    turn.return_value = (outcome, SessionResult())
    settings_db.get_session_costs.return_value = {"inner-1": 2_500_000}
    settings_db.get_delegation_settings.return_value = _no_over_cap_question()

    r = await _delegate(_parent(None))

    assert (r.status, r.error) == ("error", "cap reached")
    assert "$2.00" in r.message
    cancel.assert_awaited_once_with("inner-1")
    parked.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_working_delegation_past_its_cap_asks_to_raise_it(
    settings_db, delegate, cancel, parked
):
    """With ``ask_before_over_cap`` on (the default): stop spending, and park
    the thread on the cap question instead of failing it."""
    _, turn = delegate
    settings_db.get_session_costs.return_value = {"inner-1": 2_500_000}

    r = await _delegate(_parent(None))

    assert r.status == "needs_input"
    assert (
        r.question == "This hand-off has reached its $2.00 cap. Raise it and continue?"
    )
    assert r.question_options == ["Raise by $1", "Raise by $5", "Stop"]
    cancel.assert_awaited_once_with("inner-1")
    parked.assert_awaited_once()
    assert parked.await_args.args[:3] == ("inner-1", "alice", r.question)
    assert parked.await_args.kwargs["options"] == r.question_options


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
def _thread(**metadata) -> ChatSessionInfo:
    return ChatSessionInfo(
        session_id="inner-1",
        user_id="alice",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        expert_id="expert-b",
        metadata=ChatSessionMetadata(
            delegated_by_session_id="parent", delegation_cap_usd=1.0, **metadata
        ),
    )


async def _poll(monkeypatch, fresh: ChatSessionInfo, running: bool):
    sub = MagicMock(user_id="alice", expert_id="expert-b", messages=[])
    sub.metadata = fresh.metadata
    registry = MagicMock(status="running") if running else None
    last = MagicMock(role="assistant", content="stopped", tool_calls=None)
    last.created_at = datetime.now(UTC)
    sub.messages = [] if running else [last]
    for target, value in {
        "get_chat_session": AsyncMock(return_value=sub),
        "get_chat_session_metadata": AsyncMock(return_value=fresh),
        "stream_registry.get_session": AsyncMock(return_value=registry),
        "wait_for_session_result": AsyncMock(return_value=("running", SessionResult())),
        "_delegated_expert_info": AsyncMock(return_value=None),
        "list_sub_workspace_files": AsyncMock(return_value=[]),
    }.items():
        monkeypatch.setattr(
            f"backend.copilot.tools.get_sub_session_result.{target}", value
        )
    return await GetSubSessionResultTool()._execute(
        user_id="alice", session=_parent(None), sub_session_id="inner-1"
    )


@pytest.mark.asyncio
async def test_a_poll_stops_a_running_thread_past_its_stored_cap(
    monkeypatch, settings_db, cancel, parked
):
    settings_db.get_session_costs.return_value = {"inner-1": 1_000_000}
    settings_db.get_delegation_settings.return_value = _no_over_cap_question()

    r = await _poll(monkeypatch, _thread(), running=True)

    assert (r.status, r.error) == ("error", "cap reached")
    cancel.assert_awaited_once_with("inner-1")


@pytest.mark.asyncio
async def test_a_poll_of_a_parked_thread_asks_again_without_re_parking(
    monkeypatch, settings_db, cancel, parked
):
    settings_db.get_session_costs.return_value = {"inner-1": 1_000_000}
    question = PendingQuestion(
        text="This hand-off has reached its $1.00 cap. Raise it and continue?",
        asked_at=datetime.now(UTC),
        options=["Raise by $1", "Raise by $5", "Stop"],
    )

    r = await _poll(monkeypatch, _thread(pending_question=question), running=False)

    assert (r.status, r.question) == ("needs_input", question.text)
    parked.assert_not_awaited()
    cancel.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_thread_the_user_stopped_at_its_cap_stays_stopped(
    monkeypatch, settings_db, cancel, parked
):
    settings_db.get_session_costs.return_value = {"inner-1": 1_000_000}

    r = await _poll(monkeypatch, _thread(delegation_cap_stopped=True), running=False)

    assert (r.status, r.error) == ("error", "cap reached")
    parked.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_raised_cap_lets_the_thread_run_on(
    monkeypatch, settings_db, cancel, parked
):
    """The poll reads the cap fresh, so a raise takes effect at once."""
    settings_db.get_session_costs.return_value = {"inner-1": 1_000_000}

    r = await _poll(monkeypatch, _thread(), running=True)
    raised = _thread()
    raised.metadata.delegation_cap_usd = 2.0
    cancel.reset_mock()
    r = await _poll(monkeypatch, raised, running=True)

    assert r.status == "running"
    cancel.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hired_days_ago,toggle,expected",
    [
        (2, True, "ask_first"),
        (8, True, "unsupervised"),
        (2, False, "unsupervised"),
    ],
)
async def test_a_newly_hired_teammate_starts_in_ask_first(
    settings_db, delegate, hired_days_ago, toggle, expected
):
    """Only while the toggle is on, and only in the teammate's first week."""
    create, _ = delegate
    settings_db.get_delegation_settings.return_value = DelegationSettings(
        mode="unsupervised", new_experts_ask_first=toggle
    )
    settings_db.get_expert_hired_at.return_value = datetime.now(UTC) - timedelta(
        days=hired_days_ago
    )

    await _delegate(_parent(None))

    assert create.await_args.kwargs["autopilot_mode"] == expected
    # The hire date is only read while the toggle is on.
    assert settings_db.get_expert_hired_at.await_count == (1 if toggle else 0)
