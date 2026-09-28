"""Cost and timing on every child-session status the model and the UI read."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.sdk.cancelled_output import (
    pop_cancelled_output,
    reset_cancelled_outputs,
    tool_call,
)
from backend.copilot.sdk.session_waiter import SessionOutcome, SessionResult

from .delegate_to_expert import DelegateToExpertTool
from .get_sub_session_result import GetSubSessionResultTool
from .models import SubSessionStatusResponse
from .run_sub_session import response_from_outcome
from .sub_session_facts import RunFacts

_ASKED = datetime(2026, 9, 28, 10, 41, tzinfo=UTC)
_DONE = _ASKED + timedelta(minutes=6)


@pytest.fixture
def costs(monkeypatch):
    db = MagicMock()
    db.get_session_costs = AsyncMock(return_value={"inner-1": 120_000})
    monkeypatch.setattr(
        "backend.copilot.tools.sub_session_facts.delegation_db", lambda: db
    )
    return db


def _parent(expert_id: str | None = None) -> MagicMock:
    sess = MagicMock()
    sess.session_id = "parent"
    sess.user_id = "alice"
    sess.dry_run = False
    sess.expert_id = expert_id
    sess.metadata.llm_auth_provider = "platform"
    sess.metadata.llm_credential_id = None
    sess.metadata.origin = "interactive"
    sess.metadata.autopilot_mode = "ask_first"
    sess.metadata.delegated_by_session_id = None
    return sess


def _message(role: str, at: datetime, content: str = "") -> MagicMock:
    msg = MagicMock()
    msg.role = role
    msg.content = content
    msg.tool_calls = None
    msg.created_at = at
    return msg


@pytest.mark.parametrize(
    "outcome",
    [
        "queued",
        "running",
        "refused",
        "rejected_concurrent_turn_cap",
        "failed",
        "completed",
    ],
)
def test_every_outcome_carries_the_run_facts(outcome: SessionOutcome):
    facts = RunFacts(cost_usd=0.12, started_at=_ASKED, finished_at=_DONE)
    r = response_from_outcome(
        outcome=outcome,
        result=SessionResult(response_text="ok"),
        inner_session_id="inner-1",
        parent_session_id="parent",
        elapsed=1.0,
        workspace_files=[],
        facts=facts,
    )
    assert (r.cost_usd, r.started_at, r.finished_at) == (0.12, _ASKED, _DONE)


class TestDelegateReportsCostAndTimes:
    @pytest.fixture(autouse=True)
    def wiring(self, monkeypatch, costs):
        target = MagicMock(id="expert-b", role="PM", avatar_url=None, color="")
        target.name = "Bea"
        target.is_archived = False
        target.schedules_paused_at = None
        monkeypatch.setattr(
            "backend.copilot.tools.delegate_to_expert.resolve_target_expert",
            AsyncMock(return_value=target),
        )
        monkeypatch.setattr(
            "backend.copilot.tools.delegate_to_expert.chain_refusal",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "backend.copilot.tools.delegate_to_expert.create_chat_session",
            AsyncMock(return_value=MagicMock(session_id="inner-1")),
        )
        monkeypatch.setattr(
            "backend.copilot.tools.delegate_to_expert.list_sub_workspace_files",
            AsyncMock(return_value=[]),
        )
        monkeypatch.setattr(
            "backend.copilot.tools.delegate_to_expert.build_spawn_state_note",
            AsyncMock(return_value=""),
        )

    async def _run(self, monkeypatch, outcome: SessionOutcome):
        monkeypatch.setattr(
            "backend.copilot.tools.delegate_to_expert.run_copilot_turn_via_queue",
            AsyncMock(return_value=(outcome, SessionResult(response_text="done"))),
        )
        return await DelegateToExpertTool()._execute(
            user_id="alice", session=_parent(), expert_id="expert-b", prompt="go"
        )

    @pytest.mark.asyncio
    async def test_a_finished_delegation_has_cost_start_and_end(self, monkeypatch):
        before = datetime.now(UTC)
        r = await self._run(monkeypatch, "completed")
        assert r.cost_usd == 0.12
        assert r.started_at is not None and r.started_at >= before
        assert r.finished_at is not None and r.finished_at >= r.started_at

    @pytest.mark.asyncio
    async def test_a_running_delegation_has_no_end_yet(self, monkeypatch):
        r = await self._run(monkeypatch, "running")
        assert r.status == "running"
        assert r.started_at is not None
        assert r.finished_at is None
        assert r.cost_usd == 0.12


@pytest.mark.asyncio
async def test_a_cold_poll_reads_times_from_the_persisted_turn(monkeypatch, costs):
    sub = MagicMock(user_id="alice", expert_id=None)
    sub.metadata.delegated_by_session_id = None
    sub.metadata.delegation_cap_usd = None
    sub.metadata.pending_question = None
    sub.messages = [_message("user", _ASKED), _message("assistant", _DONE, "done")]
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.get_chat_session",
        AsyncMock(return_value=sub),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.stream_registry.get_session",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.list_sub_workspace_files",
        AsyncMock(return_value=[]),
    )
    r = await GetSubSessionResultTool()._execute(
        user_id="alice", session=_parent(), sub_session_id="inner-1"
    )
    assert r.status == "completed"
    assert (r.started_at, r.finished_at, r.cost_usd) == (_ASKED, _DONE, 0.12)


@pytest.mark.asyncio
async def test_a_delegation_stopped_mid_wait_reads_as_cancelled(monkeypatch, costs):
    """Stop cancels the waiting call; its transcript row must say the
    teammate was stopped and name the thread, not read as still working."""
    target = MagicMock(id="expert-b", role="PM", avatar_url=None, color="violet")
    target.name = "Bea"
    target.is_archived = False
    target.schedules_paused_at = None
    for name, value in {
        "resolve_target_expert": AsyncMock(return_value=target),
        "chain_refusal": AsyncMock(return_value=None),
        "create_chat_session": AsyncMock(return_value=MagicMock(session_id="inner-1")),
        "run_copilot_turn_via_queue": AsyncMock(side_effect=asyncio.CancelledError),
    }.items():
        monkeypatch.setattr(f"backend.copilot.tools.delegate_to_expert.{name}", value)
    reset_cancelled_outputs()

    with pytest.raises(asyncio.CancelledError), tool_call("delegate-call"):
        await DelegateToExpertTool()._execute(
            user_id="alice", session=_parent(), expert_id="expert-b", prompt="go"
        )

    recorded = pop_cancelled_output("delegate-call")
    assert recorded is not None
    stopped = SubSessionStatusResponse.model_validate_json(recorded)
    assert (stopped.status, stopped.sub_session_id) == ("cancelled", "inner-1")
    assert stopped.expert is not None and stopped.expert.name == "Bea"
    assert stopped.started_at is not None
    assert "stopped" in stopped.message
