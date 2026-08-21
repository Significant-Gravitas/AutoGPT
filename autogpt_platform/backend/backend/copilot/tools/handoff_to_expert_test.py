"""Tests for handoff_to_expert (ownership transfer between experts).

Handoff reuses delegation's queue-backed sub-session, so these patch the same
seams. What is tested here is what handoff adds on top: it never waits, it
frames the task as transferred rather than borrowed, and it records who let
it go.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.sdk.session_waiter import SessionResult
from backend.copilot.tools import TOOL_GROUPS, get_available_tools

from .get_sub_session_result import _in_caller_scope
from .handoff_to_expert import HandoffToExpertTool
from .models import ErrorResponse, SubSessionStatusResponse


def _session(
    user_id: str = "alice",
    session_id: str = "s1",
    expert_id: str | None = "expert-a",
) -> MagicMock:
    sess = MagicMock()
    sess.session_id = session_id
    sess.user_id = user_id
    sess.dry_run = False
    sess.metadata.llm_auth_provider = "platform"
    sess.metadata.llm_credential_id = None
    sess.expert_id = expert_id
    return sess


def _expert(
    expert_id: str = "expert-b",
    name: str = "Bea",
    *,
    is_archived: bool = False,
    schedules_paused_at=None,
) -> MagicMock:
    expert = MagicMock()
    expert.id = expert_id
    expert.name = name
    expert.role = "Ops lead"
    expert.avatar_url = None
    expert.color = "violet"
    expert.is_archived = is_archived
    expert.schedules_paused_at = schedules_paused_at
    return expert


@pytest.fixture
def roster(monkeypatch):
    experts = {"expert-a": _expert("expert-a", "Ari"), "expert-b": _expert()}

    async def fake_get_expert(user_id, expert_id, *, include_workflows=True, **_):
        return experts.get(expert_id)

    db = MagicMock()
    db.get_expert = fake_get_expert
    monkeypatch.setattr(
        "backend.copilot.tools.handoff_to_expert.experts_db",
        lambda: db,
        raising=True,
    )
    return experts


@pytest.fixture
def mock_turn(monkeypatch):
    turn = AsyncMock(return_value=("running", SessionResult()))
    monkeypatch.setattr(
        "backend.copilot.tools.handoff_to_expert.run_copilot_turn_via_queue", turn
    )
    return turn


@pytest.fixture
def mock_sessions(monkeypatch):
    created: list[MagicMock] = []

    async def fake_create(user_id, **kwargs):
        sess = _session(user_id, f"inner-{len(created) + 1}", kwargs.get("expert_id"))
        sess.dry_run = kwargs.get("dry_run", False)
        sess.metadata.delegated_by_expert_id = kwargs.get("delegated_by_expert_id")
        sess.metadata.delegated_by_session_id = kwargs.get("delegated_by_session_id")
        sess.metadata.handed_off_from_expert_id = kwargs.get(
            "handed_off_from_expert_id"
        )
        created.append(sess)
        return sess

    monkeypatch.setattr(
        "backend.copilot.tools.handoff_to_expert.create_chat_session", fake_create
    )
    return created


class TestGuards:
    @pytest.mark.asyncio
    async def test_missing_expert_id_returns_error(self):
        r = await HandoffToExpertTool()._execute(
            user_id="alice", session=_session(), prompt="do it"
        )
        assert isinstance(r, ErrorResponse)

    @pytest.mark.asyncio
    async def test_missing_prompt_returns_error(self):
        r = await HandoffToExpertTool()._execute(
            user_id="alice", session=_session(), expert_id="expert-b"
        )
        assert isinstance(r, ErrorResponse)

    @pytest.mark.asyncio
    async def test_handoff_to_self_rejected(self, roster, mock_turn, mock_sessions):
        r = await HandoffToExpertTool()._execute(
            user_id="alice",
            session=_session(expert_id="expert-a"),
            expert_id="expert-a",
            prompt="hi",
        )
        assert isinstance(r, ErrorResponse)
        mock_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_archived_target_rejected(self, roster, mock_turn, mock_sessions):
        roster["expert-b"].is_archived = True
        r = await HandoffToExpertTool()._execute(
            user_id="alice", session=_session(), expert_id="expert-b", prompt="hi"
        )
        assert isinstance(r, ErrorResponse)
        mock_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_paused_target_rejected(self, roster, mock_turn, mock_sessions):
        roster["expert-b"].schedules_paused_at = "2026-01-01T00:00:00Z"
        r = await HandoffToExpertTool()._execute(
            user_id="alice", session=_session(), expert_id="expert-b", prompt="hi"
        )
        assert isinstance(r, ErrorResponse)
        assert "paused" in r.message
        mock_turn.assert_not_awaited()


class TestTransfer:
    @pytest.mark.asyncio
    async def test_sub_runs_as_target_and_records_handoff_provenance(
        self, roster, mock_turn, mock_sessions
    ):
        await HandoffToExpertTool()._execute(
            user_id="alice",
            session=_session(session_id="s1", expert_id="expert-a"),
            expert_id="expert-b",
            prompt="own the weekly summary",
        )
        assert len(mock_sessions) == 1
        sub = mock_sessions[0]
        assert sub.expert_id == "expert-b"
        assert sub.metadata.handed_off_from_expert_id == "expert-a"
        assert sub.metadata.delegated_by_session_id == "s1"

    @pytest.mark.asyncio
    async def test_never_waits_for_a_result(self, roster, mock_turn, mock_sessions):
        await HandoffToExpertTool()._execute(
            user_id="alice",
            session=_session(),
            expert_id="expert-b",
            prompt="hi",
        )
        assert mock_turn.await_args.kwargs["timeout"] == 0

    @pytest.mark.asyncio
    async def test_framing_transfers_ownership(self, roster, mock_turn, mock_sessions):
        await HandoffToExpertTool()._execute(
            user_id="alice",
            session=_session(expert_id="expert-a"),
            expert_id="expert-b",
            prompt="own the weekly summary",
            context="Q3 numbers are final",
        )
        message = mock_turn.await_args.kwargs["message"]
        assert "Ari" in message
        assert "yours now" in message
        assert "Q3 numbers are final" in message
        assert "own the weekly summary" in message

    @pytest.mark.asyncio
    async def test_response_names_the_new_owner(self, roster, mock_turn, mock_sessions):
        r = await HandoffToExpertTool()._execute(
            user_id="alice",
            session=_session(),
            expert_id="expert-b",
            prompt="hi",
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.expert is not None and r.expert.name == "Bea"
        assert "Sub-AutoPilot" not in r.message


class TestGating:
    def test_handoff_is_hidden_in_autopilot_sessions(self) -> None:
        assert TOOL_GROUPS["handoff_to_expert"] == "experts"
        names = {t["function"]["name"] for t in get_available_tools()}
        hidden = {
            t["function"]["name"]
            for t in get_available_tools(disabled_groups=["experts"])
        }
        assert "handoff_to_expert" in names
        assert "handoff_to_expert" not in hidden

    def test_team_changes_are_hidden_in_expert_sessions(self) -> None:
        remaining = {
            t["function"]["name"]
            for t in get_available_tools(disabled_groups=["expert_admin"])
        }
        assert (
            not {
                "hire_expert",
                "raise_expert",
                "confirm_expert_change",
            }
            & remaining
        )


class TestHandoffPolling:
    """A handoff transfers ownership — the source keeps no poll capability."""

    def _sub(self) -> MagicMock:
        sub = _session(session_id="inner-1", expert_id="expert-b")
        sub.metadata.delegated_by_session_id = "s1"
        sub.metadata.handed_off_from_expert_id = "expert-a"
        return sub

    def test_source_session_cannot_poll_a_handed_off_sub(self) -> None:
        source = _session(session_id="s1", expert_id="expert-a")
        assert _in_caller_scope(self._sub(), source) is False

    def test_receiving_expert_scope_still_reads_its_own_task(self) -> None:
        target = _session(session_id="s2", expert_id="expert-b")
        assert _in_caller_scope(self._sub(), target) is True

    def test_delegation_without_handoff_keeps_the_capability(self) -> None:
        sub = _session(session_id="inner-1", expert_id="expert-b")
        sub.metadata.delegated_by_session_id = "s1"
        sub.metadata.handed_off_from_expert_id = None
        source = _session(session_id="s1", expert_id="expert-a")
        assert _in_caller_scope(sub, source) is True
