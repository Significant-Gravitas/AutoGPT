"""Tests for pod-lead routing on delegate_to_expert.

A delegation aimed at a pod member from outside the pod must land on the
pod's lead — who then distributes the work within the members with the same
tools — while intra-pod delegation keeps its direct target. The harness
mirrors ``delegate_to_expert_test``: an in-memory roster plus pods, faked
session CRUD, and a mocked queue turn.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.sdk.session_waiter import SessionResult

from .delegate_to_expert import DelegateToExpertTool
from .models import ErrorResponse


def _session(
    user_id: str = "alice",
    session_id: str = "s1",
    expert_id: str | None = "expert-out",
) -> MagicMock:
    sess = MagicMock()
    sess.session_id = session_id
    sess.user_id = user_id
    sess.dry_run = False
    sess.metadata.llm_auth_provider = "platform"
    sess.metadata.llm_credential_id = None
    sess.metadata.delegated_by_session_id = None
    sess.metadata.origin = "interactive"
    sess.expert_id = expert_id
    return sess


def _expert(expert_id: str, name: str, *, pod_id: str | None = None) -> MagicMock:
    expert = MagicMock()
    expert.id = expert_id
    expert.name = name
    expert.role = "Specialist"
    expert.avatar_url = None
    expert.color = "violet"
    expert.is_archived = False
    expert.schedules_paused_at = None
    expert.pod_id = pod_id
    return expert


def _pod(pod_id: str, name: str, lead_expert_id: str | None) -> MagicMock:
    pod = MagicMock()
    pod.id = pod_id
    pod.name = name
    pod.lead_expert_id = lead_expert_id
    return pod


@pytest.fixture
def team(monkeypatch):
    """A pod with a lead and two members, plus an outsider."""
    experts = {
        "expert-lead": _expert("expert-lead", "Lena", pod_id="pod-growth"),
        "expert-m1": _expert("expert-m1", "Mia", pod_id="pod-growth"),
        "expert-m2": _expert("expert-m2", "Moe", pod_id="pod-growth"),
        "expert-out": _expert("expert-out", "Omar"),
    }
    pods = [_pod("pod-growth", "Growth", "expert-lead")]

    async def fake_get_expert(user_id, expert_id, *, include_workflows=True, **_):
        return experts.get(expert_id)

    async def fake_list_experts(user_id, *, with_metrics=True, **_):
        return list(experts.values())

    async def fake_list_pods(user_id):
        return list(pods)

    db = MagicMock()
    db.get_expert = fake_get_expert
    db.list_experts = fake_list_experts
    db.list_pods = fake_list_pods
    for module in ("delegate_to_expert", "expert_delegation"):
        monkeypatch.setattr(
            f"backend.copilot.tools.{module}.experts_db", lambda: db, raising=True
        )
    return experts


@pytest.fixture
def mock_turn(monkeypatch):
    turn = AsyncMock(return_value=("running", SessionResult()))
    monkeypatch.setattr(
        "backend.copilot.tools.delegate_to_expert.run_copilot_turn_via_queue", turn
    )
    monkeypatch.setattr(
        "backend.copilot.tools.delegate_to_expert.list_sub_workspace_files",
        AsyncMock(return_value=None),
    )
    return turn


@pytest.fixture
def mock_sessions(monkeypatch):
    created: list[MagicMock] = []

    async def fake_create(user_id, **kwargs):
        sess = _session(user_id, f"inner-{len(created) + 1}", kwargs.get("expert_id"))
        sess.metadata.delegated_by_expert_id = kwargs.get("delegated_by_expert_id")
        sess.metadata.delegated_by_session_id = kwargs.get("delegated_by_session_id")
        sess.metadata.origin = kwargs.get("origin")
        sess.metadata.handed_off_from_expert_id = None
        created.append(sess)
        return sess

    async def fake_get(session_id, user_id=None):
        return next((s for s in created if s.session_id == session_id), None)

    monkeypatch.setattr(
        "backend.copilot.tools.delegate_to_expert.create_chat_session", fake_create
    )
    monkeypatch.setattr(
        "backend.copilot.tools.delegate_to_expert.get_chat_session", fake_get
    )
    monkeypatch.setattr(
        "backend.copilot.tools.expert_delegation.get_chat_session", fake_get
    )
    return created


async def _delegate(session: MagicMock, expert_id: str):
    return await DelegateToExpertTool()._execute(
        user_id="alice",
        session=session,
        expert_id=expert_id,
        prompt="grow the newsletter",
        wait_for_result=0,
    )


class TestPodLeadRouting:
    @pytest.mark.asyncio
    async def test_outside_delegation_to_a_pod_member_lands_on_the_lead(
        self, team, mock_turn, mock_sessions
    ):
        r = await _delegate(_session(expert_id="expert-out"), "expert-m1")

        assert not isinstance(r, ErrorResponse)
        assert mock_sessions[0].expert_id == "expert-lead"

    @pytest.mark.asyncio
    async def test_autopilot_delegation_to_a_pod_member_lands_on_the_lead(
        self, team, mock_turn, mock_sessions
    ):
        r = await _delegate(_session(expert_id=None), "expert-m1")

        assert not isinstance(r, ErrorResponse)
        assert mock_sessions[0].expert_id == "expert-lead"

    @pytest.mark.asyncio
    async def test_an_ask_naming_the_pod_lands_on_the_lead(
        self, team, mock_turn, mock_sessions
    ):
        r = await _delegate(_session(expert_id="expert-out"), "growth")

        assert not isinstance(r, ErrorResponse)
        assert mock_sessions[0].expert_id == "expert-lead"

    @pytest.mark.asyncio
    async def test_the_lead_delegates_directly_to_members(
        self, team, mock_turn, mock_sessions
    ):
        r = await _delegate(_session(expert_id="expert-lead"), "expert-m1")

        assert not isinstance(r, ErrorResponse)
        assert mock_sessions[0].expert_id == "expert-m1"

    @pytest.mark.asyncio
    async def test_intra_pod_delegation_keeps_its_direct_target(
        self, team, mock_turn, mock_sessions
    ):
        r = await _delegate(_session(expert_id="expert-m1"), "expert-m2")

        assert not isinstance(r, ErrorResponse)
        assert mock_sessions[0].expert_id == "expert-m2"

    @pytest.mark.asyncio
    async def test_a_paused_lead_falls_back_to_the_direct_target(
        self, team, mock_turn, mock_sessions
    ):
        team["expert-lead"].schedules_paused_at = "2026-01-01T00:00:00Z"

        r = await _delegate(_session(expert_id="expert-out"), "expert-m1")

        assert not isinstance(r, ErrorResponse)
        assert mock_sessions[0].expert_id == "expert-m1"

    @pytest.mark.asyncio
    async def test_a_delegation_to_a_member_of_a_leadless_pod_is_direct(
        self, team, mock_turn, mock_sessions
    ):
        team["expert-m1"].pod_id = "pod-leadless"

        r = await _delegate(_session(expert_id="expert-out"), "expert-m1")

        assert not isinstance(r, ErrorResponse)
        assert mock_sessions[0].expert_id == "expert-m1"


class TestPodChainGuards:
    """The routed hop rides the same provenance chain as any delegation, so
    the depth bound and the loop check must still hold around it."""

    def _chain(self, mock_sessions, expert_ids: list[str | None]) -> MagicMock:
        parent_id = None
        for depth, expert_id in enumerate(expert_ids):
            sess = _session(session_id=f"chain-{depth}", expert_id=expert_id)
            sess.metadata.delegated_by_session_id = parent_id
            mock_sessions.append(sess)
            parent_id = sess.session_id
        return mock_sessions[-1]

    @pytest.mark.asyncio
    async def test_router_to_lead_to_member_chain_clears_the_depth_guard(
        self, team, mock_turn, mock_sessions
    ):
        """Autopilot delegated to the lead; the lead now delegates to a
        member. Two hops sit well inside MAX_DELEGATION_DEPTH, so the extra
        pod hop must not trip the guard."""
        lead_session = self._chain(mock_sessions, [None, "expert-lead"])

        r = await _delegate(lead_session, "expert-m1")

        assert not isinstance(r, ErrorResponse)
        mock_turn.assert_awaited_once()
        assert mock_sessions[-1].expert_id == "expert-m1"

    @pytest.mark.asyncio
    async def test_a_member_handing_back_to_the_lead_is_refused_as_a_loop(
        self, team, mock_turn, mock_sessions
    ):
        """The lead delegated to a member; the member delegating back to the
        lead must still be refused by the chain walk — pod routing does not
        open a way around the ``seen`` guard."""
        member_session = self._chain(mock_sessions, ["expert-lead", "expert-m1"])

        r = await _delegate(member_session, "expert-lead")

        assert isinstance(r, ErrorResponse)
        assert "loop" in r.message
        mock_turn.assert_not_awaited()
