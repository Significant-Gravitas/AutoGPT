"""Tests for run_sub_session + get_sub_session_result (queue-backed flow).

Sub-AutoPilots are enqueued on the copilot_execution RabbitMQ queue and
executed by any copilot_executor worker. The tools wait for completion
by subscribing to ``stream_registry`` for the sub's ChatSession. These
tests patch the three integration seams — ``enqueue_copilot_turn``,
``wait_for_session_completion``, and ``stream_registry.create_session``
— to exercise the tool logic without needing RabbitMQ or Redis.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from .get_sub_session_result import GetSubSessionResultTool
from .models import ErrorResponse, SubSessionStatusResponse
from .run_sub_session import MAX_SUB_SESSION_WAIT_SECONDS, RunSubSessionTool


def _session(user_id: str = "u", session_id: str = "s1") -> MagicMock:
    sess = MagicMock()
    sess.session_id = session_id
    sess.dry_run = False
    return sess


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_queue(monkeypatch):
    """Patch the enqueue helpers + the stream-registry session creator so
    tests don't need RabbitMQ or Redis. Returns a dict of the mocks so
    individual tests can assert on them.
    """
    enqueue_turn = AsyncMock()
    enqueue_cancel = AsyncMock()
    create_session = AsyncMock()

    monkeypatch.setattr(
        "backend.copilot.tools.run_sub_session.enqueue_copilot_turn",
        enqueue_turn,
    )
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.enqueue_cancel_task",
        enqueue_cancel,
    )
    monkeypatch.setattr(
        "backend.copilot.tools.run_sub_session.stream_registry.create_session",
        create_session,
    )
    return {
        "enqueue_turn": enqueue_turn,
        "enqueue_cancel": enqueue_cancel,
        "create_session": create_session,
    }


@pytest.fixture
def mock_waiter(monkeypatch):
    """Patch wait_for_session_completion in both tool modules. The waiter
    takes keyword-only args (session_id, user_id, timeout) and returns a
    SessionOutcome string."""
    waiter = AsyncMock(return_value="running")
    monkeypatch.setattr(
        "backend.copilot.tools.run_sub_session.wait_for_session_completion",
        waiter,
    )
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.wait_for_session_completion",
        waiter,
    )
    return waiter


@pytest.fixture
def mock_model(monkeypatch):
    """Patch the model-layer helpers the tools call for session CRUD +
    ownership checks. The create side returns a fake ChatSession with a
    fresh uuid each call."""
    created: list[MagicMock] = []

    async def fake_create(user_id: str, *, dry_run: bool):
        sess = MagicMock()
        sess.session_id = f"inner-{len(created) + 1}"
        sess.user_id = user_id
        sess.dry_run = dry_run
        sess.messages = []
        created.append(sess)
        return sess

    async def fake_get(session_id: str):
        for s in created:
            if s.session_id == session_id:
                return s
        return None

    # The tool modules bind these names at import time, so patch the
    # local module bindings (not the source in backend.copilot.model).
    monkeypatch.setattr(
        "backend.copilot.tools.run_sub_session.create_chat_session", fake_create
    )
    monkeypatch.setattr(
        "backend.copilot.tools.run_sub_session.get_chat_session", fake_get
    )
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.get_chat_session", fake_get
    )
    return {"created": created, "get": fake_get}


# ---------------------------------------------------------------------------
# RunSubSessionTool
# ---------------------------------------------------------------------------


class TestRunSubSession:
    @pytest.mark.asyncio
    async def test_missing_prompt_returns_error(self):
        r = await RunSubSessionTool()._execute(
            user_id="u", session=_session(), prompt=""
        )
        assert isinstance(r, ErrorResponse)

    @pytest.mark.asyncio
    async def test_no_user_returns_error(self):
        r = await RunSubSessionTool()._execute(
            user_id=None, session=_session(), prompt="hi"
        )
        assert isinstance(r, ErrorResponse)

    @pytest.mark.asyncio
    async def test_resume_with_other_users_session_id_rejected(
        self, monkeypatch, mock_queue, mock_waiter
    ):
        """Ownership must be re-verified when the caller passes a resume id."""
        foreign = MagicMock(session_id="alien-sess", user_id="not-caller", messages=[])

        async def fake_get(session_id: str):
            if session_id == "alien-sess":
                return foreign
            return None

        monkeypatch.setattr(
            "backend.copilot.tools.run_sub_session.get_chat_session", fake_get
        )

        r = await RunSubSessionTool()._execute(
            user_id="alice",
            session=_session("alice"),
            prompt="continue",
            sub_autopilot_session_id="alien-sess",
        )
        assert isinstance(r, ErrorResponse)
        assert "is not a session you own" in r.message
        mock_queue["enqueue_turn"].assert_not_awaited()

    @pytest.mark.asyncio
    async def test_propagates_dry_run_to_sub(self, mock_queue, mock_waiter, mock_model):
        """Fresh sub-session must inherit the parent's dry_run flag."""
        parent = _session("alice")
        parent.dry_run = True
        await RunSubSessionTool()._execute(
            user_id="alice",
            session=parent,
            prompt="hi",
            wait_for_result=0,  # skip the wait helper for this assertion
        )
        assert mock_model["created"], "create_chat_session was never awaited"
        assert mock_model["created"][0].dry_run is True

    @pytest.mark.asyncio
    async def test_forwards_parent_permissions_to_queue(
        self, monkeypatch, mock_queue, mock_waiter, mock_model
    ):
        """The parent's CopilotPermissions must be passed on the enqueued
        CoPilotExecutionEntry so the worker applies the same filter."""
        from backend.copilot.permissions import CopilotPermissions

        perms = CopilotPermissions(tools=["run_block"], tools_exclude=False)
        monkeypatch.setattr(
            "backend.copilot.tools.run_sub_session.get_current_permissions",
            lambda: perms,
        )
        await RunSubSessionTool()._execute(
            user_id="alice",
            session=_session("alice"),
            prompt="hi",
            wait_for_result=0,
        )
        mock_queue["enqueue_turn"].assert_awaited_once()
        kwargs = mock_queue["enqueue_turn"].await_args.kwargs
        assert kwargs["permissions"] is perms

    @pytest.mark.asyncio
    async def test_wait_for_result_zero_returns_running(
        self, mock_queue, mock_waiter, mock_model
    ):
        """wait_for_result=0 must skip wait_for_session_completion entirely
        and return status=running so the caller polls."""
        r = await RunSubSessionTool()._execute(
            user_id="alice",
            session=_session("alice"),
            prompt="hi",
            wait_for_result=0,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "running"
        assert r.sub_session_id == r.sub_autopilot_session_id == "inner-1"
        assert r.sub_autopilot_session_link == "/copilot?sessionId=inner-1"
        mock_waiter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_wait_for_result_completed_returns_final_response(
        self, monkeypatch, mock_queue, mock_waiter, mock_model
    ):
        """When the waiter returns 'completed', the tool reads the sub's
        last assistant message for response_text + tool_calls."""
        mock_waiter.return_value = "completed"

        async def fake_get(session_id: str):
            sess = MagicMock()
            sess.session_id = session_id
            sess.user_id = "alice"
            assistant = MagicMock()
            assistant.role = "assistant"
            assistant.content = "the answer"
            assistant.tool_calls = [{"function": {"name": "foo"}}]
            sess.messages = [assistant]
            return sess

        monkeypatch.setattr(
            "backend.copilot.tools.run_sub_session.get_chat_session", fake_get
        )

        r = await RunSubSessionTool()._execute(
            user_id="alice",
            session=_session("alice"),
            prompt="hi",
            wait_for_result=60,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "completed"
        assert r.response == "the answer"
        assert r.tool_calls == [{"function": {"name": "foo"}}]
        mock_waiter.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wait_clamps_above_maximum(self, mock_queue, mock_waiter, mock_model):
        """wait_for_result values above the cap are clamped."""
        await RunSubSessionTool()._execute(
            user_id="alice",
            session=_session("alice"),
            prompt="hi",
            wait_for_result=MAX_SUB_SESSION_WAIT_SECONDS + 999,
        )
        mock_waiter.assert_awaited_once()
        assert mock_waiter.await_args.kwargs["timeout"] == MAX_SUB_SESSION_WAIT_SECONDS


# ---------------------------------------------------------------------------
# GetSubSessionResultTool
# ---------------------------------------------------------------------------


class TestGetSubSessionResult:
    @pytest.mark.asyncio
    async def test_missing_id_returns_error(self):
        r = await GetSubSessionResultTool()._execute(
            user_id="u", session=_session(), sub_session_id=""
        )
        assert isinstance(r, ErrorResponse)

    @pytest.mark.asyncio
    async def test_unknown_id_returns_error(self, monkeypatch):
        async def none_get(_sid):
            return None

        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.get_chat_session",
            none_get,
        )
        r = await GetSubSessionResultTool()._execute(
            user_id="u", session=_session(), sub_session_id="missing"
        )
        assert isinstance(r, ErrorResponse)
        assert "No sub-session with id missing" in r.message

    @pytest.mark.asyncio
    async def test_other_user_cannot_access(self, monkeypatch):
        """Cross-user lookups are indistinguishable from 'not found'."""
        foreign = MagicMock(user_id="bob", messages=[])

        async def foreign_get(_sid):
            return foreign

        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.get_chat_session",
            foreign_get,
        )
        r = await GetSubSessionResultTool()._execute(
            user_id="alice", session=_session("alice"), sub_session_id="bobs-sess"
        )
        assert isinstance(r, ErrorResponse)
        assert "No sub-session" in r.message

    @pytest.mark.asyncio
    async def test_wait_returns_running(self, monkeypatch, mock_waiter):
        sub = MagicMock(user_id="alice", messages=[])

        async def fake_get(_sid):
            return sub

        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.get_chat_session",
            fake_get,
        )
        mock_waiter.return_value = "running"

        r = await GetSubSessionResultTool()._execute(
            user_id="alice",
            session=_session("alice"),
            sub_session_id="inner-7",
            wait_if_running=30,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "running"
        assert r.sub_session_id == "inner-7"
        mock_waiter.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wait_returns_completed_with_response(self, monkeypatch, mock_waiter):
        """'completed' outcome reads the sub's last assistant message."""
        running_sub = MagicMock(user_id="alice", messages=[])
        final_sub = MagicMock(user_id="alice")
        final_assistant = MagicMock()
        final_assistant.role = "assistant"
        final_assistant.content = "done"
        final_assistant.tool_calls = None
        final_sub.messages = [final_assistant]

        calls = {"n": 0}

        async def fake_get(_sid):
            calls["n"] += 1
            # First call (ownership check) sees no messages; second call
            # (after the waiter returns completed) sees the terminal one.
            return running_sub if calls["n"] == 1 else final_sub

        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.get_chat_session",
            fake_get,
        )
        monkeypatch.setattr(
            "backend.copilot.tools.run_sub_session.get_chat_session",
            fake_get,
        )
        mock_waiter.return_value = "completed"

        r = await GetSubSessionResultTool()._execute(
            user_id="alice",
            session=_session("alice"),
            sub_session_id="inner-3",
            wait_if_running=30,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "completed"
        assert r.response == "done"

    @pytest.mark.asyncio
    async def test_already_terminal_skips_waiter(self, monkeypatch, mock_waiter):
        """If the sub's last message is already terminal, the tool returns
        'completed' without ever calling wait_for_session_completion."""
        sub = MagicMock(user_id="alice")
        assistant = MagicMock()
        assistant.role = "assistant"
        assistant.content = "already done"
        assistant.tool_calls = None
        sub.messages = [assistant]

        async def fake_get(_sid):
            return sub

        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.get_chat_session",
            fake_get,
        )
        monkeypatch.setattr(
            "backend.copilot.tools.run_sub_session.get_chat_session",
            fake_get,
        )

        r = await GetSubSessionResultTool()._execute(
            user_id="alice",
            session=_session("alice"),
            sub_session_id="inner-9",
            wait_if_running=30,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "completed"
        assert r.response == "already done"
        mock_waiter.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cancel_publishes_cancel_event(
        self, monkeypatch, mock_queue, mock_waiter
    ):
        """cancel=true fans out a CancelCoPilotEvent and returns 'cancelled'
        without waiting for the sub to finish (the worker will finalise)."""
        sub = MagicMock(user_id="alice", messages=[])

        async def fake_get(_sid):
            return sub

        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.get_chat_session",
            fake_get,
        )

        r = await GetSubSessionResultTool()._execute(
            user_id="alice",
            session=_session("alice"),
            sub_session_id="inner-5",
            cancel=True,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "cancelled"
        mock_queue["enqueue_cancel"].assert_awaited_once_with("inner-5")
        mock_waiter.assert_not_awaited()
