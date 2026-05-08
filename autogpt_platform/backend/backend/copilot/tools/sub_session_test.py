"""Tests for run_sub_session + get_sub_session_result (queue-backed flow).

Sub-AutoPilots are enqueued on the copilot_execution RabbitMQ queue and
executed by any copilot_executor worker. The tools wait for completion
by subscribing to ``stream_registry`` for the sub's ChatSession. These
tests patch the three integration seams — ``enqueue_copilot_turn``,
``wait_for_session_result``, and ``stream_registry.create_session``
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
    """Patch the enqueue helpers + the stream-registry session creator at
    the source modules (session_waiter / get_sub_session_result) so tests
    don't need RabbitMQ or Redis. Returns a dict of the mocks so
    individual tests can assert on them.
    """
    enqueue_turn = AsyncMock()
    enqueue_cancel = AsyncMock()
    create_session = AsyncMock()

    # run_sub_session calls enqueue_copilot_turn via session_waiter's
    # run_copilot_turn_via_queue helper — patch at the helper's source.
    monkeypatch.setattr(
        "backend.copilot.sdk.session_waiter.enqueue_copilot_turn",
        enqueue_turn,
    )
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.enqueue_cancel_task",
        enqueue_cancel,
    )
    monkeypatch.setattr(
        "backend.copilot.sdk.session_waiter.stream_registry.create_session",
        create_session,
    )
    return {
        "enqueue_turn": enqueue_turn,
        "enqueue_cancel": enqueue_cancel,
        "create_session": create_session,
    }


@pytest.fixture
def mock_waiter(monkeypatch):
    """Patch the queue-backed primitive and the lightweight waiter so
    tests can drive outcome + result deterministically. Returns the
    ``run_copilot_turn_via_queue`` mock (used by run_sub_session) and
    the ``wait_for_session_result`` mock (used by get_sub_session_result)
    wired to return ``("running", SessionResult())`` by default."""
    from backend.copilot.sdk.session_waiter import SessionResult

    turn_mock = AsyncMock(return_value=("running", SessionResult()))
    result_mock = AsyncMock(return_value=("running", SessionResult()))
    monkeypatch.setattr(
        "backend.copilot.tools.run_sub_session.run_copilot_turn_via_queue",
        turn_mock,
    )
    monkeypatch.setattr(
        "backend.copilot.tools.get_sub_session_result.wait_for_session_result",
        result_mock,
    )
    # Single handle with both attrs for tests that only care about one.
    turn_mock.result_mock = result_mock
    return turn_mock


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
        """The parent's CopilotPermissions must be passed through to the
        queue primitive so the worker applies the same filter."""
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
        mock_waiter.assert_awaited_once()
        assert mock_waiter.await_args.kwargs["permissions"] is perms

    @pytest.mark.asyncio
    async def test_wait_for_result_zero_returns_running(
        self, mock_queue, mock_waiter, mock_model
    ):
        """wait_for_result=0 still dispatches the job (so the sub starts)
        but the primitive returns 'running' immediately because timeout=0,
        and the tool surfaces that to the caller."""
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
        mock_waiter.assert_awaited_once()
        assert mock_waiter.await_args.kwargs["timeout"] == 0

    @pytest.mark.asyncio
    async def test_wait_for_result_completed_returns_final_response(
        self, mock_queue, mock_waiter, mock_model
    ):
        """When the queue primitive returns 'completed' + a SessionResult,
        the tool surfaces response_text + tool_calls directly — no DB
        round-trip needed for the content."""
        from backend.copilot.sdk.session_waiter import SessionResult
        from backend.copilot.sdk.stream_accumulator import ToolCallEntry

        res = SessionResult()
        res.response_text = "the answer"
        res.tool_calls = [
            ToolCallEntry(
                tool_call_id="tc-1",
                tool_name="foo",
                input={"x": 1},
                output="ok",
                success=True,
            )
        ]
        mock_waiter.return_value = ("completed", res)

        r = await RunSubSessionTool()._execute(
            user_id="alice",
            session=_session("alice"),
            prompt="hi",
            wait_for_result=60,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "completed"
        assert r.response == "the answer"
        assert r.tool_calls is not None and len(r.tool_calls) == 1
        assert r.tool_calls[0]["tool_name"] == "foo"
        mock_waiter.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_queued_outcome_surfaces_queued_status(
        self, mock_queue, mock_waiter, mock_model
    ):
        """When the shared primitive reports the target session already has
        a turn running, the tool surfaces ``status='queued'`` so the LLM can
        decide whether to poll or move on."""
        from backend.copilot.sdk.session_waiter import SessionResult

        queued_res = SessionResult(queued=True, pending_buffer_length=2)
        mock_waiter.return_value = ("queued", queued_res)

        r = await RunSubSessionTool()._execute(
            user_id="alice",
            session=_session("alice"),
            prompt="please do another thing",
            wait_for_result=0,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "queued"
        assert r.sub_session_id == "inner-1"
        assert "queued" in (r.message or "").lower()

    @pytest.mark.asyncio
    async def test_wait_clamps_above_maximum(self, mock_queue, mock_waiter, mock_model):
        """wait_for_result values above the cap are clamped before being
        passed to the queue primitive."""
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

        async def no_active_session(_sid):
            return None

        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.get_chat_session",
            fake_get,
        )
        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.stream_registry.get_session",
            no_active_session,
        )

        r = await GetSubSessionResultTool()._execute(
            user_id="alice",
            session=_session("alice"),
            sub_session_id="inner-7",
            wait_if_running=30,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "running"
        assert r.sub_session_id == "inner-7"
        mock_waiter.result_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wait_returns_completed_with_response(self, monkeypatch, mock_waiter):
        """'completed' outcome surfaces the SessionResult directly."""
        from backend.copilot.sdk.session_waiter import SessionResult

        sub = MagicMock(user_id="alice", messages=[])  # not terminal-looking

        async def fake_get(_sid):
            return sub

        async def no_active_session(_sid):
            return None

        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.get_chat_session",
            fake_get,
        )
        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.stream_registry.get_session",
            no_active_session,
        )

        res = SessionResult()
        res.response_text = "done"
        mock_waiter.result_mock.return_value = ("completed", res)

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
        """If the sub's last message is already terminal AND no turn is
        in flight, the tool returns 'completed' without ever calling
        wait_for_session_result — it rebuilds the response from the
        persisted message instead."""
        sub = MagicMock(user_id="alice")
        assistant = MagicMock()
        assistant.role = "assistant"
        assistant.content = "already done"
        assistant.tool_calls = None
        sub.messages = [assistant]

        async def fake_get(_sid):
            return sub

        async def no_active_session(_sid):
            return None

        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.get_chat_session",
            fake_get,
        )
        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.stream_registry.get_session",
            no_active_session,
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
        mock_waiter.result_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resume_turn_in_flight_does_not_return_stale(
        self, monkeypatch, mock_waiter
    ):
        """Regression for sentry r3105409601: on a resumed session whose
        stream_registry status is 'running' (new turn is mid-flight) the
        tool must NOT short-circuit to the prior turn's terminal message.
        It subscribes to the stream like a normal running-session poll."""
        # DB state reflects the PREVIOUS turn's terminal assistant message.
        prior = MagicMock()
        prior.role = "assistant"
        prior.content = "OLD stale result"
        prior.tool_calls = None
        sub = MagicMock(user_id="alice", messages=[prior])

        async def fake_get(_sid):
            return sub

        running_meta = MagicMock(status="running")

        async def active_registry(_sid):
            return running_meta

        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.get_chat_session",
            fake_get,
        )
        monkeypatch.setattr(
            "backend.copilot.tools.get_sub_session_result.stream_registry.get_session",
            active_registry,
        )

        r = await GetSubSessionResultTool()._execute(
            user_id="alice",
            session=_session("alice"),
            sub_session_id="inner-11",
            wait_if_running=30,
        )
        # The waiter must have been awaited — stale short-circuit was skipped.
        mock_waiter.result_mock.assert_awaited_once()
        assert isinstance(r, SubSessionStatusResponse)
        # Default mock_waiter.result_mock.return_value = ("running", SessionResult())
        assert r.status == "running"
        # And crucially NOT the stale content.
        assert r.response is None or r.response == ""

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
        mock_waiter.result_mock.assert_not_awaited()
