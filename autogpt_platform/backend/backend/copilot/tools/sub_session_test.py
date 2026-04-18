"""Tests for run_sub_session, get_sub_session_result, and the registry."""

import asyncio
import contextlib
from unittest.mock import MagicMock

import pytest

from backend.copilot.sdk.sub_session_registry import (
    MAX_SUB_SESSION_WAIT_SECONDS,
    _reset_for_test,
    get_sub_session,
    notify_shutdown_and_cancel_all,
    prune_finished,
    register_sub_session,
)

from .get_sub_session_result import GetSubSessionResultTool
from .models import ErrorResponse, SubSessionStatusResponse
from .run_sub_session import RunSubSessionTool, _SubAutopilotResult


def _session(user_id: str = "u", session_id: str = "s1") -> MagicMock:
    sess = MagicMock()
    sess.session_id = session_id
    sess.dry_run = False
    return sess


@pytest.fixture(autouse=True)
def _reset():
    _reset_for_test()


# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------


class TestRegistry:
    @pytest.mark.asyncio
    async def test_register_then_lookup_by_owner(self):
        async def hang():
            await asyncio.sleep(60)

        task = asyncio.create_task(hang())
        sid = register_sub_session(task, "alice", "sess", "do thing", "inner-sess-id")

        assert sid.startswith("sub-")
        entry = get_sub_session(sid, "alice")
        assert entry is not None
        assert entry.user_id == "alice"

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_other_user_cannot_lookup(self):
        async def hang():
            await asyncio.sleep(60)

        task = asyncio.create_task(hang())
        sid = register_sub_session(task, "alice", "sess", "do thing", "inner-sess-id")

        assert get_sub_session(sid, "bob") is None

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_prune_cancels_abandoned_running_tasks(self):
        """A sub the agent stopped polling would otherwise live forever in
        the registry. prune_finished enforces a hard age cap for running
        tasks and cancels + evicts them."""
        import time as _time

        from backend.copilot.sdk.sub_session_registry import (
            _sub_sessions,
            prune_finished,
        )

        async def hang():
            await asyncio.sleep(60)

        task = asyncio.create_task(hang())
        await asyncio.sleep(0)
        sid = register_sub_session(task, "alice", "sess", "do thing", "inner-sess-id")

        # Fresh sub — prune should not touch it yet.
        assert prune_finished() == 0
        assert get_sub_session(sid, "alice") is not None

        # Simulate the sub having run for 7h with no polls.
        _sub_sessions[sid].started_at = _time.monotonic() - (7 * 60 * 60)
        assert prune_finished() == 1
        assert get_sub_session(sid, "alice") is None

        # The real task was cancelled as part of the eviction.
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_prune_drops_old_terminal_entries(self):
        async def quick():
            return "done"

        task = asyncio.create_task(quick())
        await task
        sid = register_sub_session(task, "alice", "sess", "done", "inner-sess-id")

        # Newly finished — not yet stale, should NOT prune.
        assert prune_finished() == 0
        assert get_sub_session(sid, "alice") is not None

        # Simulate stale finished_at.
        import time

        from backend.copilot.sdk.sub_session_registry import _sub_sessions

        _sub_sessions[sid].finished_at = time.monotonic() - (60 * 60)
        assert prune_finished() == 1
        assert get_sub_session(sid, "alice") is None

    @pytest.mark.asyncio
    async def test_notify_shutdown_cancels_and_marks_sessions(self, monkeypatch):
        """notify_shutdown_and_cancel_all cancels in-flight tasks AND writes a
        retryable error marker into the sub's ChatSession so the user sees
        what happened when they reopen the conversation."""

        async def hang():
            await asyncio.sleep(60)

        task = asyncio.create_task(hang())
        sid = register_sub_session(
            task, "alice", "parent-sess", "do thing", "inner-sess"
        )

        # Stub the model layer so the test doesn't hit the real DB / Redis.
        from backend.copilot.constants import COPILOT_RETRYABLE_ERROR_PREFIX
        from backend.copilot.model import ChatMessage

        class _InnerSessionStub:
            def __init__(self):
                self.messages: list[ChatMessage] = []

        inner = _InnerSessionStub()
        persisted: list[object] = []

        async def fake_get(session_id: str):
            assert session_id == "inner-sess"
            return inner

        async def fake_upsert(session):
            persisted.append(session)
            return session

        monkeypatch.setattr("backend.copilot.model.get_chat_session", fake_get)
        monkeypatch.setattr("backend.copilot.model.upsert_chat_session", fake_upsert)

        notified = await notify_shutdown_and_cancel_all(reason="test shutdown")

        assert notified == 1
        # Task must be cancelled.
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()
        # Entry must be evicted (shutdown is terminal).
        assert get_sub_session(sid, "alice") is None
        # Inner session must carry a retryable error marker the frontend can render.
        assert len(inner.messages) == 1
        marker = inner.messages[0]
        assert marker.role == "assistant"
        assert marker.content.startswith(COPILOT_RETRYABLE_ERROR_PREFIX)
        assert "test shutdown" in marker.content
        assert persisted == [inner]

    @pytest.mark.asyncio
    async def test_notify_shutdown_is_noop_when_registry_empty(self):
        assert await notify_shutdown_and_cancel_all("nothing here") == 0


# ---------------------------------------------------------------------------
# RunSubSessionTool
# ---------------------------------------------------------------------------


class TestRunSubSession:
    @pytest.fixture(autouse=True)
    def _mock_create_chat_session(self, monkeypatch):
        """run_sub_session creates a ChatSession for the sub before spawning
        the task. Unit tests mock this so they don't hit the DB."""
        counter = {"n": 0}

        async def fake_create(user_id: str, *, dry_run: bool):
            counter["n"] += 1
            session = MagicMock()
            session.session_id = f"inner-sub-{counter['n']}"
            return session

        # The tool does a local import from backend.copilot.model, so patch
        # at the source.
        monkeypatch.setattr(
            "backend.copilot.model.create_chat_session",
            fake_create,
        )

    @pytest.mark.asyncio
    async def test_missing_prompt_returns_error(self):
        tool = RunSubSessionTool()
        r = await tool._execute(
            user_id="u",
            session=_session(),
            prompt="",
        )
        assert isinstance(r, ErrorResponse)

    @pytest.mark.asyncio
    async def test_no_user_returns_error(self):
        tool = RunSubSessionTool()
        r = await tool._execute(
            user_id=None,
            session=_session(),
            prompt="hi",
        )
        assert isinstance(r, ErrorResponse)

    @pytest.mark.asyncio
    async def test_resume_with_other_users_session_id_rejected(self, monkeypatch):
        """sub_autopilot_session_id must belong to the caller — prevents
        one user hijacking another user's session via run_sub_session."""
        other_session = MagicMock()
        other_session.user_id = "bob"
        other_session.session_id = "bob-session"

        async def fake_get_chat_session(session_id: str):
            if session_id == "bob-session":
                return other_session
            return None

        monkeypatch.setattr(
            "backend.copilot.model.get_chat_session",
            fake_get_chat_session,
        )

        tool = RunSubSessionTool()
        r = await tool._execute(
            user_id="alice",
            session=_session(),
            prompt="hi",
            sub_autopilot_session_id="bob-session",
            wait_for_result=0,
        )
        assert isinstance(r, ErrorResponse)
        assert "not a session you own" in r.message

    @pytest.mark.asyncio
    async def test_propagates_dry_run_to_sub(self, monkeypatch):
        """Parent session.dry_run is inherited by the created sub — a sub
        spawned inside a dry-run conversation must not silently escalate
        to a real run."""
        captured_kwargs: dict = {}

        async def fake_create(user_id: str, *, dry_run: bool):
            captured_kwargs["dry_run"] = dry_run
            sess = MagicMock()
            sess.session_id = "inner-dry"
            return sess

        # Override the autouse fixture's patch for this test.
        monkeypatch.setattr(
            "backend.copilot.model.create_chat_session",
            fake_create,
        )

        async def fake_run(**_kwargs):
            return _SubAutopilotResult(
                session_id="inner-dry", response_text="", tool_calls=[]
            )

        monkeypatch.setattr(
            "backend.copilot.tools.run_sub_session._run_sub_autopilot",
            fake_run,
        )

        parent = _session()
        parent.dry_run = True

        tool = RunSubSessionTool()
        await tool._execute(
            user_id="alice",
            session=parent,
            prompt="hi",
            wait_for_result=2,
        )
        assert captured_kwargs["dry_run"] is True

    @pytest.mark.asyncio
    async def test_forwards_parent_permissions_to_sub(self, monkeypatch):
        """CopilotPermissions set on the parent context are forwarded to the
        spawned sub so the sub can't escalate past the parent's capability
        filter."""
        from backend.copilot.context import _current_permissions
        from backend.copilot.permissions import CopilotPermissions

        captured_permissions: list = []

        async def fake_run(**kwargs):
            captured_permissions.append(kwargs.get("permissions"))
            return _SubAutopilotResult(
                session_id="inner-1",
                response_text="ok",
                tool_calls=[],
            )

        monkeypatch.setattr(
            "backend.copilot.tools.run_sub_session._run_sub_autopilot",
            fake_run,
        )

        parent_perms = CopilotPermissions(tools=["bash_exec"], tools_exclude=True)
        token = _current_permissions.set(parent_perms)
        try:
            tool = RunSubSessionTool()
            await tool._execute(
                user_id="alice",
                session=_session(),
                prompt="hi",
                wait_for_result=2,
            )
        finally:
            _current_permissions.reset(token)

        assert len(captured_permissions) == 1
        assert captured_permissions[0] is parent_perms

    @pytest.mark.asyncio
    async def test_wait_for_result_returns_completed(self, monkeypatch):
        async def fake_run(**_kwargs):
            await asyncio.sleep(0.05)
            return _SubAutopilotResult(
                session_id="sub-abc", response_text="hi!", tool_calls=[]
            )

        monkeypatch.setattr(
            "backend.copilot.tools.run_sub_session._run_sub_autopilot",
            fake_run,
        )

        tool = RunSubSessionTool()
        r = await tool._execute(
            user_id="alice",
            session=_session(),
            prompt="hi",
            wait_for_result=5,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "completed"
        assert r.sub_session_id.startswith("sub-")
        assert r.sub_autopilot_session_id == "sub-abc"
        assert r.response == "hi!"

    @pytest.mark.asyncio
    async def test_wait_times_out_returns_running(self, monkeypatch):
        async def fake_run(**_kwargs):
            await asyncio.sleep(60)
            return _SubAutopilotResult(
                session_id="sub-abc", response_text="", tool_calls=[]
            )

        monkeypatch.setattr(
            "backend.copilot.tools.run_sub_session._run_sub_autopilot",
            fake_run,
        )

        tool = RunSubSessionTool()
        r = await tool._execute(
            user_id="alice",
            session=_session(),
            prompt="hi",
            wait_for_result=1,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "running"
        # Sub-session survives — the task is registered and still going.
        entry = get_sub_session(r.sub_session_id, "alice")
        assert entry is not None
        assert not entry.task.done()

        entry.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await entry.task

    @pytest.mark.asyncio
    async def test_clamps_wait_above_maximum(self, monkeypatch):
        """wait_for_result values above MAX_SUB_SESSION_WAIT_SECONDS are capped."""

        collected_wait: list[float] = []

        async def slow_run(**_kwargs):
            # Wait long enough to NEVER finish in the test.
            await asyncio.sleep(60)
            return _SubAutopilotResult(
                session_id="sub-abc", response_text="", tool_calls=[]
            )

        real_wait_for = asyncio.wait_for

        async def tracking_wait_for(awaitable, timeout):
            collected_wait.append(timeout)
            return await real_wait_for(awaitable, timeout=0.1)

        monkeypatch.setattr(
            "backend.copilot.tools.run_sub_session._run_sub_autopilot",
            slow_run,
        )
        monkeypatch.setattr(asyncio, "wait_for", tracking_wait_for)

        tool = RunSubSessionTool()
        r = await tool._execute(
            user_id="alice",
            session=_session(),
            prompt="hi",
            wait_for_result=MAX_SUB_SESSION_WAIT_SECONDS + 999,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert collected_wait and collected_wait[0] == MAX_SUB_SESSION_WAIT_SECONDS

        entry = get_sub_session(r.sub_session_id, "alice")
        if entry is not None:
            entry.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await entry.task


# ---------------------------------------------------------------------------
# GetSubSessionResultTool
# ---------------------------------------------------------------------------


class TestGetSubSessionResult:
    @pytest.mark.asyncio
    async def test_missing_id_returns_error(self):
        tool = GetSubSessionResultTool()
        r = await tool._execute(user_id="alice", session=_session(), sub_session_id="")
        assert isinstance(r, ErrorResponse)

    @pytest.mark.asyncio
    async def test_unknown_id_returns_error(self):
        tool = GetSubSessionResultTool()
        r = await tool._execute(
            user_id="alice",
            session=_session(),
            sub_session_id="sub-nonexistent",
        )
        assert isinstance(r, ErrorResponse)

    @pytest.mark.asyncio
    async def test_other_user_cannot_access(self):
        async def hang():
            await asyncio.sleep(60)

        task = asyncio.create_task(hang())
        sid = register_sub_session(task, "alice", "sess", "do thing", "inner-sess-id")

        tool = GetSubSessionResultTool()
        r = await tool._execute(
            user_id="bob",
            session=_session(),
            sub_session_id=sid,
        )
        assert isinstance(r, ErrorResponse)

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_running_task_wait_returns_still_running(self):
        async def hang():
            await asyncio.sleep(60)

        task = asyncio.create_task(hang())
        sid = register_sub_session(task, "alice", "sess", "do thing", "inner-sess-id")

        tool = GetSubSessionResultTool()
        r = await tool._execute(
            user_id="alice",
            session=_session(),
            sub_session_id=sid,
            wait_if_running=1,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "running"

        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_completed_task_returns_result(self):
        result = _SubAutopilotResult(
            session_id="sub-test",
            response_text="done!",
            tool_calls=[{"name": "find_agent"}],
        )

        async def quick():
            return result

        task = asyncio.create_task(quick())
        await task
        sid = register_sub_session(task, "alice", "sess", "do thing", "inner-sess-id")

        tool = GetSubSessionResultTool()
        r = await tool._execute(
            user_id="alice",
            session=_session(),
            sub_session_id=sid,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "completed"
        assert r.response == "done!"
        assert r.tool_calls == [{"name": "find_agent"}]

    @pytest.mark.asyncio
    async def test_cancel_kills_task(self):
        observed = asyncio.Event()

        async def hang_until_cancel():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                observed.set()
                raise

        task = asyncio.create_task(hang_until_cancel())
        await asyncio.sleep(0)  # let task start
        sid = register_sub_session(task, "alice", "sess", "do thing", "inner-sess-id")

        tool = GetSubSessionResultTool()
        r = await tool._execute(
            user_id="alice",
            session=_session(),
            sub_session_id=sid,
            cancel=True,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "cancelled"

        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert observed.is_set()

    @pytest.mark.asyncio
    async def test_cancel_after_done_returns_real_result(self):
        """Race guard: if the task finished between lookup and cancel, return
        the real completed result instead of reporting cancelled."""

        result = _SubAutopilotResult(
            session_id="sub-test", response_text="already done", tool_calls=[]
        )

        async def quick():
            return result

        task = asyncio.create_task(quick())
        await task
        sid = register_sub_session(task, "alice", "sess", "do thing", "inner-sess-id")

        tool = GetSubSessionResultTool()
        r = await tool._execute(
            user_id="alice",
            session=_session(),
            sub_session_id=sid,
            cancel=True,
        )
        assert isinstance(r, SubSessionStatusResponse)
        assert r.status == "completed"
        assert r.response == "already done"


class TestCancelRetentionContract:
    """`cancel_sub_session` must keep the entry in the registry so a caller
    that polls again after cancel gets the terminal record, not a "not found"
    (sentry r3105237509)."""

    @pytest.mark.asyncio
    async def test_cancelled_entry_is_retained_until_pruned(self):
        from backend.copilot.sdk.sub_session_registry import cancel_sub_session

        async def hang():
            await asyncio.sleep(60)

        task = asyncio.create_task(hang())
        sid = register_sub_session(task, "alice", "sess", "x", "inner-sess-id")

        assert cancel_sub_session(sid, "alice") is True

        # Let the cancellation propagate so the done-callback records
        # finished_at — the entry should still be here for a late poll.
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()

        still_there = get_sub_session(sid, "alice")
        assert still_there is not None, (
            "cancelled entry was dropped immediately — violates the "
            "terminal-retention contract"
        )
        assert still_there.finished_at is not None
