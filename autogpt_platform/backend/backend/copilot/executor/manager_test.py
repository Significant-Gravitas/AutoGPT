"""Tests for the engine-switch continuation dispatch in the copilot manager.

The dispatch is the handoff between a finished baseline turn and the
server-initiated SDK continuation turn — the highest-risk link in the
engine-switch flow (see ``backend.copilot.engine_switch``). These tests
pin its retry/give-up contract.
"""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.engine_switch import CONTINUATION_MESSAGE, SwitchRequest

from .manager import (
    _SWITCH_DISPATCH_ATTEMPTS,
    CoPilotExecutor,
    _dispatch_engine_switch_continuation,
    _maybe_dispatch_engine_switch,
    _persist_switch_failure_marker,
)

_SWITCH = SwitchRequest(user_id="user-1", organization_id="org-1", team_id=None)


@pytest.fixture(autouse=True)
def mock_chat_session():
    session = SimpleNamespace(
        metadata=SimpleNamespace(
            llm_auth_provider=None,
            llm_credential_id=None,
        )
    )
    with patch(
        "backend.copilot.model.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    ) as mock_get_session:
        yield mock_get_session


def test_dispatch_succeeds_first_try():
    with (
        patch(
            "backend.copilot.executor.manager.schedule_turn", new_callable=AsyncMock
        ) as mock_schedule,
    ):
        _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    assert mock_schedule.await_count == 1
    kwargs = mock_schedule.call_args.kwargs
    assert kwargs["session_id"] == "sess-1"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["organization_id"] == "org-1"
    assert kwargs["message"] == CONTINUATION_MESSAGE
    assert kwargs["is_user_message"] is False
    assert kwargs["mode"] == "extended_thinking"


def test_dispatch_preserves_codex_transport_for_extended_thinking(
    mock_chat_session,
):
    mock_chat_session.return_value.metadata.llm_auth_provider = "codex"
    mock_chat_session.return_value.metadata.llm_credential_id = "cred-codex"
    with patch(
        "backend.copilot.executor.manager.schedule_turn",
        new_callable=AsyncMock,
    ) as mock_schedule:
        _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    kwargs = mock_schedule.call_args.kwargs
    assert kwargs["mode"] == "extended_thinking"
    assert kwargs["llm_auth_provider"] == "codex"
    assert kwargs["llm_credential_id"] == "cred-codex"


def test_dispatch_retries_until_success():
    with (
        patch(
            "backend.copilot.executor.manager.schedule_turn",
            new_callable=AsyncMock,
            side_effect=[RuntimeError("rmq down"), RuntimeError("rmq down"), None],
        ) as mock_schedule,
        patch("backend.copilot.executor.manager.time.sleep") as mock_sleep,
    ):
        _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    assert mock_schedule.await_count == 3
    assert mock_sleep.call_count == 2


def test_dispatch_gives_up_after_bounded_attempts_with_user_visible_marker():
    with (
        patch(
            "backend.copilot.executor.manager.schedule_turn",
            new_callable=AsyncMock,
            side_effect=RuntimeError("rmq down"),
        ) as mock_schedule,
        patch("backend.copilot.executor.manager.time.sleep"),
        patch(
            "backend.copilot.executor.manager._persist_switch_failure_marker"
        ) as mock_marker,
    ):
        _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    assert mock_schedule.await_count == _SWITCH_DISPATCH_ATTEMPTS
    mock_marker.assert_called_once_with("sess-1")


def test_no_failure_marker_on_success():
    with (
        patch("backend.copilot.executor.manager.schedule_turn", new_callable=AsyncMock),
        patch(
            "backend.copilot.executor.manager._persist_switch_failure_marker"
        ) as mock_marker,
    ):
        _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    mock_marker.assert_not_called()


def test_failure_marker_persists_error_row():
    with patch(
        "backend.copilot.model.append_and_save_message", new_callable=AsyncMock
    ) as mock_append:
        _persist_switch_failure_marker("sess-1")

    assert mock_append.await_count == 1
    session_id, message = mock_append.call_args.args
    assert session_id == "sess-1"
    assert message.role == "assistant"
    assert "Could not start the agent-building engine" in message.content


def test_turn_done_with_pending_switch_dispatches_on_success():
    with (
        patch(
            "backend.copilot.executor.manager.engine_switch.pop_switch",
            return_value=_SWITCH,
        ),
        patch("backend.copilot.executor.manager.threading.Thread") as mock_thread,
    ):
        _maybe_dispatch_engine_switch("sess-1", error_msg=None)

    mock_thread.assert_called_once()
    kwargs = mock_thread.call_args.kwargs
    assert kwargs["target"] is _dispatch_engine_switch_continuation
    assert kwargs["args"] == ("sess-1", _SWITCH)
    mock_thread.return_value.start.assert_called_once()


def test_turn_done_with_error_skips_dispatch_but_consumes_switch():
    with (
        patch(
            "backend.copilot.executor.manager.engine_switch.pop_switch",
            return_value=_SWITCH,
        ) as mock_pop,
        patch("backend.copilot.executor.manager.threading.Thread") as mock_thread,
    ):
        _maybe_dispatch_engine_switch("sess-1", error_msg="boom")

    mock_pop.assert_called_once_with("sess-1")
    mock_thread.assert_not_called()


def test_turn_done_without_switch_is_noop():
    with (
        patch(
            "backend.copilot.executor.manager.engine_switch.pop_switch",
            return_value=None,
        ),
        patch("backend.copilot.executor.manager.threading.Thread") as mock_thread,
    ):
        _maybe_dispatch_engine_switch("sess-1", error_msg=None)

    mock_thread.assert_not_called()


def test_codex_runtime_pool_shutdown_runs_once():
    executor = object.__new__(CoPilotExecutor)
    executor._active_tasks_lock_obj = threading.Lock()
    executor._codex_runtime_pool_closed = False
    transport = MagicMock()
    transport.close_runtime_pool = AsyncMock()

    with patch(
        "backend.copilot.executor.manager.get_codex_transport",
        return_value=transport,
    ) as mock_get_transport:
        executor._close_codex_runtime_pool("[test]")
        executor._close_codex_runtime_pool("[test]")

    mock_get_transport.assert_called_once_with()
    transport.close_runtime_pool.assert_awaited_once_with()


def test_cleanup_closes_codex_pool_between_consumers_and_workers():
    executor = object.__new__(CoPilotExecutor)
    executor.active_tasks = {}
    executor._task_locks = {}
    executor._stop_consuming = threading.Event()
    executor._run_client = MagicMock()
    executor._run_thread = MagicMock()
    executor._cancel_thread = None
    executor._executor = MagicMock()
    executor._executor._max_workers = 1
    executor._active_tasks_lock_obj = threading.Lock()
    executor._codex_runtime_pool_closed = False
    worker_future = MagicMock()
    lifecycle = []

    def record_worker_cleanup(_cleanup_worker):
        lifecycle.append("worker_cleanup")
        return worker_future

    executor._executor.submit.side_effect = record_worker_cleanup

    with (
        patch.object(
            executor,
            "_stop_message_consumers",
            side_effect=lambda *_args: lifecycle.append("consumer_stop"),
        ),
        patch.object(
            executor,
            "_close_codex_runtime_pool",
            side_effect=lambda _prefix: lifecycle.append("codex_pool_close"),
        ),
    ):
        executor.cleanup()

    assert lifecycle == ["consumer_stop", "codex_pool_close", "worker_cleanup"]
