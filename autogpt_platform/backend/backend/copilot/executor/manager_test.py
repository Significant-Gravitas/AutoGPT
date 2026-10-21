"""Tests for the copilot manager: what runs after a turn, and the cancel consumer.

The engine-switch dispatch is the handoff between a finished baseline turn and
the server-initiated SDK continuation turn — the highest-risk link in the
engine-switch flow (see ``backend.copilot.engine_switch``). These tests pin its
retry/give-up contract, and the order it keeps with a turn's deferred work.
"""

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from backend.copilot import after_turn
from backend.copilot.engine_switch import CONTINUATION_MESSAGE, SwitchRequest
from backend.copilot.executor.utils import COPILOT_CANCEL_QUEUE_PREFIX

from .manager import (
    _SWITCH_DISPATCH_ATTEMPTS,
    CoPilotExecutor,
    _dispatch_after_turn,
    _dispatch_engine_switch_continuation,
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


async def test_dispatch_succeeds_first_try():
    with (
        patch(
            "backend.copilot.executor.manager.schedule_turn", new_callable=AsyncMock
        ) as mock_schedule,
    ):
        await _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    assert mock_schedule.await_count == 1
    kwargs = mock_schedule.call_args.kwargs
    assert kwargs["session_id"] == "sess-1"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["organization_id"] == "org-1"
    assert kwargs["message"] == CONTINUATION_MESSAGE
    assert kwargs["is_user_message"] is False
    assert "mode" not in kwargs


async def test_dispatch_preserves_codex_transport_for_engine_switch(
    mock_chat_session,
):
    mock_chat_session.return_value.metadata.llm_auth_provider = "codex"
    mock_chat_session.return_value.metadata.llm_credential_id = "cred-codex"
    with patch(
        "backend.copilot.executor.manager.schedule_turn",
        new_callable=AsyncMock,
    ) as mock_schedule:
        await _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    kwargs = mock_schedule.call_args.kwargs
    assert "mode" not in kwargs
    assert kwargs["llm_auth_provider"] == "codex"
    assert kwargs["llm_credential_id"] == "cred-codex"


@pytest.fixture
def mock_dispatch_sleep():
    mock_sleep = AsyncMock()
    with patch(
        "backend.copilot.executor.manager.asyncio", SimpleNamespace(sleep=mock_sleep)
    ):
        yield mock_sleep


async def test_dispatch_retries_until_success(mock_dispatch_sleep):
    with (
        patch(
            "backend.copilot.executor.manager.schedule_turn",
            new_callable=AsyncMock,
            side_effect=[RuntimeError("rmq down"), RuntimeError("rmq down"), None],
        ) as mock_schedule,
    ):
        await _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    assert mock_schedule.await_count == 3
    assert mock_dispatch_sleep.await_args_list == [call(1), call(2)]


async def test_dispatch_gives_up_after_bounded_attempts_with_user_visible_marker(
    mock_dispatch_sleep,
):
    with (
        patch(
            "backend.copilot.executor.manager.schedule_turn",
            new_callable=AsyncMock,
            side_effect=RuntimeError("rmq down"),
        ) as mock_schedule,
        patch(
            "backend.copilot.executor.manager._persist_switch_failure_marker",
            new_callable=AsyncMock,
        ) as mock_marker,
    ):
        await _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    assert mock_schedule.await_count == _SWITCH_DISPATCH_ATTEMPTS
    assert mock_dispatch_sleep.await_args_list == [call(1), call(2)]
    mock_marker.assert_awaited_once_with("sess-1")


async def test_no_failure_marker_on_success():
    with (
        patch("backend.copilot.executor.manager.schedule_turn", new_callable=AsyncMock),
        patch(
            "backend.copilot.executor.manager._persist_switch_failure_marker",
            new_callable=AsyncMock,
        ) as mock_marker,
    ):
        await _dispatch_engine_switch_continuation("sess-1", _SWITCH)

    mock_marker.assert_not_awaited()


async def test_failure_marker_persists_error_row():
    with patch(
        "backend.copilot.model.append_and_save_message", new_callable=AsyncMock
    ) as mock_append:
        await _persist_switch_failure_marker("sess-1")

    assert mock_append.await_count == 1
    session_id, message = mock_append.call_args.args
    assert session_id == "sess-1"
    assert message.role == "assistant"
    assert "Could not start the agent-building engine" in message.content


async def test_a_finished_turn_starts_its_continuation_before_its_deferred_work():
    """The wake must find the continuation's turn running and leave its cards to
    it; the other order starts two turns for one chat."""
    order: list[str] = []
    after_turn.turn_started("turn-1")
    await after_turn.run_after_turn("turn-1", _recording(order, "wake"))
    threads = _RecordedThreads()

    with (
        patch(
            "backend.copilot.executor.manager.engine_switch.pop_switch",
            return_value=_SWITCH,
        ),
        patch(
            "backend.copilot.executor.manager._dispatch_engine_switch_continuation",
            new=AsyncMock(side_effect=lambda *_: order.append("continuation")),
        ),
        patch("backend.copilot.executor.manager.threading.Thread", new=threads),
    ):
        assert order == []
        _dispatch_after_turn("sess-1", "turn-1", error_msg=None)
        threads.join()

    assert order == ["continuation", "wake"]
    assert after_turn.turn_finished("turn-1") == []


def test_turn_done_with_error_skips_dispatch_but_consumes_switch():
    with (
        patch(
            "backend.copilot.executor.manager.engine_switch.pop_switch",
            return_value=_SWITCH,
        ) as mock_pop,
        patch("backend.copilot.executor.manager.threading.Thread") as mock_thread,
    ):
        _dispatch_after_turn("sess-1", "turn-2", error_msg="boom")

    mock_pop.assert_called_once_with("sess-1")
    mock_thread.assert_not_called()


def test_turn_done_without_switch_or_deferred_work_is_noop():
    with (
        patch(
            "backend.copilot.executor.manager.engine_switch.pop_switch",
            return_value=None,
        ),
        patch("backend.copilot.executor.manager.threading.Thread") as mock_thread,
    ):
        _dispatch_after_turn("sess-1", "turn-3", error_msg=None)

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


def test_cancel_consumer_consumes_the_queue_it_just_declared():
    """Pins the consumer to the per-pod queue the declare helper returned.

    Re-pinning ``basic_consume`` to a fixed name restores the original defect —
    one shared queue, so the broker hands each cancel to one arbitrary pod —
    and no other test in the suite would notice.
    """
    executor = CoPilotExecutor()
    channel = MagicMock()
    executor._cancel_client = MagicMock(is_ready=True)
    executor._cancel_client.get_channel.return_value = channel
    channel.start_consuming.side_effect = lambda: executor.stop_consuming.set()

    CoPilotExecutor._consume_cancel.__wrapped__(executor)

    declared = channel.queue_declare.call_args.kwargs["queue"]
    assert declared.startswith(COPILOT_CANCEL_QUEUE_PREFIX)
    assert channel.basic_consume.call_args.kwargs["queue"] == declared


def _recording(order: list[str], name: str) -> after_turn.Work:
    async def work() -> None:
        order.append(name)

    return work


class _RecordedThreads:
    """Starts real threads, so ``asyncio.run`` never touches the test's loop,
    and joins them."""

    def __init__(self) -> None:
        self.started: list[threading.Thread] = []
        self._thread = threading.Thread

    def __call__(self, *args, **kwargs) -> threading.Thread:
        thread = self._thread(*args, **kwargs)
        self.started.append(thread)
        return thread

    def join(self) -> None:
        for thread in self.started:
            thread.join(timeout=10)
            assert not thread.is_alive()
