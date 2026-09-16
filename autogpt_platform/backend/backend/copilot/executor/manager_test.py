"""Tests for CoPilotExecutor lifecycle unrelated to turn dispatch."""

import threading
from unittest.mock import AsyncMock, MagicMock, patch

from .manager import CoPilotExecutor


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
