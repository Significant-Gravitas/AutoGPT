"""Tests for the copilot manager: the Codex runtime pool shutdown and the
cancel consumer."""

import contextlib
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import after_turn
from backend.copilot.executor.utils import COPILOT_CANCEL_QUEUE_PREFIX

from . import cli_warmup, manager
from .manager import CoPilotExecutor


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


@pytest.fixture
def mock_dispatch_sleep():
    mock_sleep = AsyncMock()
    with patch(
        "backend.copilot.executor.manager.asyncio", SimpleNamespace(sleep=mock_sleep)
    ):
        yield mock_sleep


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


class _Started(Exception):
    """Ends ``run()`` at the first step after the CLI warm-up is launched."""


def test_executor_start_does_not_wait_for_the_cli_warm_up(monkeypatch, tmp_path):
    """The executor goes on to set up its workers while the warm-up is still
    reading the CLI binary."""
    reading = threading.Event()
    release = threading.Event()

    def stuck_read(_path: str) -> int:
        reading.set()
        release.wait(30)
        return 0

    # Nothing there to run once the read is released.
    missing = str(tmp_path / "claude")
    monkeypatch.setattr(cli_warmup, "_resolve_cli_path", lambda: missing)
    monkeypatch.setattr(cli_warmup, "_read_through", stuck_read)
    monkeypatch.setattr(manager, "start_http_server", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_load_catalog", lambda: None)
    monkeypatch.setattr(manager, "get_blocks", dict)
    monkeypatch.setattr(manager, "freeze_startup_heap", lambda: None)
    started = threading.Event()

    def worker_setup(_size: int) -> None:
        started.set()
        raise _Started

    monkeypatch.setattr(manager, "pool_size_gauge", MagicMock(set=worker_setup))
    executor = object.__new__(CoPilotExecutor)
    executor.executor_id = "executor-1"
    executor.pool_size = 1
    executor._active_tasks_lock_obj = threading.Lock()

    def run() -> None:
        with contextlib.suppress(_Started):
            executor.run()

    starting = threading.Thread(target=run, daemon=True)
    starting.start()
    try:
        # The warm-up is held until ``release``, so a start that waited for
        # it would not reach the workers' setup in time.
        assert started.wait(5)
        assert reading.wait(5)
    finally:
        release.set()
        starting.join(5)
        for thread in threading.enumerate():
            if thread.name == "copilot-cli-warmup":
                thread.join(5)


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
