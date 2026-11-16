"""A graph run that finishes while the executor's graceful shutdown waits on its
other runs must not end that shutdown, and whatever goes wrong in the wait, the
pool shutdown, lock release and consumer disconnect after it still run.
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from backend.executor.manager import ExecutionManager

RUN_A = "aaaaaaaa-0000-4000-8000-000000000000"
RUN_B = "bbbbbbbb-0000-4000-8000-000000000000"


@pytest.mark.parametrize("finishes_during", ["sweep", "lock_refresh"])
def test_a_run_finishing_mid_wait_does_not_end_the_shutdown(
    finishes_during: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Run B finishes while the wait walks the run map: in its sweep for
    finished runs, or in its refresh of run A's lock."""
    manager, mocks = _stopping_manager()
    run_b: Future = Future()
    lock_a, lock_b = MagicMock(), MagicMock()

    def run_b_finishes() -> None:
        if run_b.done():
            return
        # What run B's done-callback does on its worker thread.
        run_b.set_result(None)
        manager._execution_locks.pop(RUN_B).release()
        manager._cleanup_completed_runs()
        # Run A finishes too, so the wait's next pass can end it.
        run_a.set_result(None)

    run_a = _PolledRun(run_b_finishes if finishes_during == "sweep" else _no_op)
    if finishes_during == "lock_refresh":
        lock_a.refresh.side_effect = run_b_finishes
    manager.active_graph_runs = {
        RUN_A: (run_a, threading.Event()),
        RUN_B: (run_b, threading.Event()),
    }
    manager._execution_locks = {RUN_A: lock_a, RUN_B: lock_b}

    with caplog.at_level(logging.INFO, logger="backend.executor.manager"):
        manager.cleanup()

    assert _logged_exceptions(caplog) == []
    assert run_b.done() and manager.active_graph_runs == {}
    _assert_shutdown_finished(manager, mocks, released=[lock_a])


def test_a_failing_wait_still_releases_the_pods_runs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The wait fails with both runs still going, and run B finishes while
    their locks are being released."""
    manager, mocks = _stopping_manager()
    manager.active_graph_runs = {
        RUN_A: (Future(), threading.Event()),
        RUN_B: (Future(), threading.Event()),
    }
    lock_a, lock_b = MagicMock(), MagicMock()
    lock_a.refresh.side_effect = RuntimeError("refresh failed")
    lock_a.release.side_effect = lambda: manager._execution_locks.pop(RUN_B).release()
    manager._execution_locks = {RUN_A: lock_a, RUN_B: lock_b}

    with caplog.at_level(logging.ERROR, logger="backend.executor.manager"):
        manager.cleanup()

    assert _logged_exceptions(caplog) == ["RuntimeError: refresh failed"]
    _assert_shutdown_finished(manager, mocks, released=[lock_a, lock_b])


class _PolledRun(Future):
    """A run's future that calls ``on_poll`` whenever it is checked for done."""

    def __init__(self, on_poll: Callable[[], None]):
        super().__init__()
        self._on_poll = on_poll

    def done(self) -> bool:
        self._on_poll()
        return super().done()


def _no_op() -> None:
    pass


def _stopping_manager() -> tuple[ExecutionManager, MagicMock]:
    manager = ExecutionManager()
    manager.shutdown_poll_seconds = 0
    mocks = MagicMock()
    manager._executor = mocks.pool
    manager._run_client = mocks.run_client
    manager._cancel_client = mocks.cancel_client
    manager._run_thread = mocks.run_thread
    manager._cancel_thread = mocks.cancel_thread
    mocks.run_thread.is_alive.return_value = False
    mocks.cancel_thread.is_alive.return_value = False
    return manager, mocks


def _logged_exceptions(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        f"{type(r.exc_info[1]).__name__}: {r.exc_info[1]}"
        for r in caplog.records
        if r.exc_info and r.exc_info[1]
    ]


def _assert_shutdown_finished(
    manager: ExecutionManager, mocks: MagicMock, released: list[MagicMock]
) -> None:
    mocks.pool.shutdown.assert_called_once_with(cancel_futures=True, wait=False)
    for lock in released:
        lock.release.assert_called_with()
    assert manager._execution_locks == {}
    mocks.run_client.disconnect.assert_called_once_with()
    mocks.cancel_client.disconnect.assert_called_once_with()
