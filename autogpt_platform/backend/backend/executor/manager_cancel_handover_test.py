"""A cancel the executor takes while it hands a run to a worker reaches that
run: the run is registered before its worker exists (#17182)."""

from concurrent.futures import Future
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from backend.data.execution import ExecutionContext, GraphExecutionEntry
from backend.executor import manager as manager_module
from backend.executor.manager import ExecutionManager
from backend.executor.utils import CancelExecutionEvent

RUN = "run-1"


@pytest.fixture
def cluster_lock() -> MagicMock:
    """The run's cluster-wide lock, which this executor always gets."""
    return MagicMock()


@pytest.fixture
def manager(cluster_lock: MagicMock):
    """An executor whose worker pool is a stand-in the test controls."""
    manager = ExecutionManager()
    manager._run_client = MagicMock()
    # The broker runs these on its connection thread; run them here.
    channel = manager._run_client.get_channel.return_value
    channel.connection.add_callback_threadsafe.side_effect = lambda callback: callback()
    manager._executor = MagicMock()
    cluster_lock.try_acquire.return_value = manager.executor_id
    db = MagicMock()
    db.get_graph_executions_count.return_value = 0
    with (
        patch.object(manager_module, "ClusterLock", return_value=cluster_lock),
        patch.object(manager_module.redis, "get_redis"),
        patch.object(manager_module, "get_db_client", return_value=db),
    ):
        yield manager


def _deliver_run(manager: ExecutionManager) -> None:
    entry = GraphExecutionEntry(
        user_id="alice",
        graph_exec_id=RUN,
        graph_id="graph-1",
        graph_version=1,
        execution_context=ExecutionContext(root_execution_id=RUN),
    )
    manager._handle_run_message(
        MagicMock(), MagicMock(), MagicMock(), entry.model_dump_json().encode()
    )


def _deliver_cancel(manager: ExecutionManager) -> None:
    event = CancelExecutionEvent(graph_exec_id=RUN)
    manager._handle_cancel_message(
        MagicMock(), MagicMock(), MagicMock(), event.model_dump_json().encode()
    )


def test_a_cancel_taken_while_the_run_is_handed_to_a_worker_reaches_it(manager):
    """The cancel consumer runs on its own thread, so a cancel can arrive
    between the hand-over and its return. A stop's second cancel can be the
    only one left by then."""
    handed: list[Event] = []

    def submit(_execute, _entry, cancel_event, _lock):
        _deliver_cancel(manager)
        handed.append(cancel_event)
        return Future()

    manager._executor.submit.side_effect = submit

    _deliver_run(manager)

    assert len(handed) == 1
    assert handed[0].is_set()


def test_a_handed_over_run_is_held_by_its_workers_future(manager):
    """The stand-in registered before the hand-over is replaced, so the run is
    freed when its worker ends."""
    worker: Future = Future()
    manager._executor.submit.side_effect = lambda *_: worker

    _deliver_run(manager)

    assert manager.active_graph_runs[RUN][0] is worker
    worker.set_result(None)
    assert RUN not in manager.active_graph_runs


def test_a_run_no_worker_took_is_left_to_another_executor(
    manager, cluster_lock: MagicMock
):
    """It isn't left registered, which would hold a worker slot of this
    executor forever. Its cluster lock is released and its message requeued,
    so another executor can run it."""
    manager._executor.submit.side_effect = RuntimeError("the pool is shut down")

    _deliver_run(manager)

    assert RUN not in manager.active_graph_runs
    cluster_lock.release.assert_called_once()
    manager._run_client.publish_message.assert_called_once()
