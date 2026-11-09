"""One user's top-level runs hold at most their share of an executor's workers."""

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from backend.data.execution import ExecutionContext, GraphExecutionEntry
from backend.executor import manager as manager_module
from backend.executor.manager import ExecutionManager
from backend.executor.utils import (
    GRAPH_EXECUTION_DELAY_ROUTING_KEY,
    GRAPH_EXECUTION_EXCHANGE,
)


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch):
    """A four-worker executor whose users may each hold two, with every
    submitted run left in flight until its future is resolved."""
    yield from _manager(monkeypatch, workers=4, share=2)


@pytest.fixture
def small_manager(monkeypatch: pytest.MonkeyPatch):
    """A three-worker executor whose configured share exceeds its pool."""
    yield from _manager(monkeypatch, workers=3, share=5)


def _manager(monkeypatch: pytest.MonkeyPatch, *, workers: int, share: int):
    monkeypatch.setattr(manager_module.settings.config, "num_graph_workers", workers)
    monkeypatch.setattr(
        manager_module.settings.config, "max_graph_workers_per_user", share
    )
    manager = ExecutionManager()
    manager._run_client = MagicMock()
    # The broker runs these on its connection thread; run them here, so a
    # requeue reaches publish_message where the test can read it.
    channel = manager._run_client.get_channel.return_value
    channel.connection.add_callback_threadsafe.side_effect = lambda callback: callback()
    manager._executor = MagicMock()
    manager._executor.submit.side_effect = lambda *_: Future()
    lock = MagicMock()
    lock.try_acquire.return_value = manager.executor_id
    db = MagicMock()
    db.get_graph_executions_count.return_value = 0
    with (
        patch.object(manager_module, "ClusterLock", return_value=lock),
        patch.object(manager_module.redis, "get_redis"),
        patch.object(manager_module, "get_db_client", return_value=db),
    ):
        yield manager


def test_a_users_runs_past_their_share_wait_while_others_still_start(manager):
    for run in ("a1", "a2", "a3"):
        _deliver(manager, user="alice", run=run)
    _deliver(manager, user="bob", run="b1")

    assert set(manager.active_graph_runs) == {"a1", "a2", "b1"}


def test_a_sub_run_starts_while_its_parent_holds_a_worker(manager):
    for run in ("a1", "a2"):
        _deliver(manager, user="alice", run=run)
    _deliver(manager, user="alice", run="a1-child", root="a1")

    assert "a1-child" in manager.active_graph_runs


def test_a_finished_run_frees_its_users_share(manager):
    for run in ("a1", "a2"):
        _deliver(manager, user="alice", run=run)
    manager.active_graph_runs["a1"][0].set_result(None)
    _deliver(manager, user="alice", run="a3")

    assert set(manager.active_graph_runs) == {"a2", "a3"}


def test_one_account_cannot_hold_every_worker_of_a_small_pool(small_manager):
    for run in ("a1", "a2", "a3"):
        _deliver(small_manager, user="alice", run=run)

    assert set(small_manager.active_graph_runs) == {"a1", "a2"}


def test_a_run_refused_for_want_of_a_worker_waits_in_the_delay_queue(manager):
    for run in ("a1", "a2", "a3"):
        _deliver(manager, user="alice", run=run)

    # Republished to the delay queue, which nothing consumes, with the TTL
    # after which the broker dead-letters it back onto the run queue.
    (published,) = manager.run_client.publish_message.call_args_list
    assert published.kwargs["routing_key"] == GRAPH_EXECUTION_DELAY_ROUTING_KEY
    assert published.kwargs["exchange"] == GRAPH_EXECUTION_EXCHANGE
    assert published.kwargs["properties"].expiration == str(
        manager_module.settings.config.graph_run_requeue_delay_seconds * 1000
    )
    assert "a3" in published.kwargs["message"]
    channel = manager.run_client.get_channel.return_value
    channel.basic_nack.assert_called_once()
    assert channel.basic_nack.call_args.kwargs["requeue"] is False


def test_a_run_over_the_per_graph_limit_waits_in_the_delay_queue(manager):
    manager_module.get_db_client().get_graph_executions_count.return_value = 10**6

    _deliver(manager, user="alice", run="a1")

    assert manager.active_graph_runs == {}
    (published,) = manager.run_client.publish_message.call_args_list
    assert published.kwargs["routing_key"] == GRAPH_EXECUTION_DELAY_ROUTING_KEY


def test_a_run_finishing_while_the_share_is_counted_leaves_admission_working(
    manager,
):
    for run in ("a1", "a2"):
        _deliver(manager, user="alice", run=run)
    manager._root_run_owners["a2"] = _OwnerWhoseRunFinishesMidCount(manager, "a1")

    _deliver(manager, user="bob", run="b1")

    assert "b1" in manager.active_graph_runs


def _deliver(
    manager: ExecutionManager, *, user: str, run: str, root: str | None = None
) -> None:
    entry = GraphExecutionEntry(
        user_id=user,
        graph_exec_id=run,
        graph_id=f"graph-{run}",
        graph_version=1,
        execution_context=ExecutionContext(
            root_execution_id=root or run, parent_execution_id=root
        ),
    )
    manager._handle_run_message(
        MagicMock(), MagicMock(), MagicMock(), entry.model_dump_json().encode()
    )


class _OwnerWhoseRunFinishesMidCount(str):
    """Drops another run's owner when compared, as a worker thread's
    done-callback can while the consumer thread counts the share."""

    def __new__(cls, manager: ExecutionManager, finishing_run: str):
        owner = super().__new__(cls, "alice")
        owner.manager = manager
        owner.finishing_run = finishing_run
        return owner

    def __eq__(self, other):
        self.manager._root_run_owners.pop(self.finishing_run, None)
        return str.__eq__(self, other)

    __hash__ = str.__hash__
