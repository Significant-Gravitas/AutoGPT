"""One user's top-level runs hold at most their share of an executor's workers."""

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from backend.data.execution import ExecutionContext, GraphExecutionEntry
from backend.executor import manager as manager_module
from backend.executor.manager import ExecutionManager


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch):
    """A four-worker executor whose users may each hold two, with every
    submitted run left in flight until its future is resolved."""
    monkeypatch.setattr(manager_module.settings.config, "num_graph_workers", 4)
    monkeypatch.setattr(manager_module.settings.config, "max_graph_workers_per_user", 2)
    manager = ExecutionManager()
    manager._run_client = MagicMock()
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


def test_a_sub_run_starts_even_when_its_user_holds_their_share(manager):
    # Its parent holds a worker and waits on it; refusing it would deadlock.
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
