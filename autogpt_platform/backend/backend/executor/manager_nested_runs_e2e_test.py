"""End to end on one executor with the default worker settings: an account
whose workflows run themselves as sub-workflows and start many sub-workflows
at once holds at most its share of the workers plus one per level of nesting,
another account's run still starts, and every run of the first account still
finishes, none of them waiting for good on a worker its other runs hold.

The broker and the graph workers are simulated around the executor's own
admission. An admitted run starts its sub-workflows as a sub-workflow node
does, through the chain checks `add_graph_execution` applies before queueing
one; a refused start fails that node and its run carries on.
"""

from collections import deque
from concurrent.futures import Future
from itertools import count
from unittest.mock import MagicMock

import pytest

from backend.data.execution import ExecutionContext, GraphExecutionEntry
from backend.executor import manager as manager_module
from backend.executor.manager import ExecutionManager
from backend.executor.nesting import MAX_SUB_WORKFLOW_DEPTH, nested_context
from backend.executor.utils import GRAPH_EXECUTION_DELAY_ROUTING_KEY
from backend.util.exceptions import GraphValidationError
from backend.util.settings import Config

POOL = Config.model_fields["num_graph_workers"].default
SHARE = Config.model_fields["max_graph_workers_per_user"].default
DEEPEST = f"chain-{MAX_SUB_WORKFLOW_DEPTH}"
TOO_DEEP = f"chain-{MAX_SUB_WORKFLOW_DEPTH + 1}"

# The sub-workflows each workflow's nodes start, one entry per node run
WORKFLOWS: dict[str, list[str]] = {
    # Saved before saving one that runs itself was refused
    "runs-itself": ["runs-itself"],
    # Starts twenty sub-workflows at once, each of which starts one more
    "fan-out": ["step"] * 20,
    "step": ["leaf"],
    "leaf": [],
    # One level deeper than sub-workflows may nest
    **{f"chain-{i}": [f"chain-{i + 1}"] for i in range(MAX_SUB_WORKFLOW_DEPTH + 1)},
    TOO_DEEP: [],
    "small": [],
}


def test_one_accounts_nested_runs_leave_workers_for_another_and_all_finish(
    monkeypatch: pytest.MonkeyPatch,
):
    executor = _SimulatedExecutor(monkeypatch)
    for workflow in ["runs-itself", "fan-out", "fan-out", "chain-0"] + ["fan-out"] * 2:
        executor.queue("alice", workflow)

    # Every run admitted is still running, so this is alice at her most
    executor.deliver_queued()
    alice_at_her_most = executor.held("alice")
    executor.queue("bob", "small")
    executor.deliver_queued()

    assert executor.started("bob") == ["small"]
    assert SHARE < alice_at_her_most <= SHARE + MAX_SUB_WORKFLOW_DEPTH < POOL

    executor.run_until_settled()

    assert executor.peak["alice"] <= SHARE + MAX_SUB_WORKFLOW_DEPTH
    assert executor.unfinished() == []
    started = executor.started("alice")
    assert started.count("runs-itself") == 1
    assert started.count("leaf") == 4 * 20
    assert DEEPEST in started and TOO_DEEP not in started
    assert executor.refused_starts == {
        ("runs-itself", "runs-itself"),
        (DEEPEST, TOO_DEEP),
    }


class _Run:
    def __init__(self, entry: GraphExecutionEntry):
        self.entry = entry
        self.future: Future = Future()
        self.children: list[str] = []


class _SimulatedExecutor:
    """An ExecutionManager with its broker and graph workers simulated."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch):
        config = manager_module.settings.config
        monkeypatch.setattr(config, "num_graph_workers", POOL)
        monkeypatch.setattr(config, "max_graph_workers_per_user", SHARE)
        self.manager = ExecutionManager()
        lock = MagicMock()
        lock.try_acquire.return_value = self.manager.executor_id
        db = MagicMock()
        db.get_graph_executions_count.return_value = 0
        monkeypatch.setattr(manager_module, "ClusterLock", MagicMock(return_value=lock))
        monkeypatch.setattr(manager_module.redis, "get_redis", MagicMock())
        monkeypatch.setattr(manager_module, "get_db_client", MagicMock(return_value=db))

        self.manager._run_client = MagicMock()
        channel = self.manager._run_client.get_channel.return_value
        channel.connection.add_callback_threadsafe.side_effect = lambda call: call()
        self.manager._run_client.publish_message.side_effect = self._requeued
        self.manager._executor = MagicMock()
        self.manager._executor.submit.side_effect = self._start

        self.queued: deque[str] = deque()
        self.delayed: list[str] = []
        self.runs: dict[str, _Run] = {}
        self.refused_starts: set[tuple[str, str]] = set()
        self.peak: dict[str, int] = {}
        self._ids = count(1)
        self._progress = True

    def queue(self, user: str, workflow: str) -> None:
        run_id = f"run-{next(self._ids)}"
        context = ExecutionContext(
            user_id=user,
            graph_id=workflow,
            graph_exec_id=run_id,
            root_execution_id=run_id,
        )
        self.queued.append(_message(user, run_id, workflow, context))

    def deliver_queued(self) -> None:
        """Deliver queued runs until each one left has come straight back: a
        run refused for a full pool is queued again rather than delayed."""
        requeued_in_a_row = 0
        while self.queued and requeued_in_a_row < len(self.queued):
            runs, queued = len(self.runs), len(self.queued)
            body = self.queued.popleft().encode()
            self.manager._handle_run_message(MagicMock(), MagicMock(), None, body)
            came_back = len(self.runs) == runs and len(self.queued) == queued
            requeued_in_a_row = requeued_in_a_row + 1 if came_back else 0

    def run_until_settled(self) -> None:
        for _ in range(100_000):
            self.deliver_queued()
            if run := self._finishable():
                run.future.set_result(None)
                self._progress = True
            elif self.delayed:
                assert self._progress, f"runs wait for good: {self.unfinished()}"
                self._progress = False
                # The delay queue's hold has passed
                self.queued.extend(self.delayed)
                self.delayed.clear()
            else:
                return
        raise AssertionError("the runs never settled")

    def held(self, user: str) -> int:
        return sum(
            1
            for run in self.runs.values()
            if run.entry.user_id == user and not run.future.done()
        )

    def started(self, user: str) -> list[str]:
        return [
            run.entry.graph_id
            for run in self.runs.values()
            if run.entry.user_id == user
        ]

    def unfinished(self) -> list[str]:
        waiting = [
            GraphExecutionEntry.model_validate_json(message).graph_id
            for message in [*self.queued, *self.delayed]
        ]
        running = [
            run.entry.graph_id for run in self.runs.values() if not run.future.done()
        ]
        return waiting + running

    def _start(self, _execute, entry: GraphExecutionEntry, *_args) -> Future:
        run = _Run(entry)
        self.runs[entry.graph_exec_id] = run
        self._progress = True
        user = entry.user_id
        self.peak[user] = max(self.peak.get(user, 0), self.held(user))
        for workflow in WORKFLOWS[entry.graph_id]:
            self._start_sub_workflow(run, workflow)
        return run.future

    def _start_sub_workflow(self, parent: _Run, workflow: str) -> None:
        try:
            context = nested_context(parent.entry.execution_context, workflow)
        except GraphValidationError:
            self.refused_starts.add((parent.entry.graph_id, workflow))
            return
        run_id = f"run-{next(self._ids)}"
        parent.children.append(run_id)
        context = context.model_copy(
            update={
                "graph_id": workflow,
                "graph_exec_id": run_id,
                "parent_execution_id": parent.entry.graph_exec_id,
            }
        )
        self.queued.append(_message(parent.entry.user_id, run_id, workflow, context))

    def _finishable(self) -> _Run | None:
        return next(
            (
                run
                for run in self.runs.values()
                if not run.future.done()
                and all(
                    child in self.runs and self.runs[child].future.done()
                    for child in run.children
                )
            ),
            None,
        )

    def _requeued(self, *, routing_key: str, message: str, **_kwargs) -> None:
        if routing_key == GRAPH_EXECUTION_DELAY_ROUTING_KEY:
            self.delayed.append(message)
        else:
            self.queued.append(message)


def _message(user: str, run_id: str, workflow: str, context: ExecutionContext) -> str:
    return GraphExecutionEntry(
        user_id=user,
        graph_exec_id=run_id,
        graph_id=workflow,
        graph_version=1,
        execution_context=context,
    ).model_dump_json()
