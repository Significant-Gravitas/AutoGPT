"""A stop that lands after a node's block has finished must not cut its
bookkeeping short: the post-flight charge, the node's final status and the
run's totals are still written before the run's cleanup returns, and the
cancel is not reported as an error.
"""

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockType
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.graph import Node
from backend.data.model import GraphExecutionStats
from backend.executor.manager import ExecutionProcessor, _finish_despite_cancel
from backend.executor.utils import NodeExecutionProgress
from backend.util.decorator import TimingInfo

GRAPH_EXEC_ID = str(uuid.uuid4())

_BLOCK = SimpleNamespace(
    name="AITextGeneratorBlock",
    block_type=BlockType.AI,
    execution_timeout_seconds=None,
)


def _node_exec() -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id=GRAPH_EXEC_ID,
        graph_id="graph-1",
        graph_version=1,
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id="block-1",
        inputs={},
        execution_context=ExecutionContext(root_execution_id=GRAPH_EXEC_ID),
    )


async def _completed_block(**kwargs):
    return TimingInfo(cpu_time=0.1, wall_time=1.0), ExecutionStatus.COMPLETED


async def _failing_block(**kwargs):
    raise RuntimeError("provider down")
    yield  # pragma: no cover - makes this an async generator


async def _block_with_no_outputs(**kwargs):
    return
    yield  # pragma: no cover - makes this an async generator


async def _answers_at_once():
    return None


class _Run:
    """The node's collaborators, with the reconciliation charge held open
    until ``charge_answers`` returns."""

    def __init__(self, charge_started, charge_answers=_answers_at_once):
        self.charge_started = charge_started
        self.charge_answers = charge_answers
        self.db_client = MagicMock(
            get_node=AsyncMock(return_value=Node(id="node-1", block_id="block-1")),
            upsert_execution_output=AsyncMock(),
            get_node_execution=AsyncMock(return_value=None),
        )
        self.charge_calls = 0
        self.update_node = AsyncMock()
        self.update_graph = AsyncMock()
        self.graph_stats = GraphExecutionStats()

    async def charge(self, **kwargs):
        self.charge_calls += 1
        self.charge_started.set()
        await self.charge_answers()
        return 7, None

    def patches(self, stack: ExitStack) -> None:
        for target, kwargs in (
            ("backend.executor.manager.get_block", {"return_value": _BLOCK}),
            ("backend.data.graph.get_block", {"return_value": _BLOCK}),
            (
                "backend.executor.manager.get_db_async_client",
                {"return_value": self.db_client},
            ),
            (
                "backend.executor.manager.log_system_credential_cost",
                {"new": AsyncMock()},
            ),
            (
                "backend.executor.manager.activity_events.log_node_integration_activity",
                {"new": AsyncMock()},
            ),
            (
                "backend.executor.manager.billing.charge_reconciled_usage",
                {"new": self.charge},
            ),
            (
                "backend.executor.manager.async_update_node_execution_status",
                {"new": self.update_node},
            ),
            (
                "backend.executor.manager.async_update_graph_execution_state",
                {"new": self.update_graph},
            ),
        ):
            stack.enter_context(patch(target, **kwargs))

    def settled(self) -> list[dict]:
        """The node's final status writes, as opposed to its RUNNING one."""
        return [
            c.kwargs
            for c in self.update_node.await_args_list
            if c.kwargs["status"] != ExecutionStatus.RUNNING
        ]

    def settle_call(self) -> dict:
        [call] = self.settled()
        return call

    def node_run(self, processor: ExecutionProcessor, progress: NodeExecutionProgress):
        return processor.on_node_execution(
            node_exec=_node_exec(),
            node_exec_progress=progress,
            nodes_input_masks=None,
            graph_stats_pair=(self.graph_stats, threading.Lock()),
        )


@contextmanager
def _node_loop():
    """The executor's node loop on its own thread, as in production, so the
    run's thread stops the node with stop() and wait_for_done()."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


def _until(flag: threading.Event):
    async def wait():
        while not flag.is_set():
            await asyncio.sleep(0.01)

    return wait


def _start_node(loop, run: _Run, processor, progress: NodeExecutionProgress):
    future = asyncio.run_coroutine_threadsafe(run.node_run(processor, progress), loop)
    progress.add_task("node-exec-1", future)
    return future


@pytest.mark.asyncio
async def test_a_stop_during_the_charge_still_settles_the_node(caplog):
    processor = ExecutionProcessor()
    charge_answers = asyncio.Event()
    run = _Run(asyncio.Event(), charge_answers.wait)

    with ExitStack() as stack:
        run.patches(stack)
        stack.enter_context(
            patch.object(processor, "_on_node_execution", new=_completed_block)
        )
        stack.enter_context(caplog.at_level(logging.WARNING))
        node_run = asyncio.ensure_future(
            run.node_run(processor, NodeExecutionProgress())
        )
        await run.charge_started.wait()
        node_run.cancel()
        await asyncio.sleep(0)
        assert not node_run.done()

        charge_answers.set()
        await node_run

    settled = run.settle_call()
    assert settled["status"] == ExecutionStatus.COMPLETED
    assert settled["stats"]["reconciled_cost_delta"] == 7
    run.update_graph.assert_awaited_once()
    assert run.graph_stats.cost == 7
    assert run.graph_stats.node_count == 1
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("cancelled" in r.getMessage() for r in caplog.records)


def test_the_runs_cleanup_waits_for_a_stopped_nodes_bookkeeping():
    """The run's thread cancels the node with stop(), whose future is done at
    once, and wait_for_done must still hold the cleanup until the node's
    final status is written."""
    charge_answers = threading.Event()
    run = _Run(threading.Event(), _until(charge_answers))
    processor = ExecutionProcessor()
    progress = NodeExecutionProgress()
    with _node_loop() as loop, ExitStack() as stack:
        run.patches(stack)
        stack.enter_context(
            patch.object(processor, "_on_node_execution", new=_completed_block)
        )
        future = _start_node(loop, run, processor, progress)
        assert run.charge_started.wait(timeout=5)

        assert progress.stop() == ["node-exec-1"]
        assert future.done()
        with pytest.raises(TimeoutError):
            progress.wait_for_done(timeout=0.3)
        assert not run.settled()

        charge_answers.set()
        progress.wait_for_done(timeout=5)

        assert run.settle_call()["status"] == ExecutionStatus.COMPLETED
        run.update_graph.assert_awaited_once()


def test_the_runs_cleanup_waits_for_a_failed_blocks_error_write():
    """The stop lands while a failed block still writes its error output, and
    the write outlasts the unwind grace: the cleanup still waits for it and
    for the bookkeeping, so the node's FAILED status lands first."""
    error_write_started = threading.Event()
    error_write_answers = threading.Event()
    run = _Run(threading.Event())

    async def write_error(**kwargs):
        error_write_started.set()
        await _until(error_write_answers)()

    run.db_client.upsert_execution_output = AsyncMock(side_effect=write_error)
    processor = ExecutionProcessor()
    progress = NodeExecutionProgress()
    with _node_loop() as loop, ExitStack() as stack:
        run.patches(stack)
        stack.enter_context(
            patch("backend.executor.manager.execute_node", new=_failing_block)
        )
        stack.enter_context(
            patch("backend.executor.utils.NODE_UNWIND_GRACE_SECONDS", 0.1)
        )
        _start_node(loop, run, processor, progress)
        assert error_write_started.wait(timeout=5)

        assert progress.stop() == ["node-exec-1"]
        with pytest.raises(TimeoutError):
            progress.wait_for_done(timeout=0.5)
        assert not run.settled()

        error_write_answers.set()
        progress.wait_for_done(timeout=5)

        assert run.charge_calls == 1
        settled = run.settle_call()
        assert settled["status"] == ExecutionStatus.FAILED
        assert settled["stats"]["reconciled_cost_delta"] == 7
        run.update_graph.assert_awaited_once()


def test_a_block_that_ignores_the_stop_holds_the_cleanup_only_briefly():
    """A block that swallows the cancel must not hold a stopped run past the
    unwind grace, or the stop request would time out."""
    release = threading.Event()
    block_started = threading.Event()

    async def stubborn_block(**kwargs):
        block_started.set()
        while not release.is_set():
            try:
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                pass
        return TimingInfo(cpu_time=0.1, wall_time=1.0), ExecutionStatus.COMPLETED

    run = _Run(threading.Event())
    processor = ExecutionProcessor()
    progress = NodeExecutionProgress()
    with _node_loop() as loop, ExitStack() as stack:
        run.patches(stack)
        stack.enter_context(
            patch.object(processor, "_on_node_execution", new=stubborn_block)
        )
        stack.enter_context(
            patch("backend.executor.utils.NODE_UNWIND_GRACE_SECONDS", 0.3)
        )
        _start_node(loop, run, processor, progress)
        assert block_started.wait(timeout=5)

        progress.stop()
        progress.wait_for_done(timeout=5)

        release.set()
        deadline = time.monotonic() + 5
        while not run.update_graph.await_count and time.monotonic() < deadline:
            time.sleep(0.01)
        run.update_graph.assert_awaited_once()


def test_a_tool_node_run_inline_holds_the_cleanup_only_for_itself():
    """The Orchestrator runs a tool node inline from its own long-lived task,
    tracked by a bare future. Once the tool node is done, a stop's cleanup
    must not also wait for the Orchestrator's task to end."""
    tool_done = threading.Event()
    orchestrator_ends = threading.Event()
    run = _Run(threading.Event())
    processor = ExecutionProcessor()
    progress = NodeExecutionProgress()
    progress.add_task("node-exec-1", Future())

    async def orchestrator():
        await run.node_run(processor, progress)
        tool_done.set()
        await _until(orchestrator_ends)()

    with _node_loop() as loop, ExitStack() as stack:
        run.patches(stack)
        stack.enter_context(
            patch("backend.executor.manager.execute_node", new=_block_with_no_outputs)
        )
        orchestrator_run = asyncio.run_coroutine_threadsafe(orchestrator(), loop)
        assert tool_done.wait(timeout=5)
        assert run.settle_call()["status"] == ExecutionStatus.COMPLETED

        progress.stop()
        progress.wait_for_done(timeout=2)

        orchestrator_ends.set()
        orchestrator_run.result(timeout=5)


def test_the_unwind_grace_is_one_deadline_for_the_whole_cleanup():
    """A node whose block ignores the cancel is waited for until the deadline
    the cleanup passes, not for a fresh grace of its own."""
    loop = asyncio.new_event_loop()
    try:
        progress = NodeExecutionProgress()
        progress.track_unwinding("node-exec-1", loop.create_future())

        with pytest.raises(TimeoutError):
            progress.wait_for_done(timeout=0.3)
        progress.wait_for_done(timeout=5, unwind_until=time.time())
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_finishing_despite_cancel_survives_a_second_cancel():
    answers = asyncio.Event()
    finished = []

    async def work():
        await answers.wait()
        finished.append(True)

    caller = asyncio.ensure_future(_finish_despite_cancel(work()))
    await asyncio.sleep(0)
    for _ in range(2):
        caller.cancel()
        await asyncio.sleep(0)
    assert not caller.done()

    answers.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert finished == [True]
