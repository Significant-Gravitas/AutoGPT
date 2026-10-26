"""A stop that lands after a node's block has finished must not cut its
bookkeeping short: the post-flight charge, the node's final status and the
run's totals are still written before the run's cleanup returns, and the
cancel is not reported as an error.
"""

import asyncio
import logging
import threading
import uuid
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockType
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.graph import Node
from backend.data.model import GraphExecutionStats
from backend.executor.manager import ExecutionProcessor
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


class _Run:
    """The node's collaborators, with the reconciliation charge held open."""

    def __init__(self, charge_started, charge_answers):
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
    """As in a real stop: the node runs on the node loop, the run's thread
    cancels it with stop(), whose future is done at once, and wait_for_done
    must still hold the cleanup until the node's final status is written."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    charge_answers = threading.Event()

    async def wait_for_answer():
        while not charge_answers.is_set():
            await asyncio.sleep(0.01)

    run = _Run(threading.Event(), wait_for_answer)
    processor = ExecutionProcessor()
    progress = NodeExecutionProgress()
    try:
        with ExitStack() as stack:
            run.patches(stack)
            stack.enter_context(
                patch.object(processor, "_on_node_execution", new=_completed_block)
            )
            future = asyncio.run_coroutine_threadsafe(
                run.node_run(processor, progress), loop
            )
            progress.add_task("node-exec-1", future)
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
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


@pytest.mark.asyncio
async def test_a_stop_while_a_failed_block_writes_its_error_still_settles_it():
    """The block has failed and its error output is being written when the
    stop lands: the write finishes and the node is still charged and failed."""
    processor = ExecutionProcessor()
    error_write_started = asyncio.Event()
    error_write_answers = asyncio.Event()

    async def charge_answers():
        return None

    run = _Run(asyncio.Event(), charge_answers)

    async def write_error(**kwargs):
        error_write_started.set()
        await error_write_answers.wait()

    run.db_client.upsert_execution_output = AsyncMock(side_effect=write_error)

    async def failing_block(**kwargs):
        raise RuntimeError("provider down")
        yield  # pragma: no cover - makes this an async generator

    with ExitStack() as stack:
        run.patches(stack)
        stack.enter_context(
            patch("backend.executor.manager.execute_node", new=failing_block)
        )
        node_run = asyncio.ensure_future(
            run.node_run(processor, NodeExecutionProgress())
        )
        await error_write_started.wait()
        node_run.cancel()
        await asyncio.sleep(0)
        error_write_answers.set()
        await node_run

    run.db_client.upsert_execution_output.assert_awaited_once()
    assert run.charge_calls == 1
    settled = run.settle_call()
    assert settled["status"] == ExecutionStatus.FAILED
    assert settled["stats"]["reconciled_cost_delta"] == 7
    assert run.graph_stats.node_error_count == 1
