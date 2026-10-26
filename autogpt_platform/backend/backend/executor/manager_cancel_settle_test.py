"""A stop that lands after a node's block has finished must not cut its
bookkeeping short: the post-flight charge, the node's final status and the
run's totals are still written, and the cancel is not reported as an error.
"""

import asyncio
import logging
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockType
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.model import GraphExecutionStats
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import NodeExecutionProgress
from backend.util.decorator import TimingInfo

GRAPH_EXEC_ID = str(uuid.uuid4())

_BLOCK = SimpleNamespace(name="AITextGeneratorBlock", block_type=BlockType.AI)


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


@pytest.mark.asyncio
async def test_a_stop_during_the_charge_still_settles_the_node(caplog):
    processor = ExecutionProcessor()

    async def run_block(**kwargs):
        return TimingInfo(cpu_time=0.1, wall_time=1.0), ExecutionStatus.COMPLETED

    processor._on_node_execution = run_block  # type: ignore[method-assign]

    charge_started = asyncio.Event()
    charge_answers = asyncio.Event()

    async def charge(**kwargs):
        charge_started.set()
        await charge_answers.wait()
        return 7, None

    db_client = MagicMock()
    db_client.get_node = AsyncMock(return_value=MagicMock(block=_BLOCK))
    update_node = AsyncMock()
    update_graph = AsyncMock()
    graph_stats = GraphExecutionStats()

    with (
        patch("backend.executor.manager.get_block", return_value=_BLOCK),
        patch("backend.executor.manager.get_db_async_client", return_value=db_client),
        patch("backend.executor.manager.log_system_credential_cost", new=AsyncMock()),
        patch(
            "backend.executor.manager.activity_events.log_node_integration_activity",
            new=AsyncMock(),
        ),
        patch("backend.executor.manager.billing.charge_reconciled_usage", new=charge),
        patch(
            "backend.executor.manager.async_update_node_execution_status",
            new=update_node,
        ),
        patch(
            "backend.executor.manager.async_update_graph_execution_state",
            new=update_graph,
        ),
        caplog.at_level(logging.WARNING),
    ):
        node_run = asyncio.ensure_future(
            processor.on_node_execution(
                node_exec=_node_exec(),
                node_exec_progress=NodeExecutionProgress(),
                nodes_input_masks=None,
                graph_stats_pair=(graph_stats, threading.Lock()),
            )
        )
        await charge_started.wait()
        node_run.cancel()
        await asyncio.sleep(0)
        assert not node_run.done()

        charge_answers.set()
        await node_run

    assert update_node.await_args is not None
    assert update_node.await_args.kwargs["status"] == ExecutionStatus.COMPLETED
    assert update_node.await_args.kwargs["stats"]["reconciled_cost_delta"] == 7
    update_graph.assert_awaited_once()
    assert graph_stats.cost == 7
    assert graph_stats.node_count == 1
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("cancelled" in r.getMessage() for r in caplog.records)
