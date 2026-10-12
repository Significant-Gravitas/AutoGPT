"""Tests for the Sentry transactions of graph node runs and output routing.

They run against a live SDK client (the ``sentry_transactions`` fixture) and
assert on the transactions it would have sent.
"""

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import sentry_sdk

from backend.blocks._base import BlockType
from backend.data.execution import (
    ExecutionContext,
    ExecutionQueue,
    ExecutionStatus,
    GraphExecutionEntry,
    NodeExecutionEntry,
)
from backend.data.graph import Node
from backend.data.model import GraphExecutionStats, NodeExecutionStats
from backend.executor import node_tracing
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import (
    ExecutionOutputEntry,
    LogMetadata,
    NodeExecutionProgress,
)
from backend.util import tracing
from backend.util.decorator import TimingInfo

ROOT_EXEC_ID = str(uuid.uuid4())
SUB_EXEC_ID = str(uuid.uuid4())

_AI_BLOCK = SimpleNamespace(name="AITextGeneratorBlock", block_type=BlockType.AI)


def _node_exec(graph_exec_id: str = ROOT_EXEC_ID, **context) -> NodeExecutionEntry:
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id=graph_exec_id,
        graph_id="graph-1",
        graph_version=3,
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id="block-1",
        inputs={},
        execution_context=ExecutionContext(
            root_execution_id=context.pop("root_execution_id", graph_exec_id),
            **context,
        ),
    )


def _trace(transaction: dict) -> dict:
    return transaction["contexts"]["trace"]


# ---------- node_transaction ----------


def test_a_node_run_is_a_transaction_named_after_its_block(sentry_transactions):
    with node_tracing.node_transaction(_node_exec(), _AI_BLOCK):
        pass

    [run] = sentry_transactions
    assert run["transaction"] == "AITextGeneratorBlock"
    assert _trace(run)["op"] == "executor.node"
    assert run["tags"] == {"executor.block_type": "AI", "executor.dry_run": "false"}
    data = _trace(run)["data"]
    assert data["executor.graph_exec_id"] == ROOT_EXEC_ID
    assert data["executor.node_exec_id"] == "node-exec-1"
    assert data["executor.block_id"] == "block-1"
    assert data["executor.user_id"] == "user-1"


def test_every_node_run_of_a_run_shares_the_runs_trace(sentry_transactions):
    with node_tracing.node_transaction(_node_exec(), _AI_BLOCK):
        pass
    with node_tracing.node_transaction(
        _node_exec(SUB_EXEC_ID, root_execution_id=ROOT_EXEC_ID), _AI_BLOCK
    ):
        pass

    root_trace = uuid.UUID(ROOT_EXEC_ID).hex
    assert [_trace(t)["trace_id"] for t in sentry_transactions] == [root_trace] * 2


def test_a_node_run_inside_another_is_its_child(sentry_transactions):
    """An orchestrator runs its tool nodes from inside its own node run."""
    with node_tracing.node_transaction(_node_exec(), _AI_BLOCK) as orchestrator:
        with node_tracing.node_transaction(_node_exec(), _AI_BLOCK):
            pass

    tool, _ = sentry_transactions
    assert _trace(tool)["parent_span_id"] == orchestrator.span_id


def test_the_queue_wait_runs_from_when_the_node_was_queued(sentry_transactions):
    node_exec = _node_exec()
    node_exec.queued_at = datetime.now(timezone.utc) - timedelta(seconds=3)
    node_exec.queued_by = "node_output"

    with node_tracing.node_transaction(node_exec, _AI_BLOCK):
        pass

    [run] = sentry_transactions
    assert 3000 <= _trace(run)["data"]["executor.node.queue_wait_ms"] < 60_000
    assert run["tags"]["executor.node.queued_by"] == "node_output"


def test_an_unknown_block_still_gets_a_low_cardinality_name(sentry_transactions):
    with node_tracing.node_transaction(_node_exec(), None):
        pass

    assert sentry_transactions[0]["transaction"] == "UnknownBlock"


# ---------- record_node_result ----------


@pytest.mark.parametrize(
    "status, error, span_status",
    [
        (ExecutionStatus.COMPLETED, None, "ok"),
        (ExecutionStatus.FAILED, ValueError("missing credentials"), "invalid_argument"),
        (ExecutionStatus.FAILED, RuntimeError("provider down"), "internal_error"),
        (ExecutionStatus.TERMINATED, None, "cancelled"),
    ],
)
def test_how_a_node_run_ended_is_recorded(
    sentry_transactions, status, error, span_status
):
    stats = NodeExecutionStats(walltime=1.25)
    stats.error = error

    with node_tracing.node_transaction(_node_exec(), _AI_BLOCK) as span:
        node_tracing.record_node_result(span, status, stats)

    [run] = sentry_transactions
    assert _trace(run)["status"] == span_status
    assert _trace(run)["data"]["executor.node.run_ms"] == 1250
    assert run["tags"]["executor.node.status"] == status.value.lower()


# ---------- ExecutionProcessor ----------


def _db_client() -> MagicMock:
    client = MagicMock()
    client.get_node = AsyncMock(return_value=MagicMock(block=_AI_BLOCK))
    return client


@pytest.mark.asyncio
async def test_on_node_execution_runs_the_node_as_one_transaction(sentry_transactions):
    processor = ExecutionProcessor()

    async def run_block(**kwargs):
        with sentry_sdk.start_span(op="http.client", name="POST /upsert_output"):
            return TimingInfo(cpu_time=0.1, wall_time=2.0), ExecutionStatus.COMPLETED

    processor._on_node_execution = run_block  # type: ignore[method-assign]
    node_exec = _node_exec()
    node_exec.queued_at = datetime.now(timezone.utc)
    node_exec.queued_by = "graph_start"

    with (
        patch("backend.executor.manager.get_block", return_value=_AI_BLOCK),
        patch(
            "backend.executor.manager.get_db_async_client", return_value=_db_client()
        ),
        patch("backend.executor.manager.log_system_credential_cost", new=AsyncMock()),
        patch(
            "backend.executor.manager.activity_events.log_node_integration_activity",
            new=AsyncMock(),
        ),
        patch(
            "backend.executor.manager.billing.charge_reconciled_usage",
            new=AsyncMock(return_value=(0, None)),
        ),
        patch(
            "backend.executor.manager.async_update_node_execution_status",
            new=AsyncMock(),
        ),
        patch(
            "backend.executor.manager.async_update_graph_execution_state",
            new=AsyncMock(),
        ),
    ):
        stats = await processor.on_node_execution(
            node_exec=node_exec,
            node_exec_progress=NodeExecutionProgress(),
            nodes_input_masks=None,
            graph_stats_pair=(GraphExecutionStats(), threading.Lock()),
        )

    assert stats is not None and stats.walltime == 2.0
    [run] = sentry_transactions
    assert run["transaction"] == "AITextGeneratorBlock"
    assert _trace(run)["trace_id"] == uuid.UUID(ROOT_EXEC_ID).hex
    assert _trace(run)["status"] == "ok"
    assert _trace(run)["data"]["executor.node.run_ms"] == 2000
    assert "executor.node.queue_wait_ms" in _trace(run)["data"]
    assert run["tags"]["executor.node.queued_by"] == "graph_start"
    assert [span["op"] for span in run["spans"]] == ["http.client"]


@pytest.mark.asyncio
async def test_routing_an_output_is_a_child_of_the_node_run(sentry_transactions):
    processor = ExecutionProcessor()
    graph_exec = GraphExecutionEntry(
        user_id="user-1",
        graph_exec_id=ROOT_EXEC_ID,
        graph_id="graph-1",
        graph_version=3,
        execution_context=ExecutionContext(root_execution_id=ROOT_EXEC_ID),
    )
    with node_tracing.node_transaction(_node_exec(), _AI_BLOCK) as node_run:
        output = ExecutionOutputEntry(
            node=Node(id="node-1", block_id="block-1"),
            node_exec_id="node-exec-1",
            data=("response", "hello"),
            trace_headers=tracing.trace_propagation_headers(),
        )
    queue: ExecutionQueue[NodeExecutionEntry] = ExecutionQueue()

    with (
        patch("backend.data.graph.get_block", return_value=_AI_BLOCK),
        patch("backend.executor.manager.get_db_async_client"),
        patch(
            "backend.executor.manager._enqueue_next_nodes",
            new=AsyncMock(return_value=[_node_exec(), _node_exec()]),
        ),
    ):
        await processor._process_node_output(
            output=output,
            node_id="node-1",
            graph_exec=graph_exec,
            log_metadata=LogMetadata(
                logger=logging.getLogger("test"),
                user_id="user-1",
                graph_eid=ROOT_EXEC_ID,
                graph_id="graph-1",
                node_eid="node-exec-1",
                node_id="node-1",
                block_name="AITextGeneratorBlock",
            ),
            nodes_input_masks=None,
            execution_queue=queue,
        )

    assert queue.get_or_none() is not None and queue.get_or_none() is not None
    routing = next(
        t for t in sentry_transactions if _trace(t)["op"] == "executor.node.output"
    )
    assert routing["transaction"] == "executor.node.output"
    assert _trace(routing)["trace_id"] == node_run.trace_id
    assert _trace(routing)["parent_span_id"] == node_run.span_id
    assert routing["tags"] == {"executor.block": "AITextGeneratorBlock"}
    data = _trace(routing)["data"]
    assert data["executor.node.output.enqueued_count"] == 2
    assert data["executor.output_name"] == "response"
    assert data["executor.node.output.wait_ms"] >= 0
