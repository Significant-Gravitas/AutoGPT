"""Sentry transactions for graph node runs.

Every node run is one ``executor.node`` transaction named after its block, and
the routing of each of its first outputs (persisting downstream inputs,
queueing the nodes they complete) is one ``executor.node.output`` transaction.
All of a graph execution's transactions, and those of the sub-graphs it
starts, share one trace whose id is the root execution's id without dashes,
so a run's trace can be opened from its id. A node run inside another node's
(an orchestrator's tool) is that run's child.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sentry_sdk.tracing import NoOpSpan, Span

from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionEntry,
    NodeExecutionEntry,
)
from backend.util import tracing

if TYPE_CHECKING:
    from backend.blocks._base import AnyBlockSchema
    from backend.data.model import NodeExecutionStats
    from backend.executor.utils import ExecutionOutputEntry

NODE_OP = "executor.node"
OUTPUT_OP = "executor.node.output"

TRACED_OUTPUTS_PER_NODE_RUN = 20
"""How many of a node run's outputs get their routing traced. A node can yield
thousands (an iterator over a list), and a transaction each would make the
run's tracing cost grow with its output instead of staying bounded."""


def graph_trace_id(graph_exec_id: str, execution_context: ExecutionContext) -> str:
    """The trace shared by a graph execution and the sub-graphs it starts."""
    return tracing.trace_id_for(execution_context.root_execution_id or graph_exec_id)


@contextmanager
def node_transaction(
    node_exec: NodeExecutionEntry, block: "AnyBlockSchema | None"
) -> Iterator[Span]:
    """Run one node execution as an ``executor.node`` transaction."""
    context = node_exec.execution_context
    with tracing.start_transaction(
        NODE_OP,
        block.name if block else "UnknownBlock",
        trace_id=graph_trace_id(node_exec.graph_exec_id, context),
        tags={
            "executor.block_type": block.block_type.value if block else "unknown",
            "executor.dry_run": "true" if context.dry_run else "false",
        },
        data={
            "executor.graph_exec_id": node_exec.graph_exec_id,
            "executor.node_exec_id": node_exec.node_exec_id,
            "executor.graph_id": node_exec.graph_id,
            "executor.graph_version": node_exec.graph_version,
            "executor.node_id": node_exec.node_id,
            "executor.block_id": node_exec.block_id,
            "executor.user_id": node_exec.user_id,
        },
    ) as span:
        if node_exec.queued_at is not None:
            queue_wait = datetime.now(timezone.utc) - node_exec.queued_at
            tracing.set_timing(
                "executor.node.queue_wait_ms",
                queue_wait.total_seconds() * 1000,
                span=span,
            )
        if node_exec.queued_by is not None:
            span.set_tag("executor.node.queued_by", node_exec.queued_by)
        yield span


def record_node_result(
    span: Span, status: ExecutionStatus, stats: "NodeExecutionStats"
) -> None:
    """Record how a node run ended and how long its block ran."""
    tracing.set_timing("executor.node.run_ms", stats.walltime * 1000, span=span)
    span.set_tag("executor.node.status", status.value.lower())
    if status == ExecutionStatus.FAILED:
        # A ValueError is the executor's mark of an expected, user-caused
        # failure (bad input, missing credentials) rather than a platform one.
        failed_by_user = isinstance(stats.error, ValueError)
        tracing.set_status(
            "invalid_argument" if failed_by_user else "internal_error", span=span
        )
    elif status == ExecutionStatus.TERMINATED:
        tracing.set_status("cancelled", span=span)


def output_trace_headers(output_number: int) -> dict[str, str]:
    """The trace the routing of a node run's ``output_number``-th output
    continues: the node run's, for its first ``TRACED_OUTPUTS_PER_NODE_RUN``
    outputs, and none after, so their routing runs untraced."""
    if output_number > TRACED_OUTPUTS_PER_NODE_RUN:
        return {}
    return tracing.trace_propagation_headers()


@contextmanager
def output_transaction(
    output: "ExecutionOutputEntry", graph_exec: GraphExecutionEntry
) -> Iterator[Span]:
    """Run the routing of one node output as an ``executor.node.output``
    transaction, a child of the node run that produced it, when that run
    handed the output its trace (see ``output_trace_headers``)."""
    if not output.trace_headers:
        yield NoOpSpan()
        return
    block = output.node.block
    with tracing.start_transaction(
        OUTPUT_OP,
        OUTPUT_OP,
        trace_headers=output.trace_headers,
        tags={"executor.block": block.name if block else "UnknownBlock"},
        data={
            "executor.graph_exec_id": graph_exec.graph_exec_id,
            "executor.node_exec_id": output.node_exec_id,
            "executor.output_name": output.data[0],
        },
    ) as span:
        tracing.set_timing(
            "executor.node.output.wait_ms",
            (time.monotonic() - output.produced_at) * 1000,
            span=span,
        )
        yield span
