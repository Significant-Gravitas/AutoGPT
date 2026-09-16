"""Best-effort activity-event emission from the graph executor.

The executor runs Prisma-less, so writes go through the DatabaseManager
RPC clients, mirroring cost_tracking. Every helper swallows its own
failures: recording the work must never disrupt the execution that
produced it.
"""

import logging
from typing import TYPE_CHECKING, cast

from backend.blocks._base import Block, BlockSchema
from backend.data.activity_event import ActivityEventDraft
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEntry,
    GraphExecutionMeta,
    NodeExecutionEntry,
)
from backend.data.model import GraphExecutionStats

if TYPE_CHECKING:
    from backend.data.db_manager import (
        DatabaseManagerAsyncClient,
        DatabaseManagerClient,
    )

logger = logging.getLogger(__name__)


async def log_node_integration_activity(
    node_exec: NodeExecutionEntry,
    block: Block,
    db_client: "DatabaseManagerAsyncClient",
) -> None:
    """Record one integration.action per credentialed field of a completed node."""
    try:
        if node_exec.execution_context.dry_run:
            return
        input_model = cast(type[BlockSchema], block.input_schema)
        for field_name in input_model.get_credentials_fields():
            cred_data = node_exec.inputs.get(field_name)
            if not cred_data or not isinstance(cred_data, dict):
                continue
            provider = cred_data.get("provider")
            if not cred_data.get("id") or not provider:
                continue
            await db_client.create_activity_event(
                user_id=node_exec.user_id,
                draft=ActivityEventDraft(
                    category="INTEGRATION",
                    event_type="integration.action",
                    title=block.name,
                    provider=provider,
                    graph_exec_id=node_exec.graph_exec_id,
                    node_exec_id=node_exec.node_exec_id,
                    object_id=node_exec.block_id,
                ),
            )
    except Exception:
        logger.warning(
            "Failed to log integration activity for node execution %s",
            node_exec.node_exec_id,
            exc_info=True,
        )


def handle_run_completed(
    db_client: "DatabaseManagerClient",
    graph_exec: GraphExecutionEntry,
    exec_meta: GraphExecutionMeta,
    exec_stats: GraphExecutionStats,
) -> None:
    """Record the run's terminal event, carrying the AI summary when present."""
    try:
        if exec_meta.status not in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
            return
        if exec_stats.is_dry_run:
            return
        completed = exec_meta.status == ExecutionStatus.COMPLETED
        fallback = "Run finished" if completed else "Run failed"
        db_client.create_activity_event(
            user_id=graph_exec.user_id,
            draft=ActivityEventDraft(
                category="RUN",
                event_type="run.completed" if completed else "run.failed",
                title=exec_stats.activity_status or fallback,
                expert_id=exec_meta.expert_id,
                graph_exec_id=graph_exec.graph_exec_id,
                data={"graph_id": graph_exec.graph_id},
            ),
        )
    except Exception:
        logger.warning(
            "Failed to log run completion for %s",
            graph_exec.graph_exec_id,
            exc_info=True,
        )
