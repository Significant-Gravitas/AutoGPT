import asyncio
import logging
from datetime import datetime

from pydantic import BaseModel, Field

from backend.data.db import query_raw_with_schema
from backend.data.graph import get_graph_metadata
from backend.data.model import UserExecutionSummaryStats
from backend.util.exceptions import DatabaseError
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[SummaryData]")


class _ExecutionAggregates(BaseModel):
    total_executions: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_cost_cents: float = 0.0
    total_walltime: float = 0.0
    walltime_count: int = 0
    cost_by_graph_id: dict[str, float] = Field(default_factory=dict)
    usage_by_graph_id: dict[str, int] = Field(default_factory=dict)


async def get_user_execution_summary_data(
    user_id: str, start_time: datetime, end_time: datetime
) -> UserExecutionSummaryStats:
    """Aggregate per-graph execution stats for a user via grouped SQL.

    Pulls only one summary row per agentGraphId instead of every individual
    AgentGraphExecution: heavy users had thousands of executions per call,
    making this the dominant row-egress query in the prior implementation.
    """
    try:
        rows = await _fetch_execution_aggregate_rows(user_id, start_time, end_time)
        agg = _reduce_execution_rows(rows)
        name_by_graph_id = await _resolve_agent_names(list(agg.cost_by_graph_id))
        return _assemble_summary(agg, name_by_graph_id)
    except Exception as e:
        logger.error(f"Failed to get user summary data: {e}")
        raise DatabaseError(f"Failed to get user summary data: {e}") from e


async def _fetch_execution_aggregate_rows(
    user_id: str, start_time: datetime, end_time: datetime
) -> list[dict]:
    return await query_raw_with_schema(
        """
        SELECT
            "agentGraphId" AS graph_id,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE "executionStatus" = 'COMPLETED') AS successful,
            COUNT(*) FILTER (WHERE "executionStatus" IN ('FAILED', 'TERMINATED')) AS failed,
            COALESCE(SUM(("stats"::jsonb->>'cost')::numeric), 0) AS total_cost_cents,
            COALESCE(SUM(("stats"::jsonb->>'walltime')::numeric), 0) AS total_walltime,
            COUNT(*) FILTER (WHERE ("stats"::jsonb->>'walltime') IS NOT NULL) AS walltime_count
        FROM {schema_prefix}"AgentGraphExecution"
        WHERE "userId" = $1
          AND "isDeleted" = false
          AND "createdAt" >= $2::timestamp
          AND "createdAt" <= $3::timestamp
        GROUP BY "agentGraphId"
        """,
        user_id,
        start_time,
        end_time,
    )


def _reduce_execution_rows(rows: list[dict]) -> _ExecutionAggregates:
    agg = _ExecutionAggregates()
    for row in rows:
        graph_id = row["graph_id"]
        count = int(row["total"])
        cost_cents = float(row["total_cost_cents"])

        agg.total_executions += count
        agg.successful_runs += int(row["successful"])
        agg.failed_runs += int(row["failed"])
        agg.total_cost_cents += cost_cents
        agg.total_walltime += float(row["total_walltime"])
        agg.walltime_count += int(row["walltime_count"])
        agg.usage_by_graph_id[graph_id] = count
        agg.cost_by_graph_id[graph_id] = cost_cents / 100
    return agg


async def _resolve_agent_names(graph_ids: list[str]) -> dict[str, str]:
    resolved = await asyncio.gather(*(_resolve_agent_name(gid) for gid in graph_ids))
    return dict(zip(graph_ids, resolved))


def _assemble_summary(
    agg: _ExecutionAggregates, name_by_graph_id: dict[str, str]
) -> UserExecutionSummaryStats:
    most_used_agent = "No agents used"
    if agg.usage_by_graph_id:
        # Tuple key gives a deterministic tie-break by graph_id when execution
        # counts match, instead of relying on dict insertion order.
        most_used_agent_id = max(
            agg.usage_by_graph_id,
            key=lambda k: (agg.usage_by_graph_id[k], k),
        )
        most_used_agent = name_by_graph_id[most_used_agent_id]

    # Sum on name collisions (two distinct graphs can resolve to the same
    # display name); a plain assign would silently drop one bucket.
    cost_breakdown: dict[str, float] = {}
    for graph_id, cost in agg.cost_by_graph_id.items():
        name = name_by_graph_id[graph_id]
        cost_breakdown[name] = cost_breakdown.get(name, 0.0) + cost

    average_execution_time = (
        agg.total_walltime / agg.walltime_count if agg.walltime_count else 0
    )

    return UserExecutionSummaryStats(
        total_credits_used=agg.total_cost_cents / 100,
        total_executions=agg.total_executions,
        successful_runs=agg.successful_runs,
        failed_runs=agg.failed_runs,
        most_used_agent=most_used_agent,
        total_execution_time=agg.total_walltime,
        average_execution_time=average_execution_time,
        cost_breakdown=cost_breakdown,
    )


async def _resolve_agent_name(graph_id: str) -> str:
    try:
        graph_meta = await get_graph_metadata(graph_id=graph_id)
        return graph_meta.name if graph_meta else f"Agent {graph_id[:8]}"
    except Exception:
        logger.warning(f"Could not get metadata for graph {graph_id}", exc_info=True)
        return f"Agent {graph_id[:8]}"
