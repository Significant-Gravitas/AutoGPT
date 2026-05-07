import logging
from datetime import datetime

from backend.data.db import query_raw_with_schema
from backend.data.graph import get_graph_metadata
from backend.data.model import UserExecutionSummaryStats
from backend.util.exceptions import DatabaseError
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[SummaryData]")


async def get_user_execution_summary_data(
    user_id: str, start_time: datetime, end_time: datetime
) -> UserExecutionSummaryStats:
    """Aggregate per-graph execution stats for a user via grouped SQL.

    Pulls only one summary row per agentGraphId instead of every individual
    AgentGraphExecution: heavy users had thousands of executions per call,
    making this the dominant row-egress query in the prior implementation.
    """
    try:
        rows = await query_raw_with_schema(
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
              AND "createdAt" >= $2
              AND "createdAt" <= $3
            GROUP BY "agentGraphId"
            """,
            user_id,
            start_time,
            end_time,
        )

        total_executions = 0
        successful_runs = 0
        failed_runs = 0
        total_cost_cents = 0.0
        total_walltime = 0.0
        walltime_count = 0
        cost_by_graph_id: dict[str, float] = {}
        usage_by_graph_id: dict[str, int] = {}

        for row in rows:
            graph_id = row["graph_id"]
            count = int(row["total"])
            cost_cents = float(row["total_cost_cents"])

            total_executions += count
            successful_runs += int(row["successful"])
            failed_runs += int(row["failed"])
            total_cost_cents += cost_cents
            total_walltime += float(row["total_walltime"])
            walltime_count += int(row["walltime_count"])

            usage_by_graph_id[graph_id] = count
            cost_by_graph_id[graph_id] = cost_cents / 100

        total_credits_used = total_cost_cents / 100
        average_execution_time = (
            total_walltime / walltime_count if walltime_count else 0
        )

        most_used_agent = "No agents used"
        if usage_by_graph_id:
            most_used_agent_id = max(
                usage_by_graph_id, key=lambda k: usage_by_graph_id[k]
            )
            most_used_agent = await _resolve_agent_name(most_used_agent_id)

        cost_breakdown: dict[str, float] = {}
        for graph_id, cost in cost_by_graph_id.items():
            cost_breakdown[await _resolve_agent_name(graph_id)] = cost

        return UserExecutionSummaryStats(
            total_credits_used=total_credits_used,
            total_executions=total_executions,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            most_used_agent=most_used_agent,
            total_execution_time=total_walltime,
            average_execution_time=average_execution_time,
            cost_breakdown=cost_breakdown,
        )

    except Exception as e:
        logger.error(f"Failed to get user summary data: {e}")
        raise DatabaseError(f"Failed to get user summary data: {e}") from e


async def _resolve_agent_name(graph_id: str) -> str:
    try:
        graph_meta = await get_graph_metadata(graph_id=graph_id)
        return graph_meta.name if graph_meta else f"Agent {graph_id[:8]}"
    except Exception:
        logger.warning(f"Could not get metadata for graph {graph_id}")
        return f"Agent {graph_id[:8]}"
