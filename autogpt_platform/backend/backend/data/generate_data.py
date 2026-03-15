import logging
from collections import defaultdict
from datetime import datetime

from prisma.enums import AgentExecutionStatus

from backend.data.execution import get_graph_executions
from backend.data.graph import get_graph_metadata
from backend.data.model import UserExecutionSummaryStats
from backend.util.exceptions import DatabaseError
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[SummaryData]")


async def get_user_execution_summary_data(
    user_id: str, start_time: datetime, end_time: datetime
) -> UserExecutionSummaryStats:
    """Gather all summary data for a user in a time range.

    This function fetches graph executions once and aggregates all required
    statistics in a single pass for efficiency.
    """
    try:
        # Fetch graph executions once
        executions = await get_graph_executions(
            user_id=user_id,
            created_time_gte=start_time,
            created_time_lte=end_time,
        )

        # Initialize aggregation variables
        total_credits_used = 0.0
        total_executions = len(executions)
        successful_runs = 0
        failed_runs = 0
        terminated_runs = 0
        execution_times = []
        agent_usage = defaultdict(int)
        cost_by_graph_id = defaultdict(float)

        # Single pass through executions to aggregate all stats
        for execution in executions:
            # Count execution statuses (including TERMINATED as failed)
            if execution.status == AgentExecutionStatus.COMPLETED:
                successful_runs += 1
            elif execution.status == AgentExecutionStatus.FAILED:
                failed_runs += 1
            elif execution.status == AgentExecutionStatus.TERMINATED:
                terminated_runs += 1

            # Aggregate costs from stats
            if execution.stats and hasattr(execution.stats, "cost"):
                cost_in_dollars = execution.stats.cost / 100
                total_credits_used += cost_in_dollars
                cost_by_graph_id[execution.graph_id] += cost_in_dollars

            # Collect execution times
            if execution.stats and hasattr(execution.stats, "duration"):
                execution_times.append(execution.stats.duration)

            # Count agent usage
            agent_usage[execution.graph_id] += 1

        # Calculate derived stats
        total_execution_time = sum(execution_times)
        average_execution_time = (
            total_execution_time / len(execution_times) if execution_times else 0
        )

        # Find most used agent
        most_used_agent = "No agents used"
        if agent_usage:
            most_used_agent_id = max(agent_usage, key=lambda k: agent_usage[k])
            try:
                graph_meta = await get_graph_metadata(graph_id=most_used_agent_id)
                most_used_agent = (
                    graph_meta.name if graph_meta else f"Agent {most_used_agent_id[:8]}"
                )
            except Exception:
                logger.warning(f"Could not get metadata for graph {most_used_agent_id}")
                most_used_agent = f"Agent {most_used_agent_id[:8]}"

        # Convert graph_ids to agent names for cost breakdown
        cost_breakdown = {}
        for graph_id, cost in cost_by_graph_id.items():
            try:
                graph_meta = await get_graph_metadata(graph_id=graph_id)
                agent_name = graph_meta.name if graph_meta else f"Agent {graph_id[:8]}"
            except Exception:
                logger.warning(f"Could not get metadata for graph {graph_id}")
                agent_name = f"Agent {graph_id[:8]}"
            cost_breakdown[agent_name] = cost

        # Build the summary stats object (include terminated runs as failed)
        return UserExecutionSummaryStats(
            total_credits_used=total_credits_used,
            total_executions=total_executions,
            successful_runs=successful_runs,
            failed_runs=failed_runs + terminated_runs,
            most_used_agent=most_used_agent,
            total_execution_time=total_execution_time,
            average_execution_time=average_execution_time,
            cost_breakdown=cost_breakdown,
        )

    except Exception as e:
        logger.error(f"Failed to get user summary data: {e}")
        raise DatabaseError(f"Failed to get user summary data: {e}") from e
