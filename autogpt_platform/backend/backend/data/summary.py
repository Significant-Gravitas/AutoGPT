import logging
from collections import defaultdict
from datetime import datetime

from prisma.enums import AgentExecutionStatus, CreditTransactionType

from backend.data.credit import get_user_credit_model
from backend.data.execution import get_graph_executions
from backend.data.graph import get_graph_metadata
from backend.server.v2.store.exceptions import DatabaseError
from backend.util.logging import TruncatedLogger

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[SummaryData]")


async def get_user_credits_summary(
    user_id: str, start_time: datetime, end_time: datetime
) -> float:
    """Get total credits used by a user in a time range.

    Returns:
        Total credits used in dollars (not cents)
    """
    try:
        credit_model = get_user_credit_model()
        transaction_history = await credit_model.get_transaction_history(
            user_id=user_id,
            transaction_count_limit=1000,  # Get up to 1000 transactions
            transaction_time_ceiling=end_time,
            transaction_type=CreditTransactionType.USAGE,
        )

        # Calculate total credits used (negative amounts are usage)
        total_credits_used = sum(
            abs(tx.amount)
            for tx in transaction_history.transactions
            if tx.transaction_time >= start_time and tx.amount < 0
        )

        # Convert from cents to dollars
        return total_credits_used / 100

    except Exception as e:
        logger.error(f"Failed to get user credits summary: {e}")
        raise DatabaseError(f"Failed to get user credits summary: {e}") from e


async def get_user_execution_summary(
    user_id: str, start_time: datetime, end_time: datetime
) -> dict:
    """Get execution statistics for a user in a time range.

    Returns:
        Dictionary with total_executions, successful_runs, failed_runs
    """
    try:
        executions = await get_graph_executions(
            user_id=user_id,
            created_time_gte=start_time,
            created_time_lte=end_time,
        )

        total_executions = len(executions)
        successful_runs = len(
            [e for e in executions if e.status == AgentExecutionStatus.COMPLETED]
        )
        failed_runs = len(
            [e for e in executions if e.status == AgentExecutionStatus.FAILED]
        )

        return {
            "total_executions": total_executions,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
        }

    except Exception as e:
        logger.error(f"Failed to get user execution summary: {e}")
        raise DatabaseError(f"Failed to get user execution summary: {e}") from e


async def get_most_used_agent(
    user_id: str, start_time: datetime, end_time: datetime
) -> str:
    """Find the most frequently used agent by a user in a time range.

    Returns:
        Name of the most used agent, or "No agents used" if none
    """
    try:
        executions = await get_graph_executions(
            user_id=user_id,
            created_time_gte=start_time,
            created_time_lte=end_time,
        )

        if not executions:
            return "No agents used"

        # Count usage by graph_id
        agent_usage = defaultdict(int)
        for execution in executions:
            agent_usage[execution.graph_id] += 1

        # Find most used
        most_used_agent_id = max(agent_usage, key=lambda k: agent_usage[k])

        # Get agent name
        try:
            graph_meta = await get_graph_metadata(graph_id=most_used_agent_id)
            return graph_meta.name if graph_meta else "Unknown Agent"
        except Exception:
            logger.warning(f"Could not get metadata for graph {most_used_agent_id}")
            return "Unknown Agent"

    except Exception as e:
        logger.error(f"Failed to get most used agent: {e}")
        raise DatabaseError(f"Failed to get most used agent: {e}") from e


async def get_execution_time_stats(
    user_id: str, start_time: datetime, end_time: datetime
) -> dict:
    """Calculate execution time statistics for a user in a time range.

    Returns:
        Dictionary with total_execution_time, average_execution_time, execution_times list
    """
    try:
        executions = await get_graph_executions(
            user_id=user_id,
            created_time_gte=start_time,
            created_time_lte=end_time,
        )

        execution_times = []
        for execution in executions:
            if execution.stats and execution.stats.duration:
                execution_times.append(execution.stats.duration)

        total_execution_time = sum(execution_times)
        average_execution_time = (
            total_execution_time / len(execution_times) if execution_times else 0
        )

        return {
            "total_execution_time": total_execution_time,
            "average_execution_time": average_execution_time,
            "execution_times": execution_times,
        }

    except Exception as e:
        logger.error(f"Failed to get execution time stats: {e}")
        raise DatabaseError(f"Failed to get execution time stats: {e}") from e


async def get_cost_breakdown_by_agent(
    user_id: str, start_time: datetime, end_time: datetime
) -> dict[str, float]:
    """Get cost breakdown by agent for a user in a time range.

    Returns:
        Dictionary mapping agent names to costs in dollars
    """
    try:
        credit_model = get_user_credit_model()
        transaction_history = await credit_model.get_transaction_history(
            user_id=user_id,
            transaction_count_limit=1000,  # Get up to 1000 transactions
            transaction_time_ceiling=end_time,
            transaction_type=CreditTransactionType.USAGE,
        )

        # Group costs by graph_id
        cost_by_graph_id = defaultdict(float)
        for tx in transaction_history.transactions:
            if (
                tx.transaction_time >= start_time
                and tx.amount < 0
                and tx.usage_graph_id
            ):
                # Convert cents to dollars
                cost_by_graph_id[tx.usage_graph_id] += abs(tx.amount) / 100

        # Convert graph_ids to agent names
        cost_breakdown = {}
        for graph_id, cost in cost_by_graph_id.items():
            try:
                graph_meta = await get_graph_metadata(graph_id=graph_id)
                agent_name = graph_meta.name if graph_meta else f"Agent {graph_id[:8]}"
            except Exception:
                logger.warning(f"Could not get metadata for graph {graph_id}")
                agent_name = f"Agent {graph_id[:8]}"

            cost_breakdown[agent_name] = cost

        return cost_breakdown

    except Exception as e:
        logger.error(f"Failed to get cost breakdown by agent: {e}")
        raise DatabaseError(f"Failed to get cost breakdown by agent: {e}") from e


async def get_user_summary_data(
    user_id: str, start_time: datetime, end_time: datetime
) -> dict:
    """Gather all summary data for a user in a time range.

    This is a convenience function that calls all the other summary functions.

    Returns:
        Dictionary with all summary data needed for notifications
    """
    try:
        # Get all the data in parallel where possible
        credits_summary = await get_user_credits_summary(user_id, start_time, end_time)
        execution_summary = await get_user_execution_summary(
            user_id, start_time, end_time
        )
        most_used_agent = await get_most_used_agent(user_id, start_time, end_time)
        time_stats = await get_execution_time_stats(user_id, start_time, end_time)
        cost_breakdown = await get_cost_breakdown_by_agent(
            user_id, start_time, end_time
        )

        return {
            "total_credits_used": credits_summary,
            "total_executions": execution_summary["total_executions"],
            "successful_runs": execution_summary["successful_runs"],
            "failed_runs": execution_summary["failed_runs"],
            "most_used_agent": most_used_agent,
            "total_execution_time": time_stats["total_execution_time"],
            "average_execution_time": time_stats["average_execution_time"],
            "cost_breakdown": cost_breakdown,
        }

    except Exception as e:
        logger.error(f"Failed to get user summary data: {e}")
        raise DatabaseError(f"Failed to get user summary data: {e}") from e
