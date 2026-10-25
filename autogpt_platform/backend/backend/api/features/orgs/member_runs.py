"""The runs a member has going in an org, stopped when they leave it."""

import asyncio
import logging

from backend.data import execution as execution_db
from backend.data.execution import ExecutionStatus
from backend.executor.utils import stop_graph_execution

logger = logging.getLogger(__name__)

# A run in any of these hasn't finished: it's waiting in the queue or for
# input, running, or paused for a review.
UNFINISHED_STATUSES = [
    ExecutionStatus.QUEUED,
    ExecutionStatus.INCOMPLETE,
    ExecutionStatus.RUNNING,
    ExecutionStatus.REVIEW,
]


async def stop_member_runs(org_id: str, user_id: str) -> None:
    """Stop the runs ``user_id`` started in ``org_id`` that haven't finished.

    Called once they've been removed from the org: a run they started while
    they were a member is stopped there rather than left running, billed to
    the org. A run that fails to stop is logged and doesn't fail the removal.
    """
    runs = await execution_db.get_graph_executions(
        user_id=user_id, organization_id=org_id, statuses=UNFINISHED_STATUSES
    )
    await asyncio.gather(*[_stop(org_id, user_id, run.id) for run in runs])


async def _stop(org_id: str, user_id: str, graph_exec_id: str) -> None:
    try:
        await stop_graph_execution(user_id=user_id, graph_exec_id=graph_exec_id)
    except Exception:
        logger.warning(
            f"Couldn't stop run #{graph_exec_id} of user {user_id}, "
            f"who was removed from org {org_id}",
            exc_info=True,
        )
