"""The runs a member has going in an org, stopped when they leave it."""

import asyncio
import logging

from backend.data import execution as execution_db
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
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
    the org. That includes runs they deleted from their history, as deleting
    a run doesn't stop it. The removal has already happened, so nothing here
    fails it: a lookup or a stop that fails is logged, and the runs the other
    lookup found are still stopped.
    """
    visible, deleted = await asyncio.gather(
        _unfinished_runs(org_id, user_id, deleted=False),
        _unfinished_runs(org_id, user_id, deleted=True),
    )
    await asyncio.gather(
        *[_stop(org_id, user_id, run.id) for run in visible],
        # Lookups skip deleted runs, so a deleted one's stop can only send its
        # cancel, not wait to see it end. That's enough: only a running one
        # would carry on, since a deleted run is never resumed or picked up.
        *[_stop(org_id, user_id, run.id, wait_timeout=0) for run in deleted],
    )


async def _unfinished_runs(
    org_id: str, user_id: str, deleted: bool
) -> list[GraphExecutionMeta]:
    try:
        return await execution_db.get_graph_executions(
            user_id=user_id,
            organization_id=org_id,
            statuses=UNFINISHED_STATUSES,
            deleted=deleted,
        )
    except Exception:
        logger.warning(
            f"Couldn't look up the {'deleted ' if deleted else ''}runs of user "
            f"{user_id}, who was removed from org {org_id}, to stop them",
            exc_info=True,
        )
        return []


async def _stop(
    org_id: str, user_id: str, graph_exec_id: str, wait_timeout: float = 15.0
) -> None:
    try:
        await stop_graph_execution(
            user_id=user_id, graph_exec_id=graph_exec_id, wait_timeout=wait_timeout
        )
    except Exception:
        logger.warning(
            f"Couldn't stop run #{graph_exec_id} of user {user_id}, "
            f"who was removed from org {org_id}",
            exc_info=True,
        )
