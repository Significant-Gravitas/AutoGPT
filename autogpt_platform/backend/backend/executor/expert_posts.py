"""Post run results into the owning expert's CoPilot thread.

Runs in the executor's completion path (same spot as the run notification),
so it must never raise and must work Prisma-less — all DB access goes
through the sync DatabaseManagerClient. Dedup is a deterministic message id
per execution; the daily cap keeps a chatty schedule from flooding the
thread (the activity strip is the overflow surface).
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

from backend.data.execution import ExecutionStatus, GraphExecutionEntry
from backend.data.model import GraphExecutionStats
from backend.data.redis_client import get_redis

if TYPE_CHECKING:
    from backend.data.db_manager import DatabaseManagerClient

logger = logging.getLogger(__name__)

_POST_NAMESPACE = uuid.UUID("0b7c8a52-3d1e-4f6a-9c0d-7e5b2a91c4d8")
_DAILY_POST_CAP = 10
_CAP_KEY_TTL_SECONDS = 2 * 24 * 3600


def handle_expert_run_post(
    db_client: "DatabaseManagerClient",
    graph_exec: GraphExecutionEntry,
    status: ExecutionStatus,
    exec_stats: GraphExecutionStats,
) -> None:
    """Best-effort: a failed post must never affect execution handling."""
    try:
        _post_run_result(db_client, graph_exec, status, exec_stats)
    except Exception as e:
        logger.warning(
            f"Failed to post expert run result for execution "
            f"#{graph_exec.graph_exec_id}: {type(e).__name__}: {e}"
        )


def _post_run_result(
    db_client: "DatabaseManagerClient",
    graph_exec: GraphExecutionEntry,
    status: ExecutionStatus,
    exec_stats: GraphExecutionStats,
) -> None:
    context = graph_exec.execution_context
    expert_id = context.expert_id if context else None
    if not expert_id or (context and context.dry_run):
        return
    if status not in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
        return
    if not _under_daily_cap(graph_exec.user_id, expert_id):
        logger.info(
            f"Expert #{expert_id} hit the daily thread-post cap; "
            f"run #{graph_exec.graph_exec_id} stays on the activity feed only"
        )
        return

    metadata = db_client.get_graph_metadata(
        graph_exec.graph_id, graph_exec.graph_version
    )
    content = build_expert_run_message(
        agent_name=metadata.name if metadata else "your workflow",
        succeeded=status == ExecutionStatus.COMPLETED,
        summary=exec_stats.activity_status,
        error=str(exec_stats.error) if exec_stats.error else None,
        library_agent_id=db_client.get_library_agent_id_by_graph_id(
            graph_exec.user_id, graph_exec.graph_id
        ),
    )
    db_client.append_expert_run_message(
        user_id=graph_exec.user_id,
        expert_id=expert_id,
        content=content,
        message_id=str(
            uuid.uuid5(_POST_NAMESPACE, f"run-post:{graph_exec.graph_exec_id}")
        ),
    )


def build_expert_run_message(
    agent_name: str,
    succeeded: bool,
    summary: str | None = None,
    error: str | None = None,
    library_agent_id: str | None = None,
) -> str:
    link = (
        f"\n\n[View the run](/library/agents/{library_agent_id})"
        if library_agent_id
        else ""
    )
    if succeeded:
        body = summary or "It completed successfully."
        return f"I just finished a run of **{agent_name}**. {body}{link}"
    reason = error or "it hit an error partway through"
    return (
        f"My run of **{agent_name}** didn't finish: {reason}. "
        f"I'll try again on the next schedule — if this keeps happening, "
        f"check the workflow's setup in your library.{link}"
    )


def _under_daily_cap(user_id: str, expert_id: str) -> bool:
    """INCR-first so concurrent completions can't slip past the cap; errs
    open on Redis failure (a missed cap beats a silent thread)."""
    today = datetime.now(timezone.utc).date().isoformat()
    key = f"expert-thread-posts:{user_id}:{expert_id}:{today}"
    try:
        redis = get_redis()
        # The sync client is typed ResponseT (sync|async union); this call
        # path always uses the sync client, which returns int.
        count = cast(int, redis.incr(key))
        redis.expire(key, _CAP_KEY_TTL_SECONDS)
        return count <= _DAILY_POST_CAP
    except Exception as e:
        logger.warning(
            f"Daily post cap check failed for expert #{expert_id}: "
            f"{type(e).__name__}: {e}"
        )
        return True
