"""Close the DelegatedTask receipt when its run finishes.

Sits in the executor's completion path next to ``expert_posts``, so it must
never raise and must work Prisma-less — all DB access goes through the sync
DatabaseManagerClient. Where ``expert_posts`` posts into the expert's *latest*
thread, this posts into the session the delegation was born in, which is the
whole point of carrying ``originSessionId`` on the task.
"""

import logging
import uuid
from typing import TYPE_CHECKING

from backend.copilot.briefing.outcome import DEFAULT_AGENT_NAME
from backend.data.execution import ExecutionStatus, GraphExecutionEntry
from backend.data.model import GraphExecutionStats

if TYPE_CHECKING:
    from backend.data.db_manager import DatabaseManagerClient

logger = logging.getLogger(__name__)

_OUTCOME_NAMESPACE = uuid.UUID("2f4c9d17-8b3a-4e05-9d62-1a7c05f3b8e4")
_MAX_SUMMARY_LENGTH = 500

# Discriminator the frontend keys on to render an inline task card.
TASK_METADATA_KIND = "delegated_task"


def handle_task_outcome(
    db_client: "DatabaseManagerClient",
    graph_exec: GraphExecutionEntry,
    status: ExecutionStatus,
    exec_stats: GraphExecutionStats,
) -> None:
    """Best-effort: a failed receipt close must never affect execution
    handling."""
    try:
        _close_task(db_client, graph_exec, status, exec_stats)
    except Exception as e:
        logger.warning(
            f"Failed to close delegated task for execution "
            f"#{graph_exec.graph_exec_id}: {type(e).__name__}: {e}"
        )


def _close_task(
    db_client: "DatabaseManagerClient",
    graph_exec: GraphExecutionEntry,
    status: ExecutionStatus,
    exec_stats: GraphExecutionStats,
) -> None:
    context = graph_exec.execution_context
    task_id = context.delegated_task_id if context else None
    if not task_id or (context and context.dry_run):
        return
    # Sub-graph executions inherit the task id from the parent context; only
    # the top-level run closes the receipt, or a nested agent would report
    # the delegation done while the real work is still running.
    if context and context.parent_execution_id is not None:
        return
    if status not in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
        return

    succeeded = status == ExecutionStatus.COMPLETED
    summary = _summarize(succeeded, exec_stats)

    # Returns the origin session only when THIS call closed the task, so a
    # cancelled task stays cancelled and a re-fired completion posts nothing.
    session_id = db_client.close_delegated_task(
        user_id=graph_exec.user_id,
        task_id=task_id,
        succeeded=succeeded,
        outcome_summary=summary,
        spend=exec_stats.cost,
    )
    if not session_id:
        return

    metadata = db_client.get_graph_metadata(
        graph_exec.graph_id, graph_exec.graph_version
    )
    agent_name = metadata.name if metadata else DEFAULT_AGENT_NAME
    library_agent_id = db_client.get_library_agent_id_by_graph_id(
        graph_exec.user_id, graph_exec.graph_id
    )
    db_client.append_message_to_session(
        user_id=graph_exec.user_id,
        session_id=session_id,
        content=_build_outcome_message(agent_name, succeeded, summary),
        message_id=str(uuid.uuid5(_OUTCOME_NAMESPACE, f"task-outcome:{task_id}")),
        metadata={
            "kind": TASK_METADATA_KIND,
            "task_id": task_id,
            "execution_id": graph_exec.graph_exec_id,
            "graph_id": graph_exec.graph_id,
            "library_agent_id": library_agent_id,
            "graph_name": agent_name,
            "status": "DONE" if succeeded else "FAILED",
        },
    )


def _summarize(succeeded: bool, exec_stats: GraphExecutionStats) -> str:
    """One line describing how the run ended, capped so a runaway error
    string can't be replayed into every later turn of the thread."""
    raw = exec_stats.activity_status if succeeded else str(exec_stats.error or "")
    text = " ".join(raw.split()) if raw else ""
    if not text:
        return "Finished." if succeeded else "The run did not finish."
    return (
        text if len(text) <= _MAX_SUMMARY_LENGTH else f"{text[:_MAX_SUMMARY_LENGTH]}…"
    )


def _build_outcome_message(agent_name: str, succeeded: bool, summary: str) -> str:
    """The summary derives from workflow output — untrusted text this message
    replays into the thread's history. It is blockquoted with explicit
    provenance, matching ``expert_posts.build_expert_run_message``, so scraped
    "ignore previous instructions" content stays attributed to the run rather
    than reading as assistant speech."""
    quoted = "\n".join(f"> {line}" for line in summary.splitlines() or [""])
    if succeeded:
        return f"**{agent_name}** finished the task.\n\n{quoted}"
    return (
        f"**{agent_name}** couldn't finish the task.\n\n{quoted}\n\n"
        "Tell me how you'd like to proceed and I'll pick it back up."
    )
