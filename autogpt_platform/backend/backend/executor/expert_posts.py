"""Post run results into the owning expert's CoPilot thread.

Runs in the executor's completion path (same spot as the run notification),
so it must never raise and must work Prisma-less — all DB access goes
through the sync DatabaseManagerClient. Dedup is a deterministic message id
per execution; the daily cap keeps a chatty schedule from flooding the
thread (the activity strip is the overflow surface).
"""

import logging
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from backend.data.execution import ExecutionStatus, GraphExecutionEntry
from backend.data.model import GraphExecutionStats
from backend.data.redis_client import get_redis

if TYPE_CHECKING:
    from backend.data.db_manager import DatabaseManagerClient

logger = logging.getLogger(__name__)

_POST_NAMESPACE = uuid.UUID("0b7c8a52-3d1e-4f6a-9c0d-7e5b2a91c4d8")
_DAILY_POST_CAP = 10
_CAP_KEY_TTL_SECONDS = 2 * 24 * 3600
_MAX_ERROR_LENGTH = 500

# Discriminator the frontend keys on to render a WorkCard instead of a raw
# markdown wall. Rides in the message's per-row JSONB metadata bag.
RUN_METADATA_KIND = "expert_run"
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp")
# A string must clear this length to read as a "doc" rather than an incidental
# label; below it stays "unknown" and falls back to the run-details link.
_DOC_MIN_LENGTH = 200


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
    # Sub-graph executions (AgentExecutorBlock) inherit expert_id from the
    # parent context; only the top-level run may post, or one logical run
    # would produce a message (and burn a cap slot) per nested sub-agent.
    if context and context.parent_execution_id is not None:
        return
    if status not in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
        return
    # The key is captured once at admission and reused for release — a UTC
    # midnight rollover between the two must not decrement the new day's
    # counter (which would mint extra slots).
    cap_key = _cap_key(graph_exec.user_id, expert_id)
    if not _under_daily_cap(cap_key):
        logger.info(
            f"Expert #{expert_id} hit the daily thread-post cap; "
            f"run #{graph_exec.graph_exec_id} stays on the activity feed only"
        )
        return

    # The cap slot was consumed by the check above; give it back whenever no
    # message actually lands (failure or retry-dedup), so failed attempts
    # and re-fired completions can't silently eat the day's budget.
    try:
        metadata = db_client.get_graph_metadata(
            graph_exec.graph_id, graph_exec.graph_version
        )
        agent_name = metadata.name if metadata else "your workflow"
        succeeded = status == ExecutionStatus.COMPLETED
        library_agent_id = db_client.get_library_agent_id_by_graph_id(
            graph_exec.user_id, graph_exec.graph_id
        )
        content = build_expert_run_message(
            agent_name=agent_name,
            succeeded=succeeded,
            summary=exec_stats.activity_status,
            error=str(exec_stats.error) if exec_stats.error else None,
            library_agent_id=library_agent_id,
        )
        posted_session = db_client.append_expert_run_message(
            user_id=graph_exec.user_id,
            expert_id=expert_id,
            content=content,
            message_id=str(
                uuid.uuid5(_POST_NAMESPACE, f"run-post:{graph_exec.graph_exec_id}")
            ),
            metadata={
                "kind": RUN_METADATA_KIND,
                "execution_id": graph_exec.graph_exec_id,
                "graph_id": graph_exec.graph_id,
                "library_agent_id": library_agent_id,
                "graph_name": agent_name,
                "status": "completed" if succeeded else "failed",
                "output_type": _resolve_output_type(db_client, graph_exec),
            },
        )
    except Exception:
        _release_cap_slot(cap_key, expert_id)
        raise
    if posted_session is None:
        _release_cap_slot(cap_key, expert_id)


def build_expert_run_message(
    agent_name: str,
    succeeded: bool,
    summary: str | None = None,
    error: str | None = None,
    library_agent_id: str | None = None,
) -> str:
    """The summary and error both derive from workflow output — untrusted
    text that this message replays into the thread's conversation history.
    Both are blockquoted with explicit provenance instead of being emitted
    in the expert's own voice, so scraped "ignore previous instructions"
    content stays attributed to the run rather than reading as assistant
    speech; errors are also capped so one failed run can't persist a
    multi-MB message that reloads on every later turn."""
    link = (
        f"\n\n[View the run](/library/agents/{library_agent_id})"
        if library_agent_id
        else ""
    )
    if succeeded:
        body = (
            f"\n\nThe run's generated summary:\n\n{_quote(summary)}"
            if summary
            else " It completed successfully."
        )
        return f"I just finished a run of **{agent_name}**.{body}{link}"
    detail = (
        f"\n\nThe reported error:\n\n{_quote(_truncate(error, _MAX_ERROR_LENGTH))}"
        if error
        else ""
    )
    return (
        f"My run of **{agent_name}** didn't finish.{detail}\n\n"
        f"I'll try again on the next schedule — if this keeps happening, "
        f"check the workflow's setup in your library.{link}"
    )


def classify_output_type(value: object) -> str:
    """Classify a single run-output value for the typed work viewer.

    Deliberately conservative — only shapes the viewer can actually render
    earn a specific type. Everything else stays ``"unknown"`` and falls back
    to the run-details link.
    """
    if (
        isinstance(value, list)
        and value
        and all(isinstance(row, dict) for row in value)
    ):
        return "table"
    if isinstance(value, str):
        stripped = value.strip()
        if _is_image_url(stripped):
            return "image"
        if len(stripped) >= _DOC_MIN_LENGTH:
            return "doc"
    return "unknown"


def classify_run_output(outputs: Mapping[str, list[Any]]) -> str:
    """Classify a completed run's primary (first non-empty) output pin.

    A pin that emitted a single list-of-dicts value and one that emitted
    several dict rows both read as a ``"table"``.
    """
    for values in outputs.values():
        if not values:
            continue
        return classify_output_type(values[0] if len(values) == 1 else values)
    return "unknown"


def _is_image_url(value: str) -> bool:
    if not value.lower().startswith(("http://", "https://")):
        return False
    return urlsplit(value).path.lower().endswith(_IMAGE_EXTENSIONS)


def _resolve_output_type(
    db_client: "DatabaseManagerClient", graph_exec: GraphExecutionEntry
) -> str:
    """Best-effort output classification; any retrieval failure degrades to
    ``"unknown"`` so a thread post never hinges on fetching run outputs."""
    try:
        execution = db_client.get_graph_execution(
            user_id=graph_exec.user_id,
            execution_id=graph_exec.graph_exec_id,
        )
        return classify_run_output(execution.outputs) if execution else "unknown"
    except Exception as e:
        logger.warning(
            f"Failed to classify output for run #{graph_exec.graph_exec_id}: "
            f"{type(e).__name__}: {e}"
        )
        return "unknown"


def _quote(text: str) -> str:
    return "\n".join(f"> {line}" for line in text.splitlines() or [""])


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}… (truncated)"


def _cap_key(user_id: str, expert_id: str) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"expert-thread-posts:{user_id}:{expert_id}:{today}"


def _release_cap_slot(key: str, expert_id: str) -> None:
    try:
        get_redis().decr(key)
    except Exception as e:
        logger.warning(
            f"Failed to release post-cap slot for expert #{expert_id}: "
            f"{type(e).__name__}: {e}"
        )


def _under_daily_cap(key: str) -> bool:
    """INCR-first so concurrent completions can't slip past the cap; errs
    open on Redis failure (a missed cap beats a silent thread)."""
    try:
        redis = get_redis()
        # The sync client is typed ResponseT (sync|async union); this call
        # path always uses the sync client, which returns int.
        count = cast(int, redis.incr(key))
        redis.expire(key, _CAP_KEY_TTL_SECONDS)
        return count <= _DAILY_POST_CAP
    except Exception as e:
        logger.warning(
            f"Daily post cap check failed for {key}: {type(e).__name__}: {e}"
        )
        return True
