"""Delivery for the proactive watchers.

Every entry point here is best-effort and returns rather than raises: the
callers are an executor completion hook, a run-start budget gate and a
review upsert, none of which may fail because a chat message didn't land.
That is the same swallow-and-log contract ``expert_posts`` and the morning
briefing already use.

Three guardrails, in order:

* **Flag.** ``copilot-proactive-watchers``, default off. Checked per user.
* **Rate cap.** Per user per day, so a bad night of cron runs can't turn the
  thread into a log file. Deliberately per *user*, not per expert: the point
  is how many unprompted messages a person receives.
* **Dedupe.** A deterministic message id per event; ChatMessage's primary
  key does the rest. Replaying an event is a no-op, not a second card.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, cast

from prisma.enums import TriggerSource
from prisma.models import AgentGraphExecution

from backend.copilot import db as chat_db
from backend.copilot.briefing.outcome import DEFAULT_AGENT_NAME
from backend.data.db_accessors import library_db
from backend.data.redis_client import get_redis_async
from backend.util.feature_flag import Flag, is_feature_enabled

from .events import (
    WATCHER_METADATA_KIND,
    WatcherEvent,
    build_expert_paused_message,
    build_review_waiting_message,
    build_run_failed_message,
    watcher_message_id,
)

logger = logging.getLogger(__name__)

# Unprompted messages are a scarcer budget than run results: eight is a
# working day's worth of "you need to look at this" without the thread
# becoming something the user learns to ignore.
_DAILY_WATCHER_CAP = 8
_CAP_KEY_TTL_SECONDS = 2 * 24 * 3600


async def deliver_run_failed(
    user_id: str,
    expert_id: str,
    graph_exec_id: str,
    graph_id: str,
    trigger_source: TriggerSource,
    error: str | None = None,
) -> bool:
    """Post the "a run failed" card.

    Returns whether the watcher **owns** this failure — i.e. whether the flag
    is on for this user. The executor's legacy failure post falls back only
    on ``False``; a watcher that was enabled but chose not to post (deduped,
    or over the daily cap) still returns ``True``, because "we decided this
    one shouldn't be sent" must not turn into the older message being sent
    instead.

    The agent's name and link are resolved here rather than passed in: the
    caller is the Prisma-less executor, and this is already an RPC hop, so
    resolving them on this side keeps the hook to a single round trip.
    """
    if not await _watchers_enabled(user_id):
        return False
    try:
        name, library_agent_id = await _resolve_agent_ref(user_id, graph_id)
    except Exception as e:
        logger.warning(
            f"Failed to resolve agent for run-failed watcher on "
            f"#{graph_exec_id}: {type(e).__name__}: {e}"
        )
        name, library_agent_id = DEFAULT_AGENT_NAME, None
    await _post(
        user_id=user_id,
        expert_id=expert_id,
        event=WatcherEvent.RUN_FAILED,
        dedupe_key=graph_exec_id,
        content=build_run_failed_message(
            agent_name=name,
            trigger_source=trigger_source,
            error=error,
            library_agent_id=library_agent_id,
        ),
        metadata={
            "execution_id": graph_exec_id,
            "graph_id": graph_id,
            "graph_name": name,
            "library_agent_id": library_agent_id,
            "trigger_source": trigger_source,
        },
    )
    return True


async def deliver_expert_paused(
    user_id: str,
    expert_id: str,
    spent: int,
    budget: int,
) -> bool:
    """Post the "I paused myself on budget" card.

    Returns whether the watcher owns the pause announcement, on the same
    terms as :func:`deliver_run_failed`: ``False`` only when the flag is off,
    so a suppressed-by-dedupe post never resurrects the legacy message.

    The dedupe key is the expert plus the ISO week, matching the budget
    gate's own once-per-week posting rule: a pause that is re-detected by
    the next firing is the same event, not a new one.
    """
    if not await _watchers_enabled(user_id):
        return False
    year, week, _ = datetime.now(timezone.utc).isocalendar()
    await _post(
        user_id=user_id,
        expert_id=expert_id,
        event=WatcherEvent.EXPERT_PAUSED,
        dedupe_key=f"{expert_id}:{year}-W{week:02d}",
        content=build_expert_paused_message(spent=spent, budget=budget),
        metadata={"spent": spent, "budget": budget},
    )
    return True


async def deliver_review_waiting(
    user_id: str,
    graph_exec_id: str,
    node_exec_id: str,
    instructions: str | None = None,
) -> None:
    """Post the "this run is waiting on your decision" card.

    Unlike the other two, the caller (the review upsert) has no expert or
    provenance in scope — ``PendingHumanReview`` carries neither — so both
    are resolved from the execution row here. A run with no expert has no
    thread to be proactive in and is skipped.
    """
    if not await _watchers_enabled(user_id):
        return
    try:
        execution = await AgentGraphExecution.prisma().find_first(
            where={"id": graph_exec_id, "userId": user_id},
        )
        if execution is None or execution.expertId is None:
            return
        name, library_agent_id = await _resolve_agent_ref(
            user_id, execution.agentGraphId
        )
    except Exception as e:
        # The caller is the review upsert itself, on the path that pauses a
        # running graph. Losing the card is survivable; failing the pause is
        # not.
        logger.warning(
            f"Failed to resolve context for review-waiting watcher on "
            f"#{graph_exec_id}: {type(e).__name__}: {e}"
        )
        return
    await _post(
        user_id=user_id,
        expert_id=execution.expertId,
        event=WatcherEvent.REVIEW_WAITING,
        dedupe_key=node_exec_id,
        content=build_review_waiting_message(
            agent_name=name,
            trigger_source=execution.triggerSource,
            instructions=instructions,
            library_agent_id=library_agent_id,
        ),
        metadata={
            "execution_id": graph_exec_id,
            "node_exec_id": node_exec_id,
            "trigger_source": execution.triggerSource,
        },
    )


async def _resolve_agent_ref(user_id: str, graph_id: str) -> tuple[str, str | None]:
    """Display name and library id for the graph, for labelling and linking.
    Degrades to an unlinked default name rather than blocking the card."""
    refs = await library_db().get_library_agent_refs_by_graph_ids(user_id, [graph_id])
    ref = refs[0] if refs else None
    if ref is None:
        return DEFAULT_AGENT_NAME, None
    return ref.name or DEFAULT_AGENT_NAME, ref.id


async def _watchers_enabled(user_id: str) -> bool:
    try:
        return await is_feature_enabled(
            Flag.COPILOT_PROACTIVE_WATCHERS, user_id, default=False
        )
    except Exception as e:
        logger.warning(
            f"Proactive-watcher flag check failed for user #{user_id}; "
            f"treating as off: {type(e).__name__}: {e}"
        )
        return False


async def _post(
    user_id: str,
    expert_id: str,
    event: WatcherEvent,
    dedupe_key: str,
    content: str,
    metadata: dict[str, Any],
) -> None:
    # The key is captured once at admission and reused for release, so a UTC
    # midnight rollover between the two can't decrement the new day's
    # counter and mint the user extra slots.
    cap_key = _cap_key(user_id)
    if not await _under_daily_cap(cap_key):
        logger.info(
            f"User #{user_id} hit the daily proactive-watcher cap; "
            f"{event.value} for {dedupe_key} stays off the thread"
        )
        return
    try:
        posted_session = await chat_db.append_expert_run_message(
            user_id=user_id,
            expert_id=expert_id,
            content=content,
            message_id=watcher_message_id(event, dedupe_key),
            session_id=await chat_db.get_expert_post_session_id(user_id, expert_id),
            metadata={
                "kind": WATCHER_METADATA_KIND,
                "event": event.value,
                **metadata,
            },
        )
    except Exception as e:
        # Give the slot back on every path where no message lands, so failed
        # attempts and replayed events can't silently eat the day's budget.
        await _release_cap_slot(cap_key, user_id)
        logger.warning(
            f"Failed to deliver {event.value} watcher for user #{user_id}: "
            f"{type(e).__name__}: {e}"
        )
        return
    if posted_session is None:
        await _release_cap_slot(cap_key, user_id)


def _cap_key(user_id: str) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"copilot-watchers:{user_id}:{today}"


async def _under_daily_cap(key: str) -> bool:
    """INCR-first so concurrent events can't slip past the cap; errs open on
    Redis failure. Erring open is bounded here in a way it wouldn't be for a
    retry loop: message-id dedupe already guarantees each distinct event
    posts at most once, so the worst case is a burst of genuinely different
    things going wrong — which is exactly when staying quiet is worse."""
    try:
        redis = await get_redis_async()
        # INCR and EXPIRE in one transaction: a failure between them leaves the
        # key with no TTL, which caps that user forever instead of until the
        # date rolls over.
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, _CAP_KEY_TTL_SECONDS)
            results = await cast(Awaitable[list[Any]], pipe.execute())
        return int(results[0]) <= _DAILY_WATCHER_CAP
    except Exception as e:
        logger.warning(
            f"Proactive-watcher cap check failed for {key}: {type(e).__name__}: {e}"
        )
        return True


async def _release_cap_slot(key: str, user_id: str) -> None:
    try:
        redis = await get_redis_async()
        await redis.decr(key)
    except Exception as e:
        logger.warning(
            f"Failed to release watcher cap slot for user #{user_id}: "
            f"{type(e).__name__}: {e}"
        )
