import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, cast

from prisma.models import AgentGraph, AgentGraphExecution

from backend.copilot import db as chat_db
from backend.data.db_accessors import library_db
from backend.data.redis_client import get_redis_async
from backend.util.feature_flag import Flag, is_feature_enabled

from .events import (
    WATCHER_METADATA_KIND,
    WatcherEvent,
    build_expert_paused_message,
    build_overflow_message,
    build_review_waiting_message,
    build_run_failed_message,
    run_href,
    watcher_message_id,
)

logger = logging.getLogger(__name__)

_DAILY_WATCHER_CAP = 8
_CAP_KEY_TTL_SECONDS = 2 * 24 * 3600


async def deliver_run_failed(
    user_id: str,
    expert_id: str,
    graph_exec_id: str,
    graph_id: str,
    trigger_source: str,
    error: str | None = None,
) -> bool:
    if not await _watchers_enabled(user_id):
        return False
    try:
        name, library_agent_id = await _resolve_agent_ref(user_id, graph_id)
    except Exception:
        name, library_agent_id = "Workflow", None
    try:
        href = run_href(library_agent_id, graph_exec_id)
        await _post(
            user_id=user_id,
            expert_id=expert_id,
            event=WatcherEvent.RUN_FAILED,
            dedupe_key=graph_exec_id,
            content=build_run_failed_message(name, trigger_source, error),
            metadata={
                "title": f"{name} needs attention",
                "description": "Workflow run failed",
                "action_label": "Open run" if library_agent_id else "Open Home",
                "action_href": href,
                "status": "failed",
            },
        )
    except Exception as error_value:
        logger.warning(
            "Could not deliver workflow failure notice for user %s: %s",
            user_id,
            type(error_value).__name__,
        )
    return True


async def deliver_expert_paused(
    user_id: str,
    expert_id: str,
    expert_name: str,
) -> bool:
    if not await _watchers_enabled(user_id):
        return False
    year, week, _ = datetime.now(timezone.utc).isocalendar()
    try:
        await _post(
            user_id=user_id,
            expert_id=expert_id,
            event=WatcherEvent.EXPERT_PAUSED,
            dedupe_key=f"{expert_id}:{year}-W{week:02d}",
            content=build_expert_paused_message(expert_name),
            metadata={
                "title": f"{expert_name} needs your decision",
                "description": "Weekly work limit reached",
                "action_label": "Open Team",
                "action_href": f"/team?expert={expert_id}",
                "status": "blocked",
            },
        )
    except Exception as error_value:
        logger.warning(
            "Could not deliver expert pause notice for user %s: %s",
            user_id,
            type(error_value).__name__,
        )
    return True


async def deliver_review_waiting(
    user_id: str,
    graph_exec_id: str,
    node_exec_id: str,
    instructions: str | None = None,
) -> bool:
    if not await _watchers_enabled(user_id):
        return False
    try:
        execution = await AgentGraphExecution.prisma().find_first(
            where={"id": graph_exec_id, "userId": user_id}
        )
        if execution is None or execution.expertId is None:
            return True
        try:
            name, library_agent_id = await _resolve_agent_ref(
                user_id, execution.agentGraphId
            )
        except Exception:
            name, library_agent_id = "Workflow", None
        href = run_href(library_agent_id, graph_exec_id)
        await _post(
            user_id=user_id,
            expert_id=execution.expertId,
            event=WatcherEvent.REVIEW_WAITING,
            dedupe_key=node_exec_id,
            content=build_review_waiting_message(name, instructions),
            metadata={
                "title": f"{name} needs your approval",
                "description": "A workflow is waiting for your decision",
                "action_label": "Review run" if library_agent_id else "Open Home",
                "action_href": href,
                "status": "blocked",
            },
        )
    except Exception as error_value:
        logger.warning(
            "Could not deliver workflow review notice for user %s: %s",
            user_id,
            type(error_value).__name__,
        )
    return True


async def _resolve_agent_ref(user_id: str, graph_id: str) -> tuple[str, str | None]:
    refs = await library_db().get_library_agent_refs_by_graph_ids(user_id, [graph_id])
    if refs:
        return refs[0].name or "Workflow", refs[0].id
    graph = await AgentGraph.prisma().find_first(
        where={"id": graph_id, "isActive": True}
    )
    return (graph.name if graph and graph.name else "Workflow"), None


async def _watchers_enabled(user_id: str) -> bool:
    try:
        return await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False)
    except Exception as error_value:
        logger.warning(
            "Expert watcher flag check failed for user %s: %s",
            user_id,
            type(error_value).__name__,
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
    cap_key = _cap_key(user_id)
    if not await _under_daily_cap(cap_key):
        await _post_overflow(user_id, expert_id)
        return
    try:
        posted = await chat_db.append_expert_run_message(
            user_id=user_id,
            expert_id=expert_id,
            content=content,
            message_id=watcher_message_id(event, dedupe_key),
            metadata={"kind": WATCHER_METADATA_KIND, "event": event.value, **metadata},
        )
    except Exception:
        await _release_cap_slot(cap_key)
        raise
    if posted is None:
        await _release_cap_slot(cap_key)


async def _post_overflow(user_id: str, expert_id: str) -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        await chat_db.append_expert_run_message(
            user_id=user_id,
            expert_id=expert_id,
            content=build_overflow_message(),
            message_id=watcher_message_id(WatcherEvent.OVERFLOW, f"{user_id}:{today}"),
            metadata={
                "kind": WATCHER_METADATA_KIND,
                "event": WatcherEvent.OVERFLOW.value,
                "title": "More expert updates need attention",
                "description": "Open Home for the current state",
                "action_label": "Open Home",
                "action_href": "/home",
                "status": "blocked",
            },
        )
    except Exception as error_value:
        logger.warning(
            "Could not deliver watcher overflow notice for user %s: %s",
            user_id,
            type(error_value).__name__,
        )


def _cap_key(user_id: str) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"copilot-watchers:{user_id}:{today}"


async def _under_daily_cap(key: str) -> bool:
    try:
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, _CAP_KEY_TTL_SECONDS)
            results = await cast(Awaitable[list[Any]], pipe.execute())
        return int(results[0]) <= _DAILY_WATCHER_CAP
    except Exception as error_value:
        logger.warning("Watcher cap unavailable for %s: %s", key, error_value)
        return True


async def _release_cap_slot(key: str) -> None:
    try:
        redis = await get_redis_async()
        await redis.decr(key)
    except Exception:
        logger.debug("Could not release watcher cap slot for %s", key)
