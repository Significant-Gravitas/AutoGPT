"""Async episode ingestion with per-user serialization.

graphiti-core requires sequential ``add_episode()`` calls within the same
group_id.  This module provides a per-user asyncio.Queue that serializes
ingestion while keeping it fire-and-forget from the caller's perspective.
"""

import asyncio
import logging
from datetime import datetime, timezone

from graphiti_core.nodes import EpisodeType

from .client import derive_group_id, get_graphiti_client

logger = logging.getLogger(__name__)

_user_queues: dict[str, asyncio.Queue] = {}
_user_workers: dict[str, asyncio.Task] = {}
_workers_lock = asyncio.Lock()

# Idle workers are cleaned up after this many seconds of inactivity.
_WORKER_IDLE_TIMEOUT = 60

CUSTOM_EXTRACTION_INSTRUCTIONS = """
- Do not extract "User", "Assistant", "AI", "System", "CoPilot", or "human" as entity nodes.
- Do not extract software tool names, block names, API endpoint names, or internal system identifiers as entities.
- Do not extract action descriptions like "the assistant created..." as facts. Extract only the underlying user intent or real-world information.
- Focus on real-world entities: people, companies, products, projects, concepts, and preferences.
- Use canonical names: if the speaker says "my company" and context reveals it is "Acme Corp", use "Acme Corp".
"""


async def _ingestion_worker(user_id: str, queue: asyncio.Queue) -> None:
    """Process episodes sequentially for a single user.

    Exits after ``_WORKER_IDLE_TIMEOUT`` seconds of inactivity so that
    idle workers don't leak memory indefinitely.
    """
    try:
        while True:
            try:
                payload = await asyncio.wait_for(
                    queue.get(), timeout=_WORKER_IDLE_TIMEOUT
                )
            except asyncio.TimeoutError:
                break  # idle — clean up below

            try:
                group_id = derive_group_id(user_id)
                client = await get_graphiti_client(group_id)
                await client.add_episode(**payload)
            except Exception:
                logger.warning(
                    "Graphiti ingestion failed for user %s",
                    user_id[:12],
                    exc_info=True,
                )
            finally:
                queue.task_done()
    except asyncio.CancelledError:
        logger.debug("Ingestion worker cancelled for user %s", user_id[:12])
        raise
    finally:
        # Clean up so the next message re-creates the worker.
        _user_queues.pop(user_id, None)
        _user_workers.pop(user_id, None)


async def enqueue_conversation_turn(
    user_id: str,
    session_id: str,
    user_msg: str,
) -> None:
    """Enqueue a conversation turn for async background ingestion.

    This returns almost immediately — the actual graphiti-core
    ``add_episode()`` call (which triggers LLM entity extraction)
    runs in a background worker task.
    """
    if not user_id:
        return

    try:
        group_id = derive_group_id(user_id)
    except ValueError:
        logger.warning("Invalid user_id for ingestion: %s", user_id[:12])
        return

    user_display_name = await _resolve_user_name(user_id)

    episode_name = f"conversation_{session_id}"

    # User's own words only, in graphiti's expected "Speaker: content" format.
    # Assistant response is excluded from extraction
    # (Zep Cloud approach: ignore_roles=["assistant"]).
    episode_body_for_graphiti = f"{user_display_name}: {user_msg}"

    source_description = f"User message in session {session_id}"

    queue = await _ensure_worker(user_id)

    try:
        queue.put_nowait(
            {
                "name": episode_name,
                "episode_body": episode_body_for_graphiti,
                "source": EpisodeType.message,
                "source_description": source_description,
                "reference_time": datetime.now(timezone.utc),
                "group_id": group_id,
                "custom_extraction_instructions": CUSTOM_EXTRACTION_INSTRUCTIONS,
            }
        )
    except asyncio.QueueFull:
        logger.warning(
            "Graphiti ingestion queue full for user %s — dropping episode",
            user_id[:12],
        )


async def enqueue_episode(
    user_id: str,
    session_id: str,
    *,
    name: str,
    episode_body: str,
    source_description: str = "Conversation memory",
) -> bool:
    """Enqueue an arbitrary episode for background ingestion.

    Used by ``MemoryStoreTool`` so that explicit memory-store calls go
    through the same per-user serialization queue as conversation turns.

    Returns ``True`` if the episode was queued, ``False`` if it was dropped.
    """
    if not user_id:
        return False

    try:
        group_id = derive_group_id(user_id)
    except ValueError:
        logger.warning("Invalid user_id for episode ingestion: %s", user_id[:12])
        return False

    queue = await _ensure_worker(user_id)

    try:
        queue.put_nowait(
            {
                "name": name,
                "episode_body": episode_body,
                "source": EpisodeType.text,
                "source_description": source_description,
                "reference_time": datetime.now(timezone.utc),
                "group_id": group_id,
                "custom_extraction_instructions": CUSTOM_EXTRACTION_INSTRUCTIONS,
            }
        )
        return True
    except asyncio.QueueFull:
        logger.warning(
            "Graphiti ingestion queue full for user %s — dropping episode",
            user_id[:12],
        )
        return False


async def _ensure_worker(user_id: str) -> asyncio.Queue:
    """Create a queue and worker for *user_id* if one doesn't exist.

    Returns the queue directly so callers don't need to look it up from
    ``_user_queues`` (which avoids a TOCTOU race if the worker times out
    and cleans up between this call and the put_nowait).
    """
    async with _workers_lock:
        if user_id not in _user_queues:
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            _user_queues[user_id] = q
            _user_workers[user_id] = asyncio.create_task(
                _ingestion_worker(user_id, q),
                name=f"graphiti-ingest-{user_id[:12]}",
            )
        return _user_queues[user_id]


async def _resolve_user_name(user_id: str) -> str:
    """Get the user's display name from BusinessUnderstanding, or fall back to 'User'."""
    try:
        from backend.data.db_accessors import understanding_db

        understanding = await understanding_db().get_business_understanding(user_id)
        if understanding and understanding.user_name:
            return understanding.user_name
    except Exception:
        logger.debug("Could not resolve user name for %s", user_id[:12])
    return "User"
