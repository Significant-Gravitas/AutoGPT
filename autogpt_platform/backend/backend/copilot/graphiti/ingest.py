"""Async episode ingestion with per-user serialization.

graphiti-core requires sequential ``add_episode()`` calls within the same
group_id.  This module provides a per-user asyncio.Queue that serializes
ingestion while keeping it fire-and-forget from the caller's perspective.

Every episode is also written to the ``MemoryEpisodeLog`` table as an
append-only replay log, so we can migrate to a different memory service
without data loss.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

from .client import derive_group_id, get_graphiti_client

logger = logging.getLogger(__name__)

_user_queues: dict[str, asyncio.Queue] = {}
_user_workers: dict[str, asyncio.Task] = {}

CUSTOM_EXTRACTION_INSTRUCTIONS = """
- Do not extract "User", "Assistant", "AI", "System", "CoPilot", or "human" as entity nodes.
- Do not extract software tool names, block names, API endpoint names, or internal system identifiers as entities.
- Do not extract action descriptions like "the assistant created..." as facts. Extract only the underlying user intent or real-world information.
- Focus on real-world entities: people, companies, products, projects, concepts, and preferences.
- Use canonical names: if the speaker says "my company" and context reveals it is "Acme Corp", use "Acme Corp".
"""


async def _persist_to_replay_log(
    user_id: str,
    session_id: str,
    group_id: str,
    episode_name: str,
    episode_body: str,
    source: str,
    source_description: str,
) -> None:
    """Write an append-only row to the MemoryEpisodeLog table.

    This is the durable record — even if Graphiti ingestion fails or we
    migrate to a different service, we can replay this log.
    """
    try:
        from backend.data.db_accessors import chat_db

        await chat_db().create_memory_episode_log(
            user_id=user_id,
            session_id=session_id,
            group_id=group_id,
            episode_name=episode_name,
            episode_body=episode_body,
            source=source,
            source_description=source_description,
        )
    except Exception:
        logger.warning(
            "Failed to persist memory episode to replay log for user %s",
            user_id[:12],
            exc_info=True,
        )


async def _ingestion_worker(user_id: str, queue: asyncio.Queue) -> None:
    """Process episodes sequentially for a single user."""
    while True:
        payload = await queue.get()
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


async def enqueue_conversation_turn(
    user_id: str,
    session_id: str,
    user_msg: str,
    assistant_msg: str,
) -> None:
    """Enqueue a conversation turn for async background ingestion.

    This returns almost immediately — the actual graphiti-core
    ``add_episode()`` call (which triggers LLM entity extraction)
    runs in a background worker task.

    The episode is also persisted to the ``MemoryEpisodeLog`` table
    as an append-only replay log before being queued for Graphiti.
    """
    if not user_id:
        return

    from graphiti_core.nodes import EpisodeType

    group_id = derive_group_id(user_id)
    user_display_name = await _resolve_user_name(user_id)

    episode_name = f"conversation_{session_id}"

    # Canonical lane: user's own words only, in graphiti's expected
    # "Speaker: content" format.  Assistant response is excluded from
    # extraction (Zep Cloud approach: ignore_roles=["assistant"]).
    episode_body_for_graphiti = f"{user_display_name}: {user_msg}"

    # Replay log gets the FULL turn (both user + assistant) for audit/replay.
    replay_body = json.dumps(
        {
            "session_id": session_id,
            "messages": [
                {"role": "human", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
        },
        ensure_ascii=False,
    )

    source = str(EpisodeType.message)
    source_description = f"User message in session {session_id}"

    await _persist_to_replay_log(
        user_id=user_id,
        session_id=session_id,
        group_id=group_id,
        episode_name=episode_name,
        episode_body=replay_body,
        source=source,
        source_description=source_description,
    )

    if user_id not in _user_queues:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        _user_queues[user_id] = q
        _user_workers[user_id] = asyncio.create_task(
            _ingestion_worker(user_id, q),
            name=f"graphiti-ingest-{user_id[:12]}",
        )

    try:
        _user_queues[user_id].put_nowait(
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
