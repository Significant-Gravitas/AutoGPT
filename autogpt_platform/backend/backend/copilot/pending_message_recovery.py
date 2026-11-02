"""Restore drained follow-ups when their database persistence fails."""

import logging

from backend.copilot.pending_messages import (
    _NOTIFY_PAYLOAD,
    _PENDING_TTL_SECONDS,
    PendingMessage,
    _buffer_key,
    _notify_channel,
)
from backend.data.redis_client import get_redis_async
from backend.data.redis_list_recovery import prepend_json_items_by_id

logger = logging.getLogger(__name__)


async def restore_pending_messages(
    session_id: str, messages: list[PendingMessage]
) -> None:
    """Put accepted messages ahead of newer arrivals without evicting either."""
    if not messages:
        return
    redis = await get_redis_async()
    await prepend_json_items_by_id(
        redis,
        _buffer_key(session_id),
        [message.model_dump_json() for message in messages],
        ttl_seconds=_PENDING_TTL_SECONDS,
    )
    try:
        await redis.execute_command(
            "SPUBLISH", _notify_channel(session_id), _NOTIFY_PAYLOAD
        )
    except Exception:
        logger.warning("Pending recovery notification failed for %s", session_id)
