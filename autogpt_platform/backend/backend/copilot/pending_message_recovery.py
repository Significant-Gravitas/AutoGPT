"""Restore drained follow-ups when their database persistence fails."""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager

from backend.copilot.pending_messages import (
    _NOTIFY_PAYLOAD,
    _PENDING_TTL_SECONDS,
    PendingMessage,
    _buffer_key,
    _notify_channel,
    finish_despite_cancellation,
)
from backend.data.redis_client import get_redis_async
from backend.data.redis_list_recovery import prepend_json_items_by_id

logger = logging.getLogger(__name__)


@asynccontextmanager
async def recover_pending_on_failure(
    session_id: str, drain: Awaitable[list[PendingMessage]]
) -> AsyncIterator[list[PendingMessage]]:
    """Own a drain until the caller persists it, restoring on early failure."""
    draining = asyncio.ensure_future(drain)
    try:
        yield await asyncio.shield(draining)
    except BaseException:
        try:
            pending = await finish_despite_cancellation(draining)
            await finish_despite_cancellation(
                asyncio.ensure_future(restore_pending_messages(session_id, pending))
            )
        except Exception:
            logger.exception(f"Failed to recover pending messages for {session_id}")
        raise


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
        logger.warning(f"Pending recovery notification failed for {session_id}")
