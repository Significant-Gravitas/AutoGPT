"""Chat-scoped ask rules: a subject the user rejected asks for the rest of the chat."""

import logging

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# Outlives any chat anyone is still in; dead chats expire out of Redis.
_TTL_SECONDS = 90 * 24 * 60 * 60
_ASK_KEY = "copilot:gate:ask:"


async def set_ask(session_id: str, tool_name: str) -> None:
    """Monotone: there is no un-ask, so re-proposing the call with a space
    added to its arguments cannot buy a fresh verdict."""
    try:
        redis = await get_redis_async()
        await redis.setex(_key(session_id, tool_name), _TTL_SECONDS, "1")
    except Exception:
        logger.warning(
            f"Gate could not persist an ask rule for session {session_id}",
            exc_info=True,
        )


async def asks(session_id: str, tool_name: str) -> bool:
    try:
        redis = await get_redis_async()
        return await redis.get(_key(session_id, tool_name)) is not None
    except Exception:
        logger.warning(
            f"Gate could not read ask rules for session {session_id}; "
            "assuming the subject asks",
            exc_info=True,
        )
        return True


def _key(session_id: str, tool_name: str) -> str:
    # A flag per (session, tool): the cluster client's set operations are not
    # typed as awaitable.
    return f"{_ASK_KEY}{session_id}:{tool_name}"
