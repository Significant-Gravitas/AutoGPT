"""Per-target copilot session cache.

The bot remembers which copilot session a DM or thread is currently talking
to. The key lives here so the handler and the ``/new`` command stay in sync:
the handler owns the read/write, and ``/new`` clears it so the next message
starts a fresh AutoPilot conversation.
"""

from backend.data.redis_client import get_redis_async

from .config import SESSION_TTL


def session_cache_key(platform: str, target_id: str) -> str:
    return f"copilot-bot:session:{platform}:{target_id}"


async def set_session(platform: str, target_id: str, session_id: str) -> None:
    redis = await get_redis_async()
    await redis.set(session_cache_key(platform, target_id), session_id, ex=SESSION_TTL)


async def clear_session(platform: str, target_id: str) -> None:
    redis = await get_redis_async()
    await redis.delete(session_cache_key(platform, target_id))
