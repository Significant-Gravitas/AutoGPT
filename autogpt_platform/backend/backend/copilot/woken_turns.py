"""Which turn a wake started to carry an answered card, so a channel that
answered the card can stream that turn's reply and no other."""

import logging
from typing import Iterable

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# Read moments after the wake, or once the turn that was running ends.
_TTL_SECONDS = 60 * 60
_KEY = "copilot:woken-turn:"


async def record(session_id: str, review_ids: Iterable[str], turn_id: str) -> None:
    """Best effort: the turn runs either way; only a channel's reply is lost."""
    try:
        redis = await get_redis_async()
        async with redis.pipeline(transaction=True) as pipe:
            pipe.hset(_key(session_id), mapping={r: turn_id for r in review_ids})
            pipe.expire(_key(session_id), _TTL_SECONDS)
            await pipe.execute()
    except Exception:
        logger.warning(f"Woken turn {turn_id} not recorded", exc_info=True)


async def turn_for(session_id: str, review_id: str) -> str | None:
    redis = await get_redis_async()
    turn_id = await redis.hget(_key(session_id), review_id)
    return turn_id.decode() if isinstance(turn_id, bytes) else turn_id


def _key(session_id: str) -> str:
    return f"{_KEY}{session_id}"
