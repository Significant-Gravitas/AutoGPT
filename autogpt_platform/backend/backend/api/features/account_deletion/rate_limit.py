"""Per-user cap on ``POST /api/account/deletion/feedback``.

Every call sends one anonymous PostHog event, and nothing in PostHog can tell
a script's repeats from real answers or remove them afterwards, so the cap is
kept here. The flow sends one answer and at most a few clicks per pass, so 10
an hour is far above real use.

Fixed-window counter in Redis, as in
:mod:`backend.api.features.search.rate_limit`: approximate by design, and it
fails open, since losing a few extra answers to a Redis blip is far better
than refusing them all.
"""

import logging
from datetime import UTC, datetime

from fastapi import HTTPException

from backend.data.redis_client import get_redis_async
from backend.monitoring.instrumentation import record_rate_limit_hit

logger = logging.getLogger(__name__)

FEEDBACK_WINDOW_SECONDS = 60 * 60
FEEDBACK_MAX_REQUESTS = 10
FEEDBACK_RATE_LIMIT_DETAIL = "Too many answers. Try again later."
_ENDPOINT = "/api/account/deletion/feedback"


async def enforce_feedback_rate_limit(user_id: str) -> None:
    """Raise HTTP 429 once ``user_id`` has sent its quota for this window."""
    key = _window_key(user_id, now=datetime.now(UTC))
    try:
        redis = await get_redis_async()
        # SET NX EX makes the TTL part of the write that creates the key.
        await redis.set(key, 0, ex=FEEDBACK_WINDOW_SECONDS, nx=True)
        count = await redis.incr(key)
    except Exception as e:
        logger.warning(
            f"Account deletion feedback rate limit failed open for user "
            f"{user_id}: {type(e).__name__}"
        )
        return

    if count > FEEDBACK_MAX_REQUESTS:
        record_rate_limit_hit(_ENDPOINT, user_id)
        raise HTTPException(status_code=429, detail=FEEDBACK_RATE_LIMIT_DETAIL)


def _window_key(user_id: str, *, now: datetime) -> str:
    bucket = int(now.timestamp()) // FEEDBACK_WINDOW_SECONDS
    return f"account_deletion:feedback:rl:{user_id}:{bucket}"
