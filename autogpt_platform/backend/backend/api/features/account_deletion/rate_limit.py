"""Per-user cap on ``POST /api/account/deletion/feedback``.

Every call sends one anonymous PostHog event, and nothing in PostHog can tell
a script's repeats from real answers or remove them afterwards, so the cap is
kept here. The flow sends one open, one answer and at most a few clicks per
pass, so 20 an hour is far above real use.

Fixed-window counter in Redis (:func:`backend.data.redis_helpers.incr_with_ttl`):
approximate by design, and it fails open, since losing a few extra answers to a
Redis blip is far better than refusing them all. The whole Redis interaction
has a short deadline, so an outage falls through at once instead of holding
the request while the client retries its connect.
"""

import asyncio
import logging
from datetime import UTC, datetime

from fastapi import HTTPException

from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import incr_with_ttl
from backend.monitoring.instrumentation import record_rate_limit_hit

logger = logging.getLogger(__name__)

FEEDBACK_WINDOW_SECONDS = 60 * 60
FEEDBACK_MAX_REQUESTS = 20
FEEDBACK_RATE_LIMIT_DETAIL = "Too many answers. Try again later."
# Same budget as billing/credits_rate_limit.py.
FEEDBACK_REDIS_TIMEOUT_SECONDS = 0.25
_ENDPOINT = "/api/account/deletion/feedback"


async def enforce_feedback_rate_limit(user_id: str) -> None:
    """Raise HTTP 429 once ``user_id`` has sent its quota for this window."""
    key = _window_key(user_id, now=datetime.now(UTC))
    try:
        count = await asyncio.wait_for(
            _incr_window(key), timeout=FEEDBACK_REDIS_TIMEOUT_SECONDS
        )
    except Exception as e:
        logger.warning(
            f"Account deletion feedback rate limit failed open for user "
            f"{user_id}: {type(e).__name__}"
        )
        return

    if count > FEEDBACK_MAX_REQUESTS:
        record_rate_limit_hit(_ENDPOINT, user_id)
        raise HTTPException(status_code=429, detail=FEEDBACK_RATE_LIMIT_DETAIL)


async def _incr_window(key: str) -> int:
    # The connect shares the deadline: its retry ladder is what stalls in an
    # outage.
    return await incr_with_ttl(await get_redis_async(), key, FEEDBACK_WINDOW_SECONDS)


def _window_key(user_id: str, *, now: datetime) -> str:
    bucket = int(now.timestamp()) // FEEDBACK_WINDOW_SECONDS
    return f"account_deletion:feedback:rl:{user_id}:{bucket}"
