"""Bound new prepayment personalization work; saved drafts and retries stay usable."""

import asyncio
import hashlib
import logging
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import cast

from fastapi import HTTPException
from prisma.enums import SubscriptionTier

from backend.data.redis_client import get_redis_async
from backend.data.user import get_user_subscription_tier
from backend.monitoring.instrumentation import record_rate_limit_hit
from backend.util.feature_flag import Flag, evaluate_feature_flag
from backend.util.settings import BehaveAs, Settings

logger = logging.getLogger(__name__)
settings = Settings()

HOURLY_ATTEMPTS = 10
DAILY_ATTEMPTS = 30
REDIS_TIMEOUT_SECONDS = 0.25

# Both counters share a cluster slot. Rejection changes neither counter, and
# admission checks/increments both atomically across API workers.
_ADMIT = """
local hourly = tonumber(redis.call('GET', KEYS[1]) or '0')
local daily = tonumber(redis.call('GET', KEYS[2]) or '0')
local retry_after = 0
if hourly >= tonumber(ARGV[1]) then retry_after = tonumber(ARGV[3]) end
if daily >= tonumber(ARGV[2]) then
    retry_after = math.max(retry_after, tonumber(ARGV[4]))
end
if retry_after > 0 then return retry_after end
if redis.call('INCR', KEYS[1]) == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[3])
end
if redis.call('INCR', KEYS[2]) == 1 then
    redis.call('EXPIRE', KEYS[2], ARGV[4])
end
return 0
"""


async def enforce_personalization_budget(user_id: str) -> None:
    try:
        if not await _requires_budget(user_id):
            return
        retry_after = await asyncio.wait_for(
            _admit(user_id), timeout=REDIS_TIMEOUT_SECONDS
        )
    except Exception as exc:
        logger.warning(f"Onboarding personalization admission unavailable: {exc}")
        raise HTTPException(
            status_code=503,
            detail="Personalization is temporarily unavailable. Your progress is saved; please retry shortly.",
            headers={"Retry-After": "30"},
        ) from exc

    if retry_after:
        record_rate_limit_hit("/api/onboarding/personalization", user_id)
        raise HTTPException(
            status_code=429,
            detail="You have made several personalization attempts. Your progress is saved; please try again later.",
            headers={"Retry-After": str(retry_after)},
        )


async def _requires_budget(user_id: str) -> bool:
    if settings.config.behave_as == BehaveAs.LOCAL:
        return False
    try:
        tier = await get_user_subscription_tier(user_id)
    except ValueError:
        tier = SubscriptionTier.NO_TIER
    if tier != SubscriptionTier.NO_TIER:
        return False
    enabled, authoritative = await evaluate_feature_flag(
        Flag.ENABLE_PLATFORM_PAYMENT, user_id
    )
    if not authoritative:
        raise RuntimeError("Payment flag could not be evaluated")
    return enabled


async def _admit(user_id: str) -> int:
    hourly_key, daily_key, hourly_ttl, daily_ttl = _budget_windows(
        user_id, datetime.now(UTC)
    )
    redis = await get_redis_async()
    return await cast(
        Awaitable[int],
        redis.eval(
            _ADMIT,
            2,
            hourly_key,
            daily_key,
            str(HOURLY_ATTEMPTS),
            str(DAILY_ATTEMPTS),
            str(hourly_ttl),
            str(daily_ttl),
        ),
    )


def _budget_windows(user_id: str, now: datetime) -> tuple[str, str, int, int]:
    identity = hashlib.sha256(user_id.encode()).hexdigest()
    prefix = f"onboarding:personalization:{{{identity}}}"
    timestamp = int(now.timestamp())
    return (
        f"{prefix}:hour:{timestamp // 3600}",
        f"{prefix}:day:{timestamp // 86400}",
        3600 - timestamp % 3600,
        86400 - timestamp % 86400,
    )
