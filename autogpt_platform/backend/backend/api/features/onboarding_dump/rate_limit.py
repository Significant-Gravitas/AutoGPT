"""Bound new prepayment personalization work; saved drafts and retries stay usable."""

import asyncio
import hashlib
import logging
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import NamedTuple, cast

from fastapi import HTTPException
from prisma.enums import SubscriptionTier

from backend.copilot.onboarding_interview import INTERVIEW_TURN_CAP
from backend.data.redis_client import AsyncRedisClient, get_redis_async
from backend.data.user import get_user_subscription_tier
from backend.monitoring.instrumentation import record_rate_limit_hit
from backend.util.platform_payment import is_platform_payment_enabled
from backend.util.settings import BehaveAs, Settings

logger = logging.getLogger(__name__)
settings = Settings()

HOURLY_ATTEMPTS = 10
DAILY_ATTEMPTS = 30
REDIS_TIMEOUT_SECONDS = 0.25

# An onboarding interview's turns, counted apart from personalization
# attempts: its kickoff, every reply and the hand-off would spend a long
# interview's share of those in one sitting. Sized from the reply cap: the
# whole interview and a few resends in an hour, twice the cap in a day. Each
# person has one interview, and the cap still ends it.
INTERVIEW_HOURLY_TURNS = INTERVIEW_TURN_CAP + 4
INTERVIEW_DAILY_TURNS = 2 * INTERVIEW_TURN_CAP

# Onboarding voice calls a person may make: a call and a retry in an hour,
# three in a day. The relay caps each call, so these bound how many calls,
# not how long any one runs. A call counts once the relay has it ready,
# before the model is sent anything; the start route only reads the count.
VOICE_HOURLY_CALLS = 2
VOICE_DAILY_CALLS = 3

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

# The admission's check alone: the wait before another, or 0, counting
# nothing.
_CHECK = """
local hourly = tonumber(redis.call('GET', KEYS[1]) or '0')
local daily = tonumber(redis.call('GET', KEYS[2]) or '0')
local retry_after = 0
if hourly >= tonumber(ARGV[1]) then retry_after = tonumber(ARGV[3]) end
if daily >= tonumber(ARGV[2]) then
    retry_after = math.max(retry_after, tonumber(ARGV[4]))
end
return retry_after
"""

# The admission's count alone, past any limit. KEYS as above; ARGV[1] and
# ARGV[2]: the windows' TTLs.
_COUNT = """
if redis.call('INCR', KEYS[1]) == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
if redis.call('INCR', KEYS[2]) == 1 then
    redis.call('EXPIRE', KEYS[2], ARGV[2])
end
return 0
"""


class _Budget(NamedTuple):
    scope: str
    hourly: int
    daily: int
    route: str
    unavailable: str
    exhausted: str
    # Counted on every subscription tier, not only for someone with no plan.
    all_tiers: bool = False


_PERSONALIZATION = _Budget(
    scope="personalization",
    hourly=HOURLY_ATTEMPTS,
    daily=DAILY_ATTEMPTS,
    route="/api/onboarding/personalization",
    unavailable="Personalization is temporarily unavailable. Your progress is saved; please retry shortly.",
    exhausted="You have made several personalization attempts. Your progress is saved; please try again later.",
)

_INTERVIEW = _Budget(
    scope="interview",
    hourly=INTERVIEW_HOURLY_TURNS,
    daily=INTERVIEW_DAILY_TURNS,
    route="/api/onboarding/interview",
    unavailable="The interview is temporarily unavailable. Your progress is saved; please retry shortly.",
    exhausted="You have sent a lot of messages in a short time. Your progress is saved; please try again later.",
    all_tiers=True,
)

_VOICE_INTERVIEW = _Budget(
    scope="voice_interview",
    hourly=VOICE_HOURLY_CALLS,
    daily=VOICE_DAILY_CALLS,
    route="/api/onboarding/brain-dump/interview/voice",
    unavailable="Voice calls are temporarily unavailable. Your progress is saved; please retry shortly.",
    exhausted="You have started several voice calls. Your progress is saved; you can type instead, or call again later.",
    all_tiers=True,
)


async def enforce_personalization_budget(user_id: str) -> None:
    await _enforce(user_id, _PERSONALIZATION)


async def enforce_interview_turn_budget(user_id: str) -> None:
    """Admit one onboarding interview turn, its kickoff included.

    The personalization budget's rules on the interview's own counters,
    except that it applies on every subscription tier where the platform
    takes payment: a turn past it is a 429 with Retry-After, and a budget
    that cannot be read is a 503 rather than a turn nobody counted.
    """
    await _enforce(user_id, _INTERVIEW)


async def enforce_voice_call_budget(user_id: str) -> None:
    """Refuse an onboarding voice call past the person's call allowance, on
    the interview turn budget's rules and its own counters, counting nothing:
    the relay counts a call once it is ready (:func:`count_voice_call`), so a
    start that never got that far costs the person nothing. On every
    subscription tier where the platform takes payment, a call past it is a
    429 with Retry-After, and an allowance that cannot be read is a 503."""
    await _enforce(user_id, _VOICE_INTERVIEW, script=_CHECK)


async def count_voice_call(redis: AsyncRedisClient, user_id: str) -> None:
    """Count one call against *user_id*'s voice call allowance, where the
    allowance applies: on the counters :func:`enforce_voice_call_budget`
    reads, with *redis*, the caller's client."""
    if not await _requires_budget(user_id, _VOICE_INTERVIEW):
        return
    hourly_key, daily_key, hourly_ttl, daily_ttl = _budget_windows(
        user_id, datetime.now(UTC), _VOICE_INTERVIEW.scope
    )
    await cast(
        Awaitable[int],
        redis.eval(_COUNT, 2, hourly_key, daily_key, str(hourly_ttl), str(daily_ttl)),
    )


async def _enforce(user_id: str, budget: _Budget, *, script: str = _ADMIT) -> None:
    try:
        if not await _requires_budget(user_id, budget):
            return
        retry_after = await asyncio.wait_for(
            _admit(user_id, budget, script), timeout=REDIS_TIMEOUT_SECONDS
        )
    except Exception as exc:
        logger.warning(f"Onboarding {budget.scope} admission unavailable: {exc}")
        raise HTTPException(
            status_code=503,
            detail=budget.unavailable,
            headers={"Retry-After": "30"},
        ) from exc

    if retry_after:
        record_rate_limit_hit(budget.route, user_id)
        raise HTTPException(
            status_code=429,
            detail=budget.exhausted,
            headers={"Retry-After": str(retry_after)},
        )


async def _requires_budget(user_id: str, budget: _Budget) -> bool:
    if settings.config.behave_as == BehaveAs.LOCAL:
        return False
    if not budget.all_tiers:
        try:
            tier = await get_user_subscription_tier(user_id)
        except ValueError:
            tier = SubscriptionTier.NO_TIER
        if tier != SubscriptionTier.NO_TIER:
            return False
    return is_platform_payment_enabled()


async def _admit(user_id: str, budget: _Budget, script: str) -> int:
    hourly_key, daily_key, hourly_ttl, daily_ttl = _budget_windows(
        user_id, datetime.now(UTC), budget.scope
    )
    redis = await get_redis_async()
    return await cast(
        Awaitable[int],
        redis.eval(
            script,
            2,
            hourly_key,
            daily_key,
            str(budget.hourly),
            str(budget.daily),
            str(hourly_ttl),
            str(daily_ttl),
        ),
    )


def _budget_windows(
    user_id: str, now: datetime, scope: str = _PERSONALIZATION.scope
) -> tuple[str, str, int, int]:
    identity = hashlib.sha256(user_id.encode()).hexdigest()
    prefix = f"onboarding:{scope}:{{{identity}}}"
    timestamp = int(now.timestamp())
    return (
        f"{prefix}:hour:{timestamp // 3600}",
        f"{prefix}:day:{timestamp // 86400}",
        3600 - timestamp % 3600,
        86400 - timestamp % 86400,
    )
