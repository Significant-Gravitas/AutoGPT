"""Per-expert credit spend counters (issue #13717, SECRT-2599).

Redis is the hot-path store: billing increments on every charge that carries
an expert-attributed execution context, and the budget gate reads it before
starting a scheduled/triggered run. The durable source of truth remains
CreditTransaction joined through AgentGraphExecution.expertId — these
counters are a cache, safe to lose (a lost key under-counts one window's
spend, which only ever errs in the user's favor).

Every charge lands in two buckets, the ISO week the weekly budget enforces
and the UTC day, so the spend-approval window can switch between them
without losing data.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

from backend.data.redis_client import get_redis, get_redis_async

logger = logging.getLogger(__name__)

SpendWindow = Literal["week", "day"]

# Keys outlive the window they track so a just-rolled-over one is still
# readable for display.
_KEY_TTL_SECONDS = {"week": 14 * 24 * 3600, "day": 2 * 24 * 3600}


def weekly_spend_key(expert_id: str, now: datetime | None = None) -> str:
    return spend_key(expert_id, "week", now)


def spend_key(expert_id: str, window: SpendWindow, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    if window == "day":
        return f"expert-spend:{expert_id}:{now:%Y-%m-%d}"
    year, week, _ = now.isocalendar()
    return f"expert-spend:{expert_id}:{year}-W{week:02d}"


def window_start(window: SpendWindow, now: datetime | None = None) -> datetime:
    """First instant of the bucket ``now`` falls in (UTC)."""
    now = now or datetime.now(timezone.utc)
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if window == "day":
        return day
    return day - timedelta(days=day.weekday())


async def add_weekly_spend(expert_id: str, amount: int) -> None:
    """Add *amount* credits (may be negative for refund reconciliation) to
    the expert's current week and day counters. Never raises — a metering
    failure must not fail the charge that triggered it."""
    if amount == 0:
        return
    try:
        redis = await get_redis_async()
        for window in ("week", "day"):
            key = spend_key(expert_id, window)
            await redis.incrby(key, amount)
            await redis.expire(key, _KEY_TTL_SECONDS[window])
    except Exception as e:
        logger.warning(
            f"Failed to record spend for expert #{expert_id}: "
            f"{type(e).__name__}: {e}"
        )


def add_weekly_spend_sync(expert_id: str, amount: int) -> None:
    """Sync variant for the pre-flight charge path (``charge_usage`` runs
    on a worker thread with the sync clients). Same never-raises contract."""
    if amount == 0:
        return
    try:
        redis = get_redis()
        for window in ("week", "day"):
            key = spend_key(expert_id, window)
            redis.incrby(key, amount)
            redis.expire(key, _KEY_TTL_SECONDS[window])
    except Exception as e:
        logger.warning(
            f"Failed to record spend for expert #{expert_id}: "
            f"{type(e).__name__}: {e}"
        )


async def reset_weekly_spend(expert_id: str) -> None:
    """Zero the current counters. Called when the user resumes a paused
    expert: keeping the count would re-pause her on the next gate check,
    making Resume a no-op until the ISO week rolls over. Billing is
    unaffected — only the guardrail's counters restart. Never raises; on
    failure the user can simply resume again."""
    try:
        redis = await get_redis_async()
        await redis.delete(spend_key(expert_id, "week"), spend_key(expert_id, "day"))
    except Exception as e:
        logger.warning(
            f"Failed to reset spend for expert #{expert_id}: "
            f"{type(e).__name__}: {e}"
        )


async def get_weekly_spend(expert_id: str) -> int:
    return await get_spend(expert_id, "week")


async def get_spend(expert_id: str, window: SpendWindow) -> int:
    """Current-window spend in credits; 0 on any read failure (errs open —
    a gate must not block runs because Redis hiccuped). Clamped to
    non-negative: a refund reconciled in a later window than its charge
    decrements the new counter and could otherwise go below zero."""
    try:
        redis = await get_redis_async()
        value = await redis.get(spend_key(expert_id, window))
        return max(0, int(value)) if value is not None else 0
    except Exception as e:
        logger.warning(
            f"Failed to read spend for expert #{expert_id}: " f"{type(e).__name__}: {e}"
        )
        return 0
