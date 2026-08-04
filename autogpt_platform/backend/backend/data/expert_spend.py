"""Per-expert weekly credit spend counters (issue #13717).

Redis is the hot-path store: billing increments on every charge that carries
an expert-attributed execution context, and the budget gate reads it before
starting a scheduled/triggered run. The durable source of truth remains
CreditTransaction joined through AgentGraphExecution.expertId — these
counters are a cache, safe to lose (a lost key under-counts one week's
spend, which only ever errs in the user's favor).
"""

import logging
from datetime import datetime, timezone

from backend.data.redis_client import get_redis, get_redis_async

logger = logging.getLogger(__name__)

# Keys outlive the week they track so a just-rolled-over week is still
# readable for display; two weeks is plenty.
_KEY_TTL_SECONDS = 14 * 24 * 3600


def weekly_spend_key(expert_id: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    year, week, _ = now.isocalendar()
    return f"expert-spend:{expert_id}:{year}-W{week:02d}"


async def add_weekly_spend(expert_id: str, amount: int) -> None:
    """Add *amount* credits (may be negative for refund reconciliation) to
    the expert's current-week counter. Never raises — a metering failure
    must not fail the charge that triggered it."""
    if amount == 0:
        return
    try:
        redis = await get_redis_async()
        key = weekly_spend_key(expert_id)
        await redis.incrby(key, amount)
        await redis.expire(key, _KEY_TTL_SECONDS)
    except Exception as e:
        logger.warning(
            f"Failed to record weekly spend for expert #{expert_id}: "
            f"{type(e).__name__}: {e}"
        )


def add_weekly_spend_sync(expert_id: str, amount: int) -> None:
    """Sync variant for the pre-flight charge path (``charge_usage`` runs
    on a worker thread with the sync clients). Same never-raises contract."""
    if amount == 0:
        return
    try:
        redis = get_redis()
        key = weekly_spend_key(expert_id)
        redis.incrby(key, amount)
        redis.expire(key, _KEY_TTL_SECONDS)
    except Exception as e:
        logger.warning(
            f"Failed to record weekly spend for expert #{expert_id}: "
            f"{type(e).__name__}: {e}"
        )


async def get_weekly_spend(expert_id: str) -> int:
    """Current-week spend in credits; 0 on any read failure (errs open —
    the budget gate must not block runs because Redis hiccuped). Clamped to
    non-negative: a refund reconciled in a later ISO week than its charge
    decrements the new week's counter and could otherwise go below zero."""
    try:
        redis = await get_redis_async()
        value = await redis.get(weekly_spend_key(expert_id))
        return max(0, int(value)) if value is not None else 0
    except Exception as e:
        logger.warning(
            f"Failed to read weekly spend for expert #{expert_id}: "
            f"{type(e).__name__}: {e}"
        )
        return 0
