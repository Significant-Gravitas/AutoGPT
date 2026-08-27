"""One-shot claims for anything that must email a person exactly once.

Two independent sources of duplicates:

* **Stripe replays webhooks.** Several billing events legitimately arrive more
  than once (a retried delivery, an out-of-order subscription update), so each
  billing email claims a key derived from the resource it is about.
* **Scheduled passes fan out over a queue.** Delivery is at-least-once and the
  service can run more than one replica, so the per-user work published by a
  pass claims the user plus the period it covers.

Both need the same primitive: the first caller wins, everyone else no-ops.

Falls open on a Redis failure. A rare duplicate is a better outcome than
dropping a payment-failed notice — or a whole briefing — entirely.
"""

import logging
from collections.abc import Awaitable
from datetime import date
from typing import cast
from uuid import uuid4

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# Comfortably longer than Stripe's retry window, and long enough that an
# out-of-order redelivery days later still finds the claim.
CLAIM_TTL_SECONDS = 60 * 60 * 24 * 30
# A scheduled pass only has to dedupe within the period it covers; holding the
# key for a month would suppress the *next* period's legitimate send.
PASS_CLAIM_TTL_SECONDS = 60 * 60 * 36
_PREFIX = "notification_claim"


async def claim_once(key: str, ttl_seconds: int = CLAIM_TTL_SECONDS) -> bool:
    """Returns True for the first caller with this key, False afterwards."""
    try:
        redis_client = await get_redis_async()
        claimed = await redis_client.set(
            f"{_PREFIX}:{key}", "1", nx=True, ex=ttl_seconds
        )
        return bool(claimed)
    except Exception:
        logger.warning(
            "Dedupe claim failed for %s; proceeding anyway", key, exc_info=True
        )
        return True


async def release_claim(key: str) -> None:
    """Give a claim back after the work it guarded failed.

    A claim is taken *before* the send so a replay cannot double-send. That
    ordering has a cost: if the send then fails, the key is spent and every
    retry is deduped away, so the email is lost permanently rather than
    retried. Releasing on failure keeps the replay protection while letting a
    genuine retry through.
    """
    try:
        redis_client = await get_redis_async()
        await redis_client.delete(f"{_PREFIX}:{key}")
    except Exception:
        logger.warning(
            "Could not release the dedupe claim for %s; a retry of this "
            "message will be suppressed until the key expires",
            key,
            exc_info=True,
        )


# Counters outlive the day they cover so a late send is still counted against
# the right day, and are dropped well before they could be reused.
_DAILY_COUNTER_TTL_SECONDS = 60 * 60 * 48

_CLAIM_DAILY_SEND_SCRIPT = """
local existing = redis.call('GET', KEYS[2])
if existing then
  return tonumber(existing)
end
local sent = redis.call('INCR', KEYS[1])
if sent == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
local allowed = 0
if sent <= tonumber(ARGV[1]) then
  allowed = 1
end
redis.call('SET', KEYS[2], allowed, 'EX', ARGV[2])
return allowed
"""

_RELEASE_DAILY_SEND_SCRIPT = """
local reserved = redis.call('GET', KEYS[2])
if reserved == '1' then
  local remaining = redis.call('DECR', KEYS[1])
  if remaining < 0 then
    redis.call('SET', KEYS[1], 0, 'EX', ARGV[1])
  end
end
redis.call('DEL', KEYS[2])
return 1
"""


def _daily_send_keys(user_id: str, on_day: date, delivery_id: str) -> tuple[str, str]:
    slot_tag = f"{user_id}:{on_day.isoformat()}"
    counter_key = f"{_PREFIX}:daily_send:{{{slot_tag}}}"
    return counter_key, f"{counter_key}:delivery:{delivery_id}"


async def claim_daily_send(
    user_id: str,
    limit: int,
    on_day: date,
    delivery_id: str | None = None,
) -> bool:
    """Take one of the user's sends for `on_day`, or refuse.

    The volume knob's `daily_limit` is the user's own ceiling across every
    product notification. It is counted here rather than in the database
    because it is ephemeral per-day state: a counter with a TTL needs no table,
    no reset job, and no migration to change.

    `limit <= 0` refuses everything, which is what a one-click unsubscribe
    means when it sets the limit to zero.

    Fails *open* on a Redis error, matching `claim_once`: a rare extra email is
    a better outcome than silently swallowing someone's briefing.
    """
    if limit <= 0:
        return False
    counter_key, reservation_key = _daily_send_keys(
        user_id, on_day, delivery_id or str(uuid4())
    )
    try:
        redis_client = await get_redis_async()
        allowed = await cast(
            Awaitable[object],
            redis_client.eval(
                _CLAIM_DAILY_SEND_SCRIPT,
                2,
                counter_key,
                reservation_key,
                str(limit),
                str(_DAILY_COUNTER_TTL_SECONDS),
            ),
        )
        return bool(allowed)
    except Exception:
        logger.warning(
            "Daily-send counter unavailable for %s; allowing the send",
            user_id,
            exc_info=True,
        )
        return True


async def release_daily_send(user_id: str, on_day: date, delivery_id: str) -> None:
    counter_key, reservation_key = _daily_send_keys(user_id, on_day, delivery_id)
    try:
        redis_client = await get_redis_async()
        await cast(
            Awaitable[object],
            redis_client.eval(
                _RELEASE_DAILY_SEND_SCRIPT,
                2,
                counter_key,
                reservation_key,
                str(_DAILY_COUNTER_TTL_SECONDS),
            ),
        )
    except Exception:
        logger.warning(
            "Could not release daily-send reservation for %s",
            user_id,
            exc_info=True,
        )
