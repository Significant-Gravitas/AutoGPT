"""Whether usage limits run on each customer's billing clock yet.

The ``usage-billing-clock`` flag names the launch second::

    {"live_from": 1763460000}

Before it, and whenever the flag is conclusively off, limits run as they
always have (Monday and midnight UTC windows). Because it is a moment set in
advance, every server switches at the same second.

An outage must not flip servers back and forth, so only an answer PostHog
actually gave changes anything: it is remembered in this process and in
Redis, and "no answer" falls back to the last one given.
"""

import logging
import time
from typing import Any

from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.data.redis_client import get_redis_async
from backend.util.cache import cached
from backend.util.feature_flag import Flag, posthog, read_feature_flag

logger = logging.getLogger(__name__)

_config = ChatConfig()

LAST_ANSWER_KEY = "usage:billing_clock:live_from"
# A launch second outside this range is a typo (milliseconds, a date in the
# past decade): treated as no answer rather than acted on.
EARLIEST_LAUNCH = 1_767_225_600  # 2026-01-01
FURTHEST_AHEAD = 366 * 86_400


class LaunchAnswer(BaseModel):
    """What PostHog last said: the launch second, or None for "not live"."""

    live_from: int | None


_last_answer: LaunchAnswer | None = None


def now_seconds() -> int:
    """The moment every usage window is worked out for, as whole UTC seconds."""
    return int(time.time())


async def billing_clock_live(now: int | None = None) -> bool:
    live_from = await launch_second()
    moment = now_seconds() if now is None else now
    return live_from is not None and moment >= live_from


@cached(ttl_seconds=60, maxsize=1)
async def launch_second() -> int | None:
    """The launch second in force, or None while limits run the old way."""
    global _last_answer
    answer = await _posthog_answer()
    if answer is None:
        answer = _last_answer or await _stored_answer()
    elif answer != _last_answer:
        await _store_answer(answer)
        logger.info(f"Usage billing clock: launch second is now {answer.live_from}")
    _last_answer = answer
    if answer is None:
        return _config.usage_billing_clock_live_from
    return answer.live_from


async def _posthog_answer() -> LaunchAnswer | None:
    """PostHog's conclusive answer, or None when it gave none."""
    try:
        value, answered = await read_feature_flag(
            Flag.USAGE_BILLING_CLOCK.value, "system"
        )
    except Exception:
        logger.warning("Usage billing clock flag lookup failed", exc_info=True)
        return None
    if not answered:
        return None
    return parse_launch_answer(value, now_seconds())


def parse_launch_answer(value: Any, now: int) -> LaunchAnswer | None:
    """A flag value as an answer; None when it is malformed."""
    if value is False or value == {}:
        return LaunchAnswer(live_from=None)
    if not isinstance(value, dict) or "live_from" not in value:
        logger.error(f"usage-billing-clock has an unexpected value: {value!r}")
        return None
    live_from = value["live_from"]
    if live_from is None:
        return LaunchAnswer(live_from=None)
    if (
        isinstance(live_from, bool)
        or not isinstance(live_from, int)
        or not EARLIEST_LAUNCH <= live_from <= now + FURTHEST_AHEAD
    ):
        logger.error(f"usage-billing-clock live_from is not a sane second: {value!r}")
        return None
    return LaunchAnswer(live_from=live_from)


async def _stored_answer() -> LaunchAnswer | None:
    try:
        # Without PostHog nothing was ever stored, and there may be no Redis.
        if posthog.get_flag_client() is None:
            return None
        raw = await (await get_redis_async()).get(LAST_ANSWER_KEY)
    except Exception:
        logger.warning("Could not read the stored launch second", exc_info=True)
        return None
    if raw is None:
        return None
    return parse_launch_answer({"live_from": int(raw)}, now_seconds())


async def _store_answer(answer: LaunchAnswer) -> None:
    try:
        redis = await get_redis_async()
        if answer.live_from is None:
            await redis.delete(LAST_ANSWER_KEY)
        else:
            await redis.set(LAST_ANSWER_KEY, str(answer.live_from))
    except Exception:
        logger.warning("Could not store the launch second", exc_info=True)
