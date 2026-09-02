"""Lazy auto-registration of the per-user task-overseer cron.

Mirrors ``backend/copilot/briefing/scheduling.py`` minus the timezone
machinery: the overseer fires every 15 minutes, so local time is
irrelevant and the Redis marker only has to bound out-of-band drift (an
externally deleted cron). Fired via ``asyncio.create_task`` from the hot
chat path — never raises.
"""

from __future__ import annotations

import logging

from backend.data.redis_client import get_redis_async
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

REGISTRATION_TTL_SECONDS = 7 * 24 * 3600

OVERSEER_REGISTRATION_PREFIX = "task_overseer_registered"


async def ensure_task_overseer_scheduled(user_id: str) -> None:
    """Idempotently register the 15-minute overseer cron for a user.

    The pass itself no-ops cheaply when the user has no open tasks, so
    registration errs toward "on": any chat turn from a flagged user
    keeps the cron alive.
    """
    try:
        if not user_id:
            return
        if not await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False):
            return
        if not await is_feature_enabled(
            Flag.EXPERT_TASK_MANAGEMENT, user_id, default=False
        ):
            return
        if not await _claim_marker(user_id):
            return

        from backend.util.clients import get_scheduler_client

        try:
            await get_scheduler_client().add_task_overseer_schedule(user_id=user_id)
        except Exception:
            await _release_marker(user_id)
            raise
        logger.info("Task overseer: registered cron for user %s", user_id[:12])
    except Exception:
        logger.warning(
            "Task overseer: failed to register cron for user %s",
            user_id[:12],
            exc_info=True,
        )


async def _claim_marker(user_id: str) -> bool:
    """Atomically take the registration slot for *user_id*.

    ``SET NX`` makes the check-and-write one Redis op, so two chat turns
    racing the first registration produce one scheduler call — a second
    ``add_task_overseer_schedule`` would replace the job and reset its
    pending ``next_run_time``. A Redis failure reads as "unclaimed" so an
    outage never silently leaves a user without the cron.
    """
    try:
        redis = await get_redis_async()
        claimed = await redis.set(
            f"{OVERSEER_REGISTRATION_PREFIX}:{user_id}",
            "1",
            ex=REGISTRATION_TTL_SECONDS,
            nx=True,
        )
        return bool(claimed)
    except Exception:
        logger.debug(
            "Redis claim failed for %s:%s; treating as not-registered",
            OVERSEER_REGISTRATION_PREFIX,
            user_id[:12],
            exc_info=True,
        )
        return True


async def _release_marker(user_id: str) -> None:
    """Give the slot back when registration failed after the claim, so the
    next turn retries instead of waiting out the TTL."""
    try:
        redis = await get_redis_async()
        await redis.delete(f"{OVERSEER_REGISTRATION_PREFIX}:{user_id}")
    except Exception:
        logger.debug(
            "Redis release failed for %s:%s; marker expires with its TTL",
            OVERSEER_REGISTRATION_PREFIX,
            user_id[:12],
            exc_info=True,
        )
