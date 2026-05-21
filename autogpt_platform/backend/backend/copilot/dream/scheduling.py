"""Lazy auto-registration of per-user dream-system schedules.

A single helper, :func:`ensure_dream_system_scheduled`, that registers
ALL of a user's dream-related background passes the first time we see
them write a memory:

  * Community rebuild  — ``graphiti-communities-enabled`` (P-1.7)
  * Dream pass         — ``dream-pass-enabled`` (P-0.2)
  * Ratification pass  — ``dream-pass-enabled`` (P-0.4, rides the
                          same master gate as the dream pass)

Per ``dream/p0-spec.md`` §8 and the "holy grail" architecture
discussion: rather than three near-identical helpers, the per-user
jobs are described in :data:`DREAM_SYSTEM_JOBS` and the helper walks
that registry. Adding a new dream-system job (e.g. a P9 daydreaming
pass) is a one-row registry change, not a new helper module.

Three layers of flag gating in this design:

  1. **Registration helper (this file)** — cheapest gate; the flag
     check runs before the Redis SETNX dedup so a flag-off user
     never burns an RPC or a Redis key.
  2. **Scheduler ``@expose`` body** — defense-in-depth for direct
     callers (admin endpoint, external API, ad-hoc Python scripts)
     that bypass this helper. The scheduler refuses to register the
     job when the flag is off and returns a structured "skipped" dict.
  3. **Execution wrapper (``execute_*_sync``)** — runtime gate. If
     the flag was on at registration time and flips off later, the
     scheduled job keeps firing until the cron is deleted; the
     execution wrapper short-circuits to a skip-log so the body
     never runs.

Per-job idempotency: each job has its own Redis SETNX key (NOT a
shared one). Flipping a single flag from off→on after another job
already registered must let the newly-enabled job in without the
shared dedup key blocking it.

Per-job failure isolation: a scheduler RPC failure for one job does
not skip the others — every job's outcome is captured independently
in the helper's return dict.

Failures are logged at WARN and swallowed; the caller (the graphiti
ingestion path) must never break because dream-system registration
failed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)


# Matches the longest cron cadence in the registry (weekly community
# rebuild) so we re-check at least once per cron-tick. Going longer
# leaves a window where Redis says "registered" but the schedule was
# deleted out-of-band; going shorter wastes scheduler RPC calls.
REGISTRATION_TTL_SECONDS = 7 * 24 * 3600


# A SchedulerClient is the caller's handle to the scheduler service.
# We don't import the concrete type here to avoid a circular import
# during the executor's own bootstrap; the helper just calls the
# named coroutine on whatever the caller hands in.
SchedulerLike = Any


@dataclass(frozen=True)
class DreamSystemJob:
    """One row of the dream-system schedule registry.

    The registry is the single source of truth for "what background
    passes does the dream system run per user". Adding a new pass
    (e.g. a P9 daydreaming job) means appending a row here; the helper
    picks it up automatically, the scheduler call gets the right
    job_id, and the Redis dedup key falls out naturally.
    """

    name: str
    """Human-readable, used only for log messages."""

    job_id_prefix: str
    """Job-id naming convention: ``f"{job_id_prefix}_{user_id}"``.
    Must match the scheduler ``@expose`` method's own job_id format."""

    registration_key_prefix: str
    """Redis SETNX key prefix. Each job has its OWN key so flipping
    a single flag mid-life lets only that job re-enter the helper —
    a shared key would block recovery on flag drift."""

    flag: Flag
    """LD feature flag gate. Evaluated per-user before any other work."""

    skip_reason: str
    """The ``reason`` string returned in a flag-off skip result. Keyed
    on the flag value so the helper's return dict is grep-able for
    "why did this user not get a schedule"."""

    register: Callable[[SchedulerLike, str, str], Awaitable[dict]]
    """``(client, user_id, user_timezone) -> awaitable[result dict]``.
    The actual SchedulerClient method that creates the cron job. Kept
    as a callable rather than a method name so the registry decouples
    from the SchedulerClient class symbol (eases mocking in tests
    and future cross-process call shapes)."""


def _register_community_rebuild(
    client: SchedulerLike, user_id: str, user_timezone: str
) -> Awaitable[dict]:
    return client.add_community_rebuild_schedule(
        user_id=user_id, user_timezone=user_timezone
    )


def _register_dream_pass(
    client: SchedulerLike, user_id: str, user_timezone: str
) -> Awaitable[dict]:
    return client.add_dream_pass_schedule(
        user_id=user_id, user_timezone=user_timezone
    )


def _register_ratification_pass(
    client: SchedulerLike, user_id: str, user_timezone: str
) -> Awaitable[dict]:
    return client.add_ratification_pass_schedule(
        user_id=user_id, user_timezone=user_timezone
    )


# The registry. Listed in cron-frequency order (rarest first) so the
# log trail when a new user lands reads "weekly → daily → 6h" — the
# narrative matches how the schedules build up over time.
DREAM_SYSTEM_JOBS: list[DreamSystemJob] = [
    DreamSystemJob(
        name="Community rebuild",
        job_id_prefix="community_rebuild",
        registration_key_prefix="community_rebuild_registered",
        flag=Flag.GRAPHITI_COMMUNITIES_ENABLED,
        skip_reason="graphiti_communities_disabled",
        register=_register_community_rebuild,
    ),
    DreamSystemJob(
        name="Dream pass",
        job_id_prefix="dream_pass",
        registration_key_prefix="dream_pass_registered",
        flag=Flag.DREAM_PASS_ENABLED,
        skip_reason="dream_pass_disabled",
        register=_register_dream_pass,
    ),
    DreamSystemJob(
        name="Ratification pass",
        job_id_prefix="ratification_pass",
        # Distinct key prefix so flipping the master flag from off→on
        # lets the ratification pass register even if the dream pass
        # already used its own key in a prior call.
        registration_key_prefix="ratification_pass_registered",
        # Ratification rides the same master gate as the dream pass —
        # a tentative edge that the dream pass writes is the only
        # thing ratification has to do, so they share a lifecycle.
        flag=Flag.DREAM_PASS_ENABLED,
        skip_reason="dream_pass_disabled",
        register=_register_ratification_pass,
    ),
]


async def _resolve_user_timezone(user_id: str) -> str:
    """Look up the user's IANA timezone from Postgres, falling back to UTC.

    Single DB call — cached at the helper level for the duration of
    one ``ensure_dream_system_scheduled`` invocation so registering
    three jobs doesn't take three round-trips.
    """
    try:
        from prisma import Prisma  # noqa: F401 — ensures registry
        from prisma.models import User

        from backend.data.model import USER_TIMEZONE_NOT_SET

        user = await User.prisma().find_unique(where={"id": user_id})
        if user is None:
            return "UTC"
        tz = (user.timezone or "").strip()
        if not tz or tz == USER_TIMEZONE_NOT_SET:
            return "UTC"
        return tz
    except Exception:
        logger.debug(
            "Could not resolve timezone for user %s; defaulting to UTC",
            user_id[:12],
            exc_info=True,
        )
        return "UTC"


async def _try_redis_setnx(key: str) -> bool | None:
    """SETNX with TTL — returns:
      * True  → we are the first writer; proceed to register.
      * False → another writer already registered within TTL; skip.
      * None  → Redis unavailable; caller should still try to register
                (the scheduler's ``replace_existing=True`` is the
                durable backstop).
    """
    try:
        from backend.data.redis_client import get_redis_async

        redis = await get_redis_async()
        ok = await redis.set(key, "1", nx=True, ex=REGISTRATION_TTL_SECONDS)
        return bool(ok)
    except Exception:
        logger.debug(
            "Redis SETNX failed for %s; falling back to scheduler-side dedup",
            key,
            exc_info=True,
        )
        return None


async def ensure_dream_system_scheduled(user_id: str) -> dict[str, Any]:
    """Idempotently register every flag-enabled dream-system job for a user.

    Fire-and-forget callable — designed to be invoked from the graphiti
    ingestion path on first memory write. Walks :data:`DREAM_SYSTEM_JOBS`,
    gating each entry on its LD flag, then on a per-job Redis SETNX
    dedup key, then on the actual scheduler RPC. Each step's failure
    is isolated; a single bad job never blocks the others.

    Returns a dict keyed by ``job_id_prefix`` so callers can audit
    "what happened for this user this call":

      * Missing key   → that job wasn't attempted (shouldn't happen,
        the registry is iterated unconditionally; here only for
        debugging if we ever short-circuit).
      * ``None``      → SETNX said "already registered", no RPC made.
      * ``{"skipped": True, "reason": ...}`` → flag off, or
        registration RPC failed.
      * Anything else → the scheduler's own result dict (job id,
        next_run_time, etc.).

    Empty ``user_id`` → ``{}`` (no work, no error).
    """
    if not user_id:
        return {}

    results: dict[str, Any] = {}
    tz_cached: str | None = None
    client_cached: SchedulerLike | None = None

    for job in DREAM_SYSTEM_JOBS:
        try:
            # Layer 1 of gating: the LD flag check. Cheapest of the
            # three (sub-ms LD lookup, cached for the user's targeting
            # context), so we always run it first.
            if not await is_feature_enabled(job.flag, user_id):
                results[job.job_id_prefix] = {
                    "skipped": True,
                    "reason": job.skip_reason,
                }
                continue

            # Per-job Redis dedup. Each job's own key — flipping one
            # flag on after another already registered MUST let the
            # newly-enabled job in.
            setnx_key = f"{job.registration_key_prefix}:{user_id}"
            setnx_status = await _try_redis_setnx(setnx_key)
            if setnx_status is False:
                # Already registered within TTL.
                results[job.job_id_prefix] = None
                continue
            # setnx_status is True (we got the key) OR None (Redis
            # unavailable — proceed anyway; scheduler-side
            # ``replace_existing=True`` is the backstop).

            # Resolve the inputs to the scheduler RPC lazily. Both the
            # timezone (DB call) and the client handle are shared
            # across all enabled jobs in this invocation.
            if tz_cached is None:
                tz_cached = await _resolve_user_timezone(user_id)
            if client_cached is None:
                from backend.util.clients import get_scheduler_client

                client_cached = get_scheduler_client()

            result = await job.register(client_cached, user_id, tz_cached)
            logger.info(
                "Dream-system: registered %s for user %s (tz=%s)",
                job.name,
                user_id[:12],
                tz_cached,
            )
            results[job.job_id_prefix] = result
        except Exception:
            logger.warning(
                "Dream-system: failed to register %s for user %s",
                job.name,
                user_id[:12],
                exc_info=True,
            )
            results[job.job_id_prefix] = {
                "skipped": True,
                "reason": "registration_failed",
            }

    return results
