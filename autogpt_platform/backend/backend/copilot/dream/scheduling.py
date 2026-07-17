"""Lazy auto-registration of per-user dream-system schedules.

Two per-user APScheduler cron jobs cover the whole dream system through
P12 (the cross-scope insight cron for P8 will add a third row when it
builds):

  * ``community_rebuild_{user_id}``    — Sun 04:00 user-local (P-1.7),
    direct LLM (not batch), activity-gated inside the function.
  * ``dream_nightly_batch_{user_id}``  — daily 03:00 user-local,
    submits all nightly-batch-family work (dream pass, ratification
    supersession sweep, plus future P2 / P3 / P4 / P11 stages).

Both crons share the same registration helper. Adding a future cron
(P8 cross-scope insight, P9 lucid dream queue if it grows beyond
nightly batch, etc.) is a single row in :data:`DREAM_SYSTEM_JOBS`.

Three layers of flag gating:

  1. **Registration helper (this file)** — cheapest gate; LD flag
     check runs before the Redis CAS-style dedup so a flag-off user
     never burns an RPC or a Redis key.
  2. **Scheduler ``@expose`` body** — defense-in-depth for direct
     callers (admin endpoint, ad-hoc scripts) that bypass this helper.
  3. **Execution wrapper (``execute_*_sync``)** — runtime gate; if the
     flag flips off after registration, the scheduled job still fires
     but short-circuits before the body runs.

**Timezone drift handling.** APScheduler binds the cron trigger to
the timezone at job-creation time — a later ``User.timezone`` change
silently leaves the cron firing at the old local time. To detect and
recover from drift, the Redis dedup key stores the timezone the cron
was registered with (not the literal ``"1"`` it stored historically).
Every call to :func:`ensure_dream_system_scheduled` compares the
stored value to the user's current timezone; on mismatch, re-registers
via ``replace_existing=True``. The eager path (``force_refresh=True``)
is invoked from the ``User.timezone`` update endpoint; the lazy path
catches direct-DB / webhook / SSO writes that bypass the API.

Per-job idempotency: each cron has its OWN Redis dedup key. Flipping
a single flag from off→on after the other cron already registered
must let the newly-enabled cron in.

Per-job failure isolation: a scheduler RPC failure for one cron
surfaces in this call's return dict but never blocks the other crons.

Failures are logged at WARN and swallowed; the caller (the graphiti
ingestion path) must never break because dream-system registration
failed.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from pydantic.dataclasses import dataclass

from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)


# Matches the longest cron cadence in the registry (weekly community
# rebuild) so the lazy drift-detection path re-checks at least once
# per cron-tick. Going longer leaves a window where Redis says
# "registered with tz X" but the cron was deleted out-of-band; going
# shorter wastes scheduler RPC calls when nothing changed.
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
    crons does the dream system run per user". Adding a new cron
    (e.g. weekly cross-scope insight for P8) means appending a row
    here; the helper picks it up automatically, the scheduler call
    gets the right job_id, and the Redis dedup key falls out naturally.
    """

    name: str
    """Human-readable, used only for log messages."""

    job_id_prefix: str
    """Job-id naming convention: ``f"{job_id_prefix}_{user_id}"``.
    Must match the scheduler ``@expose`` method's own job_id format."""

    registration_key_prefix: str
    """Redis dedup key prefix. Each cron has its OWN key so flipping
    a single flag mid-life lets only that cron re-enter the helper —
    a shared key would block recovery on flag drift."""

    flag: Flag
    """LD feature flag gate. Evaluated per-user before any other work."""

    skip_reason: str
    """The ``reason`` string returned in a flag-off skip result. Keyed
    on the flag value so the helper's return dict is grep-able for
    "why did this user not get a schedule"."""

    register: Callable[[SchedulerLike, str, str], Awaitable[dict]]
    """``(client, user_id, user_timezone) -> awaitable[result dict]``.
    The actual SchedulerClient method that creates the cron job."""


def _register_community_rebuild(
    client: SchedulerLike, user_id: str, user_timezone: str
) -> Awaitable[dict]:
    return client.add_community_rebuild_schedule(
        user_id=user_id, user_timezone=user_timezone
    )


def _register_nightly_batch(
    client: SchedulerLike, user_id: str, user_timezone: str
) -> Awaitable[dict]:
    return client.add_nightly_batch_schedule(
        user_id=user_id, user_timezone=user_timezone
    )


# The registry. Listed in cron-frequency order (rarest first) so the
# log trail when a new user lands reads "weekly → daily" — the
# narrative matches how the schedules build up over time. The future
# P8 cross-scope cron (weekly, batch) will land between these two.
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
        name="Dream nightly batch",
        job_id_prefix="dream_nightly_batch",
        # NOT shared with the now-removed individual dream/ratification
        # crons — those keys (``dream_pass_registered``,
        # ``ratification_pass_registered``) are orphaned by the
        # consolidation and naturally expire via their 7-day TTL.
        registration_key_prefix="dream_nightly_batch_registered",
        # The nightly batch cron carries dream pass + ratification
        # supersession + future P2/P3/P4/P11 work. All ride the same
        # master gate; finer-grained flags inside individual submitters
        # control whether each stage actually runs within the cron.
        flag=Flag.DREAM_PASS_ENABLED,
        skip_reason="dream_pass_disabled",
        register=_register_nightly_batch,
    ),
]


async def _resolve_user_timezone(user_id: str) -> str | None:
    """Look up the user's IANA timezone from Postgres.

    Returns ``"UTC"`` only when the answer is authoritative (user
    missing or timezone genuinely unset) and ``None`` when the lookup
    itself failed — a transient DB blip is "unknown", not "UTC", and
    must never silently re-register the user's local-time crons onto
    UTC.

    Single DB call — cached at the helper level for the duration of
    one ``ensure_dream_system_scheduled`` invocation so registering
    multiple crons doesn't take multiple round-trips.

    Routes through the ``user_db()`` accessor, NOT ``User.prisma()``:
    this runs in the copilot-executor and scheduler processes, which
    never connect a local Prisma client. A direct Prisma call raises
    ``ClientNotConnectedError`` on every invocation there — a
    *permanent* failure the keep-existing-schedules fallback was never
    designed for, which silently prevented dream crons from ever being
    registered. The accessor falls back to the DatabaseManager RPC in
    Prisma-less processes (same pattern as ``dream/apply.py``).
    """
    try:
        from backend.data.db_accessors import user_db
        from backend.data.model import USER_TIMEZONE_NOT_SET

        try:
            user = await user_db().get_user_by_id(user_id)
        except ValueError:
            # Authoritative: the user row doesn't exist.
            return "UTC"
        tz = (user.timezone or "").strip()
        if not tz or tz == USER_TIMEZONE_NOT_SET:
            return "UTC"
        return tz
    except Exception:
        logger.warning(
            "Could not resolve timezone for user %s; leaving existing "
            "dream-system schedules untouched this cycle",
            user_id[:12],
            exc_info=True,
        )
        return None


async def _read_registration_tz(user_id: str, key_prefix: str) -> str | None:
    """Read the timezone the cron was last registered with.

    Returns:
      * The stored timezone string when the key exists.
      * ``None`` when the key is missing OR Redis is unavailable. The
        caller treats both as "needs registration" — scheduler-side
        ``replace_existing=True`` makes a redundant call a cheap no-op.
    """
    try:
        from backend.data.redis_client import get_redis_async

        redis = await get_redis_async()
        key = f"{key_prefix}:{user_id}"
        stored = await redis.get(key)
        if stored is None:
            return None
        if isinstance(stored, bytes):
            return stored.decode("utf-8", errors="replace")
        return str(stored)
    except Exception:
        logger.debug(
            "Redis read failed for %s:%s; treating as not-registered",
            key_prefix,
            user_id[:12],
            exc_info=True,
        )
        return None


async def _write_registration_tz(
    user_id: str, key_prefix: str, current_tz: str
) -> None:
    """Persist the timezone we just registered the cron with.

    Best-effort — a Redis write failure means the next call will see
    the key as missing and force a redundant re-register (cheap via
    ``replace_existing=True``).
    """
    try:
        from backend.data.redis_client import get_redis_async

        redis = await get_redis_async()
        key = f"{key_prefix}:{user_id}"
        await redis.set(key, current_tz, ex=REGISTRATION_TTL_SECONDS)
    except Exception:
        logger.debug(
            "Redis write failed for %s:%s; lazy path will re-detect later",
            key_prefix,
            user_id[:12],
            exc_info=True,
        )


async def ensure_dream_system_scheduled(
    user_id: str, *, force_refresh: bool = False
) -> dict[str, Any]:
    """Idempotently register every flag-enabled dream-system cron for a user.

    Fire-and-forget callable from two trigger points:

    * **Lazy path** — called from the graphiti ingestion's
      ``_ensure_worker`` the first time we see a memory write for a
      user in this process. Drift-detects timezone changes via the
      Redis stored value and re-registers when the user's current
      timezone differs from the stored one.
    * **Eager path** — called with ``force_refresh=True`` from the
      ``User.timezone`` update endpoint so a profile change takes
      effect within a single APScheduler tick instead of waiting for
      the dedup key's 7-day TTL to expire.

    Walks :data:`DREAM_SYSTEM_JOBS`, gating each entry on its LD flag,
    then on per-job drift detection, then the actual scheduler RPC.
    Each step's failure is isolated; a single bad cron never blocks
    the others.

    Returns a dict keyed by ``job_id_prefix`` so callers can audit
    "what happened for this user this call":

      * ``None`` — already registered with the current timezone; no
        RPC made. (Lazy path's happy case.)
      * ``{"skipped": True, "reason": "<flag>_disabled"}`` — flag off.
      * ``{"skipped": True, "reason": "timezone_lookup_failed"}`` —
        timezone resolution failed; the existing cron and stored tz
        are left untouched until a later call succeeds.
      * ``{"skipped": True, "reason": "registration_failed"}`` — RPC
        raised; logged.
      * Anything else — the scheduler's own result dict (job id,
        next_run_time, etc.).

    Empty ``user_id`` → ``{}`` (no work, no error).
    """
    if not user_id:
        return {}

    results: dict[str, Any] = {}
    tz_cached: str | None = None
    tz_lookup_failed = False
    client_cached: SchedulerLike | None = None

    for job in DREAM_SYSTEM_JOBS:
        try:
            # Layer 1 of gating: the LD flag check. Cheapest of the
            # three; always run it first.
            if not await is_feature_enabled(job.flag, user_id):
                results[job.job_id_prefix] = {
                    "skipped": True,
                    "reason": job.skip_reason,
                }
                continue

            # Resolve current timezone once per invocation (single DB
            # call shared across enabled crons).
            if tz_cached is None and not tz_lookup_failed:
                tz_cached = await _resolve_user_timezone(user_id)
                tz_lookup_failed = tz_cached is None

            if tz_cached is None:
                # Lookup failed — "unknown" is not "UTC". Re-registering
                # would silently rebind the user's 03:00-local crons to
                # UTC; keep the existing cron and stored tz untouched
                # until a later call resolves the real timezone.
                results[job.job_id_prefix] = {
                    "skipped": True,
                    "reason": "timezone_lookup_failed",
                }
                continue

            # Drift detection (unless caller explicitly forced refresh).
            if not force_refresh:
                stored_tz = await _read_registration_tz(
                    user_id, job.registration_key_prefix
                )
                if stored_tz == tz_cached:
                    # Same tz, still within TTL → no work.
                    results[job.job_id_prefix] = None
                    continue
                if stored_tz is not None and stored_tz != tz_cached:
                    logger.info(
                        "Dream-system: timezone drift for user %s job %s "
                        "(stored=%s, current=%s) — re-registering",
                        user_id[:12],
                        job.name,
                        stored_tz,
                        tz_cached,
                    )
                # else: stored_tz is None → first registration OR
                # Redis was unavailable. Either way, register.

            # Lazy client handle so a fully-flag-off user never even
            # constructs the scheduler client.
            if client_cached is None:
                from backend.util.clients import get_scheduler_client

                client_cached = get_scheduler_client()

            result = await job.register(client_cached, user_id, tz_cached)
            await _write_registration_tz(
                user_id, job.registration_key_prefix, tz_cached
            )
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
