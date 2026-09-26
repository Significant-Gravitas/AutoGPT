"""The dream-system cron table, the owner-timezone lookup and the Redis markers.

Two APScheduler cron jobs serve every memory scope (the account, or one
hired expert):

  * ``community_rebuild_{scope_key}``    — Sun 04:00 owner-local (P-1.7),
    direct LLM (not batch), activity-gated inside the function.
  * ``dream_nightly_batch_{scope_key}``  — daily 03:00 owner-local,
    submits all nightly-batch-family work (dream pass, ratification
    supersession sweep, plus future P2 / P3 / P4 / P11 stages).

The account's scope key is its user id, so its job ids are unchanged from
when the crons were keyed per user. Registering, pausing and resuming them
is ``registry.py``'s job; this module only says what the crons are. Adding a
future cron (P8 cross-scope insight, ...) is a row in
:data:`DREAM_SYSTEM_JOBS` plus a job-id column on ``MemoryScopeSchedule``.

Three layers of flag gating, all on the scope owner's flags:

  1. **Registry** — the cheapest gate; runs before any database read, so
     a flag-off user costs neither a query nor a scheduler RPC.
  2. **Scheduler ``@expose`` method** — defense-in-depth for direct
     callers (admin endpoint, ad-hoc scripts) that bypass the registry.
  3. **Job body** — if the flag flips off after registration, the job
     still fires but short-circuits before the work runs.

**Redis markers.** ``{prefix}:{scope_key}`` holds the timezone a cron was
last registered in, for seven days. The registry used to decide on them;
it now reads the ``MemoryScopeSchedule`` table instead and only keeps the
markers written (and cleared on an in-band delete) as a cache.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Literal

from pydantic.dataclasses import dataclass

from backend.copilot.graphiti.scope import MemoryScope
from backend.data import redis_client
from backend.data.db_accessors import user_db
from backend.data.model import USER_TIMEZONE_NOT_SET
from backend.util.feature_flag import Flag

logger = logging.getLogger(__name__)


# Matches the longest cron cadence in the table (weekly community rebuild).
REGISTRATION_TTL_SECONDS = 7 * 24 * 3600
# A marker is a cache: never wait longer than this on Redis for one.
MARKER_TIMEOUT_SECONDS = 5

# Redis marker prefixes, one per cron. Must stay in sync with the table rows
# below — pinned by a test.
COMMUNITY_REBUILD_REGISTRATION_PREFIX = "community_rebuild_registered"
NIGHTLY_BATCH_REGISTRATION_PREFIX = "dream_nightly_batch_registered"

# Job-id prefixes, shared with the scheduler's ``@expose`` methods so the two
# can never disagree on a job id.
COMMUNITY_REBUILD_JOB_PREFIX = "community_rebuild"
NIGHTLY_BATCH_JOB_PREFIX = "dream_nightly_batch"


# A SchedulerClient is the caller's handle to the scheduler service. The
# concrete type is not imported here to avoid a circular import during the
# executor's own bootstrap; the table just calls the named coroutine on
# whatever the caller hands in.
SchedulerLike = Any


@dataclass(frozen=True)
class DreamSystemJob:
    """One row of the dream-system cron table."""

    name: str
    """Human-readable, used only for log messages."""

    job_id_prefix: str
    """Job ids are ``f"{job_id_prefix}_{scope_key}"``."""

    registration_key_prefix: str
    """Redis marker prefix. Each cron has its own marker."""

    flag: Flag
    """LD feature flag gate, evaluated for the scope's owner."""

    skip_reason: str
    """The ``reason`` recorded when the flag is off, so "why does this scope
    have no schedule" is grep-able."""

    row_field: Literal["community_job_id", "nightly_job_id"]
    """The ``MemoryScopeSchedule`` field holding this cron's job id."""

    register: Callable[[SchedulerLike, MemoryScope, str], Awaitable[dict]]
    """``(client, scope, owner_timezone) -> awaitable[result dict]``: the
    SchedulerClient method that creates the cron job."""

    def job_id(self, scope: MemoryScope) -> str:
        """This cron's APScheduler job id for ``scope``."""
        return f"{self.job_id_prefix}_{scope.scope_key}"


def _register_community_rebuild(
    client: SchedulerLike, scope: MemoryScope, user_timezone: str
) -> Awaitable[dict]:
    return client.add_scope_community_rebuild_schedule(
        scope=scope, user_timezone=user_timezone
    )


def _register_nightly_batch(
    client: SchedulerLike, scope: MemoryScope, user_timezone: str
) -> Awaitable[dict]:
    return client.add_scope_nightly_batch_schedule(
        scope=scope, user_timezone=user_timezone
    )


# Listed in cron-frequency order (rarest first) so a new scope's log trail
# reads "weekly → daily". The future P8 cross-scope cron (weekly, batch)
# lands between these two.
DREAM_SYSTEM_JOBS: list[DreamSystemJob] = [
    DreamSystemJob(
        name="Community rebuild",
        job_id_prefix=COMMUNITY_REBUILD_JOB_PREFIX,
        registration_key_prefix=COMMUNITY_REBUILD_REGISTRATION_PREFIX,
        flag=Flag.GRAPHITI_COMMUNITIES_ENABLED,
        skip_reason="graphiti_communities_disabled",
        row_field="community_job_id",
        register=_register_community_rebuild,
    ),
    DreamSystemJob(
        name="Dream nightly batch",
        job_id_prefix=NIGHTLY_BATCH_JOB_PREFIX,
        registration_key_prefix=NIGHTLY_BATCH_REGISTRATION_PREFIX,
        # One master gate for the dream pass, the ratification sweep and
        # the future P2/P3/P4/P11 stages; finer flags inside individual
        # submitters decide whether each stage runs within the cron.
        flag=Flag.DREAM_PASS_ENABLED,
        skip_reason="dream_pass_disabled",
        row_field="nightly_job_id",
        register=_register_nightly_batch,
    ),
]


def dream_system_job(job_id_prefix: str) -> DreamSystemJob:
    """The table row for ``job_id_prefix``; raises ``KeyError`` if unknown."""
    for job in DREAM_SYSTEM_JOBS:
        if job.job_id_prefix == job_id_prefix:
            return job
    raise KeyError(job_id_prefix)


async def resolve_user_timezone(user_id: str) -> str | None:
    """The owner's IANA timezone, which every scope of theirs is scheduled in.

    Returns ``"UTC"`` only when the answer is authoritative (user missing or
    timezone genuinely unset) and ``None`` when the lookup itself failed — a
    transient DB blip is "unknown", not "UTC", and must never silently
    re-register the owner's local-time crons onto UTC.

    Routes through the ``user_db()`` accessor, NOT ``User.prisma()``: this
    runs in the copilot-executor and scheduler processes, which never connect
    a local Prisma client, and the accessor falls back to the DatabaseManager
    RPC there.
    """
    try:
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


async def write_registration_marker(
    scope: MemoryScope, key_prefix: str, user_timezone: str
) -> None:
    """Cache the timezone a cron was just registered in. Best-effort, and
    bounded: a hire must not wait on the Redis client's connect retries."""
    try:
        await asyncio.wait_for(
            _set_marker(_registration_key(scope, key_prefix), user_timezone),
            timeout=MARKER_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.debug("Redis write failed for %s:%s", key_prefix, scope.scope_key[:12])


async def clear_registration_marker(scope: MemoryScope, key_prefix: str) -> None:
    """Delete one cron's Redis marker after the cron was removed in-band.

    Single-key DEL so it routes on Redis Cluster. Best-effort and bounded —
    on Redis failure the marker simply expires via its TTL.
    """
    try:
        await asyncio.wait_for(
            _delete_marker(_registration_key(scope, key_prefix)),
            timeout=MARKER_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning(
            "Redis delete failed for %s:%s; marker will expire via TTL",
            key_prefix,
            scope.scope_key[:12],
            exc_info=True,
        )


def _registration_key(scope: MemoryScope, key_prefix: str) -> str:
    return scope.redis_key("registration", registration_prefix=key_prefix)


async def _set_marker(key: str, value: str) -> None:
    redis = await redis_client.get_redis_async()
    await redis.set(key, value, ex=REGISTRATION_TTL_SECONDS)


async def _delete_marker(key: str) -> None:
    redis = await redis_client.get_redis_async()
    await redis.delete(key)
