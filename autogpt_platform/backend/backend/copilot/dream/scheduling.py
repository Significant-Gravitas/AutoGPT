"""The dream-system cron table and the owner-timezone lookup.

Two APScheduler cron jobs serve every memory scope (the account, or one
hired expert), and a third the account alone:

  * ``community_rebuild_{scope_key}``    — weekly, owner-local 04:00 on
    Mondays (P-1.7): its crontab ``0 4 * * 0`` means Monday to APScheduler,
    whose weekdays start at 0 = Monday. Direct LLM (not batch),
    activity-gated inside the function.
  * ``dream_nightly_batch_{scope_key}``  — daily 03:00 owner-local,
    submits all nightly-batch-family work (dream pass, ratification
    supersession sweep, plus future P2 / P3 / P4 / P11 stages).
  * ``skill_learning_nightly_{user_id}`` — daily 03:00 owner-local, the
    account's skill-learning pass (``copilot/learning/nightly.py``), on its
    own flag; one pass covers the account's skills and its experts'.

The account's scope key is its user id, so its job ids are unchanged from
when the crons were keyed per user. Registering, pausing and resuming them
is ``registry.py``'s job; this module only says what the crons are. Adding a
future cron (P8 cross-scope insight, ...) is a row in
:data:`DREAM_SYSTEM_JOBS` plus a job-id column on ``MemoryScopeSchedule``.

Community rebuilds are part of memory maintenance. Dream passes and skill
learning retain their independent flags, checked by the registry, scheduler
registration methods and job bodies. Every cron still checks scope lifecycle
and owner restrictions before running.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Literal

from pydantic.dataclasses import dataclass

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import user_db
from backend.data.model import USER_TIMEZONE_NOT_SET
from backend.util.feature_flag import Flag

from .deadline import within_deadline

logger = logging.getLogger(__name__)


# Job-id prefixes, shared with the scheduler's ``@expose`` methods so the two
# can never disagree on a job id.
COMMUNITY_REBUILD_JOB_PREFIX = "community_rebuild"
NIGHTLY_BATCH_JOB_PREFIX = "dream_nightly_batch"
SKILL_LEARNING_JOB_PREFIX = "skill_learning_nightly"


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

    flag: Flag | None
    """Optional feature flag gate, evaluated for the scope's owner."""

    skip_reason: str
    """The ``reason`` recorded when the flag is off, so "why does this scope
    have no schedule" is grep-able."""

    row_field: Literal["community_job_id", "nightly_job_id", "learning_job_id"]
    """The ``MemoryScopeSchedule`` field holding this cron's job id."""

    register: Callable[[SchedulerLike, MemoryScope, str], Awaitable[dict]]
    """``(client, scope, owner_timezone) -> awaitable[result dict]``: the
    SchedulerClient method that creates the cron job."""

    account_only: bool = False
    """The cron runs for the account alone: an expert's scope has none."""

    def job_id(self, scope: MemoryScope) -> str:
        """This cron's APScheduler job id for ``scope``."""
        return f"{self.job_id_prefix}_{scope.scope_key}"

    def serves(self, scope: MemoryScope) -> bool:
        """Whether ``scope`` has this cron at all."""
        return not (self.account_only and scope.is_expert)


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


def _register_skill_learning(
    client: SchedulerLike, scope: MemoryScope, user_timezone: str
) -> Awaitable[dict]:
    return client.add_scope_skill_learning_schedule(
        scope=scope, user_timezone=user_timezone
    )


# Listed in cron-frequency order (rarest first) so a new scope's log trail
# reads "weekly → daily". The future P8 cross-scope cron (weekly, batch)
# lands between the first two.
DREAM_SYSTEM_JOBS: list[DreamSystemJob] = [
    DreamSystemJob(
        name="Community rebuild",
        job_id_prefix=COMMUNITY_REBUILD_JOB_PREFIX,
        flag=None,
        skip_reason="",
        row_field="community_job_id",
        register=_register_community_rebuild,
    ),
    DreamSystemJob(
        name="Dream nightly batch",
        job_id_prefix=NIGHTLY_BATCH_JOB_PREFIX,
        # One master gate for the dream pass, the ratification sweep and
        # the future P2/P3/P4/P11 stages; finer flags inside individual
        # submitters decide whether each stage runs within the cron.
        flag=Flag.DREAM_PASS_ENABLED,
        skip_reason="dream_pass_disabled",
        row_field="nightly_job_id",
        register=_register_nightly_batch,
    ),
    DreamSystemJob(
        name="Skill learning nightly",
        job_id_prefix=SKILL_LEARNING_JOB_PREFIX,
        # Its own gate: skill learning can run with the dream pass off and
        # the dream pass never implies skill learning. Same 03:00
        # user-local cadence, separate lease, separate cost rows. One pass
        # per account reviews the account's and its experts' skills, so an
        # expert's scope has no cron of its own.
        flag=Flag.DREAM_SKILL_LEARNING_ENABLED,
        skip_reason="skill_learning_disabled",
        row_field="learning_job_id",
        register=_register_skill_learning,
        account_only=True,
    ),
]


def jobs_for(scope: MemoryScope) -> list[DreamSystemJob]:
    """The crons ``scope`` has: every row for the account, and those not
    ``account_only`` for an expert."""
    return [job for job in DREAM_SYSTEM_JOBS if job.serves(scope)]


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
    RPC there. The lookup runs under the registry deadline; a timeout is a
    failed lookup.
    """
    try:
        try:
            user = await within_deadline(user_db().get_user_by_id(user_id))
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
