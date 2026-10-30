"""Keep the scheduler's dream-system crons in step with the registry rows.

The registry acts on events, and events interleave (a stale takedown after a
resume, a failed re-read, a registration landing after a timezone change, a
restore landing after a delete), leaving the scheduler's crons out of step
with a row.

:func:`reconcile` is the level-triggered backstop the scheduler runs every
``RECONCILE_INTERVAL_MINUTES``. It lists the dream-system crons and every row,
then walks the scopes as a ring, from just after the last scope a pass in this
process took (a random one after the process starts), and settles each the
listing shows out of step, on the row as read again just before:

  * an ACTIVE row's crons are the ones it records, in its timezone: what the
    scheduler lacks or runs elsewhere is registered, tagged as the
    reconcile's (``RECONCILED_SUFFIX`` on the job's name);
  * a tagged cron its row does not record, or whose row is not ACTIVE or
    gone, is removed, before the scope's crons are repaired and whatever
    that repair does. So a restore that raced a delete or a pause, failed
    partway, was cancelled or finished after its pass, leaves at worst a
    tagged cron that the next pass reaching the scope takes away, even
    while the scope's repair keeps failing;
  * every cron of a PAUSED or WIPED row, of an expert with no row, or of an
    account with no row whose user is gone, is removed. Which of those
    accounts are gone is asked once per pass, in pages, each under the
    registry deadline; when that cannot be answered, their crons wait for a
    later pass. An untagged cron no row records (a live account's from
    before the registry, a registration whose record failed) is the
    registry's, and stays.

No flag is read here. Every change is idempotent, so two schedulers (a
rollout's overlap) each walk on their own and converge. A pass takes a scope
as its place before it works it, so a pass cut off at its ceiling hands the
rest of the ring to the next: a lost, stale or stray cron is put right once
the walk next reaches its scope. A pass that cannot list the crons or the rows
changes nothing.

Known limit: a pass's changes are not fenced against the registry's. An
ensure or a resume that registers and records a cron after the pass's
re-read can lose it to the pass's take-back or removal, and the next pass
that reaches the scope restores it from the row. For the nightly learning
cron that can cost one night's run; its sources persist, so the next
night's run reviews them.
"""

import bisect
import logging
import random
from collections.abc import Awaitable, Sequence
from typing import ClassVar, Protocol

from prisma.enums import MemoryScopeScheduleState
from pydantic import BaseModel, Field

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import memory_schedule_db
from backend.data.memory_schedule import MemoryScopeSchedule

from .deadline import within_deadline
from .scheduling import DREAM_SYSTEM_JOBS, DreamSystemJob
from .scope_jobs import recorded_job_id, scope_label

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL_MINUTES = 30
# Rows read from the registry per call while a pass lists them.
PAGE_SIZE = 500
# Ends the name of every cron the reconcile registers: a later pass takes
# such a cron back once its row stops recording it.
RECONCILED_SUFFIX = " (reconciled)"
_ACTIVE = MemoryScopeScheduleState.ACTIVE


class ListedCron(BaseModel):
    """One dream-system cron the scheduler holds: its job id, the scope its
    kwargs name, the timezone its trigger runs in, and whether the reconcile
    registered it."""

    job_id: str
    scope: MemoryScope
    timezone: str | None
    reconciled: bool = False


class CronStore(Protocol):
    """The scheduler side of a pass: its dream-system crons, and adding one
    cron of a scope (tagged as the reconcile's), dropping one, or removing
    them all."""

    async def listing(self) -> Sequence[ListedCron]: ...

    async def add(
        self, job: DreamSystemJob, scope: MemoryScope, user_timezone: str
    ) -> None: ...

    async def drop(self, job: DreamSystemJob, scope: MemoryScope) -> None: ...

    async def remove(self, scope: MemoryScope) -> None: ...


class ReconcileReport(BaseModel):
    """What one pass changed: the job ids it registered and the tagged ones
    it took back, the scopes whose crons it removed, and the scopes a change
    failed for."""

    restored: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


async def reconcile(store: CronStore) -> ReconcileReport:
    """One pass (above). Never raises; the scheduler cancels it at its
    ceiling, and the place it took carries the next pass on."""
    report = ReconcileReport()
    try:
        listed = await store.listing()
        rows = {row.scope_key: row for row in await _all_rows()}
    except Exception:
        logger.warning(
            "Memory cron reconcile skipped: the crons or the registry could "
            "not be listed",
            exc_info=True,
        )
        return report
    held: dict[str, list[ListedCron]] = {}
    for cron in listed:
        held.setdefault(cron.scope.scope_key, []).append(cron)
    gone = await _gone_accounts(rows, held)
    for key in _ring(sorted(rows.keys() | held.keys())):
        row, crons = rows.get(key), held.get(key, [])
        if not _out_of_step(row, crons, gone):
            continue
        _Place.last = key
        scope = _scope_of(row) if row else crons[0].scope
        await _settle(scope, crons, store, report, gone)
    return report


async def _gone_accounts(
    rows: dict[str, MemoryScopeSchedule], held: dict[str, list[ListedCron]]
) -> set[str]:
    """The owners of account crons with no row whose user is gone, asked in
    pages under the registry deadline. Empty when the answer cannot be had:
    those crons wait for a later pass rather than go on a guess."""
    owners = sorted(
        {
            crons[0].scope.owner_user_id
            for key, crons in held.items()
            if key not in rows and not crons[0].scope.is_expert
        }
    )
    live: set[str] = set()
    try:
        for start in range(0, len(owners), PAGE_SIZE):
            page = owners[start : start + PAGE_SIZE]
            live |= await within_deadline(memory_schedule_db().existing_user_ids(page))
    except Exception:
        logger.warning(
            "Memory cron reconcile could not tell which accounts are gone; "
            "their crons wait for a later pass",
            exc_info=True,
        )
        return set()
    return set(owners) - live


def _ring(keys: list[str]) -> list[str]:
    """``keys`` from just after the last scope a pass in this process took,
    wrapping round; from a random one when it has taken none."""
    if not keys:
        return []
    if _Place.last is None:
        start = random.randrange(len(keys))
    else:
        start = bisect.bisect_right(keys, _Place.last)
    return keys[start:] + keys[:start]


class _Place:
    """The last scope a pass in this process began to settle."""

    last: ClassVar[str | None] = None


def _out_of_step(
    row: MemoryScopeSchedule | None, crons: list[ListedCron], gone: set[str]
) -> bool:
    """Whether the listing shows the scope out of step: an ACTIVE row's
    recorded crons missing or elsewhere, a tagged cron it does not record,
    or crons that must go."""
    if row is not None and row.state == _ACTIVE:
        scope = _scope_of(row)
        return bool(_off(row, scope, _zones(crons))) or bool(_stray(row, crons))
    if crons and _straggling(crons[0].scope, row, gone):
        return True
    return bool(_stray(row, crons))


async def _settle(
    scope: MemoryScope,
    crons: list[ListedCron],
    store: CronStore,
    report: ReconcileReport,
    gone: set[str],
) -> None:
    """Settle the scope on its row as read now: remove the crons of a scope
    that must have none; else take back the tagged crons the row does not
    record, then restore an ACTIVE row's crons. Each step stands on its own,
    the take-back first, so a restore that keeps failing never keeps a stray
    running."""
    try:
        row = await _read(scope)
    except Exception:
        logger.warning(f"Could not read {scope_label(scope)} again", exc_info=True)
        return
    if _straggling(scope, row, gone):
        await _step(scope, report, _remove(scope, store, report))
        return
    await _step(scope, report, _drop(scope, _stray(row, crons), store, report))
    if row is not None and row.state == _ACTIVE:
        await _step(scope, report, _restore(scope, row, crons, store, report))


async def _step(
    scope: MemoryScope, report: ReconcileReport, change: Awaitable[None]
) -> None:
    """Make one of the scope's changes; a failure is logged and the scope
    reported failed once, and the pass goes on."""
    try:
        await change
    except Exception:
        logger.warning(
            f"Could not settle the crons of {scope_label(scope)}", exc_info=True
        )
        if scope.scope_key not in report.failed:
            report.failed.append(scope.scope_key)


async def _remove(
    scope: MemoryScope, store: CronStore, report: ReconcileReport
) -> None:
    await store.remove(scope)
    report.removed.append(scope.scope_key)
    logger.info(f"Memory cron reconcile removed the crons of {scope_label(scope)}")


async def _restore(
    scope: MemoryScope,
    row: MemoryScopeSchedule,
    crons: list[ListedCron],
    store: CronStore,
    report: ReconcileReport,
) -> None:
    """Register, tagged, each cron the row records that ``crons`` lacks or
    runs in another timezone."""
    for job in _off(row, scope, _zones(crons)):
        await store.add(job, scope, row.timezone)
        report.restored.append(job.job_id(scope))
        logger.info(
            f"Memory cron reconcile registered {job.name} for "
            f"{scope_label(scope)} in {row.timezone}"
        )


async def _drop(
    scope: MemoryScope,
    jobs: list[DreamSystemJob],
    store: CronStore,
    report: ReconcileReport,
) -> None:
    """Take back the tagged crons ``jobs`` of ``scope``."""
    for job in jobs:
        await store.drop(job, scope)
        report.dropped.append(job.job_id(scope))
        logger.info(
            f"Memory cron reconcile took back {job.name} of {scope_label(scope)}: "
            "its row does not record it"
        )


def _off(
    row: MemoryScopeSchedule, scope: MemoryScope, zones: dict[str, str | None]
) -> list[DreamSystemJob]:
    """The crons ``row`` records that ``zones`` (each listed job id's
    timezone) lacks, or has in a timezone other than the row's."""
    return [
        job
        for job in DREAM_SYSTEM_JOBS
        if recorded_job_id(row, job) is not None
        and zones.get(job.job_id(scope), "") != row.timezone
    ]


def _stray(
    row: MemoryScopeSchedule | None, crons: list[ListedCron]
) -> list[DreamSystemJob]:
    """The tagged crons among ``crons`` that ``row`` does not keep: it does
    not record them, is not ACTIVE, or is gone."""
    tagged = {cron.job_id for cron in crons if cron.reconciled}
    if not tagged:
        return []
    scope = crons[0].scope
    return [
        job
        for job in DREAM_SYSTEM_JOBS
        if job.job_id(scope) in tagged
        and (row is None or row.state != _ACTIVE or recorded_job_id(row, job) is None)
    ]


def _straggling(
    scope: MemoryScope, row: MemoryScopeSchedule | None, gone: set[str]
) -> bool:
    """Whether crons of ``scope`` must go: its row says PAUSED or WIPED, or it
    has no row and is an expert's or a gone account's. A live account with
    no row keeps its crons until the registry adopts them."""
    if row is None:
        return scope.is_expert or scope.owner_user_id in gone
    return row.state != _ACTIVE


def _scope_of(row: MemoryScopeSchedule) -> MemoryScope:
    return MemoryScope.build(row.user_id, row.expert_id)


def _zones(crons: list[ListedCron]) -> dict[str, str | None]:
    return {cron.job_id: cron.timezone for cron in crons}


async def _read(scope: MemoryScope) -> MemoryScopeSchedule | None:
    """The scope's row as it stands now, None when it has none, under the
    registry deadline. Raises when it cannot be read."""
    return await within_deadline(
        memory_schedule_db().get_scope_schedule(scope.owner_user_id, scope.scope_key)
    )


async def _all_rows() -> list[MemoryScopeSchedule]:
    """Every registry row, paged by scope key, each page under the registry
    deadline."""
    rows: list[MemoryScopeSchedule] = []
    while True:
        after = rows[-1].scope_key if rows else None
        page = await within_deadline(
            memory_schedule_db().list_scope_schedules(after=after, limit=PAGE_SIZE)
        )
        rows += page
        if len(page) < PAGE_SIZE:
            return rows
