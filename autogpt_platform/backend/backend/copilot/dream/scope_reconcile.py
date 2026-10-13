"""Keep the scheduler's dream-system crons in step with the registry rows.

The registry's own paths act on events (a pause removes a scope's crons, a
resume or an ensure registers them), and events interleave: a stale takedown
lands after a resume, a re-read fails, a registration that read the row
before a timezone change lands after it. Each leaves an ACTIVE row whose
recorded crons the scheduler lacks or runs in another timezone, and an
ensure, trusting the recorded job ids, never notices.

:func:`reconcile` is the level-triggered backstop, run by the scheduler every
``RECONCILE_INTERVAL_MINUTES``. It lists the scheduler's dream-system crons
and every registry row, then settles each scope either names, in scope-key
order:

- an ACTIVE row's crons are the ones it records: each the scheduler lacks,
  or runs in a timezone other than the row's, is registered in the row's
  timezone. A cron the row does not record is the registry's to decide (an
  ensure registers what the owner's flags enable), so no flag is read here;
- every cron of a PAUSED or WIPED row, or of an expert whose row is gone, is
  removed. An account's crons with no row stay: they predate the registry,
  and the cron gate lets them run.

Each scope is read again right before it is changed, so a pause, resume or
timezone change landing during the pass wins, and one landing between that
read and the change is set right by the next pass. Every change is
idempotent, so two schedulers reconciling at once end in the same state.

A pass cut off at the scheduler's ceiling leaves the last scope it settled
in Redis (``CURSOR_KEY``) and the next pass carries on after it; one that
settles the last scope clears it. So a lost or stale cron is back after the
next pass that reaches its scope: about one interval plus that pass, not a
fixed bound. Without Redis every pass starts from the first scope. A pass
that cannot list the crons or the rows changes nothing.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from prisma.enums import MemoryScopeScheduleState
from pydantic import BaseModel, Field

from backend.copilot.graphiti.scope import MemoryScope
from backend.data import redis_client
from backend.data.db_accessors import memory_schedule_db
from backend.data.memory_schedule import MemoryScopeSchedule
from backend.data.redis_client import AsyncRedisClient

from .deadline import within_deadline
from .scheduling import DREAM_SYSTEM_JOBS, MARKER_TIMEOUT_SECONDS, DreamSystemJob
from .scope_jobs import recorded_job_id, scope_label

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL_MINUTES = 30
# Rows read from the registry per call while a pass lists them.
PAGE_SIZE = 500
# The last scope key a pass that was cut off settled. It outlives a day of
# passes that could not write it, then the next pass starts over.
CURSOR_KEY = "dream:cron_reconcile:after"
CURSOR_TTL_SECONDS = 24 * 3600
_ACTIVE = MemoryScopeScheduleState.ACTIVE


class ListedCron(BaseModel):
    """One dream-system cron the scheduler holds: its job id, the scope its
    kwargs name and the timezone its trigger runs in."""

    job_id: str
    scope: MemoryScope
    timezone: str | None


class CronStore(Protocol):
    """The scheduler side of a pass: listing its dream-system crons,
    registering one cron of a scope, and removing every dream-system cron of
    a scope."""

    async def listing(self) -> Sequence[ListedCron]: ...

    async def add(
        self, job: DreamSystemJob, scope: MemoryScope, user_timezone: str
    ) -> None: ...

    async def remove(self, scope: MemoryScope) -> None: ...


class ReconcileReport(BaseModel):
    """What one pass changed: the job ids it registered, the scopes whose
    crons it removed, the scopes a change failed for, and whether it settled
    the last scope."""

    restored: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    finished: bool = False


async def reconcile(store: CronStore) -> ReconcileReport:
    """One pass: list the scheduler's crons and every registry row, then
    settle each scope after the cursor in scope-key order. Never raises; the
    scheduler cancels it at its ceiling, and the cursor keeps its place."""
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
    place = _Place()
    after = await place.read()
    for key in sorted(rows.keys() | held.keys()):
        if after is not None and key <= after:
            continue
        if await _settle(rows.get(key), held.get(key, []), store, report):
            await place.keep(key)
    await place.keep(None)
    report.finished = True
    return report


async def _settle(
    row: MemoryScopeSchedule | None,
    crons: list[ListedCron],
    store: CronStore,
    report: ReconcileReport,
) -> bool:
    """Settle one scope; whether it had to read the scope again, the work
    the cursor keeps."""
    if row is not None and row.state == _ACTIVE:
        return await _restore(row, crons, store, report)
    if crons and _straggling(crons[0].scope, row):
        return await _remove(crons[0].scope, store, report)
    return False


async def _restore(
    row: MemoryScopeSchedule,
    crons: list[ListedCron],
    store: CronStore,
    report: ReconcileReport,
) -> bool:
    """Register each cron ``row`` records that the scheduler lacks or runs in
    another timezone, as a fresh read of the row still says."""
    scope = MemoryScope.build(row.user_id, row.expert_id)
    zones = {cron.job_id: cron.timezone for cron in crons}
    if not _off(row, scope, zones):
        return False
    try:
        current = await _read(scope)
    except Exception:
        logger.warning(f"Could not read {scope_label(scope)} again", exc_info=True)
        return True
    if current is None or current.state != _ACTIVE:
        return True
    off = _off(current, scope, zones)
    try:
        for job in off:
            await store.add(job, scope, current.timezone)
            report.restored.append(job.job_id(scope))
    except Exception:
        logger.warning(
            f"Could not restore the crons of {scope_label(scope)}", exc_info=True
        )
        report.failed.append(scope.scope_key)
        return True
    if off:
        logger.info(
            f"Memory cron reconcile registered {[job.name for job in off]} for "
            f"{scope_label(scope)} in {current.timezone}"
        )
    return True


def _off(
    row: MemoryScopeSchedule, scope: MemoryScope, zones: dict[str, str | None]
) -> list[DreamSystemJob]:
    """The crons ``row`` records that ``zones`` (each listed job id's
    timezone) lacks, or has in a timezone other than the row's."""
    return [
        job
        for job in DREAM_SYSTEM_JOBS
        if recorded_job_id(row, job) is not None
        and (job.job_id(scope) not in zones or zones[job.job_id(scope)] != row.timezone)
    ]


async def _remove(
    scope: MemoryScope, store: CronStore, report: ReconcileReport
) -> bool:
    """Remove the crons of a scope that should have none, once a fresh read
    still says so."""
    try:
        row = await _read(scope)
    except Exception:
        logger.warning(f"Could not read {scope_label(scope)} again", exc_info=True)
        return True
    if not _straggling(scope, row):
        return True
    try:
        await store.remove(scope)
    except Exception:
        logger.warning(
            f"Could not remove the crons of {scope_label(scope)}", exc_info=True
        )
        report.failed.append(scope.scope_key)
        return True
    report.removed.append(scope.scope_key)
    logger.info(f"Memory cron reconcile removed the crons of {scope_label(scope)}")
    return True


def _straggling(scope: MemoryScope, row: MemoryScopeSchedule | None) -> bool:
    """Whether crons of ``scope`` must go: its row says PAUSED or WIPED, or it
    is an expert's and has no row. An account with no row keeps its crons."""
    if row is None:
        return scope.is_expert
    return row.state != _ACTIVE


async def _read(scope: MemoryScope) -> MemoryScopeSchedule | None:
    """The scope's row as it stands now, None when it has none, under the
    registry deadline. Raises when it cannot be read: the caller then
    changes nothing for the scope this pass."""
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


class _Place:
    """Where a pass stands (``CURSOR_KEY``), kept for a successor in case it
    is cut off. Best-effort: once a Redis call fails the pass stops keeping
    it, so an outage costs a pass one timeout, not one per scope, and its
    successor starts further back, never further on."""

    def __init__(self) -> None:
        self._redis_up = True

    async def read(self) -> str | None:
        """The last scope a pass that was cut off settled; None to start
        from the first."""
        place = await self._call(lambda redis: redis.get(CURSOR_KEY))
        return place.decode() if isinstance(place, bytes) else place

    async def keep(self, scope_key: str | None) -> None:
        """Keep ``scope_key`` as settled; None once the last one is."""
        if scope_key is None:
            await self._call(lambda redis: redis.delete(CURSOR_KEY))
        else:
            await self._call(
                lambda redis: redis.set(CURSOR_KEY, scope_key, ex=CURSOR_TTL_SECONDS)
            )

    async def _call(self, command: Callable[[AsyncRedisClient], Awaitable[Any]]) -> Any:
        if not self._redis_up:
            return None
        try:
            async with asyncio.timeout(MARKER_TIMEOUT_SECONDS):
                return await command(await redis_client.get_redis_async())
        except Exception:
            self._redis_up = False
            logger.warning(
                "Memory cron reconcile cannot reach Redis; this pass keeps no "
                "place for the next one",
                exc_info=True,
            )
            return None
