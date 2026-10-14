"""Keep the scheduler's dream-system crons in step with the registry rows.

The registry acts on events, and events interleave (a stale takedown after a
resume, a failed re-read, a registration landing after a timezone change),
leaving an ACTIVE row whose recorded crons the scheduler lacks or runs in
another timezone, which an ensure, trusting the recorded ids, never notices.

:func:`reconcile` is the level-triggered backstop the scheduler runs every
``RECONCILE_INTERVAL_MINUTES``. It lists the dream-system crons and every
row, then settles each scope in key order. An ACTIVE row's crons are the ones
it records, in its timezone: what the scheduler lacks or runs elsewhere is
registered, and taken back if a fresh read shows the row stopped recording it
meanwhile (a pause, a deleted cron). A cron no row records is the registry's
to decide, so no flag is read here. Every cron of a PAUSED or WIPED row, or of
an expert with no row, is removed; an account's crons with no row predate the
registry and stay. Each scope is read again before it is changed, and every
change is idempotent, so two schedulers reconciling at once converge (one may
replay scopes the other settled, never skip one).

A pass keeps its place before it works a scope, in Redis (``CURSOR_KEY``) and
in this process, and the next pass carries on after it: one cut off at its
ceiling, mid-scope or not, still moves on, and the scope it was cut in is
retried next cycle. A pass that settles the last scope clears the place. A
lost or stale cron is back after the next pass that reaches its scope: about
one interval plus that pass, not a fixed bound. A pass that cannot list the
crons or the rows changes nothing.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, ClassVar, Protocol

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
# The last scope a pass began to work, for the pass after it.
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
    """The scheduler side of a pass: its dream-system crons, and adding one
    cron of a scope, dropping one, or removing them all."""

    async def listing(self) -> Sequence[ListedCron]: ...

    async def add(
        self, job: DreamSystemJob, scope: MemoryScope, user_timezone: str
    ) -> None: ...

    async def drop(self, job: DreamSystemJob, scope: MemoryScope) -> None: ...

    async def remove(self, scope: MemoryScope) -> None: ...


class ReconcileReport(BaseModel):
    """What one pass changed: the job ids it registered and kept, the scopes
    whose crons it removed, and the scopes a change failed for."""

    restored: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


async def reconcile(store: CronStore) -> ReconcileReport:
    """One pass (above). Never raises; the scheduler cancels it at its
    ceiling, and the place it kept carries the next pass on."""
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
        row, crons = rows.get(key), held.get(key, [])
        if (after is not None and key <= after) or not _out_of_step(row, crons):
            continue
        await place.keep(key)
        await _settle(row, crons, store, report)
    await place.keep(None)
    return report


def _out_of_step(row: MemoryScopeSchedule | None, crons: list[ListedCron]) -> bool:
    """Whether the listing shows the scope out of step: an ACTIVE row's
    recorded crons missing or elsewhere, or crons it should not have."""
    if row is not None and row.state == _ACTIVE:
        return bool(_off(row, _scope_of(row), _zones(crons)))
    return bool(crons) and _straggling(crons[0].scope, row)


async def _settle(
    row: MemoryScopeSchedule | None,
    crons: list[ListedCron],
    store: CronStore,
    report: ReconcileReport,
) -> None:
    if row is not None and row.state == _ACTIVE:
        await _restore(_scope_of(row), _zones(crons), store, report)
    else:
        await _remove(crons[0].scope, store, report)


async def _restore(
    scope: MemoryScope,
    zones: dict[str, str | None],
    store: CronStore,
    report: ReconcileReport,
) -> None:
    """Register each cron the scope's row records that ``zones`` lacks or
    has elsewhere, as a fresh read still says, then take back any a second
    read shows the row stopped recording."""
    try:
        current = await _read(scope)
    except Exception:
        logger.warning(f"Could not read {scope_label(scope)} again", exc_info=True)
        return
    if current is None or current.state != _ACTIVE:
        return
    off = _off(current, scope, zones)
    try:
        for job in off:
            await store.add(job, scope, current.timezone)
        taken_back = await _take_back(scope, off, store)
    except Exception:
        logger.warning(
            f"Could not restore the crons of {scope_label(scope)}", exc_info=True
        )
        report.failed.append(scope.scope_key)
        return
    kept = [job for job in off if job not in taken_back]
    report.restored += [job.job_id(scope) for job in kept]
    if kept:
        logger.info(
            f"Memory cron reconcile registered {[job.name for job in kept]} for "
            f"{scope_label(scope)} in {current.timezone}"
        )


async def _take_back(
    scope: MemoryScope, added: list[DreamSystemJob], store: CronStore
) -> list[DreamSystemJob]:
    """Drop what a restore added that the row no longer records as read now:
    a pause or a deleted cron landed between the restore's read and its
    write. A read that fails drops it all; a cron still recorded is restored
    by the next pass. The crons dropped."""
    if not added:
        return []
    try:
        row = await _read(scope)
    except Exception:
        logger.warning(
            f"Could not read {scope_label(scope)} after restoring its crons; "
            "dropping them for the next pass",
            exc_info=True,
        )
        row = None
    stale = [
        job
        for job in added
        if row is None or row.state != _ACTIVE or recorded_job_id(row, job) is None
    ]
    for job in stale:
        await store.drop(job, scope)
    return stale


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


async def _remove(
    scope: MemoryScope, store: CronStore, report: ReconcileReport
) -> None:
    """Remove the crons of a scope that should have none, once a fresh read
    still says so."""
    try:
        row = await _read(scope)
    except Exception:
        logger.warning(f"Could not read {scope_label(scope)} again", exc_info=True)
        return
    if not _straggling(scope, row):
        return
    try:
        await store.remove(scope)
    except Exception:
        logger.warning(
            f"Could not remove the crons of {scope_label(scope)}", exc_info=True
        )
        report.failed.append(scope.scope_key)
        return
    report.removed.append(scope.scope_key)
    logger.info(f"Memory cron reconcile removed the crons of {scope_label(scope)}")


def _straggling(scope: MemoryScope, row: MemoryScopeSchedule | None) -> bool:
    """Whether crons of ``scope`` must go: its row says PAUSED or WIPED, or it
    is an expert's and has no row. An account with no row keeps its crons."""
    if row is None:
        return scope.is_expert
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


class _Place:
    """Where the reconcile stands, kept in Redis for every scheduler and in
    this process (``local``) for when Redis fails: a pass whose Redis read
    fails or finds nothing carries on from the local place. Once a Redis
    call fails the pass stops calling it, so an outage costs one timeout a
    pass."""

    local: ClassVar[str | None] = None

    def __init__(self) -> None:
        self._redis_up = True

    async def read(self) -> str | None:
        place = await self._call(lambda redis: redis.get(CURSOR_KEY))
        if isinstance(place, bytes):
            place = place.decode()
        return place or _Place.local

    async def keep(self, scope_key: str | None) -> None:
        """Keep ``scope_key`` as the place; None once the last scope is
        settled, clearing it."""
        _Place.local = scope_key
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
                "Memory cron reconcile cannot reach Redis; this pass keeps its "
                "place in this process only",
                exc_info=True,
            )
            return None
