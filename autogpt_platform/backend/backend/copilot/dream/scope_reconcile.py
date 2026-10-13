"""Keep the scheduler's dream-system crons in step with the registry rows.

The registry's own paths act on events: a pause removes a scope's crons, a
resume or an ensure registers them, and a pause reads the row once more to
put back crons that a resume registered just before its removal. Events can
interleave in ways one re-read does not settle: a stale takedown can land
after a second resume, and a re-read can fail. Either leaves an ACTIVE row
whose recorded job ids name crons the scheduler no longer has, and an
ensure, trusting those ids, never notices.

:func:`reconcile` is the level-triggered backstop. The scheduler runs it
every ``RECONCILE_INTERVAL_MINUTES`` (``executor/scheduler.py``) and compares
one listing of its dream-system crons with every registry row:

- every cron that the owner's flags enable for an ACTIVE row, and that the
  scheduler lacks, is registered in the row's timezone;
- every cron of a PAUSED or WIPED row, or of an expert whose row is gone, is
  removed. An account's crons with no row are left alone: they predate the
  registry, and the cron gate lets them run.

Recorded job ids are never trusted here: only what the scheduler lists
counts. Each scope is read once more right before it is changed, so a pause
or a resume that lands during the pass wins, and one that lands between that
read and the change is set right by the next pass. Every change is
idempotent: a registration replaces the job of the same id, and removing a
missing job does nothing. So two schedulers reconciling at once, as in a
rolling deploy, end in the same state. A flag that is off, or that cannot be
read, registers nothing and removes nothing, and a pass that cannot list the
rows changes nothing. A lost cron is therefore back within one interval.

A scope that lacks a cron costs one flag check per pass, its owner's flags
checked once per pass whatever number of scopes they have.
"""

import logging
from collections.abc import Sequence
from typing import Protocol

from prisma.enums import MemoryScopeScheduleState
from pydantic import BaseModel, Field

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import memory_schedule_db
from backend.data.memory_schedule import MemoryScopeSchedule

from .deadline import within_deadline
from .scheduling import DREAM_SYSTEM_JOBS, DreamSystemJob
from .scope_jobs import flag_gate, scope_label

logger = logging.getLogger(__name__)

RECONCILE_INTERVAL_MINUTES = 30
# Rows read from the registry per call while a pass lists them.
PAGE_SIZE = 500
_ACTIVE = MemoryScopeScheduleState.ACTIVE


class ListedCron(BaseModel):
    """One dream-system cron the scheduler holds: its job id and the scope
    its kwargs name."""

    job_id: str
    scope: MemoryScope


class CronStore(Protocol):
    """The scheduler side of a pass: registering a scope's cron, and removing
    every dream-system cron of a scope."""

    async def add(
        self, job: DreamSystemJob, scope: MemoryScope, user_timezone: str
    ) -> None: ...

    async def remove(self, scope: MemoryScope) -> None: ...


class ReconcileReport(BaseModel):
    """What one pass changed: the job ids it registered, the scopes whose
    crons it removed, and the scopes a change failed for."""

    restored: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


async def reconcile(listed: Sequence[ListedCron], store: CronStore) -> ReconcileReport:
    """One pass over every registry row, against ``listed``, the scheduler's
    dream-system crons as one listing read them before the rows. Never
    raises."""
    report = ReconcileReport()
    try:
        rows = {row.scope_key: row for row in await _all_rows()}
    except Exception:
        logger.warning(
            "Memory cron reconcile skipped: the registry could not be listed",
            exc_info=True,
        )
        return report
    present = {cron.job_id for cron in listed}
    enabled_by_owner: dict[str, set[str]] = {}
    for row in rows.values():
        if row.state == _ACTIVE:
            await _restore(row, present, enabled_by_owner, store, report)
    stragglers = {
        cron.scope.scope_key: cron.scope
        for cron in listed
        if _straggling(cron.scope, rows.get(cron.scope.scope_key))
    }
    for scope in stragglers.values():
        await _remove(scope, store, report)
    return report


async def _restore(
    row: MemoryScopeSchedule,
    present: set[str],
    enabled_by_owner: dict[str, set[str]],
    store: CronStore,
    report: ReconcileReport,
) -> None:
    """Register every cron the owner's flags enable for ``row`` that the
    scheduler lacks, once the row still says ACTIVE."""
    scope = MemoryScope.build(row.user_id, row.expert_id)
    missing = [job for job in DREAM_SYSTEM_JOBS if job.job_id(scope) not in present]
    if not missing:
        return
    if row.user_id not in enabled_by_owner:
        _, enabled = await flag_gate(scope)
        enabled_by_owner[row.user_id] = {job.job_id_prefix for job in enabled}
    wanted = [
        job for job in missing if job.job_id_prefix in enabled_by_owner[row.user_id]
    ]
    if not wanted:
        return
    try:
        current = await _read(scope)
    except Exception:
        logger.warning(f"Could not read {scope_label(scope)} again", exc_info=True)
        return
    if current is None or current.state != _ACTIVE:
        return
    try:
        for job in wanted:
            await store.add(job, scope, current.timezone)
            report.restored.append(job.job_id(scope))
    except Exception:
        logger.warning(
            f"Could not restore the crons of {scope_label(scope)}", exc_info=True
        )
        report.failed.append(scope.scope_key)
        return
    logger.info(
        f"Memory cron reconcile restored {[job.name for job in wanted]} for "
        f"{scope_label(scope)}"
    )


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
