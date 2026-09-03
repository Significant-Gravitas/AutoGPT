"""REL-005 durable scheduler occurrence claim helpers.

Provides canonical fireTime calculation and transactional claim via the
``ScheduleOccurrence`` unique constraint ``(scheduleId, fireTime)``.

Status lifecycle:
  claimed    — winner inserted, not yet dispatched (retryable)
  dispatched — DB recorded AND queue publish succeeded
  missed     — APScheduler reported EVENT_JOB_MISSED (no billing)
  completed/failed — terminal execution states (not used by scheduler claim path)

The claim is a blind INSERT catching ``UniqueViolationError`` — never
check-then-insert. Duplicate callers converge to the existing row's
``executionId`` without creating a second chargeable execution.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from prisma.errors import UniqueViolationError
from prisma.models import ScheduleOccurrence

logger = logging.getLogger(__name__)


def canonical_fire_time(dt: datetime | None = None) -> datetime:
    """Truncate to minute boundary in UTC — deterministic idempotency key."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(second=0, microsecond=0)


async def claim_occurrence(
    schedule_id: str, fire_time: datetime
) -> tuple[object, bool]:
    """Attempt durable claim for (schedule_id, fire_time).

    Returns (occurrence, is_winner). is_winner True means this caller
    inserted the row and may create/dispatch execution. False means
    duplicate — caller must converge to existing occurrence.executionId.

    Uses blind INSERT + UniqueViolationError, never check-then-insert.
    Falls back to DatabaseManager RPC when local prisma is not connected
    (scheduler process).
    """
    # Ensure UTC and minute-truncated for stable key
    fire_time = canonical_fire_time(fire_time)
    # Try direct prisma if connected; else delegate via RPC (scheduler)
    try:
        from backend.data.db import prisma as _prisma

        if _prisma.is_connected():
            try:
                occ = await ScheduleOccurrence.prisma().create(
                    data={"scheduleId": schedule_id, "fireTime": fire_time, "status": "claimed"}
                )
                return occ, True
            except UniqueViolationError:
                occ = await ScheduleOccurrence.prisma().find_unique(
                    where={"scheduleId_fireTime": {"scheduleId": schedule_id, "fireTime": fire_time}}  # type: ignore
                )
                if occ is None:
                    raise
                return occ, False
    except Exception:
        pass
    # Fallback: use DatabaseManager RPC (indirect, but still hits same DB unique constraint)
    from backend.util.clients import get_database_manager_async_client

    db = get_database_manager_async_client()
    # The RPC path re-enters this same file on the DB manager process where
    # prisma IS connected, so the unique constraint still applies.
    # We call the direct path again via the RPC stub (which will be the real
    # function on that side). To avoid recursion, call via the DB client's
    # exposed method if available, else try direct again.
    try:
        # If the client has the method, use it (real DB manager)
        if hasattr(db, "claim_occurrence"):
            return await db.claim_occurrence(schedule_id, fire_time)  # type: ignore
    except Exception:
        pass
    # Last resort: try direct again (may raise)
    try:
        occ = await ScheduleOccurrence.prisma().create(
            data={"scheduleId": schedule_id, "fireTime": fire_time, "status": "claimed"}
        )
        return occ, True
    except UniqueViolationError:
        occ = await ScheduleOccurrence.prisma().find_unique(
            where={"scheduleId_fireTime": {"scheduleId": schedule_id, "fireTime": fire_time}}  # type: ignore
        )
        if occ is None:
            raise
        return occ, False


async def get_occurrence(schedule_id: str, fire_time: datetime):
    fire_time = canonical_fire_time(fire_time)
    return await ScheduleOccurrence.prisma().find_unique(
        where={"scheduleId_fireTime": {"scheduleId": schedule_id, "fireTime": fire_time}}  # type: ignore
    )


async def mark_dispatched(occurrence_id: str) -> None:
    await ScheduleOccurrence.prisma().update(
        where={"id": occurrence_id}, data={"status": "dispatched"}
    )


async def link_execution(occurrence_id: str, execution_id: str) -> None:
    await ScheduleOccurrence.prisma().update(
        where={"id": occurrence_id}, data={"executionId": execution_id}
    )


async def create_missed_occurrence(schedule_id: str, fire_time: datetime):
    """Create technical missed-occurrence record without billing decision.

    Idempotent: if the unique key already exists keep it as-is (do not
    overwrite a claimed/dispatched row). On UniqueViolationError fetch
    existing and leave it.
    """
    fire_time = canonical_fire_time(fire_time)
    try:
        occ = await ScheduleOccurrence.prisma().create(
            data={"scheduleId": schedule_id, "fireTime": fire_time, "status": "missed"}
        )
        return occ
    except UniqueViolationError:
        occ = await ScheduleOccurrence.prisma().find_unique(
            where={"scheduleId_fireTime": {"scheduleId": schedule_id, "fireTime": fire_time}}  # type: ignore
        )
        return occ
