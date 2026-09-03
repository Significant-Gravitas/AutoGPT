"""REL-005 ScheduleOccurrence durable unique constraint.

Proves the claim algorithm in backend.data.schedule_occurrence converges
duplicates onto one logical execution via the (scheduleId, fireTime)
unique constraint — not check-then-insert.
"""
import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.errors import UniqueViolationError


def _uve(msg: str = "Unique constraint failed") -> UniqueViolationError:
    """Prisma's UniqueViolationError requires a dict-shaped error body."""
    return UniqueViolationError(
        {"user_facing_error": {"message": msg, "code": "P2002", "meta": {}}}
    )


FIRE = datetime.datetime(2025, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)


def _occ_row(id="occ-1", status="claimed", executionId=None, scheduleId="sched-1", fireTime=FIRE):
    row = MagicMock()
    row.id = id
    row.status = status
    row.executionId = executionId
    row.scheduleId = scheduleId
    row.fireTime = fireTime
    return row


@pytest.mark.asyncio
async def test_first_claim_wins():
    """A fresh occurrence claim creates the row and reports winner."""
    from backend.data.schedule_occurrence import claim_occurrence

    with patch("backend.data.schedule_occurrence.ScheduleOccurrence.prisma") as mock:
        mock.return_value.create = AsyncMock(
            return_value=_occ_row(executionId=None)
        )
        occ, is_winner = await claim_occurrence("sched-1", FIRE)
        assert is_winner is True
        assert occ.id == "occ-1"
        # Blind insert carries the canonical key + claimed status
        data = mock.return_value.create.call_args.kwargs["data"]
        assert data["scheduleId"] == "sched-1"
        assert data["fireTime"] == FIRE
        assert data["status"] == "claimed"


@pytest.mark.asyncio
async def test_same_occurrence_twice_one_logical():
    """Same (scheduleId, fireTime) claimed twice → second converges to existing row, no duplicate."""
    from backend.data.schedule_occurrence import claim_occurrence

    existing = _occ_row(status="claimed", executionId="exec-1")
    with patch("backend.data.schedule_occurrence.ScheduleOccurrence.prisma") as mock:
        mock.return_value.create = AsyncMock(side_effect=_uve())
        mock.return_value.find_unique = AsyncMock(return_value=existing)

        occ, is_winner = await claim_occurrence("sched-1", FIRE)
        # Loser converges onto the existing occurrence
        assert is_winner is False
        assert occ.id == "occ-1"
        assert occ.executionId == "exec-1"
        # Convergence lookup used the canonical composite key
        where = mock.return_value.find_unique.call_args.kwargs["where"]
        assert where["scheduleId_fireTime"] == {"scheduleId": "sched-1", "fireTime": FIRE}


@pytest.mark.asyncio
async def test_concurrent_two_schedulers_one_wins():
    """Two schedulers racing the same fireTime → unique constraint admits exactly one winner."""
    import asyncio

    from backend.data.schedule_occurrence import claim_occurrence

    existing = _occ_row(status="claimed", executionId="exec-race")
    with patch("backend.data.schedule_occurrence.ScheduleOccurrence.prisma") as mock:
        mock.return_value.create = AsyncMock(
            side_effect=[_occ_row(executionId=None), _uve()]
        )
        mock.return_value.find_unique = AsyncMock(return_value=existing)

        results = await asyncio.gather(
            claim_occurrence("sched-1", FIRE),
            claim_occurrence("sched-1", FIRE),
            return_exceptions=True,
        )
        errors = [r for r in results if isinstance(r, Exception)]
        assert not errors, f"unexpected exceptions: {errors}"
        winners = [r for r in results if r[1] is True]
        losers = [r for r in results if r[1] is False]
        assert len(winners) == 1, "exactly one scheduler must win the claim"
        assert len(losers) == 1, "loser must converge, not error"
        # Loser reports the same logical execution as the winner's row
        assert losers[0][0].executionId == "exec-race"


@pytest.mark.asyncio
async def test_dispatch_marks_status_and_links_execution():
    """Dispatch is two durable writes: link executionId, then mark dispatched."""
    from backend.data.schedule_occurrence import link_execution, mark_dispatched

    with patch("backend.data.schedule_occurrence.ScheduleOccurrence.prisma") as mock:
        mock.return_value.update = AsyncMock(
            return_value=_occ_row(status="dispatched", executionId="exec-9")
        )
        await link_execution("occ-1", execution_id="exec-9")
        await mark_dispatched("occ-1")

        first = mock.return_value.update.call_args_list[0]
        second = mock.return_value.update.call_args_list[1]
        assert first.kwargs["where"] == {"id": "occ-1"}
        assert first.kwargs["data"] == {"executionId": "exec-9"}
        assert second.kwargs["where"] == {"id": "occ-1"}
        assert second.kwargs["data"] == {"status": "dispatched"}


@pytest.mark.asyncio
async def test_missed_tick_record_created_without_billing():
    """Missed ticks produce a technical record with no executionId (billing-decoupled)."""
    from backend.data.schedule_occurrence import create_missed_occurrence

    with patch("backend.data.schedule_occurrence.ScheduleOccurrence.prisma") as mock:
        mock.return_value.create = AsyncMock(
            return_value=_occ_row(id="occ-missed", status="missed")
        )
        await create_missed_occurrence("sched-1", FIRE)
        data = mock.return_value.create.call_args.kwargs["data"]
        assert data["status"] == "missed"
        assert "executionId" not in data, "missed record must not link chargeable execution"


@pytest.mark.asyncio
async def test_missed_tick_duplicate_converges():
    """A duplicate missed-tick record converges silently instead of crashing the listener."""
    from backend.data.schedule_occurrence import create_missed_occurrence

    with patch("backend.data.schedule_occurrence.ScheduleOccurrence.prisma") as mock:
        mock.return_value.create = AsyncMock(side_effect=_uve())
        mock.return_value.find_unique = AsyncMock(
            return_value=_occ_row(id="occ-missed", status="missed")
        )
        # Must not raise — the listener treats it as already-recorded
        await create_missed_occurrence("sched-1", FIRE)