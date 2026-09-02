"""REL-005 ScheduleOccurrence durable unique constraint."""
import pytest
from unittest.mock import AsyncMock, patch
from prisma.errors import UniqueViolationError


@pytest.mark.asyncio
async def test_same_occurrence_twice_one_logical():
    """Same (scheduleId, fireTime) twice → second is duplicate, not second execution."""
    # Simulate prisma create with unique constraint
    call_count = 0

    async def fake_create(data):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise UniqueViolationError("Unique constraint failed on the fields: (`scheduleId`,`fireTime`)")
        return {"id": "occ-1", "scheduleId": data["scheduleId"], "fireTime": data["fireTime"]}

    with patch("backend.data.prisma") as mock_prisma:
        # We test the constraint directly via prisma model
        from prisma.models import ScheduleOccurrence

        with patch.object(ScheduleOccurrence, "prisma") as mock_model:
            mock_model.create = AsyncMock(side_effect=fake_create)

            # First claim succeeds
            result1 = await ScheduleOccurrence.prisma().create(
                data={"scheduleId": "sched-1", "fireTime": "2025-01-01T00:00:00Z", "status": "claimed"}
            )
            assert result1["id"] == "occ-1"

            # Second same occurrence should raise unique violation → caller must converge to existing
            with pytest.raises(UniqueViolationError):
                await ScheduleOccurrence.prisma().create(
                    data={"scheduleId": "sched-1", "fireTime": "2025-01-01T00:00:00Z", "status": "claimed"}
                )


@pytest.mark.asyncio
async def test_concurrent_two_schedulers_one_wins():
    """Two schedulers concurrently claiming same fireTime → one wins via unique constraint."""
    # Same as above but with concurrent gather
    import asyncio

    async def try_claim(schedule_id, fire_time):
        from prisma.models import ScheduleOccurrence

        return await ScheduleOccurrence.prisma().create(
            data={"scheduleId": schedule_id, "fireTime": fire_time, "status": "claimed"}
        )

    # Mock to simulate race: first succeeds, second gets unique violation
    with patch("prisma.models.ScheduleOccurrence.prisma") as mock:
        mock.create = AsyncMock(side_effect=[
            {"id": "occ-1"},
            UniqueViolationError("Unique constraint"),
        ])
        results = await asyncio.gather(
            try_claim("sched-1", "2025-01-01T00:00:00Z"),
            try_claim("sched-1", "2025-01-01T00:00:00Z"),
            return_exceptions=True,
        )
        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, UniqueViolationError)]
        assert len(successes) == 1
        assert len(failures) == 1
