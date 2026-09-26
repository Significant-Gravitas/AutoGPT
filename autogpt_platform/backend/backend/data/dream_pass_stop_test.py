"""The two transitions that stop a dream pass from outside, against a real
database: a cancel lands only on the owner's open row and bumps its
generation exactly once however many race for it, and an expiry lands only on
an open row nothing has written since the instant it names."""

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone

import pytest
from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from prisma.models import User

from backend.copilot.dream.pass_record import cancelled, expired, failed
from backend.copilot.dream.schemas import ConsolidationOutput
from backend.data.db import execute_raw_with_schema
from backend.util.json import SafeJson

from .dream_pass import create_dream_pass, get_dream_pass, update_dream_pass
from .dream_pass_models import (
    DreamPassDraft,
    DreamPassRecord,
    DreamPassUpdate,
    DreamPhaseOutputs,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.fixture
async def owner() -> AsyncIterator[str]:
    """A throwaway user; deleting it afterwards cascades to its passes."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"dream-pass-stop-{user_id}@example.com",
            "topUpConfig": SafeJson({}),
            "timezone": "UTC",
        }
    )
    yield user_id
    await User.prisma().delete(where={"id": user_id})


async def _new_pass(owner: str) -> str:
    pass_id = str(uuid.uuid4())
    await create_dream_pass(
        DreamPassDraft(
            id=pass_id,
            user_id=owner,
            scope_key=owner,
            route=DreamPassRoute.ANTHROPIC_BATCH,
            trigger=DreamPassTrigger.CRON,
            started_at=datetime.now(timezone.utc),
        )
    )
    return pass_id


async def _row(pass_id: str) -> DreamPassRecord:
    row = await get_dream_pass(pass_id)
    assert row is not None
    return row


async def test_a_cancel_closes_an_open_row_and_bumps_its_generation(owner):
    pass_id = await _new_pass(owner)

    assert await update_dream_pass(pass_id, cancelled("testing", owner_user_id=owner))

    row = await _row(pass_id)
    assert (row.status, row.error, row.cancel_generation) == (
        DreamPassStatus.CANCELLED,
        "testing",
        1,
    )
    assert row.completed_at is not None


async def test_a_cancel_reaches_only_an_open_row(owner):
    pass_id = await _new_pass(owner)
    await update_dream_pass(
        pass_id,
        DreamPassUpdate(
            status=DreamPassStatus.COMPLETE,
            phase=DreamPassPhase.DONE,
            completed_at=datetime.now(timezone.utc),
        ),
    )

    assert not await update_dream_pass(
        pass_id, cancelled("testing", owner_user_id=owner)
    )

    row = await _row(pass_id)
    assert (row.status, row.error, row.cancel_generation) == (
        DreamPassStatus.COMPLETE,
        None,
        0,
    )


async def test_a_cancel_reaches_only_the_owners_row(owner):
    pass_id = await _new_pass(owner)

    assert not await update_dream_pass(
        pass_id, cancelled("testing", owner_user_id=str(uuid.uuid4()))
    )

    row = await _row(pass_id)
    assert (row.status, row.cancel_generation) == (DreamPassStatus.RUNNING, 0)


async def test_nothing_moves_a_cancelled_row(owner):
    pass_id = await _new_pass(owner)
    await update_dream_pass(pass_id, cancelled("testing", owner_user_id=owner))
    later = datetime.now(timezone.utc)

    for update in (
        failed("cancelled: testing", None, later),
        cancelled("again", owner_user_id=owner),
        expired("stale", not_updated_since=None),
        DreamPassUpdate(
            phase_outputs=DreamPhaseOutputs(consolidate=ConsolidationOutput())
        ),
    ):
        assert not await update_dream_pass(pass_id, update)

    row = await _row(pass_id)
    assert (row.status, row.error, row.cancel_generation) == (
        DreamPassStatus.CANCELLED,
        "testing",
        1,
    )
    assert row.phase_outputs == DreamPhaseOutputs()


async def test_racing_cancels_bump_the_generation_exactly_once(owner):
    for _ in range(5):
        pass_id = await _new_pass(owner)

        landed = await asyncio.gather(
            *(
                update_dream_pass(
                    pass_id, cancelled(f"cancel {n}", owner_user_id=owner)
                )
                for n in range(8)
            )
        )

        assert landed.count(True) == 1
        row = await _row(pass_id)
        assert (row.status, row.cancel_generation) == (DreamPassStatus.CANCELLED, 1)
        assert row.error == f"cancel {landed.index(True)}"


async def test_a_cancel_and_an_expiry_racing_close_the_row_once(owner):
    for _ in range(5):
        pass_id = await _new_pass(owner)

        landed = await asyncio.gather(
            update_dream_pass(pass_id, cancelled("testing", owner_user_id=owner)),
            update_dream_pass(pass_id, expired("stale", not_updated_since=None)),
        )

        assert landed.count(True) == 1
        row = await _row(pass_id)
        closed_by = DreamPassStatus.CANCELLED if landed[0] else DreamPassStatus.EXPIRED
        assert (row.status, row.cancel_generation) == (closed_by, 1)


async def test_an_expiry_lands_only_if_nothing_wrote_the_row_since(owner):
    pass_id = await _new_pass(owner)
    await update_dream_pass(pass_id, DreamPassUpdate(phase=DreamPassPhase.CONSOLIDATE))
    seen = await _row(pass_id)
    await asyncio.sleep(0.02)
    await update_dream_pass(pass_id, DreamPassUpdate(phase=DreamPassPhase.RECOMBINE))

    assert not await update_dream_pass(
        pass_id, expired("stale", not_updated_since=seen.updated_at)
    )
    assert (await _row(pass_id)).status is DreamPassStatus.RUNNING

    latest = await _row(pass_id)
    assert await update_dream_pass(
        pass_id, expired("stale", not_updated_since=latest.updated_at)
    )
    row = await _row(pass_id)
    assert (row.status, row.error, row.cancel_generation) == (
        DreamPassStatus.EXPIRED,
        "stale",
        1,
    )
    assert row.completed_at is not None


async def test_an_expiry_with_a_stale_cutoff_lands_only_on_an_old_row(owner):
    pass_id = await _new_pass(owner)
    await update_dream_pass(pass_id, DreamPassUpdate(phase=DreamPassPhase.CONSOLIDATE))
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)

    assert not await update_dream_pass(
        pass_id, expired("stale", not_updated_since=cutoff)
    )

    await execute_raw_with_schema(
        'UPDATE {schema_prefix}"DreamPass" '
        """SET "updatedAt" = "updatedAt" - interval '1 hour' WHERE "id" = $1""",
        pass_id,
    )
    assert await update_dream_pass(pass_id, expired("stale", not_updated_since=cutoff))
    assert (await _row(pass_id)).status is DreamPassStatus.EXPIRED


async def test_a_forced_expiry_lands_on_a_fresh_row(owner):
    pass_id = await _new_pass(owner)
    await update_dream_pass(pass_id, DreamPassUpdate(phase=DreamPassPhase.CONSOLIDATE))

    assert await update_dream_pass(pass_id, expired("forced", not_updated_since=None))

    row = await _row(pass_id)
    assert (row.status, row.cancel_generation) == (DreamPassStatus.EXPIRED, 1)


async def test_the_times_the_guard_compares_read_back_in_utc(owner):
    """The guard compares a row's lease and last write with an aware "now"."""
    pass_id = await _new_pass(owner)
    lease = datetime.now(timezone.utc) + timedelta(hours=24)
    await update_dream_pass(pass_id, DreamPassUpdate(lease_expires_at=lease))

    row = await _row(pass_id)

    assert row.updated_at.tzinfo is not None
    assert row.lease_expires_at is not None
    assert row.lease_expires_at.tzinfo is not None
    assert abs(row.lease_expires_at - lease) < timedelta(milliseconds=1)
