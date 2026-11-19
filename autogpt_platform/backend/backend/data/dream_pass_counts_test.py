"""What a pass's apply dropped, through a DreamPass row against a real
database: the record the sync route writes (``pass_record.outcome``) reads
back through ``dream_pass_result_from_row`` with its drop counts, and a row
written before them reads none dropped."""

import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest
from prisma.enums import DreamPassRoute, DreamPassTrigger
from prisma.models import DreamPass as PrismaDreamPass
from prisma.models import User

from backend.copilot.dream.pass_record import dream_pass_result_from_row, outcome
from backend.copilot.dream.schemas import DreamPassResult
from backend.util.json import SafeJson

from .dream_pass import create_dream_pass, get_dream_pass, update_dream_pass
from .dream_pass_models import DreamPassDraft

pytestmark = pytest.mark.asyncio(loop_scope="session")

_NOW = datetime.now(timezone.utc).replace(microsecond=0)


@pytest.fixture
async def pass_id() -> AsyncIterator[str]:
    """A new pass of a throwaway user; deleting the user afterwards cascades
    to it."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"dream-pass-counts-{user_id}@example.com",
            "topUpConfig": SafeJson({}),
            "timezone": "UTC",
        }
    )
    draft = DreamPassDraft(
        id=str(uuid.uuid4()),
        user_id=user_id,
        scope_key=user_id,
        route=DreamPassRoute.SYNC,
        trigger=DreamPassTrigger.CRON,
        started_at=_NOW,
    )
    await create_dream_pass(draft)
    yield draft.id
    await User.prisma().delete(where={"id": user_id})


async def test_an_applied_pass_reads_back_what_it_dropped(pass_id):
    row = await get_dream_pass(pass_id)
    assert row is not None
    result = DreamPassResult(
        user_id=row.user_id,
        pass_id=pass_id,
        completed_at=_NOW,
        consolidated_count=2,
        dropped_forgotten=1,
        uncited_writes_dropped=3,
        cross_scope_citations_dropped=4,
        failed_writes=5,
        provenance_pending=6,
    )

    assert await update_dream_pass(pass_id, outcome(result, None))

    row = await get_dream_pass(pass_id)
    assert row is not None
    read_back = dream_pass_result_from_row(row)
    assert (
        read_back.consolidated_count,
        read_back.dropped_forgotten,
        read_back.uncited_writes_dropped,
        read_back.cross_scope_citations_dropped,
        read_back.failed_writes,
        read_back.provenance_pending,
    ) == (2, 1, 3, 4, 5, 6)


async def test_a_row_applied_before_the_drop_counts_reads_none_dropped(pass_id):
    await PrismaDreamPass.prisma().update(
        where={"id": pass_id},
        data={"operations": SafeJson({"applied": {"consolidated_count": 2}})},
    )

    row = await get_dream_pass(pass_id)
    assert row is not None and row.operations.applied is not None
    applied = row.operations.applied
    assert (
        applied.consolidated_count,
        applied.dropped_forgotten,
        applied.uncited_writes_dropped,
        applied.cross_scope_citations_dropped,
        applied.failed_writes,
        applied.provenance_pending,
    ) == (2, 0, 0, 0, 0, 0)
