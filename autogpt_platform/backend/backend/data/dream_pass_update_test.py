"""A DreamPass transition against a real database: each column only moves
forward, the JSON columns merge inside the statement, and writes that land
out of order or at the same time end in the row they should."""

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

from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.schemas import (
    ConsolidationOutput,
    DreamOperations,
    RecombinationOutput,
)
from backend.data.db import query_raw_with_schema
from backend.util.json import SafeJson

from .dream_pass import create_dream_pass, get_dream_pass, update_dream_pass
from .dream_pass_models import (
    DreamPassApplied,
    DreamPassDraft,
    DreamPassOperations,
    DreamPassRecord,
    DreamPassUpdate,
    DreamPhaseOutputs,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

_NOW = datetime.now(timezone.utc).replace(microsecond=0)


@pytest.fixture
async def owner() -> AsyncIterator[str]:
    """A throwaway user; deleting it afterwards cascades to its passes."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"dream-pass-update-{user_id}@example.com",
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
            started_at=_NOW,
        )
    )
    return pass_id


async def _row(pass_id: str) -> DreamPassRecord:
    row = await get_dream_pass(pass_id)
    assert row is not None
    return row


async def test_status_and_phase_only_move_forward(owner):
    pass_id = await _new_pass(owner)
    await update_dream_pass(
        pass_id,
        DreamPassUpdate(
            status=DreamPassStatus.SUBMITTED, phase=DreamPassPhase.SANITIZE
        ),
    )

    behind = DreamPassUpdate(
        status=DreamPassStatus.RUNNING, phase=DreamPassPhase.CONSOLIDATE
    )
    assert await update_dream_pass(pass_id, behind)
    row = await _row(pass_id)
    assert (row.status, row.phase) == (
        DreamPassStatus.SUBMITTED,
        DreamPassPhase.SANITIZE,
    )

    ahead = DreamPassUpdate(status=DreamPassStatus.APPLYING, phase=DreamPassPhase.APPLY)
    assert await update_dream_pass(pass_id, ahead)
    row = await _row(pass_id)
    assert (row.status, row.phase) == (DreamPassStatus.APPLYING, DreamPassPhase.APPLY)


async def test_a_late_submit_keeps_the_first_callbacks_phase_and_batch(owner):
    """The batch submit's write reaching the database after the consolidate
    callback's must leave the row at recombine, on the recombine batch."""
    pass_id = await _new_pass(owner)
    bundle = DreamInput(
        user_id=owner, group_id=f"user_{owner}", window_start=_NOW, window_end=_NOW
    )
    consolidated = ConsolidationOutput()
    await update_dream_pass(
        pass_id,
        DreamPassUpdate(
            phase=DreamPassPhase.RECOMBINE,
            phase_outputs=DreamPhaseOutputs(consolidate=consolidated),
        ),
    )
    await update_dream_pass(
        pass_id,
        DreamPassUpdate(
            phase=DreamPassPhase.RECOMBINE, provider_batch_id="batch-recombine"
        ),
    )

    await update_dream_pass(
        pass_id,
        DreamPassUpdate(
            status=DreamPassStatus.SUBMITTED,
            phase=DreamPassPhase.CONSOLIDATE,
            provider_batch_id="batch-consolidate",
            input_bundle=bundle,
            lease_token="tok",
            lease_expires_at=_NOW + timedelta(hours=24),
            submitted_at=_NOW,
        ),
    )

    row = await _row(pass_id)
    assert row.status is DreamPassStatus.SUBMITTED
    assert row.phase is DreamPassPhase.RECOMBINE
    assert row.provider_batch_id == "batch-recombine"
    assert row.phase_outputs == DreamPhaseOutputs(consolidate=consolidated)
    assert (row.input_bundle, row.lease_token) == (bundle, "tok")
    assert row.lease_expires_at == _NOW + timedelta(hours=24)


async def test_a_batch_for_an_earlier_phase_never_replaces_a_later_one(owner):
    pass_id = await _new_pass(owner)
    await update_dream_pass(
        pass_id,
        DreamPassUpdate(phase=DreamPassPhase.SANITIZE, provider_batch_id="batch-s"),
    )

    await update_dream_pass(
        pass_id,
        DreamPassUpdate(phase=DreamPassPhase.RECOMBINE, provider_batch_id="batch-r"),
    )

    row = await _row(pass_id)
    assert (row.phase, row.provider_batch_id) == (DreamPassPhase.SANITIZE, "batch-s")


async def test_writers_racing_on_the_json_columns_all_land(owner):
    """Each writer merges its own key in the statement, so none of them can
    overwrite another's with a stale copy of the column."""
    parts = DreamPassUpdate(
        phase_outputs=DreamPhaseOutputs(
            consolidate=ConsolidationOutput(),
            recombine=RecombinationOutput(),
            sanitize=DreamOperations(summary_for_user="sanitized"),
        ),
        operations=DreamPassOperations(
            planned=DreamOperations(summary_for_user="planned"),
            applied=DreamPassApplied(consolidated_count=2),
        ),
    )
    writers = [
        DreamPassUpdate(
            phase_outputs=DreamPhaseOutputs(consolidate=ConsolidationOutput())
        ),
        DreamPassUpdate(
            phase_outputs=DreamPhaseOutputs(recombine=RecombinationOutput())
        ),
        DreamPassUpdate(
            phase_outputs=DreamPhaseOutputs(
                sanitize=DreamOperations(summary_for_user="sanitized")
            )
        ),
        DreamPassUpdate(
            operations=DreamPassOperations(
                planned=DreamOperations(summary_for_user="planned")
            )
        ),
        DreamPassUpdate(
            operations=DreamPassOperations(
                applied=DreamPassApplied(consolidated_count=2)
            )
        ),
    ]
    for _ in range(5):
        pass_id = await _new_pass(owner)

        written = await asyncio.gather(
            *(update_dream_pass(pass_id, writer) for writer in writers)
        )

        assert written == [True] * len(writers)
        row = await _row(pass_id)
        assert (row.phase_outputs, row.operations) == (
            parts.phase_outputs,
            parts.operations,
        )


async def test_timestamps_land_as_the_same_instant(owner):
    """A timestamp in any zone is stored as its UTC wall clock, the way the
    Prisma client writes the table's other timestamps."""
    pass_id = await _new_pass(owner)
    brisbane = datetime(2026, 9, 26, 13, 0, tzinfo=timezone(timedelta(hours=10)))
    before = await _row(pass_id)

    await update_dream_pass(
        pass_id,
        DreamPassUpdate(lease_expires_at=brisbane, window_start=_NOW),
    )

    row = await _row(pass_id)
    assert row.lease_expires_at == datetime(2026, 9, 26, 3, 0, tzinfo=timezone.utc)
    assert row.window_start == _NOW
    assert row.updated_at >= before.updated_at


async def test_deleting_an_expert_finds_its_passes_by_index():
    indexes = await query_raw_with_schema(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'DreamPass'"
    )

    assert "DreamPass_expertId_idx" in {row["indexname"] for row in indexes}
