"""The DreamPass lease, the reaper's scans and retention against a real
database: a new row carries its lease, a renewal moves it, the sync route's
own end empties it and the input bundle (the statement's ``clear``), a close
that leaves a cleanup behind (a cancel, the reaper's expiry) marks the row
and keeps its lease until the one write a closed row takes clears both, the
reaper lists open rows whose lease lapsed and closed rows whose cleanup is
due, oldest first, on their indexes, and retention deletes closed rows past
their retention, a batch at a time, none whose cleanup is pending."""

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
from prisma.models import DreamPass as PrismaDreamPass
from prisma.models import User

from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.pass_record import (
    cancelled,
    cleanup_finished,
    expired,
    lease,
    outcome,
    reaped,
)
from backend.copilot.dream.schemas import ConsolidationOutput, DreamPassResult
from backend.data.db import query_raw_with_schema
from backend.util.json import SafeJson

from .dream_pass import (
    create_dream_pass,
    delete_old_dream_passes,
    get_dream_pass,
    list_dream_pass_cleanups,
    list_dream_passes,
    list_expired_dream_passes,
    update_dream_pass,
)
from .dream_pass_models import (
    DreamPassDraft,
    DreamPassRecord,
    DreamPassUpdate,
    DreamPhaseOutputs,
)

pytestmark = pytest.mark.asyncio(loop_scope="session")

# Postgres keeps TIMESTAMP(3): whole milliseconds read back exactly.
_NOW = datetime.now(timezone.utc).replace(microsecond=0)
# Lapses far older than any other test's rows, so the global scans that
# list the oldest first reach these.
_LONG_AGO = datetime(2001, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
async def owner() -> AsyncIterator[str]:
    """A throwaway user; deleting it afterwards cascades to its passes."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(
        data={
            "id": user_id,
            "email": f"dream-pass-lease-{user_id}@example.com",
            "topUpConfig": SafeJson({}),
            "timezone": "UTC",
        }
    )
    yield user_id
    await User.prisma().delete(where={"id": user_id})


async def test_a_new_row_carries_its_lease_and_a_renewal_moves_it(owner):
    pass_id = await _new_pass(owner, lease_expires_at=_NOW)

    row = await _row(pass_id)
    assert (row.lease_token, row.lease_expires_at) == ("tok", _NOW)

    assert await update_dream_pass(pass_id, lease("tok", 1800))
    renewed = await _row(pass_id)
    assert renewed.lease_token == "tok"
    assert renewed.lease_expires_at is not None
    assert renewed.lease_expires_at > _NOW + timedelta(seconds=1700)


async def test_a_closing_transition_empties_the_lease_and_the_bundle(owner):
    """Written by the statement's ``clear``: the sync route's own end drops
    the lease and the bundle, the outputs stay; an update that clears
    nothing leaves them."""
    pass_id = await _new_pass_with_a_bundle(owner)

    assert await update_dream_pass(
        pass_id, outcome(DreamPassResult(user_id=owner, pass_id=pass_id), None)
    )

    row = await _row(pass_id)
    assert row.status is DreamPassStatus.COMPLETE
    assert (row.lease_token, row.lease_expires_at, row.input_bundle) == (
        None,
        None,
        None,
    )
    assert row.cleanup_pending_at is None
    assert row.phase_outputs.consolidate == ConsolidationOutput()


async def test_a_close_that_leaves_a_cleanup_marks_the_row_and_keeps_its_lease(
    owner,
):
    """A cancel marks the row for the cleanup after the pass, drops the
    bundle and keeps the lease that cleanup needs, until the write that says
    it finished empties the mark and the lease."""
    pass_id = await _new_pass_with_a_bundle(owner)

    assert await update_dream_pass(pass_id, cancelled("testing", owner_user_id=owner))

    row = await _row(pass_id)
    assert row.status is DreamPassStatus.CANCELLED
    assert (row.lease_token, row.lease_expires_at) == ("tok", _NOW)
    assert (row.input_bundle, row.cleanup_pending_at is not None) == (None, True)
    assert pass_id in [r.id for r in await list_dream_pass_cleanups(_NOW)]

    assert await update_dream_pass(pass_id, cleanup_finished())

    done = await _row(pass_id)
    assert (done.cleanup_pending_at, done.lease_token, done.lease_expires_at) == (
        None,
        None,
        None,
    )
    assert pass_id not in [r.id for r in await list_dream_pass_cleanups(_NOW)]


async def test_an_update_that_clears_nothing_leaves_the_lease(owner):
    pass_id = await _new_pass(owner, lease_expires_at=_NOW)

    await update_dream_pass(pass_id, DreamPassUpdate(phase=DreamPassPhase.CONSOLIDATE))

    row = await _row(pass_id)
    assert (row.lease_token, row.lease_expires_at) == ("tok", _NOW)


async def test_the_reaper_lists_open_passes_whose_lease_lapsed_oldest_first(owner):
    oldest = await _new_pass(owner, lease_expires_at=_LONG_AGO)
    older = await _new_pass(owner, lease_expires_at=_LONG_AGO + timedelta(days=1))
    await _new_pass(owner, lease_expires_at=_NOW + timedelta(hours=1))
    await _new_pass(owner, lease_expires_at=None)
    closed = await _new_pass(owner, lease_expires_at=_LONG_AGO)
    await update_dream_pass(closed, expired("stale", not_updated_since=None))
    # A closed row marked for its cleanup still carries its lapsed lease.
    assert (await _row(closed)).lease_expires_at == _LONG_AGO

    rows = await list_expired_dream_passes(_NOW, limit=1000)
    mine = [row.id for row in rows if row.user_id == owner]
    assert mine == [oldest, older]

    first = await list_expired_dream_passes(_NOW, limit=1)
    assert [row.id for row in first] == [oldest]


async def test_the_reaper_scans_an_index_on_status_and_lease_expiry():
    indexes = await query_raw_with_schema(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'DreamPass'"
    )

    by_name = {row["indexname"]: row["indexdef"] for row in indexes}
    assert (
        '(status, "leaseExpiresAt")' in by_name["DreamPass_status_leaseExpiresAt_idx"]
    )


async def test_the_reapers_expiry_marks_the_row_until_its_cleanup_finishes(owner):
    pass_id = await _new_pass(owner, lease_expires_at=_LONG_AGO)
    row = await _row(pass_id)
    assert not await update_dream_pass(pass_id, cleanup_finished())

    assert await update_dream_pass(
        pass_id, reaped("lapsed", not_updated_since=row.updated_at)
    )

    closed = await _row(pass_id)
    assert (closed.status, closed.error) == (DreamPassStatus.EXPIRED, "lapsed")
    assert (closed.lease_token, closed.lease_expires_at) == ("tok", None)
    assert closed.cleanup_pending_at is not None
    assert closed.cancel_generation == 1
    listed = await list_dream_pass_cleanups(_NOW, limit=1000)
    assert pass_id in [r.id for r in listed]

    assert await update_dream_pass(pass_id, cleanup_finished())

    done = await _row(pass_id)
    assert (done.cleanup_pending_at, done.lease_token) == (None, None)
    assert (done.status, done.error, done.cancel_generation) == (
        DreamPassStatus.EXPIRED,
        "lapsed",
        1,
    )
    cleanups = await list_dream_pass_cleanups(_NOW, limit=1000)
    assert pass_id not in [r.id for r in cleanups]


async def test_the_cleanup_scan_lists_closed_marked_rows_longest_pending_first(
    owner,
):
    marked = []
    for minutes in (5, 50):
        pass_id = await _new_pass(owner, lease_expires_at=_LONG_AGO)
        row = await _row(pass_id)
        assert await update_dream_pass(
            pass_id, reaped("lapsed", not_updated_since=row.updated_at)
        )
        await PrismaDreamPass.prisma().update(
            where={"id": pass_id},
            data={"cleanupPendingAt": _LONG_AGO + timedelta(minutes=minutes)},
        )
        marked.append(pass_id)
    unmarked = await _new_pass(owner, lease_expires_at=_LONG_AGO)
    await update_dream_pass(
        unmarked, outcome(DreamPassResult(user_id=owner, pass_id=unmarked), None)
    )
    await _new_pass(owner, lease_expires_at=_LONG_AGO)

    rows = await list_dream_pass_cleanups(_NOW, limit=1000)

    assert [r.id for r in rows if r.user_id == owner] == marked
    first = await list_dream_pass_cleanups(_NOW, limit=1)
    assert [r.id for r in first] == marked[:1]


async def test_the_cleanup_scan_lists_a_row_once_its_cleanup_is_due(owner):
    """Due once the row was marked, or its lease lapsed, before the cutoff,
    or it holds no lease: a pass stopped while it ran has had its grace."""
    due_before = _NOW - timedelta(minutes=30)
    shapes = {
        "marked long ago": (_LONG_AGO + timedelta(hours=1), _NOW + timedelta(days=1)),
        "lease lapsed": (_NOW, _LONG_AGO),
        "no lease": (_NOW, None),
        "marked just now, lease fresh": (_NOW, _NOW + timedelta(days=1)),
    }
    ids = {}
    for name, (marked, lease_expires_at) in shapes.items():
        pass_id = await _new_pass(owner, lease_expires_at=_LONG_AGO)
        await update_dream_pass(pass_id, cancelled("testing", owner_user_id=owner))
        await PrismaDreamPass.prisma().update(
            where={"id": pass_id},
            data={"cleanupPendingAt": marked, "leaseExpiresAt": lease_expires_at},
        )
        ids[pass_id] = name

    rows = await list_dream_pass_cleanups(due_before, limit=1000)

    listed = [ids[r.id] for r in rows if r.id in ids]
    assert listed[0] == "marked long ago"
    assert set(listed) == {"marked long ago", "lease lapsed", "no lease"}


async def test_the_cleanup_scan_has_its_index():
    indexes = await query_raw_with_schema(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'DreamPass'"
    )

    by_name = {row["indexname"]: row["indexdef"] for row in indexes}
    assert (
        '(status, "cleanupPendingAt")'
        in by_name["DreamPass_status_cleanupPendingAt_idx"]
    )


async def test_retention_keeps_a_closed_row_whose_cleanup_is_pending(owner):
    pending = await _new_pass(owner, lease_expires_at=_LONG_AGO)
    row = await _row(pending)
    assert await update_dream_pass(
        pending, reaped("lapsed", not_updated_since=row.updated_at)
    )
    await PrismaDreamPass.prisma().update(
        where={"id": pending}, data={"createdAt": _LONG_AGO}
    )

    assert await delete_old_dream_passes(_LONG_AGO + timedelta(days=1), limit=10) == 0
    assert await update_dream_pass(pending, cleanup_finished())
    assert await delete_old_dream_passes(_LONG_AGO + timedelta(days=1), limit=10) == 1


async def test_retention_deletes_closed_passes_past_their_time_a_batch_at_a_time(
    owner,
):
    old_closed = [await _new_pass(owner) for _ in range(3)]
    old_open = await _new_pass(owner)
    recent_closed = await _new_pass(owner)
    for pass_id in [*old_closed, recent_closed]:
        await update_dream_pass(pass_id, cancelled("testing", owner_user_id=owner))
        await update_dream_pass(pass_id, cleanup_finished())
    for pass_id in [*old_closed, old_open]:
        await PrismaDreamPass.prisma().update(
            where={"id": pass_id}, data={"createdAt": _LONG_AGO}
        )
    cutoff = _LONG_AGO + timedelta(days=1)

    assert await delete_old_dream_passes(cutoff, limit=2) == 2
    assert await delete_old_dream_passes(cutoff, limit=2) == 1
    assert await delete_old_dream_passes(cutoff, limit=2) == 0

    left = {row.id for row in await list_dream_passes(owner, limit=10)}
    assert left == {old_open, recent_closed}


async def test_a_users_passes_list_the_open_ones_only_when_asked(owner):
    open_pass = await _new_pass(owner)
    closed = await _new_pass(owner)
    await update_dream_pass(closed, cancelled("testing", owner_user_id=owner))

    everything = await list_dream_passes(owner, limit=10)
    only_open = await list_dream_passes(owner, limit=10, open_only=True)

    assert {row.id for row in everything} == {closed, open_pass}
    assert [row.id for row in only_open] == [open_pass]


async def _new_pass_with_a_bundle(owner: str) -> str:
    """A leased pass with its input bundle and consolidate's output."""
    pass_id = await _new_pass(owner, lease_expires_at=_NOW)
    await update_dream_pass(
        pass_id,
        DreamPassUpdate(
            input_bundle=DreamInput(
                user_id=owner,
                group_id=f"user_{owner}",
                window_start=_NOW,
                window_end=_NOW,
            ),
            phase=DreamPassPhase.RECOMBINE,
            phase_outputs=DreamPhaseOutputs(consolidate=ConsolidationOutput()),
        ),
    )
    assert (await _row(pass_id)).input_bundle is not None
    return pass_id


async def _new_pass(owner: str, *, lease_expires_at: datetime | None = None) -> str:
    pass_id = str(uuid.uuid4())
    await create_dream_pass(
        DreamPassDraft(
            id=pass_id,
            user_id=owner,
            scope_key=owner,
            route=DreamPassRoute.ANTHROPIC_BATCH,
            trigger=DreamPassTrigger.CRON,
            started_at=_NOW,
            lease_token="tok" if lease_expires_at else None,
            lease_expires_at=lease_expires_at,
        )
    )
    return pass_id


async def _row(pass_id: str) -> DreamPassRecord:
    row = await get_dream_pass(pass_id)
    assert row is not None
    return row
