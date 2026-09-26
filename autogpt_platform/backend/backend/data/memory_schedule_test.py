"""Unit tests for the memory-scope schedule data module: the ownership rules
and the writes the registry relies on, against a mocked Prisma table."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import MemoryScopeScheduleState, ResourceVisibility

from backend.data import memory_schedule
from backend.util.exceptions import NotAuthorizedError

ACTIVE = MemoryScopeScheduleState.ACTIVE
PAUSED = MemoryScopeScheduleState.PAUSED


def _row(**overrides) -> SimpleNamespace:
    now = datetime.now(timezone.utc)
    fields = {
        "scopeKey": "user-1",
        "userId": "user-1",
        "expertId": None,
        "timezone": "Europe/Paris",
        "state": ACTIVE,
        "communityJobId": "community_rebuild_user-1",
        "nightlyJobId": None,
        "lastNightlyRunAt": None,
        "lastCommunityRunAt": None,
        "createdAt": now,
        "updatedAt": now,
    }
    return SimpleNamespace(**(fields | overrides))


@pytest.fixture
def table(mocker) -> MagicMock:
    client = MagicMock()
    client.find_first = AsyncMock(return_value=None)
    client.find_many = AsyncMock(return_value=[])
    client.upsert = AsyncMock(return_value=_row())
    client.update_many = AsyncMock(return_value=1)
    mocker.patch.object(
        prisma.models.MemoryScopeSchedule, "prisma", return_value=client
    )
    return client


@pytest.mark.asyncio
async def test_get_matches_on_scope_and_owner(table):
    table.find_first.return_value = _row()

    row = await memory_schedule.get_scope_schedule("user-1", "user-1")

    table.find_first.assert_awaited_once_with(
        where={"scopeKey": "user-1", "userId": "user-1"}
    )
    assert row is not None
    assert (row.scope_key, row.community_job_id) == (
        "user-1",
        "community_rebuild_user-1",
    )


@pytest.mark.asyncio
async def test_claim_creates_in_the_given_state_and_never_overwrites(table):
    await memory_schedule.claim_scope_schedule(
        "user-1", "expert_abc", "expert-1", "Asia/Tokyo", PAUSED
    )

    data = table.upsert.await_args.kwargs["data"]
    assert data["create"] == {
        "scopeKey": "expert_abc",
        "userId": "user-1",
        "expertId": "expert-1",
        "timezone": "Asia/Tokyo",
        "state": PAUSED,
    }
    # An existing row is returned as it is: its creator set its state.
    assert data["update"] == {}


@pytest.mark.asyncio
async def test_claim_refuses_another_users_scope(table):
    table.upsert.return_value = _row(userId="someone-else")

    with pytest.raises(NotAuthorizedError):
        await memory_schedule.claim_scope_schedule("user-1", "user-1", None, "UTC")


@pytest.mark.asyncio
async def test_record_jobs_only_writes_an_active_row(table):
    table.update_many.return_value = 0

    recorded = await memory_schedule.record_scope_jobs(
        "user-1",
        "user-1",
        user_timezone="UTC",
        community_job_id="c",
        nightly_job_id="n",
    )

    assert recorded is False
    where = table.update_many.await_args.kwargs["where"]
    assert where == {"scopeKey": "user-1", "userId": "user-1", "state": ACTIVE}


@pytest.mark.asyncio
async def test_leaving_active_forgets_the_job_ids(table):
    assert await memory_schedule.set_scope_state("user-1", "user-1", PAUSED)

    assert table.update_many.await_args.kwargs == {
        "where": {"scopeKey": "user-1", "userId": "user-1"},
        "data": {"state": PAUSED, "communityJobId": None, "nightlyJobId": None},
    }


@pytest.mark.asyncio
async def test_resuming_leaves_the_job_ids_to_the_next_registration(table):
    await memory_schedule.set_scope_state("user-1", "user-1", ACTIVE)

    assert table.update_many.await_args.kwargs["data"] == {"state": ACTIVE}


@pytest.mark.asyncio
async def test_forget_clears_whichever_column_holds_the_job(table):
    table.update_many.side_effect = [0, 1]

    assert await memory_schedule.forget_scope_job("user-1", "user-1", "job-1")

    community, nightly = table.update_many.await_args_list
    assert community.kwargs["where"]["communityJobId"] == "job-1"
    assert nightly.kwargs["data"] == {"nightlyJobId": None}
    assert all(c.kwargs["where"]["userId"] == "user-1" for c in (community, nightly))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind, column",
    [("nightly", "lastNightlyRunAt"), ("community", "lastCommunityRunAt")],
)
async def test_record_run_stamps_the_kinds_column(
    table, kind: memory_schedule.ScopeRunKind, column: str
):
    await memory_schedule.record_scope_run("user-1", "user-1", kind)

    assert list(table.update_many.await_args.kwargs["data"]) == [column]


@pytest.mark.asyncio
async def test_active_rows_page_by_scope_key(table):
    await memory_schedule.list_active_scope_schedules(after="k", limit=10)

    assert table.find_many.await_args.kwargs == {
        "where": {"state": ACTIVE, "scopeKey": {"gt": "k"}},
        "order": {"scopeKey": "asc"},
        "take": 10,
    }


@pytest.mark.asyncio
async def test_live_experts_are_hired_unarchived_and_owner_only(mocker):
    experts = MagicMock()
    experts.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(id="e1", ownerUserId="u1", schedulesPausedAt=None),
            SimpleNamespace(
                id="e2", ownerUserId="u1", schedulesPausedAt=datetime.now(timezone.utc)
            ),
        ]
    )
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=experts)

    scopes = await memory_schedule.list_live_expert_scopes()

    where = experts.find_many.await_args.kwargs["where"]
    assert where == {
        "isTemplate": False,
        "isArchived": False,
        "ownerUserId": {"not": None},
        "visibility": ResourceVisibility.PRIVATE,
    }
    assert [(s.expert_id, s.paused) for s in scopes] == [("e1", False), ("e2", True)]


@pytest.mark.asyncio
async def test_existing_user_ids_skips_the_query_for_nothing(mocker):
    users = MagicMock()
    users.find_many = AsyncMock()
    mocker.patch.object(prisma.models.User, "prisma", return_value=users)

    assert await memory_schedule.existing_user_ids([]) == set()
    users.find_many.assert_not_awaited()
