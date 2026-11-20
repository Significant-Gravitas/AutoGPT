"""A user's workspace is read before it is created: nearly every call finds
it, and a read is one statement where an upsert is several."""

import datetime
from unittest.mock import AsyncMock, MagicMock

import prisma.errors
import prisma.models
import pytest

from backend.data import workspace as workspace_db


def _row(workspace_id: str = "workspace-1") -> prisma.models.UserWorkspace:
    now = datetime.datetime.now(datetime.timezone.utc)
    return prisma.models.UserWorkspace(
        id=workspace_id, userId="user-1", createdAt=now, updatedAt=now
    )


@pytest.fixture
def table(mocker):
    """The workspace table's read and its two writes."""
    calls = MagicMock(
        find_unique=AsyncMock(return_value=None),
        create=AsyncMock(),
        upsert=AsyncMock(),
    )
    mocker.patch.object(workspace_db.UserWorkspace, "prisma", return_value=calls)
    return calls


@pytest.mark.asyncio
async def test_an_existing_workspace_is_read_and_nothing_is_written(table):
    table.find_unique.return_value = _row()

    workspace = await workspace_db.get_or_create_workspace("user-1")

    assert workspace.id == "workspace-1"
    table.find_unique.assert_awaited_once_with(where={"userId": "user-1"})
    table.create.assert_not_awaited()
    table.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_missing_workspace_is_created(table):
    table.create.return_value = _row("workspace-new")

    workspace = await workspace_db.get_or_create_workspace("user-1")

    assert workspace.id == "workspace-new"
    table.create.assert_awaited_once_with(data={"userId": "user-1"})
    table.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_workspace_another_call_created_first_is_read_back(table):
    table.find_unique.side_effect = [None, _row("workspace-won")]
    table.create.side_effect = prisma.errors.UniqueViolationError({})

    workspace = await workspace_db.get_or_create_workspace("user-1")

    assert workspace.id == "workspace-won"
    assert table.find_unique.await_count == 2


@pytest.mark.asyncio
async def test_a_lost_race_with_no_row_to_read_back_raises(table):
    table.create.side_effect = prisma.errors.UniqueViolationError({})

    with pytest.raises(prisma.errors.UniqueViolationError):
        await workspace_db.get_or_create_workspace("user-1")
