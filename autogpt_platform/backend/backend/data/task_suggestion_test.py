"""The suggestion data layer, with Prisma stubbed at the model."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.data import task_suggestion as module
from backend.data.task import Task
from backend.data.task_suggestion import (
    RETIRED_WINDOW,
    SuggestedTaskInput,
    create_suggested_task,
    dismiss_task_suggestion,
    dismissed_suggestion_titles,
    retire_stale_suggestions,
)
from backend.util.exceptions import NotFoundError

_NOW = datetime(2026, 11, 14, 8, 0, tzinfo=UTC)


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": "user-1",
        "title": "Pricing deck",
        "status": TaskStatus.TODO,
        "created_by": TaskCreator.OTTO,
        "created_at": _NOW,
        "updated_at": _NOW,
        "is_suggested": True,
    }
    return Task(**(fields | overrides))


@pytest.fixture
def task_table(mocker: MockerFixture) -> MagicMock:
    table = MagicMock(
        update_many=AsyncMock(return_value=1),
        create=AsyncMock(return_value=MagicMock(id="task-1")),
        find_many=AsyncMock(return_value=[]),
    )
    mocker.patch.object(module.prisma.models.Task, "prisma", return_value=table)
    return table


@pytest.mark.asyncio
async def test_dismiss_stamps_only_a_live_suggestion_of_the_user(
    mocker: MockerFixture, task_table: MagicMock
):
    mocker.patch.object(module, "get_task", AsyncMock(return_value=_task()))

    await dismiss_task_suggestion("task-1", "user-1")

    where = task_table.update_many.await_args.kwargs["where"]
    assert where == {
        "id": "task-1",
        "userId": "user-1",
        "isSuggested": True,
        "dismissedAt": None,
    }
    assert "dismissedAt" in task_table.update_many.await_args.kwargs["data"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "found, error", [(None, NotFoundError), (_task(is_suggested=False), ValueError)]
)
async def test_dismiss_refuses_a_missing_or_board_task(
    mocker: MockerFixture, task_table: MagicMock, found, error
):
    mocker.patch.object(module, "get_task", AsyncMock(return_value=found))
    with pytest.raises(error):
        await dismiss_task_suggestion("task-1", "user-1")
    task_table.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("expert_found, owner", [(True, "exp-1"), (False, None)])
async def test_create_stores_a_todo_suggestion(
    mocker: MockerFixture, task_table: MagicMock, expert_found, owner
):
    mocker.patch.object(module, "get_task", AsyncMock(return_value=_task()))
    mocker.patch.object(module, "_is_live_expert", AsyncMock(return_value=expert_found))

    await create_suggested_task(
        "user-1",
        SuggestedTaskInput(
            title="  Pricing   deck ",
            goal="Show the board why plan B wins.",
            done_when="Five   slides are in the shared folder.",
            brief="Turn the comparison into slides.",
            reason="The comparison finished yesterday.",
            owner_expert_id="exp-1",
        ),
    )

    data = task_table.create.await_args.kwargs["data"]
    assert data["title"] == "Pricing deck"
    assert data["isSuggested"] is True
    assert data["status"] == TaskStatus.TODO
    assert data["createdBy"] == TaskCreator.OTTO
    assert data["suggestedReason"] == "The comparison finished yesterday."
    assert data["ownerExpertId"] == owner
    assert data["goal"] == "Show the board why plan B wins."
    assert data["doneWhen"] == "Five slides are in the shared folder."


@pytest.mark.asyncio
async def test_set_aside_titles_cover_dismissed_and_recently_retired(
    task_table: MagicMock,
):
    task_table.find_many.return_value = [MagicMock(title="Pricing deck")]

    assert await dismissed_suggestion_titles("user-1") == ["Pricing deck"]

    where = task_table.find_many.await_args.kwargs["where"]
    assert (where["userId"], where["isSuggested"]) == ("user-1", True)
    dismissed, retired = where["OR"]
    assert dismissed == {"dismissedAt": {"not": None}}
    assert retired["isArchived"] is True
    since = retired["updatedAt"]["gte"]
    assert abs(datetime.now(UTC) - RETIRED_WINDOW - since).total_seconds() < 60


@pytest.mark.asyncio
async def test_retiring_stamps_when_it_happened(task_table: MagicMock):
    await retire_stale_suggestions("user-1", _NOW)

    kwargs = task_table.update_many.await_args.kwargs
    assert kwargs["where"]["createdAt"] == {"lt": _NOW}
    assert kwargs["where"]["dismissedAt"] is None
    assert kwargs["data"]["isArchived"] is True
    assert "updatedAt" in kwargs["data"]
