"""A second start of a task that is already starting or being worked on
(QA pass 3, P4) gets the same clean 409 every time, and starts nothing."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.data import task_run
from backend.data.task import TASK_GONE, Task
from backend.util.exceptions import ConflictError, NotFoundError

_USER = "user-1"
_NOW = datetime(2026, 11, 19, 9, 0, tzinfo=UTC)


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": _USER,
        "title": "Call supplier",
        "status": TaskStatus.TODO,
        "created_by": TaskCreator.USER,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Task(**(fields | overrides))


@pytest.fixture
def tasks(mocker: MockerFixture) -> MagicMock:
    client = MagicMock(update_many=AsyncMock(return_value=0))
    mocker.patch.object(prisma.models.Task, "prisma", return_value=client)
    return client


@pytest.mark.parametrize("status", [TaskStatus.IN_PROGRESS, TaskStatus.NEEDS_INPUT])
async def test_a_task_already_worked_on_is_refused_the_same_way(
    tasks: MagicMock, mocker: MockerFixture, status
):
    mocker.patch.object(
        task_run, "get_task", AsyncMock(return_value=_task(status=status))
    )

    with pytest.raises(ConflictError) as refused:
        await task_run.claim_task_run("task-1", _USER)

    assert str(refused.value) == task_run.ALREADY_STARTED
    assert task_run.ALREADY_STARTED == "It is already being worked on."
    tasks.update_many.assert_not_awaited()


async def test_a_start_that_loses_the_race_says_already_started(
    tasks: MagicMock, mocker: MockerFixture
):
    """Both clicks read TODO; the first claims it, so the second's write
    moves nothing and it reads the task again: being worked on now."""
    mocker.patch.object(
        task_run,
        "get_task",
        AsyncMock(side_effect=[_task(), _task(status=TaskStatus.IN_PROGRESS)]),
    )

    with pytest.raises(ConflictError) as refused:
        await task_run.claim_task_run("task-1", _USER)

    assert str(refused.value) == task_run.ALREADY_STARTED
    tasks.update_many.assert_awaited_once()


async def test_a_missing_task_says_so_without_its_id(
    tasks: MagicMock, mocker: MockerFixture
):
    mocker.patch.object(task_run, "get_task", AsyncMock(return_value=None))

    with pytest.raises(NotFoundError) as gone:
        await task_run.claim_task_run("task-1", _USER)

    assert str(gone.value) == TASK_GONE
