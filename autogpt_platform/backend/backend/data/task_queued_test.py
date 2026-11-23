"""A run waiting for a free slot says so on its task (QA A5), and stops
saying so when its first turn starts; a mark an earlier run left never
counts for a later one."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import ResourceVisibility, TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.data import task_run
from backend.data.task import Task

_USER = "user-1"
_STARTED = datetime(2026, 11, 19, 9, 0, tzinfo=UTC)


def _row(metadata: dict | None = None, **overrides) -> prisma.models.Task:
    fields = {
        "id": "task-1",
        "createdAt": _STARTED,
        "updatedAt": _STARTED,
        "userId": _USER,
        "title": "Plan",
        "status": TaskStatus.IN_PROGRESS,
        "createdBy": TaskCreator.OTTO,
        "isArchived": False,
        "metadata": json.dumps(metadata or {}),
        "visibility": ResourceVisibility.PRIVATE,
        "attempt": 1,
        "isSuggested": False,
        "startedAt": _STARTED,
    }
    return prisma.models.Task(**(fields | overrides))


@pytest.fixture
def tasks(mocker: MockerFixture) -> MagicMock:
    client = MagicMock(
        find_first=AsyncMock(return_value=_row()),
        update_many=AsyncMock(return_value=1),
    )
    mocker.patch.object(prisma.models.Task, "prisma", return_value=client)
    return client


async def test_a_queued_run_is_marked_on_its_task(tasks: MagicMock, mocker):
    mocker.patch.object(task_run, "get_task", AsyncMock(return_value=MagicMock()))

    await task_run.mark_run_queued("task-1", _USER)

    update = tasks.update_many.await_args.kwargs
    assert update["where"]["userId"] == _USER
    assert update["data"]["statusText"] == "Queued: starts when a slot frees"
    marked = _row(update["data"]["metadata"].data)
    assert Task.from_db(marked).is_queued


async def test_the_runs_first_turn_clears_the_mark(tasks: MagicMock):
    tasks.find_first.return_value = _row(
        {"run_queued": _STARTED.isoformat()},
        statusText=task_run.QUEUED_TEXT,
    )

    cleared = await task_run.clear_run_queued("task-1", _USER)

    assert cleared
    data = tasks.update_many.await_args.kwargs["data"]
    assert data["statusText"] is None
    assert not Task.from_db(_row(data["metadata"].data)).is_queued


def test_an_earlier_runs_mark_does_not_count():
    later = _row(
        {"run_queued": _STARTED.isoformat()},
        startedAt=_STARTED + timedelta(hours=1),
    )

    assert not Task.from_db(later).is_queued


def test_only_a_task_being_worked_on_reads_as_queued():
    settled = _row({"run_queued": _STARTED.isoformat()}, status=TaskStatus.CANCELLED)

    assert not Task.from_db(settled).is_queued
