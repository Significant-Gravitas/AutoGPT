"""A task's start time in the data layer, against a mocked Prisma client:
every write is scoped to the owner and conditional on the task being TODO,
and the API model reads "scheduled" and "missed" from the row."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import ResourceVisibility, TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.data import task as task_data
from backend.data import task_due, task_run, task_schedule
from backend.data.task import Task, TaskPatch
from backend.util.exceptions import ConflictError, NotFoundError

_USER = "user-1"
_NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
_START = datetime(2026, 11, 15, 14, 0, tzinfo=UTC)


def _row(**overrides) -> prisma.models.Task:
    metadata = overrides.pop("metadata", {})
    fields = {
        "id": "task-1",
        "createdAt": _NOW,
        "updatedAt": _NOW,
        "userId": _USER,
        "title": "Research AutoGPT",
        "status": TaskStatus.TODO,
        "createdBy": TaskCreator.OTTO,
        "isArchived": False,
        "metadata": json.dumps(metadata),
        "visibility": ResourceVisibility.PRIVATE,
        "attempt": 1,
        "isSuggested": False,
    }
    return prisma.models.Task(**(fields | overrides))


@pytest.fixture
def tasks(mocker: MockerFixture) -> MagicMock:
    client = MagicMock()
    client.find_first = AsyncMock(return_value=_row())
    client.find_unique = AsyncMock(return_value=_row())
    client.find_many = AsyncMock(return_value=[])
    client.update_many = AsyncMock(return_value=1)
    client.update = AsyncMock()
    client.count = AsyncMock(return_value=0)
    mocker.patch.object(prisma.models.Task, "prisma", return_value=client)
    return client


def _task(**overrides) -> Task:
    return Task.from_db(_row(**overrides))


# ---------------------------------------------------------------- API model


@pytest.mark.parametrize(
    "fields, scheduled, missed",
    [
        ({"startAt": _START}, True, False),
        ({"metadata": {"schedule": {"missed": True}}}, False, True),
        # A flag left behind once the task moved on reads as not missed.
        (
            {"status": TaskStatus.DONE, "metadata": {"schedule": {"missed": True}}},
            False,
            False,
        ),
        ({"metadata": {"schedule": "junk"}}, False, False),
    ],
)
def test_the_task_says_whether_it_is_scheduled_or_missed(fields, scheduled, missed):
    task = _task(**fields)

    assert (task.start_at is not None, task.missed_start) == (scheduled, missed)


# ---------------------------------------------------------------- set_task_start


async def test_a_start_time_is_set_on_the_owners_todo_task(
    tasks: MagicMock, mocker: MockerFixture
):
    tasks.find_first.return_value = _row(
        metadata={"schedule": {"missed": True}, "other": 1}
    )
    mocker.patch.object(
        task_schedule, "get_task", AsyncMock(return_value=_task(startAt=_START))
    )

    task = await task_schedule.set_task_start("task-1", _USER, _START, "chat-1")

    assert task.start_at == _START
    assert tasks.find_first.await_args.kwargs["where"]["userId"] == _USER
    update = tasks.update_many.await_args.kwargs
    assert update["where"] == {
        "id": "task-1",
        "userId": _USER,
        "status": TaskStatus.TODO,
    }
    assert update["data"]["startAt"] == _START
    assert update["data"]["statusText"] is None
    metadata = update["data"]["metadata"].data
    assert metadata["other"] == 1
    assert metadata["schedule"] == {
        "due_at": _START.isoformat().replace("+00:00", "Z"),
        "origin_session_id": "chat-1",
        "missed": False,
        "repeat": None,
        "events": [],
    }


async def test_removing_the_start_keeps_the_chat_that_asked(
    tasks: MagicMock, mocker: MockerFixture
):
    tasks.find_first.return_value = _row(
        startAt=_START, metadata={"schedule": {"origin_session_id": "chat-1"}}
    )
    mocker.patch.object(task_schedule, "get_task", AsyncMock(return_value=_task()))

    await task_schedule.set_task_start("task-1", _USER, None)

    data = tasks.update_many.await_args.kwargs["data"]
    assert data["startAt"] is None
    assert data["metadata"].data["schedule"]["origin_session_id"] == "chat-1"


@pytest.mark.parametrize(
    "row, error",
    [
        (None, NotFoundError),
        (_row(status=TaskStatus.IN_PROGRESS), ConflictError),
    ],
)
async def test_a_start_time_is_refused_for_a_missing_or_started_task(
    tasks: MagicMock, row, error
):
    tasks.find_first.return_value = row

    with pytest.raises(error):
        await task_schedule.set_task_start("task-1", _USER, _START)

    tasks.update_many.assert_not_awaited()


async def test_a_task_that_started_meanwhile_is_not_rescheduled(tasks: MagicMock):
    tasks.update_many.return_value = 0

    with pytest.raises(ConflictError):
        await task_schedule.set_task_start("task-1", _USER, _START)


async def test_counting_and_cancelling_are_scoped_to_the_user(tasks: MagicMock):
    await task_schedule.count_scheduled_tasks(_USER)
    await task_schedule.cancel_scheduled_tasks(_USER)

    counted = tasks.count.await_args.kwargs["where"]
    assert counted["userId"] == _USER and counted["startAt"] == {"not": None}
    cancel = tasks.update_many.await_args.kwargs
    # Waiting for a start time, or for an event (none watch here).
    assert cancel["where"] == {
        "userId": _USER,
        "status": TaskStatus.TODO,
        "OR": [{"startAt": {"not": None}}, {"id": {"in": []}}],
    }
    assert cancel["data"]["status"] == TaskStatus.CANCELLED
    assert cancel["data"]["startAt"] is None


# ---------------------------------------------------------------- the sweep


async def test_due_tasks_are_read_with_one_indexed_query(tasks: MagicMock):
    tasks.find_many.return_value = [
        _row(startAt=_START, metadata={"schedule": {"origin_session_id": "chat-1"}})
    ]

    [due] = await task_due.list_due_tasks(_NOW, 20)

    query = tasks.find_many.await_args.kwargs
    assert query["where"] == {
        "status": TaskStatus.TODO,
        "startAt": {"lte": _NOW},
        "isArchived": False,
        "isSuggested": False,
    }
    assert query["order"] == {"startAt": "asc"} and query["take"] == 20
    assert (due.start_at, due.due_at, due.origin_session_id) == (
        _START,
        _START,
        "chat-1",
    )


async def test_the_sweeps_claim_is_conditional_on_the_start_it_read(
    tasks: MagicMock, mocker: MockerFixture
):
    mocker.patch.object(
        task_due,
        "get_task",
        AsyncMock(
            side_effect=[
                _task(startAt=_START),
                _task(status=TaskStatus.IN_PROGRESS),
            ]
        ),
    )

    claim = await task_due.claim_due_task("task-1", _USER, _START)

    assert claim is not None and claim.scheduled
    update = tasks.update_many.await_args.kwargs
    assert update["where"] == {
        "id": "task-1",
        "userId": _USER,
        "status": TaskStatus.TODO,
        "startAt": _START,
        "isArchived": False,
        "isSuggested": False,
    }
    assert update["data"]["status"] == TaskStatus.IN_PROGRESS
    assert update["data"]["startAt"] is None


async def test_a_claim_that_lost_the_race_returns_nothing(
    tasks: MagicMock, mocker: MockerFixture
):
    mocker.patch.object(
        task_due, "get_task", AsyncMock(return_value=_task(startAt=_START))
    )
    tasks.update_many.return_value = 0

    assert await task_due.claim_due_task("task-1", _USER, _START) is None


async def test_a_missed_task_keeps_its_chat_and_is_flagged(tasks: MagicMock):
    tasks.find_first.return_value = _row(
        startAt=_START, metadata={"schedule": {"origin_session_id": "chat-1"}}
    )

    assert await task_due.mark_task_missed(
        "task-1", _USER, _START, "Missed its start time"
    )

    update = tasks.update_many.await_args.kwargs
    assert update["where"]["startAt"] == _START and update["where"]["userId"] == _USER
    assert update["data"]["startAt"] is None
    assert update["data"]["statusText"] == "Missed its start time"
    schedule = update["data"]["metadata"].data["schedule"]
    assert schedule["missed"] and schedule["origin_session_id"] == "chat-1"


async def test_a_held_task_keeps_its_first_due_time(tasks: MagicMock):
    later = datetime(2026, 11, 15, 12, 10, tzinfo=UTC)

    assert await task_due.hold_due_task(
        "task-1", _USER, _START, later, "Waiting to start", _START
    )

    data = tasks.update_many.await_args.kwargs["data"]
    assert data["startAt"] == later
    assert data["metadata"].data["schedule"]["due_at"].startswith("2026-11-15T14:00")


async def test_a_task_no_longer_due_is_not_held(tasks: MagicMock):
    tasks.find_first.return_value = None

    assert not await task_due.hold_due_task("task-1", _USER, _START, _NOW, None, _START)
    tasks.update_many.assert_not_awaited()


# ---------------------------------------------------------------- other paths


async def test_start_now_clears_the_start_time_and_the_missed_flag(
    tasks: MagicMock, mocker: MockerFixture
):
    missed = _task(metadata={"schedule": {"missed": True}})
    mocker.patch.object(
        task_run,
        "get_task",
        AsyncMock(side_effect=[missed, _task(status=TaskStatus.IN_PROGRESS)]),
    )
    tasks.find_unique.return_value = _row(metadata={"schedule": {"missed": True}})

    await task_run.claim_task_run("task-1", _USER)

    data = tasks.update_many.await_args.kwargs["data"]
    assert data["startAt"] is None
    assert data["metadata"].data["schedule"]["missed"] is False


async def test_a_task_leaving_todo_loses_its_start_time(
    tasks: MagicMock, mocker: MockerFixture
):
    mocker.patch.object(
        task_data, "get_task", AsyncMock(return_value=_task(startAt=_START))
    )

    await task_data.update_task(
        "task-1", _USER, TaskPatch(status=TaskStatus.IN_PROGRESS)
    )

    assert tasks.update.await_args.kwargs["data"]["startAt"] is None


async def test_a_question_in_the_chat_does_not_park_a_scheduled_task(
    tasks: MagicMock,
):
    tasks.find_many.return_value = [_row()]

    await task_data.move_session_tasks(
        "chat-1", _USER, [TaskStatus.TODO], TaskStatus.NEEDS_INPUT
    )

    assert tasks.find_many.await_args.kwargs["where"]["startAt"] is None
    update = tasks.update_many.await_args.kwargs
    assert update["where"]["startAt"] is None
    assert update["where"]["id"] == {"in": ["task-1"]}


async def test_a_question_in_the_chat_does_not_park_a_paused_repeating_task(
    tasks: MagicMock,
):
    paused = {"schedule": {"repeat": {"crons": ["0 8 * * *"], "label": "x"}}}
    tasks.find_many.return_value = [_row(id="task-2", metadata=paused)]

    moved = await task_data.move_session_tasks(
        "chat-1", _USER, [TaskStatus.TODO], TaskStatus.NEEDS_INPUT
    )

    assert moved == 0
    tasks.update_many.assert_not_awaited()


async def test_scheduled_tasks_are_listed_soonest_first(tasks: MagicMock):
    await task_data.list_tasks(_USER, scheduled=True)

    query = tasks.find_many.await_args.kwargs
    assert query["where"]["userId"] == _USER
    assert query["where"]["status"] == TaskStatus.TODO
    assert query["where"]["startAt"] == {"not": None}
    assert query["order"] == {"startAt": "asc"}
