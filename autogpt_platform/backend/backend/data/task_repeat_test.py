"""A repeating task in the data layer, against a mocked Prisma client: a run
that ends puts it back to waiting (a one-off still settles), pause, resume and
stop repeating, stopping one run, and the account's repeats stopped when it
closes."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import ResourceVisibility, TaskActivityKind, TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.data import task_record, task_schedule
from backend.data.task import Task
from backend.data.task_schedule_meta import TaskRepeat, read_schedule_meta
from backend.util.exceptions import ConflictError

_USER = "user-1"
_NOW = datetime(2026, 11, 17, 7, 0, tzinfo=UTC)
_DAILY_8 = TaskRepeat(crons=["0 8 * * *"], label="Every day at 08:00")


def _row(repeat: TaskRepeat | None = _DAILY_8, **overrides) -> prisma.models.Task:
    schedule = {"repeat": repeat.model_dump(mode="json")} if repeat else {}
    fields = {
        "id": "task-1",
        "createdAt": _NOW,
        "updatedAt": _NOW,
        "userId": _USER,
        "title": "Morning brief",
        "status": TaskStatus.IN_PROGRESS,
        "createdBy": TaskCreator.OTTO,
        "isArchived": False,
        "metadata": json.dumps({"schedule": schedule}),
        "visibility": ResourceVisibility.PRIVATE,
        "attempt": 1,
        "isSuggested": False,
        "workSessionId": "run-chat",
    }
    return prisma.models.Task(**(fields | overrides))


@pytest.fixture
def tasks(mocker: MockerFixture) -> MagicMock:
    client = MagicMock()
    client.find_first = AsyncMock(return_value=_row())
    client.find_many = AsyncMock(return_value=[])
    client.update_many = AsyncMock(return_value=1)
    client.update = AsyncMock()
    mocker.patch.object(prisma.models.Task, "prisma", return_value=client)
    asks = MagicMock(update_many=AsyncMock(return_value=0))
    mocker.patch.object(prisma.models.TaskAsk, "prisma", return_value=asks)
    mocker.patch.object(
        task_schedule,
        "get_task",
        AsyncMock(return_value=Task.from_db(_row(status=TaskStatus.TODO))),
    )
    return client


@pytest.fixture
def lines(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch.object(task_record, "_append", AsyncMock())


def _schedule(data: dict):
    return read_schedule_meta(data["metadata"].data)


# ---------------------------------------------------------------- a run ends


async def test_a_finished_run_puts_a_repeating_task_back_for_its_next_run(
    tasks: MagicMock, lines: AsyncMock
):
    tasks.find_many.return_value = [_row(result="Three new signups.")]

    moved = await task_record.settle_work_tasks(
        "run-chat", _USER, TaskStatus.DONE, "ignored"
    )

    assert moved == 1
    data = tasks.update.await_args.kwargs["data"]
    assert data["status"] == TaskStatus.TODO
    assert data["startAt"] is not None and data["startAt"] > datetime.now(UTC)
    # The last run's result stays on the task until the next run starts.
    assert data["result"] == "Three new signups."
    repeat = _schedule(data).repeat
    assert repeat is not None and repeat.last_run == "done"
    entry = lines.await_args.args[1]
    assert entry.kind == TaskActivityKind.DONE
    assert entry.chat_session_id == "run-chat"
    assert entry.data == {"run_end": "done", "result": "Three new signups."}


async def test_a_one_off_task_still_settles(tasks: MagicMock, lines: AsyncMock):
    tasks.find_many.return_value = [_row(repeat=None, result="Done it.")]

    await task_record.settle_work_tasks("run-chat", _USER, TaskStatus.DONE)

    data = tasks.update.await_args.kwargs["data"]
    assert data["status"] == TaskStatus.DONE
    assert "startAt" not in data and "metadata" not in data


async def test_a_repeating_task_waiting_for_its_next_run_is_not_settled_again(
    tasks: MagicMock, lines: AsyncMock
):
    tasks.find_many.return_value = [_row(status=TaskStatus.TODO)]

    moved = await task_record.settle_work_tasks("run-chat", _USER, TaskStatus.DONE)

    assert moved == 0
    tasks.update.assert_not_awaited()
    lines.assert_not_awaited()


async def test_the_third_failed_run_pauses_it_and_says_why(
    tasks: MagicMock, lines: AsyncMock
):
    failing = _DAILY_8.model_copy(update={"failed_in_a_row": 2})
    tasks.find_many.return_value = [_row(repeat=failing)]

    await task_record.settle_work_tasks(
        "run-chat", _USER, TaskStatus.FAILED, None, "The work ran out of time."
    )

    data = tasks.update.await_args.kwargs["data"]
    assert data["status"] == TaskStatus.TODO
    assert data["startAt"] is None
    assert data["statusText"] == "Paused: the last 3 runs failed"
    assert data["failureReason"] == "The work ran out of time."
    repeat = _schedule(data).repeat
    assert repeat is not None and repeat.paused and repeat.failed_in_a_row == 3
    texts = [call.args[1].text for call in lines.await_args_list]
    assert texts == [
        "Failed: The work ran out of time.",
        "Paused: the last 3 runs failed",
    ]


# ---------------------------------------------------------------- the user


async def test_pause_keeps_the_rhythm_and_takes_the_next_run_off(tasks: MagicMock):
    tasks.find_first.return_value = _row(status=TaskStatus.TODO, startAt=_NOW)

    await task_schedule.pause_task_repeat("task-1", _USER)

    update = tasks.update_many.await_args.kwargs
    assert update["where"] == {
        "id": "task-1",
        "userId": _USER,
        "status": TaskStatus.TODO,
    }
    assert update["data"]["startAt"] is None
    repeat = _schedule(update["data"]).repeat
    assert repeat is not None and repeat.paused and repeat.crons == ["0 8 * * *"]


async def test_pause_during_a_run_lets_the_run_finish(tasks: MagicMock):
    await task_schedule.pause_task_repeat("task-1", _USER)

    data = tasks.update_many.await_args.kwargs["data"]
    assert "startAt" not in data and "status" not in data
    repeat = _schedule(data).repeat
    assert repeat is not None and repeat.paused


async def test_resume_sets_the_next_run_and_forgets_the_failures(tasks: MagicMock):
    paused = _DAILY_8.model_copy(update={"paused": True, "failed_in_a_row": 3})
    tasks.find_first.return_value = _row(repeat=paused, status=TaskStatus.TODO)

    await task_schedule.resume_task_repeat("task-1", _USER)

    data = tasks.update_many.await_args.kwargs["data"]
    assert data["startAt"] is not None and data["startAt"] > datetime.now(UTC)
    assert data["startAt"].astimezone(UTC).hour == 8
    meta = _schedule(data)
    assert meta.due_at == data["startAt"]
    assert meta.repeat is not None
    assert not meta.repeat.paused and meta.repeat.failed_in_a_row == 0


async def test_stop_repeating_settles_a_waiting_task_as_its_last_run_ended(
    tasks: MagicMock,
):
    last = _DAILY_8.model_copy(update={"last_run": "failed"})
    tasks.find_first.return_value = _row(
        repeat=last, status=TaskStatus.TODO, startAt=_NOW
    )

    await task_schedule.stop_task_repeat("task-1", _USER)

    data = tasks.update_many.await_args.kwargs["data"]
    assert data["status"] == TaskStatus.FAILED
    assert data["startAt"] is None
    assert _schedule(data).repeat is None


async def test_stop_repeating_during_a_run_lets_it_settle_once(tasks: MagicMock):
    await task_schedule.stop_task_repeat("task-1", _USER)

    data = tasks.update_many.await_args.kwargs["data"]
    assert "status" not in data
    assert _schedule(data).repeat is None


async def test_only_a_repeating_task_can_be_paused(tasks: MagicMock):
    tasks.find_first.return_value = _row(repeat=None)

    with pytest.raises(ConflictError):
        await task_schedule.pause_task_repeat("task-1", _USER)


async def test_stopping_one_run_puts_the_task_back_for_its_next_run(
    tasks: MagicMock,
):
    await task_schedule.end_repeat_run("task-1", _USER)

    update = tasks.update_many.await_args.kwargs
    assert update["where"]["status"] == TaskStatus.IN_PROGRESS
    assert update["data"]["status"] == TaskStatus.TODO
    assert update["data"]["startAt"] is not None
    repeat = _schedule(update["data"]).repeat
    assert repeat is not None and repeat.last_run == "cancelled"


async def test_a_closing_account_stops_every_repeat(tasks: MagicMock):
    tasks.find_many.return_value = [
        _row(),
        _row(id="task-2", repeat=None),
        _row(id="task-3", status=TaskStatus.TODO, repeat=_DAILY_8),
    ]

    await task_schedule.cancel_scheduled_tasks(_USER)

    stripped = [
        call.kwargs["where"]["id"]
        for call in tasks.update_many.await_args_list
        if "metadata" in call.kwargs["data"]
    ]
    assert stripped == ["task-1", "task-3"]
    for call in tasks.update_many.await_args_list:
        if "metadata" in call.kwargs["data"]:
            assert _schedule(call.kwargs["data"]).repeat is None
    # The waiting ones are cancelled as before.
    last = tasks.update_many.await_args.kwargs
    assert last["data"]["status"] == TaskStatus.CANCELLED
