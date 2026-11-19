"""A worker that delivered nothing says so (QA A1): ``finish_task`` with a
not-done outcome keeps the reason on the task, and every settle path then
settles it FAILED with that reason instead of DONE, a repeating task counting
it as a failed run."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import ResourceVisibility, TaskActivityKind, TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.data import task_record, task_result, task_run
from backend.data.task import Task
from backend.data.task_schedule_meta import TaskRepeat, read_schedule_meta

_USER = "user-1"
_NOW = datetime(2026, 11, 19, 7, 0, tzinfo=UTC)
_REASON = "No web search here, so 0 of 25 companies were listed."


def _row(repeat: TaskRepeat | None = None, **overrides) -> prisma.models.Task:
    schedule = {"repeat": repeat.model_dump(mode="json")} if repeat else {}
    fields = {
        "id": "task-1",
        "createdAt": _NOW,
        "updatedAt": _NOW,
        "userId": _USER,
        "title": "List 25 companies",
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
    client.find_first = AsyncMock(return_value=None)
    client.find_many = AsyncMock(return_value=[])
    client.update = AsyncMock()
    client.update_many = AsyncMock(return_value=1)
    mocker.patch.object(prisma.models.Task, "prisma", return_value=client)
    asks = MagicMock(update_many=AsyncMock(return_value=0))
    mocker.patch.object(prisma.models.TaskAsk, "prisma", return_value=asks)
    return client


@pytest.fixture
def lines(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch.object(task_record, "_append", AsyncMock())


async def test_a_not_done_finish_keeps_the_reason_on_the_task(
    tasks: MagicMock, mocker: MockerFixture
):
    working = Task.from_db(_row())
    mocker.patch.object(task_result, "work_tasks", AsyncMock(return_value=[working]))
    mocker.patch.object(
        task_result, "_is_background_chat", AsyncMock(return_value=True)
    )
    # The worker tried (it ran tools), so not-done is a real outcome.
    mocker.patch.object(task_result, "_tool_work_done", AsyncMock(return_value=True))

    finish = await task_result.finish_task(
        "run-chat", _USER, _REASON, None, outcome="blocked"
    )

    assert finish is not None and finish.outcome == "blocked"
    data = tasks.update.await_args.kwargs["data"]
    assert data["result"] == _REASON
    assert data["failureReason"] == _REASON


async def test_a_done_finish_leaves_no_failure_reason(
    tasks: MagicMock, mocker: MockerFixture
):
    working = Task.from_db(_row())
    mocker.patch.object(task_result, "work_tasks", AsyncMock(return_value=[working]))
    mocker.patch.object(
        task_result, "_is_background_chat", AsyncMock(return_value=True)
    )

    await task_result.finish_task("run-chat", _USER, "Listed 25.", None)

    data = tasks.update.await_args.kwargs["data"]
    assert data["failureReason"] is None


async def test_a_worker_that_finished_not_done_settles_failed_not_done(
    tasks: MagicMock, lines: AsyncMock
):
    """The background worker's turn completed, so delivery asks for DONE;
    the task's own finish says otherwise and wins."""
    tasks.find_many.return_value = [_row(result=_REASON, failureReason=_REASON)]

    await task_record.settle_work_tasks("run-chat", _USER, TaskStatus.DONE, "ok")

    data = tasks.update.await_args.kwargs["data"]
    assert data["status"] == TaskStatus.FAILED
    assert data["failureReason"] == _REASON
    entry = lines.await_args.args[1]
    assert entry.kind == TaskActivityKind.FAILED
    assert entry.text.startswith("Did not finish: No web search here")
    assert entry.data["run_end"] == "failed"


async def test_a_finished_task_says_finished(tasks: MagicMock, lines: AsyncMock):
    tasks.find_many.return_value = [_row(result="Listed 25.")]

    await task_record.settle_work_tasks("run-chat", _USER, TaskStatus.DONE)

    assert tasks.update.await_args.kwargs["data"]["status"] == TaskStatus.DONE
    assert lines.await_args.args[1].text == "Finished"


async def test_a_repeating_task_counts_a_not_done_finish_as_a_failed_run(
    tasks: MagicMock, lines: AsyncMock
):
    daily = TaskRepeat(crons=["0 8 * * *"], label="Every day at 08:00")
    tasks.find_many.return_value = [
        _row(repeat=daily, result=_REASON, failureReason=_REASON)
    ]

    await task_record.settle_work_tasks("run-chat", _USER, TaskStatus.DONE)

    data = tasks.update.await_args.kwargs["data"]
    assert data["status"] == TaskStatus.TODO
    repeat = read_schedule_meta(data["metadata"].data).repeat
    assert repeat is not None
    assert repeat.last_run == "failed" and repeat.failed_in_a_row == 1


async def test_a_page_run_that_finished_not_done_settles_failed(
    tasks: MagicMock, lines: AsyncMock
):
    row = _row(result=_REASON, failureReason=_REASON)
    tasks.find_first.return_value = row
    tasks.find_many.return_value = [row]

    status = await task_run.settle_task_run(
        "task-1", "run-chat", _USER, "completed", None
    )

    assert status == TaskStatus.FAILED
    assert tasks.update.await_args.kwargs["data"]["status"] == TaskStatus.FAILED


async def test_a_title_only_task_finished_not_done_without_trying_is_a_question(
    tasks: MagicMock, mocker: MockerFixture
):
    """QA pass 3, P2: nothing says what to do and nothing was tried, so the
    worker is sent to ask instead; nothing is saved."""
    thin = Task.from_db(_row(description=None, goal=None, doneWhen=None))
    mocker.patch.object(task_result, "work_tasks", AsyncMock(return_value=[thin]))
    mocker.patch.object(task_result, "_tool_work_done", AsyncMock(return_value=False))

    finish = await task_result.finish_task(
        "run-chat", _USER, "No instructions, so I did nothing.", None, "blocked"
    )

    assert finish is not None and finish.needs_question
    tasks.update.assert_not_awaited()


@pytest.mark.parametrize(
    "fields, tried",
    [
        ({"description": "Ask about the March invoices."}, False),
        ({"goal": "Get the invoices paid"}, False),
        ({}, True),
    ],
)
async def test_a_task_with_a_brief_or_tool_work_can_finish_not_done(
    tasks: MagicMock, mocker: MockerFixture, fields, tried
):
    working = Task.from_db(_row(**({"description": None} | fields)))
    mocker.patch.object(task_result, "work_tasks", AsyncMock(return_value=[working]))
    mocker.patch.object(task_result, "_tool_work_done", AsyncMock(return_value=tried))
    mocker.patch.object(
        task_result, "_is_background_chat", AsyncMock(return_value=True)
    )

    finish = await task_result.finish_task(
        "run-chat", _USER, _REASON, None, outcome="failed"
    )

    assert finish is not None and not finish.needs_question
    assert tasks.update.await_args.kwargs["data"]["failureReason"] == _REASON
