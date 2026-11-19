"""A task that starts on an event, in the data layer, against a mocked Prisma
client: an event is queued and makes the task due, the filter and a pause
turn events away, the queue is capped, a claimed run takes the oldest event,
a run that ends puts the task back to watching, and stopping it lets go of
the webhook."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import ResourceVisibility, TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.data import task_due, task_record, task_schedule, task_watch
from backend.data.task import Task
from backend.data.task_schedule_meta import (
    MAX_PENDING_EVENTS,
    TaskRepeat,
    TaskWatch,
    WatchEvent,
    read_schedule_meta,
)

_USER = "user-1"
_NOW = datetime(2026, 11, 17, 7, 0, tzinfo=UTC)
_PR_TRIGGER = "6c60ec01-8128-419e-988f-96a063ee2fea"
_WATCH = TaskWatch(
    webhook_id="wh-1",
    provider="github",
    trigger_block_id=_PR_TRIGGER,
    trigger_config={"repo": "acme/web", "events": {"opened": True}},
    line="PR opened on acme/web",
)
_WATCHING = TaskRepeat(label=_WATCH.line, watch=_WATCH)


def _event(number: int = 12) -> WatchEvent:
    return WatchEvent(
        received_at=_NOW,
        event_type="pull_request.opened",
        summary=f"PR #{number} opened: Fix login",
        payload_text=json.dumps({"number": number}),
    )


def _row(
    repeat: TaskRepeat | None = _WATCHING,
    events: list[WatchEvent] | None = None,
    **overrides,
) -> prisma.models.Task:
    schedule: dict = {"origin_session_id": "asking-chat"}
    if repeat:
        schedule["repeat"] = repeat.model_dump(mode="json")
    schedule["events"] = [e.model_dump(mode="json") for e in events or []]
    fields = {
        "id": "task-1",
        "createdAt": _NOW,
        "updatedAt": _NOW,
        "userId": _USER,
        "title": "Review new PRs",
        "status": TaskStatus.TODO,
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
    return client


@pytest.fixture
def watching(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch.object(
        task_watch, "watching_task_rows", AsyncMock(return_value=[_row()])
    )


@pytest.fixture
def notes(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch.object(task_watch, "add_activity", AsyncMock())


@pytest.fixture
def released(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch.object(task_schedule, "release_watch_webhook", AsyncMock())


def _schedule(data: dict):
    return read_schedule_meta(data["metadata"].data)


def _pr(action: str = "opened") -> dict:
    return {
        "action": action,
        "number": 12,
        "pull_request": {"number": 12, "title": "Fix login"},
        "sender": {"login": "someone"},
    }


# ---------------------------------------------------------------- the ingress


async def test_an_event_is_queued_and_makes_the_waiting_task_due_now(
    tasks: MagicMock, watching: AsyncMock, notes: AsyncMock
):
    queued = await task_watch.deliver_webhook_event(
        _USER, "wh-1", "pull_request.opened", _pr(), now=_NOW
    )

    assert queued == 1
    watching.assert_awaited_once_with(_USER, "wh-1")
    update = tasks.update_many.await_args.kwargs
    # Only the row as it was read: a claim meanwhile would lose the event.
    assert update["where"] == {
        "id": "task-1",
        "userId": _USER,
        "isArchived": False,
        "updatedAt": _NOW,
    }
    assert update["data"]["startAt"] == _NOW
    meta = _schedule(update["data"])
    assert meta.due_at == _NOW
    assert [e.summary for e in meta.events] == ["PR #12 opened: Fix login"]
    assert "someone" not in meta.events[0].payload_text
    notes.assert_not_awaited()


async def test_an_event_the_filter_does_not_take_is_ignored(
    tasks: MagicMock, watching: AsyncMock
):
    queued = await task_watch.deliver_webhook_event(
        _USER, "wh-1", "pull_request.closed", _pr("closed"), now=_NOW
    )

    assert queued == 0
    tasks.update_many.assert_not_awaited()


async def test_a_paused_task_ignores_events(tasks: MagicMock, watching: AsyncMock):
    paused = _WATCHING.model_copy(update={"paused": True})
    watching.return_value = [_row(repeat=paused)]

    queued = await task_watch.deliver_webhook_event(
        _USER, "wh-1", "pull_request.opened", _pr(), now=_NOW
    )

    assert queued == 0
    tasks.update_many.assert_not_awaited()


async def test_a_full_queue_drops_the_event_and_says_so(
    tasks: MagicMock, watching: AsyncMock, notes: AsyncMock
):
    full = [_event(n) for n in range(MAX_PENDING_EVENTS)]
    watching.return_value = [_row(events=full, startAt=_NOW)]

    queued = await task_watch.deliver_webhook_event(
        _USER, "wh-1", "pull_request.opened", _pr(), now=_NOW
    )

    assert queued == 0
    tasks.update_many.assert_not_awaited()
    entry = notes.await_args.args[2]
    assert entry.text == task_watch.DROPPED_TEXT


async def test_an_event_during_a_run_waits_for_it_to_end(
    tasks: MagicMock, watching: AsyncMock
):
    watching.return_value = [_row(status=TaskStatus.IN_PROGRESS)]

    await task_watch.deliver_webhook_event(
        _USER, "wh-1", "pull_request.opened", _pr(), now=_NOW
    )

    data = tasks.update_many.await_args.kwargs["data"]
    assert "startAt" not in data
    assert len(_schedule(data).events) == 1


async def test_a_task_changed_meanwhile_is_read_again(
    tasks: MagicMock, watching: AsyncMock
):
    tasks.update_many.side_effect = [0, 1]
    tasks.find_first.return_value = _row(events=[_event(7)], startAt=_NOW)

    queued = await task_watch.deliver_webhook_event(
        _USER, "wh-1", "pull_request.opened", _pr(), now=_NOW
    )

    assert queued == 1
    data = tasks.update_many.await_args.kwargs["data"]
    assert [e.summary for e in _schedule(data).events] == [
        "PR #7 opened: Fix login",
        "PR #12 opened: Fix login",
    ]
    # Already due for the first event: its start stays.
    assert "startAt" not in data


# ---------------------------------------------------------------- a run


async def test_a_claimed_run_takes_the_oldest_event(
    tasks: MagicMock, mocker: MockerFixture
):
    row = _row(events=[_event(1), _event(2)], startAt=_NOW)
    tasks.find_first.return_value = row
    mocker.patch.object(task_due, "get_task", AsyncMock(return_value=Task.from_db(row)))

    claim = await task_due.claim_due_task("task-1", _USER, _NOW)

    assert claim is not None and claim.event is not None
    assert claim.event.summary == "PR #1 opened: Fix login"
    update = tasks.update_many.await_args.kwargs
    assert update["where"]["updatedAt"] == _NOW
    assert update["data"]["status"] == TaskStatus.IN_PROGRESS
    left = _schedule(update["data"]).events
    assert [e.summary for e in left] == ["PR #2 opened: Fix login"]


@pytest.mark.parametrize("pending, start", [([], None), ([_event(2)], "now")])
async def test_a_finished_run_goes_back_to_watching(
    tasks: MagicMock, mocker: MockerFixture, pending, start
):
    mocker.patch.object(task_record, "_append", AsyncMock())
    tasks.find_many.return_value = [
        _row(status=TaskStatus.IN_PROGRESS, events=pending, result="Reviewed.")
    ]

    await task_record.settle_work_tasks("run-chat", _USER, TaskStatus.DONE)

    data = tasks.update.await_args.kwargs["data"]
    assert data["status"] == TaskStatus.TODO
    if start is None:
        assert data["startAt"] is None
    else:
        # More events wait: the next run is due at once.
        assert data["startAt"] is not None
        assert abs((data["startAt"] - datetime.now(UTC)).total_seconds()) < 60
    meta = _schedule(data)
    assert meta.repeat is not None and meta.repeat.watch == _WATCH
    assert meta.repeat.last_run == "done"


async def test_the_third_failed_run_pauses_a_watching_task(
    tasks: MagicMock, mocker: MockerFixture
):
    mocker.patch.object(task_record, "_append", AsyncMock())
    failing = _WATCHING.model_copy(update={"failed_in_a_row": 2})
    tasks.find_many.return_value = [
        _row(repeat=failing, status=TaskStatus.IN_PROGRESS, events=[_event()])
    ]

    await task_record.settle_work_tasks(
        "run-chat", _USER, TaskStatus.FAILED, None, "It broke."
    )

    data = tasks.update.await_args.kwargs["data"]
    assert data["status"] == TaskStatus.TODO
    assert data["startAt"] is None
    repeat = _schedule(data).repeat
    assert repeat is not None and repeat.paused and repeat.watch is not None


# ---------------------------------------------------------------- the user


@pytest.mark.parametrize("pending, starts", [([], False), ([_event()], True)])
async def test_resume_waits_for_the_next_event_or_runs_one_waiting(
    tasks: MagicMock, mocker: MockerFixture, pending, starts
):
    paused = _WATCHING.model_copy(update={"paused": True, "failed_in_a_row": 3})
    tasks.find_first.return_value = _row(repeat=paused, events=pending)
    mocker.patch.object(task_schedule, "get_task", AsyncMock())

    await task_schedule.resume_task_repeat("task-1", _USER)

    data = tasks.update_many.await_args.kwargs["data"]
    assert (data["startAt"] is not None) is starts
    repeat = _schedule(data).repeat
    assert repeat is not None and not repeat.paused and repeat.failed_in_a_row == 0


async def test_stop_watching_cancels_a_task_that_never_ran_and_lets_go_of_the_webhook(
    tasks: MagicMock, mocker: MockerFixture, released: AsyncMock
):
    tasks.find_first.return_value = _row(events=[_event()], startAt=_NOW)
    mocker.patch.object(task_schedule, "get_task", AsyncMock())

    await task_schedule.stop_task_repeat("task-1", _USER)

    data = tasks.update_many.await_args.kwargs["data"]
    assert data["status"] == TaskStatus.CANCELLED
    assert data["startAt"] is None
    meta = _schedule(data)
    assert meta.repeat is None and meta.events == []
    released.assert_awaited_once_with(_USER, _WATCH)


async def test_cancelling_or_archiving_clears_the_watch_and_the_webhook(
    tasks: MagicMock, released: AsyncMock
):
    assert await task_schedule.clear_task_repeat("task-1", _USER)

    assert _schedule(tasks.update_many.await_args.kwargs["data"]).repeat is None
    released.assert_awaited_once_with(_USER, _WATCH)


async def test_a_closing_account_stops_and_cancels_its_watching_tasks(
    tasks: MagicMock, released: AsyncMock
):
    tasks.find_many.return_value = [_row(), _row(id="task-2", repeat=None)]

    await task_schedule.cancel_scheduled_tasks(_USER)

    released.assert_awaited_once_with(_USER, _WATCH)
    cancel = tasks.update_many.await_args.kwargs
    assert cancel["data"]["status"] == TaskStatus.CANCELLED
    assert {"id": {"in": ["task-1"]}} in cancel["where"]["OR"]
