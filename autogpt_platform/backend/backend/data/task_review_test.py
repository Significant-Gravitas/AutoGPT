"""A result's review kept in the task's metadata (QA D1), and the task list's
filters and page (QA B5), against a mocked Prisma client."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import ResourceVisibility, TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.data import task_review
from backend.data.task import Task, unarchive_task
from backend.data.task_list import TaskFilter, list_task_page
from backend.data.task_record import is_unverified
from backend.util.exceptions import ConflictError

_USER = "user-1"
_STARTED = datetime(2026, 11, 19, 9, 0, tzinfo=UTC)


def _row(metadata: dict | None = None, **overrides) -> prisma.models.Task:
    fields = {
        "id": "task-1",
        "createdAt": _STARTED,
        "updatedAt": _STARTED,
        "userId": _USER,
        "title": "Plan",
        "status": TaskStatus.DONE,
        "createdBy": TaskCreator.OTTO,
        "isArchived": False,
        "metadata": json.dumps(metadata or {}),
        "visibility": ResourceVisibility.PRIVATE,
        "attempt": 1,
        "isSuggested": False,
        "startedAt": _STARTED,
        "doneWhen": "A plan in the doc",
    }
    return prisma.models.Task(**(fields | overrides))


@pytest.fixture
def tasks(mocker: MockerFixture) -> MagicMock:
    client = MagicMock(
        find_first=AsyncMock(return_value=_row()),
        find_many=AsyncMock(return_value=[]),
        update_many=AsyncMock(return_value=1),
    )
    mocker.patch.object(prisma.models.Task, "prisma", return_value=client)
    return client


async def test_accepting_marks_the_result_accepted(tasks: MagicMock, mocker):
    mocker.patch.object(task_review, "get_task", AsyncMock(return_value=MagicMock()))

    await task_review.set_task_review("task-1", _USER, "accepted")

    update = tasks.update_many.await_args.kwargs
    assert update["where"]["userId"] == _USER
    task = Task.from_db(_row(update["data"]["metadata"].data))
    assert task.review is not None and task.review.state == "accepted"
    # Accepted by the user: the record no longer reads "Not verified yet".
    assert not is_unverified(task, [])
    assert is_unverified(Task.from_db(_row()), [])


async def test_only_a_finished_result_can_be_accepted(tasks: MagicMock):
    tasks.find_first.return_value = _row(status=TaskStatus.FAILED)

    with pytest.raises(ConflictError):
        await task_review.set_task_review("task-1", _USER, "accepted")
    tasks.update_many.assert_not_awaited()


def test_a_review_of_an_earlier_run_does_not_count():
    review = {"state": "accepted", "at": (_STARTED - timedelta(hours=1)).isoformat()}

    assert Task.from_db(_row({"review": review})).review is None


async def test_the_list_pages_and_filters_for_the_user(tasks: MagicMock):
    tasks.find_many.return_value = [_row(id=f"t{i}") for i in range(3)]

    page = await list_task_page(
        _USER,
        TaskFilter(
            statuses=[TaskStatus.DONE, TaskStatus.FAILED],
            owned_by_otto=True,
            search="  launch   plan ",
        ),
        limit=2,
        offset=4,
    )

    query = tasks.find_many.await_args.kwargs
    assert query["where"]["userId"] == _USER
    assert query["where"]["status"] == {"in": [TaskStatus.DONE, TaskStatus.FAILED]}
    assert query["where"]["ownerExpertId"] is None
    # QA pass 3, M4: the words can be anywhere the task says something.
    match = {"contains": "launch plan", "mode": "insensitive"}
    assert query["where"]["OR"] == [
        {"title": match},
        {"description": match},
        {"goal": match},
        {"doneWhen": match},
        {"result": match},
    ]
    assert query["order"] == [{"updatedAt": "desc"}, {"id": "asc"}]
    assert (query["take"], query["skip"]) == (3, 4)
    assert [t.id for t in page.tasks] == ["t0", "t1"]
    assert (page.has_more, page.next_offset) == (True, 6)


async def test_the_last_page_says_nothing_follows(tasks: MagicMock):
    tasks.find_many.return_value = [_row()]

    page = await list_task_page(_USER, TaskFilter(), limit=50, offset=0)

    assert (page.has_more, page.next_offset) == (False, None)


@pytest.mark.parametrize(
    "sort, order",
    [
        ("updated", {"updatedAt": "desc"}),
        ("created", {"createdAt": "desc"}),
        ("title", {"title": "asc"}),
    ],
)
async def test_the_list_is_ordered_as_asked(tasks: MagicMock, sort, order):
    """QA pass 3, M3: newest first for the dates, A to Z for title, ties by
    id so offset pages never overlap."""
    await list_task_page(_USER, TaskFilter(sort=sort), limit=50, offset=0)

    assert tasks.find_many.await_args.kwargs["order"] == [order, {"id": "asc"}]


async def test_unarchiving_brings_back_only_the_users_archived_task(tasks: MagicMock):
    """QA pass 3, M5: visibility only; nothing else on the task changes."""
    restored = await unarchive_task("task-1", _USER)

    update = tasks.update_many.await_args.kwargs
    assert restored
    assert update["where"] == {"id": "task-1", "userId": _USER, "isArchived": True}
    assert update["data"] == {"isArchived": False}
