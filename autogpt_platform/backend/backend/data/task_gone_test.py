"""A task that is missing, archived or someone else's reads the same way on
every task path, with no id in the words the user sees (QA pass 3, P5/U8)."""

from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from pytest_mock import MockerFixture

from backend.copilot import task_run as copilot_task_run
from backend.data import task_review, task_schedule
from backend.data.task import TASK_GONE, attach_task_session, update_task
from backend.util.exceptions import NotFoundError

_USER = "user-1"


@pytest.fixture(autouse=True)
def nothing_found(mocker: MockerFixture) -> MagicMock:
    client = MagicMock(
        find_first=AsyncMock(return_value=None),
        find_unique=AsyncMock(return_value=None),
        update_many=AsyncMock(return_value=0),
    )
    mocker.patch.object(prisma.models.Task, "prisma", return_value=client)
    return client


@pytest.mark.parametrize(
    "call",
    [
        lambda: task_schedule.pause_task_repeat("task-1", _USER),
        lambda: task_schedule.resume_task_repeat("task-1", _USER),
        lambda: task_schedule.stop_task_repeat("task-1", _USER),
        lambda: task_schedule.set_task_start("task-1", _USER, None),
        lambda: task_review.set_task_review("task-1", _USER, "accepted"),
        lambda: update_task("task-1", _USER, MagicMock()),
        lambda: attach_task_session("task-1", "chat-1", _USER),
    ],
)
async def test_a_task_that_is_gone_reads_without_its_id(call):
    with pytest.raises(NotFoundError) as gone:
        await call()

    assert str(gone.value) == TASK_GONE
    assert "task-1" not in str(gone.value)


async def test_cancelling_a_task_that_is_gone_reads_without_its_id(
    mocker: MockerFixture,
):
    store = MagicMock(get_task=AsyncMock(return_value=None))
    mocker.patch.object(copilot_task_run, "task_db", return_value=store)

    with pytest.raises(NotFoundError) as gone:
        await copilot_task_run.cancel_task_run(_USER, "task-1")

    assert str(gone.value) == TASK_GONE
