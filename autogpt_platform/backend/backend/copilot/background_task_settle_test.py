"""A finished background worker closes the tasks it was running."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import TaskStatus

from backend.copilot.background_delivery import settle_worker_tasks
from backend.copilot.background_task import BackgroundTask


def _task(outcome: str | None) -> MagicMock:
    task = MagicMock(spec=BackgroundTask)
    task.id, task.user_id, task.child_session_id = "bg-1", "user-1", "worker-chat"
    task.outcome = outcome
    return task


@pytest.mark.parametrize(
    "outcome, status",
    [
        ("completed", TaskStatus.DONE),
        ("failed", TaskStatus.FAILED),
        ("timed_out", TaskStatus.FAILED),
        ("cancelled", TaskStatus.CANCELLED),
    ],
)
async def test_the_outcome_settles_the_open_tasks(outcome, status):
    store = MagicMock(move_session_tasks=AsyncMock(return_value=1))
    with patch("backend.copilot.background_delivery.task_db", return_value=store):
        await settle_worker_tasks(_task(outcome))

    store.move_session_tasks.assert_awaited_once_with(
        "worker-chat", "user-1", [TaskStatus.TODO, TaskStatus.IN_PROGRESS], status
    )


async def test_no_outcome_moves_nothing():
    store = MagicMock(move_session_tasks=AsyncMock())
    with patch("backend.copilot.background_delivery.task_db", return_value=store):
        await settle_worker_tasks(_task(None))

    store.move_session_tasks.assert_not_awaited()
