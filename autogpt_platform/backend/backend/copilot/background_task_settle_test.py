"""A finished background worker closes the tasks it was running, and the chat
that started the work is told the result in a few lines."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import TaskStatus

from backend.copilot.background_delivery import (
    continuation_message,
    settle_worker_tasks,
)
from backend.copilot.background_task import BackgroundTask
from backend.data.task_result import TaskOutcome


def _task(outcome: str | None) -> MagicMock:
    task = MagicMock(spec=BackgroundTask)
    task.id, task.user_id, task.child_session_id = "bg-1", "user-1", "worker-chat"
    task.outcome = outcome
    return task


@pytest.fixture
def stores():
    record = MagicMock(settle_work_tasks=AsyncMock(return_value=1))
    child = MagicMock(response_text="The worker's last reply")
    background = MagicMock(background_child_result=AsyncMock(return_value=child))
    with (
        patch(
            "backend.copilot.background_delivery.task_record_db", return_value=record
        ),
        patch(
            "backend.copilot.background_delivery.background_task_db",
            return_value=background,
        ),
    ):
        yield record


@pytest.mark.parametrize(
    "outcome, status, reply, reason",
    [
        ("completed", TaskStatus.DONE, "The worker's last reply", None),
        ("failed", TaskStatus.FAILED, None, "The work stopped before it finished."),
        ("timed_out", TaskStatus.FAILED, None, "The work ran out of time."),
        ("cancelled", TaskStatus.CANCELLED, None, None),
    ],
)
async def test_the_outcome_settles_the_workers_tasks(
    stores, outcome, status, reply, reason
):
    await settle_worker_tasks(_task(outcome))
    stores.settle_work_tasks.assert_awaited_once_with(
        "worker-chat", "user-1", status, reply, reason
    )


async def test_no_outcome_moves_nothing(stores):
    await settle_worker_tasks(_task(None))
    stores.settle_work_tasks.assert_not_awaited()


def _outcome(**overrides) -> TaskOutcome:
    fields = {
        "task_id": "task-1",
        "title": "Research brief: Cursor",
        "status": TaskStatus.DONE,
        "result": "Brief ready: Pro is $20 a month.",
        "document_title": "cursor-brief.md",
        "document_file_id": "file-1",
    }
    return TaskOutcome(**(fields | overrides))


def test_a_finished_task_wakes_the_chat_with_its_result_and_document():
    message = continuation_message(_task("completed"), _outcome())
    assert "Research brief: Cursor" in message
    assert "Brief ready: Pro is $20 a month." in message
    assert "[cursor-brief.md](workspace://file-1)" in message
    assert "already sees this result" in message
    assert "at most 2 short sentences" in message


def test_a_task_with_no_document_links_none():
    message = continuation_message(
        _task("completed"), _outcome(document_title=None, document_file_id=None)
    )
    assert "workspace://" not in message and "Brief ready" in message


@pytest.mark.parametrize(
    "outcome",
    [None, _outcome(status=TaskStatus.FAILED, result=None), _outcome(result=None)],
)
def test_without_a_result_the_chat_reads_the_workers_reply_and_stays_short(outcome):
    message = continuation_message(_task("failed"), outcome)
    assert 'get_sub_session_result(sub_session_id="worker-chat")' in message
    assert "say so rather than claiming a result" in message
    assert "at most 3 short lines" in message
