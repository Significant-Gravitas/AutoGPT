"""Tests for start_task: a task, a background worker, and the turn ends."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import TaskCreator, TaskStatus

from backend.copilot import turn_end
from backend.copilot.background_task import BackgroundTask
from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.start_task import StartTaskTool, worker_prompt
from backend.copilot.tools.task_models import TaskStartedResponse
from backend.copilot.tree import TreeRefusal
from backend.data.task import Task, TaskPatch

from ._test_data import make_session

_USER = "test-user-start-task"
_NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": _USER,
        "title": "Plan the launch funnel",
        "status": TaskStatus.IN_PROGRESS,
        "created_by": TaskCreator.OTTO,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Task(**(fields | overrides))


@pytest.fixture
def store():
    store = MagicMock()
    store.create_task = AsyncMock(return_value=_task(status=TaskStatus.TODO))
    store.update_task = AsyncMock(return_value=_task())
    store.attach_task_session = AsyncMock(
        return_value=_task(session_ids=["chat", "worker-chat"])
    )
    with patch("backend.copilot.tools.start_task.task_db", return_value=store):
        yield store


@pytest.fixture
def session():
    session = make_session(_USER)
    yield session
    turn_end.clear(session.session_id)


def _background() -> MagicMock:
    background = MagicMock(spec=BackgroundTask)
    background.id = "bg-1"
    background.child_session_id = "worker-chat"
    return background


async def test_starts_the_work_in_the_background_and_ends_the_turn(store, session):
    start = AsyncMock(return_value=_background())
    with patch("backend.copilot.tools.start_task.start_background_subtask", start):
        result = await StartTaskTool()._execute(
            _USER, session, title="Plan the launch funnel", brief="Email + ads, Nov 20"
        )

    assert isinstance(result, TaskStartedResponse)
    assert result.in_background
    assert result.worker_session_id == "worker-chat"
    assert result.task.session_ids == ["chat", "worker-chat"]
    store.update_task.assert_awaited_once_with(
        "task-1", _USER, TaskPatch(status=TaskStatus.IN_PROGRESS)
    )
    prompt = start.await_args.args[2]
    assert "Email + ads, Nov 20" in prompt and "task-1" in prompt
    store.attach_task_session.assert_awaited_once_with("task-1", "worker-chat", _USER)
    assert turn_end.is_ending(session.session_id)
    assert not turn_end.refuses_calls(session.session_id)


async def test_without_background_work_the_task_is_worked_on_here(store, session):
    refusal = TreeRefusal("Background subtasks are not enabled for this account.")
    with patch(
        "backend.copilot.tools.start_task.start_background_subtask",
        AsyncMock(side_effect=refusal),
    ):
        result = await StartTaskTool()._execute(
            _USER, session, title="Plan the launch funnel", brief="Email + ads"
        )

    assert isinstance(result, TaskStartedResponse)
    assert not result.in_background
    assert "Work on it here now" in result.message
    assert not turn_end.is_ending(session.session_id)


@pytest.mark.parametrize("brief", ["", "   "])
async def test_a_task_needs_a_brief(store, session, brief):
    result = await StartTaskTool()._execute(_USER, session, title="x", brief=brief)

    assert isinstance(result, ErrorResponse)
    store.create_task.assert_not_awaited()


async def test_a_bad_title_is_reported_to_the_model(store, session):
    store.create_task = AsyncMock(side_effect=ValueError("A task needs a title."))

    result = await StartTaskTool()._execute(_USER, session, title=" ", brief="b")

    assert isinstance(result, ErrorResponse)
    assert result.message == "A task needs a title."


def test_the_worker_is_told_to_decide_and_only_ask_when_blocked():
    prompt = worker_prompt(_task(), "Write the funnel.")

    assert "Write the funnel." in prompt
    assert "assumption" in prompt and "ask_question" in prompt
