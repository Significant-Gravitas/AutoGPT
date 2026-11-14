"""Tests for the task tools: what they ask the data layer and what they answer."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import TaskCreator, TaskStatus

from backend.copilot.tools.models import ErrorResponse
from backend.copilot.tools.task_models import (
    TaskArchivedResponse,
    TaskCreatedResponse,
    TaskDetailsResponse,
    TaskListResponse,
    TaskUpdatedResponse,
)
from backend.copilot.tools.tasks import (
    ArchiveTaskTool,
    CreateTaskTool,
    GetTaskTool,
    ListTasksTool,
    UpdateTaskTool,
)
from backend.data.task import Task, TaskPatch
from backend.util.exceptions import NotFoundError

from ._test_data import make_session

_USER = "test-user-tasks"
_NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": _USER,
        "title": "Build feature X",
        "status": TaskStatus.TODO,
        "created_by": TaskCreator.OTTO,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Task(**(fields | overrides))


@pytest.fixture
def db():
    store = MagicMock()
    with patch("backend.copilot.tools.tasks.task_db", return_value=store):
        yield store


@pytest.mark.parametrize(
    "expert_id, creator",
    [(None, TaskCreator.OTTO), ("expert-7", TaskCreator.EXPERT)],
)
async def test_create_links_this_chat_and_names_who_made_it(db, expert_id, creator):
    session = make_session(_USER, expert_id=expert_id)
    db.create_task = AsyncMock(return_value=_task(session_ids=[session.session_id]))

    result = await CreateTaskTool()._execute(
        _USER, session, title="Build feature X", expert_id="expert-2"
    )

    assert isinstance(result, TaskCreatedResponse)
    assert result.task.session_ids == [session.session_id]
    kwargs = db.create_task.await_args.kwargs
    assert kwargs["user_id"] == _USER
    assert kwargs["session_id"] == session.session_id
    assert kwargs["owner_expert_id"] == "expert-2"
    assert kwargs["created_by"] == creator
    assert kwargs["created_by_expert_id"] == expert_id


async def test_create_reports_a_bad_title_to_the_model(db):
    db.create_task = AsyncMock(side_effect=ValueError("A task needs a title."))

    result = await CreateTaskTool()._execute(_USER, make_session(_USER), title=" ")

    assert isinstance(result, ErrorResponse)
    assert result.error == "create_task_failed"
    assert result.message == "A task needs a title."


@pytest.mark.parametrize("this_chat_only", [True, False])
async def test_list_scopes_to_this_chat_only_when_asked(db, this_chat_only):
    session = make_session(_USER)
    db.list_tasks = AsyncMock(return_value=[_task(), _task(id="task-2")])

    result = await ListTasksTool()._execute(
        _USER, session, this_chat_only=this_chat_only, status="IN_PROGRESS"
    )

    assert isinstance(result, TaskListResponse)
    assert result.count == 2
    db.list_tasks.assert_awaited_once_with(
        user_id=_USER,
        status=TaskStatus.IN_PROGRESS,
        owner_expert_id=None,
        session_id=session.session_id if this_chat_only else None,
    )


async def test_list_refuses_an_unknown_status(db):
    db.list_tasks = AsyncMock(return_value=[])

    result = await ListTasksTool()._execute(_USER, make_session(_USER), status="LATE")

    assert isinstance(result, ErrorResponse)
    db.list_tasks.assert_not_awaited()


async def test_get_finds_the_task_or_says_it_is_missing(db):
    db.get_task = AsyncMock(side_effect=[_task(), None])
    tool, session = GetTaskTool(), make_session(_USER)

    found = await tool._execute(_USER, session, task_id="task-1")
    missing = await tool._execute(_USER, session, task_id="task-9")

    assert isinstance(found, TaskDetailsResponse) and found.task.id == "task-1"
    assert isinstance(missing, ErrorResponse) and missing.error == "task_not_found"


async def test_update_builds_the_patch_from_the_call(db):
    db.update_task = AsyncMock(return_value=_task(status=TaskStatus.DONE))

    result = await UpdateTaskTool()._execute(
        _USER, make_session(_USER), task_id="task-1", status="DONE"
    )

    assert isinstance(result, TaskUpdatedResponse)
    assert result.task.status == TaskStatus.DONE
    db.update_task.assert_awaited_once_with(
        "task-1", _USER, TaskPatch(status=TaskStatus.DONE)
    )


async def test_update_can_give_a_task_its_goal_and_done_when_cut_to_a_line(db):
    db.update_task = AsyncMock(return_value=_task())

    await UpdateTaskTool()._execute(
        _USER,
        make_session(_USER),
        task_id="task-1",
        goal="Know the evidence set is complete",
        done_when="check " * 80,
    )

    patch = db.update_task.await_args.args[2]
    assert patch.goal == "Know the evidence set is complete"
    assert patch.done_when is not None and len(patch.done_when) <= 160


async def test_update_passes_not_found_through_as_an_error(db):
    db.update_task = AsyncMock(side_effect=NotFoundError("Task task-9 not found"))

    result = await UpdateTaskTool()._execute(
        _USER, make_session(_USER), task_id="task-9", title="x"
    )

    assert isinstance(result, ErrorResponse)
    assert result.message == "Task task-9 not found"


@pytest.mark.parametrize("archived", [True, False])
async def test_archive(db, archived):
    db.archive_task = AsyncMock(return_value=archived)

    result = await ArchiveTaskTool()._execute(
        _USER, make_session(_USER), task_id="task-1"
    )

    expected = TaskArchivedResponse if archived else ErrorResponse
    assert isinstance(result, expected)
    db.archive_task.assert_awaited_once_with("task-1", _USER)
