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
    store.update_task = AsyncMock(
        return_value=_task(session_ids=["chat", "worker-chat"])
    )
    store.attach_task_session = AsyncMock(
        return_value=_task(session_ids=["chat", "worker-chat"])
    )
    store.get_task = AsyncMock(return_value=None)
    # The record: this chat works on no task and nothing with the title is open.
    store.work_tasks = AsyncMock(return_value=[])
    store.find_open_task_by_title = AsyncMock(return_value=None)
    store.add_activity = AsyncMock()
    with (
        patch("backend.copilot.tools.start_task.task_db", return_value=store),
        patch("backend.copilot.tools.start_task.task_record_db", return_value=store),
    ):
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


async def test_starts_the_work_in_the_background_and_wraps_up_the_turn(store, session):
    start = AsyncMock(return_value=_background())
    with patch("backend.copilot.tools.start_task.start_background_subtask", start):
        result = await StartTaskTool()._execute(
            _USER, session, title="Plan the launch funnel", brief="Email + ads, Nov 20"
        )

    assert isinstance(result, TaskStartedResponse)
    assert result.in_background
    assert result.worker_session_id == "worker-chat"
    assert result.task.session_ids == ["chat", "worker-chat"]
    store.update_task.assert_any_await(
        "task-1", _USER, TaskPatch(status=TaskStatus.IN_PROGRESS)
    )
    # The worker's chat is named as the one doing the work, so its tool calls
    # land on this task's record.
    store.update_task.assert_any_await(
        "task-1", _USER, TaskPatch(work_session_id="worker-chat")
    )
    prompt = start.await_args.args[2]
    assert "Email + ads, Nov 20" in prompt and "task-1" in prompt
    store.attach_task_session.assert_awaited_once_with("task-1", "worker-chat", _USER)
    assert not turn_end.is_ending(session.session_id)
    assert turn_end.refusal(session.session_id, "start_task") is None
    assert turn_end.refusal(session.session_id, "web_search") is not None


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
    assert turn_end.refusal(session.session_id, "web_search") is None


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


async def test_a_task_carries_its_goal_and_what_done_looks_like(store, session):
    start = AsyncMock(return_value=_background())
    with patch("backend.copilot.tools.start_task.start_background_subtask", start):
        await StartTaskTool()._execute(
            _USER,
            session,
            title="Plan the launch funnel",
            brief="Email + ads",
            goal="Launch on Nov 20",
            done_when="Emails and ads scheduled",
        )

    created = store.create_task.await_args.kwargs
    assert created["goal"] == "Launch on Nov 20"
    assert created["done_when"] == "Emails and ads scheduled"
    assert created["parent_task_id"] is None
    assert "Done when: " not in worker_prompt(_task(), "brief")
    assert "Done when: Emails scheduled" in worker_prompt(
        _task(done_when="Emails scheduled"), "brief"
    )


async def test_the_same_work_twice_returns_the_first_task_and_starts_no_worker(
    store, session
):
    store.find_open_task_by_title.return_value = _task(work_session_id="worker-chat")
    start = AsyncMock(return_value=_background())
    with patch("backend.copilot.tools.start_task.start_background_subtask", start):
        result = await StartTaskTool()._execute(
            _USER, session, title="Plan the launch funnel", brief="again"
        )

    assert isinstance(result, TaskStartedResponse)
    assert "already running" in result.message
    assert result.worker_session_id == "worker-chat"
    store.create_task.assert_not_awaited()
    start.assert_not_awaited()


async def test_started_inside_a_task_it_becomes_that_tasks_subtask(store, session):
    store.work_tasks.return_value = [_task(id="parent-1")]
    start = AsyncMock(return_value=_background())
    with patch("backend.copilot.tools.start_task.start_background_subtask", start):
        await StartTaskTool()._execute(
            _USER, session, title="Draft the emails", brief="b", expert_id="exp-1"
        )

    assert store.create_task.await_args.kwargs["parent_task_id"] == "parent-1"
    noted = [call.args[0] for call in store.add_activity.await_args_list]
    assert noted == ["task-1", "parent-1"]


async def test_a_subtask_that_cannot_be_made_is_refused_with_the_reason(store, session):
    store.create_task.side_effect = ValueError(
        "A subtask cannot have subtasks: do this part yourself."
    )
    result = await StartTaskTool()._execute(_USER, session, title="Deeper", brief="b")

    assert isinstance(result, ErrorResponse)
    assert "do this part yourself" in result.message


@pytest.mark.parametrize(
    "status, retry, line",
    [
        (TaskStatus.FAILED, True, "Trying again"),
        (TaskStatus.DONE, False, "Picked up again"),
    ],
)
async def test_an_existing_task_is_run_again_on_its_own_record(
    store, session, status, retry, line
):
    store.get_task.return_value = _task(status=status)
    start = AsyncMock(return_value=_background())
    with patch("backend.copilot.tools.start_task.start_background_subtask", start):
        result = await StartTaskTool()._execute(
            _USER, session, brief="Linear is back", task_id="task-1"
        )

    assert isinstance(result, TaskStartedResponse)
    store.create_task.assert_not_awaited()
    store.update_task.assert_any_await(
        "task-1",
        _USER,
        TaskPatch(
            status=TaskStatus.IN_PROGRESS,
            description="Linear is back",
            retry=retry,
            result="",
        ),
    )
    assert line in store.add_activity.await_args.args[2].text
    start.assert_awaited_once()


async def test_a_task_already_being_worked_on_is_not_started_again(store, session):
    store.get_task.return_value = _task(status=TaskStatus.NEEDS_INPUT)
    result = await StartTaskTool()._execute(
        _USER, session, brief="again", task_id="task-1"
    )

    assert isinstance(result, ErrorResponse)
    assert "already being worked on" in result.message


async def test_a_task_kept_in_the_chat_names_that_chat_as_its_work_chat(store, session):
    refused = AsyncMock(side_effect=TreeRefusal("Background subtasks are off."))
    with patch("backend.copilot.tools.start_task.start_background_subtask", refused):
        result = await StartTaskTool()._execute(
            _USER, session, title="Plan the launch funnel", brief="b"
        )

    assert isinstance(result, TaskStartedResponse)
    assert not result.in_background
    store.update_task.assert_any_await(
        "task-1", _USER, TaskPatch(work_session_id=session.session_id)
    )


async def test_a_task_an_expert_starts_from_their_own_chat_is_theirs(store, session):
    session.expert_id = "expert-riya"
    start = AsyncMock(return_value=_background())
    with patch("backend.copilot.tools.start_task.start_background_subtask", start):
        await StartTaskTool()._execute(_USER, session, title="Brief", brief="Do it")
    assert store.create_task.await_args.kwargs["owner_expert_id"] == "expert-riya"

    # Handed to someone else: theirs, not the chat's expert's.
    await_count = store.create_task.await_count
    with patch("backend.copilot.tools.start_task.start_background_subtask", start):
        await StartTaskTool()._execute(
            _USER, session, title="Deck", brief="Do it", expert_id="expert-max"
        )
    assert store.create_task.await_count == await_count + 1
    assert store.create_task.await_args.kwargs["owner_expert_id"] == "expert-max"


async def test_a_chat_whose_expert_left_the_team_still_starts_the_task(store, session):
    session.expert_id = "expert-gone"
    store.create_task.side_effect = [
        ValueError("Expert expert-gone not found"),
        _task(status=TaskStatus.TODO),
    ]
    start = AsyncMock(return_value=_background())
    with patch("backend.copilot.tools.start_task.start_background_subtask", start):
        result = await StartTaskTool()._execute(
            _USER, session, title="Brief", brief="Do it"
        )
    assert isinstance(result, TaskStartedResponse)
    assert store.create_task.await_args.kwargs["owner_expert_id"] is None


async def test_a_long_goal_is_cut_to_a_line_the_user_can_skim(store, session):
    start = AsyncMock(return_value=_background())
    with patch("backend.copilot.tools.start_task.start_background_subtask", start):
        await StartTaskTool()._execute(
            _USER,
            session,
            title="Brief",
            brief="Do it",
            goal="word " * 80,
            done_when="check " * 80,
        )
    fields = store.create_task.await_args.kwargs
    assert len(fields["goal"]) <= 120 and len(fields["done_when"]) <= 160


def test_the_worker_is_told_how_to_write_on_the_task_and_how_to_finish():
    prompt = worker_prompt(_task(done_when="Brief saved"), "Research Cursor.")
    for needed in (
        'run_capability(id="tool:<name>"',
        'tool:report_progress {"status_text"',
        'tool:add_evidence {"claim"',
        'tool:finish_task {"summary"',
        "write_workspace_file",
        "Never put the document in your reply",
        "Done when: Brief saved",
    ):
        assert needed in prompt
