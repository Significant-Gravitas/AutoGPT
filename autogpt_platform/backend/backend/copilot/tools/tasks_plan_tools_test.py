from types import SimpleNamespace

"""The task tools hold the parts of the task rubric a tool can check."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.tasks.models import (
    Assignee,
    AssigneeType,
    Task,
    TaskPriority,
    TaskStatus,
)

from ._test_data import make_session
from .models import ErrorResponse, TaskPlanToolResponse, TaskUpdatedToolResponse
from .tasks_plan_tools import ConfirmTaskPlanTool, CreateTaskTool, ProposePlanTool
from .tasks_tools import ListMyTasksTool

_USER = "user-tasks-rubric"
_NOW = datetime(2026, 11, 3, tzinfo=UTC)


def _task(
    number: int,
    title: str,
    *,
    status: TaskStatus = TaskStatus.TODO,
    parent_id: str | None = None,
    assignee: Assignee | None = None,
) -> Task:
    return Task(
        id=f"task-{number}",
        number=number,
        key=f"T-{number}",
        title=title,
        status=status,
        priority=TaskPriority.NONE,
        parent_id=parent_id,
        assignee=assignee,
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.fixture
def store(monkeypatch):
    store = MagicMock()
    store.get_task_for_work_session = AsyncMock(return_value=None)
    store.list_tasks = AsyncMock(return_value=[])
    store.create_task = AsyncMock(return_value=_task(7, "Hire a sales rep"))
    store.create_plan = AsyncMock(return_value=MagicMock())
    for module in ("tasks_helpers", "tasks_plan_tools", "tasks_tools"):
        monkeypatch.setattr(
            f"backend.copilot.tools.{module}.tasks_store", lambda: store
        )
    monkeypatch.setattr(
        "backend.copilot.tools.tasks_helpers.is_feature_enabled",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.tasks_plan_tools.publish_task_event", AsyncMock()
    )
    monkeypatch.setattr(
        "backend.copilot.tools.tasks_plan_tools.dispatch.on_task_assigned",
        AsyncMock(),
    )
    return store


async def _create(title: str, **kwargs):
    return await CreateTaskTool()._execute(
        user_id=_USER, session=make_session(_USER), title=title, **kwargs
    )


@pytest.mark.asyncio
async def test_a_plain_chat_creates_one_top_level_task(store):
    result = await _create("Hire a sales rep")

    assert isinstance(result, TaskUpdatedToolResponse)
    assert result.message == "Created T-7."
    assert store.create_task.await_args.args[1].parent_id is None


@pytest.mark.asyncio
async def test_a_task_chat_still_creates_a_sub_task(store):
    store.get_task_for_work_session.return_value = (_USER, _task(3, "Launch"))

    result = await _create("Write the announcement")

    assert isinstance(result, TaskUpdatedToolResponse)
    assert result.message == "Created T-7 under T-3."
    assert store.create_task.await_args.args[1].parent_id == "task-3"


@pytest.mark.asyncio
async def test_an_open_task_with_the_same_title_is_updated_not_duplicated(store):
    store.list_tasks.return_value = [_task(4, "Hire a  Sales Rep")]

    result = await _create("hire a sales rep")

    assert isinstance(result, ErrorResponse)
    assert "T-4" in result.message
    assert "update_task_status" in result.message
    store.create_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_finished_task_with_the_same_title_does_not_block_a_new_one(store):
    store.list_tasks.return_value = [
        _task(4, "Hire a sales rep", status=TaskStatus.DONE)
    ]

    result = await _create("Hire a sales rep")

    assert isinstance(result, TaskUpdatedToolResponse)


@pytest.mark.asyncio
async def test_the_same_title_for_someone_else_is_a_different_task(store):
    store.list_experts_for_user = AsyncMock(
        return_value=[SimpleNamespace(id="exp-1", name="Maya")]
    )
    store.resolve_assignee = AsyncMock(
        return_value=Assignee(type=AssigneeType.EXPERT, id="exp-1", name="Maya")
    )
    store.list_tasks.return_value = [
        _task(
            4,
            "Review the pricing page",
            assignee=Assignee(type=AssigneeType.OTTO, name="Otto"),
        )
    ]

    result = await _create("Review the pricing page", assignee="Maya")

    assert isinstance(result, TaskUpdatedToolResponse)


async def _propose(*titles: str, **kwargs):
    return await ProposePlanTool()._execute(
        user_id=_USER,
        session=make_session(_USER),
        goal="Reach out to leads",
        steps=[{"title": t} for t in titles],
        **kwargs,
    )


@pytest.mark.asyncio
async def test_batch_work_is_proposed_as_one_task(store):
    store.create_plan.return_value.model_dump.return_value = {"id": "plan-1"}

    result = await _propose("Hire a writer", "Hire an editor", "Hire a designer")

    assert isinstance(result, TaskPlanToolResponse)
    assert "one task with 3 items" in result.message
    assert store.create_plan.await_args.args[-1] is False


@pytest.mark.asyncio
async def test_clearly_different_pieces_of_work_split_into_sub_tasks(store):
    store.create_plan.return_value.model_dump.return_value = {"id": "plan-1"}

    result = await _propose("Blog post", "Launch email", split=True)

    assert isinstance(result, TaskPlanToolResponse)
    assert "2 sub-tasks" in result.message
    assert store.create_plan.await_args.args[-1] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "titles, split, reason",
    [
        ((), False, "at least one step"),
        (("Email the leads",), True, "split=false"),
        (("Email lead", "email  LEAD", "Email lead"), True, "split=false"),
        (tuple(f"Translate into language {n}" for n in range(11)), True, "at most 10"),
    ],
)
async def test_a_plan_that_breaks_the_shape_rules_is_refused(
    store, titles, split, reason
):
    result = await _propose(*titles, split=split)

    assert isinstance(result, ErrorResponse)
    assert reason in result.message
    store.create_plan.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_approved_one_task_plan_starts_that_task(store, monkeypatch):
    on_assigned = AsyncMock()
    monkeypatch.setattr(
        "backend.copilot.tools.tasks_plan_tools.dispatch.on_task_assigned", on_assigned
    )
    plan = MagicMock(split=False)
    parent = MagicMock(id="task-9", key="T-9", subtasks=[])
    store.confirm_plan = AsyncMock(return_value=(plan, parent))
    plan.model_dump.return_value = {}
    parent.model_dump.return_value = {}

    result = await ConfirmTaskPlanTool()._execute(
        user_id=_USER, session=make_session(_USER), plan_id="plan-1"
    )

    assert result.message == "Plan started as T-9."
    on_assigned.assert_awaited_once_with(_USER, "task-9")


@pytest.mark.asyncio
async def test_include_others_lists_the_whole_board(store):
    store.list_tasks.return_value = [
        _task(
            5,
            "Hire a sales rep",
            assignee=Assignee(type=AssigneeType.EXPERT, id="exp-1", name="Maya"),
        ),
    ]

    result = await ListMyTasksTool()._execute(
        user_id=_USER, session=make_session(_USER), include_others=True
    )

    filters = store.list_tasks.await_args.args[1]
    assert filters.assignee_type is None
    assert filters.assignee_id is None
    assert "assigned to Maya" in result.message


@pytest.mark.asyncio
async def test_without_include_others_only_your_own_tasks_are_listed(store):
    await ListMyTasksTool()._execute(user_id=_USER, session=make_session(_USER))

    assert store.list_tasks.await_args.args[1].assignee_type == AssigneeType.OTTO
