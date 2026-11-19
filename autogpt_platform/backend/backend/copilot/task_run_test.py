"""Starting and stopping a task's work from the task itself."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import TaskActivityKind, TaskActor, TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.copilot import task_run
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.model import ChatSession
from backend.copilot.tools.start_task import BRIEF_MAX_CHARS, worker_prompt_text
from backend.copilot.turn_queue import InflightCapExceeded
from backend.data.task import Task, TaskPatch
from backend.data.task_run import QUEUED_TEXT, TaskRunClaim
from backend.data.task_schedule_meta import TaskRepeat
from backend.util.exceptions import ConflictError, ExpertNotFoundError, NotFoundError

_MOD = "backend.copilot.task_run"
_NOW = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)
USER = "user-1"


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": USER,
        "title": "Plan the launch",
        "description": "Write a launch plan for Nov 20.",
        "status": TaskStatus.TODO,
        "created_by": TaskCreator.USER,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Task(**(fields | overrides))


def _claim(before: TaskStatus = TaskStatus.TODO, attempt: int = 1) -> TaskRunClaim:
    return TaskRunClaim(
        before=_task(status=before),
        task=_task(status=TaskStatus.IN_PROGRESS, attempt=attempt),
        retry=before == TaskStatus.FAILED,
    )


class _Deps(SimpleNamespace):
    run: MagicMock
    tasks: MagicMock
    record: MagicMock
    create_chat: AsyncMock
    delete_chat: AsyncMock
    schedule: AsyncMock
    enqueue: AsyncMock


@pytest.fixture
def deps(mocker: MockerFixture) -> _Deps:
    run, tasks, record = MagicMock(), MagicMock(), MagicMock()
    run.claim_task_run = AsyncMock(return_value=_claim())
    run.undo_task_run = AsyncMock()
    run.mark_run_queued = AsyncMock(
        return_value=_task(status=TaskStatus.IN_PROGRESS, is_queued=True)
    )
    tasks.attach_task_session = AsyncMock()
    tasks.update_task = AsyncMock(
        side_effect=lambda task_id, user_id, patch: _task(
            status=patch.status or TaskStatus.IN_PROGRESS,
            work_session_id=patch.work_session_id or "chat-new",
        )
    )
    record.add_activity = AsyncMock()
    mocker.patch(f"{_MOD}.task_run_db", return_value=run)
    mocker.patch(f"{_MOD}.task_db", return_value=tasks)
    mocker.patch(f"{_MOD}.task_record_db", return_value=record)
    session = ChatSession.new(USER, dry_run=False, work_task_id="task-1")
    session.session_id = "chat-new"
    return _Deps(
        run=run,
        tasks=tasks,
        record=record,
        create_chat=mocker.patch(
            f"{_MOD}.create_chat_session", new_callable=AsyncMock, return_value=session
        ),
        delete_chat=mocker.patch(f"{_MOD}.delete_chat_session", new_callable=AsyncMock),
        schedule=mocker.patch(
            f"{_MOD}.schedule_chat_turn", new_callable=AsyncMock, return_value="turn-1"
        ),
        enqueue=mocker.patch(
            f"{_MOD}.turn_queue.try_enqueue_turn", new_callable=AsyncMock
        ),
    )


def _activity(deps: _Deps):
    deps.record.add_activity.assert_awaited_once()
    return deps.record.add_activity.await_args.args[2]


@pytest.mark.asyncio
async def test_start_opens_a_work_chat_and_sends_the_brief(deps: _Deps):
    task = await task_run.start_task_run(USER, "task-1", organization_id="org-1")

    assert task.work_session_id == "chat-new"
    deps.create_chat.assert_awaited_once()
    chat_args = deps.create_chat.await_args.kwargs
    assert chat_args["work_task_id"] == "task-1"
    assert chat_args["expert_id"] is None
    assert chat_args["organization_id"] == "org-1"
    deps.tasks.attach_task_session.assert_awaited_once_with("task-1", "chat-new", USER)
    deps.tasks.update_task.assert_awaited_once_with(
        "task-1", USER, TaskPatch(work_session_id="chat-new")
    )
    turn = deps.schedule.await_args.kwargs
    assert turn["session_id"] == "chat-new"
    assert "Write a launch plan for Nov 20." in turn["message"]
    assert "task id task-1" in turn["message"]
    assert turn["message_metadata"] == {task_run.TASK_RUN_KEY: "task-1"}
    line = _activity(deps)
    assert (line.kind, line.text, line.actor_type) == (
        TaskActivityKind.STARTED,
        "You started it",
        TaskActor.USER,
    )
    assert line.data == {}
    deps.run.undo_task_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_try_again_counts_the_attempt_and_keeps_the_instructions(deps: _Deps):
    deps.run.claim_task_run.return_value = _claim(TaskStatus.FAILED, attempt=2)

    await task_run.start_task_run(USER, "task-1", "  Use the Q3 numbers.  ")

    message = deps.schedule.await_args.kwargs["message"]
    assert "The user asked for this change: Use the Q3 numbers." in message
    line = _activity(deps)
    assert line.kind == TaskActivityKind.RETRIED
    assert line.text == "You asked for another try (attempt 2)"
    assert line.data == {"instructions": "Use the Q3 numbers."}


_RULE = f"\n\n{task_run.BLOCKED_RULE}"


def test_the_brief_falls_back_to_the_goal_then_the_title():
    assert task_run.run_brief(_task(description=None, goal="Ship by Fri"), None) == (
        "Goal: Ship by Fri" + _RULE
    )
    assert task_run.run_brief(_task(description=None), None) == (
        "Plan the launch" + _RULE
    )


def test_a_page_run_is_told_to_finish_a_task_it_cannot_do_as_blocked():
    prompt = task_run.run_prompt(_task(), "Use the Q3 numbers.")

    assert 'call finish_task with "outcome": "blocked"' in prompt
    assert prompt.index("Use the Q3 numbers.") < prompt.index(task_run.BLOCKED_RULE)


def test_the_chat_workers_instructions_are_unchanged():
    chat = worker_prompt_text("Plan", "task-1", "Write a plan.")

    assert task_run.BLOCKED_RULE not in chat


def test_the_brief_carries_the_goal_beside_the_description():
    task = _task(goal="Launch on   Nov 20", done_when="A plan is saved")

    brief = task_run.run_brief(task, None)
    prompt = task_run.run_prompt(task, None)

    assert brief == "Goal: Launch on Nov 20\n\nWrite a launch plan for Nov 20." + _RULE
    assert "Done when: A plan is saved" in prompt


def test_a_goal_the_description_already_says_is_not_repeated():
    task = _task(goal="a launch plan for nov 20")

    assert task_run.run_brief(task, None) == "Write a launch plan for Nov 20." + _RULE


def test_the_note_survives_a_description_cut_to_fit():
    note = "Name the two partner schools in section 4; they agreed"
    task = _task(description="Field notes. " * 600, goal="Submit the report")

    brief = task_run.run_brief(task, note)

    assert len(brief) <= BRIEF_MAX_CHARS
    assert brief.startswith("Goal: Submit the report\n\nField notes.")
    assert brief.endswith(f"\n\nThe user asked for this change: {note}" + _RULE)


@pytest.mark.asyncio
async def test_trying_a_cancelled_task_again_says_which_attempt(deps: _Deps):
    deps.run.claim_task_run.return_value = TaskRunClaim(
        before=_task(status=TaskStatus.CANCELLED),
        task=_task(status=TaskStatus.IN_PROGRESS, attempt=2),
        retry=True,
    )

    await task_run.start_task_run(USER, "task-1")

    line = _activity(deps)
    assert line.kind == TaskActivityKind.RETRIED
    assert line.text == "You asked for another try (attempt 2)"


@pytest.mark.asyncio
async def test_an_owner_who_left_hands_the_run_to_otto(deps: _Deps):
    deps.run.claim_task_run.return_value = TaskRunClaim(
        before=_task(owner_expert_id="exp-1"),
        task=_task(status=TaskStatus.IN_PROGRESS, owner_expert_id="exp-1"),
        retry=False,
    )
    session = deps.create_chat.return_value
    deps.create_chat.side_effect = [ExpertNotFoundError("exp-1"), session]

    await task_run.start_task_run(USER, "task-1")

    owners = [call.kwargs["expert_id"] for call in deps.create_chat.await_args_list]
    assert owners == ["exp-1", None]


@pytest.mark.asyncio
async def test_a_full_running_pool_queues_the_turn(deps: _Deps):
    deps.schedule.side_effect = ConcurrentTurnLimitError("busy", cap=5, used=5)

    await task_run.start_task_run(USER, "task-1")

    deps.enqueue.assert_awaited_once()
    assert deps.enqueue.await_args.kwargs["session_id"] == "chat-new"
    deps.run.undo_task_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refused_turn_leaves_the_task_as_it_was(deps: _Deps):
    deps.schedule.side_effect = ConcurrentTurnLimitError("busy", cap=5, used=5)
    deps.enqueue.side_effect = InflightCapExceeded(used=15, cap=15)

    with pytest.raises(InflightCapExceeded):
        await task_run.start_task_run(USER, "task-1")

    deps.run.undo_task_run.assert_awaited_once()
    deps.delete_chat.assert_awaited_once_with("chat-new", USER)
    deps.record.add_activity.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ConflictError("busy"), NotFoundError("gone")])
async def test_a_task_that_cannot_be_claimed_starts_nothing(deps: _Deps, error):
    deps.run.claim_task_run.side_effect = error

    with pytest.raises(type(error)):
        await task_run.start_task_run(USER, "task-1")

    deps.create_chat.assert_not_awaited()
    deps.schedule.assert_not_awaited()
    deps.run.undo_task_run.assert_not_awaited()


@pytest.fixture
def stops(mocker: MockerFixture) -> SimpleNamespace:
    background = MagicMock()
    background.cancel_queued_background_followup = AsyncMock(return_value=False)
    mocker.patch(f"{_MOD}.background_task_db", return_value=background)
    return SimpleNamespace(
        subtasks=mocker.patch(
            f"{_MOD}.cancel_background_subtasks", new_callable=AsyncMock
        ),
        queued=mocker.patch(
            f"{_MOD}.turn_queue.cancel_queued_turn",
            new_callable=AsyncMock,
            return_value=False,
        ),
        active=mocker.patch(
            f"{_MOD}.stream_registry.get_active_session",
            new_callable=AsyncMock,
            return_value=(SimpleNamespace(turn_id="turn-9"), ""),
        ),
        cancel=mocker.patch(f"{_MOD}.enqueue_cancel_task", new_callable=AsyncMock),
    )


@pytest.mark.asyncio
async def test_cancel_stops_the_running_turn_and_its_subtasks(
    deps: _Deps, stops: SimpleNamespace
):
    deps.tasks.get_task = AsyncMock(
        return_value=_task(status=TaskStatus.IN_PROGRESS, work_session_id="chat-w")
    )
    deps.tasks.update_task.side_effect = None
    deps.tasks.update_task.return_value = _task(
        status=TaskStatus.CANCELLED, work_session_id="chat-w"
    )

    task = await task_run.cancel_task_run(USER, "task-1")

    assert task.status == TaskStatus.CANCELLED
    deps.tasks.update_task.assert_awaited_once_with(
        "task-1", USER, TaskPatch(status=TaskStatus.CANCELLED)
    )
    stops.subtasks.assert_awaited_once_with(USER, "chat-w")
    stops.cancel.assert_awaited_once_with("chat-w", "turn-9")
    line = _activity(deps)
    assert (line.text, line.actor_type) == ("You stopped it", TaskActor.USER)


@pytest.mark.asyncio
async def test_cancel_takes_a_queued_turn_off_the_queue(
    deps: _Deps, stops: SimpleNamespace
):
    deps.tasks.get_task = AsyncMock(
        return_value=_task(status=TaskStatus.IN_PROGRESS, work_session_id="chat-w")
    )
    stops.queued.return_value = True

    await task_run.cancel_task_run(USER, "task-1")

    stops.cancel.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED]
)
async def test_cancel_with_nothing_running_changes_nothing(
    deps: _Deps, stops: SimpleNamespace, status: TaskStatus
):
    deps.tasks.get_task = AsyncMock(
        return_value=_task(status=status, work_session_id="chat-w")
    )

    task = await task_run.cancel_task_run(USER, "task-1")

    assert task.status == status
    deps.tasks.update_task.assert_not_awaited()
    stops.cancel.assert_not_awaited()
    deps.record.add_activity.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_of_another_users_task_is_not_found(
    deps: _Deps, stops: SimpleNamespace
):
    deps.tasks.get_task = AsyncMock(return_value=None)

    with pytest.raises(NotFoundError):
        await task_run.cancel_task_run(USER, "task-1")


@pytest.mark.asyncio
async def test_a_task_started_at_its_time_uses_the_sweeps_claim(deps: _Deps):
    claim = _claim().model_copy(update={"scheduled": True})

    await task_run.start_task_run(USER, "task-1", claim=claim)

    deps.run.claim_task_run.assert_not_awaited()
    deps.schedule.assert_awaited_once()
    line = _activity(deps)
    assert (line.kind, line.text, line.actor_type) == (
        TaskActivityKind.STARTED,
        "Started at its scheduled time",
        TaskActor.SYSTEM,
    )


@pytest.mark.asyncio
async def test_a_task_started_at_its_time_queues_past_the_running_cap(deps: _Deps):
    deps.schedule.side_effect = ConcurrentTurnLimitError("busy", cap=5, used=5)
    claim = _claim().model_copy(update={"scheduled": True})

    await task_run.start_task_run(USER, "task-1", claim=claim)

    deps.enqueue.assert_awaited_once()
    deps.run.undo_task_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_of_a_task_waiting_for_its_start_starts_nothing(
    deps: _Deps, stops: SimpleNamespace
):
    deps.tasks.get_task = AsyncMock(
        return_value=_task(start_at=datetime(2026, 11, 15, 9, 0, tzinfo=UTC))
    )
    deps.tasks.update_task.side_effect = None
    deps.tasks.update_task.return_value = _task(status=TaskStatus.CANCELLED)

    task = await task_run.cancel_task_run(USER, "task-1")

    assert task.status == TaskStatus.CANCELLED
    deps.tasks.update_task.assert_awaited_once_with(
        "task-1", USER, TaskPatch(status=TaskStatus.CANCELLED)
    )
    stops.subtasks.assert_not_awaited()
    stops.cancel.assert_not_awaited()
    assert _activity(deps).text == "You cancelled it"


_DAILY = TaskRepeat(crons=["0 8 * * *"], label="Every day at 08:00")


@pytest.mark.asyncio
async def test_stopping_one_run_of_a_repeating_task_keeps_it_repeating(
    deps: _Deps, stops: SimpleNamespace, mocker: MockerFixture
):
    deps.tasks.get_task = AsyncMock(
        return_value=_task(
            status=TaskStatus.IN_PROGRESS, work_session_id="chat-w", repeat=_DAILY
        )
    )
    schedule = MagicMock()
    schedule.end_repeat_run = AsyncMock(
        return_value=_task(status=TaskStatus.TODO, repeat=_DAILY)
    )
    mocker.patch(f"{_MOD}.task_schedule_db", return_value=schedule)

    task = await task_run.cancel_task_run(USER, "task-1", this_run_only=True)

    assert task.status == TaskStatus.TODO and task.repeat == _DAILY
    schedule.end_repeat_run.assert_awaited_once_with("task-1", USER)
    deps.tasks.update_task.assert_not_awaited()
    stops.cancel.assert_awaited_once()
    entry = deps.record.add_activity.await_args.args[2]
    assert entry.text == "You stopped this run"
    assert entry.chat_session_id == "chat-w"
    assert entry.data == {"run_end": "cancelled"}


@pytest.mark.asyncio
async def test_cancelling_a_repeating_task_stops_it_repeating(
    deps: _Deps, stops: SimpleNamespace, mocker: MockerFixture
):
    deps.tasks.get_task = AsyncMock(
        return_value=_task(status=TaskStatus.TODO, repeat=_DAILY)
    )
    deps.tasks.update_task.side_effect = None
    deps.tasks.update_task.return_value = _task(
        status=TaskStatus.CANCELLED, repeat=_DAILY
    )
    schedule = MagicMock(clear_task_repeat=AsyncMock(return_value=True))
    mocker.patch(f"{_MOD}.task_schedule_db", return_value=schedule)

    await task_run.cancel_task_run(USER, "task-1")

    schedule.clear_task_repeat.assert_awaited_once_with("task-1", USER)


@pytest.mark.asyncio
async def test_a_run_waiting_for_a_free_slot_says_it_is_queued(deps: _Deps):
    """A5: queued past the running cap, the task must not read as Working."""
    deps.schedule.side_effect = ConcurrentTurnLimitError("busy", cap=5, used=5)
    deps.run.mark_run_queued = AsyncMock(
        return_value=_task(
            status=TaskStatus.IN_PROGRESS,
            status_text=QUEUED_TEXT,
            is_queued=True,
        )
    )

    task = await task_run.start_task_run(USER, "task-1")

    deps.run.mark_run_queued.assert_awaited_once_with("task-1", USER)
    assert task.is_queued and task.status_text == "Queued: starts when a slot frees"


@pytest.mark.asyncio
async def test_a_run_that_starts_at_once_is_not_queued(deps: _Deps):
    deps.run.mark_run_queued = AsyncMock()

    task = await task_run.start_task_run(USER, "task-1")

    deps.run.mark_run_queued.assert_not_awaited()
    assert not task.is_queued


@pytest.mark.asyncio
async def test_a_busy_work_chat_is_a_conflict_not_a_crash(deps: _Deps):
    """A8: a 409 with words for the user, not a bare RuntimeError."""
    deps.schedule.return_value = None

    with pytest.raises(ConflictError, match="busy"):
        await task_run.start_task_run(USER, "task-1")

    deps.run.undo_task_run.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TaskStatus.DONE, TaskStatus.CANCELLED])
async def test_marking_work_going_done_settles_it_then_stops_it(
    deps: _Deps, stops: SimpleNamespace, status
):
    """QA A2: settled first, so the stopped turn's end cannot flip it back."""
    order: list[str] = []
    deps.tasks.update_task.side_effect = lambda *a: order.append("settle") or _task(
        status=status, work_session_id="chat-w"
    )
    stops.cancel.side_effect = lambda *a: order.append("stop")

    task = await task_run.settle_by_hand(USER, "task-1", status)

    assert task.status == status
    deps.tasks.update_task.assert_awaited_once_with(
        "task-1", USER, TaskPatch(status=status)
    )
    assert order == ["settle", "stop"]
