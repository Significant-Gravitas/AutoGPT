"""The due-task sweep: start each due task once, or hold, move, miss or
cancel it, and never let one task stop the batch."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest
from prisma.enums import TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.copilot import task_due
from backend.copilot.tree import TreeRefusal
from backend.copilot.turn_queue import InflightCapExceeded, TurnRefusal
from backend.data import task_due as task_due_data
from backend.data.task import Task, TaskPatch
from backend.data.task_due import DueTask
from backend.data.task_run import TaskRunClaim
from backend.data.task_schedule_meta import TaskRepeat

_MODULE = "backend.copilot.task_due"
_USER = "user-1"
_NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)
_DUE = _NOW - timedelta(seconds=20)


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": _USER,
        "title": "Research AutoGPT",
        "description": "Find the latest release notes",
        "status": TaskStatus.TODO,
        "created_by": TaskCreator.OTTO,
        "created_at": _NOW,
        "updated_at": _NOW,
        "start_at": _DUE,
    }
    return Task(**(fields | overrides))


def _item(**overrides) -> DueTask:
    task = overrides.pop("task", _task())
    fields = {
        "task": task,
        "start_at": _DUE,
        "due_at": _DUE,
        "origin_session_id": "chat-1",
    }
    return DueTask(**(fields | overrides))


def _claim(item: DueTask) -> TaskRunClaim:
    return TaskRunClaim(
        before=item.task,
        task=item.task.model_copy(
            update={"status": TaskStatus.IN_PROGRESS, "start_at": None}
        ),
        retry=False,
        scheduled=True,
    )


def _chat(**overrides) -> MagicMock:
    chat = MagicMock()
    chat.session_id = "chat-1"
    chat.expert_id = overrides.get("expert_id")
    chat.archived_at = overrides.get("archived_at")
    chat.metadata.llm_auth_provider = "platform"
    chat.metadata.builder_graph_id = None
    return chat


@pytest.fixture
def db(mocker: MockerFixture) -> MagicMock:
    """Every store the sweep reaches, as one mock."""
    store = MagicMock()
    store.list_due_tasks = AsyncMock(return_value=[])
    store.claim_due_task = AsyncMock(side_effect=lambda *a: _claim(_item()))
    store.mark_task_missed = AsyncMock(return_value=True)
    store.hold_due_task = AsyncMock(return_value=True)
    store.cancel_scheduled_tasks = AsyncMock(return_value=1)
    store.undo_task_run = AsyncMock()
    store.update_task = AsyncMock()
    store.add_activity = AsyncMock()
    store.get_expert = AsyncMock()
    store.expert_row_exists = AsyncMock(return_value=False)
    for accessor in (
        "task_due_db",
        "task_db",
        "task_record_db",
        "task_run_db",
        "task_schedule_db",
        "experts_db",
    ):
        mocker.patch(f"{_MODULE}.{accessor}", return_value=store)
    return store


@pytest.fixture
def chat(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{_MODULE}.get_chat_session", new_callable=AsyncMock, return_value=_chat()
    )


@pytest.fixture
def refusal(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{_MODULE}.turn_refusal", new_callable=AsyncMock, return_value=None
    )


@pytest.fixture
def launch(mocker: MockerFixture) -> AsyncMock:
    background = MagicMock(child_session_id="worker-chat")
    return mocker.patch(
        f"{_MODULE}.launch_background_subtask",
        new_callable=AsyncMock,
        return_value=background,
    )


@pytest.fixture
def run(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(f"{_MODULE}.start_task_run", new_callable=AsyncMock)


@pytest.fixture
def link(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(f"{_MODULE}.link_worker", new_callable=AsyncMock)


def _lines(db: MagicMock) -> list[str]:
    return [call.args[2].text for call in db.add_activity.await_args_list]


async def test_a_due_task_starts_under_the_chat_that_asked(
    db, chat, refusal, launch, run, link
):
    await task_due.start_due_task(_item(), _NOW)

    db.claim_due_task.assert_awaited_once_with("task-1", _USER, _DUE)
    user_id, parent, prompt = launch.await_args.args
    assert (user_id, parent.session_id) == (_USER, "chat-1")
    assert "Find the latest release notes" in prompt and "task-1" in prompt
    # Its own launching turn, named after the task and its start, so a retry
    # of the same start finds the same work instead of starting it twice.
    envelope = launch.await_args.kwargs["envelope"]
    assert envelope.tree_id == launch.await_args.kwargs["turn_id"]
    assert (
        launch.await_args.kwargs["turn_id"] == f"task-start:task-1:{_DUE.isoformat()}"
    )
    assert envelope.spend_session_id == "chat-1"
    link.assert_awaited_once()
    assert link.await_args.args[1:] == ("worker-chat", _USER)
    assert _lines(db) == ["Started at its scheduled time"]
    run.assert_not_awaited()


async def test_two_passes_at_once_start_the_task_once(
    db, chat, refusal, launch, run, link
):
    """Both passes read the task as due before either claims it; the claim is
    conditional on the start time, so only one wins."""
    both_read = asyncio.Barrier(2)

    async def checked(*_: object) -> None:
        await asyncio.wait_for(both_read.wait(), timeout=5)

    refusal.side_effect = checked
    state = {"start_at": _DUE}

    async def claim(task_id: str, user_id: str, start_at: datetime):
        if state["start_at"] != start_at:
            return None
        state["start_at"] = None
        return _claim(_item())

    db.claim_due_task.side_effect = claim

    await asyncio.gather(
        task_due.start_due_task(_item(), _NOW),
        task_due.start_due_task(_item(), _NOW),
    )

    assert db.claim_due_task.await_count == 2
    launch.assert_awaited_once()


async def test_a_task_started_by_hand_meanwhile_is_skipped(
    db, chat, refusal, launch, run
):
    db.claim_due_task.side_effect = None
    db.claim_due_task.return_value = None

    await task_due.start_due_task(_item(), _NOW)

    launch.assert_not_awaited()
    run.assert_not_awaited()


async def test_a_task_hours_late_is_marked_missed_and_not_run(
    db, chat, refusal, launch, run
):
    late = _item(start_at=_NOW - timedelta(hours=7), due_at=_NOW - timedelta(hours=7))

    await task_due.start_due_task(late, _NOW)

    db.mark_task_missed.assert_awaited_once_with(
        "task-1", _USER, late.start_at, "Missed its start time"
    )
    assert _lines(db) == ["Missed its start time"]
    db.claim_due_task.assert_not_awaited()
    launch.assert_not_awaited()


async def test_a_hold_keeps_the_first_due_time_for_the_lateness_rule(
    db, chat, refusal, launch
):
    """Held for hours past its first due time, it is missed, not run."""
    held = _item(start_at=_NOW - timedelta(minutes=1), due_at=_NOW - timedelta(hours=7))

    await task_due.start_due_task(held, _NOW)

    db.mark_task_missed.assert_awaited_once()
    launch.assert_not_awaited()


async def test_without_a_plan_it_stays_scheduled_and_says_why(
    db, chat, refusal, launch, run
):
    refusal.return_value = TurnRefusal(reason="paywalled", kind="no_plan")

    await task_due.start_due_task(_item(), _NOW)

    db.hold_due_task.assert_awaited_once_with(
        "task-1",
        _USER,
        _DUE,
        _NOW + task_due.HOLD_AFTER,
        "Waiting to start: your account needs a plan",
        _DUE,
    )
    db.claim_due_task.assert_not_awaited()
    launch.assert_not_awaited()


async def test_a_used_up_daily_allowance_moves_it_to_the_reset(
    db, chat, refusal, launch, mocker: MockerFixture
):
    tracked = mocker.patch(f"{_MODULE}.track_turn_refusal")
    resets_at = _NOW + timedelta(hours=5)
    refusal.return_value = TurnRefusal(
        reason="rate-limited", kind="usage", window="daily", resets_at=resets_at
    )

    await task_due.start_due_task(_item(), _NOW)

    task_id, user_id, expected, at, text, due_at = db.hold_due_task.await_args.args
    assert resets_at <= at < resets_at + timedelta(seconds=300)
    # The reset is its new due time, so it is not missed for waiting on it.
    assert due_at == at
    assert text == "Moved to when your daily usage resets"
    tracked.assert_called_once()
    launch.assert_not_awaited()


async def test_a_closed_account_has_its_scheduled_tasks_cancelled_quietly(
    db, chat, refusal, launch
):
    refusal.return_value = TurnRefusal(reason="is banned or closed", kind="banned")

    await task_due.start_due_task(_item(), _NOW)

    db.cancel_scheduled_tasks.assert_awaited_once_with(_USER)
    assert _lines(db) == []
    launch.assert_not_awaited()


@pytest.mark.parametrize(
    "found", [None, _chat(archived_at=_NOW)], ids=["deleted", "archived"]
)
async def test_without_the_chat_that_asked_it_runs_in_a_chat_of_its_own(
    db, chat, refusal, launch, run, found
):
    chat.return_value = found

    await task_due.start_due_task(_item(), _NOW)

    launch.assert_not_awaited()
    run.assert_awaited_once()
    assert run.await_args.args == (_USER, "task-1")
    assert run.await_args.kwargs["claim"].scheduled


async def test_background_work_refused_falls_back_to_a_chat_of_its_own(
    db, chat, refusal, launch, run, link
):
    launch.side_effect = TreeRefusal("Background subtasks are not enabled.")

    await task_due.start_due_task(_item(), _NOW)

    db.undo_task_run.assert_awaited_once()
    assert db.claim_due_task.await_count == 2
    run.assert_awaited_once()
    link.assert_not_awaited()


async def test_at_the_hard_cap_it_waits_for_a_slot(db, chat, refusal, run):
    chat.return_value = None
    run.side_effect = InflightCapExceeded(used=15, cap=15)

    await task_due.start_due_task(_item(), _NOW)

    db.hold_due_task.assert_awaited_once_with(
        "task-1",
        _USER,
        _DUE,
        _NOW + task_due.SLOT_WAIT,
        "Waiting for a free slot to start",
        _DUE,
    )


async def test_an_expert_no_longer_on_the_team_hands_it_to_otto(
    db, chat, refusal, launch, run
):
    item = _item(task=_task(owner_expert_id="expert-gone"))
    chat.return_value = _chat(expert_id="expert-gone")
    db.get_expert.return_value = None

    await task_due.start_due_task(item, _NOW)

    db.update_task.assert_awaited_once_with(
        "task-1", _USER, TaskPatch(assign_to_otto=True)
    )
    assert "Handed to Otto" in _lines(db)[0]
    # The chat that asked was the expert's own, so the work runs as Otto's.
    launch.assert_not_awaited()
    run.assert_awaited_once()


async def test_a_paused_expert_holds_the_task(db, chat, refusal, launch):
    item = _item(task=_task(owner_expert_id="expert-1"))
    db.get_expert.return_value = MagicMock(is_archived=False, schedules_paused_at=_NOW)

    await task_due.start_due_task(item, _NOW)

    assert (
        db.hold_due_task.await_args.args[4] == "Waiting to start: its expert is paused"
    )
    db.claim_due_task.assert_not_awaited()
    launch.assert_not_awaited()


async def test_one_task_failing_does_not_stop_the_batch(
    db, chat, refusal, launch, link
):
    broken = _item(task=_task(id="task-broken"), origin_session_id="chat-broken")
    fine = _item()
    db.list_due_tasks.return_value = [broken, fine]

    async def load(session_id: str, user_id: str):
        if session_id == "chat-broken":
            raise RuntimeError("cache down")
        return _chat()

    chat.side_effect = load

    await task_due.start_due_tasks(_NOW)

    # The broken one is put back to try again shortly; the next one starts.
    db.hold_due_task.assert_awaited_once_with(
        "task-broken", _USER, _DUE, _NOW + task_due.RETRY_AFTER, None, _DUE
    )
    launch.assert_awaited_once()


async def test_a_failed_launch_puts_the_claimed_task_back(
    db, chat, refusal, launch, link
):
    db.list_due_tasks.return_value = [_item()]
    launch.side_effect = RuntimeError("redis down")

    await task_due.start_due_tasks(_NOW)

    db.undo_task_run.assert_awaited_once()
    assert db.hold_due_task.await_args.args[3] == _NOW + task_due.RETRY_AFTER


async def test_nothing_due_is_one_query(db, chat, refusal):
    await task_due.start_due_tasks(_NOW)

    db.list_due_tasks.assert_awaited_once_with(_NOW, task_due.DUE_BATCH_SIZE)
    refusal.assert_not_awaited()
    chat.assert_not_awaited()


# ---------------------------------------------------------------- repeating

_EVERY_3_MIN = TaskRepeat(crons=["*/3 * * * *"], label="Every 3 minutes")


async def test_a_repeating_task_hours_late_skips_to_its_next_run(
    db, chat, refusal, launch, run
):
    late_at = _NOW - timedelta(hours=7)
    late = _item(start_at=late_at, due_at=late_at, repeat=_EVERY_3_MIN)

    await task_due.start_due_task(late, _NOW)

    db.mark_task_missed.assert_not_awaited()
    next_run = datetime(2026, 11, 15, 12, 3, tzinfo=UTC)
    db.hold_due_task.assert_awaited_once_with(
        "task-1", _USER, late_at, next_run, None, next_run
    )
    assert _lines(db) == ["Skipped a run: it could not start in time"]
    launch.assert_not_awaited()


async def test_each_run_of_a_repeating_task_is_its_own_work(
    db, chat, refusal, launch, run, link
):
    """Run now and the scheduled run it moved share a start time; the claim
    time keeps their background work apart."""
    item = _item(repeat=_EVERY_3_MIN)
    claimed = _claim(item)
    first = claimed.model_copy(
        update={"task": claimed.task.model_copy(update={"started_at": _NOW})}
    )
    second = claimed.model_copy(
        update={
            "task": claimed.task.model_copy(
                update={"started_at": _NOW + timedelta(minutes=5)}
            )
        }
    )
    db.claim_due_task.side_effect = [first, second]

    await task_due.start_due_task(item, _NOW)
    await task_due.start_due_task(item, _NOW)

    turn_ids = [call.kwargs["turn_id"] for call in launch.await_args_list]
    assert len(set(turn_ids)) == 2
    assert all(t.startswith(f"task-start:task-1:{_DUE.isoformat()}:") for t in turn_ids)


async def test_the_sweep_only_picks_tasks_waiting_so_a_run_never_overlaps(
    mocker: MockerFixture,
):
    """A repeating task whose run is still going is IN_PROGRESS with no start
    time, so the due query, which asks for TODO tasks whose time came, skips
    that tick; the run's end sets the next one."""
    client = MagicMock(find_many=AsyncMock(return_value=[]))
    mocker.patch.object(prisma.models.Task, "prisma", return_value=client)

    await task_due_data.list_due_tasks(_NOW, 20)

    where = client.find_many.await_args.kwargs["where"]
    assert where["status"] == TaskStatus.TODO
    assert where["startAt"] == {"lte": _NOW}
