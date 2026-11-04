from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.tasks.models import (
    Ask,
    Assignee,
    AssigneeType,
    Task,
    TaskPriority,
    TaskStatus,
)

from . import hire_task

_NOW = datetime(2026, 11, 3, tzinfo=UTC)


class FakeStore:
    """The slice of the tasks store a hire touches, in memory."""

    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.attachments: dict[str, list] = {}

    async def create_task(self, user_id, data, actor):
        number = len(self.tasks) + 1
        task = Task(
            id=f"task-{number}",
            number=number,
            key=f"T-{number}",
            priority=TaskPriority.NONE,
            title=data.title,
            description=data.description,
            status=data.status or TaskStatus.TODO,
            assignee=Assignee(type=data.assignee_type, name="Otto"),
            created_at=_NOW,
            updated_at=_NOW,
        )
        self.tasks[task.id] = task
        return task

    async def add_attachment(self, user_id, task_id, data, actor):
        self.attachments.setdefault(task_id, []).append(data)

    async def get_task(self, user_id, task_id):
        return self.tasks.get(task_id)

    async def get_task_for_work_session(self, session_id):
        return None

    async def update_task(self, user_id, task_id, data, actor):
        return self._set(task_id, **data.model_dump(exclude_unset=True))

    async def set_task_status(
        self, user_id, task_id, status, actor, note=None, outcome_summary=None
    ):
        return self._set(task_id, status=status, outcome_summary=outcome_summary)

    async def set_live(self, task_id, running, current_step=None):
        task = self.tasks[task_id]
        live = task.live.model_copy(update={"running": running})
        return self._set(task_id, live=live)

    async def contract_add_approval(self, user_id, task_id, card, actor):
        kept = [a for a in self.tasks[task_id].approvals if a.name != card.name]
        task = self._set(task_id, approvals=[*kept, card], status=TaskStatus.BLOCKED)
        return SimpleNamespace(task=task, contract=task)

    async def contract_ask(self, user_id, task_id, question, actor, why=None):
        ask = Ask(id="ask-1", question=question, why=why)
        task = self._set(task_id, ask=ask, status=TaskStatus.BLOCKED)
        return SimpleNamespace(task=task, contract=task)

    async def contract_drop_approvals(self, user_id, task_id, ids):
        approvals = [
            a for a in self.tasks[task_id].approvals if a.confirmation_id not in ids
        ]
        task = self._set(task_id, approvals=approvals)
        return SimpleNamespace(task=task, contract=task)

    def _set(self, task_id, **fields):
        task = self.tasks[task_id].model_copy(update=fields)
        self.tasks[task_id] = task
        return task


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value


@pytest.fixture
def board():
    store, redis = FakeStore(), FakeRedis()
    with (
        patch.object(hire_task, "tasks_store", return_value=store),
        patch.object(hire_task, "get_redis_async", AsyncMock(return_value=redis)),
        patch.object(hire_task, "is_feature_enabled", AsyncMock(return_value=True)),
        patch.object(hire_task, "find_work_task", AsyncMock(return_value=None)),
        patch.object(hire_task, "caller_actor", AsyncMock()),
        patch.object(hire_task, "remember_hire_chat", AsyncMock()),
        patch.object(hire_task, "publish_contract_update", AsyncMock()),
        patch.object(hire_task, "publish_task_event", AsyncMock()),
    ):
        yield store


def _session(session_id: str = "chat-1"):
    return SimpleNamespace(session_id=session_id, expert_id=None)


async def _propose(name: str, role: str, confirmation_id: str, chat: str = "chat-1"):
    """What hire_expert does: track the hire, then put its card on the task."""
    task = await hire_task.track_proposed_hire("u", _session(chat), name, role)
    await hire_task.wait_on_hire_card(
        "u", _session(chat), task, confirmation_id, "hire", name, role
    )
    return task


@pytest.mark.asyncio
async def test_a_proposed_hire_goes_on_the_board_as_its_own_task(board):
    task = await hire_task.track_proposed_hire("u", _session(), "Maya", "Editor")

    assert task is not None
    assert task.title == "Hire Maya"
    assert task.description == "Hire Maya (Editor)"
    assert task.assignee is not None
    assert task.assignee.type == AssigneeType.OTTO
    note = hire_task.tracked_note(task)
    assert "T-1" in note
    assert f"task_id {task.id} and status cancelled" in note


@pytest.mark.asyncio
async def test_a_hire_task_links_the_chat_with_its_card(board):
    task = await hire_task.track_proposed_hire("u", _session(), "Maya", "Editor")

    assert task is not None
    [link] = board.attachments[task.id]
    assert (link.kind, link.ref_id) == ("chat", "chat-1")
    hire_task.remember_hire_chat.assert_awaited_once_with(task.id, "chat-1")


@pytest.mark.asyncio
async def test_a_waiting_hire_says_what_it_waits_for(board):
    task = await _propose("Sam", "Support", "c-sam")
    assert task is not None

    waiting = board.tasks[task.id]
    assert waiting.status == TaskStatus.BLOCKED
    assert waiting.ask is not None
    assert waiting.ask.question == "Waiting for your OK on Sam's hire card"
    assert [a.confirmation_id for a in waiting.approvals] == ["c-sam"]


@pytest.mark.asyncio
async def test_each_hire_in_one_chat_is_its_own_task(board):
    tess = await _propose("Tess", "Support", "c-tess")
    sam = await _propose("Sam", "Support", "c-sam")

    assert tess is not None and sam is not None
    assert tess.id != sam.id
    assert [t.title for t in board.tasks.values()] == ["Hire Tess", "Hire Sam"]


@pytest.mark.asyncio
async def test_proposing_the_same_hire_again_keeps_its_task(board):
    first = await _propose("Maya", "Editor", "c-1")
    again = await _propose("Maya", "Senior Editor", "c-2")

    assert first is not None and again is not None
    assert again.id == first.id
    assert len(board.tasks) == 1
    task = board.tasks[first.id]
    assert task.description == "Hire Maya (Senior Editor)"
    assert [a.confirmation_id for a in task.approvals] == ["c-2"]


@pytest.mark.asyncio
async def test_confirming_a_hire_completes_only_its_own_task(board):
    tess = await _propose("Tess", "Support", "c-tess")
    sam = await _propose("Sam", "Support", "c-sam")
    assert tess is not None and sam is not None

    await hire_task.forget_card("u", "chat-1", ["c-tess"], tess.id)
    await hire_task.complete_hire("u", tess.id, "Tess", "Support")

    done, waiting = board.tasks[tess.id], board.tasks[sam.id]
    assert done.status == TaskStatus.DONE
    assert done.outcome_summary == "Hired Tess (Support)"
    assert done.approvals == []
    assert waiting.status == TaskStatus.BLOCKED
    assert waiting.ask is not None and "Sam" in waiting.ask.question


@pytest.mark.asyncio
async def test_a_hire_approved_on_the_board_stops_hiring_when_done(board):
    task = await _propose("Sam", "Support", "c-sam")
    assert task is not None
    await board.set_live(task.id, True)

    await hire_task.complete_hire("u", task.id, "Sam", "Support")

    assert board.tasks[task.id].live.running is False
    assert board.tasks[task.id].status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_a_hire_after_the_task_is_done_starts_a_new_task(board):
    task = await hire_task.track_proposed_hire("u", _session(), "Maya", "Editor")
    assert task is not None
    await hire_task.complete_hire("u", task.id, "Maya", "Editor")

    later = await hire_task.track_proposed_hire("u", _session(), "Maya", "Editor")

    assert later is not None and later.id != task.id


@pytest.mark.asyncio
async def test_a_chat_working_a_task_adds_no_hire_task(board):
    with patch.object(hire_task, "find_work_task", AsyncMock(return_value=object())):
        task = await hire_task.track_proposed_hire("u", _session(), "Maya", "Editor")
    await hire_task.wait_on_hire_card(
        "u", _session(), task, "c-maya", "hire", "Maya", "Editor"
    )

    assert task is None
    assert board.tasks == {}
    assert hire_task.tracked_note(task) == ""


@pytest.mark.asyncio
async def test_accounts_without_the_board_get_no_hire_task(board):
    with patch.object(hire_task, "is_feature_enabled", AsyncMock(return_value=False)):
        task = await hire_task.track_proposed_hire("u", _session(), "Maya", "Editor")

    assert task is None
    assert board.tasks == {}


@pytest.mark.asyncio
async def test_a_board_failure_never_blocks_the_hire(board):
    with patch.object(
        hire_task, "find_work_task", AsyncMock(side_effect=RuntimeError("db down"))
    ):
        task = await hire_task.track_proposed_hire("u", _session(), "Maya", "Editor")

    assert task is None


class CardStore:
    """A task's card list, the way contract_db keeps it."""

    def __init__(self, task: Task) -> None:
        self.task = task
        self.added: list = []
        self.dropped: list[list[str]] = []

    async def get_task_for_work_session(self, session_id):
        return ("user-1", self.task) if session_id == "work-chat" else None

    async def contract_add_approval(self, user_id, task_id, card, actor):
        self.added.append(card)
        self.task = self.task.model_copy(update={"approvals": [card]})
        return SimpleNamespace(task=self.task, status_event="task_needs_input")

    async def contract_drop_approvals(self, user_id, task_id, ids):
        self.dropped.append(ids)
        return SimpleNamespace(task=self.task, status_event=None)


def _work_task() -> Task:
    return Task(
        id="task-9",
        number=9,
        key="T-9",
        title="Raise two experts",
        priority=TaskPriority.NONE,
        status=TaskStatus.IN_PROGRESS,
        work_session_id="work-chat",
        created_at=_NOW,
        updated_at=_NOW,
    )


@pytest.mark.asyncio
async def test_a_card_proposed_in_a_task_chat_waits_on_that_task():
    store = CardStore(_work_task())
    session = SimpleNamespace(session_id="work-chat", expert_id=None)
    with (
        patch.object(hire_task, "tasks_store", return_value=store),
        patch.object(hire_task, "is_feature_enabled", AsyncMock(return_value=True)),
        patch.object(hire_task, "find_work_task", AsyncMock(return_value=store.task)),
        patch.object(hire_task, "caller_actor", AsyncMock()),
        patch.object(hire_task, "publish_contract_update", AsyncMock()),
        patch.object(hire_task, "publish_task_event", AsyncMock()),
    ):
        task = await hire_task.park_on_work_task(
            "user-1", session, "c-devin", "hire", "Devin", "Web Development"
        )

    assert task is not None and task.key == "T-9"
    assert [c.confirmation_id for c in store.added] == ["c-devin"]
    assert "T-9" in hire_task.parked_note(task)


@pytest.mark.asyncio
async def test_a_chat_with_no_task_keeps_no_card():
    store = CardStore(_work_task())
    session = SimpleNamespace(session_id="plain-chat", expert_id=None)
    with (
        patch.object(hire_task, "tasks_store", return_value=store),
        patch.object(hire_task, "find_work_task", AsyncMock(return_value=None)),
    ):
        task = await hire_task.park_on_work_task(
            "user-1", session, "c-devin", "hire", "Devin", ""
        )

    assert task is None
    assert store.added == []


@pytest.mark.asyncio
async def test_a_confirmed_hire_leaves_its_task_whichever_id_was_used():
    store = CardStore(_work_task())
    card = hire_task.CardApproval(confirmation_id="c-start", name="Devin")
    store.task = store.task.model_copy(update={"approvals": [card]})
    with (
        patch.object(hire_task, "tasks_store", return_value=store),
        patch.object(hire_task, "publish_contract_update", AsyncMock()),
        patch.object(hire_task, "publish_task_event", AsyncMock()),
    ):
        await hire_task.forget_card("user-1", "work-chat", ["c-plain", "c-start"])

    assert store.dropped == [["c-plain", "c-start"]]
