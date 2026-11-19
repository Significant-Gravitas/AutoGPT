"""The end of a turn in a task's run chat settles the task."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.copilot.background_task import BACKGROUND_TIMEOUT_MESSAGE
from backend.copilot.executor.utils import CoPilotExecutionEntry
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.copilot.task_run_settle import (
    last_reply,
    settle_task_run_turn,
    turn_ending,
)

_MOD = "backend.copilot.task_run_settle"


def _entry(user_id: str | None = "user-1") -> CoPilotExecutionEntry:
    return CoPilotExecutionEntry(
        session_id="chat-w", turn_id="turn-1", user_id=user_id, message="go"
    )


@pytest.fixture
def run_db(mocker: MockerFixture) -> MagicMock:
    run = MagicMock()
    run.settle_task_run = AsyncMock()
    mocker.patch(f"{_MOD}.task_run_db", return_value=run)
    return run


@pytest.fixture
def handoffs(mocker: MockerFixture) -> AsyncMock:
    background = MagicMock()
    background.has_open_handoffs = AsyncMock(return_value=False)
    mocker.patch(f"{_MOD}.background_task_db", return_value=background)
    return background.has_open_handoffs


def _chat(
    mocker: MockerFixture,
    work_task_id: str | None,
    messages: list[ChatMessage] | None = None,
) -> None:
    session = ChatSession.new("user-1", dry_run=False, work_task_id=work_task_id)
    session.messages = messages or []
    mocker.patch(
        f"{_MOD}.get_chat_session", new_callable=AsyncMock, return_value=session
    )


@pytest.mark.parametrize(
    "error,ending",
    [
        (None, "completed"),
        (CANCELLED_MESSAGE, "cancelled"),
        (BACKGROUND_TIMEOUT_MESSAGE, "timed_out"),
        ("boom", "failed"),
    ],
)
def test_turn_ending(error, ending):
    assert turn_ending(error) == ending


@pytest.mark.asyncio
async def test_the_end_of_a_run_turn_settles_its_task(
    mocker: MockerFixture, run_db: MagicMock, handoffs: AsyncMock
):
    _chat(mocker, "task-1")

    await settle_task_run_turn(_entry(), "boom")

    run_db.settle_task_run.assert_awaited_once_with(
        "task-1", "chat-w", "user-1", "failed", None
    )


@pytest.mark.asyncio
async def test_a_plain_chat_settles_nothing(
    mocker: MockerFixture, run_db: MagicMock, handoffs: AsyncMock
):
    _chat(mocker, None)

    await settle_task_run_turn(_entry(), None)

    run_db.settle_task_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_work_handed_to_background_subtasks_waits_for_their_wake(
    mocker: MockerFixture, run_db: MagicMock, handoffs: AsyncMock
):
    _chat(mocker, "task-1")
    handoffs.return_value = True

    await settle_task_run_turn(_entry(), None)
    run_db.settle_task_run.assert_not_awaited()

    # A stop is a stop, subtasks or not.
    await settle_task_run_turn(_entry(), CANCELLED_MESSAGE)
    run_db.settle_task_run.assert_awaited_once_with(
        "task-1", "chat-w", "user-1", "cancelled", None
    )


@pytest.mark.asyncio
async def test_a_settle_failure_never_fails_the_turn(
    mocker: MockerFixture, run_db: MagicMock, handoffs: AsyncMock
):
    _chat(mocker, "task-1")
    run_db.settle_task_run.side_effect = RuntimeError("db down")

    await settle_task_run_turn(_entry(), None)


@pytest.mark.asyncio
async def test_the_workers_last_words_go_to_the_settle(
    mocker: MockerFixture, run_db: MagicMock, handoffs: AsyncMock
):
    _chat(
        mocker,
        "task-1",
        [
            ChatMessage(role="user", content="Work on the task"),
            ChatMessage(role="assistant", content="Looking at Shopify."),
            ChatMessage(role="tool", content="{}", tool_call_id="c1"),
            ChatMessage(role="assistant", content="Shopify isn't connected."),
        ],
    )

    await settle_task_run_turn(_entry(), None)

    run_db.settle_task_run.assert_awaited_once_with(
        "task-1", "chat-w", "user-1", "completed", "Shopify isn't connected."
    )


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [ChatMessage(role="assistant", content="Old turn"), ChatMessage(role="user")],
        [ChatMessage(role="user"), ChatMessage(role="assistant", content="  ")],
    ],
)
def test_a_turn_that_said_nothing_has_no_last_reply(messages):
    session = ChatSession.new("user-1", dry_run=False)
    session.messages = messages

    assert last_reply(session) is None
