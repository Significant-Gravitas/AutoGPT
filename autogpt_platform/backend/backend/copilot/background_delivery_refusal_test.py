"""A closed or banned account's background subtasks don't carry on: the
recovery pass stops the child and suppresses the follow-up for good, so
keeping the account later doesn't wake the original turn."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import background_delivery

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"


def _task():
    return SimpleNamespace(
        id="task-1",
        user_id=USER_ID,
        child_session_id="child-1",
        wake_suppressed=False,
        outcome="cancelled",
        followup_state="queued",
    )


@pytest.fixture
def deps(monkeypatch):
    tasks = MagicMock()
    tasks.suppress_background_followup = AsyncMock()
    chats = MagicMock()
    chats.get_chat_session_status = AsyncMock(return_value="idle")
    monkeypatch.setattr(background_delivery, "background_task_db", lambda: tasks)
    monkeypatch.setattr(background_delivery, "chat_db", lambda: chats)
    stop = AsyncMock()
    monkeypatch.setattr(background_delivery, "_stop_child", stop)
    recover = AsyncMock()
    monkeypatch.setattr(background_delivery, "_recover_followup", recover)
    monkeypatch.setattr(background_delivery, "keep_tree_alive", AsyncMock())
    return SimpleNamespace(tasks=tasks, chats=chats, stop=stop, recover=recover)


@pytest.mark.asyncio
async def test_a_refused_accounts_followup_is_suppressed_not_recovered(
    monkeypatch, deps
):
    monkeypatch.setattr(
        background_delivery, "is_user_refused", AsyncMock(return_value=True)
    )
    task = _task()

    await background_delivery._reconcile(task)

    deps.stop.assert_awaited_once_with(task)
    deps.tasks.suppress_background_followup.assert_awaited_once_with("task-1", USER_ID)
    deps.recover.assert_not_awaited()
    deps.chats.get_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_open_accounts_queued_followup_is_still_recovered(monkeypatch, deps):
    monkeypatch.setattr(
        background_delivery, "is_user_refused", AsyncMock(return_value=False)
    )
    task = _task()

    await background_delivery._reconcile(task)

    deps.tasks.suppress_background_followup.assert_not_awaited()
    deps.recover.assert_awaited_once_with(task)
