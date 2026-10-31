"""A closing account's AutoPilot turns are cancelled and its chat links unlinked."""

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.autopilot_work import (
    cancel_autopilot_turns,
    unlink_chat_platforms,
)
from backend.platform_linking.models import PlatformLinkInfo, PlatformUserLinkInfo
from backend.util.exceptions import NotFoundError

MODULE = "backend.api.features.account_deletion.autopilot_work"
BEST_EFFORT_MODULE = "backend.api.features.account_deletion.best_effort"
USER_ID = "user-1"
LINKED_AT = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _chat(session_id: str) -> MagicMock:
    chat = MagicMock()
    chat.session_id = session_id
    return chat


def _server_link(link_id: str) -> PlatformLinkInfo:
    return PlatformLinkInfo(
        id=link_id,
        platform="DISCORD",
        platform_server_id=f"guild-{link_id}",
        owner_platform_user_id="discord-user",
        server_name="Team server",
        linked_at=LINKED_AT,
    )


def _dm_link(link_id: str) -> PlatformUserLinkInfo:
    return PlatformUserLinkInfo(
        id=link_id,
        platform="TELEGRAM",
        platform_user_id=f"tg-{link_id}",
        platform_username="someone",
        linked_at=LINKED_AT,
    )


@pytest.fixture(autouse=True)
def fast_polls(mocker: MockerFixture) -> None:
    mocker.patch(f"{MODULE}.TURN_STOP_POLL_SECONDS", 0.01)


@pytest.fixture
def list_queued(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{MODULE}.list_queued_sessions", new_callable=AsyncMock, return_value=[]
    )


@pytest.fixture
def dequeue(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{MODULE}.cancel_queued_turn", new_callable=AsyncMock, return_value=True
    )


@pytest.fixture
def list_running(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{MODULE}.get_running_session_ids", new_callable=AsyncMock, return_value=set()
    )


@pytest.fixture
def cancel(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(f"{MODULE}.enqueue_cancel_task", new_callable=AsyncMock)


@pytest.fixture
def links(mocker: MockerFixture) -> MagicMock:
    db = MagicMock()
    db.list_server_links = AsyncMock(return_value=[])
    db.list_user_links = AsyncMock(return_value=[])
    db.delete_server_link = AsyncMock()
    db.delete_user_link = AsyncMock()
    mocker.patch(f"{MODULE}.platform_linking_db", return_value=db)
    return db


def _errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


# ---- AutoPilot turns ----


@pytest.mark.asyncio
async def test_queued_turns_are_dequeued_with_the_users_id(
    list_queued: AsyncMock, dequeue: AsyncMock, list_running: AsyncMock
):
    list_queued.return_value = [_chat("chat-1"), _chat("chat-2")]

    await cancel_autopilot_turns(USER_ID)

    list_queued.assert_awaited_once_with(USER_ID)
    assert dequeue.await_args_list == [
        call(user_id=USER_ID, session_id="chat-1"),
        call(user_id=USER_ID, session_id="chat-2"),
    ]


@pytest.mark.asyncio
async def test_running_turns_are_cancelled_and_waited_for(
    list_queued: AsyncMock,
    dequeue: AsyncMock,
    list_running: AsyncMock,
    cancel: AsyncMock,
):
    list_running.side_effect = [{"chat-1", "chat-2"}, {"chat-2"}, set()]

    await cancel_autopilot_turns(USER_ID)

    assert list_running.await_args_list == [call(USER_ID)] * 3
    assert sorted(c.args[0] for c in cancel.await_args_list) == ["chat-1", "chat-2"]


@pytest.mark.asyncio
async def test_queued_turns_are_dequeued_before_running_ones_are_cancelled(
    list_queued: AsyncMock,
    dequeue: AsyncMock,
    list_running: AsyncMock,
    cancel: AsyncMock,
):
    """A running turn's end promotes the user's oldest queued turn."""
    list_queued.return_value = [_chat("chat-queued")]
    list_running.side_effect = [{"chat-running"}, set()]
    order = MagicMock()
    order.attach_mock(list_queued, "list_queued")
    order.attach_mock(dequeue, "dequeue")
    order.attach_mock(list_running, "list_running")
    order.attach_mock(cancel, "cancel")

    await cancel_autopilot_turns(USER_ID)

    assert [c[0] for c in order.mock_calls] == [
        "list_queued",
        "dequeue",
        "list_running",
        "cancel",
        "list_running",
    ]


@pytest.mark.asyncio
async def test_a_turn_that_starts_during_the_wait_is_cancelled_too(
    list_queued: AsyncMock,
    dequeue: AsyncMock,
    list_running: AsyncMock,
    cancel: AsyncMock,
):
    list_running.side_effect = [{"chat-1"}, {"chat-1", "chat-2"}, set()]

    await cancel_autopilot_turns(USER_ID)

    assert cancel.await_args_list == [call("chat-1"), call("chat-2")]


@pytest.mark.asyncio
async def test_a_turn_is_cancelled_once_while_it_winds_down(
    list_queued: AsyncMock,
    dequeue: AsyncMock,
    list_running: AsyncMock,
    cancel: AsyncMock,
):
    list_running.side_effect = [{"chat-1"}, {"chat-1"}, {"chat-1"}, set()]

    await cancel_autopilot_turns(USER_ID)

    cancel.assert_awaited_once_with("chat-1")


@pytest.mark.asyncio
async def test_no_running_turns_means_no_cancel_and_no_wait(
    list_queued: AsyncMock,
    dequeue: AsyncMock,
    list_running: AsyncMock,
    cancel: AsyncMock,
):
    await cancel_autopilot_turns(USER_ID)

    list_running.assert_awaited_once_with(USER_ID)
    cancel.assert_not_awaited()
    dequeue.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_turn_that_does_not_end_in_time_is_logged_and_left(
    mocker: MockerFixture,
    list_queued: AsyncMock,
    dequeue: AsyncMock,
    list_running: AsyncMock,
    cancel: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    mocker.patch(f"{MODULE}.TURN_STOP_WAIT_SECONDS", 0.05)
    list_running.return_value = {"chat-1"}
    caplog.set_level(logging.INFO, logger=MODULE)

    await asyncio.wait_for(cancel_autopilot_turns(USER_ID), timeout=2)

    cancel.assert_awaited_once_with("chat-1")
    assert _errors(caplog) == []
    assert any(USER_ID in w and "could not confirm" in w for w in _warnings(caplog))


@pytest.mark.asyncio
async def test_one_failing_cancel_is_logged_and_the_others_still_cancel(
    list_queued: AsyncMock,
    dequeue: AsyncMock,
    list_running: AsyncMock,
    cancel: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    list_running.side_effect = [{"chat-1", "chat-2"}, set()]

    async def publish(session_id: str) -> None:
        if session_id == "chat-1":
            raise RuntimeError("broker down")

    cancel.side_effect = publish
    caplog.set_level(logging.INFO, logger=MODULE)

    await cancel_autopilot_turns(USER_ID)

    assert cancel.await_count == 2
    [error] = _errors(caplog)
    assert "chat-1" in error and USER_ID in error and "broker down" in error


@pytest.mark.asyncio
async def test_one_failing_dequeue_does_not_stop_the_others(
    list_queued: AsyncMock,
    dequeue: AsyncMock,
    list_running: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    list_queued.return_value = [_chat("chat-1"), _chat("chat-2")]
    dequeue.side_effect = [RuntimeError("db down"), True]
    caplog.set_level(logging.INFO, logger=MODULE)

    await cancel_autopilot_turns(USER_ID)

    assert dequeue.await_count == 2
    [error] = _errors(caplog)
    assert "chat-1" in error and USER_ID in error


@pytest.mark.asyncio
async def test_a_failing_queue_listing_still_cancels_running_turns(
    list_queued: AsyncMock,
    dequeue: AsyncMock,
    list_running: AsyncMock,
    cancel: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    list_queued.side_effect = RuntimeError("db down")
    list_running.side_effect = [{"chat-1"}, set()]
    caplog.set_level(logging.INFO, logger=MODULE)

    await cancel_autopilot_turns(USER_ID)

    cancel.assert_awaited_once_with("chat-1")
    [error] = _errors(caplog)
    assert "queued" in error and USER_ID in error


@pytest.mark.asyncio
async def test_a_hung_queue_listing_is_cut_off(
    mocker: MockerFixture,
    list_queued: AsyncMock,
    dequeue: AsyncMock,
    list_running: AsyncMock,
    cancel: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    mocker.patch(f"{BEST_EFFORT_MODULE}.STEP_TIMEOUT_SECONDS", 0.05)

    async def hang(_: str) -> list[object]:
        await asyncio.sleep(10)
        return []

    list_queued.side_effect = hang
    list_running.side_effect = [{"chat-1"}, set()]
    caplog.set_level(logging.INFO, logger=MODULE)

    await asyncio.wait_for(cancel_autopilot_turns(USER_ID), timeout=2)

    cancel.assert_awaited_once_with("chat-1")
    [error] = _errors(caplog)
    assert "queued" in error and USER_ID in error


@pytest.mark.asyncio
async def test_nothing_raises_when_every_turn_step_fails(
    list_queued: AsyncMock,
    list_running: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    list_queued.side_effect = RuntimeError("db down")
    list_running.side_effect = RuntimeError("db down")
    caplog.set_level(logging.INFO, logger=MODULE)

    await cancel_autopilot_turns(USER_ID)

    errors = _errors(caplog)
    assert len(errors) == 2
    assert all(USER_ID in error for error in errors)


@pytest.mark.asyncio
async def test_every_turn_step_hung_is_still_bounded(
    mocker: MockerFixture,
    list_queued: AsyncMock,
    list_running: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    mocker.patch(f"{BEST_EFFORT_MODULE}.STEP_TIMEOUT_SECONDS", 0.05)
    mocker.patch(f"{MODULE}.TURN_STOP_WAIT_SECONDS", 0.05)

    async def hang(_: str) -> object:
        await asyncio.sleep(10)

    list_queued.side_effect = hang
    list_running.side_effect = hang
    caplog.set_level(logging.INFO, logger=MODULE)

    await asyncio.wait_for(cancel_autopilot_turns(USER_ID), timeout=2)

    assert _errors(caplog) and _warnings(caplog)


# ---- Chat platform links ----


@pytest.mark.asyncio
async def test_every_server_and_dm_link_is_unlinked_with_the_users_id(
    links: MagicMock,
):
    links.list_server_links.return_value = [_server_link("s1"), _server_link("s2")]
    links.list_user_links.return_value = [_dm_link("d1")]

    await unlink_chat_platforms(USER_ID)

    links.list_server_links.assert_awaited_once_with(USER_ID)
    links.list_user_links.assert_awaited_once_with(USER_ID)
    assert links.delete_server_link.await_args_list == [
        call("s1", USER_ID),
        call("s2", USER_ID),
    ]
    links.delete_user_link.assert_awaited_once_with("d1", USER_ID)


@pytest.mark.asyncio
async def test_one_failing_unlink_is_logged_and_the_others_still_unlink(
    links: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    links.list_server_links.return_value = [_server_link("s1"), _server_link("s2")]
    links.list_user_links.return_value = [_dm_link("d1")]
    links.delete_server_link.side_effect = [RuntimeError("db down"), None]
    caplog.set_level(logging.INFO, logger=MODULE)

    await unlink_chat_platforms(USER_ID)

    assert links.delete_server_link.await_count == 2
    links.delete_user_link.assert_awaited_once()
    [error] = _errors(caplog)
    assert "s1" in error and "DISCORD" in error and USER_ID in error


@pytest.mark.asyncio
async def test_a_link_already_gone_is_not_an_error(
    links: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    links.list_user_links.return_value = [_dm_link("d1")]
    links.delete_user_link.side_effect = NotFoundError("Link not found.")
    caplog.set_level(logging.INFO, logger=BEST_EFFORT_MODULE)

    await unlink_chat_platforms(USER_ID)

    assert _errors(caplog) == []
    assert any("already gone" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_failing_server_listing_still_unlinks_dms(
    links: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    links.list_server_links.side_effect = RuntimeError("db down")
    links.list_user_links.return_value = [_dm_link("d1")]
    caplog.set_level(logging.INFO, logger=MODULE)

    await unlink_chat_platforms(USER_ID)

    links.delete_user_link.assert_awaited_once_with("d1", USER_ID)
    [error] = _errors(caplog)
    assert "server" in error and USER_ID in error


@pytest.mark.asyncio
async def test_unreachable_links_are_logged_without_raising(
    mocker: MockerFixture,
    caplog: pytest.LogCaptureFixture,
):
    mocker.patch(
        f"{MODULE}.platform_linking_db", side_effect=ConnectionError("no route")
    )
    caplog.set_level(logging.INFO, logger=MODULE)

    await unlink_chat_platforms(USER_ID)

    [error] = _errors(caplog)
    assert USER_ID in error and "no route" in error


@pytest.mark.asyncio
async def test_a_hung_link_listing_is_cut_off(
    mocker: MockerFixture,
    links: MagicMock,
    caplog: pytest.LogCaptureFixture,
):
    mocker.patch(f"{BEST_EFFORT_MODULE}.STEP_TIMEOUT_SECONDS", 0.05)

    async def hang(_: str) -> list[object]:
        await asyncio.sleep(10)
        return []

    links.list_server_links.side_effect = hang
    links.list_user_links.return_value = [_dm_link("d1")]
    caplog.set_level(logging.INFO, logger=MODULE)

    await asyncio.wait_for(unlink_chat_platforms(USER_ID), timeout=2)

    links.delete_user_link.assert_awaited_once_with("d1", USER_ID)
    [error] = _errors(caplog)
    assert "server" in error and USER_ID in error
