"""A follow-up the turn ended without reading is reported, never dropped."""

import time

import pytest

from backend.copilot import held_follow_ups
from backend.copilot import pending_messages as pm_module
from backend.copilot.held_follow_ups import (
    follow_up_state,
    held_follow_up_reason,
    held_reason_for,
    hold_follow_ups_at_turn_end,
    restore_held_follow_up,
    take_follow_up,
)
from backend.copilot.pending_messages import (
    PendingMessage,
    drain_pending_messages,
    peek_pending_messages,
    push_pending_message,
)
from backend.copilot.pending_messages_test import _FakeRedis


@pytest.fixture()
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    redis = _FakeRedis()

    async def _get_redis_async() -> _FakeRedis:
        return redis

    monkeypatch.setattr(pm_module, "get_redis_async", _get_redis_async)
    monkeypatch.setattr(held_follow_ups, "get_redis_async", _get_redis_async)
    return redis


async def _queued(*texts: str) -> list[str]:
    ids = []
    for text in texts:
        message = PendingMessage(content=text)
        await push_pending_message("sess-1", message)
        ids.append(message.id)
    return ids


@pytest.mark.asyncio
async def test_a_turn_that_ends_before_reading_its_follow_ups_reports_them(
    fake_redis: _FakeRedis,
) -> None:
    await _queued("What is my team working on?")

    held = await hold_follow_ups_at_turn_end("sess-1", "turn-1", "error")

    assert held is not None
    assert (held.reason, held.count) == ("error", 1)
    assert await held_follow_up_reason("sess-1") == "error"
    # Reported, not consumed: the message still waits to be sent again.
    assert [m.content for m in await peek_pending_messages("sess-1")] == [
        "What is my team working on?"
    ]


@pytest.mark.asyncio
async def test_a_turn_that_read_every_follow_up_clears_an_earlier_hold(
    fake_redis: _FakeRedis,
) -> None:
    await _queued("status?")
    await hold_follow_ups_at_turn_end("sess-1", "turn-1", "paused")
    await drain_pending_messages("sess-1")

    assert await hold_follow_ups_at_turn_end("sess-1", "turn-2", "ended") is None
    assert await held_follow_up_reason("sess-1") is None


@pytest.mark.asyncio
async def test_a_resend_takes_the_follow_up_it_names_once(
    fake_redis: _FakeRedis,
) -> None:
    status, monday = await _queued("status?", "and Monday")
    await hold_follow_ups_at_turn_end("sess-1", "turn-1", "stopped")

    taken = await take_follow_up("sess-1", status)

    assert taken is not None and taken.content == "status?"
    assert [m.id for m in await peek_pending_messages("sess-1")] == [monday]
    # A second Retry of it finds it gone, and read: it must not run again.
    assert await take_follow_up("sess-1", status) is None
    assert await follow_up_state("sess-1", status) == "read"


@pytest.mark.asyncio
async def test_a_follow_up_another_turn_read_meanwhile_is_read_not_unknown(
    fake_redis: _FakeRedis,
) -> None:
    [status] = await _queued("status?")
    await hold_follow_ups_at_turn_end("sess-1", "turn-1", "error")
    assert await follow_up_state("sess-1", status) == "waiting"

    # Another tab's turn (or an approval's) reads the queue before the Retry.
    await drain_pending_messages("sess-1")

    assert await take_follow_up("sess-1", status) is None
    assert await follow_up_state("sess-1", status) == "read"


@pytest.mark.asyncio
async def test_a_follow_up_that_expired_unread_is_unknown(
    fake_redis: _FakeRedis,
) -> None:
    assert await follow_up_state("sess-1", "expired-id") == "unknown"


@pytest.mark.asyncio
async def test_the_next_turns_read_of_the_queue_ends_the_hold(
    fake_redis: _FakeRedis,
) -> None:
    await _queued("status?")
    await hold_follow_ups_at_turn_end("sess-1", "turn-1", "error")

    await drain_pending_messages("sess-1")

    assert await held_follow_up_reason("sess-1") is None


@pytest.mark.asyncio
async def test_a_queue_with_a_message_newer_than_the_hold_is_not_reported_held(
    fake_redis: _FakeRedis,
) -> None:
    await _queued("status?")
    await hold_follow_ups_at_turn_end("sess-1", "turn-1", "error")
    covered = await peek_pending_messages("sess-1")
    assert await held_follow_up_reason("sess-1", covering=covered) == "error"

    await push_pending_message(
        "sess-1", PendingMessage(content="also", enqueued_at=time.time() + 60)
    )

    waiting = await peek_pending_messages("sess-1")
    assert await held_follow_up_reason("sess-1", covering=waiting) is None


@pytest.mark.asyncio
async def test_a_refused_resend_puts_the_follow_up_back(
    fake_redis: _FakeRedis,
) -> None:
    [status] = await _queued("status?")
    await hold_follow_ups_at_turn_end("sess-1", "turn-1", "error")
    taken = await take_follow_up("sess-1", status)

    await restore_held_follow_up("sess-1", taken)

    assert [m.id for m in await peek_pending_messages("sess-1")] == [status]
    assert await follow_up_state("sess-1", status) == "waiting"


@pytest.mark.parametrize(
    "failed, cancelled, paused, reason",
    [
        (True, True, False, "stopped"),
        (True, False, False, "error"),
        (True, False, True, "error"),
        (False, False, True, "paused"),
        (False, False, False, "ended"),
    ],
)
def test_the_reason_names_how_the_turn_ended(
    failed: bool, cancelled: bool, paused: bool, reason: str
) -> None:
    assert held_reason_for(failed=failed, cancelled=cancelled, paused=paused) == reason
