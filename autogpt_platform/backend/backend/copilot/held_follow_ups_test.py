"""A follow-up the turn ended without reading is reported, never dropped."""

import time

import pytest

from backend.copilot import held_follow_ups
from backend.copilot import pending_messages as pm_module
from backend.copilot.held_follow_ups import (
    held_follow_up_reason,
    held_reason_for,
    hold_follow_ups_at_turn_end,
    holds_follow_up,
    restore_held_follow_up,
    take_held_follow_up,
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


async def _queued(*texts: str) -> None:
    for text in texts:
        await push_pending_message("sess-1", PendingMessage(content=text))


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
async def test_a_resend_of_a_held_follow_up_takes_its_waiting_copy_out_once(
    fake_redis: _FakeRedis,
) -> None:
    await _queued("status?", "and Monday")
    await hold_follow_ups_at_turn_end("sess-1", "turn-1", "stopped")

    taken = await take_held_follow_up("sess-1", "status?")

    assert taken is not None and taken.content == "status?"
    assert [m.content for m in await peek_pending_messages("sess-1")] == ["and Monday"]
    assert await take_held_follow_up("sess-1", "status?") is None


@pytest.mark.asyncio
async def test_the_next_turns_read_of_the_queue_ends_the_hold(
    fake_redis: _FakeRedis,
) -> None:
    await _queued("status?")
    await hold_follow_ups_at_turn_end("sess-1", "turn-1", "error")
    assert await holds_follow_up("sess-1", "status?")

    await drain_pending_messages("sess-1")
    await _queued("status?")

    # Typed again while that turn runs: queued, not held.
    assert await held_follow_up_reason("sess-1") is None
    assert not await holds_follow_up("sess-1", "status?")
    assert await take_held_follow_up("sess-1", "status?") is None


@pytest.mark.asyncio
async def test_a_message_typed_again_after_the_turn_ended_is_not_the_held_one(
    fake_redis: _FakeRedis,
) -> None:
    await _queued("yes")
    await hold_follow_ups_at_turn_end("sess-1", "turn-1", "ended")
    # Retried: the only held copy goes out, and its turn reads nothing more.
    assert await take_held_follow_up("sess-1", "yes") is not None

    # The user types the same word again while that turn runs.
    await push_pending_message(
        "sess-1", PendingMessage(content="yes", enqueued_at=time.time() + 60)
    )

    assert not await holds_follow_up("sess-1", "yes")
    assert await take_held_follow_up("sess-1", "yes") is None
    assert [m.content for m in await peek_pending_messages("sess-1")] == ["yes"]


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
    await _queued("status?")
    await hold_follow_ups_at_turn_end("sess-1", "turn-1", "error")
    taken = await take_held_follow_up("sess-1", "status?")

    await restore_held_follow_up("sess-1", taken)

    assert [m.content for m in await peek_pending_messages("sess-1")] == ["status?"]


@pytest.mark.asyncio
async def test_a_send_takes_nothing_while_follow_ups_wait_for_a_running_turn(
    fake_redis: _FakeRedis,
) -> None:
    # Queued for a turn that is still running: it will read them itself.
    await _queued("status?")

    assert await take_held_follow_up("sess-1", "status?") is None
    assert len(await peek_pending_messages("sess-1")) == 1


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
