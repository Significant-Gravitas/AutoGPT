"""The follow-up queue a turn closes at its last look, on the real Lua scripts.

``follow_up_after_last_look_test`` runs the race through the SDK engine; these
pin the gate itself: who can still push, who can close it, and what reopens it.
"""

import contextlib
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from fakeredis.aioredis import FakeRedis

from backend.copilot import pending_messages, stream_registry
from backend.copilot.pending_message_helpers import drain_pending_safe
from backend.copilot.pending_messages import (
    PendingMessage,
    drain_pending_or_close,
    peek_pending_messages,
    push_pending_message_if_session_running,
    reopen_follow_ups,
)

SESSION = "gate-session"
META = stream_registry.get_session_meta_key(SESSION)


@pytest.fixture
async def redis() -> AsyncIterator[FakeRedis]:
    async with FakeRedis(decode_responses=True) as fake:
        with contextlib.ExitStack() as stack:
            for module in (pending_messages, stream_registry):
                stack.enter_context(
                    patch.object(
                        module, "get_redis_async", AsyncMock(return_value=fake)
                    )
                )
            chat_db = stack.enter_context(patch.object(stream_registry, "chat_db"))
            chat_db.return_value.set_turn_duration = AsyncMock()
            yield fake


async def _start_turn(turn_id: str) -> None:
    await stream_registry.create_session(
        SESSION, None, "chat_stream", "chat", turn_id=turn_id
    )


async def _push(text: str, *, message_id: str | None = None) -> int | None:
    message = PendingMessage(content=text)
    if message_id is not None:
        message.id = message_id
    return await push_pending_message_if_session_running(
        SESSION,
        message,
        session_meta_key=META,
        deduplicate=message_id is not None,
    )


async def _last_look(turn_id: str = "turn-1") -> list[str]:
    drained = await drain_pending_or_close(
        SESSION, session_meta_key=META, turn_id=turn_id
    )
    return [message.content for message in drained]


@pytest.mark.asyncio
async def test_a_last_look_that_finds_follow_ups_takes_them_and_stays_open(redis):
    await _start_turn("turn-1")
    await _push("first")
    await _push("second")

    assert await _last_look() == ["first", "second"]
    # The turn continues with them, so it can still take more.
    assert await _push("third") == 1
    assert [m.content for m in await peek_pending_messages(SESSION)] == ["third"]


@pytest.mark.asyncio
async def test_a_last_look_at_an_empty_queue_closes_it_to_that_turn(redis):
    await _start_turn("turn-1")

    assert await _last_look() == []
    assert await _push("too late") is None
    assert await peek_pending_messages(SESSION) == []


@pytest.mark.asyncio
async def test_a_last_look_that_finds_only_unreadable_entries_still_closes_it(
    redis,
):
    await _start_turn("turn-1")
    # One no reader can parse, such as an entry a mismatched release wrote.
    await redis.rpush(pending_messages._buffer_key(SESSION), "null")

    assert await _last_look() == []
    assert await _push("too late") is None
    assert await peek_pending_messages(SESSION) == []


@pytest.mark.asyncio
async def test_the_next_turn_takes_follow_ups_again(redis):
    await _start_turn("turn-1")
    await _last_look()
    await stream_registry.mark_session_completed(SESSION, turn_id="turn-1")

    await _start_turn("turn-2")

    assert await _push("for the new turn") == 1


@pytest.mark.asyncio
async def test_a_rerun_of_the_turn_reopens_the_queue_its_last_attempt_closed(redis):
    """A redelivered turn keeps its turn id and gets no new ``create_session``,
    so the marker its crashed attempt wrote is still there when it reruns."""
    await _start_turn("turn-1")
    await _last_look()

    await reopen_follow_ups(SESSION, session_meta_key=META, turn_id="turn-1")

    assert await _push("during the rerun") == 1


@pytest.mark.asyncio
async def test_a_first_look_still_takes_the_backlog_when_reopening_fails(redis):
    """Reopening is best effort: after a Redis error the rerun still runs with
    what was queued, and only its later follow-ups get the 409."""
    await _start_turn("turn-1")
    await _push("waiting")

    with patch.object(
        pending_messages,
        "hash_compare_and_set",
        AsyncMock(side_effect=ConnectionError("Redis went away")),
    ):
        drained = await drain_pending_safe(
            SESSION, drain_backlog=True, open_for_turn="turn-1"
        )

    assert [message.content for message in drained] == ["waiting"]


@pytest.mark.asyncio
async def test_reopening_for_another_turn_leaves_a_closed_queue_closed(redis):
    await _start_turn("turn-1")
    await _last_look()

    await reopen_follow_ups(SESSION, session_meta_key=META, turn_id="turn-0")

    assert await _push("too late") is None


@pytest.mark.asyncio
async def test_a_last_look_closes_only_its_own_turns_queue(redis):
    """An old turn that outlived its registration reaches its last look after
    the chat's next turn took the session meta."""
    await _start_turn("turn-2")
    before = await redis.hgetall(META)

    assert await _last_look("turn-1") == []

    assert await redis.hgetall(META) == before
    assert await _push("for the new turn") == 1


@pytest.mark.asyncio
async def test_a_resent_accepted_follow_up_is_still_answered_as_accepted(redis):
    """A retransmit of a send the queue took before it closed must not be told
    to start a turn: the client would send the same message twice."""
    await _start_turn("turn-1")
    assert await _push("sent once", message_id="msg-1") == 1
    assert await _last_look() == ["sent once"]
    await _last_look()

    assert await _push("sent once", message_id="msg-1") == 0
    assert await _push("a different one", message_id="msg-2") is None


@pytest.mark.parametrize("status", ["completed", "failed", None])
@pytest.mark.asyncio
async def test_only_a_running_turn_closes_the_queue(redis, status):
    """The turn the meta names, but no longer running (or no meta at all)."""
    if status is not None:
        await _start_turn("turn-1")
        await redis.hset(META, "status", status)
    before = await redis.hgetall(META)

    assert await _last_look() == []

    assert await redis.hgetall(META) == before
