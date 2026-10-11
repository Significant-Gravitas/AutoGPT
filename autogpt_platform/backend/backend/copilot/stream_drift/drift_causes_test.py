"""Backend causes of chat state drifting from server state, via the real registry.

Each live cause is a strict expected failure whose reason names it: CI stays
green while the bug is there, and the test turns red once a fix makes it
pass. ``raises=AssertionError`` keeps a harness error from passing as the bug.
The completion writers' turn guard (#14974) is already pinned by
``stream_registry_test.py::TestCompletionOnRealRedis``.
"""

import asyncio
import functools
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from backend.copilot import stream_heartbeat, stream_registry
from backend.copilot.baseline import service as baseline
from backend.copilot.executor.manager import CoPilotExecutor
from backend.copilot.executor.utils import CoPilotExecutionEntry, get_session_lock_key
from backend.copilot.model import ChatSession
from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamReasoningDelta,
    StreamReasoningEnd,
    StreamReasoningStart,
    StreamStart,
    StreamStatus,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
)
from backend.data import redis_client
from backend.data.redis_client import get_redis_async
from backend.util.testing import is_tcp_port_reachable

from .scripted import baseline_turn, provider_round, session_with_prompt

requires_redis = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no local Redis reachable; the stream registry needs one to run",
)


@pytest.fixture
async def session_id():
    session_id = f"drift-{uuid.uuid4().hex}"
    yield session_id
    redis = await get_redis_async()
    await redis.delete(stream_registry.get_session_meta_key(session_id))
    await redis.delete(get_session_lock_key(session_id))


@pytest.fixture
async def turn_id():
    turn_id = str(uuid.uuid4())
    yield turn_id
    redis = await get_redis_async()
    await redis.delete(stream_registry._get_turn_stream_key(turn_id))


@requires_redis
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "W2: create_session overwrites the meta of a turn that is still "
        "running, so every reader follows the new turn and the running one "
        "streams to nobody"
    ),
)
async def test_a_dispatch_does_not_take_over_a_running_turns_meta(
    session_id: str,
) -> None:
    await stream_registry.create_session(session_id, None, "", "", turn_id="a")
    # A second POST let through acquire_turn_slot's refresh branch.
    await stream_registry.create_session(session_id, None, "", "", turn_id="b")

    active, _ = await stream_registry.get_active_session(session_id)
    assert active is not None and active.turn_id == "a"


@requires_redis
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "W3: a turn whose executor died reads as running until its meta "
        "expires, up to an hour; the lapsed cluster lock is never consulted"
    ),
)
async def test_a_turn_whose_executor_died_is_not_reported_running(
    session_id: str, turn_id: str
) -> None:
    await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
    # The executor publishes this only after taking the cluster lock.
    await stream_registry.publish_chunk(
        turn_id,
        StreamStatus(message="Setting up your environment…"),
        session_id=session_id,
    )
    # Its pod is gone: nothing refreshes the lock and it has lapsed.
    redis = await get_redis_async()
    if await redis.exists(get_session_lock_key(session_id)):
        pytest.fail("harness: the cluster lock should be gone")

    active, _ = await stream_registry.get_active_session(session_id)
    assert active is None


@requires_redis
async def test_a_resume_of_a_long_running_turn_replays_whole_blocks(
    session_id: str, turn_id: str
) -> None:
    """W5: a running turn's stream used to be capped at 10,000 entries, so a
    resume replayed from mid-block and the client's parser threw."""
    await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
    for chunk in [
        StreamStart(messageId="m1"),
        StreamTextStart(id="t1"),
        *(StreamTextDelta(id="t1", delta=f"{i} ") for i in range(_PAST_THE_OLD_CAP)),
    ]:
        await stream_registry.publish_chunk(turn_id, chunk, session_id=session_id)

    queue = await stream_registry.subscribe_to_session(session_id, None, "0-0")
    if queue is None:
        pytest.fail("harness: the running turn has no subscribable stream")
    try:
        replay = [queue.get_nowait()[1] for _ in range(queue.qsize())]
    finally:
        await stream_registry.unsubscribe_from_session(session_id, queue)

    assert len(replay) == _PAST_THE_OLD_CAP + 2
    assert _orphan_parts(replay) == []


# Past the old 10,000 cap by several of Redis's 100-entry stream nodes, which
# is where an approximate MAXLEN starts trimming.
_PAST_THE_OLD_CAP = 10_500


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "W6: after the silence watchdog's last tier a silent turn publishes "
        "nothing, so nothing refreshes the session meta's TTL and it expires "
        "under a live turn"
    ),
)
async def test_a_silent_turn_keeps_publishing_within_the_stream_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ttl = stream_registry.config.stream_ttl
    now = 0.0

    def clock() -> float:
        nonlocal now
        now += 5.0
        return now

    monkeypatch.setattr(
        stream_heartbeat,
        "SilenceWatchdog",
        functools.partial(stream_heartbeat.SilenceWatchdog, clock=clock),
    )
    ended = asyncio.Event()

    async def silent_engine() -> AsyncGenerator[StreamBaseResponse, None]:
        yield StreamStart(messageId="m1")
        await ended.wait()

    published_at: list[float] = []

    async def consume() -> None:
        async for _ in stream_heartbeat.wrap_stream_with_heartbeat(
            silent_engine(), tick_s=0.001
        ):
            published_at.append(now)

    consumer = asyncio.create_task(consume())
    try:
        async with asyncio.timeout(30):
            while now < 2 * ttl:
                await asyncio.sleep(0.01)
            ended.set()
            await consumer
    finally:
        ended.set()
        consumer.cancel()

    gaps = [b - a for a, b in zip(published_at, [*published_at[1:], now])]
    assert max(gaps) < ttl


@requires_redis
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "Approval wake: the end-of-turn wake dispatches the follow-up turn while "
        "the ending turn is still registered, and the executor drops it as a "
        "duplicate without closing it, so the chat reads running forever"
    ),
)
async def test_a_turn_the_executor_drops_is_not_left_running(
    session_id: str, turn_id: str
) -> None:
    # held.wake -> dispatch_turn writes the follow-up turn's meta, then enqueues it.
    await stream_registry.create_session(session_id, None, "", "", turn_id=turn_id)
    executor = CoPilotExecutor()
    # The turn whose end woke it has not left the executor yet.
    executor.active_tasks[session_id] = (Future(), threading.Event())
    channel = MagicMock(is_open=True)
    channel.connection.add_callback_threadsafe.side_effect = lambda ack: ack()
    entry = CoPilotExecutionEntry(
        session_id=session_id, turn_id=turn_id, user_id=None, message="wake"
    )

    executor._handle_run_message(
        channel,
        MagicMock(delivery_tag=1),
        MagicMock(),
        entry.model_dump_json().encode(),
    )

    dropped = channel.basic_nack.call_args == call(1, requeue=False)
    # A fix may close the dropped turn asynchronously.
    deadline = time.monotonic() + 5
    while dropped and time.monotonic() < deadline:
        if not await _reads_running(session_id):
            break
        await asyncio.sleep(0.1)
    assert not (
        dropped and await _reads_running(session_id)
    ), "the executor dropped the turn and it still reads running"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "W8: the baseline engine logs and swallows a failed final persist, so "
        "the turn finishes as a success and the streamed reply is not in the "
        "rows the client hydrates from"
    ),
)
async def test_a_reply_that_failed_to_persist_ends_the_turn_in_error(
    monkeypatch: pytest.MonkeyPatch, baseline_io: list[ChatSession]
) -> None:
    monkeypatch.setattr(
        baseline,
        "call_provider_stream",
        AsyncMock(return_value=provider_round(["The answer ", "is 42."])),
    )
    persist = AsyncMock(side_effect=ConnectionError("database unavailable"))
    monkeypatch.setattr(baseline, "upsert_chat_session", persist)

    turn = baseline_turn(session_with_prompt("Why?"), str(uuid.uuid4()))
    events = [event async for event in turn]

    streamed = "".join(e.delta for e in events if isinstance(e, StreamTextDelta))
    if streamed != "The answer is 42." or not persist.await_count:
        pytest.fail("harness: the reply should stream and its persist should run")
    assert any(isinstance(event, StreamError) for event in events)


async def _reads_running(session_id: str) -> bool:
    active, _ = await stream_registry.get_active_session(session_id)
    return active is not None


def _orphan_parts(chunks: list[StreamBaseResponse]) -> list[str]:
    """Deltas and ends whose start the replay does not contain."""
    open_ids: set[str] = set()
    orphans = []
    for chunk in chunks:
        if isinstance(chunk, (StreamTextStart, StreamReasoningStart)):
            open_ids.add(chunk.id)
        elif isinstance(
            chunk,
            (StreamTextDelta, StreamTextEnd, StreamReasoningDelta, StreamReasoningEnd),
        ):
            if chunk.id not in open_ids:
                orphans.append(f"{chunk.type.value}:{chunk.id}")
    return orphans
