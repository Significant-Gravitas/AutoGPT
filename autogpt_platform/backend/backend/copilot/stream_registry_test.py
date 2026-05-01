"""Tests for disconnect_all_listeners in stream_registry."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from redis.exceptions import RedisError

from backend.copilot import stream_registry
from backend.copilot.executor.utils import get_session_lock_key


@pytest.fixture(autouse=True)
def _clear_listener_sessions():
    stream_registry._listener_sessions.clear()
    yield
    stream_registry._listener_sessions.clear()


async def _sleep_forever():
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise


@pytest.mark.asyncio
async def test_disconnect_all_listeners_cancels_matching_session():
    task_a = asyncio.create_task(_sleep_forever())
    task_b = asyncio.create_task(_sleep_forever())
    task_other = asyncio.create_task(_sleep_forever())

    stream_registry._listener_sessions[1] = ("sess-1", task_a)
    stream_registry._listener_sessions[2] = ("sess-1", task_b)
    stream_registry._listener_sessions[3] = ("sess-other", task_other)

    try:
        cancelled = await stream_registry.disconnect_all_listeners("sess-1")

        assert cancelled == 2
        assert task_a.cancelled()
        assert task_b.cancelled()
        assert not task_other.done()
        # Matching entries are removed, non-matching entries remain.
        assert 1 not in stream_registry._listener_sessions
        assert 2 not in stream_registry._listener_sessions
        assert 3 in stream_registry._listener_sessions
    finally:
        task_other.cancel()
        try:
            await task_other
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_disconnect_all_listeners_no_match_returns_zero():
    task = asyncio.create_task(_sleep_forever())
    stream_registry._listener_sessions[1] = ("sess-other", task)

    try:
        cancelled = await stream_registry.disconnect_all_listeners("sess-missing")

        assert cancelled == 0
        assert not task.done()
        assert 1 in stream_registry._listener_sessions
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_disconnect_all_listeners_skips_already_done_tasks():
    async def _noop():
        return None

    done_task = asyncio.create_task(_noop())
    await done_task
    stream_registry._listener_sessions[1] = ("sess-1", done_task)

    cancelled = await stream_registry.disconnect_all_listeners("sess-1")

    # Done tasks are filtered out before cancellation.
    assert cancelled == 0


@pytest.mark.asyncio
async def test_disconnect_all_listeners_empty_registry():
    cancelled = await stream_registry.disconnect_all_listeners("sess-1")
    assert cancelled == 0


@pytest.mark.asyncio
async def test_disconnect_all_listeners_timeout_not_counted():
    """Tasks that don't respond to cancellation (timeout) are not counted."""
    task = asyncio.create_task(_sleep_forever())
    stream_registry._listener_sessions[1] = ("sess-1", task)

    with patch.object(
        asyncio, "wait_for", new=AsyncMock(side_effect=asyncio.TimeoutError)
    ):
        cancelled = await stream_registry.disconnect_all_listeners("sess-1")

    assert cancelled == 0
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# stream_and_publish: closing the wrapper forwards GeneratorExit into the
# inner stream so its finally (stream lock release, etc.) runs deterministically.
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Minimal stand-in for a StreamBaseResponse so publish_chunk is a no-op."""

    def __init__(self, idx: int):
        self.idx = idx


@pytest.mark.asyncio
async def test_stream_and_publish_aclose_propagates_to_inner_stream():
    """Closing the wrapper MUST run the inner generator's finally block."""
    inner_finally_ran = asyncio.Event()

    async def _inner():
        try:
            yield _FakeEvent(0)
            yield _FakeEvent(1)
            yield _FakeEvent(2)
        finally:
            inner_finally_ran.set()

    inner = _inner()
    # Empty turn_id skips publish_chunk — keeps the test hermetic (no Redis).
    wrapper = stream_registry.stream_and_publish(
        session_id="sess-test", turn_id="", stream=inner
    )

    # Consume one event, then close the wrapper early.
    first = await wrapper.__anext__()
    assert isinstance(first, _FakeEvent)

    await wrapper.aclose()

    # The inner generator's finally must have run deterministically
    # (not deferred to GC) so the caller's cleanup (lock release, etc.)
    # is observable right after aclose returns.
    assert inner_finally_ran.is_set()


@pytest.mark.asyncio
async def test_stream_and_publish_logs_warning_on_publish_chunk_failure():
    """``stream_and_publish`` must not propagate a Redis publish failure —
    it warns once with full stack trace, keeps yielding, and logs
    subsequent failures at WARNING (terser, no exc_info) so repeated
    errors stay visible without flooding the trace."""
    from redis.exceptions import RedisError

    async def _inner():
        yield _FakeEvent(0)
        yield _FakeEvent(1)
        yield _FakeEvent(2)

    async def _raising_publish(turn_id, event, session_id=None):
        raise RedisError("boom")

    warning_mock = patch.object(
        stream_registry.logger, "warning", autospec=True
    ).start()
    try:
        with patch.object(stream_registry, "publish_chunk", new=_raising_publish):
            wrapper = stream_registry.stream_and_publish(
                session_id="sess-test", turn_id="turn-1", stream=_inner()
            )
            received = [evt async for evt in wrapper]
    finally:
        patch.stopall()

    # Every event still yields through — publish failures don't break the stream.
    assert len(received) == 3
    # One warning per failed publish (3 total).  First call carries a
    # stack trace (``exc_info=True``); subsequent calls are terser.
    assert warning_mock.call_count == 3
    assert warning_mock.call_args_list[0].kwargs.get("exc_info") is True
    assert warning_mock.call_args_list[1].kwargs.get("exc_info") is not True


@pytest.mark.asyncio
async def test_stream_and_publish_consumer_break_then_aclose_releases_inner():
    """The processor pattern — break on cancel, then aclose — must release."""
    inner_finally_ran = asyncio.Event()

    async def _inner():
        try:
            for idx in range(100):
                yield _FakeEvent(idx)
        finally:
            inner_finally_ran.set()

    inner = _inner()
    wrapper = stream_registry.stream_and_publish(
        session_id="sess-test", turn_id="", stream=inner
    )

    # Mimic the processor: consume a few events, simulate Stop by breaking,
    # then aclose the wrapper (as processor._execute_async now does in the
    # try/finally around the async for).
    try:
        count = 0
        async for _ in wrapper:
            count += 1
            if count >= 2:
                break
    finally:
        await wrapper.aclose()

    assert inner_finally_ran.is_set()


# ---------------------------------------------------------------------------
# mark_session_completed: the atomic meta flip to completed/failed must also
# release the per-session cluster lock, so the next enqueued turn's run
# handler can acquire it without waiting for the TTL (5 min default).
# ---------------------------------------------------------------------------


class _FakeRedis:
    """Minimal async-Redis fake: only the calls mark_session_completed makes."""

    def __init__(self, meta: dict[str, str]):
        self._meta = dict(meta)
        self.deleted_keys: list[str] = []
        self.delete = AsyncMock(side_effect=self._record_delete)

    async def _record_delete(self, *keys: str):
        self.deleted_keys.extend(keys)
        for k in keys:
            self._meta.pop(k, None)
        return len(keys)

    async def hgetall(self, _key: str):
        return dict(self._meta)

    async def hdel(self, _key: str, *fields: str) -> int:
        removed = 0
        for f in fields:
            if f in self._meta:
                del self._meta[f]
                removed += 1
        return removed


@pytest.mark.asyncio
async def test_mark_session_completed_releases_cluster_lock_on_success():
    """CAS swap must be followed by a DELETE on the session's lock key so a
    stuck-because-of-stale-lock session becomes immediately claimable."""
    fake_redis = _FakeRedis({"status": "running", "turn_id": "turn-1"})

    with (
        patch.object(
            stream_registry, "get_redis_async", new=AsyncMock(return_value=fake_redis)
        ),
        patch.object(
            stream_registry, "hash_compare_and_set", new=AsyncMock(return_value=True)
        ),
        patch.object(stream_registry, "publish_chunk", new=AsyncMock()),
        patch.object(
            stream_registry.chat_db(),
            "set_turn_duration",
            new=AsyncMock(),
            create=True,
        ),
    ):
        result = await stream_registry.mark_session_completed("sess-1")

    assert result is True
    assert get_session_lock_key("sess-1") in fake_redis.deleted_keys


@pytest.mark.asyncio
async def test_mark_session_completed_skips_lock_release_when_already_completed():
    """CAS failure = someone else completed the session first; we must not
    delete their already-released lock, and we must NOT publish StreamFinish
    twice (the winning caller already published it)."""
    fake_redis = _FakeRedis({"status": "completed", "turn_id": "turn-1"})
    publish_mock = AsyncMock()

    with (
        patch.object(
            stream_registry, "get_redis_async", new=AsyncMock(return_value=fake_redis)
        ),
        patch.object(
            stream_registry, "hash_compare_and_set", new=AsyncMock(return_value=False)
        ),
        patch.object(stream_registry, "publish_chunk", new=publish_mock),
    ):
        result = await stream_registry.mark_session_completed("sess-1")

    assert result is False
    assert get_session_lock_key("sess-1") not in fake_redis.deleted_keys
    assert not any(
        isinstance(call.args[1], stream_registry.StreamFinish)
        for call in publish_mock.call_args_list
    ), "StreamFinish must NOT be re-published on the CAS-no-op branch"


@pytest.mark.asyncio
async def test_mark_session_completed_survives_lock_release_redis_error():
    """A Redis hiccup during lock DELETE must not prevent the StreamFinish
    publish — the client's SSE stream would otherwise hang on the stale meta
    status while Redis recovers."""
    fake_redis = _FakeRedis({"status": "running", "turn_id": "turn-1"})
    fake_redis.delete = AsyncMock(side_effect=RedisError("boom"))
    publish_mock = AsyncMock()

    with (
        patch.object(
            stream_registry, "get_redis_async", new=AsyncMock(return_value=fake_redis)
        ),
        patch.object(
            stream_registry, "hash_compare_and_set", new=AsyncMock(return_value=True)
        ),
        patch.object(stream_registry, "publish_chunk", new=publish_mock),
        patch.object(
            stream_registry.chat_db(),
            "set_turn_duration",
            new=AsyncMock(),
            create=True,
        ),
    ):
        result = await stream_registry.mark_session_completed("sess-1")

    assert result is True
    assert any(
        isinstance(call.args[1], stream_registry.StreamFinish)
        for call in publish_mock.call_args_list
    ), "StreamFinish must still be published even if lock DELETE raises"


# ---------------------------------------------------------------------------
# Replays must contain protocol chunks only. Redis cursor data parts are not
# emitted because AI SDK resume needs the complete stream envelope from 0-0.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_to_session_replays_chunks_without_cursor_parts():
    """During replay, the subscriber queue contains chunks plus terminal finish."""
    import orjson

    from backend.copilot.response_model import (
        StreamTextDelta,
        StreamTextEnd,
        StreamTextStart,
    )

    # Three chunks recorded in Redis for a completed turn. Completed status
    # means the listener branch is skipped and only the replay path runs,
    # which keeps the test hermetic.
    stream_key_msgs = [
        (
            "9999-0",
            {"data": orjson.dumps(StreamTextStart(id="blk-1").model_dump()).decode()},
        ),
        (
            "9999-1",
            {
                "data": orjson.dumps(
                    StreamTextDelta(id="blk-1", delta="hi").model_dump()
                ).decode()
            },
        ),
        (
            "9999-2",
            {"data": orjson.dumps(StreamTextEnd(id="blk-1").model_dump()).decode()},
        ),
    ]

    fake_redis = AsyncMock()
    fake_redis.hgetall = AsyncMock(
        return_value={
            "user_id": "u1",
            "session_id": "sess-1",
            "turn_id": "turn-1",
            "status": "completed",  # finished → no listener task
        }
    )
    fake_redis.xread = AsyncMock(return_value=[("stream-key", stream_key_msgs)])

    with patch.object(
        stream_registry, "get_redis_async", new=AsyncMock(return_value=fake_redis)
    ):
        queue = await stream_registry.subscribe_to_session(
            session_id="sess-1", user_id="u1", last_message_id="0-0"
        )

    assert queue is not None

    delivered = []
    while not queue.empty():
        delivered.append(queue.get_nowait())

    assert len(delivered) == 4
    assert isinstance(delivered[0], StreamTextStart)
    assert isinstance(delivered[1], StreamTextDelta)
    assert isinstance(delivered[2], StreamTextEnd)
    assert isinstance(delivered[3], stream_registry.StreamFinish)
