"""Tests for disconnect_all_listeners in stream_registry."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot import stream_registry


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
