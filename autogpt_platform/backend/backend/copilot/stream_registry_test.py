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
