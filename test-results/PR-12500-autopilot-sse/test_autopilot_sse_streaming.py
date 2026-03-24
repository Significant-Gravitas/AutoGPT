"""
PR #12500 - AutoPilot SSE Stream Registry Integration Tests

Tests that the AutoPilot block's sub-agent execution correctly registers
with the stream registry so SSE updates flow to the frontend in real-time.

Test scenarios:
1. stream_and_publish passes through all events and publishes to Redis
2. stream_and_publish skips StreamFinish/StreamError (handled by mark_session_completed)
3. stream_and_publish gracefully degrades when turn_id is empty
4. stream_and_publish gracefully degrades on Redis publish failures
5. collect_copilot_response creates registry session and publishes events
6. collect_copilot_response cleans up on error
7. Full lifecycle: create_session -> stream_and_publish -> mark_session_completed
8. Real Redis integration: verify keys are created and cleaned up
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---- Imports from the codebase under test ----
from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamTextDelta,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
    StreamUsage,
)
from backend.copilot.stream_registry import stream_and_publish


# ---- Helpers ----

async def make_stream(*events) -> AsyncIterator:
    """Create an async iterator from a list of events."""
    for e in events:
        yield e


# ============================================================
# Test 1: stream_and_publish passes through all events
# ============================================================

@pytest.mark.asyncio
async def test_stream_and_publish_passes_through_all_events():
    """All events from the source stream are yielded unchanged."""
    events = [
        StreamTextStart(id="t1"),
        StreamTextDelta(id="t1", delta="Hello"),
        StreamTextDelta(id="t1", delta=" world"),
        StreamToolInputAvailable(toolCallId="tc1", toolName="search", input={"q": "test"}),
        StreamToolOutputAvailable(toolCallId="tc1", output="result", success=True),
        StreamUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        StreamFinish(),
    ]

    with patch("backend.copilot.stream_registry.publish_chunk", new_callable=AsyncMock) as mock_publish:
        collected = []
        async for event in stream_and_publish(
            session_id="test-session",
            turn_id="test-turn-id",
            stream=make_stream(*events),
        ):
            collected.append(event)

    # All events should be yielded
    assert len(collected) == len(events)
    for orig, received in zip(events, collected):
        assert orig is received, f"Event {type(orig).__name__} was not passed through"


# ============================================================
# Test 2: stream_and_publish skips StreamFinish and StreamError
# ============================================================

@pytest.mark.asyncio
async def test_stream_and_publish_skips_finish_and_error():
    """StreamFinish and StreamError are NOT published to Redis."""
    events = [
        StreamTextDelta(id="t1", delta="Hello"),
        StreamError(errorText="oops"),
        StreamFinish(),
    ]

    with patch("backend.copilot.stream_registry.publish_chunk", new_callable=AsyncMock) as mock_publish:
        collected = []
        async for event in stream_and_publish(
            session_id="test-session",
            turn_id="test-turn-id",
            stream=make_stream(*events),
        ):
            collected.append(event)

    # All events are yielded (pass-through)
    assert len(collected) == 3

    # Only StreamTextDelta should be published (StreamError and StreamFinish skipped)
    assert mock_publish.await_count == 1
    published_event = mock_publish.call_args_list[0].args[1]
    assert isinstance(published_event, StreamTextDelta)
    assert published_event.delta == "Hello"


# ============================================================
# Test 3: stream_and_publish skips publishing when turn_id is empty
# ============================================================

@pytest.mark.asyncio
async def test_stream_and_publish_no_publish_when_turn_id_empty():
    """When turn_id is empty, no events are published but all are yielded."""
    events = [
        StreamTextDelta(id="t1", delta="Hello"),
        StreamTextDelta(id="t1", delta=" world"),
    ]

    with patch("backend.copilot.stream_registry.publish_chunk", new_callable=AsyncMock) as mock_publish:
        collected = []
        async for event in stream_and_publish(
            session_id="test-session",
            turn_id="",  # empty = disabled
            stream=make_stream(*events),
        ):
            collected.append(event)

    assert len(collected) == 2
    mock_publish.assert_not_awaited()


# ============================================================
# Test 4: stream_and_publish gracefully degrades on Redis failures
# ============================================================

@pytest.mark.asyncio
async def test_stream_and_publish_graceful_on_redis_failure():
    """Redis publish failures don't break the stream - events still yield."""
    from redis.exceptions import RedisError

    events = [
        StreamTextDelta(id="t1", delta="first"),
        StreamTextDelta(id="t1", delta="second"),
        StreamTextDelta(id="t1", delta="third"),
    ]

    with patch(
        "backend.copilot.stream_registry.publish_chunk",
        new_callable=AsyncMock,
        side_effect=RedisError("Connection lost"),
    ) as mock_publish:
        collected = []
        async for event in stream_and_publish(
            session_id="test-session-1234",
            turn_id="test-turn-id",
            stream=make_stream(*events),
        ):
            collected.append(event)

    # All events still yielded despite Redis failures
    assert len(collected) == 3
    assert collected[0].delta == "first"
    assert collected[1].delta == "second"
    assert collected[2].delta == "third"


# ============================================================
# Test 5: collect_copilot_response creates registry session
# ============================================================

@pytest.mark.asyncio
async def test_collect_creates_registry_session_and_publishes():
    """collect_copilot_response creates a stream registry session,
    publishes events via stream_and_publish, and finalizes."""
    events = [
        StreamTextDelta(id="t1", delta="Sub-agent "),
        StreamTextDelta(id="t1", delta="response"),
        StreamToolInputAvailable(toolCallId="tc1", toolName="web_search", input={"q": "test"}),
        StreamToolOutputAvailable(toolCallId="tc1", output="found it", success=True),
        StreamUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        StreamFinish(),
    ]

    async def mock_stream(**kwargs):
        for e in events:
            yield e

    with patch("backend.copilot.sdk.collect.stream_registry") as mock_reg, \
         patch("backend.copilot.sdk.collect.stream_chat_completion_sdk", side_effect=mock_stream):

        mock_reg.create_session = AsyncMock()
        mock_reg.publish_chunk = AsyncMock()
        mock_reg.mark_session_completed = AsyncMock()

        # Re-implement stream_and_publish as pass-through with publish tracking
        async def fake_stream_and_publish(session_id, turn_id, stream):
            async for event in stream:
                if turn_id and not isinstance(event, (StreamFinish, StreamError)):
                    await mock_reg.publish_chunk(turn_id, event)
                yield event

        mock_reg.stream_and_publish = fake_stream_and_publish

        from backend.copilot.sdk.collect import collect_copilot_response

        result = await collect_copilot_response(
            session_id="autopilot-session",
            message="summarize this text",
            user_id="test-user",
        )

    # Verify result aggregation
    assert result.response_text == "Sub-agent response"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool_name"] == "web_search"
    assert result.total_tokens == 30

    # Verify stream registry lifecycle
    mock_reg.create_session.assert_awaited_once()
    create_kwargs = mock_reg.create_session.call_args
    assert create_kwargs.kwargs["session_id"] == "autopilot-session"
    assert create_kwargs.kwargs["user_id"] == "test-user"
    assert create_kwargs.kwargs["tool_call_id"] == "autopilot_stream"
    assert create_kwargs.kwargs["tool_name"] == "autopilot"

    # Verify events were published (StreamFinish excluded)
    published_types = [
        type(call.args[1]).__name__
        for call in mock_reg.publish_chunk.call_args_list
    ]
    assert "StreamTextDelta" in published_types
    assert "StreamToolInputAvailable" in published_types
    assert "StreamToolOutputAvailable" in published_types
    assert "StreamUsage" in published_types
    assert "StreamFinish" not in published_types

    # Verify session was finalized
    mock_reg.mark_session_completed.assert_awaited_once()
    complete_kwargs = mock_reg.mark_session_completed.call_args
    assert complete_kwargs.kwargs.get("error_message") is None


# ============================================================
# Test 6: collect_copilot_response cleans up on error
# ============================================================

@pytest.mark.asyncio
async def test_collect_marks_error_on_stream_error():
    """When stream yields StreamError, session is marked as failed."""
    events = [
        StreamTextDelta(id="t1", delta="partial"),
        StreamError(errorText="sub-agent crashed"),
    ]

    async def mock_stream(**kwargs):
        for e in events:
            yield e

    with patch("backend.copilot.sdk.collect.stream_registry") as mock_reg, \
         patch("backend.copilot.sdk.collect.stream_chat_completion_sdk", side_effect=mock_stream):

        mock_reg.create_session = AsyncMock()
        mock_reg.publish_chunk = AsyncMock()
        mock_reg.mark_session_completed = AsyncMock()

        async def fake_stream_and_publish(session_id, turn_id, stream):
            async for event in stream:
                if turn_id and not isinstance(event, (StreamFinish, StreamError)):
                    await mock_reg.publish_chunk(turn_id, event)
                yield event

        mock_reg.stream_and_publish = fake_stream_and_publish

        from backend.copilot.sdk.collect import collect_copilot_response

        with pytest.raises(RuntimeError, match="sub-agent crashed"):
            await collect_copilot_response(
                session_id="autopilot-session",
                message="do something",
                user_id="test-user",
            )

    # Session should be marked as failed with the error message
    mock_reg.mark_session_completed.assert_awaited_once()
    complete_args = mock_reg.mark_session_completed.call_args
    assert complete_args.args[0] == "autopilot-session"
    assert complete_args.kwargs["error_message"] == "sub-agent crashed"
    assert complete_args.kwargs["skip_error_publish"] is False


# ============================================================
# Test 7: mark_session_completed skip_error_publish parameter
# ============================================================

@pytest.mark.asyncio
async def test_mark_session_completed_skip_error_publish():
    """Verify that skip_error_publish prevents duplicate StreamError publication."""
    from backend.copilot.stream_registry import mark_session_completed

    session_id = f"test-skip-{uuid.uuid4()}"
    turn_id = str(uuid.uuid4())

    with patch("backend.copilot.stream_registry.get_redis_async") as mock_get_redis:
        mock_redis = AsyncMock()
        mock_get_redis.return_value = mock_redis

        # Simulate session exists with status "active"
        mock_redis.hgetall.return_value = {
            b"status": b"active",
            b"turn_id": turn_id.encode(),
            b"user_id": b"test-user",
        }
        mock_redis.hset.return_value = True

        # Simulate XADD for publishing
        mock_redis.xadd.return_value = b"1234-0"

        result = await mark_session_completed(
            session_id,
            error_message="already published error",
            skip_error_publish=True,
        )

        # Should still mark as failed
        if result:
            # Check that StreamError was NOT published to xadd
            # (only StreamFinish should be published)
            xadd_calls = [
                call for call in mock_redis.xadd.call_args_list
            ]
            for call in xadd_calls:
                data = call.args[1] if len(call.args) > 1 else call.kwargs.get("fields", {})
                if isinstance(data, dict):
                    chunk_data = data.get(b"data") or data.get("data", b"{}")
                    if isinstance(chunk_data, bytes):
                        chunk_data = chunk_data.decode()
                    assert "StreamError" not in str(chunk_data) or "errorText" not in str(chunk_data), \
                        "StreamError should not be published when skip_error_publish=True"


# ============================================================
# Test 8: Real Redis integration test
# ============================================================

@pytest.mark.asyncio
async def test_real_redis_stream_registry_lifecycle():
    """Integration test: verify stream registry keys in real Redis.

    This test uses the actual Redis instance to verify that:
    1. create_session creates the session metadata hash key
    2. stream_and_publish publishes events to a Redis stream
    3. mark_session_completed updates the status correctly

    Requires Redis on localhost:6379 (Docker).
    """
    try:
        import redis.asyncio as aioredis
    except ImportError:
        pytest.skip("redis package not available")

    try:
        r = aioredis.Redis(host="localhost", port=6379, decode_responses=True)
        await r.ping()
    except Exception:
        pytest.skip("Redis not available at localhost:6379")

    session_id = f"test-autopilot-sse-{uuid.uuid4()}"
    turn_id = str(uuid.uuid4())
    # Key prefixes from ChatConfig defaults
    meta_key = f"chat:task:meta:{session_id}"
    stream_key = f"chat:stream:{turn_id}"

    try:
        # Step 1: Create session via the real stream_registry module
        from backend.copilot.stream_registry import (
            create_session,
            mark_session_completed,
        )

        session = await create_session(
            session_id=session_id,
            user_id="test-user-123",
            tool_call_id="autopilot_stream",
            tool_name="autopilot",
            turn_id=turn_id,
        )

        # Verify session metadata exists in Redis
        session_data = await r.hgetall(meta_key)
        assert session_data.get("status") == "running", \
            f"Session should be running, got: {session_data}"
        assert session_data.get("turn_id") == turn_id
        assert session_data.get("tool_name") == "autopilot"
        assert session_data.get("tool_call_id") == "autopilot_stream"
        assert session_data.get("user_id") == "test-user-123"

        # Step 2: Publish events via stream_and_publish (uses real Redis)
        events = [
            StreamTextDelta(id="t1", delta="Real-time "),
            StreamTextDelta(id="t1", delta="update!"),
            StreamToolInputAvailable(
                toolCallId="tc1", toolName="search", input={"q": "test"}
            ),
        ]

        collected = []
        async for event in stream_and_publish(
            session_id=session_id,
            turn_id=turn_id,
            stream=make_stream(*events),
        ):
            collected.append(event)

        assert len(collected) == 3, "All events should be yielded"

        # Verify events were published to the Redis stream
        stream_entries = await r.xrange(stream_key)
        assert len(stream_entries) >= 3, \
            f"Expected at least 3 stream entries, got {len(stream_entries)}"

        # Step 3: Mark session completed
        result = await mark_session_completed(session_id)
        assert result is True, "Should return True for newly completed session"

        # Verify session status updated
        session_data = await r.hgetall(meta_key)
        assert session_data.get("status") == "completed", \
            f"Session should be completed, got: {session_data.get('status')}"

        # Verify calling mark_session_completed again returns False (idempotent)
        result2 = await mark_session_completed(session_id)
        assert result2 is False, "Second call should return False (already completed)"

    finally:
        # Cleanup test keys
        await r.delete(meta_key, stream_key)
        await r.aclose()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
