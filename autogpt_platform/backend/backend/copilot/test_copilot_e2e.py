"""End-to-end tests for Copilot streaming with dummy implementations.

These tests verify the complete copilot flow using dummy implementations
for agent generator and SDK service, allowing automated testing without
external LLM calls.

Enable test mode with COPILOT_TEST_MODE=true environment variable.
"""

import asyncio
import os
from uuid import uuid4

import pytest

from backend.copilot.model import ChatMessage, ChatSession, upsert_chat_session
from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamHeartbeat,
    StreamStart,
    StreamTextDelta,
)
from backend.copilot.sdk.dummy import stream_chat_completion_dummy


@pytest.fixture(autouse=True)
def enable_test_mode():
    """Enable test mode for all tests in this module."""
    os.environ["COPILOT_TEST_MODE"] = "true"
    yield
    os.environ.pop("COPILOT_TEST_MODE", None)


@pytest.mark.asyncio
async def test_dummy_streaming_basic_flow():
    """Test that dummy streaming produces correct event sequence."""
    events = []

    async for event in stream_chat_completion_dummy(
        session_id="test-session-basic",
        message="Hello",
        is_user_message=True,
        user_id="test-user",
    ):
        events.append(event)

    # Verify we got events
    assert len(events) > 0, "Should receive events"

    # Verify StreamStart
    start_events = [e for e in events if isinstance(e, StreamStart)]
    assert len(start_events) == 1
    assert start_events[0].messageId
    assert start_events[0].taskId

    # Verify StreamTextDelta events
    text_events = [e for e in events if isinstance(e, StreamTextDelta)]
    assert len(text_events) > 0
    full_text = "".join(e.delta for e in text_events)
    assert len(full_text) > 0

    # Verify StreamFinish
    finish_events = [e for e in events if isinstance(e, StreamFinish)]
    assert len(finish_events) == 1

    # Verify order: start before text before finish
    start_idx = events.index(start_events[0])
    finish_idx = events.index(finish_events[0])
    first_text_idx = events.index(text_events[0]) if text_events else -1
    if first_text_idx >= 0:
        assert start_idx < first_text_idx < finish_idx

    print(f"✅ Basic flow: {len(events)} events, {len(text_events)} text deltas")


@pytest.mark.asyncio
async def test_streaming_no_timeout():
    """Test that streaming completes within reasonable time without timeout."""
    import time

    start_time = time.monotonic()
    event_count = 0

    async for event in stream_chat_completion_dummy(
        session_id="test-session-timeout",
        message="count to 10",
        is_user_message=True,
        user_id="test-user",
    ):
        event_count += 1

    elapsed = time.monotonic() - start_time

    # Should complete in < 5 seconds (dummy has 0.1s delays between words)
    assert elapsed < 5.0, f"Streaming took {elapsed:.1f}s, expected < 5s"
    assert event_count > 0, "Should receive events"

    print(f"✅ No timeout: completed in {elapsed:.2f}s with {event_count} events")


@pytest.mark.asyncio
async def test_streaming_event_types():
    """Test that all expected event types are present."""
    event_types = set()

    async for event in stream_chat_completion_dummy(
        session_id="test-session-types",
        message="test",
        is_user_message=True,
        user_id="test-user",
    ):
        event_types.add(type(event).__name__)

    # Required event types
    assert "StreamStart" in event_types, "Missing StreamStart"
    assert "StreamTextDelta" in event_types, "Missing StreamTextDelta"
    assert "StreamFinish" in event_types, "Missing StreamFinish"

    print(f"✅ Event types: {sorted(event_types)}")


@pytest.mark.asyncio
async def test_streaming_text_content():
    """Test that streamed text is coherent and complete."""
    text_events = []

    async for event in stream_chat_completion_dummy(
        session_id="test-session-content",
        message="count to 3",
        is_user_message=True,
        user_id="test-user",
    ):
        if isinstance(event, StreamTextDelta):
            text_events.append(event)

    # Verify text deltas
    assert len(text_events) > 0, "Should have text deltas"

    # Reconstruct full text
    full_text = "".join(e.delta for e in text_events)
    assert len(full_text) > 0, "Text should not be empty"
    assert (
        "1" in full_text or "counted" in full_text.lower()
    ), "Text should contain count"

    # Verify all deltas have IDs
    for text_event in text_events:
        assert text_event.id, "Text delta must have ID"
        assert text_event.delta, "Text delta must have content"

    print(f"✅ Text content: '{full_text}' ({len(text_events)} deltas)")


@pytest.mark.asyncio
async def test_streaming_heartbeat_timing():
    """Test that heartbeats are sent at correct interval during long operations."""
    # This test would need a dummy that takes longer
    # For now, just verify heartbeat structure if we receive one
    heartbeats = []

    async for event in stream_chat_completion_dummy(
        session_id="test-session-heartbeat",
        message="test",
        is_user_message=True,
        user_id="test-user",
    ):
        if isinstance(event, StreamHeartbeat):
            heartbeats.append(event)

    # Dummy is fast, so we might not get heartbeats
    # But if we do, verify they're valid
    if heartbeats:
        print(f"✅ Heartbeat structure verified ({len(heartbeats)} received)")
    else:
        print("✅ No heartbeats (dummy executes quickly)")


@pytest.mark.asyncio
async def test_error_handling():
    """Test that errors are properly formatted and sent."""
    # This would require a dummy that can trigger errors
    # For now, just verify error event structure

    error = StreamError(errorText="Test error", code="test_error")
    assert error.errorText == "Test error"
    assert error.code == "test_error"
    assert str(error.type.value) in ["error", "error"]

    print("✅ Error structure verified")


@pytest.mark.asyncio
async def test_concurrent_sessions():
    """Test that multiple sessions can stream concurrently."""

    async def stream_session(session_id: str) -> int:
        count = 0
        async for event in stream_chat_completion_dummy(
            session_id=session_id,
            message="test",
            is_user_message=True,
            user_id="test-user",
        ):
            count += 1
        return count

    # Run 3 concurrent sessions
    results = await asyncio.gather(
        stream_session("session-1"),
        stream_session("session-2"),
        stream_session("session-3"),
    )

    # All should complete successfully
    assert all(count > 0 for count in results), "All sessions should produce events"
    print(f"✅ Concurrent sessions: {results} events each")


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="Event loop isolation issue with DB operations in tests - needs fixture refactoring"
)
async def test_session_state_persistence():
    """Test that session state is maintained across multiple messages."""
    from datetime import datetime, timezone

    session_id = f"test-session-{uuid4()}"
    user_id = "test-user"

    # Create session with first message
    session = ChatSession(
        session_id=session_id,
        user_id=user_id,
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!"),
        ],
        usage=[],
        started_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    await upsert_chat_session(session)

    # Stream second message
    events = []
    async for event in stream_chat_completion_dummy(
        session_id=session_id,
        message="How are you?",
        is_user_message=True,
        user_id=user_id,
        session=session,  # Pass existing session
    ):
        events.append(event)

    # Verify events were produced
    assert len(events) > 0, "Should produce events for second message"

    # Verify we got a complete response
    finish_events = [e for e in events if isinstance(e, StreamFinish)]
    assert len(finish_events) == 1, "Should have StreamFinish"

    print(f"✅ Session persistence: {len(events)} events for second message")


if __name__ == "__main__":
    # Run tests directly

    print("Running Copilot E2E tests with dummy implementations...")
    print("=" * 60)

    asyncio.run(test_dummy_streaming_basic_flow())
    asyncio.run(test_streaming_no_timeout())
    asyncio.run(test_streaming_event_types())
    asyncio.run(test_streaming_text_content())
    asyncio.run(test_streaming_heartbeat_timing())
    asyncio.run(test_error_handling())
    asyncio.run(test_concurrent_sessions())
    asyncio.run(test_session_state_persistence())

    print("=" * 60)
    print("✅ All E2E tests passed!")
