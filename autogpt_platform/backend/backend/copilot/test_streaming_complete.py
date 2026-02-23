"""Complete streaming test - validates all events are published correctly."""

import asyncio

import pytest

from backend.copilot.response_model import StreamFinish, StreamStart, StreamTextDelta
from backend.copilot.sdk.dummy import stream_chat_completion_dummy


@pytest.mark.asyncio
async def test_dummy_streaming_yields_all_events():
    """Test that dummy streaming yields start, text deltas, and finish."""
    events = []

    async for event in stream_chat_completion_dummy(
        session_id="test-session",
        message="count to 3",
        is_user_message=True,
        user_id="test-user",
    ):
        events.append(event)
        print(f"Event: {type(event).__name__} - {event}")

    # Verify we got events
    assert len(events) > 0, "Should receive at least one event"

    # Verify START event
    start_events = [e for e in events if isinstance(e, StreamStart)]
    assert len(start_events) == 1, f"Expected 1 StreamStart, got {len(start_events)}"
    assert start_events[0].messageId, "StreamStart should have messageId"
    assert start_events[0].taskId, "StreamStart should have taskId"

    # Verify TEXT DELTA events
    text_events = [e for e in events if isinstance(e, StreamTextDelta)]
    assert len(text_events) > 0, f"Expected text deltas, got {len(text_events)}"
    print(f"\n✓ Received {len(text_events)} text delta events")

    # Verify all text deltas have required fields
    for text_event in text_events:
        assert text_event.id, "StreamTextDelta must have id"
        assert text_event.delta, "StreamTextDelta must have delta text"
        # Type is an enum, check the string value
        assert str(text_event.type.value) in [
            "text_delta",
            "text-delta",
        ], f"Type should be text_delta, got {text_event.type}"

    # Verify text content
    full_text = "".join(e.delta for e in text_events)
    print(f"✓ Full text: {full_text!r}")
    assert (
        "counted" in full_text.lower() or "1" in full_text
    ), "Text should contain count content"

    # Verify FINISH event
    finish_events = [e for e in events if isinstance(e, StreamFinish)]
    assert len(finish_events) == 1, f"Expected 1 StreamFinish, got {len(finish_events)}"

    # Verify event order
    start_idx = events.index(start_events[0])
    first_text_idx = events.index(text_events[0])
    finish_idx = events.index(finish_events[0])
    assert (
        start_idx < first_text_idx < finish_idx
    ), "Events should be in order: start -> text -> finish"

    print(f"\n✅ All {len(events)} events validated successfully!")
    print("   - 1 StreamStart")
    print(f"   - {len(text_events)} StreamTextDelta")
    print("   - 1 StreamFinish")


if __name__ == "__main__":
    asyncio.run(test_dummy_streaming_yields_all_events())
