"""Tests for collect_copilot_response stream registry integration."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
    StreamUsage,
)
from backend.copilot.sdk.collect import collect_copilot_response


def _mock_stream_fn(*events):
    """Return a callable that returns an async generator."""

    async def _gen(**_kwargs):
        for e in events:
            yield e

    return _gen


@pytest.fixture
def mock_registry():
    """Patch stream_registry module used by collect."""
    with patch("backend.copilot.sdk.collect.stream_registry") as m:
        m.create_session = AsyncMock()
        m.publish_chunk = AsyncMock()
        m.mark_session_completed = AsyncMock()

        # stream_and_publish: pass-through that also publishes (real logic)
        # We re-implement the pass-through here so the event loop works,
        # but still track publish_chunk calls via the mock.
        async def _stream_and_publish(session_id, turn_id, stream):
            async for event in stream:
                if turn_id and not isinstance(event, (StreamFinish, StreamError)):
                    await m.publish_chunk(turn_id, event)
                yield event

        m.stream_and_publish = _stream_and_publish
        yield m


@pytest.fixture
def stream_fn_patch():
    """Helper to patch stream_chat_completion_sdk."""

    def _patch(events):
        return patch(
            "backend.copilot.sdk.collect.stream_chat_completion_sdk",
            new=_mock_stream_fn(*events),
        )

    return _patch


@pytest.mark.asyncio
async def test_stream_registry_called_on_success(mock_registry, stream_fn_patch):
    """Stream registry create/publish/complete are called correctly on success."""
    events = [
        StreamTextDelta(id="t1", delta="Hello "),
        StreamTextDelta(id="t1", delta="world"),
        StreamUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        StreamFinish(),
    ]

    with stream_fn_patch(events):
        result = await collect_copilot_response(
            session_id="test-session",
            message="hi",
            user_id="user-1",
        )

    assert result.response_text == "Hello world"
    assert result.total_tokens == 15

    mock_registry.create_session.assert_awaited_once()
    # StreamFinish should NOT be published (mark_session_completed does it)
    published_types = [
        type(call.args[1]).__name__
        for call in mock_registry.publish_chunk.call_args_list
    ]
    assert "StreamFinish" not in published_types
    assert "StreamTextDelta" in published_types

    mock_registry.mark_session_completed.assert_awaited_once()
    _, kwargs = mock_registry.mark_session_completed.call_args
    assert kwargs.get("error_message") is None


@pytest.mark.asyncio
async def test_stream_registry_error_on_stream_error(mock_registry, stream_fn_patch):
    """mark_session_completed receives error message when StreamError occurs."""
    events = [
        StreamTextDelta(id="t1", delta="partial"),
        StreamError(errorText="something broke"),
    ]

    with stream_fn_patch(events):
        with pytest.raises(RuntimeError, match="something broke"):
            await collect_copilot_response(
                session_id="test-session",
                message="hi",
                user_id="user-1",
            )

    _, kwargs = mock_registry.mark_session_completed.call_args
    assert kwargs.get("error_message") == "something broke"
    # stream_and_publish skips StreamError, so mark_session_completed must
    # publish it (skip_error_publish=False).
    assert kwargs.get("skip_error_publish") is False

    # StreamError should NOT be published via publish_chunk — mark_session_completed
    # handles it to avoid double-publication.
    published_types = [
        type(call.args[1]).__name__
        for call in mock_registry.publish_chunk.call_args_list
    ]
    assert "StreamError" not in published_types


@pytest.mark.asyncio
async def test_graceful_degradation_when_create_session_fails(
    mock_registry, stream_fn_patch
):
    """AutoPilot still works when stream registry create_session raises."""
    events = [
        StreamTextDelta(id="t1", delta="works"),
        StreamFinish(),
    ]
    mock_registry.create_session = AsyncMock(side_effect=ConnectionError("Redis down"))

    with stream_fn_patch(events):
        result = await collect_copilot_response(
            session_id="test-session",
            message="hi",
            user_id="user-1",
        )

    assert result.response_text == "works"
    # publish_chunk should NOT be called because turn_id was cleared
    mock_registry.publish_chunk.assert_not_awaited()
    # mark_session_completed IS still called to clean up any partial state
    mock_registry.mark_session_completed.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_calls_published_and_collected(mock_registry, stream_fn_patch):
    """Tool call events are both published to registry and collected in result."""
    events = [
        StreamToolInputAvailable(
            toolCallId="tc-1", toolName="read_file", input={"path": "/tmp"}
        ),
        StreamToolOutputAvailable(
            toolCallId="tc-1", output="file contents", success=True
        ),
        StreamTextDelta(id="t1", delta="done"),
        StreamFinish(),
    ]

    with stream_fn_patch(events):
        result = await collect_copilot_response(
            session_id="test-session",
            message="hi",
            user_id="user-1",
        )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["tool_name"] == "read_file"
    assert result.tool_calls[0]["output"] == "file contents"
    assert result.tool_calls[0]["success"] is True
    assert result.response_text == "done"
