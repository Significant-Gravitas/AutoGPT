import asyncio
import json
from collections.abc import Iterator
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from claude_agent_sdk.types import HookContext, HookInput
from langfuse import Langfuse
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.trace.v1.trace_pb2 import Span
from opentelemetry.sdk.trace import TracerProvider
from requests import Response

from backend.copilot import pending_messages
from backend.copilot.context import set_execution_context
from backend.copilot.model import ChatSession
from backend.copilot.pending_messages import PendingMessage, PendingMessageContext
from backend.copilot.sdk.security_hooks import create_security_hooks


@pytest.fixture
def trace_export(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Langfuse, list[Span]]]:
    exported: list[Span] = []

    def receive(url: str, *, data: bytes, **_kwargs: object) -> Response:
        assert url.startswith("http://127.0.0.1:9/")
        request = ExportTraceServiceRequest.FromString(data)
        exported.extend(
            span
            for resource in request.resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        )
        response = Response()
        response.status_code = 200
        response._content = b""
        return response

    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    for key in (
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER",
        "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER",
        "OTEL_EXPORTER_OTLP_COMPRESSION",
        "OTEL_EXPORTER_OTLP_TRACES_COMPRESSION",
    ):
        monkeypatch.delenv(key, raising=False)
    provider = TracerProvider()
    with patch("requests.Session.post", side_effect=receive):
        client = Langfuse(
            public_key=f"pk-test-{uuid4()}",
            secret_key="sk-test",
            base_url="http://127.0.0.1:9",
            tracer_provider=provider,
            tracing_enabled=True,
        )
        monkeypatch.setattr(
            "backend.copilot.pending_message_tracing.get_client", lambda: client
        )
        try:
            yield client, exported
        finally:
            client.shutdown()
            provider.shutdown()


@pytest.fixture
def pending_redis(monkeypatch: pytest.MonkeyPatch) -> Iterator[AsyncMock]:
    redis = AsyncMock()
    redis.lpop.return_value = []
    monkeypatch.setattr(
        pending_messages, "get_redis_async", AsyncMock(return_value=redis)
    )
    set_execution_context(
        user_id="test-user",
        session=ChatSession.new("test-user", dry_run=True, session_id="test-session"),
    )
    try:
        yield redis
    finally:
        set_execution_context(user_id=None, session=None)


async def _post_tool_use(tool_name: str = "WebSearch", tool_use_id: str = "tool-1"):
    hook = create_security_hooks(user_id="test-user")["PostToolUse"][0].hooks[0]
    return await hook(
        cast(
            HookInput,
            {
                "hook_event_name": "PostToolUse",
                "session_id": "sdk-session",
                "transcript_path": "",
                "cwd": "",
                "tool_name": tool_name,
                "tool_input": {},
                "tool_response": "tool result",
                "tool_use_id": tool_use_id,
            },
        ),
        tool_use_id,
        cast(HookContext, {}),
    )


def _attributes(span: Span) -> dict[str, str]:
    return {
        item.key: (
            str(item.value.int_value)
            if item.value.HasField("int_value")
            else item.value.string_value
        )
        for item in span.attributes
    }


@pytest.mark.parametrize("tool_name", ["WebSearch", "mcp__copilot__run_block"])
async def test_consumed_followups_are_exported_as_ordered_user_messages(
    trace_export: tuple[Langfuse, list[Span]], pending_redis: AsyncMock, tool_name: str
) -> None:
    client, exported = trace_export
    pending = [
        PendingMessage(
            id="followup-1",
            content="Use the revised figures",
            enqueued_at=1000,
            file_ids=["file-1"],
            context=PendingMessageContext(url="https://example.com/report"),
        ),
        PendingMessage(
            id="followup-2", content="Keep the answer brief", enqueued_at=1001
        ),
    ]
    pending_redis.lpop.side_effect = [[item.model_dump_json() for item in pending], []]

    with client.start_as_current_span(name="copilot-sdk-turn") as turn:
        with client.start_as_current_span(name="tool"):
            pass
        result = await asyncio.create_task(_post_tool_use(tool_name))
        assert all(item.content in str(result) for item in pending)
        await asyncio.create_task(_post_tool_use(tool_name, "tool-2"))
        with client.start_as_current_span(name="assistant-continuation"):
            pass
    client.flush()

    tool = next(span for span in exported if span.name == "tool")
    continuation = next(
        span for span in exported if span.name == "assistant-continuation"
    )
    events = sorted(
        (span for span in exported if span.name == "copilot-user-follow-up"),
        key=lambda span: int(
            _attributes(span)["langfuse.observation.metadata.batch_index"]
        ),
    )
    assert len(events) == 2, "the hook consumed two messages but their trace has none"
    for index, (event, message) in enumerate(zip(events, pending), start=1):
        attrs = _attributes(event)
        assert event.trace_id.hex() == turn.trace_id
        assert event.parent_span_id.hex() == turn.id
        assert tool.end_time_unix_nano <= event.start_time_unix_nano
        assert event.end_time_unix_nano <= continuation.start_time_unix_nano
        assert attrs["langfuse.observation.type"] == "event"
        traced_message = json.loads(attrs["langfuse.observation.input"])[0]
        assert traced_message["role"] == "user"
        assert message.content in traced_message["content"]
        delivered = result["hookSpecificOutput"]["additionalContext"]
        assert f"Message {index}:\n{traced_message['content']}" in delivered
        assert attrs["langfuse.observation.metadata.source"] == "PostToolUse"
        assert attrs["langfuse.observation.metadata.message_id"] == message.id
        assert attrs["langfuse.observation.metadata.enqueued_at"] == str(
            message.enqueued_at
        )
        assert attrs["langfuse.observation.metadata.tool_use_id"] == "tool-1"
        assert attrs["langfuse.observation.metadata.tool_name"] == tool_name
        assert attrs["langfuse.observation.metadata.batch_index"] == str(index)
        assert (
            json.loads(attrs["langfuse.observation.metadata.file_ids"])
            == message.file_ids
        )
        if message.context:
            assert json.loads(attrs["langfuse.observation.metadata.context"]) == {
                "url": message.context.url
            }
    assert events[0].start_time_unix_nano <= events[1].start_time_unix_nano
    pending_redis.rpush.assert_awaited_once()


@pytest.mark.parametrize(
    "target",
    [
        "backend.copilot.pending_message_tracing.get_client",
        "langfuse.Langfuse.create_event",
    ],
)
async def test_trace_failure_does_not_prevent_delivery_or_persistence(
    trace_export: tuple[Langfuse, list[Span]], pending_redis: AsyncMock, target: str
) -> None:
    client, _ = trace_export
    message = PendingMessage(content="Still deliver this")
    pending_redis.lpop.return_value = [message.model_dump_json()]
    with (
        client.start_as_current_span(name="copilot-sdk-turn"),
        patch(target, side_effect=RuntimeError("tracing unavailable")) as failed,
    ):
        result = await _post_tool_use()
    assert message.content in str(result)
    pending_redis.rpush.assert_awaited_once()
    failed.assert_called_once()


async def test_persistence_failure_does_not_hide_consumed_message_from_trace(
    trace_export: tuple[Langfuse, list[Span]], pending_redis: AsyncMock
) -> None:
    client, exported = trace_export
    message = PendingMessage(content="Keep this in the trace")
    pending_redis.lpop.return_value = [message.model_dump_json()]
    pending_redis.rpush.side_effect = ConnectionError("persist queue unavailable")
    with client.start_as_current_span(name="copilot-sdk-turn"):
        result = await pending_messages.drain_and_format_for_injection(
            "test-session", log_prefix="[TEST]", source="between-rounds"
        )
    client.flush()
    assert message.content in str(result)
    event = next(span for span in exported if span.name == "copilot-user-follow-up")
    assert (
        _attributes(event)["langfuse.observation.metadata.source"] == "between-rounds"
    )
    assert json.loads(_attributes(event)["langfuse.observation.input"]) == [
        {"role": "user", "content": message.content}
    ]


@pytest.mark.parametrize("case", ["empty", "redis-failure", "no-session"])
async def test_no_followup_event_without_consumption(
    trace_export: tuple[Langfuse, list[Span]], pending_redis: AsyncMock, case: str
) -> None:
    client, exported = trace_export
    if case == "redis-failure":
        pending_redis.lpop.side_effect = ConnectionError("pending queue unavailable")
    elif case == "no-session":
        set_execution_context(user_id=None, session=None)
    with client.start_as_current_span(name="copilot-sdk-turn"):
        result = await _post_tool_use()
    client.flush()
    assert result == {}
    assert [span.name for span in exported] == ["copilot-sdk-turn"]
    pending_redis.rpush.assert_not_awaited()


async def test_no_orphan_trace_without_active_turn(
    trace_export: tuple[Langfuse, list[Span]], pending_redis: AsyncMock
) -> None:
    client, exported = trace_export
    message = PendingMessage(content="Tracing is optional")
    pending_redis.lpop.return_value = [message.model_dump_json()]
    result = await _post_tool_use()
    client.flush()
    assert message.content in str(result)
    assert exported == []


async def test_concurrent_turns_keep_followups_on_their_own_trace(
    trace_export: tuple[Langfuse, list[Span]], pending_redis: AsyncMock
) -> None:
    client, exported = trace_export
    messages = {key: PendingMessage(content=key) for key in ("session-a", "session-b")}

    async def pop(key: str, count: int) -> list[str]:
        await asyncio.sleep(0)
        return [
            message.model_dump_json()
            for session_id, message in messages.items()
            if key == pending_messages._buffer_key(session_id)
        ]

    pending_redis.lpop.side_effect = pop

    async def consume(session_id: str) -> tuple[str, str]:
        set_execution_context(
            user_id="test-user",
            session=ChatSession.new("test-user", dry_run=True, session_id=session_id),
        )
        with client.start_as_current_span(name="copilot-sdk-turn") as turn:
            result = await _post_tool_use(tool_use_id=session_id)
            assert messages[session_id].content in str(result)
        return session_id, turn.trace_id

    trace_ids = dict(await asyncio.gather(*(consume(key) for key in messages)))
    client.flush()
    events = [span for span in exported if span.name == "copilot-user-follow-up"]
    assert len(events) == 2
    assert len(set(trace_ids.values())) == 2
    for event in events:
        attrs = _attributes(event)
        session_id = attrs["langfuse.observation.metadata.tool_use_id"]
        assert event.trace_id.hex() == trace_ids[session_id]
        assert (
            attrs["langfuse.observation.metadata.message_id"] == messages[session_id].id
        )
