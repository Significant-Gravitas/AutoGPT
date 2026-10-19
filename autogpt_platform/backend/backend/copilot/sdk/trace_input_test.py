import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from langfuse import Langfuse
from langsmith import tracing_context
from langsmith.run_trees import RunTree
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)

from backend.copilot.model import ChatSession
from backend.copilot.response_model import StreamTextDelta, StreamToolInputAvailable
from backend.copilot.sdk.compaction import CompactionTracker
from backend.copilot.sdk.response_adapter import SDKResponseAdapter
from backend.copilot.sdk.service import (
    PreparedAttachments,
    _RetryState,
    _run_stream_attempt,
    _StreamContext,
    _TokenUsage,
)
from backend.copilot.sdk.trace_input import record_turn_input
from backend.copilot.sdk.transcript_builder import TranscriptBuilder

from .langsmith_tracing_test import _ScriptedClient, posted_runs, traced_client
from .trace_export_test import (
    _TraceServer,
    langfuse_client,
    trace_export_environment,
    trace_receiver,
)

__all__ = [
    "langfuse_client",
    "posted_runs",
    "trace_export_environment",
    "trace_receiver",
    "traced_client",
]

_SVC = "backend.copilot.sdk.service"
_PROMPT = 'What is in these pictures?\n日本語 "test"'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("image_count", "file_count"),
    [(0, 0), (1, 1), (2, 2), (1, 2), (0, 1)],
    ids=["text-only", "one-image", "two-images", "mixed-files", "document-only"],
)
async def test_turn_exports_prompt_and_references_without_image_bytes(
    image_count: int,
    file_count: int,
    traced_client: type[_ScriptedClient],
    posted_runs: list[RunTree],
    trace_receiver: _TraceServer,
    langfuse_client: Langfuse,
    tmp_path,
) -> None:
    trace_receiver.fail_requests = 0
    now = datetime.now(UTC)
    session = ChatSession(
        session_id="sess-1",
        user_id="user-1",
        usage=[],
        started_at=now,
        updated_at=now,
        messages=[],
    )
    file_ids = [f"file-{index}" for index in range(file_count)]
    image_blocks = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": f"private-image-payload-{index}",
            },
        }
        for index in range(image_count)
    ]
    ctx = _StreamContext(
        session=session,
        session_id=session.session_id,
        log_prefix="[trace-test]",
        sdk_cwd=str(tmp_path),
        current_message=_PROMPT,
        file_ids=file_ids,
        message_id="reply-1",
        attachments=PreparedAttachments(image_blocks=image_blocks),
        compaction=CompactionTracker(),
        lock=MagicMock(refresh=AsyncMock()),
    )
    state = _RetryState(
        options=ClaudeAgentOptions(),
        query_message=_PROMPT,
        compaction_stats=None,
        use_resume=False,
        resume_file=None,
        transcript_msg_count=0,
        adapter=SDKResponseAdapter(message_id="reply-1", session_id=session.session_id),
        transcript_builder=TranscriptBuilder(),
        usage=_TokenUsage(),
    )
    client = traced_client(options=state.options)
    with (
        patch(f"{_SVC}.ClaudeSDKClient", return_value=client),
        patch(
            "backend.copilot.sdk.trace_input.get_client", return_value=langfuse_client
        ),
        patch(f"{_SVC}.drain_pending_for_persist", new=AsyncMock(return_value=[])),
        patch(f"{_SVC}.upsert_chat_session", new=AsyncMock()),
        tracing_context(enabled=True, client=MagicMock()),
        langfuse_client.start_as_current_span(name="copilot-sdk-turn") as span,
    ):
        events = [event async for event in _run_stream_attempt(ctx, state)]

    langfuse_client.flush()
    exported = [
        item
        for body in trace_receiver.accepted
        for resource in ExportTraceServiceRequest.FromString(body).resource_spans
        for scope in resource.scope_spans
        for item in scope.spans
    ]
    root = next(item for item in exported if item.name == "copilot-sdk-turn")
    attributes = {item.key: item.value.string_value for item in root.attributes}
    assert b"What is in these pictures?" in b"".join(
        trace_receiver.bodies
    ), "The user prompt is missing from the exported trace"
    expected: dict = {"prompt": _PROMPT}
    if file_ids:
        expected["attachments"] = [f"workspace://{file_id}" for file_id in file_ids]
    assert json.loads(attributes.get("langfuse.trace.input", "null")) == expected
    assert json.loads(attributes["langfuse.observation.input"]) == expected
    assert root.trace_id.hex() == span.trace_id
    assert b"private-image-payload" not in b"".join(trace_receiver.bodies)
    assert "private-image-payload" not in str([run.inputs for run in posted_runs])

    assert any(isinstance(event, StreamToolInputAvailable) for event in events)
    assert (
        "".join(event.delta for event in events if isinstance(event, StreamTextDelta))
        == "done"
    )
    assert len([run for run in posted_runs if run.name == "claude.assistant.turn"]) == 2
    if image_count:
        client._transport.write.assert_awaited_once()
        sent = json.loads(client._transport.write.call_args.args[0])
        assert sent["message"]["content"] == [
            *image_blocks,
            {"type": "text", "text": _PROMPT},
        ]
        assert sent["session_id"] == session.session_id
    else:
        conversation = next(
            run for run in posted_runs if run.name == "claude.conversation"
        )
        assert conversation.inputs["prompt"] == _PROMPT
    assert image_blocks == ctx.attachments.image_blocks


@pytest.mark.parametrize(
    "operation", ["get_client", "update_current_trace", "update_current_span"]
)
def test_tracing_errors_do_not_interrupt_the_turn(operation: str) -> None:
    with patch("backend.copilot.sdk.trace_input.get_client") as get_client:
        failing_call = {
            "get_client": get_client,
            "update_current_trace": get_client.return_value.update_current_trace,
            "update_current_span": get_client.return_value.update_current_span,
        }[operation]
        failing_call.side_effect = RuntimeError("tracing unavailable")
        record_turn_input(_PROMPT, ["image-1"])
        failing_call.assert_called_once()


def test_no_active_span_does_not_create_a_trace(
    langfuse_client: Langfuse, trace_receiver: _TraceServer
) -> None:
    with patch(
        "backend.copilot.sdk.trace_input.get_client", return_value=langfuse_client
    ):
        record_turn_input(_PROMPT, ["image-1"])
    langfuse_client.flush()
    assert trace_receiver.bodies == []


def test_disabled_tracing_does_not_export_input(trace_receiver: _TraceServer) -> None:
    client = Langfuse(tracing_enabled=False)
    with (
        patch("backend.copilot.sdk.trace_input.get_client", return_value=client),
        client.start_as_current_span(name="disabled"),
    ):
        record_turn_input(_PROMPT, ["image-1"])
    client.flush()
    assert trace_receiver.bodies == []
