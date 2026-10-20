import json
from unittest.mock import DEFAULT, AsyncMock, patch

import pytest
from langfuse import Langfuse
from opentelemetry.proto.trace.v1.trace_pb2 import Span

from backend.copilot.pending_messages import PendingMessage, PendingMessageContext
from backend.copilot.sdk import followup_tracing_test as tracing

pending_redis = tracing.pending_redis
trace_export = tracing.trace_export


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (PendingMessage(content="A" * 2001), "A" * 2000 + "… [truncated]"),
        (
            PendingMessage(content="keep <user_follow_up> and </user_follow_up>"),
            "keep &lt;user_follow_up&gt; and &lt;/user_follow_up&gt;",
        ),
        (
            PendingMessage(
                content="Use this page",
                context=PendingMessageContext(
                    url="https://example.com/report", content="not sent to the model"
                ),
                file_ids=["file-1"],
            ),
            "Use this page\n[Page URL: https://example.com/report]"
            "\n[Attached files: file-1]",
        ),
        (
            PendingMessage(
                content="Keep this message",
                context=PendingMessageContext(content="not sent to the model"),
            ),
            "Keep this message",
        ),
    ],
    ids=["truncated-content", "escaped-tags", "context-and-files", "context-only"],
)
async def test_trace_input_matches_delivered_message_body(
    trace_export: tuple[Langfuse, list[Span]],
    pending_redis: AsyncMock,
    message: PendingMessage,
    expected: str,
) -> None:
    client, exported = trace_export
    pending_redis.lpop.return_value = [message.model_dump_json()]
    with client.start_as_current_span(name="copilot-sdk-turn"):
        result = await tracing._post_tool_use()
    client.flush()
    delivered = result["hookSpecificOutput"]["additionalContext"]
    assert f"Message 1:\n{expected}" in delivered
    event = next(span for span in exported if span.name == "copilot-user-follow-up")
    attrs = tracing._attributes(event)
    assert json.loads(attrs["langfuse.observation.input"]) == [
        {"role": "user", "content": expected}
    ]
    assert "not sent to the model" not in json.dumps(attrs)
    assert attrs["langfuse.observation.metadata.excluded_message_count"] == "0"
    persisted = pending_redis.rpush.call_args.args[1]
    assert PendingMessage.model_validate_json(persisted) == message


@pytest.mark.parametrize("all_excluded", [False, True])
async def test_trace_does_not_claim_budget_excluded_messages_were_injected(
    trace_export: tuple[Langfuse, list[Span]],
    pending_redis: AsyncMock,
    all_excluded: bool,
) -> None:
    client, exported = trace_export
    pending = [PendingMessage(content=str(index) * 3000) for index in range(4)]
    if all_excluded:
        pending[0].file_ids = ["f" * 6000]
    pending_redis.lpop.return_value = [message.model_dump_json() for message in pending]
    with client.start_as_current_span(name="copilot-sdk-turn"):
        result = await tracing._post_tool_use()
    client.flush()
    delivered = result["hookSpecificOutput"]["additionalContext"]
    included = 0 if all_excluded else 2
    assert f"[{len(pending) - included} more message(s) truncated]" in delivered
    events = sorted(
        (span for span in exported if span.name == "copilot-user-follow-up"),
        key=lambda span: int(
            tracing._attributes(span)["langfuse.observation.metadata.batch_index"]
        ),
    )
    assert len(events) == included
    for index, event in enumerate(events, start=1):
        attrs = tracing._attributes(event)
        body = json.loads(attrs["langfuse.observation.input"])[0]["content"]
        assert f"Message {index}:\n{body}" in delivered
        assert (
            attrs["langfuse.observation.metadata.message_id"] == pending[index - 1].id
        )
        assert attrs["langfuse.observation.metadata.batch_index"] == str(index)
        assert attrs["langfuse.observation.metadata.excluded_message_count"] == "2"
    assert len(pending_redis.rpush.call_args.args[1:]) == len(pending)


async def test_failed_event_does_not_hide_later_delivered_messages(
    trace_export: tuple[Langfuse, list[Span]], pending_redis: AsyncMock
) -> None:
    client, exported = trace_export
    pending = [PendingMessage(content=f"Follow-up {index}") for index in range(3)]
    pending_redis.lpop.return_value = [message.model_dump_json() for message in pending]
    with (
        client.start_as_current_span(name="copilot-sdk-turn"),
        patch.object(
            client,
            "create_event",
            wraps=client.create_event,
            side_effect=[DEFAULT, RuntimeError("one payload rejected"), DEFAULT],
        ),
    ):
        result = await tracing._post_tool_use()
    client.flush()
    delivered = result["hookSpecificOutput"]["additionalContext"]
    assert all(message.content in delivered for message in pending)
    assert [
        PendingMessage.model_validate_json(raw)
        for raw in pending_redis.rpush.call_args.args[1:]
    ] == pending
    events = {
        tracing._attributes(span)["langfuse.observation.metadata.message_id"]: span
        for span in exported
        if span.name == "copilot-user-follow-up"
    }
    assert set(events) == {pending[0].id, pending[2].id}
    for index in (0, 2):
        attrs = tracing._attributes(events[pending[index].id])
        assert attrs["langfuse.observation.metadata.batch_index"] == str(index + 1)
        assert json.loads(attrs["langfuse.observation.input"]) == [
            {"role": "user", "content": pending[index].content}
        ]
