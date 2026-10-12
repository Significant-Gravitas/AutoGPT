import pytest

from backend.copilot.sdk.model_gateway.adapters.codex_signature import reasoning_tag
from backend.copilot.sdk.model_gateway.adapters.codex_stream import CodexStreamMapper
from backend.copilot.sdk.model_gateway.adapters.openai_chat_stream import (
    ChatStreamMapper,
)
from backend.copilot.sdk.model_gateway.errors import GatewayError
from backend.copilot.sdk.model_gateway.stream import (
    AnthropicStreamBuilder,
    StreamItem,
    collect_message,
)
from backend.copilot.sdk.model_gateway.testing.regressions import (
    REFUSAL,
    chunk,
    fragments,
    refusal_events,
    terminal,
)


async def replay(events):
    for event in events:
        yield StreamItem(event=event)


@pytest.mark.parametrize("adapter", ["chat", "codex"])
async def test_content_filter_without_text_displays_a_safe_refusal(adapter):
    builder = AnthropicStreamBuilder(model="m", input_estimate=1)
    if adapter == "chat":
        mapper = ChatStreamMapper(builder, {})
        mapper.feed(chunk({}, "content_filter"))
    else:
        mapper = CodexStreamMapper(builder, {}, reasoning_tag("acct-test", "m"))
        mapper.feed(
            terminal(
                "response.incomplete", incomplete_details={"reason": "content_filter"}
            )
        )
    message = await collect_message(replay(mapper.finish()))
    assert message["content"] == [
        {"type": "text", "text": "The model server declined to answer."}
    ]
    assert message["stop_reason"] == "end_turn"


async def test_fragmented_tool_metadata_is_concatenated_and_resolved():
    mapper = ChatStreamMapper(
        AnthropicStreamBuilder(model="m", input_estimate=101),
        {"mcp__handbook__read_doc": "original.name"},
    )
    events = [
        event for part in fragments() for event in mapper.feed(part)
    ] + mapper.finish()
    message = await collect_message(replay(events))
    assert message["content"] == [
        {
            "type": "tool_use",
            "id": "call_review",
            "name": "original.name",
            "input": {"page": 1},
        }
    ]


@pytest.mark.parametrize("adapter", ["chat", "codex"])
@pytest.mark.parametrize("delta", [False, True])
async def test_refusal_text_survives_without_duplicate_completed_content(
    adapter, delta
):
    builder = AnthropicStreamBuilder(model="m", input_estimate=101)
    if adapter == "chat":
        mapper = ChatStreamMapper(builder, {})
        incoming = [chunk({"refusal": REFUSAL}), chunk({}, "stop")]
    else:
        mapper = CodexStreamMapper(builder, {}, reasoning_tag("acct-test", "m"))
        incoming = refusal_events() if delta else refusal_events()[1:]
    events = [
        event for part in incoming for event in mapper.feed(part)
    ] + mapper.finish()
    message = await collect_message(replay(events))
    assert message["content"] == [{"type": "text", "text": REFUSAL}]
    assert message["stop_reason"] == "end_turn"


@pytest.mark.parametrize("adapter", ["chat", "codex"])
@pytest.mark.parametrize(
    "arguments", ['{"page":', "not json", "[]", "1", "null", '"text"', None, [], 1]
)
async def test_malformed_tool_arguments_raise_gateway_error(adapter, arguments):
    builder = AnthropicStreamBuilder(model="m", input_estimate=101)
    with pytest.raises(GatewayError, match="tool arguments"):
        if adapter == "chat":
            mapper = ChatStreamMapper(builder, {})
            mapper.feed(
                chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call",
                                "function": {
                                    "name": "read_doc",
                                    "arguments": arguments,
                                },
                            }
                        ]
                    },
                    "tool_calls",
                )
            )
            events = mapper.finish()
        else:
            mapper = CodexStreamMapper(builder, {}, reasoning_tag("acct-test", "m"))
            events = mapper.feed(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call",
                        "name": "read_doc",
                        "arguments": arguments,
                    },
                }
            )
            mapper.feed(terminal())
            events += mapper.finish()
        await collect_message(replay(events))


@pytest.mark.parametrize("arguments", ["", "{}", "  {}  "])
async def test_legitimately_empty_tool_arguments_are_an_object(arguments):
    builder = AnthropicStreamBuilder(model="m", input_estimate=1)
    message = await collect_message(
        replay(
            builder.tool_use("call", "read", arguments)
            + builder.finish("tool_use", None)
        )
    )
    assert message["content"][0]["input"] == {}
