import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any
from unittest.mock import patch

import claude_agent_sdk
import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultMessage,
    StreamEvent,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    UserMessage,
)
from langsmith import trace, tracing_context
from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk
from langsmith.run_trees import RunTree

from .conversation_output import ConversationOutputClient
from .service import _iter_sdk_messages


@pytest.fixture
def script() -> list[Message | BaseException | asyncio.Event]:
    return []


@pytest.fixture
def blocked() -> asyncio.Event:
    return asyncio.Event()


@pytest.fixture
def client(
    script, blocked, monkeypatch: pytest.MonkeyPatch
) -> ConversationOutputClient:
    async def receive_messages(_: object) -> AsyncIterator[Message]:
        for item in script:
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, asyncio.Event):
                blocked.set()
                await item.wait()
            else:
                yield item

    class ScriptedClient(ConversationOutputClient):
        async def query(self, prompt: str) -> None:
            return None

    monkeypatch.setattr(
        claude_agent_sdk.ClaudeSDKClient, "receive_messages", receive_messages
    )
    with (
        patch.object(claude_agent_sdk, "ClaudeSDKClient", ScriptedClient),
        patch.object(claude_agent_sdk, "SdkMcpTool", None),
        patch("langsmith.integrations.claude_agent_sdk.set_tracing_config"),
    ):
        assert configure_claude_agent_sdk()
    return ScriptedClient(options=ClaudeAgentOptions())


@pytest.fixture
def exported(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []

    def capture(run: RunTree, **_: object) -> None:
        if run.name == "claude.conversation":
            snapshots.append(
                {
                    "outputs": dict(run.outputs or {}),
                    "error": run.error,
                    "ended": run.end_time is not None,
                }
            )

    monkeypatch.setattr(RunTree, "post", lambda *_, **__: None)
    monkeypatch.setattr(RunTree, "patch", capture)
    return snapshots


def assistant(text: str, message_id: str = "msg-1", **kwargs) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)], model="test", message_id=message_id, **kwargs
    )


def result(**kwargs) -> ResultMessage:
    return ResultMessage(
        **{
            "subtype": "success",
            "duration_ms": 1,
            "duration_api_ms": 1,
            "num_turns": 1,
            "is_error": False,
            "session_id": "session",
            **kwargs,
        }
    )


def event(kind: str, **kwargs) -> StreamEvent:
    return StreamEvent(
        uuid="event", session_id="session", event={"type": kind, **kwargs}
    )


async def consume(client: ConversationOutputClient) -> None:
    await client.query("test prompt")
    async with aclosing(_iter_sdk_messages(client)) as messages:
        async for message in messages:
            if isinstance(message, ResultMessage):
                break


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages,expected_text,expected_reason",
    [
        ([assistant("done"), result()], "done", "end_turn"),
        (
            [result(result="result-only answer", stop_reason="stop_sequence")],
            "result-only answer",
            "stop_sequence",
        ),
        ([assistant("draft"), result(result="final")], "final", "end_turn"),
        (
            [assistant("first"), assistant("final", "msg-2"), result()],
            "final",
            "end_turn",
        ),
        ([assistant("part "), assistant("two"), result()], "part two", "end_turn"),
        (
            [
                assistant("main"),
                assistant("child", parent_tool_use_id="tool-1"),
                result(),
            ],
            "main",
            "end_turn",
        ),
        (
            [
                assistant("main"),
                UserMessage(
                    content=[
                        ToolResultBlock(tool_use_id="tool-1", content="tool output")
                    ]
                ),
                result(),
            ],
            "main",
            "end_turn",
        ),
        (
            [
                AssistantMessage(
                    content=[
                        ThinkingBlock(thinking="private reasoning", signature="sig")
                    ],
                    model="test",
                ),
                result(),
            ],
            "",
            "end_turn",
        ),
        (
            [
                assistant("partial"),
                result(is_error=True, subtype="error_max_turns", result="error detail"),
            ],
            "partial",
            "error_max_turns",
        ),
        ([assistant("partial"), result(is_error=True)], "partial", "error"),
        (
            [assistant("partial"), result(terminal_reason="aborted_streaming")],
            "partial",
            "cancelled",
        ),
        (
            [assistant("partial"), result(terminal_reason="aborted_tools")],
            "partial",
            "cancelled",
        ),
        ([assistant("partial"), RuntimeError("transport failed")], "partial", "error"),
        ([RuntimeError("transport failed")], "", "error"),
        ([assistant("partial")], "partial", "error"),
    ],
)
async def test_exported_conversation_output(
    client, script, exported, messages, expected_text, expected_reason
) -> None:
    script.extend(messages)
    with tracing_context(enabled=True):
        await consume(client)
    assert len(exported) == 1
    assert exported[0]["outputs"] == {
        "output": expected_text,
        "stop_reason": expected_reason,
    }
    assert exported[0]["ended"]


@pytest.mark.asyncio
@pytest.mark.parametrize("with_summary", [False, True])
async def test_partial_stream_text_is_not_duplicated(
    client, script, exported, with_summary
):
    script.extend(
        [
            event("message_start", message={"id": "msg-1"}),
            event(
                "content_block_start",
                index=0,
                content_block={"type": "text", "text": "hello "},
            ),
            event(
                "content_block_delta",
                index=0,
                delta={"type": "text_delta", "text": "world"},
            ),
            event("message_delta", delta={"stop_reason": "max_tokens"}),
        ]
    )
    if with_summary:
        script.append(assistant("hello world", stop_reason="max_tokens"))
    script.append(result())
    with tracing_context(enabled=True):
        await consume(client)
    assert exported[0]["outputs"] == {
        "output": "hello world",
        "stop_reason": "max_tokens",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("with_text", [False, True])
async def test_cancellation_exports_before_propagating(
    client, script, blocked, exported, with_text
):
    if with_text:
        script.extend(
            [
                event("message_start", message={"id": "msg-1"}),
                event(
                    "content_block_delta",
                    index=0,
                    delta={"type": "text_delta", "text": "partial"},
                ),
            ]
        )
    script.append(asyncio.Event())
    with tracing_context(enabled=True):
        pending = asyncio.create_task(consume(client))
        await asyncio.wait_for(blocked.wait(), timeout=5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
    assert len(exported) == 1
    assert exported[0]["outputs"] == {
        "output": "partial" if with_text else "",
        "stop_reason": "cancelled",
    }
    assert exported[0]["ended"]


@pytest.mark.asyncio
async def test_close_between_messages_exports_immediately(client, script, exported):
    script.extend([assistant("partial"), asyncio.Event()])
    with tracing_context(enabled=True):
        stream = _iter_sdk_messages(client)
        await anext(stream)
        await stream.aclose()
    assert len(exported) == 1
    assert exported[0]["outputs"] == {"output": "partial", "stop_reason": "cancelled"}


@pytest.mark.asyncio
async def test_reused_client_keeps_turn_outputs_separate(client, script, exported):
    with tracing_context(enabled=True):
        for text in ("first", "second"):
            script[:] = [assistant(text), result()]
            await consume(client)
    assert [snapshot["outputs"] for snapshot in exported] == [
        {"output": "first", "stop_reason": "end_turn"},
        {"output": "second", "stop_reason": "end_turn"},
    ]


@pytest.mark.asyncio
async def test_uninstrumented_client_does_not_change_parent_output(monkeypatch):
    async def receive_messages(_: object) -> AsyncIterator[Message]:
        yield assistant("done")
        yield result()

    monkeypatch.setattr(
        claude_agent_sdk.ClaudeSDKClient, "receive_messages", receive_messages
    )
    client = ConversationOutputClient()
    with tracing_context(enabled="local"), trace("unrelated") as parent:
        parent.set(outputs={"keep": "this"})
        assert len([message async for message in client.receive_response()]) == 2
        assert parent.outputs == {"keep": "this"}
