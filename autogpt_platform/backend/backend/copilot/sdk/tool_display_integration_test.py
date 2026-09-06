import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from claude_agent_sdk import AssistantMessage, ToolUseBlock, create_sdk_mcp_server, tool
from mcp.shared.memory import create_connected_server_and_client_session

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import (
    StreamToolDisplayAvailable,
    StreamToolInputAvailable,
    ToolDisplayData,
)
from backend.copilot.sdk.response_adapter import SDKResponseAdapter
from backend.copilot.sdk.security_hooks import create_security_hooks
from backend.copilot.sdk.service import (
    _dispatch_response,
    _format_sdk_content_blocks,
    _iter_sdk_messages,
    _StreamAccumulator,
)
from backend.copilot.sdk.tool_adapter import (
    _build_input_schema,
    _make_truncating_wrapper,
    pop_pending_tool_output,
    set_execution_context,
)
from backend.copilot.sdk.tool_display import SDKToolDisplayBridge, get_sdk_tool_call_id
from backend.copilot.tool_display import emit_tool_display_name
from backend.copilot.tools import TOOL_REGISTRY


def make_session():
    now = datetime.now(timezone.utc)
    return ChatSession(
        session_id="session",
        user_id="user",
        messages=[],
        usage=[],
        started_at=now,
        updated_at=now,
    )


@pytest.mark.parametrize(
    "tool_name,args,display_name,name_key",
    [
        (
            "run_agent",
            {"library_agent_id": "agent"},
            "Actual Library Title",
            "graph_name",
        ),
        (
            "run_block",
            {"block_id": "db7d8f02-2f44-4c55-ab7a-eae0941f0c30", "input_data": {}},
            "FillTextTemplateBlock",
            "block_name",
        ),
        (
            "continue_run_block",
            {"review_id": "review"},
            "FillTextTemplateBlock",
            "block_name",
        ),
    ],
)
@pytest.mark.asyncio
async def test_hook_to_mcp_handler_keeps_provider_identity_and_clean_stash(
    tool_name, args, display_name, name_key
):
    bridge = SDKToolDisplayBridge()
    hooks = create_security_hooks("user", tool_display_bridge=bridge)
    pre_hook = hooks["PreToolUse"][0].hooks[0]
    decision = await pre_hook(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": f"mcp__copilot__{tool_name}",
            "tool_input": args,
            "session_id": "session",
            "transcript_path": "",
            "cwd": "",
            "tool_use_id": "toolu-provider",
        },
        "toolu-provider",
        {"signal": None},
    )
    tagged = decision["hookSpecificOutput"]["updatedInput"]
    session = make_session()
    set_execution_context("user", session)

    async def execute(clean_args):
        assert clean_args == args
        assert get_sdk_tool_call_id() == "toolu-provider"
        emit_tool_display_name(display_name)
        return {
            "content": [{"type": "text", "text": json.dumps({name_key: display_name})}]
        }

    wrapper = _make_truncating_wrapper(execute, tool_name, tool_display_bridge=bridge)
    schema = _build_input_schema(TOOL_REGISTRY[tool_name])
    server = create_sdk_mcp_server(
        "copilot", tools=[tool(tool_name, "Run tool", schema)(wrapper)]
    )
    try:
        async with create_connected_server_and_client_session(
            server["instance"]
        ) as client:
            result = await client.call_tool(tool_name, tagged)
            assert not result.isError
        assert bridge.ready.is_set()
        [display] = bridge.drain()
        assert display.id == "toolu-provider"
        assert display.data.displayName == display_name
        assert (
            json.loads(pop_pending_tool_output(tool_name, args))[name_key]
            == display_name
        )
    finally:
        set_execution_context(None, None)


def test_adapter_never_exposes_sdk_correlation_token_in_input():
    bridge = SDKToolDisplayBridge()
    args = {"preset_id": "preset"}
    tagged = bridge.prepare_call("run_agent", args, "toolu-real")
    adapter = SDKResponseAdapter()
    responses = adapter.convert_message(
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id="toolu-real", name="mcp__copilot__run_agent", input=tagged
                )
            ],
            model="test",
        )
    )
    [part] = [
        response
        for response in responses
        if isinstance(response, StreamToolInputAvailable)
    ]
    assert part.input == args
    assert adapter.current_tool_calls["toolu-real"]["input"] == args


def test_fallback_transcript_never_persists_sdk_correlation_token():
    bridge = SDKToolDisplayBridge()
    args = {"preset_id": "preset"}
    tagged = bridge.prepare_call("run_agent", args, "toolu-real")
    [block] = _format_sdk_content_blocks(
        [ToolUseBlock(id="toolu-real", name="mcp__copilot__run_agent", input=tagged)]
    )
    assert block["input"] == args


@pytest.mark.parametrize("name_first", [True, False])
@pytest.mark.parametrize(
    "tool_name,tool_input,display_name",
    [
        ("run_agent", {"preset_id": "p"}, "Daily Digest"),
        ("run_block", {"block_id": "b", "input_data": {}}, "FillTextTemplateBlock"),
        ("continue_run_block", {"review_id": "r"}, "FillTextTemplateBlock"),
    ],
)
def test_dispatch_persists_name_without_duplicate_call_and_flags_late_save(
    name_first, tool_name, tool_input, display_name
):
    session = make_session()
    ctx = MagicMock(session=session)
    state = MagicMock()
    acc = _StreamAccumulator(
        assistant_response=ChatMessage(role="assistant"), accumulated_tool_calls=[]
    )
    display = StreamToolDisplayAvailable(
        id="call", data=ToolDisplayData(toolCallId="call", displayName=display_name)
    )
    start = StreamToolInputAvailable(
        toolCallId="call", toolName=tool_name, input=tool_input
    )
    events = [display, start] if name_first else [start, display]
    for event in events:
        _dispatch_response(event, acc, ctx, state, False, "[test]")
        if isinstance(event, StreamToolInputAvailable):
            acc.assistant_response.sequence = 0
    _dispatch_response(display, acc, ctx, state, False, "[test]")
    assert len(session.messages) == 1
    assert len(session.messages[0].tool_calls) == 1
    call = session.messages[0].tool_calls[0]
    assert call["display_name"] == display_name
    assert call["function"]["name"] == tool_name
    assert json.loads(call["function"]["arguments"]) == tool_input
    if not name_first:
        assert session.messages[0].tool_calls_pending_save


@pytest.mark.asyncio
async def test_name_wakes_stream_without_cancelling_pending_sdk_read():
    client = AsyncMock()
    display_ready = asyncio.Event()
    release = asyncio.Event()
    cancelled = False

    async def receive():
        nonlocal cancelled
        try:
            await release.wait()
            yield "result"
        except asyncio.CancelledError:
            cancelled = True
            raise

    client.receive_response = receive
    stream = _iter_sdk_messages(client, tool_display_wake=display_ready)
    asyncio.get_running_loop().call_soon(display_ready.set)
    try:
        assert await asyncio.wait_for(anext(stream), 1) is None
        assert not cancelled
        release.set()
        assert await asyncio.wait_for(anext(stream), 1) == "result"
    finally:
        await stream.aclose()
