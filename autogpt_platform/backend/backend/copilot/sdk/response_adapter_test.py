"""Unit tests for the SDK response adapter."""

import asyncio

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamFinishStep,
    StreamStart,
    StreamStartStep,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)

from .response_adapter import SDKResponseAdapter
from .tool_adapter import MCP_TOOL_PREFIX
from .tool_adapter import _pending_tool_outputs as _pto
from .tool_adapter import _stash_event
from .tool_adapter import stash_pending_tool_output as _stash
from .tool_adapter import wait_for_stash


def _adapter() -> SDKResponseAdapter:
    return SDKResponseAdapter(message_id="msg-1", session_id="session-1")


# -- SystemMessage -----------------------------------------------------------


def test_system_init_emits_start_and_step():
    adapter = _adapter()
    results = adapter.convert_message(SystemMessage(subtype="init", data={}))
    assert len(results) == 2
    assert isinstance(results[0], StreamStart)
    assert results[0].messageId == "msg-1"
    assert results[0].sessionId == "session-1"
    assert isinstance(results[1], StreamStartStep)


def test_system_non_init_emits_nothing():
    adapter = _adapter()
    results = adapter.convert_message(SystemMessage(subtype="other", data={}))
    assert results == []


# -- AssistantMessage with TextBlock -----------------------------------------


def test_text_block_emits_step_start_and_delta():
    adapter = _adapter()
    msg = AssistantMessage(content=[TextBlock(text="hello")], model="test")
    results = adapter.convert_message(msg)
    assert len(results) == 3
    assert isinstance(results[0], StreamStartStep)
    assert isinstance(results[1], StreamTextStart)
    assert isinstance(results[2], StreamTextDelta)
    assert results[2].delta == "hello"


def test_empty_text_block_emits_only_step():
    adapter = _adapter()
    msg = AssistantMessage(content=[TextBlock(text="")], model="test")
    results = adapter.convert_message(msg)
    # Empty text skipped, but step still opens
    assert len(results) == 1
    assert isinstance(results[0], StreamStartStep)


def test_multiple_text_deltas_reuse_block_id():
    adapter = _adapter()
    msg1 = AssistantMessage(content=[TextBlock(text="a")], model="test")
    msg2 = AssistantMessage(content=[TextBlock(text="b")], model="test")
    r1 = adapter.convert_message(msg1)
    r2 = adapter.convert_message(msg2)
    # First gets step+start+delta, second only delta (block & step already started)
    assert len(r1) == 3
    assert isinstance(r1[0], StreamStartStep)
    assert isinstance(r1[1], StreamTextStart)
    assert len(r2) == 1
    assert isinstance(r2[0], StreamTextDelta)
    assert r1[1].id == r2[0].id  # same block ID


# -- AssistantMessage with ToolUseBlock --------------------------------------


def test_tool_use_emits_input_start_and_available():
    """Tool names arrive with MCP prefix and should be stripped for the frontend."""
    adapter = _adapter()
    msg = AssistantMessage(
        content=[
            ToolUseBlock(
                id="tool-1",
                name=f"{MCP_TOOL_PREFIX}find_agent",
                input={"q": "x"},
            )
        ],
        model="test",
    )
    results = adapter.convert_message(msg)
    assert len(results) == 3
    assert isinstance(results[0], StreamStartStep)
    assert isinstance(results[1], StreamToolInputStart)
    assert results[1].toolCallId == "tool-1"
    assert results[1].toolName == "find_agent"  # prefix stripped
    assert isinstance(results[2], StreamToolInputAvailable)
    assert results[2].toolName == "find_agent"  # prefix stripped
    assert results[2].input == {"q": "x"}


def test_text_then_tool_ends_text_block():
    adapter = _adapter()
    text_msg = AssistantMessage(content=[TextBlock(text="thinking...")], model="test")
    tool_msg = AssistantMessage(
        content=[ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}tool", input={})],
        model="test",
    )
    adapter.convert_message(text_msg)  # opens step + text
    results = adapter.convert_message(tool_msg)
    # Step already open, so: TextEnd, ToolInputStart, ToolInputAvailable
    assert len(results) == 3
    assert isinstance(results[0], StreamTextEnd)
    assert isinstance(results[1], StreamToolInputStart)


# -- UserMessage with ToolResultBlock ----------------------------------------


def test_tool_result_emits_output_and_finish_step():
    adapter = _adapter()
    # First register the tool call (opens step) — SDK sends prefixed name
    tool_msg = AssistantMessage(
        content=[ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}find_agent", input={})],
        model="test",
    )
    adapter.convert_message(tool_msg)

    # Now send tool result
    result_msg = UserMessage(
        content=[ToolResultBlock(tool_use_id="t1", content="found 3 agents")]
    )
    results = adapter.convert_message(result_msg)
    assert len(results) == 2
    assert isinstance(results[0], StreamToolOutputAvailable)
    assert results[0].toolCallId == "t1"
    assert results[0].toolName == "find_agent"  # prefix stripped
    assert results[0].output == "found 3 agents"
    assert results[0].success is True
    assert isinstance(results[1], StreamFinishStep)


def test_tool_result_error():
    adapter = _adapter()
    adapter.convert_message(
        AssistantMessage(
            content=[
                ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}run_agent", input={})
            ],
            model="test",
        )
    )
    result_msg = UserMessage(
        content=[ToolResultBlock(tool_use_id="t1", content="timeout", is_error=True)]
    )
    results = adapter.convert_message(result_msg)
    assert isinstance(results[0], StreamToolOutputAvailable)
    assert results[0].success is False
    assert isinstance(results[1], StreamFinishStep)


def test_tool_result_list_content():
    adapter = _adapter()
    adapter.convert_message(
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}tool", input={})],
            model="test",
        )
    )
    result_msg = UserMessage(
        content=[
            ToolResultBlock(
                tool_use_id="t1",
                content=[
                    {"type": "text", "text": "line1"},
                    {"type": "text", "text": "line2"},
                ],
            )
        ]
    )
    results = adapter.convert_message(result_msg)
    assert isinstance(results[0], StreamToolOutputAvailable)
    assert results[0].output == "line1line2"
    assert isinstance(results[1], StreamFinishStep)


def test_string_user_message_ignored():
    """A plain string UserMessage (not tool results) produces no output."""
    adapter = _adapter()
    results = adapter.convert_message(UserMessage(content="hello"))
    assert results == []


# -- ResultMessage -----------------------------------------------------------


def test_result_success_emits_finish_step_and_finish():
    adapter = _adapter()
    # Start some text first (opens step)
    adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="done")], model="test")
    )
    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id="s1",
    )
    results = adapter.convert_message(msg)
    # TextEnd + FinishStep + StreamFinish
    assert len(results) == 3
    assert isinstance(results[0], StreamTextEnd)
    assert isinstance(results[1], StreamFinishStep)
    assert isinstance(results[2], StreamFinish)


def test_result_error_emits_error_and_finish():
    adapter = _adapter()
    msg = ResultMessage(
        subtype="error",
        duration_ms=100,
        duration_api_ms=50,
        is_error=True,
        num_turns=0,
        session_id="s1",
        result="API rate limited",
    )
    results = adapter.convert_message(msg)
    # No step was open, so no FinishStep — just Error + Finish
    assert len(results) == 2
    assert isinstance(results[0], StreamError)
    assert "API rate limited" in results[0].errorText
    assert isinstance(results[1], StreamFinish)


# -- Text after tools (new block ID) ----------------------------------------


def test_text_after_tool_gets_new_block_id():
    adapter = _adapter()
    # Text -> Tool -> ToolResult -> Text should get a new text block ID and step
    adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="before")], model="test")
    )
    adapter.convert_message(
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}tool", input={})],
            model="test",
        )
    )
    # Send tool result (closes step)
    adapter.convert_message(
        UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="ok")])
    )
    results = adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="after")], model="test")
    )
    # Should get StreamStartStep (new step) + StreamTextStart (new block) + StreamTextDelta
    assert len(results) == 3
    assert isinstance(results[0], StreamStartStep)
    assert isinstance(results[1], StreamTextStart)
    assert isinstance(results[2], StreamTextDelta)
    assert results[2].delta == "after"


# -- Full conversation flow --------------------------------------------------


def test_full_conversation_flow():
    """Simulate a complete conversation: init -> text -> tool -> result -> text -> finish."""
    adapter = _adapter()
    all_responses: list[StreamBaseResponse] = []

    # 1. Init
    all_responses.extend(
        adapter.convert_message(SystemMessage(subtype="init", data={}))
    )
    # 2. Assistant text
    all_responses.extend(
        adapter.convert_message(
            AssistantMessage(content=[TextBlock(text="Let me search")], model="test")
        )
    )
    # 3. Tool use
    all_responses.extend(
        adapter.convert_message(
            AssistantMessage(
                content=[
                    ToolUseBlock(
                        id="t1",
                        name=f"{MCP_TOOL_PREFIX}find_agent",
                        input={"query": "email"},
                    )
                ],
                model="test",
            )
        )
    )
    # 4. Tool result
    all_responses.extend(
        adapter.convert_message(
            UserMessage(
                content=[ToolResultBlock(tool_use_id="t1", content="Found 2 agents")]
            )
        )
    )
    # 5. More text
    all_responses.extend(
        adapter.convert_message(
            AssistantMessage(content=[TextBlock(text="I found 2")], model="test")
        )
    )
    # 6. Result
    all_responses.extend(
        adapter.convert_message(
            ResultMessage(
                subtype="success",
                duration_ms=500,
                duration_api_ms=400,
                is_error=False,
                num_turns=2,
                session_id="s1",
            )
        )
    )

    types = [type(r).__name__ for r in all_responses]
    assert types == [
        "StreamStart",
        "StreamStartStep",  # step 1: text + tool call
        "StreamTextStart",
        "StreamTextDelta",  # "Let me search"
        "StreamTextEnd",  # closed before tool
        "StreamToolInputStart",
        "StreamToolInputAvailable",
        "StreamToolOutputAvailable",  # tool result
        "StreamFinishStep",  # step 1 closed after tool result
        "StreamStartStep",  # step 2: continuation text
        "StreamTextStart",  # new block after tool
        "StreamTextDelta",  # "I found 2"
        "StreamTextEnd",  # closed by result
        "StreamFinishStep",  # step 2 closed
        "StreamFinish",
    ]


# -- Flush unresolved tool calls --------------------------------------------


def test_flush_unresolved_at_result_message():
    """Built-in tools (WebSearch) without UserMessage results get flushed at ResultMessage."""
    adapter = _adapter()
    all_responses: list[StreamBaseResponse] = []

    # 1. Init
    all_responses.extend(
        adapter.convert_message(SystemMessage(subtype="init", data={}))
    )
    # 2. Tool use (built-in tool — no MCP prefix)
    all_responses.extend(
        adapter.convert_message(
            AssistantMessage(
                content=[
                    ToolUseBlock(id="ws-1", name="WebSearch", input={"query": "test"})
                ],
                model="test",
            )
        )
    )
    # 3. No UserMessage for this tool — go straight to ResultMessage
    all_responses.extend(
        adapter.convert_message(
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="s1",
            )
        )
    )

    types = [type(r).__name__ for r in all_responses]
    assert types == [
        "StreamStart",
        "StreamStartStep",
        "StreamToolInputStart",
        "StreamToolInputAvailable",
        "StreamToolOutputAvailable",  # flushed with empty output
        "StreamFinishStep",  # step closed by flush
        "StreamFinish",
    ]
    # The flushed output should be empty (no stash available)
    output_event = [
        r for r in all_responses if isinstance(r, StreamToolOutputAvailable)
    ][0]
    assert output_event.toolCallId == "ws-1"
    assert output_event.toolName == "WebSearch"
    assert output_event.output == ""


def test_flush_unresolved_at_next_assistant_message():
    """Built-in tools get flushed when the next AssistantMessage arrives."""
    adapter = _adapter()
    all_responses: list[StreamBaseResponse] = []

    # 1. Init
    all_responses.extend(
        adapter.convert_message(SystemMessage(subtype="init", data={}))
    )
    # 2. Tool use (built-in — no UserMessage will come)
    all_responses.extend(
        adapter.convert_message(
            AssistantMessage(
                content=[
                    ToolUseBlock(id="ws-1", name="WebSearch", input={"query": "test"})
                ],
                model="test",
            )
        )
    )
    # 3. Next AssistantMessage triggers flush before processing its blocks
    all_responses.extend(
        adapter.convert_message(
            AssistantMessage(
                content=[TextBlock(text="Here are the results")], model="test"
            )
        )
    )

    types = [type(r).__name__ for r in all_responses]
    assert types == [
        "StreamStart",
        "StreamStartStep",
        "StreamToolInputStart",
        "StreamToolInputAvailable",
        # Flush at next AssistantMessage:
        "StreamToolOutputAvailable",
        "StreamFinishStep",  # step closed by flush
        # New step for continuation text:
        "StreamStartStep",
        "StreamTextStart",
        "StreamTextDelta",
    ]


def test_flush_with_stashed_output():
    """Stashed output from PostToolUse hook is used when flushing."""
    adapter = _adapter()

    # Simulate PostToolUse hook stashing output
    _pto.set({})
    _stash("WebSearch", "Search result: 5 items found")

    all_responses: list[StreamBaseResponse] = []

    # Tool use
    all_responses.extend(
        adapter.convert_message(
            AssistantMessage(
                content=[
                    ToolUseBlock(id="ws-1", name="WebSearch", input={"query": "test"})
                ],
                model="test",
            )
        )
    )
    # ResultMessage triggers flush
    all_responses.extend(
        adapter.convert_message(
            ResultMessage(
                subtype="success",
                duration_ms=100,
                duration_api_ms=50,
                is_error=False,
                num_turns=1,
                session_id="s1",
            )
        )
    )

    output_events = [
        r for r in all_responses if isinstance(r, StreamToolOutputAvailable)
    ]
    assert len(output_events) == 1
    assert output_events[0].output == "Search result: 5 items found"

    # Cleanup
    _pto.set({})  # type: ignore[arg-type]


# -- wait_for_stash synchronisation tests --


@pytest.mark.asyncio
async def test_wait_for_stash_signaled():
    """wait_for_stash returns True when stash_pending_tool_output signals."""
    _pto.set({})
    event = asyncio.Event()
    _stash_event.set(event)

    # Simulate a PostToolUse hook that stashes output after a short delay
    async def delayed_stash():
        await asyncio.sleep(0.01)
        _stash("WebSearch", "result data")

    asyncio.create_task(delayed_stash())
    result = await wait_for_stash(timeout=1.0)

    assert result is True
    assert _pto.get({}).get("WebSearch") == ["result data"]

    # Cleanup
    _pto.set({})  # type: ignore[arg-type]
    _stash_event.set(None)


@pytest.mark.asyncio
async def test_wait_for_stash_timeout():
    """wait_for_stash returns False on timeout when no stash occurs."""
    _pto.set({})
    event = asyncio.Event()
    _stash_event.set(event)

    result = await wait_for_stash(timeout=0.05)
    assert result is False

    # Cleanup
    _pto.set({})  # type: ignore[arg-type]
    _stash_event.set(None)


@pytest.mark.asyncio
async def test_wait_for_stash_already_stashed():
    """wait_for_stash picks up a stash that happened just before the wait."""
    _pto.set({})
    event = asyncio.Event()
    _stash_event.set(event)

    # Stash before waiting — simulates hook completing before message arrives
    _stash("Read", "file contents")
    # Event is now set; wait_for_stash detects the fast path and returns
    # immediately without timing out.
    result = await wait_for_stash(timeout=0.05)
    assert result is True

    # But the stash itself is populated
    assert _pto.get({}).get("Read") == ["file contents"]

    # Cleanup
    _pto.set({})  # type: ignore[arg-type]
    _stash_event.set(None)


# -- Parallel tool call tests --


def test_parallel_tool_calls_not_flushed_prematurely():
    """Parallel tool calls should NOT be flushed when the next AssistantMessage
    only contains ToolUseBlocks (parallel continuation)."""
    adapter = SDKResponseAdapter()

    # Init
    adapter.convert_message(SystemMessage(subtype="init", data={}))

    # First AssistantMessage: tool call #1
    msg1 = AssistantMessage(
        content=[ToolUseBlock(id="t1", name="WebSearch", input={"q": "foo"})],
        model="test",
    )
    r1 = adapter.convert_message(msg1)
    assert any(isinstance(r, StreamToolInputAvailable) for r in r1)
    assert adapter.has_unresolved_tool_calls

    # Second AssistantMessage: tool call #2 (parallel continuation)
    msg2 = AssistantMessage(
        content=[ToolUseBlock(id="t2", name="WebSearch", input={"q": "bar"})],
        model="test",
    )
    r2 = adapter.convert_message(msg2)

    # No flush should have happened — t1 should NOT have StreamToolOutputAvailable
    output_events = [r for r in r2 if isinstance(r, StreamToolOutputAvailable)]
    assert len(output_events) == 0, (
        f"Tool-only AssistantMessage should not flush prior tools, "
        f"but got {len(output_events)} output events"
    )

    # Both t1 and t2 should still be unresolved
    assert "t1" not in adapter.resolved_tool_calls
    assert "t2" not in adapter.resolved_tool_calls


def test_text_assistant_message_flushes_prior_tools():
    """An AssistantMessage with text (new turn) should flush unresolved tools."""
    adapter = SDKResponseAdapter()

    # Init
    adapter.convert_message(SystemMessage(subtype="init", data={}))

    # Tool call
    msg1 = AssistantMessage(
        content=[ToolUseBlock(id="t1", name="WebSearch", input={"q": "foo"})],
        model="test",
    )
    adapter.convert_message(msg1)
    assert adapter.has_unresolved_tool_calls

    # Text AssistantMessage (new turn after tools completed)
    msg2 = AssistantMessage(
        content=[TextBlock(text="Here are the results")],
        model="test",
    )
    r2 = adapter.convert_message(msg2)

    # Flush SHOULD have happened — t1 gets empty output
    output_events = [r for r in r2 if isinstance(r, StreamToolOutputAvailable)]
    assert len(output_events) == 1
    assert output_events[0].toolCallId == "t1"
    assert "t1" in adapter.resolved_tool_calls


def test_already_resolved_tool_skipped_in_user_message():
    """A tool result in UserMessage should be skipped if already resolved by flush."""
    adapter = SDKResponseAdapter()

    adapter.convert_message(SystemMessage(subtype="init", data={}))

    # Tool call + flush via text message
    adapter.convert_message(
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="WebSearch", input={})],
            model="test",
        )
    )
    adapter.convert_message(
        AssistantMessage(
            content=[TextBlock(text="Done")],
            model="test",
        )
    )
    assert "t1" in adapter.resolved_tool_calls

    # Now UserMessage arrives with the real result — should be skipped
    user_msg = UserMessage(content=[ToolResultBlock(tool_use_id="t1", content="real")])
    r = adapter.convert_message(user_msg)
    output_events = [r_ for r_ in r if isinstance(r_, StreamToolOutputAvailable)]
    assert (
        len(output_events) == 0
    ), "Already-resolved tool should not emit duplicate output"
