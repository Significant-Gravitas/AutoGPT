"""Unit tests for the SDK response adapter."""

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from backend.api.features.chat.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamStart,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)

from .response_adapter import SDKResponseAdapter


def _adapter() -> SDKResponseAdapter:
    a = SDKResponseAdapter(message_id="msg-1")
    a.set_task_id("task-1")
    return a


# -- SystemMessage -----------------------------------------------------------


def test_system_init_emits_start():
    adapter = _adapter()
    results = adapter.convert_message(SystemMessage(subtype="init", data={}))
    assert len(results) == 1
    assert isinstance(results[0], StreamStart)
    assert results[0].messageId == "msg-1"
    assert results[0].taskId == "task-1"


def test_system_non_init_emits_nothing():
    adapter = _adapter()
    results = adapter.convert_message(SystemMessage(subtype="other", data={}))
    assert results == []


# -- AssistantMessage with TextBlock -----------------------------------------


def test_text_block_emits_start_and_delta():
    adapter = _adapter()
    msg = AssistantMessage(content=[TextBlock(text="hello")], model="test")
    results = adapter.convert_message(msg)
    assert len(results) == 2
    assert isinstance(results[0], StreamTextStart)
    assert isinstance(results[1], StreamTextDelta)
    assert results[1].delta == "hello"


def test_empty_text_block_is_skipped():
    adapter = _adapter()
    msg = AssistantMessage(content=[TextBlock(text="")], model="test")
    results = adapter.convert_message(msg)
    assert results == []


def test_multiple_text_deltas_reuse_block_id():
    adapter = _adapter()
    msg1 = AssistantMessage(content=[TextBlock(text="a")], model="test")
    msg2 = AssistantMessage(content=[TextBlock(text="b")], model="test")
    r1 = adapter.convert_message(msg1)
    r2 = adapter.convert_message(msg2)
    # First gets start+delta, second only delta (block already started)
    assert len(r1) == 2
    assert len(r2) == 1
    assert isinstance(r2[0], StreamTextDelta)
    assert isinstance(r1[0], StreamTextStart)
    assert r1[0].id == r2[0].id  # same block ID


# -- AssistantMessage with ToolUseBlock --------------------------------------


def test_tool_use_emits_input_start_and_available():
    adapter = _adapter()
    msg = AssistantMessage(
        content=[ToolUseBlock(id="tool-1", name="find_agent", input={"q": "x"})],
        model="test",
    )
    results = adapter.convert_message(msg)
    assert len(results) == 2
    assert isinstance(results[0], StreamToolInputStart)
    assert results[0].toolCallId == "tool-1"
    assert results[0].toolName == "find_agent"
    assert isinstance(results[1], StreamToolInputAvailable)
    assert results[1].input == {"q": "x"}


def test_text_then_tool_ends_text_block():
    adapter = _adapter()
    text_msg = AssistantMessage(content=[TextBlock(text="thinking...")], model="test")
    tool_msg = AssistantMessage(
        content=[ToolUseBlock(id="t1", name="tool", input={})], model="test"
    )
    adapter.convert_message(text_msg)
    results = adapter.convert_message(tool_msg)
    # Should have: TextEnd, ToolInputStart, ToolInputAvailable
    assert len(results) == 3
    assert isinstance(results[0], StreamTextEnd)
    assert isinstance(results[1], StreamToolInputStart)


# -- UserMessage with ToolResultBlock ----------------------------------------


def test_tool_result_emits_output():
    adapter = _adapter()
    # First register the tool call
    tool_msg = AssistantMessage(
        content=[ToolUseBlock(id="t1", name="find_agent", input={})], model="test"
    )
    adapter.convert_message(tool_msg)

    # Now send tool result
    result_msg = UserMessage(
        content=[ToolResultBlock(tool_use_id="t1", content="found 3 agents")]
    )
    results = adapter.convert_message(result_msg)
    assert len(results) == 1
    assert isinstance(results[0], StreamToolOutputAvailable)
    assert results[0].toolCallId == "t1"
    assert results[0].toolName == "find_agent"
    assert results[0].output == "found 3 agents"
    assert results[0].success is True


def test_tool_result_error():
    adapter = _adapter()
    adapter.convert_message(
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="run_agent", input={})], model="test"
        )
    )
    result_msg = UserMessage(
        content=[ToolResultBlock(tool_use_id="t1", content="timeout", is_error=True)]
    )
    results = adapter.convert_message(result_msg)
    assert isinstance(results[0], StreamToolOutputAvailable)
    assert results[0].success is False


def test_tool_result_list_content():
    adapter = _adapter()
    adapter.convert_message(
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="tool", input={})], model="test"
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


def test_string_user_message_ignored():
    """A plain string UserMessage (not tool results) produces no output."""
    adapter = _adapter()
    results = adapter.convert_message(UserMessage(content="hello"))
    assert results == []


# -- ResultMessage -----------------------------------------------------------


def test_result_success_emits_finish():
    adapter = _adapter()
    # Start some text first
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
    # TextEnd + StreamFinish
    assert len(results) == 2
    assert isinstance(results[0], StreamTextEnd)
    assert isinstance(results[1], StreamFinish)


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
    assert len(results) == 2
    assert isinstance(results[0], StreamError)
    assert "API rate limited" in results[0].errorText
    assert isinstance(results[1], StreamFinish)


# -- Text after tools (new block ID) ----------------------------------------


def test_text_after_tool_gets_new_block_id():
    adapter = _adapter()
    # Text -> Tool -> Text should get a new text block ID
    adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="before")], model="test")
    )
    adapter.convert_message(
        AssistantMessage(
            content=[ToolUseBlock(id="t1", name="tool", input={})], model="test"
        )
    )
    results = adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="after")], model="test")
    )
    # Should get StreamTextStart (new block) + StreamTextDelta
    assert len(results) == 2
    assert isinstance(results[0], StreamTextStart)
    assert isinstance(results[1], StreamTextDelta)
    assert results[1].delta == "after"


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
                    ToolUseBlock(id="t1", name="find_agent", input={"query": "email"})
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
        "StreamTextStart",
        "StreamTextDelta",  # "Let me search"
        "StreamTextEnd",  # closed before tool
        "StreamToolInputStart",
        "StreamToolInputAvailable",
        "StreamToolOutputAvailable",  # tool result
        "StreamTextStart",  # new block after tool
        "StreamTextDelta",  # "I found 2"
        "StreamTextEnd",  # closed by result
        "StreamFinish",
    ]
