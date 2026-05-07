"""Unit tests for the SDK response adapter."""

import asyncio

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamFinishStep,
    StreamHeartbeat,
    StreamReasoningDelta,
    StreamReasoningEnd,
    StreamStart,
    StreamStartStep,
    StreamStatus,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
)

from .compaction import compaction_events
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


def test_task_progress_emits_heartbeat():
    """task_progress events emit a StreamHeartbeat to keep Redis TTL alive."""
    adapter = _adapter()
    results = adapter.convert_message(SystemMessage(subtype="task_progress", data={}))
    assert len(results) == 1
    assert isinstance(results[0], StreamHeartbeat)


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


def test_tool_use_strips_whitespace_in_tool_name():
    adapter = _adapter()
    msg = AssistantMessage(
        content=[
            ToolUseBlock(
                id="tool-1",
                name=f" {MCP_TOOL_PREFIX}find_block",
                input={},
            )
        ],
        model="test",
    )
    results = adapter.convert_message(msg)
    tool_events = [
        r
        for r in results
        if isinstance(r, (StreamToolInputStart, StreamToolInputAvailable))
    ]
    assert tool_events, "expected tool input events"
    for event in tool_events:
        assert event.toolName == "find_block"


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
    assert len(results) == 3
    assert isinstance(results[0], StreamToolOutputAvailable)
    assert results[0].toolCallId == "t1"
    assert results[0].toolName == "find_agent"  # prefix stripped
    assert results[0].output == "found 3 agents"
    assert results[0].success is True
    assert isinstance(results[1], StreamFinishStep)
    assert isinstance(results[2], StreamStatus)
    assert results[2].message == "Analyzing result…"


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


# -- Reasoning streaming -----------------------------------------------------


def test_thinking_block_streams_as_reasoning():
    """ThinkingBlock content streams as StreamReasoningDelta so the
    frontend renders it via the ``Reasoning`` part (collapsed by
    default) instead of dropping it silently."""
    adapter = _adapter()
    msg = AssistantMessage(
        content=[
            ThinkingBlock(thinking="planning step 1", signature="sig"),
        ],
        model="test",
    )
    results = adapter.convert_message(msg)
    # Step + ReasoningStart + ReasoningDelta
    types = [type(r).__name__ for r in results]
    assert "StreamReasoningStart" in types
    assert any(
        isinstance(r, StreamReasoningDelta) and r.delta == "planning step 1"
        for r in results
    )


def test_text_after_thinking_closes_reasoning_and_opens_text():
    """Reasoning and text are distinct UI parts — opening text must
    emit ``ReasoningEnd`` first so the AI SDK transport doesn't merge
    them into the same ``Reasoning`` part."""
    adapter = _adapter()
    adapter.convert_message(
        AssistantMessage(
            content=[ThinkingBlock(thinking="warming up", signature="sig")],
            model="test",
        )
    )
    results = adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="hello")], model="test")
    )
    types = [type(r).__name__ for r in results]
    # ReasoningEnd must come before TextStart
    re_idx = types.index("StreamReasoningEnd")
    ts_idx = types.index("StreamTextStart")
    assert re_idx < ts_idx


def test_thinking_after_text_in_same_message_renders_reasoning_first():
    """Kimi K2.6 (and other non-Anthropic OpenRouter providers) place
    ``reasoning`` AFTER the visible text in the response, so the SDK
    builds an ``AssistantMessage`` with content = [TextBlock, ThinkingBlock].
    Without reordering, the UI would show the answer first and the
    reasoning panel below it — the opposite of the natural reading
    order Anthropic models produce.  response_adapter must hoist
    ThinkingBlocks to the front so ``reasoning-start/delta/end`` events
    hit the SSE stream BEFORE the ``text-*`` events."""
    adapter = _adapter()
    msg = AssistantMessage(
        content=[
            TextBlock(text="63"),
            ThinkingBlock(thinking="7 times 9 is 63", signature=""),
        ],
        model="test",
    )
    results = adapter.convert_message(msg)
    types = [type(r).__name__ for r in results]
    # ReasoningStart must land before TextStart in the emitted stream
    assert "StreamReasoningStart" in types
    assert "StreamTextStart" in types
    assert types.index("StreamReasoningStart") < types.index("StreamTextStart")
    # ReasoningDelta payload is intact
    assert any(
        isinstance(r, StreamReasoningDelta) and r.delta == "7 times 9 is 63"
        for r in results
    )


def test_tool_use_after_thinking_closes_reasoning():
    """Opening a tool also closes an open reasoning block."""
    adapter = _adapter()
    adapter.convert_message(
        AssistantMessage(
            content=[ThinkingBlock(thinking="let me search", signature="sig")],
            model="test",
        )
    )
    results = adapter.convert_message(
        AssistantMessage(
            content=[
                ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}find_block", input={})
            ],
            model="test",
        )
    )
    types = [type(r).__name__ for r in results]
    assert types.index("StreamReasoningEnd") < types.index("StreamToolInputStart")


def test_empty_thinking_block_is_ignored():
    """A ThinkingBlock with empty content shouldn't emit anything."""
    adapter = _adapter()
    msg = AssistantMessage(
        content=[ThinkingBlock(thinking="", signature="sig")],
        model="test",
    )
    results = adapter.convert_message(msg)
    # Only the StepStart fires — no reasoning events.
    assert [type(r).__name__ for r in results] == ["StreamStartStep"]


def test_render_reasoning_in_ui_false_still_emits_adapter_events():
    """With the persist/render decoupling the adapter is flag-agnostic:
    it always emits ``StreamReasoning*`` so the session transcript keeps a
    durable reasoning record.  Wire-level suppression when
    ``render_reasoning_in_ui=False`` happens at the SDK service yield
    boundary, not here — see
    ``backend/copilot/sdk/service.py::_filter_reasoning_events``.
    """
    adapter = SDKResponseAdapter(
        message_id="m",
        session_id="s",
        render_reasoning_in_ui=False,
    )
    msg = AssistantMessage(
        content=[ThinkingBlock(thinking="plan", signature="sig")],
        model="test",
    )
    results = adapter.convert_message(msg)
    types = [type(r).__name__ for r in results]
    assert "StreamReasoningStart" in types
    assert "StreamReasoningDelta" in types


def test_render_reasoning_off_text_after_thinking_still_closes_reasoning():
    """Adapter still emits a ``StreamReasoningEnd`` when text follows a
    thinking block — decoupled from the render flag.  The service layer
    drops the reasoning events at yield time; the adapter's structural
    open/close pairing must not depend on the flag or downstream filters
    would see orphan reasoning starts on the persisted transcript.
    """
    adapter = SDKResponseAdapter(
        message_id="m",
        session_id="s",
        render_reasoning_in_ui=False,
    )
    adapter.convert_message(
        AssistantMessage(
            content=[ThinkingBlock(thinking="warming up", signature="sig")],
            model="test",
        )
    )
    results = adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="hello")], model="test")
    )
    types = [type(r).__name__ for r in results]
    assert "StreamReasoningEnd" in types
    assert "StreamTextStart" in types
    assert "StreamTextDelta" in types


def test_render_reasoning_on_is_default():
    """Default is True — existing callers keep emitting reasoning events."""
    adapter = SDKResponseAdapter(message_id="m", session_id="s")
    msg = AssistantMessage(
        content=[ThinkingBlock(thinking="plan", signature="sig")],
        model="test",
    )
    results = adapter.convert_message(msg)
    types = [type(r).__name__ for r in results]
    assert "StreamReasoningStart" in types
    assert "StreamReasoningDelta" in types


def test_result_success_thinking_only_first_pass_defers_for_reprompt():
    """First time we see a thinking-only final turn the adapter must defer:
    no text, no StreamFinish.  Driver reads ``pending_thinking_only_reprompt``
    and re-prompts the model for a closing TextBlock before falling back."""
    adapter = _adapter()

    adapter.convert_message(
        AssistantMessage(
            content=[
                ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}find_block", input={}),
            ],
            model="test",
        )
    )
    adapter.convert_message(
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="t1", content="result", is_error=False)
            ],
            parent_tool_use_id=None,
        )
    )

    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=4,
        session_id="s1",
        result="",
    )
    results = adapter.convert_message(msg)

    text_deltas = [r for r in results if isinstance(r, StreamTextDelta)]
    finishes = [r for r in results if isinstance(r, StreamFinish)]
    assert text_deltas == [], "first pass must not emit placeholder"
    assert finishes == [], "first pass must skip StreamFinish so driver can re-prompt"
    assert adapter.pending_thinking_only_reprompt is True
    assert adapter.thinking_only_reprompted is False


def test_result_success_thinking_only_after_reprompt_promotes_thinking():
    """After re-prompt, if the model still produces thinking-only, the
    adapter promotes the most recent ThinkingBlock content to visible text
    rather than showing the bare placeholder."""
    adapter = _adapter()
    adapter._last_thinking_content = (
        "Here are the best restaurants: Dishoom, The Clove Club, Padella."
    )
    # Simulate the driver having already fired the re-prompt round once.
    adapter.thinking_only_reprompted = True
    adapter._any_tool_results_seen = True
    adapter._text_since_last_tool_result = False

    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=4,
        session_id="s1",
        result="",
    )
    results = adapter.convert_message(msg)

    text_deltas = [r for r in results if isinstance(r, StreamTextDelta)]
    assert len(text_deltas) == 1
    assert "Dishoom" in text_deltas[0].delta
    assert isinstance(results[-1], StreamFinish)


def test_result_success_thinking_only_two_rounds_with_driver_reset_emits_fallback():
    """Regression: the driver's reset between the deferred first round and
    the re-prompt second round must leave enough state for the guard to
    fire when the second round also returns thinking-only.  Specifically,
    ``_any_tool_results_seen`` must remain True after the reset — the
    guard requires it.
    """
    adapter = _adapter()

    # --- Round 1: tool_use → tool_result → thinking-only finish.
    adapter.convert_message(
        AssistantMessage(
            content=[
                ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}find_block", input={})
            ],
            model="test",
        )
    )
    adapter.convert_message(
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="t1", content="result", is_error=False)
            ],
            parent_tool_use_id=None,
        )
    )
    adapter.convert_message(
        AssistantMessage(
            content=[
                ThinkingBlock(thinking="Round 1 internal reasoning.", signature="")
            ],
            model="test",
        )
    )
    round1 = adapter.convert_message(
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=2,
            session_id="s1",
            result="",
        )
    )
    assert adapter.pending_thinking_only_reprompt is True
    assert [r for r in round1 if isinstance(r, StreamFinish)] == []

    # --- Driver behaviour between rounds (must mirror service.py exactly).
    adapter.pending_thinking_only_reprompt = False
    adapter.thinking_only_reprompted = True
    adapter._text_since_last_tool_result = False
    # Intentionally do NOT touch ``_any_tool_results_seen`` — the guard at
    # ResultMessage time needs it to stay True so the placeholder fires
    # if the re-prompt round also returns thinking-only.

    # --- Round 2: model returns thinking-only again.
    adapter.convert_message(
        AssistantMessage(
            content=[
                ThinkingBlock(thinking="Round 2 internal reasoning.", signature="")
            ],
            model="test",
        )
    )
    round2 = adapter.convert_message(
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=4,
            session_id="s1",
            result="",
        )
    )
    text_deltas = [r for r in round2 if isinstance(r, StreamTextDelta)]
    assert len(text_deltas) == 1, "second pass must emit fallback text"
    assert text_deltas[0].delta.strip()  # non-empty
    assert isinstance(round2[-1], StreamFinish)


def test_tool_result_clears_stale_thinking_so_fallback_does_not_leak_pre_tool_thinking():
    """Stale ``ThinkingBlock`` content from before a tool call must not be
    promoted as fallback text once the tool result has landed.  Without
    this reset, a turn that thinks → tool_use → tool_result → silent
    finish would surface the *pre-tool* reasoning to the user, which
    pre-dates the actual answer the user is waiting for."""
    adapter = _adapter()

    # Pre-tool thinking — should be discarded when the tool_result lands.
    adapter.convert_message(
        AssistantMessage(
            content=[ThinkingBlock(thinking="Stale pre-tool reasoning.", signature="")],
            model="test",
        )
    )
    adapter.convert_message(
        AssistantMessage(
            content=[
                ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}find_block", input={}),
            ],
            model="test",
        )
    )
    adapter.convert_message(
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="t1", content="result", is_error=False)
            ],
            parent_tool_use_id=None,
        )
    )
    # Tool_result must wipe ``_last_thinking_content`` — otherwise
    # ``Stale pre-tool reasoning.`` would be the fallback below.
    assert adapter._last_thinking_content == ""

    # Simulate a thinking-only finish that emits NO new ThinkingBlock at all
    # (so ``_last_thinking_content`` stays empty), and a re-prompt round that
    # also produces nothing.  The fallback must be the placeholder, not the
    # stale pre-tool reasoning.
    adapter.thinking_only_reprompted = True
    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=4,
        session_id="s1",
        result="",
    )
    results = adapter.convert_message(msg)
    text_deltas = [r for r in results if isinstance(r, StreamTextDelta)]
    assert len(text_deltas) == 1
    assert text_deltas[0].delta == "(Done — no further commentary.)"
    assert "Stale pre-tool" not in text_deltas[0].delta


def test_result_success_thinking_only_after_reprompt_falls_back_to_placeholder():
    """After re-prompt with no thinking content captured either, the
    adapter emits the placeholder so the turn still visibly completes."""
    adapter = _adapter()
    adapter._last_thinking_content = ""
    adapter.thinking_only_reprompted = True
    adapter._any_tool_results_seen = True
    adapter._text_since_last_tool_result = False

    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=4,
        session_id="s1",
        result="",
    )
    results = adapter.convert_message(msg)

    text_deltas = [r for r in results if isinstance(r, StreamTextDelta)]
    assert len(text_deltas) == 1
    assert text_deltas[0].delta == "(Done — no further commentary.)"
    assert isinstance(results[-1], StreamFinish)


def test_result_success_does_not_synthesize_when_text_already_emitted():
    """Guard: do NOT synthesize when the model DID emit closing text
    after the last tool result — the fallback is only for the silent
    thinking-only case."""
    adapter = _adapter()

    adapter.convert_message(
        AssistantMessage(
            content=[
                ToolUseBlock(id="t1", name=f"{MCP_TOOL_PREFIX}find_block", input={})
            ],
            model="test",
        )
    )
    adapter.convert_message(
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="t1", content="result", is_error=False)
            ],
            parent_tool_use_id=None,
        )
    )
    # Model responds with actual text after the tool result.
    adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="all done")], model="test")
    )

    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=4,
        session_id="s1",
        result="all done",
    )
    results = adapter.convert_message(msg)

    # No fallback — the only TextDelta came from the previous
    # AssistantMessage call, not from ResultMessage's synthesis.
    text_deltas = [r for r in results if isinstance(r, StreamTextDelta)]
    assert text_deltas == []


def test_result_success_does_not_synthesize_when_no_tools_ran():
    """Guard: no tool_results seen ⇒ no fallback.  Pure-text turns with
    no tools legitimately produce text-only responses through normal
    AssistantMessage events; we don't need a fallback there."""
    adapter = _adapter()

    adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="hello")], model="test")
    )

    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id="s1",
        result="hello",
    )
    results = adapter.convert_message(msg)
    text_deltas = [r for r in results if isinstance(r, StreamTextDelta)]
    assert text_deltas == []


def test_result_empty_success_emits_error_and_finish():
    """SECRT-2252: a ``subtype="success"`` ResultMessage with empty ``result``,
    no produced content, and ``output_tokens == 0`` is the SDK's ghost-finish
    bug. The adapter surfaces it as a ``StreamError`` *paired with*
    ``StreamFinish`` so the service-layer post-stream flow flips
    ``acc.stream_completed`` and skips the ``STREAM_INCOMPLETE_MARKER``
    branch. ``SystemMessage(subtype="init")`` opened a step, so the
    empty-completion branch must close it before emitting the error."""
    adapter = _adapter()
    adapter.convert_message(SystemMessage(subtype="init", data={}))
    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id="s1",
        result=None,
        usage={"input_tokens": 5, "output_tokens": 0},
    )
    results = adapter.convert_message(msg)
    types = [type(r).__name__ for r in results]
    assert "StreamFinishStep" in types
    assert "StreamError" in types
    assert "StreamFinish" in types
    # Open step must be closed before the error, and the error must
    # precede StreamFinish on the wire.
    assert types.index("StreamFinishStep") < types.index("StreamError")
    assert types.index("StreamError") < types.index("StreamFinish")
    err = next(r for r in results if isinstance(r, StreamError))
    assert err.code == "empty_completion"


def test_result_empty_success_with_empty_string_result_treated_as_empty():
    """An empty string (not just None) for ``result`` is also empty."""
    adapter = _adapter()
    adapter.convert_message(SystemMessage(subtype="init", data={}))
    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id="s1",
        result="",
        usage={"output_tokens": 0},
    )
    results = adapter.convert_message(msg)
    err = next(r for r in results if isinstance(r, StreamError))
    assert err.code == "empty_completion"
    assert any(isinstance(r, StreamFinish) for r in results)


def test_result_success_with_text_emits_finish_not_error():
    """Non-empty success (text was produced) keeps the existing
    ``StreamFinish`` behaviour — no spurious error."""
    adapter = _adapter()
    adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="hello")], model="test")
    )
    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id="s1",
        result="hello",
        usage={"output_tokens": 5},
    )
    results = adapter.convert_message(msg)
    types = [type(r).__name__ for r in results]
    assert "StreamFinish" in types
    assert "StreamError" not in types


def test_result_success_with_nonzero_output_tokens_not_empty():
    """If ``output_tokens > 0`` but ``result`` is empty, don't classify as
    empty — fall through to the existing success path. No prior
    AssistantMessage so the `output_tokens` guard is the only thing
    keeping `_is_empty_completion()` from firing."""
    adapter = _adapter()
    adapter.convert_message(SystemMessage(subtype="init", data={}))
    msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id="s1",
        result="",
        usage={"output_tokens": 50},
    )
    results = adapter.convert_message(msg)
    types = [type(r).__name__ for r in results]
    assert "StreamFinish" in types
    assert "StreamError" not in types


def test_result_error_emits_error_and_finish():
    adapter = _adapter()
    msg = ResultMessage(
        subtype="error",
        duration_ms=100,
        duration_api_ms=50,
        is_error=True,
        num_turns=0,
        session_id="s1",
        result="Invalid API key provided",
    )
    results = adapter.convert_message(msg)
    # No step was open, so no FinishStep — just Error + Finish
    assert len(results) == 2
    assert isinstance(results[0], StreamError)
    assert "Invalid API key provided" in results[0].errorText
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
        "StreamStatus",  # user-facing status while continuation is generated
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
    ]
    # Flush marked a tool_result as seen with no text since, so the
    # thinking-only-final-turn guard defers placeholder emission and asks
    # the driver to re-prompt (no StreamFinish yet).
    assert adapter.pending_thinking_only_reprompt is True
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
    pto = _pto.get()
    assert pto is not None
    assert pto.get("WebSearch") == ["result data"]

    # Cleanup
    _pto.set({})
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
    _pto.set({})
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
    pto = _pto.get()
    assert pto is not None
    assert pto.get("Read") == ["file contents"]

    # Cleanup
    _pto.set({})
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


# -- _end_text_if_open before compaction -------------------------------------


def test_end_text_if_open_emits_text_end_before_finish_step():
    """StreamTextEnd must be emitted before StreamFinishStep during compaction.

    When ``emit_end_if_ready`` fires compaction events while a text block is
    still open, ``_end_text_if_open`` must close it first.  If StreamFinishStep
    arrives before StreamTextEnd, the Vercel AI SDK clears ``activeTextParts``
    and raises "Received text-end for missing text part".
    """
    adapter = _adapter()

    # Open a text block by processing an AssistantMessage with text
    msg = AssistantMessage(content=[TextBlock(text="partial response")], model="test")
    adapter.convert_message(msg)
    assert adapter.has_started_text
    assert not adapter.has_ended_text

    # Simulate what service.py does before yielding compaction events
    pre_close: list[StreamBaseResponse] = []
    adapter._end_text_if_open(pre_close)
    combined = pre_close + list(compaction_events("Compacted transcript"))

    text_end_idx = next(
        (i for i, e in enumerate(combined) if isinstance(e, StreamTextEnd)), None
    )
    finish_step_idx = next(
        (i for i, e in enumerate(combined) if isinstance(e, StreamFinishStep)), None
    )

    assert text_end_idx is not None, "StreamTextEnd must be present"
    assert finish_step_idx is not None, "StreamFinishStep must be present"
    assert text_end_idx < finish_step_idx, (
        f"StreamTextEnd (idx={text_end_idx}) must precede "
        f"StreamFinishStep (idx={finish_step_idx}) — otherwise the Vercel AI SDK "
        "clears activeTextParts before text-end arrives"
    )


def test_step_open_must_reset_after_compaction_finish_step():
    """Adapter step_open must be reset when compaction emits StreamFinishStep.

    Compaction events bypass the adapter, so service.py must explicitly clear
    step_open after yielding a StreamFinishStep from compaction. Without this,
    the next AssistantMessage skips StreamStartStep because the adapter still
    thinks a step is open.
    """
    adapter = _adapter()

    # Open a step + text block via an AssistantMessage
    msg = AssistantMessage(content=[TextBlock(text="thinking...")], model="test")
    adapter.convert_message(msg)
    assert adapter.step_open is True

    # Simulate what service.py does: close text, then check compaction events
    pre_close: list[StreamBaseResponse] = []
    adapter._end_text_if_open(pre_close)

    events = list(compaction_events("Compacted transcript"))
    if any(isinstance(ev, StreamFinishStep) for ev in events):
        adapter.step_open = False

    assert (
        adapter.step_open is False
    ), "step_open must be False after compaction emits StreamFinishStep"

    # Next AssistantMessage must open a new step
    msg2 = AssistantMessage(content=[TextBlock(text="continued")], model="test")
    results = adapter.convert_message(msg2)
    assert any(
        isinstance(r, StreamStartStep) for r in results
    ), "A new StreamStartStep must be emitted after compaction closed the step"


def test_end_text_if_open_no_op_when_no_text_open():
    """_end_text_if_open emits nothing when no text block is open."""
    adapter = _adapter()
    results: list[StreamBaseResponse] = []
    adapter._end_text_if_open(results)
    assert results == []


def test_end_text_if_open_no_op_after_text_already_ended():
    """_end_text_if_open emits nothing when the text block is already closed."""
    adapter = _adapter()
    msg = AssistantMessage(content=[TextBlock(text="hello")], model="test")
    adapter.convert_message(msg)
    # Close it once
    first: list[StreamBaseResponse] = []
    adapter._end_text_if_open(first)
    assert len(first) == 1
    assert isinstance(first[0], StreamTextEnd)
    # Second call must be a no-op
    second: list[StreamBaseResponse] = []
    adapter._end_text_if_open(second)
    assert second == []


# ---------------------------------------------------------------------------
# Partial-message streaming (CHAT_SDK_INCLUDE_PARTIAL_MESSAGES)
# Covers the 10 scenarios in docs/sdk-per-token-streaming-followup.md
# ---------------------------------------------------------------------------


def _stream_event(payload: dict) -> StreamEvent:
    """Convenience constructor for a raw Anthropic StreamEvent payload."""
    return StreamEvent(
        uuid="stream-evt",
        session_id="session-1",
        parent_tool_use_id=None,
        event=payload,
    )


def _message_start() -> StreamEvent:
    return _stream_event({"type": "message_start"})


def _text_block_start(index: int) -> StreamEvent:
    return _stream_event(
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""},
        }
    )


def _text_delta(index: int, text: str) -> StreamEvent:
    return _stream_event(
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }
    )


def _thinking_block_start(index: int) -> StreamEvent:
    return _stream_event(
        {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": ""},
        }
    )


def _thinking_delta(index: int, text: str) -> StreamEvent:
    return _stream_event(
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": text},
        }
    )


def _block_stop(index: int) -> StreamEvent:
    return _stream_event({"type": "content_block_stop", "index": index})


def _collect_text_deltas(responses):
    return "".join(r.delta for r in responses if isinstance(r, StreamTextDelta))


def _collect_reasoning_deltas(responses):
    return "".join(r.delta for r in responses if isinstance(r, StreamReasoningDelta))


class TestPartialMessageStreaming:
    """Scenarios 1-10 from sdk-per-token-streaming-followup.md.

    The adapter runs unconditionally in partial-aware mode — when the
    flag ``CHAT_SDK_INCLUDE_PARTIAL_MESSAGES`` is off the CLI simply
    never emits ``StreamEvent`` messages and the diff maps stay empty
    (so the tail logic degrades to "emit the full summary content"
    which is the pre-partial behaviour).
    """

    def test_partial_and_summary_agree_no_duplicate(self):
        """Scenario 1: partial streams full text, summary matches exactly.
        No duplicate emission, no truncation — full content reaches the
        wire once."""
        adapter = _adapter()
        full = "Hello world"
        responses: list[StreamBaseResponse] = []
        adapter._handle_stream_event(_text_block_start(0), responses)
        for chunk in ("Hello", " ", "world"):
            adapter._handle_stream_event(_text_delta(0, chunk), responses)
        adapter._handle_stream_event(_block_stop(0), responses)
        # Summary arrives with the same full text
        summary = adapter.convert_message(
            AssistantMessage(content=[TextBlock(text=full)], model="test")
        )
        combined = responses + summary
        assert _collect_text_deltas(combined) == full

    def test_partial_short_summary_long_tail_emitted(self):
        """Scenario 2 (the truncation bug we saw): partial emitted a
        prefix of the real answer; summary has the full text.  The
        adapter must emit only the tail so no content is lost."""
        adapter = _adapter()
        responses: list[StreamBaseResponse] = []
        adapter._handle_stream_event(_text_block_start(0), responses)
        for chunk in ("The user ", "seems confused. They sent"):
            adapter._handle_stream_event(_text_delta(0, chunk), responses)
        # Summary has the full, un-truncated content
        full = (
            "The user seems confused. They sent a short greeting. "
            "Let me offer them concrete options."
        )
        summary = adapter.convert_message(
            AssistantMessage(content=[TextBlock(text=full)], model="test")
        )
        combined = responses + summary
        assert _collect_text_deltas(combined) == full

    def test_partial_empty_summary_only(self):
        """Scenario 3: no partial deltas (CLI emitted the block entirely
        in the summary — short blocks, proxy buffering, encrypted
        content).  Summary carries the full text."""
        adapter = _adapter()
        summary = adapter.convert_message(
            AssistantMessage(content=[TextBlock(text="short answer")], model="test")
        )
        assert _collect_text_deltas(summary) == "short answer"

    def test_partial_long_summary_matches_no_double_emit(self):
        """Scenario 4 (most common): partial streams everything, summary
        repeats the same content.  No duplication on the wire."""
        adapter = _adapter()
        responses: list[StreamBaseResponse] = []
        full = "Here is a long paragraph with several words in it."
        adapter._handle_stream_event(_text_block_start(0), responses)
        # Partition into chunks that *exactly* reconstruct ``full`` — a
        # word-split with trailing spaces would emit more content than
        # the summary carries and the reconcile would correctly flag
        # divergence.
        chunks = [full[:13], full[13:25], full[25:]]
        assert "".join(chunks) == full
        for chunk in chunks:
            adapter._handle_stream_event(_text_delta(0, chunk), responses)
        adapter._handle_stream_event(_block_stop(0), responses)
        assert _collect_text_deltas(responses) == full

        summary = adapter.convert_message(
            AssistantMessage(content=[TextBlock(text=full)], model="test")
        )
        # Summary must not add any TextDelta since partial already covered it
        assert _collect_text_deltas(summary) == ""

    def test_partial_diverges_summary_wins(self):
        """Scenario 5: partial content isn't a prefix of the summary.
        Defensive path emits the full summary content — content must
        not silently disappear."""
        adapter = _adapter()
        responses: list[StreamBaseResponse] = []
        adapter._handle_stream_event(_text_block_start(0), responses)
        adapter._handle_stream_event(_text_delta(0, "first draft"), responses)
        # Summary has totally different content (proxy rewrote it)
        summary = adapter.convert_message(
            AssistantMessage(
                content=[TextBlock(text="final polished answer")],
                model="test",
            )
        )
        # The summary's text must reach the wire even though partial
        # already emitted "first draft" (which was the proxy's draft).
        assert "final polished answer" in _collect_text_deltas(responses + summary)

    def test_thinking_only_partial_coalesced(self):
        """Scenario 6a (thinking-only permutation): a run of
        ``thinking_delta`` events below the coalesce threshold flushes
        at ``content_block_stop`` so the reasoning tail isn't lost."""
        adapter = _adapter()
        responses: list[StreamBaseResponse] = []
        adapter._handle_stream_event(_thinking_block_start(0), responses)
        # Each chunk is well under the 64-char threshold
        for chunk in ("Let ", "me ", "think"):
            adapter._handle_stream_event(_thinking_delta(0, chunk), responses)
        # At stop, the pending buffer drains
        adapter._handle_stream_event(_block_stop(0), responses)
        assert _collect_reasoning_deltas(responses) == "Let me think"
        # Block closed
        assert any(isinstance(r, StreamReasoningEnd) for r in responses)

    def test_text_only_via_partial_and_summary(self):
        """Scenario 6b (text-only permutation): partial fills a block,
        summary matches — see scenario 4 for no-double-emit assertion."""
        adapter = _adapter()
        responses: list[StreamBaseResponse] = []
        adapter._handle_stream_event(_text_block_start(0), responses)
        adapter._handle_stream_event(_text_delta(0, "hi"), responses)
        adapter._handle_stream_event(_block_stop(0), responses)
        assert _collect_text_deltas(responses) == "hi"

    def test_mixed_text_then_thinking_partial_preserves_order(self):
        """Scenario 6c (mixed, Anthropic order — reasoning then text).
        When partial emits blocks in natural order and summary matches,
        the wire order is identical to emission order."""
        adapter = _adapter()
        responses: list[StreamBaseResponse] = []
        # Anthropic-shape: thinking index 0, text index 1
        adapter._handle_stream_event(_thinking_block_start(0), responses)
        adapter._handle_stream_event(
            _thinking_delta(0, "X" * 80), responses
        )  # over threshold
        adapter._handle_stream_event(_block_stop(0), responses)
        adapter._handle_stream_event(_text_block_start(1), responses)
        adapter._handle_stream_event(_text_delta(1, "answer"), responses)
        adapter._handle_stream_event(_block_stop(1), responses)
        types = [type(r).__name__ for r in responses]
        # ReasoningStart must come before TextStart — partial streams in
        # the CLI's natural order, which is also the UI's desired order.
        assert types.index("StreamReasoningStart") < types.index("StreamTextStart")

    def test_multi_message_turn_resets_per_index_maps(self):
        """Scenario 7: tool-use loop creates multiple AssistantMessages
        per turn.  Anthropic content-block indices are scoped to a single
        message — ``message_start`` must reset the diff maps so the next
        message's index-0 text isn't silently suppressed."""
        adapter = _adapter()
        responses: list[StreamBaseResponse] = []
        # First message at index 0 = "first"
        adapter._handle_stream_event(_message_start(), responses)
        adapter._handle_stream_event(_text_block_start(0), responses)
        adapter._handle_stream_event(_text_delta(0, "first"), responses)
        adapter._handle_stream_event(_block_stop(0), responses)
        # New message starts — index 0 now refers to a fresh block
        adapter._handle_stream_event(_message_start(), responses)
        adapter._handle_stream_event(_text_block_start(0), responses)
        adapter._handle_stream_event(_text_delta(0, "second"), responses)
        adapter._handle_stream_event(_block_stop(0), responses)
        # Both texts must land on the wire
        assert _collect_text_deltas(responses) == "firstsecond"

    def test_empty_thinking_with_signature_emits_nothing(self):
        """Scenario 8: encrypted / empty thinking block.  Partial emits
        nothing, summary carries ``block.thinking == ""`` with a
        signature — the adapter must not open a reasoning block."""
        adapter = _adapter()
        summary = adapter.convert_message(
            AssistantMessage(
                content=[ThinkingBlock(thinking="", signature="sig")],
                model="test",
            )
        )
        # No reasoning events should be emitted for empty thinking
        reasoning_events = [
            r
            for r in summary
            if isinstance(r, StreamReasoningDelta)
            or type(r).__name__ in ("StreamReasoningStart", "StreamReasoningEnd")
        ]
        assert reasoning_events == []

    def test_thinking_tail_drains_on_block_stop(self):
        """Scenario 10: a thinking_delta chunk smaller than the 64-char
        threshold arrives, then ``content_block_stop``.  The tail text
        must emit in a final ``StreamReasoningDelta`` BEFORE
        ``StreamReasoningEnd``."""
        adapter = _adapter()
        responses: list[StreamBaseResponse] = []
        adapter._handle_stream_event(_thinking_block_start(0), responses)
        # One small chunk well under 64 chars
        adapter._handle_stream_event(_thinking_delta(0, "tiny chunk"), responses)
        # Block stop must flush the pending buffer
        adapter._handle_stream_event(_block_stop(0), responses)
        types = [type(r).__name__ for r in responses]
        # The final ReasoningDelta must precede ReasoningEnd
        rd_idx = types.index("StreamReasoningDelta")
        re_idx = types.index("StreamReasoningEnd")
        assert rd_idx < re_idx
        assert _collect_reasoning_deltas(responses) == "tiny chunk"

    def test_thinking_coalesces_on_char_threshold(self):
        """Extra: thinking_delta accumulating past 64 chars flushes
        mid-block without waiting for block_stop (coalesce threshold)."""
        adapter = _adapter()
        responses: list[StreamBaseResponse] = []
        adapter._handle_stream_event(_thinking_block_start(0), responses)
        # One 80-char chunk trips the threshold on a single event
        adapter._handle_stream_event(_thinking_delta(0, "x" * 80), responses)
        # A ReasoningDelta must already have been emitted (not buffered
        # until block_stop).
        assert any(isinstance(r, StreamReasoningDelta) for r in responses)


# ---------------------------------------------------------------------------
# Partial/summary reconcile — summary walk must not duplicate partial content
# ---------------------------------------------------------------------------


def test_summary_walk_skips_fully_streamed_text():
    """If the partial stream delivered the entire TextBlock, the summary
    walk must not emit a second ``StreamTextDelta`` for the same block."""
    adapter = _adapter()
    responses: list[StreamBaseResponse] = []
    adapter._handle_stream_event(_text_block_start(0), responses)
    adapter._handle_stream_event(_text_delta(0, "complete answer"), responses)
    adapter._handle_stream_event(_block_stop(0), responses)
    # Summary arrives with matching content
    summary = adapter.convert_message(
        AssistantMessage(content=[TextBlock(text="complete answer")], model="test")
    )
    # Partial path emitted exactly one StreamTextDelta
    partial_deltas = [r for r in responses if isinstance(r, StreamTextDelta)]
    summary_deltas = [r for r in summary if isinstance(r, StreamTextDelta)]
    assert len(partial_deltas) == 1
    assert summary_deltas == []
