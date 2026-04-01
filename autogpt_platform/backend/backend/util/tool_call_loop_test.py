"""Unit tests for tool_call_loop shared abstraction.

Covers:
- Happy path with tool calls (single and multi-round)
- Final text response (no tool calls)
- Max iterations reached
- No tools scenario
- Exception propagation from tool executor
- Parallel tool execution
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from backend.util.tool_call_loop import (
    LLMLoopResponse,
    LLMToolCall,
    ToolCallLoopResult,
    ToolCallResult,
    tool_call_loop,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_DEFS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def _make_response(
    text: str | None = None,
    tool_calls: list[LLMToolCall] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> LLMLoopResponse:
    return LLMLoopResponse(
        response_text=text,
        tool_calls=tool_calls or [],
        raw_response={"mock": True},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_response_no_tool_calls():
    """LLM responds with text only -- loop should yield once and finish."""

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        return _make_response(text="Hello world")

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        raise AssertionError("Should not be called")

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        messages.append({"role": "assistant", "content": response.response_text})

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Hi"}]
    results: list[ToolCallLoopResult] = []
    async for r in tool_call_loop(
        messages=msgs,
        tools=TOOL_DEFS,
        llm_call=llm_call,
        execute_tool=execute_tool,
        update_conversation=update_conversation,
    ):
        results.append(r)

    assert len(results) == 1
    assert results[0].finished_naturally is True
    assert results[0].response_text == "Hello world"
    assert results[0].iterations == 1
    assert results[0].total_prompt_tokens == 10
    assert results[0].total_completion_tokens == 5


@pytest.mark.asyncio
async def test_single_tool_call_then_text():
    """LLM makes one tool call, then responds with text on second round."""
    call_count = 0

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_response(
                tool_calls=[
                    LLMToolCall(
                        id="tc_1", name="get_weather", arguments='{"city":"NYC"}'
                    )
                ]
            )
        return _make_response(text="It's sunny in NYC")

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        return ToolCallResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content='{"temp": 72}',
        )

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        messages.append({"role": "assistant", "content": response.response_text})
        if tool_results:
            for tr in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": tr.content,
                    }
                )

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Weather?"}]
    results: list[ToolCallLoopResult] = []
    async for r in tool_call_loop(
        messages=msgs,
        tools=TOOL_DEFS,
        llm_call=llm_call,
        execute_tool=execute_tool,
        update_conversation=update_conversation,
    ):
        results.append(r)

    # First yield: tool call iteration (not finished)
    # Second yield: text response (finished)
    assert len(results) == 2
    assert results[0].finished_naturally is False
    assert results[0].iterations == 1
    assert len(results[0].last_tool_calls) == 1
    assert results[1].finished_naturally is True
    assert results[1].response_text == "It's sunny in NYC"
    assert results[1].iterations == 2
    assert results[1].total_prompt_tokens == 20
    assert results[1].total_completion_tokens == 10


@pytest.mark.asyncio
async def test_max_iterations_reached():
    """Loop should stop after max_iterations even if LLM keeps calling tools."""

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        return _make_response(
            tool_calls=[
                LLMToolCall(id="tc_x", name="get_weather", arguments='{"city":"X"}')
            ]
        )

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        return ToolCallResult(
            tool_call_id=tool_call.id, tool_name=tool_call.name, content="result"
        )

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        pass

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Go"}]
    results: list[ToolCallLoopResult] = []
    async for r in tool_call_loop(
        messages=msgs,
        tools=TOOL_DEFS,
        llm_call=llm_call,
        execute_tool=execute_tool,
        update_conversation=update_conversation,
        max_iterations=3,
    ):
        results.append(r)

    # 3 tool-call iterations + 1 final "max reached"
    assert len(results) == 4
    for r in results[:3]:
        assert r.finished_naturally is False
    final = results[-1]
    assert final.finished_naturally is False
    assert "3 iterations" in final.response_text
    assert final.iterations == 3


@pytest.mark.asyncio
async def test_no_tools_first_response_text():
    """When LLM immediately responds with text (empty tools list), finishes."""

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        return _make_response(text="No tools needed")

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        raise AssertionError("Should not be called")

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        pass

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Hi"}]
    results: list[ToolCallLoopResult] = []
    async for r in tool_call_loop(
        messages=msgs,
        tools=[],
        llm_call=llm_call,
        execute_tool=execute_tool,
        update_conversation=update_conversation,
    ):
        results.append(r)

    assert len(results) == 1
    assert results[0].finished_naturally is True
    assert results[0].response_text == "No tools needed"


@pytest.mark.asyncio
async def test_tool_executor_exception_propagates():
    """Exception in execute_tool should propagate out of the loop."""

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        return _make_response(
            tool_calls=[LLMToolCall(id="tc_err", name="get_weather", arguments="{}")]
        )

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        raise RuntimeError("Tool execution failed!")

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        pass

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Go"}]
    with pytest.raises(RuntimeError, match="Tool execution failed!"):
        async for _ in tool_call_loop(
            messages=msgs,
            tools=TOOL_DEFS,
            llm_call=llm_call,
            execute_tool=execute_tool,
            update_conversation=update_conversation,
        ):
            pass


@pytest.mark.asyncio
async def test_parallel_tool_execution():
    """Multiple tool calls in one response should execute concurrently."""
    execution_order: list[str] = []

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        if len(messages) == 1:
            return _make_response(
                tool_calls=[
                    LLMToolCall(id="tc_a", name="tool_a", arguments="{}"),
                    LLMToolCall(id="tc_b", name="tool_b", arguments="{}"),
                ]
            )
        return _make_response(text="Done")

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        # tool_b starts instantly, tool_a has a small delay.
        # With parallel execution, both should overlap.
        if tool_call.name == "tool_a":
            await asyncio.sleep(0.05)
        execution_order.append(tool_call.name)
        return ToolCallResult(
            tool_call_id=tool_call.id, tool_name=tool_call.name, content="ok"
        )

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        messages.append({"role": "assistant", "content": "called tools"})
        if tool_results:
            for tr in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": tr.content,
                    }
                )

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Run both"}]
    async for _ in tool_call_loop(
        messages=msgs,
        tools=TOOL_DEFS,
        llm_call=llm_call,
        execute_tool=execute_tool,
        update_conversation=update_conversation,
    ):
        pass

    # With parallel execution, tool_b (no delay) finishes before tool_a
    assert execution_order == ["tool_b", "tool_a"]


@pytest.mark.asyncio
async def test_sequential_tool_execution():
    """With parallel_tool_calls=False, tools execute in order regardless of speed."""
    execution_order: list[str] = []

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        if len(messages) == 1:
            return _make_response(
                tool_calls=[
                    LLMToolCall(id="tc_a", name="tool_a", arguments="{}"),
                    LLMToolCall(id="tc_b", name="tool_b", arguments="{}"),
                ]
            )
        return _make_response(text="Done")

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        # tool_b would finish first if parallel, but sequential should keep order
        if tool_call.name == "tool_a":
            await asyncio.sleep(0.05)
        execution_order.append(tool_call.name)
        return ToolCallResult(
            tool_call_id=tool_call.id, tool_name=tool_call.name, content="ok"
        )

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        messages.append({"role": "assistant", "content": "called tools"})
        if tool_results:
            for tr in tool_results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_call_id,
                        "content": tr.content,
                    }
                )

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Run both"}]
    async for _ in tool_call_loop(
        messages=msgs,
        tools=TOOL_DEFS,
        llm_call=llm_call,
        execute_tool=execute_tool,
        update_conversation=update_conversation,
        parallel_tool_calls=False,
    ):
        pass

    # With sequential execution, tool_a runs first despite being slower
    assert execution_order == ["tool_a", "tool_b"]


@pytest.mark.asyncio
async def test_last_iteration_message_appended():
    """On the final iteration, last_iteration_message should be appended."""
    captured_messages: list[list[dict[str, Any]]] = []

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        captured_messages.append(list(messages))
        return _make_response(
            tool_calls=[LLMToolCall(id="tc_1", name="get_weather", arguments="{}")]
        )

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        return ToolCallResult(
            tool_call_id=tool_call.id, tool_name=tool_call.name, content="ok"
        )

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        pass

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Go"}]
    async for _ in tool_call_loop(
        messages=msgs,
        tools=TOOL_DEFS,
        llm_call=llm_call,
        execute_tool=execute_tool,
        update_conversation=update_conversation,
        max_iterations=2,
        last_iteration_message="Please finish now.",
    ):
        pass

    # First iteration: no extra message
    assert len(captured_messages[0]) == 1
    # Second (last) iteration: should have the hint appended
    last_call_msgs = captured_messages[1]
    assert any(
        m.get("role") == "system" and "Please finish now." in m.get("content", "")
        for m in last_call_msgs
    )


@pytest.mark.asyncio
async def test_token_accumulation():
    """Tokens should accumulate across iterations."""
    call_count = 0

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return _make_response(
                tool_calls=[
                    LLMToolCall(
                        id=f"tc_{call_count}", name="get_weather", arguments="{}"
                    )
                ],
                prompt_tokens=100,
                completion_tokens=50,
            )
        return _make_response(text="Final", prompt_tokens=100, completion_tokens=50)

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        return ToolCallResult(
            tool_call_id=tool_call.id, tool_name=tool_call.name, content="ok"
        )

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        pass

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Go"}]
    final_result = None
    async for r in tool_call_loop(
        messages=msgs,
        tools=TOOL_DEFS,
        llm_call=llm_call,
        execute_tool=execute_tool,
        update_conversation=update_conversation,
    ):
        final_result = r

    assert final_result is not None
    assert final_result.total_prompt_tokens == 300  # 3 calls * 100
    assert final_result.total_completion_tokens == 150  # 3 calls * 50
    assert final_result.iterations == 3


@pytest.mark.asyncio
async def test_max_iterations_zero_no_loop():
    """max_iterations=0 should immediately yield a 'max reached' result without calling LLM."""

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        raise AssertionError("LLM should not be called when max_iterations=0")

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        raise AssertionError("Tool should not be called when max_iterations=0")

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        raise AssertionError("Updater should not be called when max_iterations=0")

    msgs: list[dict[str, Any]] = [{"role": "user", "content": "Go"}]
    results: list[ToolCallLoopResult] = []
    async for r in tool_call_loop(
        messages=msgs,
        tools=TOOL_DEFS,
        llm_call=llm_call,
        execute_tool=execute_tool,
        update_conversation=update_conversation,
        max_iterations=0,
    ):
        results.append(r)

    assert len(results) == 1
    assert results[0].finished_naturally is False
    assert results[0].iterations == 0
    assert "0 iterations" in results[0].response_text
