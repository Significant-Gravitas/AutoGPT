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
    ConversationUpdater,
    LLMCaller,
    LLMLoopResponse,
    LLMToolCall,
    ToolCallLoopResult,
    ToolCallResult,
    ToolExecutor,
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

    def update_conversation(
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None:
        messages.append({"role": "assistant", "content": response.response_text})

    results = await _run_loop("Hi", llm_call, _execute_never, update_conversation)

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

    results = await _run_loop("Weather?", llm_call, execute_tool, update_conversation)

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

    results = await _run_loop(
        "Go", llm_call, execute_tool, _skip_update, max_iterations=3
    )

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

    results = await _run_loop("Hi", llm_call, _execute_never, _skip_update, tools=[])

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

    with pytest.raises(RuntimeError, match="Tool execution failed!"):
        await _run_loop("Go", llm_call, execute_tool, _skip_update)


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

    await _run_loop("Run both", llm_call, execute_tool, _append_called_tools)

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

    await _run_loop(
        "Run both",
        llm_call,
        execute_tool,
        _append_called_tools,
        parallel_tool_calls=False,
    )

    # With sequential execution, tool_a runs first despite being slower
    assert execution_order == ["tool_a", "tool_b"]


async def _run_one_round(
    tool_calls: list[LLMToolCall],
    execute_tool,
    runs_alone=None,
) -> list[dict[str, Any]]:
    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        if len(messages) == 1:
            return _make_response(tool_calls=tool_calls)
        return _make_response(text="Done")

    results = await _run_loop(
        "go", llm_call, execute_tool, _append_called_tools, runs_alone=runs_alone
    )
    return [m for m in results[-1].messages if m["role"] == "tool"]


@pytest.mark.asyncio
async def test_copy_beside_the_command_building_the_file_gets_all_of_it(tmp_path):
    """SECRT-3178: a file copied in the same message as the command that
    builds it was stored half-written. With the builder running alone, the
    copy sees the whole file, and the results keep the model's order."""
    target = tmp_path / "out.zip"
    chunks = [b"PK\x03\x04", b"data" * 64, b"PK\x05\x06"]

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        if tool_call.name == "build":
            with target.open("wb") as f:
                for chunk in chunks:
                    f.write(chunk)
                    f.flush()
                    await asyncio.sleep(0.01)
            content = "built"
        else:
            content = str(len(target.read_bytes()) if target.exists() else 0)
        return ToolCallResult(
            tool_call_id=tool_call.id, tool_name=tool_call.name, content=content
        )

    results = await _run_one_round(
        [
            LLMToolCall(id="tc_build", name="build", arguments="{}"),
            LLMToolCall(id="tc_copy", name="copy", arguments="{}"),
        ],
        execute_tool,
        runs_alone=lambda tc: tc.name == "build",
    )

    assert [r["tool_call_id"] for r in results] == ["tc_build", "tc_copy"]
    assert results[1]["content"] == str(sum(len(c) for c in chunks))


@pytest.mark.asyncio
async def test_calls_between_lone_calls_still_overlap():
    """Only the picked calls lose parallelism: the calls on each side of
    one still run together, and no call crosses it."""
    running: set[str] = set()
    seen_together: list[set[str]] = []

    async def execute_tool(
        tool_call: LLMToolCall, tools: Sequence[Any]
    ) -> ToolCallResult:
        running.add(tool_call.name)
        await asyncio.sleep(0.01)
        seen_together.append(set(running))
        running.discard(tool_call.name)
        return ToolCallResult(
            tool_call_id=tool_call.id, tool_name=tool_call.name, content="ok"
        )

    names = ["read_a", "read_b", "write", "read_c", "read_d"]
    await _run_one_round(
        [LLMToolCall(id=n, name=n, arguments="{}") for n in names],
        execute_tool,
        runs_alone=lambda tc: tc.name == "write",
    )

    assert {"read_a", "read_b"} in seen_together
    assert {"write"} in seen_together
    assert {"read_c", "read_d"} in seen_together
    assert all(
        s <= {"read_a", "read_b"} or s == {"write"} or s <= {"read_c", "read_d"}
        for s in seen_together
    )


@pytest.mark.asyncio
async def test_last_iteration_message_appended():
    """On the final iteration, last_iteration_message should be appended
    and ``tools`` should be empty so the model is forced to produce text."""
    captured_messages: list[list[dict[str, Any]]] = []
    captured_tools: list[Sequence[Any]] = []

    async def llm_call(
        messages: list[dict[str, Any]], tools: Sequence[Any]
    ) -> LLMLoopResponse:
        captured_messages.append(list(messages))
        captured_tools.append(tools)
        return _make_response(
            tool_calls=[LLMToolCall(id="tc_1", name="get_weather", arguments="{}")]
        )

    await _run_loop(
        "Go",
        llm_call,
        _execute_ok,
        _skip_update,
        max_iterations=2,
        last_iteration_message="Please finish now.",
    )

    # First iteration: no extra message, tools available
    assert len(captured_messages[0]) == 1
    assert list(captured_tools[0]) == list(TOOL_DEFS)
    # Second (last) iteration: hint appended, tools dropped (forces the
    # model to produce a text response instead of another tool call).
    last_call_msgs = captured_messages[1]
    assert any(
        m.get("role") == "system" and "Please finish now." in m.get("content", "")
        for m in last_call_msgs
    )
    assert list(captured_tools[1]) == []


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

    results = await _run_loop("Go", llm_call, _execute_ok, _skip_update)

    final_result = results[-1]
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

    results = await _run_loop(
        "Go", llm_call, execute_tool, update_conversation, max_iterations=0
    )

    assert len(results) == 1
    assert results[0].finished_naturally is False
    assert results[0].iterations == 0
    assert "0 iterations" in results[0].response_text


async def _run_loop(
    prompt: str,
    llm_call: LLMCaller,
    execute_tool: ToolExecutor,
    update_conversation: ConversationUpdater,
    tools: Sequence[Any] = TOOL_DEFS,
    **kwargs: Any,
) -> list[ToolCallLoopResult]:
    msgs: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    return [
        r
        async for r in tool_call_loop(
            messages=msgs,
            tools=tools,
            llm_call=llm_call,
            execute_tool=execute_tool,
            update_conversation=update_conversation,
            **kwargs,
        )
    ]


async def _execute_never(
    tool_call: LLMToolCall, tools: Sequence[Any]
) -> ToolCallResult:
    raise AssertionError("Should not be called")


async def _execute_ok(tool_call: LLMToolCall, tools: Sequence[Any]) -> ToolCallResult:
    return ToolCallResult(
        tool_call_id=tool_call.id, tool_name=tool_call.name, content="ok"
    )


def _skip_update(
    messages: list[dict[str, Any]],
    response: LLMLoopResponse,
    tool_results: list[ToolCallResult] | None = None,
) -> None:
    pass


def _append_called_tools(
    messages: list[dict[str, Any]],
    response: LLMLoopResponse,
    tool_results: list[ToolCallResult] | None = None,
) -> None:
    messages.append({"role": "assistant", "content": "called tools"})
    for tr in tool_results or []:
        messages.append(
            {"role": "tool", "tool_call_id": tr.tool_call_id, "content": tr.content}
        )
