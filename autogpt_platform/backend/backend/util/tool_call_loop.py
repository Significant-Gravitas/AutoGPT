"""Shared tool-calling conversation loop.

Provides a generic, provider-agnostic conversation loop that both
the ToolOrchestratorBlock and copilot baseline can use. The loop:

1. Calls the LLM with tool definitions
2. Extracts tool calls from the response
3. Executes tools via a caller-supplied callback
4. Appends results to the conversation
5. Repeats until no more tool calls or max iterations reached

Callers provide callbacks for LLM calling, tool execution, and
conversation updating. The loop is an async generator that yields
``ToolCallLoopEvent`` objects for streaming-capable callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ToolCallResult:
    """Result of a single tool execution."""

    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False


@dataclass
class LLMToolCall:
    """A tool call extracted from an LLM response."""

    id: str
    name: str
    arguments: str  # JSON string


@dataclass
class LLMLoopResponse:
    """Response from a single LLM call in the loop."""

    response_text: str | None
    tool_calls: list[LLMToolCall]
    raw_response: Any  # Provider-specific raw response for conversation history
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning: str | None = None


class LLMCaller(Protocol):
    """Protocol for LLM call functions."""

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMLoopResponse: ...


class ToolExecutor(Protocol):
    """Protocol for tool execution functions."""

    async def __call__(
        self,
        tool_call: LLMToolCall,
        tools: list[dict[str, Any]],
    ) -> ToolCallResult: ...


class ConversationUpdater(Protocol):
    """Protocol for updating conversation history after an LLM response."""

    def __call__(
        self,
        messages: list[dict[str, Any]],
        response: LLMLoopResponse,
        tool_results: list[ToolCallResult] | None = None,
    ) -> None: ...


@dataclass
class ToolCallLoopResult:
    """Final result of the tool-calling loop."""

    response_text: str
    messages: list[dict[str, Any]]
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    iterations: int = 0
    finished_naturally: bool = True  # False if hit max iterations


async def tool_call_loop(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    llm_call: LLMCaller,
    execute_tool: ToolExecutor,
    update_conversation: ConversationUpdater,
    max_iterations: int = -1,
    last_iteration_message: str | None = None,
) -> ToolCallLoopResult:
    """Run a tool-calling conversation loop.

    Args:
        messages: Initial conversation messages (modified in-place).
        tools: Tool function definitions (OpenAI format).
        llm_call: Async function to call the LLM. The callback can
            perform streaming internally (e.g. accumulate text deltas
            and collect events) — it just needs to return the final
            ``LLMLoopResponse`` with extracted tool calls.
        execute_tool: Async function to execute a tool call.
        update_conversation: Function to update messages with LLM
            response and tool results.
        max_iterations: Max iterations. -1 = infinite, 0 would not loop.
        last_iteration_message: Optional message to append on the last
            iteration to encourage the model to finish.

    Returns:
        ToolCallLoopResult with the final response and conversation state.
    """
    result = ToolCallLoopResult(
        response_text="",
        messages=messages,
    )
    iteration = 0

    while max_iterations < 0 or iteration < max_iterations:
        iteration += 1
        result.iterations = iteration

        # On last iteration, add a hint to finish
        iteration_messages = list(messages)
        if (
            last_iteration_message
            and max_iterations > 0
            and iteration == max_iterations
        ):
            iteration_messages.append(
                {"role": "system", "content": last_iteration_message}
            )

        # Call LLM
        response = await llm_call(iteration_messages, tools)
        result.total_prompt_tokens += response.prompt_tokens
        result.total_completion_tokens += response.completion_tokens

        # No tool calls = done
        if not response.tool_calls:
            result.response_text = response.response_text or ""
            update_conversation(messages, response)
            result.finished_naturally = True
            return result

        # Execute tools
        tool_results: list[ToolCallResult] = []
        for tc in response.tool_calls:
            tr = await execute_tool(tc, tools)
            tool_results.append(tr)

        # Update conversation with response + tool results
        update_conversation(messages, response, tool_results)

    # Hit max iterations
    result.response_text = (
        f"Completed after {max_iterations} iterations (limit reached)"
    )
    result.finished_naturally = False
    return result
