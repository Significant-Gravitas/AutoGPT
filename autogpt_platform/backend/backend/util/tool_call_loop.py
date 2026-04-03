"""Shared tool-calling conversation loop.

Provides a generic, provider-agnostic conversation loop that both
the OrchestratorBlock and copilot baseline can use. The loop:

1. Calls the LLM with tool definitions
2. Extracts tool calls from the response
3. Executes tools via a caller-supplied callback
4. Appends results to the conversation
5. Repeats until no more tool calls or max iterations reached

Callers provide callbacks for LLM calling, tool execution, and
conversation updating.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed dict definitions for tool definitions and conversation messages.
# These document the expected shapes and allow callers to pass TypedDict
# subclasses (e.g. ``ChatCompletionToolParam``) without ``type: ignore``.
# ---------------------------------------------------------------------------


class FunctionParameters(TypedDict, total=False):
    """JSON Schema object describing a tool function's parameters."""

    type: str
    properties: dict[str, Any]
    required: list[str]
    additionalProperties: bool


class FunctionDefinition(TypedDict, total=False):
    """Function definition within a tool definition."""

    name: str
    description: str
    parameters: FunctionParameters


class ToolDefinition(TypedDict):
    """OpenAI-compatible tool definition (function-calling format).

    Compatible with ``openai.types.chat.ChatCompletionToolParam`` and the
    dict-based tool definitions built by ``OrchestratorBlock``.
    """

    type: str
    function: FunctionDefinition


class ConversationMessage(TypedDict, total=False):
    """A single message in the conversation (OpenAI chat format).

    Primarily for documentation; at runtime plain dicts are used because
    messages from different providers carry varying keys.
    """

    role: str
    content: str | list[Any] | None
    tool_calls: list[dict[str, Any]]
    tool_call_id: str
    name: str


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
    """Response from a single LLM call in the loop.

    ``raw_response`` is typed as ``Any`` intentionally: the loop itself
    never inspects it — it is an opaque pass-through that the caller's
    ``ConversationUpdater`` uses to rebuild provider-specific message
    history (OpenAI ChatCompletion, Anthropic Message, Ollama str, etc.).
    """

    response_text: str | None
    tool_calls: list[LLMToolCall]
    raw_response: Any
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning: str | None = None


class LLMCaller(Protocol):
    """Protocol for LLM call functions."""

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[Any],
    ) -> LLMLoopResponse: ...


class ToolExecutor(Protocol):
    """Protocol for tool execution functions."""

    async def __call__(
        self,
        tool_call: LLMToolCall,
        tools: Sequence[Any],
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
    last_tool_calls: list[LLMToolCall] = field(default_factory=list)


async def tool_call_loop(
    *,
    messages: list[dict[str, Any]],
    tools: Sequence[Any],
    llm_call: LLMCaller,
    execute_tool: ToolExecutor,
    update_conversation: ConversationUpdater,
    max_iterations: int = -1,
    last_iteration_message: str | None = None,
    parallel_tool_calls: bool = True,
) -> AsyncGenerator[ToolCallLoopResult, None]:
    """Run a tool-calling conversation loop as an async generator.

    Yields a ``ToolCallLoopResult`` after each iteration so callers can
    drain buffered events (e.g. streaming text deltas) between iterations.
    The **final** yielded result has ``finished_naturally`` set and contains
    the complete response text.

    Args:
        messages: Initial conversation messages (modified in-place).
        tools: Tool function definitions (OpenAI format).  Accepts any
            sequence of tool dicts, including ``ChatCompletionToolParam``.
        llm_call: Async function to call the LLM. The callback can
            perform streaming internally (e.g. accumulate text deltas
            and collect events) — it just needs to return the final
            ``LLMLoopResponse`` with extracted tool calls.
        execute_tool: Async function to execute a tool call.
        update_conversation: Function to update messages with LLM
            response and tool results.
        max_iterations: Max iterations. -1 = infinite, 0 = no loop
            (immediately yields a "max reached" result).
        last_iteration_message: Optional message to append on the last
            iteration to encourage the model to finish.
        parallel_tool_calls: If True (default), execute multiple tool
            calls from a single LLM response concurrently via
            ``asyncio.gather``.  Set to False when tool calls may have
            ordering dependencies or mutate shared state.

    Yields:
        ToolCallLoopResult after each iteration. Check ``finished_naturally``
        to determine if the loop completed or is still running.
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    iteration = 0

    while max_iterations < 0 or iteration < max_iterations:
        iteration += 1

        # On last iteration, add a hint to finish.  Only copy the list
        # when the hint needs to be appended to avoid per-iteration overhead
        # on long conversations.
        is_last = (
            last_iteration_message
            and max_iterations > 0
            and iteration == max_iterations
        )
        if is_last:
            iteration_messages = list(messages)
            iteration_messages.append(
                {"role": "system", "content": last_iteration_message}
            )
        else:
            iteration_messages = messages

        # Call LLM
        response = await llm_call(iteration_messages, tools)
        total_prompt_tokens += response.prompt_tokens
        total_completion_tokens += response.completion_tokens

        # No tool calls = done
        if not response.tool_calls:
            update_conversation(messages, response)
            yield ToolCallLoopResult(
                response_text=response.response_text or "",
                messages=messages,
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                iterations=iteration,
                finished_naturally=True,
            )
            return

        # Execute tools — parallel or sequential depending on caller preference.
        # NOTE: asyncio.gather does not cancel sibling tasks when one raises.
        # Callers should handle errors inside execute_tool (return error
        # ToolCallResult) rather than letting exceptions propagate.
        if parallel_tool_calls and len(response.tool_calls) > 1:
            # Parallel: side-effects from different tool executors (e.g.
            # streaming events appended to a shared list) may interleave
            # nondeterministically.  Each event carries its own tool-call
            # identifier, so consumers must correlate by ID.
            tool_results: list[ToolCallResult] = list(
                await asyncio.gather(
                    *(execute_tool(tc, tools) for tc in response.tool_calls)
                )
            )
        else:
            # Sequential: preserves ordering guarantees for callers that
            # need deterministic execution order.
            tool_results = [await execute_tool(tc, tools) for tc in response.tool_calls]

        # Update conversation with response + tool results
        update_conversation(messages, response, tool_results)

        # Yield a fresh result so callers can drain buffered events
        yield ToolCallLoopResult(
            response_text="",
            messages=messages,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            iterations=iteration,
            finished_naturally=False,
            last_tool_calls=list(response.tool_calls),
        )

    # Hit max iterations
    yield ToolCallLoopResult(
        response_text=f"Completed after {max_iterations} iterations (limit reached)",
        messages=messages,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        iterations=iteration,
        finished_naturally=False,
    )
