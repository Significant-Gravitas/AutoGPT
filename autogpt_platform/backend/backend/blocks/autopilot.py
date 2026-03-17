from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict  # Needed for Python <3.12 compatibility

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

if TYPE_CHECKING:
    from backend.data.execution import ExecutionContext

logger = logging.getLogger(__name__)

# Block ID shared between autopilot.py and copilot prompting.py.
AUTOPILOT_BLOCK_ID = "c069dc6b-c3ed-4c12-b6e5-d47361e64ce6"

# Maximum history messages to serialize (caps memory for resumed sessions).
_MAX_HISTORY_MESSAGES = 200


class ToolCallEntry(TypedDict):
    """A single tool invocation record from an autopilot execution."""

    tool_call_id: str
    tool_name: str
    input: Any
    output: Any | None
    success: bool | None


class TokenUsage(TypedDict):
    """Aggregated token counts from the autopilot stream."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AutoPilotBlock(Block):
    """Execute tasks using AutoGPT AutoPilot with full access to platform tools.

    The autopilot can manage agents, access workspace files, fetch web content,
    run blocks, and more. This block enables sub-agent patterns (autopilot calling
    autopilot) and scheduled autopilot execution via the agent executor.
    """

    class Input(BlockSchemaInput):
        """Input schema for the AutoPilot block."""

        prompt: str = SchemaField(
            description=(
                "The task or instruction for the autopilot to execute. "
                "The autopilot has access to platform tools like agent management, "
                "workspace files, web fetch, block execution, and more."
            ),
            placeholder="Find my agents and list them",
            advanced=False,
        )

        system_context: str = SchemaField(
            description=(
                "Optional additional context prepended to the prompt. "
                "Use this to constrain autopilot behavior, provide domain "
                "context, or set output format requirements."
            ),
            default="",
            advanced=True,
        )

        session_id: str = SchemaField(
            description=(
                "Session ID to continue an existing autopilot conversation. "
                "Leave empty to start a new session. "
                "Use the session_id output from a previous run to continue."
            ),
            default="",
            advanced=True,
        )

        max_recursion_depth: int = SchemaField(
            description=(
                "Maximum nesting depth when the autopilot calls this block "
                "recursively (sub-agent pattern). Prevents infinite loops."
            ),
            default=3,
            ge=1,
            le=10,
            advanced=True,
        )

        timeout_seconds: int = SchemaField(
            description=(
                "Maximum execution time in seconds. The autopilot stream will "
                "be cancelled if it exceeds this limit, preventing indefinite "
                "executor slot occupation."
            ),
            default=300,
            ge=10,
            le=3600,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        """Output schema for the AutoPilot block."""

        response: str = SchemaField(
            description="The final text response from the autopilot."
        )
        tool_calls: list[ToolCallEntry] = SchemaField(
            description=(
                "List of tools called during execution. Each entry has "
                "tool_call_id, tool_name, input, output, and success fields."
            ),
        )
        conversation_history: str = SchemaField(
            description=(
                "Full conversation history as JSON. "
                "It can be used for logging or analysis."
            ),
        )
        session_id: str = SchemaField(
            description=(
                "Session ID for this conversation. "
                "Pass this back to continue the conversation in a future run."
            ),
        )
        token_usage: TokenUsage = SchemaField(
            description=(
                "Token usage statistics: prompt_tokens, "
                "completion_tokens, total_tokens."
            ),
        )

    def __init__(self):
        super().__init__(
            id=AUTOPILOT_BLOCK_ID,
            description=(
                "Execute tasks using AutoGPT AutoPilot with full access to "
                "platform tools (agent management, workspace files, web fetch, "
                "block execution, and more). Enables sub-agent patterns and "
                "scheduled autopilot execution."
            ),
            categories={BlockCategory.AI, BlockCategory.AGENT},
            input_schema=AutoPilotBlock.Input,
            output_schema=AutoPilotBlock.Output,
            test_input={
                "prompt": "List my agents",
                "system_context": "",
                "session_id": "",
                "max_recursion_depth": 3,
                "timeout_seconds": 300,
            },
            test_output=[
                ("response", "You have 2 agents: Agent A and Agent B."),
                ("tool_calls", []),
                (
                    "conversation_history",
                    '[{"role": "user", "content": "List my agents"}]',
                ),
                ("session_id", "test-session-id"),
                (
                    "token_usage",
                    {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                ),
            ],
            test_mock={
                "create_session": lambda *args, **kwargs: "test-session-id",
                "execute_copilot": lambda *args, **kwargs: (
                    "You have 2 agents: Agent A and Agent B.",
                    [],
                    '[{"role": "user", "content": "List my agents"}]',
                    "test-session-id",
                    {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                ),
            },
        )

    async def create_session(self, user_id: str) -> str:
        """Create a new chat session and return its ID (mockable for tests)."""
        from backend.copilot.model import create_chat_session

        session = await create_chat_session(user_id)
        return session.session_id

    async def execute_copilot(
        self,
        prompt: str,
        system_context: str,
        session_id: str,
        max_recursion_depth: int,
        user_id: str,
    ) -> tuple[str, list[ToolCallEntry], str, str, TokenUsage]:
        """Invoke the copilot and collect all stream results.

        Follows the same path as the normal copilot: create session if needed,
        then let stream_chat_completion_sdk handle everything (session loading,
        message append, lock, transcript, cleanup).

        Args:
            prompt: The user task/instruction.
            system_context: Optional context prepended to the prompt.
            session_id: Chat session to use.
            max_recursion_depth: Maximum allowed recursion nesting.
            user_id: Authenticated user ID.

        Returns:
            A tuple of (response_text, tool_calls, history_json, session_id, usage).
        """
        from backend.copilot.response_model import (
            StreamError,
            StreamTextDelta,
            StreamToolInputAvailable,
            StreamToolOutputAvailable,
            StreamUsage,
        )
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        # NOTE: This calls stream_chat_completion_sdk directly in the graph
        # executor process rather than proxying through the Copilot Executor
        # service.  If the copilot executor env diverges (different SDK version,
        # tool set, or model config) this block will follow the graph executor's
        # env.  Keep shared copilot deps aligned or proxy via the executor.
        tokens = _check_recursion(max_recursion_depth)
        try:
            effective_prompt = prompt
            if system_context:
                effective_prompt = f"[System Context: {system_context}]\n\n{prompt}"

            # Consume the stream — same as the executor processor.
            # Do NOT pass a session object; let the SDK load it internally
            # so all session management (lock, persist, transcript) is handled
            # by the SDK's own finally block.
            response_parts: list[str] = []
            tool_calls: list[ToolCallEntry] = []
            tool_calls_by_id: dict[str, ToolCallEntry] = {}
            total_usage: TokenUsage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

            async for event in stream_chat_completion_sdk(
                session_id=session_id,
                message=effective_prompt,
                is_user_message=True,
                user_id=user_id,
            ):
                if isinstance(event, StreamTextDelta):
                    response_parts.append(event.delta)
                elif isinstance(event, StreamToolInputAvailable):
                    entry: ToolCallEntry = {
                        "tool_call_id": event.toolCallId,
                        "tool_name": event.toolName,
                        "input": event.input,
                        "output": None,
                        "success": None,
                    }
                    tool_calls.append(entry)
                    tool_calls_by_id[event.toolCallId] = entry
                elif isinstance(event, StreamToolOutputAvailable):
                    if tc := tool_calls_by_id.get(event.toolCallId):
                        tc["output"] = event.output
                        tc["success"] = event.success
                elif isinstance(event, StreamUsage):
                    total_usage["prompt_tokens"] += event.prompt_tokens
                    total_usage["completion_tokens"] += event.completion_tokens
                    total_usage["total_tokens"] += event.total_tokens
                elif isinstance(event, StreamError):
                    raise RuntimeError(f"AutoPilot error: {event.errorText}")

            # Build a lightweight conversation summary from streamed data.
            # The SDK already persists the full session; avoid a redundant DB
            # fetch by reconstructing the current turn from what we collected.
            response_text = "".join(response_parts)
            turn_messages: list[dict[str, Any]] = [
                {"role": "user", "content": effective_prompt},
            ]
            if tool_calls:
                turn_messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "tool_calls": tool_calls,
                    }
                )
            else:
                turn_messages.append({"role": "assistant", "content": response_text})
            # Cap serialized messages to bound memory.
            history_json = json.dumps(
                turn_messages[-_MAX_HISTORY_MESSAGES:], default=str
            )

            return (
                response_text,
                tool_calls,
                history_json,
                session_id,
                total_usage,
            )
        finally:
            _reset_recursion(tokens)

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        """Validate inputs, invoke the autopilot, and yield structured outputs.

        Yields session_id even on failure so callers can inspect/resume the session.
        """
        if not input_data.prompt.strip():
            yield "error", "Prompt cannot be empty."
            return

        if not execution_context.user_id:
            yield "error", "Cannot run autopilot without an authenticated user."
            return

        if input_data.max_recursion_depth < 1:
            yield "error", "max_recursion_depth must be at least 1."
            return

        # Create session eagerly so the user always gets the session_id,
        # even if the downstream stream fails (avoids orphaned sessions).
        sid = input_data.session_id
        if not sid:
            sid = await self.create_session(execution_context.user_id)

        timeout = input_data.timeout_seconds
        try:
            async with asyncio.timeout(timeout):
                response, tool_calls, history, _, usage = await self.execute_copilot(
                    prompt=input_data.prompt,
                    system_context=input_data.system_context,
                    session_id=sid,
                    max_recursion_depth=input_data.max_recursion_depth,
                    user_id=execution_context.user_id,
                )

            yield "response", response
            yield "tool_calls", tool_calls
            yield "conversation_history", history
            yield "session_id", sid
            yield "token_usage", usage
        except TimeoutError:
            logger.warning(
                "AutoPilot execution timed out after %ds for session %s",
                timeout,
                sid,
            )
            yield "session_id", sid
            yield "error", f"AutoPilot execution timed out after {timeout}s."
        except asyncio.CancelledError:
            yield "session_id", sid
            yield "error", "AutoPilot execution was cancelled."
            raise


# ---------------------------------------------------------------------------
# Helpers – placed after the block class for top-down readability.
# ---------------------------------------------------------------------------

# Task-scoped recursion depth counter & chain-wide limit.
# contextvars are scoped to the current asyncio task, so concurrent
# graph executions each get independent counters.
_autopilot_recursion_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_autopilot_recursion_depth", default=0
)
_autopilot_recursion_limit: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_autopilot_recursion_limit", default=None
)


def _check_recursion(
    max_depth: int,
) -> tuple[contextvars.Token[int], contextvars.Token[int | None]]:
    """Check and increment recursion depth.

    Returns ContextVar tokens that must be passed to ``_reset_recursion``
    when the caller exits to restore the previous depth.

    Raises:
        RuntimeError: If the current depth already meets or exceeds the limit.
    """
    current = _autopilot_recursion_depth.get()
    inherited = _autopilot_recursion_limit.get()
    limit = max_depth if inherited is None else min(inherited, max_depth)
    if current >= limit:
        raise RuntimeError(
            f"AutoPilot recursion depth limit reached ({limit}). "
            "The autopilot has called itself too many times."
        )
    return (
        _autopilot_recursion_depth.set(current + 1),
        _autopilot_recursion_limit.set(limit),
    )


def _reset_recursion(
    tokens: tuple[contextvars.Token[int], contextvars.Token[int | None]],
) -> None:
    """Restore recursion depth and limit to their previous values."""
    _autopilot_recursion_depth.reset(tokens[0])
    _autopilot_recursion_limit.reset(tokens[1])
