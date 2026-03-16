from __future__ import annotations

import contextvars
import json
from typing import TYPE_CHECKING

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

if TYPE_CHECKING:
    from backend.copilot.model import ChatSession
    from backend.executor.utils import ExecutionContext

# Task-scoped recursion depth counter & chain-wide limit.
# contextvars are scoped to the current asyncio task, so concurrent
# graph executions each get independent counters.
_copilot_recursion_depth: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_copilot_recursion_depth", default=0
)
_copilot_recursion_limit: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "_copilot_recursion_limit", default=None
)


class AutogptCopilotBlock(Block):
    """Execute tasks using the AutoGPT Copilot with full access to platform tools.

    The copilot can manage agents, access workspace files, fetch web content,
    run blocks, and more. This block enables sub-agent patterns (copilot calling
    copilot) and scheduled copilot execution via the agent executor.
    """

    class Input(BlockSchemaInput):
        prompt: str = SchemaField(
            description=(
                "The task or instruction for the copilot to execute. "
                "The copilot has access to platform tools like agent management, "
                "workspace files, web fetch, block execution, and more."
            ),
            placeholder="Find my agents and list them",
            advanced=False,
        )

        system_context: str = SchemaField(
            description=(
                "Optional additional context prepended to the prompt. "
                "Use this to constrain copilot behavior, provide domain "
                "context, or set output format requirements."
            ),
            default="",
            advanced=True,
        )

        session_id: str = SchemaField(
            description=(
                "Session ID to continue an existing copilot conversation. "
                "Leave empty to start a new session. "
                "Use the session_id output from a previous run to continue."
            ),
            default="",
            advanced=True,
        )

        max_recursion_depth: int = SchemaField(
            description=(
                "Maximum nesting depth when the copilot calls this block "
                "recursively (sub-agent pattern). Prevents infinite loops."
            ),
            default=3,
            ge=1,
            advanced=True,
        )

    class Output(BlockSchemaOutput):
        response: str = SchemaField(
            description="The final text response from the copilot."
        )
        tool_calls: list[dict] = SchemaField(
            description=(
                "List of tools called during execution. Each entry has "
                "toolCallId, toolName, input, output, and success fields."
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
        token_usage: dict = SchemaField(
            description=(
                "Token usage statistics: promptTokens, "
                "completionTokens, totalTokens."
            ),
        )
        error: str = SchemaField(
            description="Error message if execution failed.",
        )

    def __init__(self):
        super().__init__(
            id="c069dc6b-c3ed-4c12-b6e5-d47361e64ce6",
            description=(
                "Execute tasks using the AutoGPT Copilot with full access to "
                "platform tools (agent management, workspace files, web fetch, "
                "block execution, and more). Enables sub-agent patterns and "
                "scheduled copilot execution."
            ),
            categories={BlockCategory.AI, BlockCategory.AGENT},
            input_schema=AutogptCopilotBlock.Input,
            output_schema=AutogptCopilotBlock.Output,
            test_input={
                "prompt": "List my agents",
                "system_context": "",
                "session_id": "",
                "max_recursion_depth": 3,
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
                        "promptTokens": 100,
                        "completionTokens": 50,
                        "totalTokens": 150,
                    },
                ),
            ],
            test_mock={
                "execute_copilot": lambda *args, **kwargs: (
                    "You have 2 agents: Agent A and Agent B.",
                    [],
                    '[{"role": "user", "content": "List my agents"}]',
                    "test-session-id",
                    {
                        "promptTokens": 100,
                        "completionTokens": 50,
                        "totalTokens": 150,
                    },
                ),
            },
        )

    @staticmethod
    def _check_recursion(max_recursion_depth: int) -> tuple:
        """Check and increment recursion depth. Returns tokens to reset later."""
        current_depth = _copilot_recursion_depth.get()
        inherited_limit = _copilot_recursion_limit.get()
        effective_limit = (
            max_recursion_depth
            if inherited_limit is None
            else min(inherited_limit, max_recursion_depth)
        )
        if current_depth >= effective_limit:
            raise RuntimeError(
                f"Copilot recursion depth limit reached ({effective_limit}). "
                "The copilot has called itself too many times."
            )
        return (
            _copilot_recursion_depth.set(current_depth + 1),
            _copilot_recursion_limit.set(effective_limit),
        )

    @staticmethod
    def _reset_recursion(tokens: tuple) -> None:
        _copilot_recursion_depth.reset(tokens[0])
        _copilot_recursion_limit.reset(tokens[1])

    @staticmethod
    async def _get_or_create_session(session_id: str, user_id: str) -> ChatSession:
        from backend.copilot.model import create_chat_session, get_chat_session

        if session_id:
            session = await get_chat_session(session_id, user_id)
            if not session:
                raise ValueError(
                    f"Copilot session {session_id} not found. "
                    "Use an empty session_id to start a new session."
                )
            return session
        return await create_chat_session(user_id)

    async def execute_copilot(
        self,
        prompt: str,
        system_context: str,
        session_id: str,
        max_recursion_depth: int,
        user_id: str,
    ) -> tuple[str, list[dict], str, str, dict]:
        """Invoke the copilot and collect all stream results."""
        from backend.copilot.model import get_chat_session
        from backend.copilot.response_model import (
            StreamError,
            StreamTextDelta,
            StreamToolInputAvailable,
            StreamToolOutputAvailable,
            StreamUsage,
        )
        from backend.copilot.sdk.service import stream_chat_completion_sdk

        tokens = self._check_recursion(max_recursion_depth)
        try:
            session = await self._get_or_create_session(session_id, user_id)

            effective_prompt = prompt
            if system_context:
                effective_prompt = f"[System Context: {system_context}]\n\n{prompt}"

            response_parts: list[str] = []
            tool_calls: list[dict] = []
            tool_calls_by_id: dict[str, dict] = {}
            total_usage = {
                "promptTokens": 0,
                "completionTokens": 0,
                "totalTokens": 0,
            }

            # No asyncio.timeout — the SDK's internal stream must not be
            # cancelled mid-flight (see sdk/service.py L998-1001).
            async for event in stream_chat_completion_sdk(
                session_id=session.session_id,
                message=effective_prompt,
                is_user_message=True,
                user_id=user_id,
                session=session,
            ):
                if isinstance(event, StreamTextDelta):
                    response_parts.append(event.delta)
                elif isinstance(event, StreamToolInputAvailable):
                    entry = {
                        "toolCallId": event.toolCallId,
                        "toolName": event.toolName,
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
                    total_usage["promptTokens"] += event.promptTokens
                    total_usage["completionTokens"] += event.completionTokens
                    total_usage["totalTokens"] += event.totalTokens
                elif isinstance(event, StreamError):
                    raise RuntimeError(f"Copilot error: {event.errorText}")

            updated_session = await get_chat_session(session.session_id, user_id)
            history_json = "[]"
            if updated_session and updated_session.messages:
                history_json = json.dumps(
                    [m.model_dump(exclude_none=True) for m in updated_session.messages],
                    default=str,
                )

            return (
                "".join(response_parts),
                tool_calls,
                history_json,
                session.session_id,
                total_usage,
            )
        finally:
            self._reset_recursion(tokens)

    async def run(
        self,
        input_data: Input,
        *,
        execution_context: ExecutionContext,
        **kwargs,
    ) -> BlockOutput:
        if not input_data.prompt.strip():
            yield "error", "Prompt cannot be empty."
            return

        if not execution_context.user_id:
            yield "error", "Cannot run copilot without an authenticated user."
            return

        try:
            response, tool_calls, history, sid, usage = await self.execute_copilot(
                prompt=input_data.prompt,
                system_context=input_data.system_context,
                session_id=input_data.session_id,
                max_recursion_depth=input_data.max_recursion_depth,
                user_id=execution_context.user_id,
            )

            yield "response", response
            yield "tool_calls", tool_calls
            yield "conversation_history", history
            yield "session_id", sid
            yield "token_usage", usage
        except Exception as e:
            yield "error", str(e)
