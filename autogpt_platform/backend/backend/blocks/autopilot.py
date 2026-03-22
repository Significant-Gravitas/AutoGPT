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
from backend.copilot.permissions import (
    CopilotPermissions,
    ToolName,
    all_known_tool_names,
    validate_block_identifiers,
)
from backend.data.model import SchemaField

if TYPE_CHECKING:
    from backend.data.execution import ExecutionContext

logger = logging.getLogger(__name__)

# Block ID shared between autopilot.py and copilot prompting.py.
AUTOPILOT_BLOCK_ID = "c069dc6b-c3ed-4c12-b6e5-d47361e64ce6"


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

        tools: list[ToolName] = SchemaField(
            description=(
                "Tool names to filter. Works with tools_exclude to form an "
                "allow-list or deny-list. "
                "Leave empty to apply no tool filter."
            ),
            default=[],
            advanced=True,
        )

        tools_exclude: bool = SchemaField(
            description=(
                "Controls how the 'tools' list is interpreted. "
                "True (default): 'tools' is a deny-list — listed tools are blocked, "
                "all others are allowed. An empty 'tools' list means allow everything. "
                "False: 'tools' is an allow-list — only listed tools are permitted."
            ),
            default=True,
            advanced=True,
        )

        blocks: list[str] = SchemaField(
            description=(
                "Block identifiers to filter when the copilot uses run_block. "
                "Each entry can be: a block name (e.g. 'HTTP Request'), "
                "a full block UUID, or the first 8 hex characters of the UUID "
                "(e.g. 'c069dc6b'). Works with blocks_exclude. "
                "Leave empty to apply no block filter."
            ),
            default=[],
            advanced=True,
        )

        blocks_exclude: bool = SchemaField(
            description=(
                "Controls how the 'blocks' list is interpreted. "
                "True (default): 'blocks' is a deny-list — listed blocks are blocked, "
                "all others are allowed. An empty 'blocks' list means allow everything. "
                "False: 'blocks' is an allow-list — only listed blocks are permitted."
            ),
            default=True,
            advanced=True,
        )

        # timeout_seconds removed: the SDK manages its own heartbeat-based
        # timeouts internally; wrapping with asyncio.timeout corrupts the
        # SDK's internal stream (see service.py CRITICAL comment).

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
                "Current turn messages (user prompt + assistant reply) as JSON. "
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
        from backend.copilot.model import create_chat_session  # avoid circular import

        session = await create_chat_session(user_id)
        return session.session_id

    async def execute_copilot(
        self,
        prompt: str,
        system_context: str,
        session_id: str,
        max_recursion_depth: int,
        user_id: str,
        permissions: "CopilotPermissions | None" = None,
    ) -> tuple[str, list[ToolCallEntry], str, str, TokenUsage]:
        """Invoke the copilot and collect all stream results.

        Delegates to :func:`collect_copilot_response` — the shared helper that
        consumes ``stream_chat_completion_sdk`` without wrapping it in an
        ``asyncio.timeout`` (the SDK manages its own heartbeat-based timeouts).

        Args:
            prompt: The user task/instruction.
            system_context: Optional context prepended to the prompt.
            session_id: Chat session to use.
            max_recursion_depth: Maximum allowed recursion nesting.
            user_id: Authenticated user ID.
            permissions: Optional capability filter restricting tools/blocks.

        Returns:
            A tuple of (response_text, tool_calls, history_json, session_id, usage).
        """
        from backend.copilot.sdk.collect import (
            collect_copilot_response,  # avoid circular import
        )

        tokens = _check_recursion(max_recursion_depth)
        perm_token = None
        try:
            effective_permissions, perm_token = _merge_inherited_permissions(
                permissions
            )
            effective_prompt = prompt
            if system_context:
                effective_prompt = f"[System Context: {system_context}]\n\n{prompt}"

            result = await collect_copilot_response(
                session_id=session_id,
                message=effective_prompt,
                user_id=user_id,
                permissions=effective_permissions,
            )

            # Build a lightweight conversation summary from streamed data.
            turn_messages: list[dict[str, Any]] = [
                {"role": "user", "content": effective_prompt},
            ]
            if result.tool_calls:
                turn_messages.append(
                    {
                        "role": "assistant",
                        "content": result.response_text,
                        "tool_calls": result.tool_calls,
                    }
                )
            else:
                turn_messages.append(
                    {"role": "assistant", "content": result.response_text}
                )
            history_json = json.dumps(turn_messages, default=str)

            tool_calls: list[ToolCallEntry] = [
                {
                    "tool_call_id": tc["tool_call_id"],
                    "tool_name": tc["tool_name"],
                    "input": tc["input"],
                    "output": tc["output"],
                    "success": tc["success"],
                }
                for tc in result.tool_calls
            ]

            usage: TokenUsage = {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            }

            return (
                result.response_text,
                tool_calls,
                history_json,
                session_id,
                usage,
            )
        finally:
            _reset_recursion(tokens)
            if perm_token is not None:
                _inherited_permissions.reset(perm_token)

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

        # Validate and build permissions eagerly — fail before creating a session.
        permissions = await _build_and_validate_permissions(input_data)
        if isinstance(permissions, str):
            # Validation error returned as a string message.
            yield "error", permissions
            return

        # Create session eagerly so the user always gets the session_id,
        # even if the downstream stream fails (avoids orphaned sessions).
        sid = input_data.session_id
        if not sid:
            sid = await self.create_session(execution_context.user_id)

        # NOTE: No asyncio.timeout() here — the SDK manages its own
        # heartbeat-based timeouts internally.  Wrapping with asyncio.timeout
        # would cancel the task mid-flight, corrupting the SDK's internal
        # anyio memory stream (see service.py CRITICAL comment).
        try:
            response, tool_calls, history, _, usage = await self.execute_copilot(
                prompt=input_data.prompt,
                system_context=input_data.system_context,
                session_id=sid,
                max_recursion_depth=input_data.max_recursion_depth,
                user_id=execution_context.user_id,
                permissions=permissions,
            )

            yield "response", response
            yield "tool_calls", tool_calls
            yield "conversation_history", history
            yield "session_id", sid
            yield "token_usage", usage
        except asyncio.CancelledError:
            yield "session_id", sid
            yield "error", "AutoPilot execution was cancelled."
            raise
        except Exception as exc:
            yield "session_id", sid
            yield "error", str(exc)


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


# ---------------------------------------------------------------------------
# Permission helpers
# ---------------------------------------------------------------------------

# Inherited permissions from a parent AutoPilotBlock execution.
# This acts as a ceiling: child executions can only be more restrictive.
_inherited_permissions: contextvars.ContextVar["CopilotPermissions | None"] = (
    contextvars.ContextVar("_inherited_permissions", default=None)
)


async def _build_and_validate_permissions(
    input_data: "AutoPilotBlock.Input",
) -> "CopilotPermissions | str":
    """Build a :class:`CopilotPermissions` from block input and validate it.

    Returns a :class:`CopilotPermissions` on success or a human-readable
    error string if validation fails.
    """
    # Tool names are validated by Pydantic via the ToolName Literal type
    # at model construction time — no runtime check needed here.
    # Validate block identifiers against live block registry.
    if input_data.blocks:
        invalid_blocks = await validate_block_identifiers(input_data.blocks)
        if invalid_blocks:
            return (
                f"Unknown block identifier(s) in 'blocks': {invalid_blocks}. "
                "Use find_block to discover valid block names and IDs. "
                "You may also use the first 8 characters of a block UUID."
            )

    return CopilotPermissions(
        tools=list(input_data.tools),
        tools_exclude=input_data.tools_exclude,
        blocks=input_data.blocks,
        blocks_exclude=input_data.blocks_exclude,
    )


def _merge_inherited_permissions(
    permissions: "CopilotPermissions | None",
) -> "tuple[CopilotPermissions | None, contextvars.Token[CopilotPermissions | None] | None]":
    """Merge *permissions* with any inherited parent permissions.

    The merged result is stored back into the contextvar so that any nested
    AutoPilotBlock invocation (sub-agent) inherits the merged ceiling.

    Returns a tuple of (merged_permissions, reset_token).  The caller MUST
    reset the contextvar via ``_inherited_permissions.reset(token)`` in a
    ``finally`` block when ``reset_token`` is not None — this prevents
    permission leakage between sequential independent executions in the same
    asyncio task.
    """
    parent = _inherited_permissions.get()

    if permissions is None and parent is None:
        return None, None

    all_tools = all_known_tool_names()

    if permissions is None:
        permissions = CopilotPermissions()  # allow-all; will be narrowed by parent

    merged = (
        permissions.merged_with_parent(parent, all_tools)
        if parent is not None
        else permissions
    )

    # Store merged permissions as the new inherited ceiling for nested calls.
    # Return the token so the caller can restore the previous value in finally.
    token = _inherited_permissions.set(merged)
    return merged, token
