"""Response adapter for converting Claude Agent SDK messages to Vercel AI SDK format.

This module provides the adapter layer that converts streaming messages from
the Claude Agent SDK into the Vercel AI SDK UI Stream Protocol format that
the frontend expects.
"""

import json
import logging
import uuid

from claude_agent_sdk import (
    AssistantMessage,
    Message,
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

from .tool_adapter import MCP_TOOL_PREFIX, pop_pending_tool_output

logger = logging.getLogger(__name__)


class SDKResponseAdapter:
    """Adapter for converting Claude Agent SDK messages to Vercel AI SDK format.

    This class maintains state during a streaming session to properly track
    text blocks, tool calls, and message lifecycle.
    """

    def __init__(self, message_id: str | None = None, session_id: str | None = None):
        self.message_id = message_id or str(uuid.uuid4())
        self.session_id = session_id
        self.text_block_id = str(uuid.uuid4())
        self.has_started_text = False
        self.has_ended_text = False
        self.current_tool_calls: dict[str, dict[str, str]] = {}
        self.resolved_tool_calls: set[str] = set()
        self.step_open = False

    @property
    def has_unresolved_tool_calls(self) -> bool:
        """True when there are tool calls that haven't received output yet."""
        return bool(self.current_tool_calls.keys() - self.resolved_tool_calls)

    def convert_message(self, sdk_message: Message) -> list[StreamBaseResponse]:
        """Convert a single SDK message to Vercel AI SDK format."""
        responses: list[StreamBaseResponse] = []

        if isinstance(sdk_message, SystemMessage):
            if sdk_message.subtype == "init":
                responses.append(
                    StreamStart(messageId=self.message_id, sessionId=self.session_id)
                )
                # Open the first step (matches non-SDK: StreamStart then StreamStartStep)
                responses.append(StreamStartStep())
                self.step_open = True

        elif isinstance(sdk_message, AssistantMessage):
            # Flush any SDK built-in tool calls that didn't get a UserMessage
            # result (e.g. WebSearch, Read handled internally by the CLI).
            # BUT skip flush when this AssistantMessage is a parallel tool
            # continuation (contains only ToolUseBlocks) — the prior tools
            # are still executing concurrently and haven't finished yet.
            is_tool_only = all(isinstance(b, ToolUseBlock) for b in sdk_message.content)
            if not is_tool_only:
                self._flush_unresolved_tool_calls(responses)

            # After tool results, the SDK sends a new AssistantMessage for the
            # next LLM turn. Open a new step if the previous one was closed.
            if not self.step_open:
                responses.append(StreamStartStep())
                self.step_open = True

            for block in sdk_message.content:
                if isinstance(block, TextBlock):
                    if block.text:
                        self._ensure_text_started(responses)
                        responses.append(
                            StreamTextDelta(id=self.text_block_id, delta=block.text)
                        )

                elif isinstance(block, ToolUseBlock):
                    self._end_text_if_open(responses)

                    # Strip MCP prefix so frontend sees "find_block"
                    # instead of "mcp__copilot__find_block".
                    tool_name = block.name.removeprefix(MCP_TOOL_PREFIX)

                    responses.append(
                        StreamToolInputStart(toolCallId=block.id, toolName=tool_name)
                    )
                    responses.append(
                        StreamToolInputAvailable(
                            toolCallId=block.id,
                            toolName=tool_name,
                            input=block.input,
                        )
                    )
                    self.current_tool_calls[block.id] = {"name": tool_name}

        elif isinstance(sdk_message, UserMessage):
            # UserMessage carries tool results back from tool execution.
            content = sdk_message.content
            blocks = content if isinstance(content, list) else []
            resolved_in_blocks: set[str] = set()

            sid = (self.session_id or "?")[:12]
            parent_id_preview = getattr(sdk_message, "parent_tool_use_id", None)
            logger.info(
                "[SDK] [%s] UserMessage: %d blocks, content_type=%s, "
                "parent_tool_use_id=%s",
                sid,
                len(blocks),
                type(content).__name__,
                parent_id_preview[:12] if parent_id_preview else "None",
            )

            for block in blocks:
                if isinstance(block, ToolResultBlock) and block.tool_use_id:
                    # Skip if already resolved (e.g. by flush) — the real
                    # result supersedes the empty flush, but re-emitting
                    # would confuse the frontend's state machine.
                    if block.tool_use_id in self.resolved_tool_calls:
                        continue
                    tool_info = self.current_tool_calls.get(block.tool_use_id, {})
                    tool_name = tool_info.get("name", "unknown")

                    # Prefer the stashed full output over the SDK's
                    # (potentially truncated) ToolResultBlock content.
                    # The SDK truncates large results, writing them to disk,
                    # which breaks frontend widget parsing.
                    output = pop_pending_tool_output(tool_name) or (
                        _extract_tool_output(block.content)
                    )

                    responses.append(
                        StreamToolOutputAvailable(
                            toolCallId=block.tool_use_id,
                            toolName=tool_name,
                            output=output,
                            success=not (block.is_error or False),
                        )
                    )
                    resolved_in_blocks.add(block.tool_use_id)

            # Handle SDK built-in tool results carried via parent_tool_use_id
            # instead of (or in addition to) ToolResultBlock content.
            parent_id = sdk_message.parent_tool_use_id
            if (
                parent_id
                and parent_id not in resolved_in_blocks
                and parent_id not in self.resolved_tool_calls
            ):
                tool_info = self.current_tool_calls.get(parent_id, {})
                tool_name = tool_info.get("name", "unknown")

                # Try stashed output first (from PostToolUse hook),
                # then tool_use_result dict, then string content.
                output = pop_pending_tool_output(tool_name)
                if not output:
                    tur = sdk_message.tool_use_result
                    if tur is not None:
                        output = _extract_tool_use_result(tur)
                if not output and isinstance(content, str) and content.strip():
                    output = content.strip()

                if output:
                    responses.append(
                        StreamToolOutputAvailable(
                            toolCallId=parent_id,
                            toolName=tool_name,
                            output=output,
                            success=True,
                        )
                    )
                    resolved_in_blocks.add(parent_id)

            self.resolved_tool_calls.update(resolved_in_blocks)

            # Close the current step after tool results — the next
            # AssistantMessage will open a new step for the continuation.
            if self.step_open:
                responses.append(StreamFinishStep())
                self.step_open = False

        elif isinstance(sdk_message, ResultMessage):
            self._flush_unresolved_tool_calls(responses)
            self._end_text_if_open(responses)
            # Close the step before finishing.
            if self.step_open:
                responses.append(StreamFinishStep())
                self.step_open = False

            if sdk_message.subtype == "success":
                responses.append(StreamFinish())
            elif sdk_message.subtype in ("error", "error_during_execution"):
                error_msg = sdk_message.result or "Unknown error"
                responses.append(
                    StreamError(errorText=str(error_msg), code="sdk_error")
                )
                responses.append(StreamFinish())
            else:
                logger.warning(
                    f"Unexpected ResultMessage subtype: {sdk_message.subtype}"
                )
                responses.append(StreamFinish())

        else:
            logger.debug(f"Unhandled SDK message type: {type(sdk_message).__name__}")

        return responses

    def _ensure_text_started(self, responses: list[StreamBaseResponse]) -> None:
        """Start (or restart) a text block if needed."""
        if not self.has_started_text or self.has_ended_text:
            if self.has_ended_text:
                self.text_block_id = str(uuid.uuid4())
                self.has_ended_text = False
            responses.append(StreamTextStart(id=self.text_block_id))
            self.has_started_text = True

    def _end_text_if_open(self, responses: list[StreamBaseResponse]) -> None:
        """End the current text block if one is open."""
        if self.has_started_text and not self.has_ended_text:
            responses.append(StreamTextEnd(id=self.text_block_id))
            self.has_ended_text = True

    def _flush_unresolved_tool_calls(self, responses: list[StreamBaseResponse]) -> None:
        """Emit outputs for tool calls that didn't receive a UserMessage result.

        SDK built-in tools (WebSearch, Read, etc.) may be executed by the CLI
        internally without surfacing a separate ``UserMessage`` with
        ``ToolResultBlock`` content.  The ``PostToolUse`` hook stashes their
        output, which we pop and emit here before the next ``AssistantMessage``
        starts.
        """
        unresolved = [
            (tid, info.get("name", "unknown"))
            for tid, info in self.current_tool_calls.items()
            if tid not in self.resolved_tool_calls
        ]
        sid = (self.session_id or "?")[:12]
        if not unresolved:
            logger.info(
                "[SDK] [%s] Flush called but all %d tool(s) already resolved",
                sid,
                len(self.current_tool_calls),
            )
            return
        logger.info(
            "[SDK] [%s] Flushing %d unresolved tool call(s): %s",
            sid,
            len(unresolved),
            ", ".join(f"{name}({tid[:12]})" for tid, name in unresolved),
        )

        flushed = False
        for tool_id, tool_name in unresolved:
            output = pop_pending_tool_output(tool_name)
            if output is not None:
                responses.append(
                    StreamToolOutputAvailable(
                        toolCallId=tool_id,
                        toolName=tool_name,
                        output=output,
                        success=True,
                    )
                )
                self.resolved_tool_calls.add(tool_id)
                flushed = True
                logger.info(
                    "[SDK] [%s] Flushed stashed output for %s " "(call %s, %d chars)",
                    sid,
                    tool_name,
                    tool_id[:12],
                    len(output),
                )
            else:
                # No output available — emit an empty output so the frontend
                # transitions the tool from input-available to output-available
                # (stops the spinner).
                responses.append(
                    StreamToolOutputAvailable(
                        toolCallId=tool_id,
                        toolName=tool_name,
                        output="",
                        success=True,
                    )
                )
                self.resolved_tool_calls.add(tool_id)
                flushed = True
                logger.warning(
                    "[SDK] [%s] Flushed EMPTY output for unresolved tool %s "
                    "(call %s) — stash was empty (likely SDK hook race "
                    "condition: PostToolUse hook hadn't completed before "
                    "flush was triggered)",
                    sid,
                    tool_name,
                    tool_id[:12],
                )

        if flushed and self.step_open:
            responses.append(StreamFinishStep())
            self.step_open = False


def _extract_tool_output(content: str | list[dict[str, str]] | None) -> str:
    """Extract a string output from a ToolResultBlock's content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        if parts:
            return "".join(parts)
        try:
            return json.dumps(content)
        except (TypeError, ValueError):
            return str(content)
    if content is None:
        return ""
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _extract_tool_use_result(result: object) -> str:
    """Extract a string from a UserMessage's ``tool_use_result`` dict.

    SDK built-in tools may store their result in ``tool_use_result``
    instead of (or in addition to) ``ToolResultBlock`` content blocks.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # Try common result keys
        for key in ("content", "text", "output", "stdout", "result"):
            val = result.get(key)
            if isinstance(val, str) and val:
                return val
        # Fall back to JSON serialization of the whole dict
        try:
            return json.dumps(result)
        except (TypeError, ValueError):
            return str(result)
    if result is None:
        return ""
    try:
        return json.dumps(result)
    except (TypeError, ValueError):
        return str(result)
