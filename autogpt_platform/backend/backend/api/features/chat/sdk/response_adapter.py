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

from backend.api.features.chat.response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamHeartbeat,
    StreamStart,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
    StreamUsage,
)

logger = logging.getLogger(__name__)


class SDKResponseAdapter:
    """Adapter for converting Claude Agent SDK messages to Vercel AI SDK format.

    This class maintains state during a streaming session to properly track
    text blocks, tool calls, and message lifecycle.
    """

    def __init__(self, message_id: str | None = None):
        self.message_id = message_id or str(uuid.uuid4())
        self.text_block_id = str(uuid.uuid4())
        self.has_started_text = False
        self.has_ended_text = False
        self.current_tool_calls: dict[str, dict[str, str]] = {}
        self.task_id: str | None = None

    def set_task_id(self, task_id: str) -> None:
        """Set the task ID for reconnection support."""
        self.task_id = task_id

    def convert_message(self, sdk_message: Message) -> list[StreamBaseResponse]:
        """Convert a single SDK message to Vercel AI SDK format."""
        responses: list[StreamBaseResponse] = []

        if isinstance(sdk_message, SystemMessage):
            if sdk_message.subtype == "init":
                responses.append(
                    StreamStart(messageId=self.message_id, taskId=self.task_id)
                )

        elif isinstance(sdk_message, AssistantMessage):
            for block in sdk_message.content:
                if isinstance(block, TextBlock):
                    if block.text:
                        self._ensure_text_started(responses)
                        responses.append(
                            StreamTextDelta(id=self.text_block_id, delta=block.text)
                        )

                elif isinstance(block, ToolUseBlock):
                    self._end_text_if_open(responses)

                    responses.append(
                        StreamToolInputStart(toolCallId=block.id, toolName=block.name)
                    )
                    responses.append(
                        StreamToolInputAvailable(
                            toolCallId=block.id,
                            toolName=block.name,
                            input=block.input,
                        )
                    )
                    self.current_tool_calls[block.id] = {"name": block.name}

        elif isinstance(sdk_message, UserMessage):
            # UserMessage carries tool results back from tool execution
            content = sdk_message.content
            blocks = content if isinstance(content, list) else []
            for block in blocks:
                if isinstance(block, ToolResultBlock) and block.tool_use_id:
                    tool_info = self.current_tool_calls.get(block.tool_use_id, {})
                    tool_name = tool_info.get("name", "unknown")
                    output = _extract_tool_output(block.content)
                    responses.append(
                        StreamToolOutputAvailable(
                            toolCallId=block.tool_use_id,
                            toolName=tool_name,
                            output=output,
                            success=not (block.is_error or False),
                        )
                    )

        elif isinstance(sdk_message, ResultMessage):
            if sdk_message.subtype == "success":
                self._end_text_if_open(responses)
                responses.append(StreamFinish())

            elif sdk_message.subtype in ("error", "error_during_execution"):
                error_msg = getattr(sdk_message, "result", None) or "Unknown error"
                responses.append(
                    StreamError(errorText=str(error_msg), code="sdk_error")
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

    def create_heartbeat(self, tool_call_id: str | None = None) -> StreamHeartbeat:
        """Create a heartbeat response."""
        return StreamHeartbeat(toolCallId=tool_call_id)

    def create_usage(self, prompt_tokens: int, completion_tokens: int) -> StreamUsage:
        """Create a usage statistics response."""
        return StreamUsage(
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=prompt_tokens + completion_tokens,
        )


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
