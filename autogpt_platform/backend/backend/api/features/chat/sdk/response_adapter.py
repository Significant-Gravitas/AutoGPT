"""Response adapter for converting Claude Agent SDK messages to Vercel AI SDK format.

This module provides the adapter layer that converts streaming messages from
the Claude Agent SDK into the Vercel AI SDK UI Stream Protocol format that
the frontend expects.
"""

import json
import logging
import uuid
from typing import Any, AsyncGenerator

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
        """Initialize the adapter.

        Args:
            message_id: Optional message ID. If not provided, one will be generated.
        """
        self.message_id = message_id or str(uuid.uuid4())
        self.text_block_id = str(uuid.uuid4())
        self.has_started_text = False
        self.has_ended_text = False
        self.current_tool_calls: dict[str, dict[str, Any]] = {}
        self.task_id: str | None = None

    def set_task_id(self, task_id: str) -> None:
        """Set the task ID for reconnection support."""
        self.task_id = task_id

    def convert_message(self, sdk_message: Any) -> list[StreamBaseResponse]:
        """Convert a single SDK message to Vercel AI SDK format.

        Args:
            sdk_message: A message from the Claude Agent SDK.

        Returns:
            List of StreamBaseResponse objects (may be empty or multiple).
        """
        responses: list[StreamBaseResponse] = []

        # Handle different SDK message types - use class name since SDK uses dataclasses
        class_name = type(sdk_message).__name__
        msg_subtype = getattr(sdk_message, "subtype", None)

        if class_name == "SystemMessage":
            if msg_subtype == "init":
                # Session initialization - emit start
                responses.append(
                    StreamStart(
                        messageId=self.message_id,
                        taskId=self.task_id,
                    )
                )

        elif class_name == "AssistantMessage":
            # Assistant message with content blocks
            content = getattr(sdk_message, "content", [])
            for block in content:
                # Check block type by class name (SDK uses dataclasses) or dict type
                block_class = type(block).__name__
                block_type = block.get("type") if isinstance(block, dict) else None

                if block_class == "TextBlock" or block_type == "text":
                    # Text content
                    text = getattr(block, "text", None) or (
                        block.get("text") if isinstance(block, dict) else ""
                    )

                    if text:
                        # Start text block if needed (or restart after tool calls)
                        if not self.has_started_text or self.has_ended_text:
                            # Generate new text block ID for text after tools
                            if self.has_ended_text:
                                self.text_block_id = str(uuid.uuid4())
                                self.has_ended_text = False
                            responses.append(StreamTextStart(id=self.text_block_id))
                            self.has_started_text = True

                        # Emit text delta
                        responses.append(
                            StreamTextDelta(
                                id=self.text_block_id,
                                delta=text,
                            )
                        )

                elif block_class == "ToolUseBlock" or block_type == "tool_use":
                    # Tool call
                    tool_id_raw = getattr(block, "id", None) or (
                        block.get("id") if isinstance(block, dict) else None
                    )
                    tool_id: str = (
                        str(tool_id_raw) if tool_id_raw else str(uuid.uuid4())
                    )

                    tool_name_raw = getattr(block, "name", None) or (
                        block.get("name") if isinstance(block, dict) else None
                    )
                    tool_name: str = str(tool_name_raw) if tool_name_raw else "unknown"

                    tool_input = getattr(block, "input", None) or (
                        block.get("input") if isinstance(block, dict) else {}
                    )

                    # End text block if we were streaming text
                    if self.has_started_text and not self.has_ended_text:
                        responses.append(StreamTextEnd(id=self.text_block_id))
                        self.has_ended_text = True

                    # Emit tool input start
                    responses.append(
                        StreamToolInputStart(
                            toolCallId=tool_id,
                            toolName=tool_name,
                        )
                    )

                    # Emit tool input available with full input
                    responses.append(
                        StreamToolInputAvailable(
                            toolCallId=tool_id,
                            toolName=tool_name,
                            input=tool_input if isinstance(tool_input, dict) else {},
                        )
                    )

                    # Track the tool call
                    self.current_tool_calls[tool_id] = {
                        "name": tool_name,
                        "input": tool_input,
                    }

        elif class_name in ("ToolResultMessage", "UserMessage"):
            # Tool result - check for tool_result content
            content = getattr(sdk_message, "content", [])

            for block in content:
                block_class = type(block).__name__
                block_type = block.get("type") if isinstance(block, dict) else None

                if block_class == "ToolResultBlock" or block_type == "tool_result":
                    tool_use_id = getattr(block, "tool_use_id", None) or (
                        block.get("tool_use_id") if isinstance(block, dict) else None
                    )
                    result_content = getattr(block, "content", None) or (
                        block.get("content") if isinstance(block, dict) else ""
                    )
                    is_error = getattr(block, "is_error", False) or (
                        block.get("is_error", False)
                        if isinstance(block, dict)
                        else False
                    )

                    if tool_use_id:
                        tool_info = self.current_tool_calls.get(tool_use_id, {})
                        tool_name = tool_info.get("name", "unknown")

                        # Format the output
                        if isinstance(result_content, list):
                            # Extract text from content blocks
                            output_text = ""
                            for item in result_content:
                                if (
                                    isinstance(item, dict)
                                    and item.get("type") == "text"
                                ):
                                    output_text += item.get("text", "")
                                elif hasattr(item, "text"):
                                    output_text += getattr(item, "text", "")
                            if output_text:
                                output = output_text
                            else:
                                try:
                                    output = json.dumps(result_content)
                                except (TypeError, ValueError):
                                    output = str(result_content)
                        elif isinstance(result_content, str):
                            output = result_content
                        else:
                            try:
                                output = json.dumps(result_content)
                            except (TypeError, ValueError):
                                output = str(result_content)

                        responses.append(
                            StreamToolOutputAvailable(
                                toolCallId=tool_use_id,
                                toolName=tool_name,
                                output=output,
                                success=not is_error,
                            )
                        )

        elif class_name == "ResultMessage":
            # Final result
            if msg_subtype == "success":
                # End text block if still open
                if self.has_started_text and not self.has_ended_text:
                    responses.append(StreamTextEnd(id=self.text_block_id))
                    self.has_ended_text = True

                # Emit finish
                responses.append(StreamFinish())

            elif msg_subtype in ("error", "error_during_execution"):
                error_msg = getattr(sdk_message, "error", "Unknown error")
                responses.append(
                    StreamError(
                        errorText=str(error_msg),
                        code="sdk_error",
                    )
                )
                responses.append(StreamFinish())

        elif class_name == "ErrorMessage":
            # Error message
            error_msg = getattr(sdk_message, "message", None) or getattr(
                sdk_message, "error", "Unknown error"
            )
            responses.append(
                StreamError(
                    errorText=str(error_msg),
                    code="sdk_error",
                )
            )
            responses.append(StreamFinish())

        else:
            logger.debug(f"Unhandled SDK message type: {class_name}")

        return responses

    def create_heartbeat(self, tool_call_id: str | None = None) -> StreamHeartbeat:
        """Create a heartbeat response."""
        return StreamHeartbeat(toolCallId=tool_call_id)

    def create_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> StreamUsage:
        """Create a usage statistics response."""
        return StreamUsage(
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=prompt_tokens + completion_tokens,
        )


async def adapt_sdk_stream(
    sdk_stream: AsyncGenerator[Any, None],
    message_id: str | None = None,
    task_id: str | None = None,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Adapt a Claude Agent SDK stream to Vercel AI SDK format.

    Args:
        sdk_stream: The async generator from the Claude Agent SDK.
        message_id: Optional message ID for the response.
        task_id: Optional task ID for reconnection support.

    Yields:
        StreamBaseResponse objects in Vercel AI SDK format.
    """
    adapter = SDKResponseAdapter(message_id=message_id)
    if task_id:
        adapter.set_task_id(task_id)

    # Emit start immediately
    yield StreamStart(messageId=adapter.message_id, taskId=task_id)

    finished = False
    try:
        async for sdk_message in sdk_stream:
            responses = adapter.convert_message(sdk_message)
            for response in responses:
                # Skip duplicate start messages
                if isinstance(response, StreamStart):
                    continue
                if isinstance(response, StreamFinish):
                    finished = True
                yield response

    except Exception as e:
        logger.error(f"Error in SDK stream: {e}", exc_info=True)
        yield StreamError(
            errorText="An error occurred. Please try again.",
            code="stream_error",
        )
        yield StreamFinish()
        return

    # Ensure terminal StreamFinish if SDK stream ended without one
    if not finished:
        yield StreamFinish()
