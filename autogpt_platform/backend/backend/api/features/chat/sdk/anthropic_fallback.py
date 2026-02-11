"""Anthropic SDK fallback implementation.

This module provides the fallback streaming implementation using the Anthropic SDK
directly when the Claude Agent SDK is not available.
"""

import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

from ..config import ChatConfig
from ..model import ChatMessage, ChatSession
from ..response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
    StreamUsage,
)
from .tool_adapter import get_tool_definitions, get_tool_handlers

logger = logging.getLogger(__name__)
config = ChatConfig()

# Maximum tool-call iterations before stopping to prevent infinite loops
_MAX_TOOL_ITERATIONS = 10


async def stream_with_anthropic(
    session: ChatSession,
    system_prompt: str,
    text_block_id: str,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Stream using Anthropic SDK directly with tool calling support.

    This function accumulates messages into the session for persistence.
    The caller should NOT yield an additional StreamFinish - this function handles it.
    """
    import anthropic

    # Only use ANTHROPIC_API_KEY - don't fall back to OpenRouter keys
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield StreamError(
            errorText="ANTHROPIC_API_KEY not configured for fallback",
            code="config_error",
        )
        yield StreamFinish()
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    tool_definitions = get_tool_definitions()
    tool_handlers = get_tool_handlers()

    anthropic_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["inputSchema"],
        }
        for t in tool_definitions
    ]

    anthropic_messages = _convert_session_to_anthropic(session)

    if not anthropic_messages or anthropic_messages[-1]["role"] != "user":
        anthropic_messages.append(
            {"role": "user", "content": "Continue with the task."}
        )

    has_started_text = False
    accumulated_text = ""
    accumulated_tool_calls: list[dict[str, Any]] = []

    for _ in range(_MAX_TOOL_ITERATIONS):
        try:
            async with client.messages.stream(
                model=(
                    config.model.split("/")[-1] if "/" in config.model else config.model
                ),
                max_tokens=4096,
                system=system_prompt,
                messages=cast(Any, anthropic_messages),
                tools=cast(Any, anthropic_tools) if anthropic_tools else [],
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if hasattr(block, "type"):
                            if block.type == "text" and not has_started_text:
                                yield StreamTextStart(id=text_block_id)
                                has_started_text = True
                            elif block.type == "tool_use":
                                yield StreamToolInputStart(
                                    toolCallId=block.id, toolName=block.name
                                )

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "type") and delta.type == "text_delta":
                            accumulated_text += delta.text
                            yield StreamTextDelta(id=text_block_id, delta=delta.text)

                final_message = await stream.get_final_message()

                if final_message.stop_reason == "tool_use":
                    if has_started_text:
                        yield StreamTextEnd(id=text_block_id)
                        has_started_text = False
                        text_block_id = str(uuid.uuid4())

                    tool_results = []
                    assistant_content: list[dict[str, Any]] = []

                    for block in final_message.content:
                        if block.type == "text":
                            assistant_content.append(
                                {"type": "text", "text": block.text}
                            )
                        elif block.type == "tool_use":
                            assistant_content.append(
                                {
                                    "type": "tool_use",
                                    "id": block.id,
                                    "name": block.name,
                                    "input": block.input,
                                }
                            )

                            # Track tool call for session persistence
                            accumulated_tool_calls.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": json.dumps(
                                            block.input
                                            if isinstance(block.input, dict)
                                            else {}
                                        ),
                                    },
                                }
                            )

                            yield StreamToolInputAvailable(
                                toolCallId=block.id,
                                toolName=block.name,
                                input=(
                                    block.input if isinstance(block.input, dict) else {}
                                ),
                            )

                            output, is_error = await _execute_tool(
                                block.name, block.input, tool_handlers
                            )

                            yield StreamToolOutputAvailable(
                                toolCallId=block.id,
                                toolName=block.name,
                                output=output,
                                success=not is_error,
                            )

                            # Save tool result to session
                            session.messages.append(
                                ChatMessage(
                                    role="tool",
                                    content=output,
                                    tool_call_id=block.id,
                                )
                            )

                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": output,
                                    "is_error": is_error,
                                }
                            )

                    # Save assistant message with tool calls to session
                    session.messages.append(
                        ChatMessage(
                            role="assistant",
                            content=accumulated_text or None,
                            tool_calls=(
                                accumulated_tool_calls
                                if accumulated_tool_calls
                                else None
                            ),
                        )
                    )
                    # Reset for next iteration
                    accumulated_text = ""
                    accumulated_tool_calls = []

                    anthropic_messages.append(
                        {"role": "assistant", "content": assistant_content}
                    )
                    anthropic_messages.append({"role": "user", "content": tool_results})
                    continue

                else:
                    if has_started_text:
                        yield StreamTextEnd(id=text_block_id)

                    # Save final assistant response to session
                    if accumulated_text:
                        session.messages.append(
                            ChatMessage(role="assistant", content=accumulated_text)
                        )

                    yield StreamUsage(
                        promptTokens=final_message.usage.input_tokens,
                        completionTokens=final_message.usage.output_tokens,
                        totalTokens=final_message.usage.input_tokens
                        + final_message.usage.output_tokens,
                    )
                    yield StreamFinish()
                    return

        except Exception as e:
            logger.error(f"[Anthropic Fallback] Error: {e}", exc_info=True)
            yield StreamError(
                errorText="An error occurred. Please try again.",
                code="anthropic_error",
            )
            yield StreamFinish()
            return

    yield StreamError(errorText="Max tool iterations reached", code="max_iterations")
    yield StreamFinish()


def _convert_session_to_anthropic(session: ChatSession) -> list[dict[str, Any]]:
    """Convert session messages to Anthropic format.

    Handles merging consecutive same-role messages (Anthropic requires alternating roles).
    """
    messages: list[dict[str, Any]] = []

    for msg in session.messages:
        if msg.role == "user":
            new_msg = {"role": "user", "content": msg.content or ""}
        elif msg.role == "assistant":
            content: list[dict[str, Any]] = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    content.append(
                        {
                            "type": "tool_use",
                            "id": tc.get("id", str(uuid.uuid4())),
                            "name": func.get("name", ""),
                            "input": args,
                        }
                    )
            if content:
                new_msg = {"role": "assistant", "content": content}
            else:
                continue  # Skip empty assistant messages
        elif msg.role == "tool":
            new_msg = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content or "",
                    }
                ],
            }
        else:
            continue

        messages.append(new_msg)

    # Merge consecutive same-role messages (Anthropic requires alternating roles)
    return _merge_consecutive_roles(messages)


def _merge_consecutive_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive messages with the same role.

    Anthropic API requires alternating user/assistant roles.
    """
    if not messages:
        return []

    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            # Merge with previous message
            prev_content = merged[-1]["content"]
            new_content = msg["content"]

            # Normalize both to list-of-blocks form
            if isinstance(prev_content, str):
                prev_content = [{"type": "text", "text": prev_content}]
            if isinstance(new_content, str):
                new_content = [{"type": "text", "text": new_content}]

            # Ensure both are lists
            if not isinstance(prev_content, list):
                prev_content = [prev_content]
            if not isinstance(new_content, list):
                new_content = [new_content]

            merged[-1]["content"] = prev_content + new_content
        else:
            merged.append(msg)

    return merged


async def _execute_tool(
    tool_name: str, tool_input: Any, handlers: dict[str, Any]
) -> tuple[str, bool]:
    """Execute a tool and return (output, is_error)."""
    handler = handlers.get(tool_name)
    if not handler:
        return f"Unknown tool: {tool_name}", True

    try:
        result = await handler(tool_input)
        # Safely extract output - handle empty or missing content
        content = result.get("content") or []
        if content and isinstance(content, list) and len(content) > 0:
            first_item = content[0]
            output = first_item.get("text", "") if isinstance(first_item, dict) else ""
        else:
            output = ""
        is_error = result.get("isError", False)
        return output, is_error
    except Exception as e:
        return f"Error: {str(e)}", True
