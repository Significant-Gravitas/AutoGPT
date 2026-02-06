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
from ..model import ChatSession
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


async def stream_with_anthropic(
    session: ChatSession,
    system_prompt: str,
    text_block_id: str,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Stream using Anthropic SDK directly with tool calling support."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY") or config.api_key
    if not api_key:
        yield StreamError(
            errorText="ANTHROPIC_API_KEY not configured", code="config_error"
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
    max_iterations = 10

    for _ in range(max_iterations):
        try:
            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
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

                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": output,
                                    "is_error": is_error,
                                }
                            )

                    anthropic_messages.append(
                        {"role": "assistant", "content": assistant_content}
                    )
                    anthropic_messages.append({"role": "user", "content": tool_results})
                    continue

                else:
                    if has_started_text:
                        yield StreamTextEnd(id=text_block_id)

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
            yield StreamError(errorText=f"Error: {str(e)}", code="anthropic_error")
            yield StreamFinish()
            return

    yield StreamError(errorText="Max tool iterations reached", code="max_iterations")
    yield StreamFinish()


def _convert_session_to_anthropic(session: ChatSession) -> list[dict[str, Any]]:
    """Convert session messages to Anthropic format."""
    messages = []
    for msg in session.messages:
        if msg.role == "user":
            messages.append({"role": "user", "content": msg.content or ""})
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
                messages.append({"role": "assistant", "content": content})
        elif msg.role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content or "",
                        }
                    ],
                }
            )
    return messages


async def _execute_tool(
    tool_name: str, tool_input: Any, handlers: dict[str, Any]
) -> tuple[str, bool]:
    """Execute a tool and return (output, is_error)."""
    handler = handlers.get(tool_name)
    if not handler:
        return f"Unknown tool: {tool_name}", True

    try:
        result = await handler(tool_input)
        output = result.get("content", [{}])[0].get("text", "")
        is_error = result.get("isError", False)
        return output, is_error
    except Exception as e:
        return f"Error: {str(e)}", True
