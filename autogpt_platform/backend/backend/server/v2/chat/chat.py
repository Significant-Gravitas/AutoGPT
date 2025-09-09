"""Chat streaming functions for handling OpenAI chat completions with tool calling."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)
from prisma.enums import ChatMessageRole

from backend.server.v2.chat import db
from backend.server.v2.chat.config import get_config
from backend.server.v2.chat.models import (
    Error,
    LoginNeeded,
    StreamEnd,
    TextChunk,
    ToolCall,
    ToolResponse,
)
from backend.server.v2.chat.tool_exports import execute_tool, tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Global client cache
_client_cache: AsyncOpenAI | None = None


def get_openai_client(force_new: bool = False) -> AsyncOpenAI:
    """Get or create an OpenAI client instance.

    Args:
        force_new: Force creation of a new client instance

    Returns:
        AsyncOpenAI client instance

    """
    global _client_cache
    config = get_config()

    if not force_new and config.cache_client and _client_cache is not None:
        return _client_cache

    # Create new client with configuration
    client_kwargs = {}
    if config.api_key:
        client_kwargs["api_key"] = config.api_key
    if config.base_url:
        client_kwargs["base_url"] = config.base_url

    client = AsyncOpenAI(**client_kwargs)

    # Cache if configured
    if config.cache_client:
        _client_cache = client

    return client


async def stream_chat_response(
    messages: list[ChatCompletionMessageParam],
    user_id: str,
    model: str | None = None,
    on_assistant_message: (
        Callable[[str, list[dict[str, Any]] | None], Awaitable[None]] | None
    ) = None,
    on_tool_response: Callable[[str, str, str], Awaitable[None]] | None = None,
    session_id: str | None = None,  # Optional for login needed responses
) -> AsyncGenerator[str, None]:
    """Pure streaming function for OpenAI chat completions with tool calling.

    This function is database-agnostic and focuses only on streaming logic.

    Args:
        messages: Conversation context as ChatCompletionMessageParam list
        user_id: User ID for tool execution
        model: OpenAI model to use (overrides config)
        on_assistant_message: Callback for assistant messages (content, tool_calls)
        on_tool_response: Callback for tool responses (tool_call_id, content, role)
        session_id: Optional session ID for login responses

    Yields:
        SSE formatted JSON response objects

    """
    config = get_config()
    model = model or config.model

    try:
        logger.info("Starting pure chat stream")

        # Get OpenAI client
        client = get_openai_client()

        # Loop to handle tool calls and continue conversation
        while True:
            try:
                logger.info("Creating OpenAI chat completion stream...")

                # Create the stream with proper types
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                )

                # Variables to accumulate the response
                assistant_message: str = ""
                tool_calls: list[dict[str, Any]] = []
                finish_reason: str | None = None

                # Process the stream
                chunk: ChatCompletionChunk
                async for chunk in stream:
                    if chunk.choices:
                        choice = chunk.choices[0]
                        delta = choice.delta

                        # Capture finish reason
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
                            logger.info(f"Finish reason: {finish_reason}")

                        # Handle content streaming
                        if delta.content:
                            assistant_message += delta.content
                            # Stream the text chunk
                            text_response = TextChunk(
                                content=delta.content,
                                timestamp=datetime.utcnow().isoformat(),
                            )
                            yield text_response.to_sse()

                        # Handle tool calls
                        if delta.tool_calls:
                            for tc_chunk in delta.tool_calls:
                                idx = tc_chunk.index

                                # Ensure we have a tool call object at this index
                                while len(tool_calls) <= idx:
                                    tool_calls.append(
                                        {
                                            "id": "",
                                            "type": "function",
                                            "function": {
                                                "name": "",
                                                "arguments": "",
                                            },
                                        },
                                    )

                                # Accumulate the tool call data
                                if tc_chunk.id:
                                    tool_calls[idx]["id"] = tc_chunk.id
                                if tc_chunk.function:
                                    if tc_chunk.function.name:
                                        tool_calls[idx]["function"][
                                            "name"
                                        ] = tc_chunk.function.name
                                    if tc_chunk.function.arguments:
                                        tool_calls[idx]["function"][
                                            "arguments"
                                        ] += tc_chunk.function.arguments

                logger.info(f"Stream complete. Finish reason: {finish_reason}")

                # Notify about assistant message if callback provided
                if on_assistant_message and (assistant_message or tool_calls):
                    await on_assistant_message(
                        assistant_message if assistant_message else "",
                        tool_calls if tool_calls else None,
                    )

                # Check if we need to execute tools
                if finish_reason == "tool_calls" and tool_calls:
                    logger.info(f"Processing {len(tool_calls)} tool call(s)")

                    # Add assistant message with tool calls to context
                    assistant_msg = {
                        "role": "assistant",
                        "content": assistant_message if assistant_message else None,
                        "tool_calls": tool_calls,
                    }
                    messages.append(assistant_msg)  # type: ignore

                    # Process each tool call
                    for tool_call in tool_calls:
                        tool_name: str = tool_call.get("function", {}).get(
                            "name",
                            "",
                        )
                        tool_id: str = tool_call.get("id", "")

                        # Parse arguments
                        try:
                            tool_args: dict[str, Any] = json.loads(
                                tool_call.get("function", {}).get("arguments", "{}"),
                            )
                        except (json.JSONDecodeError, TypeError):
                            tool_args = {}

                        logger.info(
                            f"Executing tool: {tool_name} with args: {tool_args}",
                        )

                        # Stream tool call notification
                        tool_call_response = ToolCall(
                            tool_id=tool_id,
                            tool_name=tool_name,
                            arguments=tool_args,
                            timestamp=datetime.utcnow().isoformat(),
                        )
                        yield tool_call_response.to_sse()

                        # Small delay for UI responsiveness
                        await asyncio.sleep(0.1)

                        # Execute the tool (returns JSON string)
                        tool_result_str = await execute_tool(
                            tool_name,
                            tool_args,
                            user_id=user_id,
                            session_id=session_id or "",
                        )

                        # Parse the JSON result
                        try:
                            tool_result = json.loads(tool_result_str)
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, use as string
                            tool_result = tool_result_str

                        # Check for special responses (login needed, etc.)
                        if isinstance(tool_result, dict):
                            result_type = tool_result.get("type")
                            if result_type == "need_login":
                                login_response = LoginNeeded(
                                    message=tool_result.get(
                                        "message", "Authentication required"
                                    ),
                                    session_id=session_id or "",
                                    agent_info=tool_result.get("agent_info"),
                                    timestamp=datetime.utcnow().isoformat(),
                                )
                                yield login_response.to_sse()
                            else:
                                # Stream tool response
                                tool_response = ToolResponse(
                                    tool_id=tool_id,
                                    tool_name=tool_name,
                                    result=tool_result,
                                    success=True,
                                    timestamp=datetime.utcnow().isoformat(),
                                )
                                yield tool_response.to_sse()
                        else:
                            # Stream tool response
                            tool_response = ToolResponse(
                                tool_id=tool_id,
                                tool_name=tool_name,
                                result=tool_result_str,  # Use original string
                                success=True,
                                timestamp=datetime.utcnow().isoformat(),
                            )
                            yield tool_response.to_sse()

                        logger.info(
                            f"Tool result: {tool_result_str[:200] if len(tool_result_str) > 200 else tool_result_str}"
                        )

                        # Notify about tool response if callback provided
                        if on_tool_response:
                            await on_tool_response(
                                tool_id,
                                tool_result_str,  # Already a string
                                "tool",
                            )

                        # Add tool result to context
                        tool_msg: ChatCompletionToolMessageParam = {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": tool_result_str,  # Already JSON string
                        }
                        messages.append(tool_msg)

                    # Continue the loop to get final response
                    logger.info("Making follow-up call with tool results...")
                    continue
                else:
                    # No tool calls, conversation complete
                    logger.info("Conversation complete")

                    # Send stream end marker
                    end_response = StreamEnd(
                        timestamp=datetime.utcnow().isoformat(),
                        summary={
                            "message_count": len(messages),
                            "had_tool_calls": len(tool_calls) > 0,
                        },
                    )
                    yield end_response.to_sse()
                    break

            except Exception as e:
                logger.error(f"Error in stream: {e!s}", exc_info=True)
                error_response = Error(
                    message=str(e),
                    timestamp=datetime.utcnow().isoformat(),
                )
                yield error_response.to_sse()
                break

    except Exception as e:
        logger.error(f"Error in stream_chat_response: {e!s}", exc_info=True)
        error_response = Error(
            message=str(e),
            timestamp=datetime.utcnow().isoformat(),
        )
        yield error_response.to_sse()


# Wrapper function that handles database operations
async def stream_chat_completion(
    session_id: str,
    user_message: str,
    user_id: str,
    model: str = "gpt-4o",
    max_messages: int = 50,
) -> AsyncGenerator[str, None]:
    """Main entry point for streaming chat completions with database handling.

    This function handles all database operations and delegates streaming
    to the pure stream_chat_response function.

    Args:
        session_id: Chat session ID
        user_message: User's input message
        user_id: User ID for authentication
        model: OpenAI model to use
        max_messages: Maximum context messages to include

    Yields:
        SSE formatted JSON strings with response data

    """
    config = get_config()
    logger.warn(
        f"Streaming chat completion for session {session_id} with user {user_id} and message {user_message}"
    )
    # Store user message in database
    await db.create_chat_message(
        session_id=session_id,
        content=user_message,
        role=ChatMessageRole.USER,
    )

    # Get conversation context (already typed as List[ChatCompletionMessageParam])
    context = await db.get_conversation_context(
        session_id=session_id,
        max_messages=max_messages,
        include_system=True,
    )

    # Add system prompt if this is the first message
    if not any(msg.get("role") == "system" for msg in context):
        system_prompt = config.get_system_prompt()
        system_message: ChatCompletionMessageParam = {
            "role": "system",
            "content": system_prompt,
        }
        context.insert(0, system_message)

    # Add current user message to context
    user_msg: ChatCompletionMessageParam = {
        "role": "user",
        "content": user_message,
    }
    context.append(user_msg)

    # Define database callbacks
    async def save_assistant_message(
        content: str, tool_calls: list[dict[str, Any]] | None
    ) -> None:
        """Save assistant message to database."""
        await db.create_chat_message(
            session_id=session_id,
            content=content,
            role=ChatMessageRole.ASSISTANT,
            tool_calls=tool_calls,
        )

    async def save_tool_response(tool_call_id: str, content: str, role: str) -> None:
        """Save tool response to database."""
        await db.create_chat_message(
            session_id=session_id,
            content=content,
            role=ChatMessageRole.TOOL,
            tool_call_id=tool_call_id,
        )

    # Stream the response using the pure function
    async for chunk in stream_chat_response(
        messages=context,
        user_id=user_id,
        model=model,
        on_assistant_message=save_assistant_message,
        on_tool_response=save_tool_response,
        session_id=session_id,
    ):
        yield chunk
