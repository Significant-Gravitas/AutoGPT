from collections.abc import AsyncGenerator
from typing import Any
from datetime import datetime, UTC

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

import backend.server.v2.chat.config
from backend.server.v2.chat.models import (
    StreamTextChunk,
    StreamToolCall,
    StreamToolResponse,
    StreamError,
    StreamEnd,
    ResponseType,
)
import logging

logger = logging.getLogger(__name__)

config = backend.server.v2.chat.config.ChatConfig()
client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)


async def execute_tool(
    tool_name: str,
    parameters: dict[str, Any],
    user_id: str | None,
    session_id: str,
) -> StreamToolResponse:
    """
    TODO: Implement tool execution.
    """
    return StreamToolResponse(
        type=ResponseType.TOOL_RESPONSE,
        tool_id=tool_name,
        tool_name=tool_name,
        result="",
        success=True,
        timestamp=datetime.now(UTC).isoformat(),
    )


async def stream_chat_completion(
    session_id: str,
    user_message: str,
    user_id: str | None,
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
    # TODO: Implement this function once db operations are implemented
    async for chunk in stream_chat_response(
        messages=[],
        tools=[],
        session_id=session_id,
        user_id=user_id,
    ):
        yield chunk


async def stream_chat_response(
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionToolParam],
    session_id: str,
    user_id: str | None,
) -> AsyncGenerator[str, None]:
    """
    Pure streaming function for OpenAI chat completions with tool calling.

    This function is database-agnostic and focuses only on streaming logic.

    Args:
        messages: Conversation context as ChatCompletionMessageParam list
        session_id: Session ID
        user_id: User ID for tool execution

    Yields:
        SSE formatted JSON response objects

    """
    model = config.model

    logger.info("Starting pure chat stream")

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
                        text_response = StreamTextChunk(
                            content=delta.content,
                            timestamp=datetime.now(UTC).isoformat(),
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

                            # Yield each tool call as soon as it's done
                            logger.info(f"Yielding tool call: {tool_calls[idx]}")
                            yield StreamToolCall(
                                tool_id=tool_calls[idx]["id"],
                                tool_name=tool_calls[idx]["function"]["name"],
                                arguments=tool_calls[idx]["function"]["arguments"],
                                timestamp=datetime.now(UTC).isoformat(),
                            ).to_sse()

                            tool_execution_response: StreamToolResponse = (
                                await execute_tool(
                                    tool_calls[idx]["function"]["name"],
                                    tool_calls[idx]["function"]["arguments"],
                                    user_id=user_id,
                                    session_id=session_id or "",
                                )
                            )
                            yield tool_execution_response.to_sse()

            logger.info(f"Stream complete. Finish reason: {finish_reason}")
            yield StreamEnd(
                timestamp=datetime.now(UTC).isoformat(),
            ).to_sse()

        except Exception as e:
            logger.error(f"Error in stream: {e!s}", exc_info=True)
            error_response = StreamError(
                message=str(e),
                timestamp=datetime.now(UTC).isoformat(),
            )
            yield error_response.to_sse()
            yield StreamEnd(
                timestamp=datetime.now(UTC).isoformat(),
            ).to_sse()
            return
