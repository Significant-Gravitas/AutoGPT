import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import orjson
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam

import backend.server.v2.chat.config
from backend.server.v2.chat.data import (
    ChatMessage,
    ChatSession,
    Usage,
    get_chat_session,
    upsert_chat_session,
)
from backend.server.v2.chat.models import (
    StreamBaseResponse,
    StreamEnd,
    StreamError,
    StreamTextChunk,
    StreamToolCall,
    StreamToolExecutionResult,
    StreamUsage,
)
from backend.server.v2.chat.tools import execute_tool, tools
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)

config = backend.server.v2.chat.config.ChatConfig()
client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)


async def create_chat_session(
    user_id: str | None = None,
) -> ChatSession:
    """
    Create a new chat session.
    """
    return ChatSession.new(user_id)


async def get_session(
    session_id: str,
    user_id: str | None = None,
) -> ChatSession | None:
    """
    Get a chat session by ID.
    """
    return await get_chat_session(session_id, user_id)


async def assign_user_to_session(
    session_id: str,
    user_id: str,
) -> ChatSession:
    """
    Assign a user to a chat session.
    """
    session = await get_chat_session(session_id, None)
    if not session:
        raise NotFoundError(f"Session {session_id} not found")
    session.user_id = user_id
    return await upsert_chat_session(session)


async def stream_chat_completion(
    session_id: str,
    user_message: str,
    user_id: str | None = None,
) -> AsyncGenerator[StreamBaseResponse, None]:
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
    logger.info(
        f"Streaming chat completion for session {session_id} with user message {user_message} and user id {user_id}"
    )

    session = await get_chat_session(session_id, user_id)

    if not session:
        raise NotFoundError(
            f"Session {session_id} not found. Please create a new session first."
        )

    session.messages.append(ChatMessage(role="user", content=user_message))

    if len(session.messages) > config.max_context_messages:
        raise ValueError(f"Max messages exceeded: {config.max_context_messages}")

    logger.info(
        f"Upserting session: {session.session_id} with user id {session.user_id}"
    )
    session = await upsert_chat_session(session)
    assert session, "Session not found"

    assistant_repsonse = ChatMessage(
        role="assistant",
        content="",
    )

    has_yielded_end = False
    has_yielded_error = False
    try:
        async for chunk in _stream_chat_chunks(
            session=session,
            tools=tools,
        ):

            if isinstance(chunk, StreamTextChunk):
                assistant_repsonse.content += chunk.content
                yield chunk
            elif isinstance(chunk, StreamToolCall):
                tool_call_response = ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        {
                            "id": chunk.tool_id,
                            "type": "function",
                            "function": {
                                "name": chunk.tool_name,
                                "arguments": chunk.arguments,
                            },
                        }
                    ],
                )
                session.messages.append(tool_call_response)
            elif isinstance(chunk, StreamToolExecutionResult):
                session.messages.append(
                    ChatMessage(
                        role="tool",
                        content=orjson.dumps(chunk.result).decode("utf-8"),
                        tool_call_id=chunk.tool_id,
                    )
                )
                yield chunk
            elif isinstance(chunk, StreamEnd):
                has_yielded_end = True
                yield chunk
            elif isinstance(chunk, StreamError):
                has_yielded_error = True
                yield chunk
            elif isinstance(chunk, StreamUsage):
                session.usage.append(
                    Usage(
                        prompt_tokens=chunk.prompt_tokens,
                        completion_tokens=chunk.completion_tokens,
                        total_tokens=chunk.total_tokens,
                    )
                )
            else:
                logger.error(f"Unknown chunk type: {type(chunk)}", exc_info=True)
    except Exception as e:
        logger.error(f"Error in stream: {e!s}", exc_info=True)
        if not has_yielded_error:
            error_response = StreamError(
                message=str(e),
                timestamp=datetime.now(UTC).isoformat(),
            )
            yield error_response
        if not has_yielded_end:
            yield StreamEnd(
                timestamp=datetime.now(UTC).isoformat(),
            )
            has_yielded_end = True
    finally:
        # We always upsert the session even if an error occurs
        # So we dont lose track of tool call executions
        logger.info(
            f"Upserting session: {session.session_id} with user id {session.user_id}"
        )
        session.messages.append(assistant_repsonse)
        await upsert_chat_session(session)


async def _stream_chat_chunks(
    session: ChatSession,
    tools: list[ChatCompletionToolParam],
) -> AsyncGenerator[StreamBaseResponse, None]:
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
                messages=session.to_openai_messages(),
                tools=tools,
                tool_choice="auto",
                stream=True,
            )

            # Variables to accumulate the response
            assistant_message: str = ""
            tool_calls: list[dict[str, Any]] = []
            active_tool_call_idx = None
            finish_reason: str | None = None

            # Process the stream
            chunk: ChatCompletionChunk
            async for chunk in stream:
                logger.info(f"Chunk: \n\n{chunk}")
                if chunk.usage:
                    yield StreamUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                    )

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
                        yield text_response

                    # Handle tool calls
                    if delta.tool_calls:
                        for tc_chunk in delta.tool_calls:
                            idx = tc_chunk.index
                            if active_tool_call_idx is None:
                                active_tool_call_idx = idx

                            if active_tool_call_idx != idx:
                                yield_idx = idx - 1
                                async for tc in _yield_tool_call(
                                    tool_calls, yield_idx, session
                                ):
                                    yield tc

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

            if active_tool_call_idx is not None:
                async for tc in _yield_tool_call(
                    tool_calls, active_tool_call_idx, session
                ):
                    yield tc

            yield StreamEnd(
                timestamp=datetime.now(UTC).isoformat(),
            )
            return
        except Exception as e:
            logger.error(f"Error in stream: {e!s}", exc_info=True)
            error_response = StreamError(
                message=str(e),
                timestamp=datetime.now(UTC).isoformat(),
            )
            yield error_response
            yield StreamEnd(
                timestamp=datetime.now(UTC).isoformat(),
            )
            return


async def _yield_tool_call(
    tool_calls: list[dict[str, Any]],
    yield_idx: int,
    session: ChatSession,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """
    Yield a tool call.
    """
    logger.info(f"Yielding tool call: {tool_calls[yield_idx]}")
    yield StreamToolCall(
        tool_id=tool_calls[yield_idx]["id"],
        tool_name=tool_calls[yield_idx]["function"]["name"],
        arguments=orjson.loads(tool_calls[yield_idx]["function"]["arguments"]),
        timestamp=datetime.now(UTC).isoformat(),
    )

    tool_execution_response: StreamToolExecutionResult = await execute_tool(
        tool_name=tool_calls[yield_idx]["function"]["name"],
        parameters=orjson.loads(tool_calls[yield_idx]["function"]["arguments"]),
        tool_call_id=tool_calls[yield_idx]["id"],
        user_id=session.user_id,
        session_id=session.session_id,
    )
    logger.info(f"Yielding Tool execution response: {tool_execution_response}")
    yield tool_execution_response


if __name__ == "__main__":
    import asyncio

    async def main():
        session = await create_chat_session()
        async for chunk in stream_chat_completion(
            session.session_id,
            "Please find me an agent that can help me with my business. Call the tool twice once with the query 'money printing agent' and once with the query 'money generating agent'",
            user_id=session.user_id,
        ):
            print(chunk)

    asyncio.run(main())
