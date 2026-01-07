import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

import orjson
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam

from backend.data.understanding import (
    format_understanding_for_prompt,
    get_business_understanding,
)
from backend.util.exceptions import NotFoundError

from . import db as chat_db
from .config import ChatConfig
from .model import (
    ChatMessage,
    ChatSession,
    Usage,
    get_chat_session,
    upsert_chat_session,
)
from .model import (
    create_chat_session as model_create_chat_session,
)
from .response_model import (
    StreamBaseResponse,
    StreamEnd,
    StreamError,
    StreamTextChunk,
    StreamTextEnded,
    StreamToolCall,
    StreamToolCallStart,
    StreamToolExecutionResult,
    StreamUsage,
)
from .tools import execute_tool, tools

logger = logging.getLogger(__name__)

config = ChatConfig()
client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)


async def _is_first_session(user_id: str) -> bool:
    """Check if this is the user's first chat session.

    Returns True if the user has 1 or fewer sessions (meaning this is their first).
    """
    try:
        session_count = await chat_db.get_user_session_count(user_id)
        return session_count <= 1
    except Exception as e:
        logger.warning(f"Failed to check session count for user {user_id}: {e}")
        return False  # Default to non-onboarding if we can't check


async def _build_system_prompt(
    user_id: str | None, prompt_type: str = "default"
) -> str:
    """Build the full system prompt including business understanding if available.

    Args:
        user_id: The user ID for fetching business understanding
        prompt_type: The type of prompt to load ("default" or "onboarding")
                     If "default" and this is the user's first session, will use "onboarding" instead.

    Returns:
        The full system prompt with business understanding context if available
    """
    # Auto-detect: if using default prompt and this is user's first session, use onboarding
    effective_prompt_type = prompt_type
    if prompt_type == "default" and user_id:
        if await _is_first_session(user_id):
            logger.info("First session detected for user, using onboarding prompt")
            effective_prompt_type = "onboarding"

    # Start with the base system prompt for the specified type
    base_prompt = config.get_system_prompt_for_type(effective_prompt_type)

    # If user is authenticated, try to fetch their business understanding
    if user_id:
        try:
            understanding = await get_business_understanding(user_id)
            if understanding:
                context = format_understanding_for_prompt(understanding)
                if context:
                    return (
                        f"{base_prompt}\n\n---\n\n"
                        f"{context}\n\n"
                        "Use this context to provide more personalized recommendations "
                        "and to better understand the user's business needs when "
                        "suggesting agents and automations."
                    )
        except Exception as e:
            logger.warning(f"Failed to fetch business understanding: {e}")

    return base_prompt


async def _generate_session_title(message: str) -> str | None:
    """Generate a concise title for a chat session based on the first message.

    Args:
        message: The first user message in the session

    Returns:
        A short title (3-6 words) or None if generation fails
    """
    try:
        response = await client.chat.completions.create(
            model=config.title_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (3-6 words) for a chat conversation "
                        "based on the user's first message. The title should capture the "
                        "main topic or intent. Return ONLY the title, no quotes or punctuation."
                    ),
                },
                {"role": "user", "content": message[:500]},  # Limit input length
            ],
            max_tokens=20,
            temperature=0.7,
        )
        title = response.choices[0].message.content
        if title:
            # Clean up the title
            title = title.strip().strip("\"'")
            # Limit length
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        return None
    except Exception as e:
        logger.warning(f"Failed to generate session title: {e}")
        return None


async def create_chat_session(
    user_id: str | None = None,
) -> ChatSession:
    """
    Create a new chat session and persist it to the database.
    """
    return await model_create_chat_session(user_id)


async def get_session(
    session_id: str,
    user_id: str | None = None,
) -> ChatSession | None:
    """
    Get a chat session by ID.
    """
    return await get_chat_session(session_id, user_id)


async def get_user_sessions(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[ChatSession]:
    """
    Get all chat sessions for a user.
    """
    from .model import get_user_sessions as model_get_user_sessions

    return await model_get_user_sessions(user_id, limit, offset)


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
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    retry_count: int = 0,
    session: ChatSession | None = None,
    context: dict[str, str] | None = None,  # {url: str, content: str}
    prompt_type: str = "default",
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Main entry point for streaming chat completions with database handling.

    This function handles all database operations and delegates streaming
    to the internal _stream_chat_chunks function.

    Args:
        session_id: Chat session ID
        user_message: User's input message
        user_id: User ID for authentication (None for anonymous)
        session: Optional pre-loaded session object (for recursive calls to avoid Redis refetch)
        prompt_type: The type of prompt to use ("default" or "onboarding")

    Yields:
        StreamBaseResponse objects formatted as SSE

    Raises:
        NotFoundError: If session_id is invalid
        ValueError: If max_context_messages is exceeded

    """
    logger.info(
        f"Streaming chat completion for session {session_id} for message {message} and user id {user_id}. Message is user message: {is_user_message}"
    )

    # Only fetch from Redis if session not provided (initial call)
    if session is None:
        session = await get_chat_session(session_id, user_id)
        logger.info(
            f"Fetched session from Redis: {session.session_id if session else 'None'}, "
            f"message_count={len(session.messages) if session else 0}"
        )
    else:
        logger.info(
            f"Using provided session object: {session.session_id}, "
            f"message_count={len(session.messages)}"
        )

    if not session:
        raise NotFoundError(
            f"Session {session_id} not found. Please create a new session first."
        )

    if message:
        # Build message content with context if provided
        message_content = message
        if context and context.get("url") and context.get("content"):
            context_text = f"Page URL: {context['url']}\n\nPage Content:\n{context['content']}\n\n---\n\nUser Message: {message}"
            message_content = context_text
            logger.info(
                f"Including page context: URL={context['url']}, content_length={len(context['content'])}"
            )

        session.messages.append(
            ChatMessage(
                role="user" if is_user_message else "assistant", content=message_content
            )
        )
        logger.info(
            f"Appended message (role={'user' if is_user_message else 'assistant'}), "
            f"new message_count={len(session.messages)}"
        )

    if len(session.messages) > config.max_context_messages:
        raise ValueError(f"Max messages exceeded: {config.max_context_messages}")

    logger.info(
        f"Upserting session: {session.session_id} with user id {session.user_id}, "
        f"message_count={len(session.messages)}"
    )
    session = await upsert_chat_session(session)
    assert session, "Session not found"

    # Generate title for new sessions on first user message (non-blocking)
    # Check: is_user_message, no title yet, and this is the first user message
    if is_user_message and message and not session.title:
        user_messages = [m for m in session.messages if m.role == "user"]
        if len(user_messages) == 1:
            # First user message - generate title in background
            import asyncio

            async def _update_title():
                try:
                    title = await _generate_session_title(message)
                    if title:
                        session.title = title
                        await upsert_chat_session(session)
                        logger.info(
                            f"Generated title for session {session_id}: {title}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to update session title: {e}")

            # Fire and forget - don't block the chat response
            asyncio.create_task(_update_title())

    # Build system prompt with business understanding
    system_prompt = await _build_system_prompt(user_id, prompt_type)

    assistant_response = ChatMessage(
        role="assistant",
        content="",
    )

    has_yielded_end = False
    has_yielded_error = False
    has_done_tool_call = False
    has_received_text = False
    text_streaming_ended = False
    tool_response_messages: list[ChatMessage] = []
    accumulated_tool_calls: list[dict[str, Any]] = []
    should_retry = False

    try:
        async for chunk in _stream_chat_chunks(
            session=session,
            tools=tools,
            system_prompt=system_prompt,
        ):

            if isinstance(chunk, StreamTextChunk):
                content = chunk.content or ""
                assert assistant_response.content is not None
                assistant_response.content += content
                has_received_text = True
                yield chunk
            elif isinstance(chunk, StreamToolCallStart):
                # Emit text_ended before first tool call, but only if we've received text
                if has_received_text and not text_streaming_ended:
                    yield StreamTextEnded()
                    text_streaming_ended = True
                yield chunk
            elif isinstance(chunk, StreamToolCall):
                # Accumulate tool calls in OpenAI format
                accumulated_tool_calls.append(
                    {
                        "id": chunk.tool_id,
                        "type": "function",
                        "function": {
                            "name": chunk.tool_name,
                            "arguments": orjson.dumps(chunk.arguments).decode("utf-8"),
                        },
                    }
                )
            elif isinstance(chunk, StreamToolExecutionResult):
                result_content = (
                    chunk.result
                    if isinstance(chunk.result, str)
                    else orjson.dumps(chunk.result).decode("utf-8")
                )
                tool_response_messages.append(
                    ChatMessage(
                        role="tool",
                        content=result_content,
                        tool_call_id=chunk.tool_id,
                    )
                )
                has_done_tool_call = True
                # Track if any tool execution failed
                if not chunk.success:
                    logger.warning(
                        f"Tool {chunk.tool_name} (ID: {chunk.tool_id}) execution failed"
                    )
                yield chunk
            elif isinstance(chunk, StreamEnd):
                if not has_done_tool_call:
                    has_yielded_end = True
                    yield chunk
            elif isinstance(chunk, StreamError):
                has_yielded_error = True
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
        logger.error(f"Error during stream: {e!s}", exc_info=True)

        # Check if this is a retryable error (JSON parsing, incomplete tool calls, etc.)
        is_retryable = isinstance(e, (orjson.JSONDecodeError, KeyError, TypeError))

        if is_retryable and retry_count < config.max_retries:
            logger.info(
                f"Retryable error encountered. Attempt {retry_count + 1}/{config.max_retries}"
            )
            should_retry = True
        else:
            # Non-retryable error or max retries exceeded
            # Save any partial progress before reporting error
            messages_to_save: list[ChatMessage] = []

            # Add assistant message if it has content or tool calls
            if accumulated_tool_calls:
                assistant_response.tool_calls = accumulated_tool_calls
            if assistant_response.content or assistant_response.tool_calls:
                messages_to_save.append(assistant_response)

            # Add tool response messages after assistant message
            messages_to_save.extend(tool_response_messages)

            session.messages.extend(messages_to_save)
            await upsert_chat_session(session)

            if not has_yielded_error:
                error_message = str(e)
                if not is_retryable:
                    error_message = f"Non-retryable error: {error_message}"
                elif retry_count >= config.max_retries:
                    error_message = (
                        f"Max retries ({config.max_retries}) exceeded: {error_message}"
                    )

                error_response = StreamError(
                    message=error_message,
                    timestamp=datetime.now(UTC).isoformat(),
                )
                yield error_response
            if not has_yielded_end:
                yield StreamEnd(
                    timestamp=datetime.now(UTC).isoformat(),
                )
            return

    # Handle retry outside of exception handler to avoid nesting
    if should_retry and retry_count < config.max_retries:
        logger.info(
            f"Retrying stream_chat_completion for session {session_id}, attempt {retry_count + 1}"
        )
        async for chunk in stream_chat_completion(
            session_id=session.session_id,
            user_id=user_id,
            retry_count=retry_count + 1,
            session=session,
            prompt_type=prompt_type,
        ):
            yield chunk
        return  # Exit after retry to avoid double-saving in finally block

    # Normal completion path - save session and handle tool call continuation
    logger.info(
        f"Normal completion path: session={session.session_id}, "
        f"current message_count={len(session.messages)}"
    )

    # Build the messages list in the correct order
    messages_to_save: list[ChatMessage] = []

    # Add assistant message with tool_calls if any
    if accumulated_tool_calls:
        assistant_response.tool_calls = accumulated_tool_calls
        logger.info(
            f"Added {len(accumulated_tool_calls)} tool calls to assistant message"
        )
    if assistant_response.content or assistant_response.tool_calls:
        messages_to_save.append(assistant_response)
        logger.info(
            f"Saving assistant message with content_len={len(assistant_response.content or '')}, tool_calls={len(assistant_response.tool_calls or [])}"
        )

    # Add tool response messages after assistant message
    messages_to_save.extend(tool_response_messages)
    logger.info(
        f"Saving {len(tool_response_messages)} tool response messages, "
        f"total_to_save={len(messages_to_save)}"
    )

    session.messages.extend(messages_to_save)
    logger.info(f"Extended session messages, new message_count={len(session.messages)}")
    await upsert_chat_session(session)

    # If we did a tool call, stream the chat completion again to get the next response
    if has_done_tool_call:
        logger.info(
            "Tool call executed, streaming chat completion again to get assistant response"
        )
        async for chunk in stream_chat_completion(
            session_id=session.session_id,
            user_id=user_id,
            session=session,  # Pass session object to avoid Redis refetch
            prompt_type=prompt_type,
        ):
            yield chunk


async def _stream_chat_chunks(
    session: ChatSession,
    tools: list[ChatCompletionToolParam],
    system_prompt: str | None = None,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """
    Pure streaming function for OpenAI chat completions with tool calling.

    This function is database-agnostic and focuses only on streaming logic.

    Args:
        session: Chat session with conversation history
        tools: Available tools for the model
        system_prompt: System prompt to prepend to messages

    Yields:
        SSE formatted JSON response objects

    """
    model = config.model

    logger.info("Starting pure chat stream")

    # Build messages with system prompt prepended
    messages = session.to_openai_messages()
    if system_prompt:
        from openai.types.chat import ChatCompletionSystemMessageParam

        system_message = ChatCompletionSystemMessageParam(
            role="system",
            content=system_prompt,
        )
        messages = [system_message] + messages

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

            # Variables to accumulate tool calls
            tool_calls: list[dict[str, Any]] = []
            active_tool_call_idx: int | None = None
            finish_reason: str | None = None
            # Track which tool call indices have had their start event emitted
            emitted_start_for_idx: set[int] = set()

            # Process the stream
            chunk: ChatCompletionChunk
            async for chunk in stream:
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

                            # Update active tool call index if needed
                            if (
                                active_tool_call_idx is None
                                or active_tool_call_idx != idx
                            ):
                                active_tool_call_idx = idx

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

                            # Emit StreamToolCallStart only after we have the tool call ID
                            if (
                                idx not in emitted_start_for_idx
                                and tool_calls[idx]["id"]
                                and tool_calls[idx]["function"]["name"]
                            ):
                                yield StreamToolCallStart(
                                    tool_id=tool_calls[idx]["id"],
                                    tool_name=tool_calls[idx]["function"]["name"],
                                    timestamp=datetime.now(UTC).isoformat(),
                                )
                                emitted_start_for_idx.add(idx)
            logger.info(f"Stream complete. Finish reason: {finish_reason}")

            # Yield all accumulated tool calls after the stream is complete
            # This ensures all tool call arguments have been fully received
            for idx, tool_call in enumerate(tool_calls):
                try:
                    async for tc in _yield_tool_call(tool_calls, idx, session):
                        yield tc
                except (orjson.JSONDecodeError, KeyError, TypeError) as e:
                    logger.error(
                        f"Failed to parse tool call {idx}: {e}",
                        exc_info=True,
                        extra={"tool_call": tool_call},
                    )
                    yield StreamError(
                        message=f"Invalid tool call arguments for tool {tool_call.get('function', {}).get('name', 'unknown')}: {e}",
                        timestamp=datetime.now(UTC).isoformat(),
                    )
                    # Re-raise to trigger retry logic in the parent function
                    raise

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
    Yield a tool call and its execution result.

    Raises:
        orjson.JSONDecodeError: If tool call arguments cannot be parsed as JSON
        KeyError: If expected tool call fields are missing
        TypeError: If tool call structure is invalid
    """
    logger.info(f"Yielding tool call: {tool_calls[yield_idx]}")

    # Parse tool call arguments - handle empty arguments gracefully
    raw_arguments = tool_calls[yield_idx]["function"]["arguments"]
    if raw_arguments:
        arguments = orjson.loads(raw_arguments)
    else:
        arguments = {}

    yield StreamToolCall(
        tool_id=tool_calls[yield_idx]["id"],
        tool_name=tool_calls[yield_idx]["function"]["name"],
        arguments=arguments,
        timestamp=datetime.now(UTC).isoformat(),
    )

    tool_execution_response: StreamToolExecutionResult = await execute_tool(
        tool_name=tool_calls[yield_idx]["function"]["name"],
        parameters=arguments,
        tool_call_id=tool_calls[yield_idx]["id"],
        user_id=session.user_id,
        session=session,
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
