import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import orjson
from langfuse import get_client, propagate_attributes
from langfuse.openai import openai  # type: ignore
from openai import APIConnectionError, APIError, APIStatusError, RateLimitError
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam

from backend.data.understanding import (
    format_understanding_for_prompt,
    get_business_understanding,
)
from backend.util.exceptions import NotFoundError
from backend.util.settings import Settings

from .config import ChatConfig
from .model import (
    ChatMessage,
    ChatSession,
    Usage,
    get_chat_session,
    update_session_title,
    upsert_chat_session,
)
from .response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamStart,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
    StreamUsage,
)
from .tools import execute_tool, tools

logger = logging.getLogger(__name__)

config = ChatConfig()
settings = Settings()
client = openai.AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)


langfuse = get_client()


class LangfuseNotConfiguredError(Exception):
    """Raised when Langfuse is required but not configured."""

    pass


def _is_langfuse_configured() -> bool:
    """Check if Langfuse credentials are configured."""
    return bool(
        settings.secrets.langfuse_public_key and settings.secrets.langfuse_secret_key
    )


async def _build_system_prompt(user_id: str | None) -> tuple[str, Any]:
    """Build the full system prompt including business understanding if available.

    Args:
        user_id: The user ID for fetching business understanding
                     If "default" and this is the user's first session, will use "onboarding" instead.

    Returns:
        Tuple of (compiled prompt string, Langfuse prompt object for tracing)
    """

    # cache_ttl_seconds=0 disables SDK caching to always get the latest prompt
    prompt = langfuse.get_prompt(config.langfuse_prompt_name, cache_ttl_seconds=0)

    # If user is authenticated, try to fetch their business understanding
    understanding = None
    if user_id:
        try:
            understanding = await get_business_understanding(user_id)
        except Exception as e:
            logger.warning(f"Failed to fetch business understanding: {e}")
            understanding = None
    if understanding:
        context = format_understanding_for_prompt(understanding)
    else:
        context = "This is the first time you are meeting the user. Greet them and introduce them to the platform"

    compiled = prompt.compile(users_information=context)
    return compiled, understanding


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
    tool_call_response: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    retry_count: int = 0,
    session: ChatSession | None = None,
    context: dict[str, str] | None = None,  # {url: str, content: str}
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Main entry point for streaming chat completions with database handling.

    This function handles all database operations and delegates streaming
    to the internal _stream_chat_chunks function.

    Args:
        session_id: Chat session ID
        user_message: User's input message
        user_id: User ID for authentication (None for anonymous)
        session: Optional pre-loaded session object (for recursive calls to avoid Redis refetch)

    Yields:
        StreamBaseResponse objects formatted as SSE

    Raises:
        NotFoundError: If session_id is invalid
        ValueError: If max_context_messages is exceeded

    """
    logger.info(
        f"Streaming chat completion for session {session_id} for message {message} and user id {user_id}. Message is user message: {is_user_message}"
    )

    # Check if Langfuse is configured - required for chat functionality
    if not _is_langfuse_configured():
        logger.error("Chat request failed: Langfuse is not configured")
        yield StreamError(
            errorText="Chat service is not available. Langfuse must be configured "
            "with LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables."
        )
        yield StreamFinish()
        return

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

            # Capture only the values we need (not the session object) to avoid
            # stale data issues when the main flow modifies the session
            captured_session_id = session_id
            captured_message = message

            async def _update_title():
                try:
                    title = await _generate_session_title(captured_message)
                    if title:
                        # Use dedicated title update function that doesn't
                        # touch messages, avoiding race conditions
                        await update_session_title(captured_session_id, title)
                        logger.info(
                            f"Generated title for session {captured_session_id}: {title}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to update session title: {e}")

            # Fire and forget - don't block the chat response
            asyncio.create_task(_update_title())

    # Build system prompt with business understanding
    system_prompt, understanding = await _build_system_prompt(user_id)

    # Create Langfuse trace for this LLM call (each call gets its own trace, grouped by session_id)
    # Using v3 SDK: start_observation creates a root span, update_trace sets trace-level attributes
    input = message
    if not message and tool_call_response:
        input = tool_call_response

    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="span",
        name="user-copilot-request",
        input=input,
    ) as span:
        with propagate_attributes(
            session_id=session_id,
            user_id=user_id,
            tags=["copilot"],
            metadata={
                "users_information": format_understanding_for_prompt(understanding)[
                    :200
                ]  # langfuse only accepts upto to 200 chars
            },
        ):

            # Initialize variables that will be used in finally block (must be defined before try)
            assistant_response = ChatMessage(
                role="assistant",
                content="",
            )
            accumulated_tool_calls: list[dict[str, Any]] = []

            # Wrap main logic in try/finally to ensure Langfuse observations are always ended
            has_yielded_end = False
            has_yielded_error = False
            has_done_tool_call = False
            has_received_text = False
            text_streaming_ended = False
            tool_response_messages: list[ChatMessage] = []
            should_retry = False

            # Generate unique IDs for AI SDK protocol
            import uuid as uuid_module

            message_id = str(uuid_module.uuid4())
            text_block_id = str(uuid_module.uuid4())

            # Yield message start
            yield StreamStart(messageId=message_id)

            try:
                async for chunk in _stream_chat_chunks(
                    session=session,
                    tools=tools,
                    system_prompt=system_prompt,
                    text_block_id=text_block_id,
                ):

                    if isinstance(chunk, StreamTextStart):
                        # Emit text-start before first text delta
                        if not has_received_text:
                            yield chunk
                    elif isinstance(chunk, StreamTextDelta):
                        delta = chunk.delta or ""
                        assert assistant_response.content is not None
                        assistant_response.content += delta
                        has_received_text = True
                        yield chunk
                    elif isinstance(chunk, StreamTextEnd):
                        # Emit text-end after text completes
                        if has_received_text and not text_streaming_ended:
                            text_streaming_ended = True
                            if assistant_response.content:
                                logger.warn(
                                    f"StreamTextEnd: Attempting to set output {assistant_response.content}"
                                )
                                span.update_trace(output=assistant_response.content)
                                span.update(output=assistant_response.content)
                            yield chunk
                    elif isinstance(chunk, StreamToolInputStart):
                        # Emit text-end before first tool call, but only if we've received text
                        if has_received_text and not text_streaming_ended:
                            yield StreamTextEnd(id=text_block_id)
                            text_streaming_ended = True
                        yield chunk
                    elif isinstance(chunk, StreamToolInputAvailable):
                        # Accumulate tool calls in OpenAI format
                        accumulated_tool_calls.append(
                            {
                                "id": chunk.toolCallId,
                                "type": "function",
                                "function": {
                                    "name": chunk.toolName,
                                    "arguments": orjson.dumps(chunk.input).decode(
                                        "utf-8"
                                    ),
                                },
                            }
                        )
                    elif isinstance(chunk, StreamToolOutputAvailable):
                        result_content = (
                            chunk.output
                            if isinstance(chunk.output, str)
                            else orjson.dumps(chunk.output).decode("utf-8")
                        )
                        tool_response_messages.append(
                            ChatMessage(
                                role="tool",
                                content=result_content,
                                tool_call_id=chunk.toolCallId,
                            )
                        )
                        has_done_tool_call = True
                        # Track if any tool execution failed
                        if not chunk.success:
                            logger.warning(
                                f"Tool {chunk.toolName} (ID: {chunk.toolCallId}) execution failed"
                            )
                        yield chunk
                    elif isinstance(chunk, StreamFinish):
                        if not has_done_tool_call:
                            # Emit text-end before finish if we received text but haven't closed it
                            if has_received_text and not text_streaming_ended:
                                yield StreamTextEnd(id=text_block_id)
                                text_streaming_ended = True
                            has_yielded_end = True
                            yield chunk
                    elif isinstance(chunk, StreamError):
                        has_yielded_error = True
                    elif isinstance(chunk, StreamUsage):
                        session.usage.append(
                            Usage(
                                prompt_tokens=chunk.promptTokens,
                                completion_tokens=chunk.completionTokens,
                                total_tokens=chunk.totalTokens,
                            )
                        )
                    else:
                        logger.error(
                            f"Unknown chunk type: {type(chunk)}", exc_info=True
                        )
                if assistant_response.content:
                    langfuse.update_current_trace(output=assistant_response.content)
                    langfuse.update_current_span(output=assistant_response.content)
                elif tool_response_messages:
                    langfuse.update_current_trace(output=str(tool_response_messages))
                    langfuse.update_current_span(output=str(tool_response_messages))

            except Exception as e:
                logger.error(f"Error during stream: {e!s}", exc_info=True)

                # Check if this is a retryable error (JSON parsing, incomplete tool calls, etc.)
                is_retryable = isinstance(
                    e, (orjson.JSONDecodeError, KeyError, TypeError)
                )

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
                            error_message = f"Max retries ({config.max_retries}) exceeded: {error_message}"

                        error_response = StreamError(errorText=error_message)
                        yield error_response
                    if not has_yielded_end:
                        yield StreamFinish()
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
                    context=context,
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
            logger.info(
                f"Extended session messages, new message_count={len(session.messages)}"
            )
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
                    context=context,
                    tool_call_response=str(tool_response_messages),
                ):
                    yield chunk


# Retry configuration for OpenAI API calls
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 30.0


def _is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable."""
    if isinstance(error, RateLimitError):
        return True
    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIStatusError):
        # APIStatusError has a response with status_code
        # Retry on 5xx status codes (server errors)
        if error.response.status_code >= 500:
            return True
    if isinstance(error, APIError):
        # Retry on overloaded errors or 500 errors (may not have status code)
        error_message = str(error).lower()
        if "overloaded" in error_message or "internal server error" in error_message:
            return True
    return False


async def _stream_chat_chunks(
    session: ChatSession,
    tools: list[ChatCompletionToolParam],
    system_prompt: str | None = None,
    text_block_id: str | None = None,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """
    Pure streaming function for OpenAI chat completions with tool calling.

    This function is database-agnostic and focuses only on streaming logic.
    Implements exponential backoff retry for transient API errors.

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
        retry_count = 0
        last_error: Exception | None = None

        while retry_count <= MAX_RETRIES:
            try:
                logger.info(
                    f"Creating OpenAI chat completion stream..."
                    f"{f' (retry {retry_count}/{MAX_RETRIES})' if retry_count > 0 else ''}"
                )

                # Create the stream with proper types
                stream = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                    stream_options={"include_usage": True},
                )

                # Variables to accumulate tool calls
                tool_calls: list[dict[str, Any]] = []
                active_tool_call_idx: int | None = None
                finish_reason: str | None = None
                # Track which tool call indices have had their start event emitted
                emitted_start_for_idx: set[int] = set()

                # Track if we've started the text block
                text_started = False

                # Process the stream
                chunk: ChatCompletionChunk
                async for chunk in stream:
                    if chunk.usage:
                        yield StreamUsage(
                            promptTokens=chunk.usage.prompt_tokens,
                            completionTokens=chunk.usage.completion_tokens,
                            totalTokens=chunk.usage.total_tokens,
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
                            # Emit text-start on first text content
                            if not text_started and text_block_id:
                                yield StreamTextStart(id=text_block_id)
                                text_started = True
                            # Stream the text delta
                            text_response = StreamTextDelta(
                                id=text_block_id or "",
                                delta=delta.content,
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

                                # Emit StreamToolInputStart only after we have the tool call ID
                                if (
                                    idx not in emitted_start_for_idx
                                    and tool_calls[idx]["id"]
                                    and tool_calls[idx]["function"]["name"]
                                ):
                                    yield StreamToolInputStart(
                                        toolCallId=tool_calls[idx]["id"],
                                        toolName=tool_calls[idx]["function"]["name"],
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
                            errorText=f"Invalid tool call arguments for tool {tool_call.get('function', {}).get('name', 'unknown')}: {e}",
                        )
                        # Re-raise to trigger retry logic in the parent function
                        raise

                yield StreamFinish()
                return
            except Exception as e:
                last_error = e
                if _is_retryable_error(e) and retry_count < MAX_RETRIES:
                    retry_count += 1
                    # Calculate delay with exponential backoff
                    delay = min(
                        BASE_DELAY_SECONDS * (2 ** (retry_count - 1)),
                        MAX_DELAY_SECONDS,
                    )
                    logger.warning(
                        f"Retryable error in stream: {e!s}. "
                        f"Retrying in {delay:.1f}s (attempt {retry_count}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(delay)
                    continue  # Retry the stream
                else:
                    # Non-retryable error or max retries exceeded
                    logger.error(
                        f"Error in stream (not retrying): {e!s}",
                        exc_info=True,
                    )
                    error_response = StreamError(errorText=str(e))
                    yield error_response
                    yield StreamFinish()
                    return

        # If we exit the retry loop without returning, it means we exhausted retries
        if last_error:
            logger.error(
                f"Max retries ({MAX_RETRIES}) exceeded. Last error: {last_error!s}",
                exc_info=True,
            )
            yield StreamError(errorText=f"Max retries exceeded: {last_error!s}")
            yield StreamFinish()
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
    tool_name = tool_calls[yield_idx]["function"]["name"]
    tool_call_id = tool_calls[yield_idx]["id"]
    logger.info(f"Yielding tool call: {tool_calls[yield_idx]}")

    # Parse tool call arguments - handle empty arguments gracefully
    raw_arguments = tool_calls[yield_idx]["function"]["arguments"]
    if raw_arguments:
        arguments = orjson.loads(raw_arguments)
    else:
        arguments = {}

    yield StreamToolInputAvailable(
        toolCallId=tool_call_id,
        toolName=tool_name,
        input=arguments,
    )

    tool_execution_response: StreamToolOutputAvailable = await execute_tool(
        tool_name=tool_name,
        parameters=arguments,
        tool_call_id=tool_call_id,
        user_id=session.user_id,
        session=session,
    )

    yield tool_execution_response
