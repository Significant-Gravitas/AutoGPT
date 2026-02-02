import asyncio
import logging
import time
from asyncio import CancelledError
from collections.abc import AsyncGenerator
from typing import Any

import openai
import orjson
from langfuse import get_client
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    PermissionDeniedError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionChunk, ChatCompletionToolParam

from backend.data.redis_client import get_redis_async
from backend.data.understanding import (
    format_understanding_for_prompt,
    get_business_understanding,
)
from backend.util.exceptions import NotFoundError
from backend.util.settings import Settings

from . import db as chat_db
from .config import ChatConfig
from .model import (
    ChatMessage,
    ChatSession,
    Usage,
    cache_chat_session,
    get_chat_session,
    invalidate_session_cache,
    update_session_title,
    upsert_chat_session,
)
from .response_model import (
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
from .tools import execute_tool, get_tool, tools
from .tools.models import (
    ErrorResponse,
    OperationInProgressResponse,
    OperationPendingResponse,
    OperationStartedResponse,
)
from .tracking import track_user_message

logger = logging.getLogger(__name__)

config = ChatConfig()
settings = Settings()
client = openai.AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)


langfuse = get_client()

# Redis key prefix for tracking running long-running operations
# Used for idempotency across Kubernetes pods - prevents duplicate executions on browser refresh
RUNNING_OPERATION_PREFIX = "chat:running_operation:"

# Default system prompt used when Langfuse is not configured
# This is a snapshot of the "CoPilot Prompt" from Langfuse (version 11)
DEFAULT_SYSTEM_PROMPT = """You are **Otto**, an AI Co-Pilot for AutoGPT and a Forward-Deployed Automation Engineer serving small business owners. Your mission is to help users automate business tasks with AI by delivering tangible value through working automations—not through documentation or lengthy explanations.

Here is everything you know about the current user from previous interactions:

<users_information>
{users_information}
</users_information>

## YOUR CORE MANDATE

You are action-oriented. Your success is measured by:
- **Value Delivery**: Does the user think "wow, that was amazing" or "what was the point"?
- **Demonstrable Proof**: Show working automations, not descriptions of what's possible
- **Time Saved**: Focus on tangible efficiency gains
- **Quality Output**: Deliver results that meet or exceed expectations

## YOUR WORKFLOW

Adapt flexibly to the conversation context. Not every interaction requires all stages:

1. **Explore & Understand**: Learn about the user's business, tasks, and goals. Use `add_understanding` to capture important context that will improve future conversations.

2. **Assess Automation Potential**: Help the user understand whether and how AI can automate their task.

3. **Prepare for AI**: Provide brief, actionable guidance on prerequisites (data, access, etc.).

4. **Discover or Create Agents**:
   - **Always check the user's library first** with `find_library_agent` (these may be customized to their needs)
   - Search the marketplace with `find_agent` for pre-built automations
   - Find reusable components with `find_block`
   - Create custom solutions with `create_agent` if nothing suitable exists
   - Modify existing library agents with `edit_agent`

5. **Execute**: Run automations immediately, schedule them, or set up webhooks using `run_agent`. Test specific components with `run_block`.

6. **Show Results**: Display outputs using `agent_output`.

## AVAILABLE TOOLS

**Understanding & Discovery:**
- `add_understanding`: Create a memory about the user's business or use cases for future sessions
- `search_docs`: Search platform documentation for specific technical information
- `get_doc_page`: Retrieve full text of a specific documentation page

**Agent Discovery:**
- `find_library_agent`: Search the user's existing agents (CHECK HERE FIRST—these may be customized)
- `find_agent`: Search the marketplace for pre-built automations
- `find_block`: Find pre-written code units that perform specific tasks (agents are built from blocks)

**Agent Creation & Editing:**
- `create_agent`: Create a new automation agent
- `edit_agent`: Modify an agent in the user's library

**Execution & Output:**
- `run_agent`: Run an agent now, schedule it, or set up a webhook trigger
- `run_block`: Test or run a specific block independently
- `agent_output`: View results from previous agent runs

## BEHAVIORAL GUIDELINES

**Be Concise:**
- Target 2-5 short lines maximum
- Make every word count—no repetition or filler
- Use lightweight structure for scannability (bullets, numbered lists, short prompts)
- Avoid jargon (blocks, slugs, cron) unless the user asks

**Be Proactive:**
- Suggest next steps before being asked
- Anticipate needs based on conversation context and user information
- Look for opportunities to expand scope when relevant
- Reveal capabilities through action, not explanation

**Use Tools Effectively:**
- Select the right tool for each task
- **Always check `find_library_agent` before searching the marketplace**
- Use `add_understanding` to capture valuable business context
- When tool calls fail, try alternative approaches

## CRITICAL REMINDER

You are NOT a chatbot. You are NOT documentation. You are a partner who helps busy business owners get value quickly by showing proof through working automations. Bias toward action over explanation."""

# Module-level set to hold strong references to background tasks.
# This prevents asyncio from garbage collecting tasks before they complete.
# Tasks are automatically removed on completion via done_callback.
_background_tasks: set[asyncio.Task] = set()


async def _mark_operation_started(tool_call_id: str) -> bool:
    """Mark a long-running operation as started (Redis-based).

    Returns True if successfully marked (operation was not already running),
    False if operation was already running (lost race condition).
    Raises exception if Redis is unavailable (fail-closed).
    """
    redis = await get_redis_async()
    key = f"{RUNNING_OPERATION_PREFIX}{tool_call_id}"
    # SETNX with TTL - atomic "set if not exists"
    result = await redis.set(key, "1", ex=config.long_running_operation_ttl, nx=True)
    return result is not None


async def _mark_operation_completed(tool_call_id: str) -> None:
    """Mark a long-running operation as completed (remove Redis key).

    This is best-effort - if Redis fails, the TTL will eventually clean up.
    """
    try:
        redis = await get_redis_async()
        key = f"{RUNNING_OPERATION_PREFIX}{tool_call_id}"
        await redis.delete(key)
    except Exception as e:
        # Non-critical: TTL will clean up eventually
        logger.warning(f"Failed to delete running operation key {tool_call_id}: {e}")


def _is_langfuse_configured() -> bool:
    """Check if Langfuse credentials are configured."""
    return bool(
        settings.secrets.langfuse_public_key and settings.secrets.langfuse_secret_key
    )


async def _get_system_prompt_template(context: str) -> str:
    """Get the system prompt, trying Langfuse first with fallback to default.

    Args:
        context: The user context/information to compile into the prompt.

    Returns:
        The compiled system prompt string.
    """
    if _is_langfuse_configured():
        try:
            # cache_ttl_seconds=0 disables SDK caching to always get the latest prompt
            # Use asyncio.to_thread to avoid blocking the event loop
            prompt = await asyncio.to_thread(
                langfuse.get_prompt, config.langfuse_prompt_name, cache_ttl_seconds=0
            )
            return prompt.compile(users_information=context)
        except Exception as e:
            logger.warning(f"Failed to fetch prompt from Langfuse, using default: {e}")

    # Fallback to default prompt
    return DEFAULT_SYSTEM_PROMPT.format(users_information=context)


async def _build_system_prompt(user_id: str | None) -> tuple[str, Any]:
    """Build the full system prompt including business understanding if available.

    Args:
        user_id: The user ID for fetching business understanding
                     If "default" and this is the user's first session, will use "onboarding" instead.

    Returns:
        Tuple of (compiled prompt string, business understanding object)
    """
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

    compiled = await _get_system_prompt_template(context)
    return compiled, understanding


async def _generate_session_title(
    message: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str | None:
    """Generate a concise title for a chat session based on the first message.

    Args:
        message: The first user message in the session
        user_id: User ID for OpenRouter tracing (optional)
        session_id: Session ID for OpenRouter tracing (optional)

    Returns:
        A short title (3-6 words) or None if generation fails
    """
    try:
        # Build extra_body for OpenRouter tracing and PostHog analytics
        extra_body: dict[str, Any] = {}
        if user_id:
            extra_body["user"] = user_id[:128]  # OpenRouter limit
            extra_body["posthogDistinctId"] = user_id
        if session_id:
            extra_body["session_id"] = session_id[:128]  # OpenRouter limit
        extra_body["posthogProperties"] = {
            "environment": settings.config.app_env.value,
        }

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
            extra_body=extra_body,
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
        session.messages.append(
            ChatMessage(
                role="user" if is_user_message else "assistant", content=message
            )
        )
        logger.info(
            f"Appended message (role={'user' if is_user_message else 'assistant'}), "
            f"new message_count={len(session.messages)}"
        )

        # Track user message in PostHog
        if is_user_message:
            track_user_message(
                user_id=user_id,
                session_id=session_id,
                message_length=len(message),
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
            captured_user_id = user_id

            async def _update_title():
                try:
                    title = await _generate_session_title(
                        captured_message,
                        user_id=captured_user_id,
                        session_id=captured_session_id,
                    )
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

    # Initialize variables for streaming
    assistant_response = ChatMessage(
        role="assistant",
        content="",
    )
    accumulated_tool_calls: list[dict[str, Any]] = []
    has_saved_assistant_message = False
    has_appended_streaming_message = False
    last_cache_time = 0.0
    last_cache_content_len = 0

    has_yielded_end = False
    has_yielded_error = False
    has_done_tool_call = False
    has_long_running_tool_call = False  # Track if we had a long-running tool call
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
                if not has_appended_streaming_message:
                    session.messages.append(assistant_response)
                    has_appended_streaming_message = True
                current_time = time.monotonic()
                content_len = len(assistant_response.content)
                if (
                    current_time - last_cache_time >= 1.0
                    and content_len > last_cache_content_len
                ):
                    try:
                        await cache_chat_session(session)
                    except Exception as e:
                        logger.warning(
                            f"Failed to cache partial session {session.session_id}: {e}"
                        )
                    last_cache_time = current_time
                    last_cache_content_len = content_len
                yield chunk
            elif isinstance(chunk, StreamTextEnd):
                # Emit text-end after text completes
                if has_received_text and not text_streaming_ended:
                    text_streaming_ended = True
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
                            "arguments": orjson.dumps(chunk.input).decode("utf-8"),
                        },
                    }
                )
                yield chunk
            elif isinstance(chunk, StreamToolOutputAvailable):
                result_content = (
                    chunk.output
                    if isinstance(chunk.output, str)
                    else orjson.dumps(chunk.output).decode("utf-8")
                )
                # Skip saving long-running operation responses - messages already saved in _yield_tool_call
                # Use JSON parsing instead of substring matching to avoid false positives
                is_long_running_response = False
                try:
                    parsed = orjson.loads(result_content)
                    if isinstance(parsed, dict) and parsed.get("type") in (
                        "operation_started",
                        "operation_in_progress",
                    ):
                        is_long_running_response = True
                except (orjson.JSONDecodeError, TypeError):
                    pass  # Not JSON or not a dict - treat as regular response
                if is_long_running_response:
                    # Remove from accumulated_tool_calls since assistant message was already saved
                    accumulated_tool_calls[:] = [
                        tc
                        for tc in accumulated_tool_calls
                        if tc["id"] != chunk.toolCallId
                    ]
                    has_long_running_tool_call = True
                else:
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

                    # Save assistant message before yielding finish to ensure it's persisted
                    # even if client disconnects immediately after receiving StreamFinish
                    if not has_saved_assistant_message:
                        messages_to_save_early: list[ChatMessage] = []
                        if accumulated_tool_calls:
                            assistant_response.tool_calls = accumulated_tool_calls
                        if not has_appended_streaming_message and (
                            assistant_response.content or assistant_response.tool_calls
                        ):
                            messages_to_save_early.append(assistant_response)
                        messages_to_save_early.extend(tool_response_messages)

                        if messages_to_save_early:
                            session.messages.extend(messages_to_save_early)
                            logger.info(
                                f"Saving assistant message before StreamFinish: "
                                f"content_len={len(assistant_response.content or '')}, "
                                f"tool_calls={len(assistant_response.tool_calls or [])}, "
                                f"tool_responses={len(tool_response_messages)}"
                            )
                        if messages_to_save_early or has_appended_streaming_message:
                            await upsert_chat_session(session)
                            has_saved_assistant_message = True

                    has_yielded_end = True
                    yield chunk
            elif isinstance(chunk, StreamError):
                has_yielded_error = True
                yield chunk
            elif isinstance(chunk, StreamUsage):
                session.usage.append(
                    Usage(
                        prompt_tokens=chunk.promptTokens,
                        completion_tokens=chunk.completionTokens,
                        total_tokens=chunk.totalTokens,
                    )
                )
            else:
                logger.error(f"Unknown chunk type: {type(chunk)}", exc_info=True)

    except CancelledError:
        if not has_saved_assistant_message:
            if accumulated_tool_calls:
                assistant_response.tool_calls = accumulated_tool_calls
            if assistant_response.content:
                assistant_response.content = (
                    f"{assistant_response.content}\n\n[interrupted]"
                )
            else:
                assistant_response.content = "[interrupted]"
            if not has_appended_streaming_message:
                session.messages.append(assistant_response)
            if tool_response_messages:
                session.messages.extend(tool_response_messages)
            try:
                await upsert_chat_session(session)
            except Exception as e:
                logger.warning(
                    f"Failed to save interrupted session {session.session_id}: {e}"
                )
        raise
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
            if not has_appended_streaming_message and (
                assistant_response.content or assistant_response.tool_calls
            ):
                messages_to_save.append(assistant_response)

            # Add tool response messages after assistant message
            messages_to_save.extend(tool_response_messages)

            if not has_saved_assistant_message:
                if messages_to_save:
                    session.messages.extend(messages_to_save)
                if messages_to_save or has_appended_streaming_message:
                    await upsert_chat_session(session)

            if not has_yielded_error:
                error_message = str(e)
                if not is_retryable:
                    error_message = f"Non-retryable error: {error_message}"
                elif retry_count >= config.max_retries:
                    error_message = (
                        f"Max retries ({config.max_retries}) exceeded: {error_message}"
                    )

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
    # Only save if we haven't already saved when StreamFinish was received
    if not has_saved_assistant_message:
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
        if not has_appended_streaming_message and (
            assistant_response.content or assistant_response.tool_calls
        ):
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

        if messages_to_save:
            session.messages.extend(messages_to_save)
            logger.info(
                f"Extended session messages, new message_count={len(session.messages)}"
            )
        # Save if there are regular (non-long-running) tool responses or streaming message.
        # Long-running tools save their own state, but we still need to save regular tools
        # that may be in the same response.
        has_regular_tool_responses = len(tool_response_messages) > 0
        if has_regular_tool_responses or (
            not has_long_running_tool_call
            and (messages_to_save or has_appended_streaming_message)
        ):
            await upsert_chat_session(session)
    else:
        logger.info(
            "Assistant message already saved when StreamFinish was received, "
            "skipping duplicate save"
        )

    # If we did a tool call, stream the chat completion again to get the next response
    # Skip only if ALL tools were long-running (they handle their own completion)
    has_regular_tools = len(tool_response_messages) > 0
    if has_done_tool_call and (has_regular_tools or not has_long_running_tool_call):
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


def _is_region_blocked_error(error: Exception) -> bool:
    if isinstance(error, PermissionDeniedError):
        return "not available in your region" in str(error).lower()
    return "not available in your region" in str(error).lower()


async def _summarize_messages(
    messages: list,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> str:
    """Summarize a list of messages into concise context.

    Uses the same model as the chat for higher quality summaries.

    Args:
        messages: List of message dicts to summarize
        model: Model to use for summarization (same as chat model)
        api_key: API key for OpenAI client
        base_url: Base URL for OpenAI client
        timeout: Request timeout in seconds (default: 30.0)

    Returns:
        Summarized text
    """
    # Format messages for summarization
    conversation = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        # Include user, assistant, and tool messages (tool outputs are important context)
        if content and role in ("user", "assistant", "tool"):
            conversation.append(f"{role.upper()}: {content}")

    conversation_text = "\n\n".join(conversation)

    # Handle empty conversation
    if not conversation_text:
        return "No conversation history available."

    # Truncate conversation to fit within summarization model's context
    # gpt-4o-mini has 128k context, but we limit to ~25k tokens (~100k chars) for safety
    MAX_CHARS = 100_000
    if len(conversation_text) > MAX_CHARS:
        conversation_text = conversation_text[:MAX_CHARS] + "\n\n[truncated]"

    # Call LLM to summarize
    import openai

    summarization_client = openai.AsyncOpenAI(
        api_key=api_key, base_url=base_url, timeout=timeout
    )

    response = await summarization_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Create a detailed summary of the conversation so far. "
                    "This summary will be used as context when continuing the conversation.\n\n"
                    "Before writing the summary, analyze each message chronologically to identify:\n"
                    "- User requests and their explicit goals\n"
                    "- Your approach and key decisions made\n"
                    "- Technical specifics (file names, tool outputs, function signatures)\n"
                    "- Errors encountered and resolutions applied\n\n"
                    "You MUST include ALL of the following sections:\n\n"
                    "## 1. Primary Request and Intent\n"
                    "The user's explicit goals and what they are trying to accomplish.\n\n"
                    "## 2. Key Technical Concepts\n"
                    "Technologies, frameworks, tools, and patterns being used or discussed.\n\n"
                    "## 3. Files and Resources Involved\n"
                    "Specific files examined or modified, with relevant snippets and identifiers.\n\n"
                    "## 4. Errors and Fixes\n"
                    "Problems encountered, error messages, and their resolutions. "
                    "Include any user feedback on fixes.\n\n"
                    "## 5. Problem Solving\n"
                    "Issues that have been resolved and how they were addressed.\n\n"
                    "## 6. All User Messages\n"
                    "A complete list of all user inputs (excluding tool outputs) to preserve their exact requests.\n\n"
                    "## 7. Pending Tasks\n"
                    "Work items the user explicitly requested that have not yet been completed.\n\n"
                    "## 8. Current Work\n"
                    "Precise description of what was being worked on most recently, including relevant context.\n\n"
                    "## 9. Next Steps\n"
                    "What should happen next, aligned with the user's most recent requests. "
                    "Include verbatim quotes of recent instructions if relevant."
                ),
            },
            {"role": "user", "content": f"Summarize:\n\n{conversation_text}"},
        ],
        max_tokens=1500,
        temperature=0.3,
    )

    summary = response.choices[0].message.content
    return summary or "No summary available."


def _ensure_tool_pairs_intact(
    recent_messages: list[dict],
    all_messages: list[dict],
    start_index: int,
) -> list[dict]:
    """
    Ensure tool_call/tool_response pairs stay together after slicing.

    When slicing messages for context compaction, a naive slice can separate
    an assistant message containing tool_calls from its corresponding tool
    response messages. This causes API validation errors (e.g., Anthropic's
    "unexpected tool_use_id found in tool_result blocks").

    This function checks for orphan tool responses in the slice and extends
    backwards to include their corresponding assistant messages.

    Args:
        recent_messages: The sliced messages to validate
        all_messages: The complete message list (for looking up missing assistants)
        start_index: The index in all_messages where recent_messages begins

    Returns:
        A potentially extended list of messages with tool pairs intact
    """
    if not recent_messages:
        return recent_messages

    # Collect all tool_call_ids from assistant messages in the slice
    available_tool_call_ids: set[str] = set()
    for msg in recent_messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tc_id = tc.get("id")
                if tc_id:
                    available_tool_call_ids.add(tc_id)

    # Find orphan tool responses (tool messages whose tool_call_id is missing)
    orphan_tool_call_ids: set[str] = set()
    for msg in recent_messages:
        if msg.get("role") == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id and tc_id not in available_tool_call_ids:
                orphan_tool_call_ids.add(tc_id)

    if not orphan_tool_call_ids:
        # No orphans, slice is valid
        return recent_messages

    # Find the assistant messages that contain the orphan tool_call_ids
    # Search backwards from start_index in all_messages
    messages_to_prepend: list[dict] = []
    for i in range(start_index - 1, -1, -1):
        msg = all_messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            msg_tool_ids = {tc.get("id") for tc in msg["tool_calls"] if tc.get("id")}
            if msg_tool_ids & orphan_tool_call_ids:
                # This assistant message has tool_calls we need
                # Also collect its contiguous tool responses that follow it
                assistant_and_responses: list[dict] = [msg]

                # Scan forward from this assistant to collect tool responses
                for j in range(i + 1, start_index):
                    following_msg = all_messages[j]
                    if following_msg.get("role") == "tool":
                        tool_id = following_msg.get("tool_call_id")
                        if tool_id and tool_id in msg_tool_ids:
                            assistant_and_responses.append(following_msg)
                    else:
                        # Stop at first non-tool message
                        break

                # Prepend the assistant and its tool responses (maintain order)
                messages_to_prepend = assistant_and_responses + messages_to_prepend
                # Mark these as found
                orphan_tool_call_ids -= msg_tool_ids
                # Also add this assistant's tool_call_ids to available set
                available_tool_call_ids |= msg_tool_ids

        if not orphan_tool_call_ids:
            # Found all missing assistants
            break

    if orphan_tool_call_ids:
        # Some tool_call_ids couldn't be resolved - remove those tool responses
        # This shouldn't happen in normal operation but handles edge cases
        logger.warning(
            f"Could not find assistant messages for tool_call_ids: {orphan_tool_call_ids}. "
            "Removing orphan tool responses."
        )
        recent_messages = [
            msg
            for msg in recent_messages
            if not (
                msg.get("role") == "tool"
                and msg.get("tool_call_id") in orphan_tool_call_ids
            )
        ]

    if messages_to_prepend:
        logger.info(
            f"Extended recent messages by {len(messages_to_prepend)} to preserve "
            f"tool_call/tool_response pairs"
        )
        return messages_to_prepend + recent_messages

    return recent_messages


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

    # Apply context window management
    token_count = 0  # Initialize for exception handler
    try:
        from backend.util.prompt import estimate_token_count

        # Convert to dict for token counting
        # OpenAI message types are TypedDicts, so they're already dict-like
        messages_dict = []
        for msg in messages:
            # TypedDict objects are already dicts, just filter None values
            if isinstance(msg, dict):
                msg_dict = {k: v for k, v in msg.items() if v is not None}
            else:
                # Fallback for unexpected types
                msg_dict = dict(msg)
            messages_dict.append(msg_dict)

        # Estimate tokens using appropriate tokenizer
        # Normalize model name for token counting (tiktoken only supports OpenAI models)
        token_count_model = model
        if "/" in model:
            # Strip provider prefix (e.g., "anthropic/claude-opus-4.5" -> "claude-opus-4.5")
            token_count_model = model.split("/")[-1]

        # For Claude and other non-OpenAI models, approximate with gpt-4o tokenizer
        # Most modern LLMs have similar tokenization (~1 token per 4 chars)
        if "claude" in token_count_model.lower() or not any(
            known in token_count_model.lower()
            for known in ["gpt", "o1", "chatgpt", "text-"]
        ):
            token_count_model = "gpt-4o"

        # Attempt token counting with error handling
        try:
            token_count = estimate_token_count(messages_dict, model=token_count_model)
        except Exception as token_error:
            # If token counting fails, use gpt-4o as fallback approximation
            logger.warning(
                f"Token counting failed for model {token_count_model}: {token_error}. "
                "Using gpt-4o approximation."
            )
            token_count = estimate_token_count(messages_dict, model="gpt-4o")

        # If over threshold, summarize old messages
        if token_count > 120_000:
            KEEP_RECENT = 15

            # Check if we have a system prompt at the start
            has_system_prompt = (
                len(messages) > 0 and messages[0].get("role") == "system"
            )

            # Always attempt mitigation when over limit, even with few messages
            if messages:
                # Split messages based on whether system prompt exists
                # Calculate start index for the slice
                slice_start = max(0, len(messages_dict) - KEEP_RECENT)
                recent_messages = messages_dict[-KEEP_RECENT:]

                # Ensure tool_call/tool_response pairs stay together
                # This prevents API errors from orphan tool responses
                recent_messages = _ensure_tool_pairs_intact(
                    recent_messages, messages_dict, slice_start
                )

                if has_system_prompt:
                    # Keep system prompt separate, summarize everything between system and recent
                    system_msg = messages[0]
                    old_messages_dict = messages_dict[1:-KEEP_RECENT]
                else:
                    # No system prompt, summarize everything except recent
                    system_msg = None
                    old_messages_dict = messages_dict[:-KEEP_RECENT]

                # Summarize any non-empty old messages (no minimum threshold)
                # If we're over the token limit, we need to compress whatever we can
                if old_messages_dict:
                    # Summarize old messages using the same model as chat
                    summary_text = await _summarize_messages(
                        old_messages_dict,
                        model=model,
                        api_key=config.api_key,
                        base_url=config.base_url,
                    )

                    # Build new message list
                    # Use assistant role (not system) to prevent privilege escalation
                    # of user-influenced content to instruction-level authority
                    from openai.types.chat import ChatCompletionAssistantMessageParam

                    summary_msg = ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=(
                            "[Previous conversation summary — for context only]: "
                            f"{summary_text}"
                        ),
                    )

                    # Rebuild messages based on whether we have a system prompt
                    if has_system_prompt:
                        # system_prompt + summary + recent_messages
                        messages = [system_msg, summary_msg] + recent_messages
                    else:
                        # summary + recent_messages (no original system prompt)
                        messages = [summary_msg] + recent_messages

                    logger.info(
                        f"Context summarized: {token_count} tokens, "
                        f"summarized {len(old_messages_dict)} old messages, "
                        f"kept last {KEEP_RECENT} messages"
                    )

                    # Fallback: If still over limit after summarization, progressively drop recent messages
                    # This handles edge cases where recent messages are extremely large
                    new_messages_dict = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            msg_dict = {k: v for k, v in msg.items() if v is not None}
                        else:
                            msg_dict = dict(msg)
                        new_messages_dict.append(msg_dict)

                    new_token_count = estimate_token_count(
                        new_messages_dict, model=token_count_model
                    )

                    if new_token_count > 120_000:
                        # Still over limit - progressively reduce KEEP_RECENT
                        logger.warning(
                            f"Still over limit after summarization: {new_token_count} tokens. "
                            "Reducing number of recent messages kept."
                        )

                        for keep_count in [12, 10, 8, 5, 3, 2, 1, 0]:
                            if keep_count == 0:
                                # Try with just system prompt + summary (no recent messages)
                                if has_system_prompt:
                                    messages = [system_msg, summary_msg]
                                else:
                                    messages = [summary_msg]
                                logger.info(
                                    "Trying with 0 recent messages (system + summary only)"
                                )
                            else:
                                # Slice from ORIGINAL recent_messages to avoid duplicating summary
                                reduced_recent = (
                                    recent_messages[-keep_count:]
                                    if len(recent_messages) >= keep_count
                                    else recent_messages
                                )
                                # Ensure tool pairs stay intact in the reduced slice
                                # Note: Search in messages_dict (full conversation) not recent_messages
                                # (already sliced), so we can find assistants outside the current slice.
                                # Calculate where reduced_recent starts in messages_dict
                                reduced_start_in_dict = slice_start + max(
                                    0, len(recent_messages) - keep_count
                                )
                                reduced_recent = _ensure_tool_pairs_intact(
                                    reduced_recent, messages_dict, reduced_start_in_dict
                                )
                                if has_system_prompt:
                                    messages = [
                                        system_msg,
                                        summary_msg,
                                    ] + reduced_recent
                                else:
                                    messages = [summary_msg] + reduced_recent

                            new_messages_dict = []
                            for msg in messages:
                                if isinstance(msg, dict):
                                    msg_dict = {
                                        k: v for k, v in msg.items() if v is not None
                                    }
                                else:
                                    msg_dict = dict(msg)
                                new_messages_dict.append(msg_dict)

                            new_token_count = estimate_token_count(
                                new_messages_dict, model=token_count_model
                            )

                            if new_token_count <= 120_000:
                                logger.info(
                                    f"Reduced to {keep_count} recent messages, "
                                    f"now {new_token_count} tokens"
                                )
                                break
                        else:
                            logger.error(
                                f"Unable to reduce token count below threshold even with 0 messages. "
                                f"Final count: {new_token_count} tokens"
                            )
                            # ABSOLUTE LAST RESORT: Drop system prompt
                            # This should only happen if summary itself is massive
                            if has_system_prompt and len(messages) > 1:
                                messages = messages[1:]  # Drop system prompt
                                logger.critical(
                                    "CRITICAL: Dropped system prompt as absolute last resort. "
                                    "Behavioral consistency may be affected."
                                )
                                # Yield error to user
                                yield StreamError(
                                    errorText=(
                                        "Warning: System prompt dropped due to size constraints. "
                                        "Assistant behavior may be affected."
                                    )
                                )
                else:
                    # No old messages to summarize - all messages are "recent"
                    # Apply progressive truncation to reduce token count
                    logger.warning(
                        f"Token count {token_count} exceeds threshold but no old messages to summarize. "
                        f"Applying progressive truncation to recent messages."
                    )

                    # Create a base list excluding system prompt to avoid duplication
                    # This is the pool of messages we'll slice from in the loop
                    # Use messages_dict for type consistency with _ensure_tool_pairs_intact
                    base_msgs = (
                        messages_dict[1:] if has_system_prompt else messages_dict
                    )

                    # Try progressively smaller keep counts
                    new_token_count = token_count  # Initialize with current count
                    for keep_count in [12, 10, 8, 5, 3, 2, 1, 0]:
                        if keep_count == 0:
                            # Try with just system prompt (no recent messages)
                            if has_system_prompt:
                                messages = [system_msg]
                                logger.info(
                                    "Trying with 0 recent messages (system prompt only)"
                                )
                            else:
                                # No system prompt and no recent messages = empty messages list
                                # This is invalid, skip this iteration
                                continue
                        else:
                            if len(base_msgs) < keep_count:
                                continue  # Skip if we don't have enough messages

                            # Slice from base_msgs to get recent messages (without system prompt)
                            recent_messages = base_msgs[-keep_count:]

                            # Ensure tool pairs stay intact in the reduced slice
                            reduced_slice_start = max(0, len(base_msgs) - keep_count)
                            recent_messages = _ensure_tool_pairs_intact(
                                recent_messages, base_msgs, reduced_slice_start
                            )

                            if has_system_prompt:
                                messages = [system_msg] + recent_messages
                            else:
                                messages = recent_messages

                        new_messages_dict = []
                        for msg in messages:
                            if msg is None:
                                continue  # Skip None messages (type safety)
                            if isinstance(msg, dict):
                                msg_dict = {
                                    k: v for k, v in msg.items() if v is not None
                                }
                            else:
                                msg_dict = dict(msg)
                            new_messages_dict.append(msg_dict)

                        new_token_count = estimate_token_count(
                            new_messages_dict, model=token_count_model
                        )

                        if new_token_count <= 120_000:
                            logger.info(
                                f"Reduced to {keep_count} recent messages, "
                                f"now {new_token_count} tokens"
                            )
                            break
                    else:
                        # Even with 0 messages still over limit
                        logger.error(
                            f"Unable to reduce token count below threshold even with 0 messages. "
                            f"Final count: {new_token_count} tokens. Messages may be extremely large."
                        )
                        # ABSOLUTE LAST RESORT: Drop system prompt
                        if has_system_prompt and len(messages) > 1:
                            messages = messages[1:]  # Drop system prompt
                            logger.critical(
                                "CRITICAL: Dropped system prompt as absolute last resort. "
                                "Behavioral consistency may be affected."
                            )
                            # Yield error to user
                            yield StreamError(
                                errorText=(
                                    "Warning: System prompt dropped due to size constraints. "
                                    "Assistant behavior may be affected."
                                )
                            )

    except Exception as e:
        logger.error(f"Context summarization failed: {e}", exc_info=True)
        # If we were over the token limit, yield error to user
        # Don't silently continue with oversized messages that will fail
        if token_count > 120_000:
            yield StreamError(
                errorText=(
                    f"Unable to manage context window (token limit exceeded: {token_count} tokens). "
                    "Context summarization failed. Please start a new conversation."
                )
            )
            yield StreamFinish()
            return
        # Otherwise, continue with original messages (under limit)

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

                # Build extra_body for OpenRouter tracing and PostHog analytics
                extra_body: dict[str, Any] = {
                    "posthogProperties": {
                        "environment": settings.config.app_env.value,
                    },
                }
                if session.user_id:
                    extra_body["user"] = session.user_id[:128]  # OpenRouter limit
                    extra_body["posthogDistinctId"] = session.user_id
                if session.session_id:
                    extra_body["session_id"] = session.session_id[
                        :128
                    ]  # OpenRouter limit

                # Create the stream with proper types
                from typing import cast

                from openai.types.chat import (
                    ChatCompletionMessageParam,
                    ChatCompletionStreamOptionsParam,
                )

                stream = await client.chat.completions.create(
                    model=model,
                    messages=cast(list[ChatCompletionMessageParam], messages),
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                    stream_options=ChatCompletionStreamOptionsParam(include_usage=True),
                    extra_body=extra_body,
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
                    error_code = None
                    error_text = str(e)
                    if _is_region_blocked_error(e):
                        error_code = "MODEL_NOT_AVAILABLE_REGION"
                        error_text = (
                            "This model is not available in your region. "
                            "Please connect via VPN and try again."
                        )
                    error_response = StreamError(
                        errorText=error_text,
                        code=error_code,
                    )
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

    For tools marked with `is_long_running=True` (like agent generation), spawns a
    background task so the operation survives SSE disconnections. For other tools,
    yields heartbeat events every 15 seconds to keep the SSE connection alive.

    Raises:
        orjson.JSONDecodeError: If tool call arguments cannot be parsed as JSON
        KeyError: If expected tool call fields are missing
        TypeError: If tool call structure is invalid
    """
    import uuid as uuid_module

    tool_name = tool_calls[yield_idx]["function"]["name"]
    tool_call_id = tool_calls[yield_idx]["id"]

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

    # Check if this tool is long-running (survives SSE disconnection)
    tool = get_tool(tool_name)
    if tool and tool.is_long_running:
        # Atomic check-and-set: returns False if operation already running (lost race)
        if not await _mark_operation_started(tool_call_id):
            logger.info(
                f"Tool call {tool_call_id} already in progress, returning status"
            )
            # Build dynamic message based on tool name
            if tool_name == "create_agent":
                in_progress_msg = "Agent creation already in progress. Please wait..."
            elif tool_name == "edit_agent":
                in_progress_msg = "Agent edit already in progress. Please wait..."
            else:
                in_progress_msg = f"{tool_name} already in progress. Please wait..."

            yield StreamToolOutputAvailable(
                toolCallId=tool_call_id,
                toolName=tool_name,
                output=OperationInProgressResponse(
                    message=in_progress_msg,
                    tool_call_id=tool_call_id,
                ).model_dump_json(),
                success=True,
            )
            return

        # Generate operation ID
        operation_id = str(uuid_module.uuid4())

        # Build a user-friendly message based on tool and arguments
        if tool_name == "create_agent":
            agent_desc = arguments.get("description", "")
            # Truncate long descriptions for the message
            desc_preview = (
                (agent_desc[:100] + "...") if len(agent_desc) > 100 else agent_desc
            )
            pending_msg = (
                f"Creating your agent: {desc_preview}"
                if desc_preview
                else "Creating agent... This may take a few minutes."
            )
            started_msg = (
                "Agent creation started. You can close this tab - "
                "check your library in a few minutes."
            )
        elif tool_name == "edit_agent":
            changes = arguments.get("changes", "")
            changes_preview = (changes[:100] + "...") if len(changes) > 100 else changes
            pending_msg = (
                f"Editing agent: {changes_preview}"
                if changes_preview
                else "Editing agent... This may take a few minutes."
            )
            started_msg = (
                "Agent edit started. You can close this tab - "
                "check your library in a few minutes."
            )
        else:
            pending_msg = f"Running {tool_name}... This may take a few minutes."
            started_msg = (
                f"{tool_name} started. You can close this tab - "
                "check back in a few minutes."
            )

        # Track appended messages for rollback on failure
        assistant_message: ChatMessage | None = None
        pending_message: ChatMessage | None = None

        # Wrap session save and task creation in try-except to release lock on failure
        try:
            # Save assistant message with tool_call FIRST (required by LLM)
            assistant_message = ChatMessage(
                role="assistant",
                content="",
                tool_calls=[tool_calls[yield_idx]],
            )
            session.messages.append(assistant_message)

            # Then save pending tool result
            pending_message = ChatMessage(
                role="tool",
                content=OperationPendingResponse(
                    message=pending_msg,
                    operation_id=operation_id,
                    tool_name=tool_name,
                ).model_dump_json(),
                tool_call_id=tool_call_id,
            )
            session.messages.append(pending_message)
            await upsert_chat_session(session)
            logger.info(
                f"Saved pending operation {operation_id} for tool {tool_name} "
                f"in session {session.session_id}"
            )

            # Store task reference in module-level set to prevent GC before completion
            task = asyncio.create_task(
                _execute_long_running_tool(
                    tool_name=tool_name,
                    parameters=arguments,
                    tool_call_id=tool_call_id,
                    operation_id=operation_id,
                    session_id=session.session_id,
                    user_id=session.user_id,
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
        except Exception as e:
            # Roll back appended messages to prevent data corruption on subsequent saves
            if (
                pending_message
                and session.messages
                and session.messages[-1] == pending_message
            ):
                session.messages.pop()
            if (
                assistant_message
                and session.messages
                and session.messages[-1] == assistant_message
            ):
                session.messages.pop()

            # Release the Redis lock since the background task won't be spawned
            await _mark_operation_completed(tool_call_id)
            logger.error(
                f"Failed to setup long-running tool {tool_name}: {e}", exc_info=True
            )
            raise

        # Return immediately - don't wait for completion
        yield StreamToolOutputAvailable(
            toolCallId=tool_call_id,
            toolName=tool_name,
            output=OperationStartedResponse(
                message=started_msg,
                operation_id=operation_id,
                tool_name=tool_name,
            ).model_dump_json(),
            success=True,
        )
        return

    # Normal flow: Run tool execution in background task with heartbeats
    tool_task = asyncio.create_task(
        execute_tool(
            tool_name=tool_name,
            parameters=arguments,
            tool_call_id=tool_call_id,
            user_id=session.user_id,
            session=session,
        )
    )

    # Yield heartbeats every 15 seconds while waiting for tool to complete
    heartbeat_interval = 15.0  # seconds
    while not tool_task.done():
        try:
            # Wait for either the task to complete or the heartbeat interval
            await asyncio.wait_for(
                asyncio.shield(tool_task), timeout=heartbeat_interval
            )
        except asyncio.TimeoutError:
            # Task still running, send heartbeat to keep connection alive
            logger.debug(f"Sending heartbeat for tool {tool_name} ({tool_call_id})")
            yield StreamHeartbeat(toolCallId=tool_call_id)
        except CancelledError:
            # Task was cancelled, clean up and propagate
            tool_task.cancel()
            logger.warning(f"Tool execution cancelled: {tool_name} ({tool_call_id})")
            raise

    # Get the result - handle any exceptions that occurred during execution
    try:
        tool_execution_response: StreamToolOutputAvailable = await tool_task
    except Exception as e:
        # Task raised an exception - ensure we send an error response to the frontend
        logger.error(
            f"Tool execution failed: {tool_name} ({tool_call_id}): {e}", exc_info=True
        )
        error_response = ErrorResponse(
            message=f"Tool execution failed: {e!s}",
            error=type(e).__name__,
            session_id=session.session_id,
        )
        tool_execution_response = StreamToolOutputAvailable(
            toolCallId=tool_call_id,
            toolName=tool_name,
            output=error_response.model_dump_json(),
            success=False,
        )

    yield tool_execution_response


async def _execute_long_running_tool(
    tool_name: str,
    parameters: dict[str, Any],
    tool_call_id: str,
    operation_id: str,
    session_id: str,
    user_id: str | None,
) -> None:
    """Execute a long-running tool in background and update chat history with result.

    This function runs independently of the SSE connection, so the operation
    survives if the user closes their browser tab.
    """
    try:
        # Load fresh session (not stale reference)
        session = await get_chat_session(session_id, user_id)
        if not session:
            logger.error(f"Session {session_id} not found for background tool")
            return

        # Execute the actual tool
        result = await execute_tool(
            tool_name=tool_name,
            parameters=parameters,
            tool_call_id=tool_call_id,
            user_id=user_id,
            session=session,
        )

        # Update the pending message with result
        await _update_pending_operation(
            session_id=session_id,
            tool_call_id=tool_call_id,
            result=(
                result.output
                if isinstance(result.output, str)
                else orjson.dumps(result.output).decode("utf-8")
            ),
        )

        logger.info(f"Background tool {tool_name} completed for session {session_id}")

        # Generate LLM continuation so user sees response when they poll/refresh
        await _generate_llm_continuation(session_id=session_id, user_id=user_id)

    except Exception as e:
        logger.error(f"Background tool {tool_name} failed: {e}", exc_info=True)
        error_response = ErrorResponse(
            message=f"Tool {tool_name} failed: {str(e)}",
        )
        await _update_pending_operation(
            session_id=session_id,
            tool_call_id=tool_call_id,
            result=error_response.model_dump_json(),
        )
    finally:
        await _mark_operation_completed(tool_call_id)


async def _update_pending_operation(
    session_id: str,
    tool_call_id: str,
    result: str,
) -> None:
    """Update the pending tool message with final result.

    This is called by background tasks when long-running operations complete.
    """
    # Update the message in database
    updated = await chat_db.update_tool_message_content(
        session_id=session_id,
        tool_call_id=tool_call_id,
        new_content=result,
    )

    if updated:
        # Invalidate Redis cache so next load gets fresh data
        # Wrap in try/except to prevent cache failures from triggering error handling
        # that would overwrite our successful DB update
        try:
            await invalidate_session_cache(session_id)
        except Exception as e:
            # Non-critical: cache will eventually be refreshed on next load
            logger.warning(f"Failed to invalidate cache for session {session_id}: {e}")
        logger.info(
            f"Updated pending operation for tool_call_id {tool_call_id} "
            f"in session {session_id}"
        )
    else:
        logger.warning(
            f"Failed to update pending operation for tool_call_id {tool_call_id} "
            f"in session {session_id}"
        )


async def _generate_llm_continuation(
    session_id: str,
    user_id: str | None,
) -> None:
    """Generate an LLM response after a long-running tool completes.

    This is called by background tasks to continue the conversation
    after a tool result is saved. The response is saved to the database
    so users see it when they refresh or poll.
    """
    try:
        # Load fresh session from DB (bypass cache to get the updated tool result)
        await invalidate_session_cache(session_id)
        session = await get_chat_session(session_id, user_id)
        if not session:
            logger.error(f"Session {session_id} not found for LLM continuation")
            return

        # Build system prompt
        system_prompt, _ = await _build_system_prompt(user_id)

        # Build messages in OpenAI format
        messages = session.to_openai_messages()
        if system_prompt:
            from openai.types.chat import ChatCompletionSystemMessageParam

            system_message = ChatCompletionSystemMessageParam(
                role="system",
                content=system_prompt,
            )
            messages = [system_message] + messages

        # Build extra_body for tracing
        extra_body: dict[str, Any] = {
            "posthogProperties": {
                "environment": settings.config.app_env.value,
            },
        }
        if user_id:
            extra_body["user"] = user_id[:128]
            extra_body["posthogDistinctId"] = user_id
        if session_id:
            extra_body["session_id"] = session_id[:128]

        # Make non-streaming LLM call (no tools - just text response)
        from typing import cast

        from openai.types.chat import ChatCompletionMessageParam

        # No tools parameter = text-only response (no tool calls)
        response = await client.chat.completions.create(
            model=config.model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            extra_body=extra_body,
        )

        if response.choices and response.choices[0].message.content:
            assistant_content = response.choices[0].message.content

            # Reload session from DB to avoid race condition with user messages
            # that may have been sent while we were generating the LLM response
            fresh_session = await get_chat_session(session_id, user_id)
            if not fresh_session:
                logger.error(
                    f"Session {session_id} disappeared during LLM continuation"
                )
                return

            # Save assistant message to database
            assistant_message = ChatMessage(
                role="assistant",
                content=assistant_content,
            )
            fresh_session.messages.append(assistant_message)

            # Save to database (not cache) to persist the response
            await upsert_chat_session(fresh_session)

            # Invalidate cache so next poll/refresh gets fresh data
            await invalidate_session_cache(session_id)

            logger.info(
                f"Generated LLM continuation for session {session_id}, "
                f"response length: {len(assistant_content)}"
            )
        else:
            logger.warning(f"LLM continuation returned empty response for {session_id}")

    except Exception as e:
        logger.error(f"Failed to generate LLM continuation: {e}", exc_info=True)
