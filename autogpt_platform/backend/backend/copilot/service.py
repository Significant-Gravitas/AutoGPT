import asyncio
import logging
import time
from asyncio import CancelledError
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

import openai

if TYPE_CHECKING:
    from backend.util.prompt import CompressResult

import orjson
from langfuse import get_client
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    PermissionDeniedError,
    RateLimitError,
)
from openai.types.chat import (
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)

from backend.data.db_accessors import understanding_db
from backend.data.understanding import format_understanding_for_prompt
from backend.util.exceptions import NotFoundError
from backend.util.settings import AppEnvironment, Settings

from .config import ChatConfig
from .model import (
    ChatMessage,
    ChatSession,
    ChatSessionInfo,
    Usage,
    cache_chat_session,
    get_chat_session,
    update_session_title,
    upsert_chat_session,
)
from .response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamFinishStep,
    StreamHeartbeat,
    StreamStart,
    StreamStartStep,
    StreamTextDelta,
    StreamTextEnd,
    StreamTextStart,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
    StreamUsage,
)
from .tools import execute_tool, tools
from .tools.models import ErrorResponse
from .tracking import track_user_message

logger = logging.getLogger(__name__)

# Set to hold background tasks to prevent garbage collection
_background_tasks: set[asyncio.Task[Any]] = set()

config = ChatConfig()
settings = Settings()
client = openai.AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)


langfuse = get_client()

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
   - **When `create_agent` returns `suggested_goal`**: Present the suggestion to the user and ask "Would you like me to proceed with this refined goal?" If they accept, call `create_agent` again with the suggested goal.
   - **When `create_agent` returns `clarifying_questions`**: After the user answers, call `create_agent` again with the original description AND the answers in the `context` parameter.

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

**Handle Feedback Loops:**
- When a tool returns a suggested alternative (like a refined goal), present it clearly and ask the user for confirmation before proceeding
- When clarifying questions are answered, immediately re-call the tool with the accumulated context
- Don't ask redundant questions if the user has already provided context in the conversation

## CRITICAL REMINDER

You are NOT a chatbot. You are NOT documentation. You are a partner who helps busy business owners get value quickly by showing proof through working automations. Bias toward action over explanation."""


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
            # In non-production environments, fetch the latest prompt version
            # instead of the production-labeled version for easier testing
            label = (
                None
                if settings.config.app_env == AppEnvironment.PRODUCTION
                else "latest"
            )
            prompt = await asyncio.to_thread(
                langfuse.get_prompt,
                config.langfuse_prompt_name,
                label=label,
                cache_ttl_seconds=0,
            )
            return prompt.compile(users_information=context)
        except Exception as e:
            logger.warning(f"Failed to fetch prompt from Langfuse, using default: {e}")

    # Fallback to default prompt
    return DEFAULT_SYSTEM_PROMPT.format(users_information=context)


async def _build_system_prompt(
    user_id: str | None, has_conversation_history: bool = False
) -> tuple[str, Any]:
    """Build the full system prompt including business understanding if available.

    Args:
        user_id: The user ID for fetching business understanding.
        has_conversation_history: Whether there's existing conversation history.
            If True, we don't tell the model to greet/introduce (since they're
            already in a conversation).

    Returns:
        Tuple of (compiled prompt string, business understanding object)
    """
    # If user is authenticated, try to fetch their business understanding
    understanding = None
    if user_id:
        try:
            understanding = await understanding_db().get_business_understanding(user_id)
        except Exception as e:
            logger.warning(f"Failed to fetch business understanding: {e}")
            understanding = None

    if understanding:
        context = format_understanding_for_prompt(understanding)
    elif has_conversation_history:
        context = "No prior understanding saved yet. Continue the existing conversation naturally."
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
) -> ChatSessionInfo:
    """
    Assign a user to a chat session.
    """
    session = await get_chat_session(session_id, None)
    if not session:
        raise NotFoundError(f"Session {session_id} not found")
    session.user_id = user_id
    session = await upsert_chat_session(session)
    return session


async def stream_chat_completion(
    session_id: str,
    message: str | None = None,
    tool_call_response: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    retry_count: int = 0,
    session: ChatSession | None = None,
    context: dict[str, str] | None = None,  # {url: str, content: str}
    _continuation_message_id: (
        str | None
    ) = None,  # Internal: reuse message ID for tool call continuations
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

    """
    completion_start = time.monotonic()

    # Build log metadata for structured logging
    log_meta = {"component": "ChatService", "session_id": session_id}
    if user_id:
        log_meta["user_id"] = user_id

    logger.info(
        f"[TIMING] stream_chat_completion STARTED, session={session_id}, user={user_id}, "
        f"message_len={len(message) if message else 0}, is_user={is_user_message}",
        extra={
            "json_fields": {
                **log_meta,
                "message_len": len(message) if message else 0,
                "is_user_message": is_user_message,
            }
        },
    )

    # Only fetch from Redis if session not provided (initial call)
    if session is None:
        fetch_start = time.monotonic()
        session = await get_chat_session(session_id, user_id)
        fetch_time = (time.monotonic() - fetch_start) * 1000
        logger.info(
            f"[TIMING] get_chat_session took {fetch_time:.1f}ms, "
            f"n_messages={len(session.messages) if session else 0}",
            extra={
                "json_fields": {
                    **log_meta,
                    "duration_ms": fetch_time,
                    "n_messages": len(session.messages) if session else 0,
                }
            },
        )
    else:
        logger.info(
            f"[TIMING] Using provided session, messages={len(session.messages)}",
            extra={"json_fields": {**log_meta, "n_messages": len(session.messages)}},
        )

    if not session:
        raise NotFoundError(
            f"Session {session_id} not found. Please create a new session first."
        )

    # Append the new message to the session if it's not already there
    new_message_role = "user" if is_user_message else "assistant"
    if message and (
        len(session.messages) == 0
        or not (
            session.messages[-1].role == new_message_role
            and session.messages[-1].content == message
        )
    ):
        session.messages.append(ChatMessage(role=new_message_role, content=message))

    # Track user message in PostHog
    if is_user_message and message:
        posthog_start = time.monotonic()
        track_user_message(
            user_id=user_id,
            session_id=session_id,
            message_length=len(message),
        )
        posthog_time = (time.monotonic() - posthog_start) * 1000
        logger.info(
            f"[TIMING] track_user_message took {posthog_time:.1f}ms",
            extra={"json_fields": {**log_meta, "duration_ms": posthog_time}},
        )

    upsert_start = time.monotonic()
    session = await upsert_chat_session(session)
    upsert_time = (time.monotonic() - upsert_start) * 1000
    logger.info(
        f"[TIMING] upsert_chat_session took {upsert_time:.1f}ms",
        extra={"json_fields": {**log_meta, "duration_ms": upsert_time}},
    )
    assert session, "Session not found"

    # Generate title for new sessions on first user message (non-blocking)
    # Check: is_user_message, no title yet, and this is the first user message
    user_messages = [m for m in session.messages if m.role == "user"]
    first_user_msg = message or (user_messages[0].content if user_messages else None)
    if is_user_message and first_user_msg and not session.title:
        if len(user_messages) == 1:
            # First user message - generate title in background
            # Capture only the values we need (not the session object) to avoid
            # stale data issues when the main flow modifies the session
            captured_session_id = session_id
            captured_message = first_user_msg
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
            task = asyncio.create_task(_update_title())
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

    # Build system prompt with business understanding
    prompt_start = time.monotonic()
    system_prompt, understanding = await _build_system_prompt(user_id)
    prompt_time = (time.monotonic() - prompt_start) * 1000
    logger.info(
        f"[TIMING] _build_system_prompt took {prompt_time:.1f}ms",
        extra={"json_fields": {**log_meta, "duration_ms": prompt_time}},
    )

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
    has_received_text = False
    text_streaming_ended = False
    tool_response_messages: list[ChatMessage] = []
    should_retry = False

    # Generate unique IDs for AI SDK protocol
    import uuid as uuid_module

    is_continuation = _continuation_message_id is not None
    message_id = _continuation_message_id or str(uuid_module.uuid4())
    text_block_id = str(uuid_module.uuid4())

    # Only yield message start for the initial call, not for continuations.
    setup_time = (time.monotonic() - completion_start) * 1000
    logger.info(
        f"[TIMING] Setup complete, yielding StreamStart at {setup_time:.1f}ms",
        extra={"json_fields": {**log_meta, "setup_time_ms": setup_time}},
    )
    if not is_continuation:
        yield StreamStart(messageId=message_id, sessionId=session.session_id)

    # Emit start-step before each LLM call (AI SDK uses this to add step boundaries)
    yield StreamStartStep()

    try:
        logger.info(
            "[TIMING] Calling _stream_chat_chunks",
            extra={"json_fields": log_meta},
        )
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
                if has_done_tool_call:
                    # Tool calls happened — close the step but don't send message-level finish.
                    # The continuation will open a new step, and finish will come at the end.
                    yield StreamFinishStep()
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
                    # Emit finish-step (resets AI SDK text/reasoning state).
                    # StreamFinish is published by mark_session_completed.
                    yield StreamFinishStep()
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
            elif isinstance(chunk, StreamHeartbeat):
                # Pass through heartbeat to keep SSE connection alive
                yield chunk
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
            # Close the current step before retrying so the recursive call's
            # StreamStartStep doesn't produce unbalanced step events.
            if not has_yielded_end:
                yield StreamFinishStep()
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
                yield StreamFinishStep()
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
            _continuation_message_id=message_id,  # Reuse message ID since start was already sent
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

        # Add assistant message with tool_calls if any.
        # Use extend (not assign) to preserve tool_calls already added by
        # _yield_tool_call for long-running tools.
        if accumulated_tool_calls:
            if not assistant_response.tool_calls:
                assistant_response.tool_calls = []
            assistant_response.tool_calls.extend(accumulated_tool_calls)
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
        if messages_to_save or has_appended_streaming_message:
            await upsert_chat_session(session)
    else:
        logger.info(
            "Assistant message already saved when StreamFinish was received, "
            "skipping duplicate save"
        )

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
            _continuation_message_id=message_id,  # Reuse message ID to avoid duplicates
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


async def _manage_context_window(
    messages: list,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> "CompressResult":
    """
    Manage context window using the unified compress_context function.

    This is a thin wrapper that creates an OpenAI client for summarization
    and delegates to the shared compression logic in prompt.py.

    Args:
        messages: List of messages in OpenAI format
        model: Model name for token counting and summarization
        api_key: API key for summarization calls
        base_url: Base URL for summarization calls

    Returns:
        CompressResult with compacted messages and metadata
    """
    import openai

    from backend.util.prompt import compress_context

    # Convert messages to dict format
    messages_dict = []
    for msg in messages:
        if isinstance(msg, dict):
            msg_dict = {k: v for k, v in msg.items() if v is not None}
        else:
            msg_dict = dict(msg)
        messages_dict.append(msg_dict)

    # Only create client if api_key is provided (enables summarization)
    # Use context manager to avoid socket leaks
    if api_key:
        async with openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=30.0
        ) as client:
            return await compress_context(
                messages=messages_dict,
                model=model,
                client=client,
            )
    else:
        # No API key - use truncation-only mode
        return await compress_context(
            messages=messages_dict,
            model=model,
            client=None,
        )


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
    import time as time_module

    stream_chunks_start = time_module.perf_counter()
    model = config.model

    # Build log metadata for structured logging
    log_meta = {"component": "ChatService", "session_id": session.session_id}
    if session.user_id:
        log_meta["user_id"] = session.user_id

    logger.info(
        f"[TIMING] _stream_chat_chunks STARTED, session={session.session_id}, "
        f"user={session.user_id}, n_messages={len(session.messages)}",
        extra={"json_fields": {**log_meta, "n_messages": len(session.messages)}},
    )

    messages = session.to_openai_messages()
    if system_prompt:
        system_message = ChatCompletionSystemMessageParam(
            role="system",
            content=system_prompt,
        )
        messages = [system_message] + messages

    # Apply context window management
    context_start = time_module.perf_counter()
    context_result = await _manage_context_window(
        messages=messages,
        model=model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    context_time = (time_module.perf_counter() - context_start) * 1000
    logger.info(
        f"[TIMING] _manage_context_window took {context_time:.1f}ms",
        extra={"json_fields": {**log_meta, "duration_ms": context_time}},
    )

    if context_result.error:
        if "System prompt dropped" in context_result.error:
            # Warning only - continue with reduced context
            yield StreamError(
                errorText=(
                    "Warning: System prompt dropped due to size constraints. "
                    "Assistant behavior may be affected."
                )
            )
        else:
            # Any other error - abort to prevent failed LLM calls
            yield StreamError(
                errorText=(
                    f"Context window management failed: {context_result.error}. "
                    "Please start a new conversation."
                )
            )
            return

    messages = context_result.messages
    if context_result.was_compacted:
        logger.info(
            f"Context compacted for streaming: {context_result.token_count} tokens"
        )

    # Loop to handle tool calls and continue conversation
    while True:
        retry_count = 0
        last_error: Exception | None = None

        while retry_count <= MAX_RETRIES:
            try:
                elapsed = (time_module.perf_counter() - stream_chunks_start) * 1000
                retry_info = (
                    f" (retry {retry_count}/{MAX_RETRIES})" if retry_count > 0 else ""
                )
                logger.info(
                    f"[TIMING] Creating OpenAI stream at {elapsed:.1f}ms{retry_info}",
                    extra={
                        "json_fields": {
                            **log_meta,
                            "elapsed_ms": elapsed,
                            "retry_count": retry_count,
                        }
                    },
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

                # Enable adaptive thinking for Anthropic models via OpenRouter
                if config.thinking_enabled and "anthropic" in model.lower():
                    extra_body["reasoning"] = {"enabled": True}

                api_call_start = time_module.perf_counter()
                stream = await client.chat.completions.create(
                    model=model,
                    messages=cast(list[ChatCompletionMessageParam], messages),
                    tools=tools,
                    tool_choice="auto",
                    stream=True,
                    stream_options=ChatCompletionStreamOptionsParam(include_usage=True),
                    extra_body=extra_body,
                )
                api_init_time = (time_module.perf_counter() - api_call_start) * 1000
                logger.info(
                    f"[TIMING] OpenAI stream object returned in {api_init_time:.1f}ms",
                    extra={"json_fields": {**log_meta, "duration_ms": api_init_time}},
                )

                # Variables to accumulate tool calls
                tool_calls: list[dict[str, Any]] = []
                active_tool_call_idx: int | None = None
                finish_reason: str | None = None
                # Track which tool call indices have had their start event emitted
                emitted_start_for_idx: set[int] = set()

                # Track if we've started the text block
                text_started = False
                first_content_chunk = True
                chunk_count = 0

                # Process the stream
                chunk: ChatCompletionChunk
                async for chunk in stream:
                    chunk_count += 1
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
                            # Log timing for first content chunk
                            if first_content_chunk:
                                first_content_chunk = False
                                ttfc = (
                                    time_module.perf_counter() - api_call_start
                                ) * 1000
                                logger.info(
                                    f"[TIMING] FIRST CONTENT CHUNK at {ttfc:.1f}ms "
                                    f"(since API call), n_chunks={chunk_count}",
                                    extra={
                                        "json_fields": {
                                            **log_meta,
                                            "time_to_first_chunk_ms": ttfc,
                                            "n_chunks": chunk_count,
                                        }
                                    },
                                )
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
                stream_duration = time_module.perf_counter() - api_call_start
                logger.info(
                    f"[TIMING] OpenAI stream COMPLETE, finish_reason={finish_reason}, "
                    f"duration={stream_duration:.2f}s, "
                    f"n_chunks={chunk_count}, n_tool_calls={len(tool_calls)}",
                    extra={
                        "json_fields": {
                            **log_meta,
                            "stream_duration_ms": stream_duration * 1000,
                            "finish_reason": finish_reason,
                            "n_chunks": chunk_count,
                            "n_tool_calls": len(tool_calls),
                        }
                    },
                )

                # Execute all accumulated tool calls in parallel
                # Events are yielded as they arrive from each concurrent tool
                async for event in _execute_tool_calls_parallel(tool_calls, session):
                    yield event

                total_time = (time_module.perf_counter() - stream_chunks_start) * 1000
                logger.info(
                    f"[TIMING] _stream_chat_chunks COMPLETED in {total_time / 1000:.1f}s; "
                    f"session={session.session_id}, user={session.user_id}",
                    extra={"json_fields": {**log_meta, "total_time_ms": total_time}},
                )
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
                    _log_api_error(
                        error=e,
                        context="stream (not retrying)",
                        session_id=session.session_id if session else None,
                        message_count=len(messages) if messages else None,
                        model=model,
                        retry_count=retry_count,
                    )
                    error_code = None
                    error_text = str(e)

                    error_details = _extract_api_error_details(e)
                    if error_details.get("response_body"):
                        body = error_details["response_body"]
                        if isinstance(body, dict):
                            err = body.get("error")
                            if isinstance(err, dict) and err.get("message"):
                                error_text = err["message"]
                            elif body.get("message"):
                                error_text = body["message"]

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
                    return

        # If we exit the retry loop without returning, it means we exhausted retries
        if last_error:
            _log_api_error(
                error=last_error,
                context=f"stream (max retries {MAX_RETRIES} exceeded)",
                session_id=session.session_id if session else None,
                message_count=len(messages) if messages else None,
                model=model,
                retry_count=MAX_RETRIES,
            )
            yield StreamError(errorText=f"Max retries exceeded: {last_error!s}")
            return


async def _execute_tool_calls_parallel(
    tool_calls: list[dict[str, Any]],
    session: ChatSession,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Execute all tool calls concurrently, yielding stream events as they arrive.

    Each tool runs as an ``asyncio.Task``, pushing events into a shared queue.
    """
    queue: asyncio.Queue[StreamBaseResponse | None] = asyncio.Queue()
    n_tools = len(tool_calls)
    retryable_errors: list[Exception] = []

    async def _run_tool(idx: int) -> None:
        tool_name = tool_calls[idx].get("function", {}).get("name", "unknown")
        tool_call_id = tool_calls[idx].get("id", f"unknown_{idx}")
        try:
            async for event in _yield_tool_call(tool_calls, idx, session):
                await queue.put(event)
        except (orjson.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(
                f"Failed to parse tool call {idx} ({tool_name}): {e}",
                exc_info=True,
            )
            retryable_errors.append(e)
        except Exception as e:
            # Infrastructure / setup errors — emit an error output so the
            # client always sees a terminal event and doesn't hang.
            logger.error(f"Tool call {idx} ({tool_name}) failed: {e}", exc_info=True)
            await queue.put(
                StreamToolOutputAvailable(
                    toolCallId=tool_call_id,
                    toolName=tool_name,
                    output=ErrorResponse(
                        message=f"Tool execution failed: {e!s}",
                        error=type(e).__name__,
                        session_id=session.session_id,
                    ).model_dump_json(),
                    success=False,
                )
            )
        finally:
            await queue.put(None)  # sentinel

    tasks = [asyncio.create_task(_run_tool(idx)) for idx in range(n_tools)]
    try:
        finished = 0
        while finished < n_tools:
            event = await queue.get()
            if event is None:
                finished += 1
            else:
                yield event
        if retryable_errors:
            if len(retryable_errors) > 1:
                logger.warning(
                    f"{len(retryable_errors)} tool calls had retryable errors; "
                    f"re-raising first to trigger retry"
                )
            raise retryable_errors[0]
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _yield_tool_call(
    tool_calls: list[dict[str, Any]],
    yield_idx: int,
    session: ChatSession,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """
    Yield a tool call and its execution result.

    Yields heartbeat events every 10 seconds to keep the SSE connection alive
    while the tool executes.

    Raises:
        orjson.JSONDecodeError: If tool call arguments cannot be parsed as JSON
        KeyError: If expected tool call fields are missing
        TypeError: If tool call structure is invalid
    """
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

    # Run tool execution in background task with heartbeats
    tool_task = asyncio.create_task(
        execute_tool(
            tool_name=tool_name,
            parameters=arguments,
            tool_call_id=tool_call_id,
            user_id=session.user_id,
            session=session,
        )
    )

    # Yield heartbeats every 10 seconds while waiting for tool to complete
    # IMPORTANT: Must be less than frontend timeout (12s in useCopilotPage.ts)
    heartbeat_interval = 10.0  # seconds
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


def _log_api_error(
    error: Exception,
    context: str,
    session_id: str | None = None,
    message_count: int | None = None,
    model: str | None = None,
    retry_count: int = 0,
) -> None:
    """Log detailed API error information for debugging."""
    details = _extract_api_error_details(error)
    details["context"] = context
    details["session_id"] = session_id
    details["message_count"] = message_count
    details["model"] = model
    details["retry_count"] = retry_count

    if isinstance(error, RateLimitError):
        logger.warning(f"Rate limit error in {context}: {details}", exc_info=error)
    elif isinstance(error, APIConnectionError):
        logger.warning(f"API connection error in {context}: {details}", exc_info=error)
    elif isinstance(error, APIStatusError) and error.status_code >= 500:
        logger.error(f"API server error (5xx) in {context}: {details}", exc_info=error)
    else:
        logger.error(f"API error in {context}: {details}", exc_info=error)


def _extract_api_error_details(error: Exception) -> dict[str, Any]:
    """Extract detailed information from OpenAI/OpenRouter API errors."""
    error_msg = str(error)
    details: dict[str, Any] = {
        "error_type": type(error).__name__,
        "error_message": error_msg[:500] + "..." if len(error_msg) > 500 else error_msg,
    }

    if hasattr(error, "code"):
        details["code"] = getattr(error, "code", None)
    if hasattr(error, "param"):
        details["param"] = getattr(error, "param", None)

    if isinstance(error, APIStatusError):
        details["status_code"] = error.status_code
        details["request_id"] = getattr(error, "request_id", None)

        if hasattr(error, "body") and error.body:
            details["response_body"] = _sanitize_error_body(error.body)

        if hasattr(error, "response") and error.response:
            headers = error.response.headers
            details["openrouter_provider"] = headers.get("x-openrouter-provider")
            details["openrouter_model"] = headers.get("x-openrouter-model")
            details["retry_after"] = headers.get("retry-after")
            details["rate_limit_remaining"] = headers.get("x-ratelimit-remaining")

    return details


def _sanitize_error_body(
    body: Any, max_length: int = 2000
) -> dict[str, Any] | str | None:
    """Extract only safe fields from error response body to avoid logging sensitive data."""
    if not isinstance(body, dict):
        # Non-dict bodies (e.g., HTML error pages) - return truncated string
        if body is not None:
            body_str = str(body)
            if len(body_str) > max_length:
                return body_str[:max_length] + "...[truncated]"
            return body_str
        return None

    safe_fields = ("message", "type", "code", "param", "error")
    sanitized: dict[str, Any] = {}

    for field in safe_fields:
        if field in body:
            value = body[field]
            if field == "error" and isinstance(value, dict):
                sanitized[field] = _sanitize_error_body(value, max_length)
            elif isinstance(value, str) and len(value) > max_length:
                sanitized[field] = value[:max_length] + "...[truncated]"
            else:
                sanitized[field] = value

    return sanitized if sanitized else None
