"""Claude Agent SDK service layer for CoPilot chat completions."""

import asyncio
import glob
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import openai

from backend.data.understanding import (
    format_understanding_for_prompt,
    get_business_understanding,
)
from backend.util.exceptions import NotFoundError

from ..config import ChatConfig
from ..model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    update_session_title,
    upsert_chat_session,
)
from ..response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamStart,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
)
from ..tracking import track_user_message
from .anthropic_fallback import stream_with_anthropic
from .response_adapter import SDKResponseAdapter
from .security_hooks import create_security_hooks
from .session_file import cleanup_session_file, write_session_file
from .tool_adapter import (
    COPILOT_TOOL_NAMES,
    create_copilot_mcp_server,
    set_execution_context,
)

logger = logging.getLogger(__name__)
config = ChatConfig()

# Set to hold background tasks to prevent garbage collection
_background_tasks: set[asyncio.Task[Any]] = set()

# SDK tool-results directory pattern
_SDK_TOOL_RESULTS_GLOB = "/root/.claude/projects/*/tool-results/*"


def _cleanup_sdk_tool_results() -> None:
    """Remove SDK tool-result files to prevent disk accumulation."""
    for path in glob.glob(_SDK_TOOL_RESULTS_GLOB):
        try:
            os.remove(path)
        except OSError:
            pass


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


async def _build_system_prompt(
    user_id: str | None, has_conversation_history: bool = False
) -> tuple[str, Any]:
    """Build the system prompt with user's business understanding context.

    Args:
        user_id: The user ID to fetch understanding for.
        has_conversation_history: Whether there's existing conversation history.
            If True, we don't tell the model to greet/introduce (since they're
            already in a conversation).
    """
    understanding = None
    if user_id:
        try:
            understanding = await get_business_understanding(user_id)
        except Exception as e:
            logger.warning(f"Failed to fetch business understanding: {e}")

    if understanding:
        context = format_understanding_for_prompt(understanding)
    elif has_conversation_history:
        # Don't tell model to greet if there's conversation history
        context = "No prior understanding saved yet. Continue the existing conversation naturally."
    else:
        context = "This is the first time you are meeting the user. Greet them and introduce them to the platform"

    return DEFAULT_SYSTEM_PROMPT.replace("{users_information}", context), understanding


async def _generate_session_title(
    message: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str | None:
    """Generate a concise title for a chat session."""
    from backend.util.settings import Settings

    settings = Settings()
    try:
        # Build extra_body for OpenRouter tracing
        extra_body: dict[str, Any] = {
            "posthogProperties": {"environment": settings.config.app_env.value},
        }
        if user_id:
            extra_body["user"] = user_id[:128]
            extra_body["posthogDistinctId"] = user_id
        if session_id:
            extra_body["session_id"] = session_id[:128]

        client = openai.AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        response = await client.chat.completions.create(
            model=config.title_model,
            messages=[
                {
                    "role": "system",
                    "content": "Generate a very short title (3-6 words) for a chat conversation based on the user's first message. Return ONLY the title, no quotes or punctuation.",
                },
                {"role": "user", "content": message[:500]},
            ],
            max_tokens=20,
            extra_body=extra_body,
        )
        title = response.choices[0].message.content
        if title:
            title = title.strip().strip("\"'")
            return title[:47] + "..." if len(title) > 50 else title
        return None
    except Exception as e:
        logger.warning(f"Failed to generate session title: {e}")
        return None


async def stream_chat_completion_sdk(
    session_id: str,
    message: str | None = None,
    tool_call_response: str | None = None,  # noqa: ARG001
    is_user_message: bool = True,
    user_id: str | None = None,
    retry_count: int = 0,  # noqa: ARG001
    session: ChatSession | None = None,
    context: dict[str, str] | None = None,  # noqa: ARG001
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Stream chat completion using Claude Agent SDK.

    Drop-in replacement for stream_chat_completion with improved reliability.
    """

    if session is None:
        session = await get_chat_session(session_id, user_id)

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
        if is_user_message:
            track_user_message(
                user_id=user_id, session_id=session_id, message_length=len(message)
            )

    session = await upsert_chat_session(session)

    # Generate title for new sessions (first user message)
    if is_user_message and not session.title:
        user_messages = [m for m in session.messages if m.role == "user"]
        if len(user_messages) == 1:
            first_message = user_messages[0].content or message or ""
            if first_message:
                task = asyncio.create_task(
                    _update_title_async(session_id, first_message, user_id)
                )
                # Store reference to prevent garbage collection
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

    # Check if there's conversation history (more than just the current message)
    has_history = len(session.messages) > 1
    system_prompt, _ = await _build_system_prompt(
        user_id, has_conversation_history=has_history
    )
    set_execution_context(user_id, session, None)

    message_id = str(uuid.uuid4())
    text_block_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())

    yield StreamStart(messageId=message_id, taskId=task_id)

    # Track whether the stream completed normally via ResultMessage
    stream_completed = False

    try:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            # Create MCP server with CoPilot tools
            mcp_server = create_copilot_mcp_server()

            # For multi-turn conversations, write a session file so the CLI
            # loads full user+assistant context via --resume.  This enables
            # turn-level compaction for long conversations.
            resume_id: str | None = None
            if len(session.messages) > 1:
                resume_id = write_session_file(session)
                if resume_id:
                    logger.info(
                        f"[SDK] Wrote session file for --resume: "
                        f"{len(session.messages) - 1} prior messages"
                    )

            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                mcp_servers={"copilot": mcp_server},  # type: ignore[arg-type]
                allowed_tools=COPILOT_TOOL_NAMES,
                hooks=create_security_hooks(user_id),  # type: ignore[arg-type]
                resume=resume_id,
            )

            adapter = SDKResponseAdapter(message_id=message_id)
            adapter.set_task_id(task_id)

            try:
                async with ClaudeSDKClient(options=options) as client:
                    # Determine the current user message
                    current_message = message or ""
                    if not current_message and session.messages:
                        last_user = [m for m in session.messages if m.role == "user"]
                        if last_user:
                            current_message = last_user[-1].content or ""

                    # Guard against empty messages
                    if not current_message.strip():
                        yield StreamError(
                            errorText="Message cannot be empty.",
                            code="empty_prompt",
                        )
                        yield StreamFinish()
                        return

                    await client.query(current_message, session_id=session_id)
                    logger.info(
                        "[SDK] Query sent"
                        + (" (with --resume)" if resume_id else " (new)")
                    )

                    # Track assistant response to save to session
                    # We may need multiple assistant messages if text comes after tool results
                    assistant_response = ChatMessage(role="assistant", content="")
                    accumulated_tool_calls: list[dict[str, Any]] = []
                    has_appended_assistant = False
                    has_tool_results = False  # Track if we've received tool results

                    # Receive messages from the SDK
                    async for sdk_msg in client.receive_messages():
                        for response in adapter.convert_message(sdk_msg):
                            if isinstance(response, StreamStart):
                                continue
                            yield response

                            # Accumulate text deltas into assistant response
                            if isinstance(response, StreamTextDelta):
                                delta = response.delta or ""
                                # After tool results, create new assistant message for post-tool text
                                if has_tool_results and has_appended_assistant:
                                    assistant_response = ChatMessage(
                                        role="assistant", content=delta
                                    )
                                    accumulated_tool_calls = []  # Reset for new message
                                    session.messages.append(assistant_response)
                                    has_tool_results = False
                                else:
                                    assistant_response.content = (
                                        assistant_response.content or ""
                                    ) + delta
                                    if not has_appended_assistant:
                                        session.messages.append(assistant_response)
                                        has_appended_assistant = True

                            # Track tool calls on the assistant message
                            elif isinstance(response, StreamToolInputAvailable):
                                accumulated_tool_calls.append(
                                    {
                                        "id": response.toolCallId,
                                        "type": "function",
                                        "function": {
                                            "name": response.toolName,
                                            "arguments": json.dumps(
                                                response.input or {}
                                            ),
                                        },
                                    }
                                )
                                # Update assistant message with tool calls
                                assistant_response.tool_calls = accumulated_tool_calls
                                # Append assistant message if not already (tool-only response)
                                if not has_appended_assistant:
                                    session.messages.append(assistant_response)
                                    has_appended_assistant = True

                            elif isinstance(response, StreamToolOutputAvailable):
                                session.messages.append(
                                    ChatMessage(
                                        role="tool",
                                        content=(
                                            response.output
                                            if isinstance(response.output, str)
                                            else str(response.output)
                                        ),
                                        tool_call_id=response.toolCallId,
                                    )
                                )
                                has_tool_results = True

                            elif isinstance(response, StreamFinish):
                                stream_completed = True

                        # Break out of the message loop if we received finish signal
                        if stream_completed:
                            break

                    # Ensure assistant response is saved even if no text deltas
                    # (e.g., only tool calls were made)
                    if (
                        assistant_response.content or assistant_response.tool_calls
                    ) and not has_appended_assistant:
                        session.messages.append(assistant_response)

            finally:
                # Always clean up SDK tool-result files, even on error
                _cleanup_sdk_tool_results()
                # Clean up session file written for --resume
                if resume_id:
                    cleanup_session_file(resume_id)

        except ImportError:
            logger.warning(
                "[SDK] claude-agent-sdk not available, using Anthropic fallback"
            )
            async for response in stream_with_anthropic(
                session, system_prompt, text_block_id
            ):
                if isinstance(response, StreamFinish):
                    stream_completed = True
                yield response

        # Save the session with accumulated messages
        await upsert_chat_session(session)
        logger.debug(
            f"[SDK] Session {session_id} saved with {len(session.messages)} messages"
        )
        # Yield StreamFinish to signal completion to the caller (routes.py)
        # Only if one hasn't already been yielded by the stream
        if not stream_completed:
            yield StreamFinish()

    except Exception as e:
        logger.error(f"[SDK] Error: {e}", exc_info=True)
        # Save session even on error to preserve any partial response
        try:
            await upsert_chat_session(session)
        except Exception as save_err:
            logger.error(f"[SDK] Failed to save session on error: {save_err}")
        # Sanitize error message to avoid exposing internal details
        yield StreamError(
            errorText="An error occurred. Please try again.",
            code="sdk_error",
        )
        yield StreamFinish()


async def _update_title_async(
    session_id: str, message: str, user_id: str | None = None
) -> None:
    """Background task to update session title."""
    try:
        title = await _generate_session_title(
            message, user_id=user_id, session_id=session_id
        )
        if title:
            await update_session_title(session_id, title)
            logger.debug(f"[SDK] Generated title for {session_id}: {title}")
    except Exception as e:
        logger.warning(f"[SDK] Failed to update session title: {e}")
