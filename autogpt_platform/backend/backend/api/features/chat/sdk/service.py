"""Claude Agent SDK service layer for CoPilot chat completions."""

import asyncio
import json
import logging
import os
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

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
from ..service import _build_system_prompt, _generate_session_title
from ..tracking import track_user_message
from .anthropic_fallback import stream_with_anthropic
from .response_adapter import SDKResponseAdapter
from .security_hooks import create_security_hooks
from .tool_adapter import (
    COPILOT_TOOL_NAMES,
    create_copilot_mcp_server,
    set_execution_context,
)

logger = logging.getLogger(__name__)
config = ChatConfig()

# Set to hold background tasks to prevent garbage collection
_background_tasks: set[asyncio.Task[Any]] = set()


def _sanitize_session_id(session_id: str) -> str:
    """Sanitize session_id to prevent path traversal and injection.

    Only allows alphanumeric characters and hyphens, stripping everything else.
    """
    return re.sub(r"[^A-Za-z0-9-]", "", session_id)


def _cleanup_sdk_tool_results(cwd: str) -> None:
    """Remove SDK tool-result files for a specific session working directory.

    The SDK creates tool-result files under ~/.claude/projects/<encoded-cwd>/tool-results/.
    We clean only the specific cwd's results to avoid race conditions between
    concurrent sessions.
    """
    import glob as _glob
    import shutil

    # SDK encodes the cwd path by replacing '/' with '-'
    encoded_cwd = cwd.replace("/", "-")
    project_dir = os.path.expanduser(f"~/.claude/projects/{encoded_cwd}")
    results_glob = os.path.join(project_dir, "tool-results", "*")

    for path in _glob.glob(results_glob):
        try:
            os.remove(path)
        except OSError:
            pass

    # Also clean up the temp cwd directory itself
    if cwd.startswith("/tmp/copilot-"):
        try:
            shutil.rmtree(cwd, ignore_errors=True)
        except OSError:
            pass


async def _compress_conversation_history(
    session: ChatSession,
) -> list[ChatMessage]:
    """Compress prior conversation messages if they exceed the token threshold.

    Uses the shared compress_context() from prompt.py which supports:
    - LLM summarization of old messages (keeps recent ones intact)
    - Progressive content truncation as fallback
    - Middle-out deletion as last resort

    Returns the compressed prior messages (everything except the current message).
    """
    prior = session.messages[:-1]
    if len(prior) < 2:
        return prior

    from backend.util.prompt import compress_context

    # Convert ChatMessages to dicts for compress_context
    messages_dict = []
    for msg in prior:
        msg_dict: dict[str, Any] = {"role": msg.role}
        if msg.content:
            msg_dict["content"] = msg.content
        if msg.tool_calls:
            msg_dict["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            msg_dict["tool_call_id"] = msg.tool_call_id
        messages_dict.append(msg_dict)

    try:
        import openai

        async with openai.AsyncOpenAI(
            api_key=config.api_key, base_url=config.base_url, timeout=30.0
        ) as client:
            result = await compress_context(
                messages=messages_dict,
                model=config.model,
                client=client,
            )
    except Exception as e:
        logger.warning(f"[SDK] Context compression with LLM failed: {e}")
        # Fall back to truncation-only (no LLM summarization)
        result = await compress_context(
            messages=messages_dict,
            model=config.model,
            client=None,
        )

    if result.was_compacted:
        logger.info(
            f"[SDK] Context compacted: {result.original_token_count} -> "
            f"{result.token_count} tokens "
            f"({result.messages_summarized} summarized, "
            f"{result.messages_dropped} dropped)"
        )
        # Convert compressed dicts back to ChatMessages
        return [
            ChatMessage(
                role=m["role"],
                content=m.get("content"),
                tool_calls=m.get("tool_calls"),
                tool_call_id=m.get("tool_call_id"),
            )
            for m in result.messages
        ]

    return prior


def _format_conversation_context(messages: list[ChatMessage]) -> str | None:
    """Format conversation messages into a context prefix for the user message.

    Returns a string like:
        <conversation_history>
        User: hello
        You responded: Hi! How can I help?
        </conversation_history>

    Returns None if there are no messages to format.
    """
    if not messages:
        return None

    lines: list[str] = []
    for msg in messages:
        if not msg.content:
            continue
        if msg.role == "user":
            lines.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            lines.append(f"You responded: {msg.content}")
        # Skip tool messages — they're internal details

    if not lines:
        return None

    return "<conversation_history>\n" + "\n".join(lines) + "\n</conversation_history>"


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
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)

    # Build system prompt (reuses non-SDK path with Langfuse support)
    has_history = len(session.messages) > 1
    system_prompt, _ = await _build_system_prompt(
        user_id, has_conversation_history=has_history
    )
    set_execution_context(user_id, session, None)

    message_id = str(uuid.uuid4())
    text_block_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())

    yield StreamStart(messageId=message_id, taskId=task_id)

    stream_completed = False
    # Use a session-specific temp dir to avoid cleanup race conditions
    # between concurrent sessions. Sanitize session_id to prevent path traversal.
    safe_session_id = _sanitize_session_id(session_id)
    sdk_cwd = f"/tmp/copilot-{safe_session_id}"
    os.makedirs(sdk_cwd, exist_ok=True)

    try:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            mcp_server = create_copilot_mcp_server()

            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                mcp_servers={"copilot": mcp_server},  # type: ignore[arg-type]
                allowed_tools=COPILOT_TOOL_NAMES,
                hooks=create_security_hooks(user_id),  # type: ignore[arg-type]
                cwd=sdk_cwd,
            )

            adapter = SDKResponseAdapter(message_id=message_id)
            adapter.set_task_id(task_id)

            async with ClaudeSDKClient(options=options) as client:
                current_message = message or ""
                if not current_message and session.messages:
                    last_user = [m for m in session.messages if m.role == "user"]
                    if last_user:
                        current_message = last_user[-1].content or ""

                if not current_message.strip():
                    yield StreamError(
                        errorText="Message cannot be empty.",
                        code="empty_prompt",
                    )
                    yield StreamFinish()
                    return

                # Build query with conversation history context.
                # Compress history first to handle long conversations.
                query_message = current_message
                if len(session.messages) > 1:
                    compressed = await _compress_conversation_history(session)
                    history_context = _format_conversation_context(compressed)
                    if history_context:
                        query_message = (
                            f"{history_context}\n\n"
                            f"Now, the user says:\n{current_message}"
                        )

                logger.info(
                    f"[SDK] Sending query: {current_message[:80]!r}"
                    f" ({len(session.messages)} msgs in session)"
                )
                await client.query(query_message, session_id=session_id)

                assistant_response = ChatMessage(role="assistant", content="")
                accumulated_tool_calls: list[dict[str, Any]] = []
                has_appended_assistant = False
                has_tool_results = False

                async for sdk_msg in client.receive_messages():
                    logger.debug(
                        f"[SDK] Received: {type(sdk_msg).__name__} "
                        f"{getattr(sdk_msg, 'subtype', '')}"
                    )
                    for response in adapter.convert_message(sdk_msg):
                        if isinstance(response, StreamStart):
                            continue
                        yield response

                        if isinstance(response, StreamTextDelta):
                            delta = response.delta or ""
                            # After tool results, start a new assistant
                            # message for the post-tool text.
                            if has_tool_results and has_appended_assistant:
                                assistant_response = ChatMessage(
                                    role="assistant", content=delta
                                )
                                accumulated_tool_calls = []
                                has_appended_assistant = False
                                has_tool_results = False
                                session.messages.append(assistant_response)
                                has_appended_assistant = True
                            else:
                                assistant_response.content = (
                                    assistant_response.content or ""
                                ) + delta
                                if not has_appended_assistant:
                                    session.messages.append(assistant_response)
                                    has_appended_assistant = True

                        elif isinstance(response, StreamToolInputAvailable):
                            accumulated_tool_calls.append(
                                {
                                    "id": response.toolCallId,
                                    "type": "function",
                                    "function": {
                                        "name": response.toolName,
                                        "arguments": json.dumps(response.input or {}),
                                    },
                                }
                            )
                            assistant_response.tool_calls = accumulated_tool_calls
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

                    if stream_completed:
                        break

                if (
                    assistant_response.content or assistant_response.tool_calls
                ) and not has_appended_assistant:
                    session.messages.append(assistant_response)

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

        await upsert_chat_session(session)
        logger.debug(
            f"[SDK] Session {session_id} saved with {len(session.messages)} messages"
        )
        if not stream_completed:
            yield StreamFinish()

    except Exception as e:
        logger.error(f"[SDK] Error: {e}", exc_info=True)
        try:
            await upsert_chat_session(session)
        except Exception as save_err:
            logger.error(f"[SDK] Failed to save session on error: {save_err}")
        yield StreamError(
            errorText="An error occurred. Please try again.",
            code="sdk_error",
        )
        yield StreamFinish()
    finally:
        _cleanup_sdk_tool_results(sdk_cwd)


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
