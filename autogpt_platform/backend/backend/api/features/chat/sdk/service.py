"""Claude Agent SDK service layer for CoPilot chat completions."""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from backend.util.exceptions import NotFoundError

from .. import stream_registry
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
from ..service import (
    _build_system_prompt,
    _execute_long_running_tool_with_streaming,
    _generate_session_title,
)
from ..tools.models import OperationPendingResponse, OperationStartedResponse
from ..tools.sandbox import WORKSPACE_PREFIX, make_session_path
from ..tracking import track_user_message
from .response_adapter import SDKResponseAdapter
from .security_hooks import create_security_hooks
from .tool_adapter import (
    COPILOT_TOOL_NAMES,
    LongRunningCallback,
    create_copilot_mcp_server,
    set_execution_context,
)
from .transcript import (
    delete_transcript,
    download_transcript,
    read_transcript_file,
    upload_transcript,
    validate_transcript,
    write_transcript_to_tempfile,
)

logger = logging.getLogger(__name__)
config = ChatConfig()

# Set to hold background tasks to prevent garbage collection
_background_tasks: set[asyncio.Task[Any]] = set()


_SDK_CWD_PREFIX = WORKSPACE_PREFIX

# Appended to the system prompt to inform the agent about available tools.
# The SDK built-in Bash is NOT available — use mcp__copilot__bash_exec instead,
# which has kernel-level network isolation (unshare --net).
_SDK_TOOL_SUPPLEMENT = """

## Tool notes

- The SDK built-in Bash tool is NOT available.  Use the `bash_exec` MCP tool
  for shell commands — it runs in a network-isolated sandbox.
- **Shared workspace**: The SDK Read/Write tools and `bash_exec` share the
  same working directory. Files created by one are readable by the other.
  These files are **ephemeral** — they exist only for the current session.
- **Persistent storage**: Use `write_workspace_file` / `read_workspace_file`
  for files that should persist across sessions (stored in cloud storage).
- Long-running tools (create_agent, edit_agent, etc.) are handled
  asynchronously.  You will receive an immediate response; the actual result
  is delivered to the user via a background stream.
"""


def _build_long_running_callback(user_id: str | None) -> LongRunningCallback:
    """Build a callback that delegates long-running tools to the non-SDK infrastructure.

    Long-running tools (create_agent, edit_agent, etc.) are delegated to the
    existing background infrastructure: stream_registry (Redis Streams),
    database persistence, and SSE reconnection.  This means results survive
    page refreshes / pod restarts, and the frontend shows the proper loading
    widget with progress updates.

    The returned callback matches the ``LongRunningCallback`` signature:
    ``(tool_name, args, session) -> MCP response dict``.
    """

    async def _callback(
        tool_name: str, args: dict[str, Any], session: ChatSession
    ) -> dict[str, Any]:
        operation_id = str(uuid.uuid4())
        task_id = str(uuid.uuid4())
        tool_call_id = f"sdk-{uuid.uuid4().hex[:12]}"
        session_id = session.session_id

        # --- Build user-friendly messages (matches non-SDK service) ---
        if tool_name == "create_agent":
            desc = args.get("description", "")
            desc_preview = (desc[:100] + "...") if len(desc) > 100 else desc
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
            changes = args.get("changes", "")
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

        # --- Register task in Redis for SSE reconnection ---
        await stream_registry.create_task(
            task_id=task_id,
            session_id=session_id,
            user_id=user_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            operation_id=operation_id,
        )

        # --- Save OperationPendingResponse to chat history ---
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

        # --- Spawn background task (reuses non-SDK infrastructure) ---
        bg_task = asyncio.create_task(
            _execute_long_running_tool_with_streaming(
                tool_name=tool_name,
                parameters=args,
                tool_call_id=tool_call_id,
                operation_id=operation_id,
                task_id=task_id,
                session_id=session_id,
                user_id=user_id,
            )
        )
        _background_tasks.add(bg_task)
        bg_task.add_done_callback(_background_tasks.discard)
        await stream_registry.set_task_asyncio_task(task_id, bg_task)

        logger.info(
            f"[SDK] Long-running tool {tool_name} delegated to background "
            f"(operation_id={operation_id}, task_id={task_id})"
        )

        # --- Return OperationStartedResponse as MCP tool result ---
        # This flows through SDK → response adapter → frontend, triggering
        # the loading widget with SSE reconnection support.
        started_json = OperationStartedResponse(
            message=started_msg,
            operation_id=operation_id,
            tool_name=tool_name,
            task_id=task_id,
        ).model_dump_json()

        return {
            "content": [{"type": "text", "text": started_json}],
            "isError": False,
        }

    return _callback


def _resolve_sdk_model() -> str | None:
    """Resolve the model name for the Claude Agent SDK CLI.

    Uses ``config.claude_agent_model`` if set, otherwise derives from
    ``config.model`` by stripping the OpenRouter provider prefix (e.g.,
    ``"anthropic/claude-opus-4.6"`` → ``"claude-opus-4.6"``).
    """
    if config.claude_agent_model:
        return config.claude_agent_model
    model = config.model
    if "/" in model:
        return model.split("/", 1)[1]
    return model


def _build_sdk_env() -> dict[str, str]:
    """Build env vars for the SDK CLI process.

    Routes API calls through OpenRouter (or a custom base_url) using
    the same ``config.api_key`` / ``config.base_url`` as the non-SDK path.
    This gives per-call token and cost tracking on the OpenRouter dashboard.

    Only overrides ``ANTHROPIC_API_KEY`` when a valid proxy URL and auth
    token are both present — otherwise returns an empty dict so the SDK
    falls back to its default credentials.
    """
    env: dict[str, str] = {}
    if config.api_key and config.base_url:
        # Strip /v1 suffix — SDK expects the base URL without a version path
        base = config.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        if not base or not base.startswith("http"):
            # Invalid base_url — don't override SDK defaults
            return env
        env["ANTHROPIC_BASE_URL"] = base
        env["ANTHROPIC_AUTH_TOKEN"] = config.api_key
        # Must be explicitly empty so the CLI uses AUTH_TOKEN instead
        env["ANTHROPIC_API_KEY"] = ""
    return env


def _make_sdk_cwd(session_id: str) -> str:
    """Create a safe, session-specific working directory path.

    Delegates to :func:`~backend.api.features.chat.tools.sandbox.make_session_path`
    (single source of truth for path sanitization) and adds a defence-in-depth
    assertion.
    """
    cwd = make_session_path(session_id)
    # Defence-in-depth: normpath + startswith is a CodeQL-recognised sanitizer
    cwd = os.path.normpath(cwd)
    if not cwd.startswith(_SDK_CWD_PREFIX):
        raise ValueError(f"SDK cwd escaped prefix: {cwd}")
    return cwd


def _cleanup_sdk_tool_results(cwd: str) -> None:
    """Remove SDK tool-result files for a specific session working directory.

    The SDK creates tool-result files under ~/.claude/projects/<encoded-cwd>/tool-results/.
    We clean only the specific cwd's results to avoid race conditions between
    concurrent sessions.

    Security: cwd MUST be created by _make_sdk_cwd() which sanitizes session_id.
    """
    import shutil

    # Security check 1: Validate cwd is under the expected prefix
    normalized = os.path.normpath(cwd)
    if not normalized.startswith(_SDK_CWD_PREFIX):
        logger.warning(f"[SDK] Rejecting cleanup for invalid path: {cwd}")
        return

    # Security check 2: Verify path stayed within workspace after normalization
    if not normalized.startswith(_SDK_CWD_PREFIX):
        logger.warning(f"[SDK] Rejecting cleanup for path outside workspace: {cwd}")
        return

    # SDK encodes the cwd path by replacing '/' with '-'
    encoded_cwd = normalized.replace("/", "-")

    # Construct the project directory path (known-safe home expansion)
    claude_projects = os.path.expanduser("~/.claude/projects")
    project_dir = os.path.join(claude_projects, encoded_cwd)

    # Security check 3: Validate project_dir is under ~/.claude/projects
    project_dir = os.path.normpath(project_dir)
    if not project_dir.startswith(claude_projects):
        logger.warning(
            f"[SDK] Rejecting cleanup for escaped project path: {project_dir}"
        )
        return

    results_dir = os.path.join(project_dir, "tool-results")
    if os.path.isdir(results_dir):
        for filename in os.listdir(results_dir):
            file_path = os.path.join(results_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError:
                pass

    # Also clean up the temp cwd directory itself
    try:
        shutil.rmtree(normalized, ignore_errors=True)
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
    system_prompt += _SDK_TOOL_SUPPLEMENT
    message_id = str(uuid.uuid4())
    task_id = str(uuid.uuid4())

    yield StreamStart(messageId=message_id, taskId=task_id)

    stream_completed = False
    # Initialise sdk_cwd before the try so the finally can reference it
    # even if _make_sdk_cwd raises (in that case it stays as "").
    sdk_cwd = ""
    use_resume = False

    try:
        # Use a session-specific temp dir to avoid cleanup race conditions
        # between concurrent sessions.
        sdk_cwd = _make_sdk_cwd(session_id)
        os.makedirs(sdk_cwd, exist_ok=True)

        set_execution_context(
            user_id,
            session,
            long_running_callback=_build_long_running_callback(user_id),
        )
        try:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            # Fail fast when no API credentials are available at all
            sdk_env = _build_sdk_env()
            if not sdk_env and not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "No API key configured. Set OPEN_ROUTER_API_KEY "
                    "(or CHAT_API_KEY) for OpenRouter routing, "
                    "or ANTHROPIC_API_KEY for direct Anthropic access."
                )

            mcp_server = create_copilot_mcp_server()

            sdk_model = _resolve_sdk_model()

            # --- Transcript capture via Stop hook ---
            captured_transcript: dict[str, str] = {}

            def _on_stop(transcript_path: str, sdk_session_id: str) -> None:
                captured_transcript["path"] = transcript_path
                captured_transcript["session_id"] = sdk_session_id

            security_hooks = create_security_hooks(
                user_id,
                sdk_cwd=sdk_cwd,
                max_subtasks=config.claude_agent_max_subtasks,
                on_stop=_on_stop if config.claude_agent_use_resume else None,
            )

            # --- Resume strategy: download transcript from bucket ---
            resume_file: str | None = None
            use_resume = False

            if config.claude_agent_use_resume and user_id and len(session.messages) > 1:
                transcript_content = await download_transcript(user_id, session_id)
                if transcript_content and validate_transcript(transcript_content):
                    resume_file = write_transcript_to_tempfile(
                        transcript_content, session_id, sdk_cwd
                    )
                    if resume_file:
                        use_resume = True
                        logger.info(
                            f"[SDK] Using --resume with transcript "
                            f"({len(transcript_content)} bytes)"
                        )

            sdk_options_kwargs: dict[str, Any] = {
                "system_prompt": system_prompt,
                "mcp_servers": {"copilot": mcp_server},
                "allowed_tools": COPILOT_TOOL_NAMES,
                "disallowed_tools": ["Bash"],
                "hooks": security_hooks,
                "cwd": sdk_cwd,
                "max_buffer_size": config.claude_agent_max_buffer_size,
            }
            if sdk_env:
                sdk_options_kwargs["model"] = sdk_model
                sdk_options_kwargs["env"] = sdk_env
            if use_resume and resume_file:
                sdk_options_kwargs["resume"] = resume_file

            options = ClaudeAgentOptions(**sdk_options_kwargs)  # type: ignore[arg-type]

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

                # Build query: with --resume the CLI already has full context,
                # so we only send the new message.  Without resume, compress
                # history into a context prefix as before.
                query_message = current_message
                if not use_resume and len(session.messages) > 1:
                    logger.warning(
                        f"[SDK] Using compression fallback for session "
                        f"{session_id} ({len(session.messages)} messages) — "
                        f"no transcript available for --resume"
                    )
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

                # --- Capture transcript while CLI is still alive ---
                # Must happen INSIDE async with: close() sends SIGTERM
                # which kills the CLI before it can flush the JSONL.
                if (
                    config.claude_agent_use_resume
                    and user_id
                    and captured_transcript.get("path")
                ):
                    # Give CLI time to flush JSONL writes before we read
                    await asyncio.sleep(0.5)
                    raw_transcript = read_transcript_file(captured_transcript["path"])
                    if raw_transcript:
                        # Upload in background — strip + store to bucket
                        task = asyncio.create_task(
                            _upload_transcript_bg(user_id, session_id, raw_transcript)
                        )
                        _background_tasks.add(task)
                        task.add_done_callback(_background_tasks.discard)
                    else:
                        logger.debug("[SDK] Stop hook fired but transcript not usable")

        except ImportError:
            raise RuntimeError(
                "claude-agent-sdk is not installed. "
                "Disable SDK mode (CHAT_USE_CLAUDE_AGENT_SDK=false) "
                "to use the OpenAI-compatible fallback."
            )

        await upsert_chat_session(session)
        logger.debug(
            f"[SDK] Session {session_id} saved with {len(session.messages)} messages"
        )
        if not stream_completed:
            yield StreamFinish()

    except Exception as e:
        logger.error(f"[SDK] Error: {e}", exc_info=True)
        if use_resume and user_id:
            logger.warning("[SDK] Deleting transcript after resume failure")
            task = asyncio.create_task(delete_transcript(user_id, session_id))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
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
        if sdk_cwd:
            _cleanup_sdk_tool_results(sdk_cwd)


async def _upload_transcript_bg(
    user_id: str, session_id: str, raw_content: str
) -> None:
    """Background task to strip progress entries and upload transcript."""
    try:
        await upload_transcript(user_id, session_id, raw_content)
    except Exception as e:
        logger.error(f"[SDK] Failed to upload transcript for {session_id}: {e}")


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
