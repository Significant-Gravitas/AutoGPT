"""Claude Agent SDK service layer for CoPilot chat completions."""

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from backend.util.exceptions import NotFoundError

from .. import db as chat_db
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
    StreamFinishStep,
    StreamHeartbeat,
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
    SDK_DISALLOWED_TOOLS,
    LongRunningCallback,
    create_copilot_mcp_server,
    set_execution_context,
    wait_for_stash,
)
from .transcript import (
    cleanup_cli_project_dir,
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


@dataclass
class CapturedTranscript:
    """Info captured by the SDK Stop hook for stateless --resume."""

    path: str = ""
    sdk_session_id: str = ""
    raw_content: str = ""

    @property
    def available(self) -> bool:
        return bool(self.path)


_SDK_CWD_PREFIX = WORKSPACE_PREFIX

# Heartbeat interval — keep SSE alive through proxies/LBs during tool execution.
_HEARTBEAT_INTERVAL = 15.0  # seconds

# Appended to the system prompt to inform the agent about available tools.
# The SDK built-in Bash is NOT available — use mcp__copilot__bash_exec instead,
# which has kernel-level network isolation (unshare --net).
_SDK_TOOL_SUPPLEMENT = """

## Tool notes

### Shell commands
- The SDK built-in Bash tool is NOT available.  Use the `bash_exec` MCP tool
  for shell commands — it runs in a network-isolated sandbox.

### Two storage systems — CRITICAL to understand

1. **Ephemeral working directory** (`/tmp/copilot-<session>/`):
   - Shared by SDK Read/Write/Edit/Glob/Grep tools AND `bash_exec`
   - Files here are **lost between turns** — do NOT rely on them persisting
   - Use for temporary work: running scripts, processing data, etc.

2. **Persistent workspace** (cloud storage):
   - Files here **survive across turns and sessions**
   - Use `write_workspace_file` to save important files (code, outputs, configs)
   - Use `read_workspace_file` to retrieve previously saved files
   - Use `list_workspace_files` to see what files you've saved before
   - Call `list_workspace_files(include_all_sessions=True)` to see files from
     all sessions

### Moving files between ephemeral and persistent storage
- **Ephemeral → Persistent**: Use `write_workspace_file` with either:
  - `content` param (plain text) — for text files
  - `source_path` param — to copy any file directly from the ephemeral dir
- **Persistent → Ephemeral**: Use `read_workspace_file` with `save_to_path`
  param to download a workspace file to the ephemeral dir for processing

### File persistence workflow
When you create or modify important files (code, configs, outputs), you MUST:
1. Save them using `write_workspace_file` so they persist
2. At the start of a new turn, call `list_workspace_files` to see what files
   are available from previous turns

### Long-running tools
Long-running tools (create_agent, edit_agent, etc.) are handled
asynchronously.  You will receive an immediate response; the actual result
is delivered to the user via a background stream.

### Sub-agent tasks
- When using the Task tool, NEVER set `run_in_background` to true.
  All tasks must run in the foreground.
"""


def _build_long_running_callback(
    user_id: str | None,
    saved_msg_count_ref: list[int] | None = None,
) -> LongRunningCallback:
    """Build a callback that delegates long-running tools to the non-SDK infrastructure.

    Long-running tools (create_agent, edit_agent, etc.) are delegated to the
    existing background infrastructure: stream_registry (Redis Streams),
    database persistence, and SSE reconnection.  This means results survive
    page refreshes / pod restarts, and the frontend shows the proper loading
    widget with progress updates.

    Args:
        user_id: User ID for the session
        saved_msg_count_ref: Mutable reference [count] shared with streaming loop
            for coordinating message saves. When provided, the callback will update
            it after appending messages to prevent counter drift.

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
        # Layer 2: Query DB for latest count before save (defense against stale counter)
        db_count = await chat_db.get_chat_session_message_count(session_id)
        await upsert_chat_session(session, existing_message_count=db_count)
        # Layer 3: Update shared counter so streaming loop stays in sync
        if saved_msg_count_ref is not None:
            saved_msg_count_ref[0] = len(session.messages)

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

    Delegates to :func:`~backend.copilot.tools.sandbox.make_session_path`
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
    """Remove SDK session artifacts for a specific working directory.

    Cleans up:
    - ``~/.claude/projects/<encoded-cwd>/`` — CLI session transcripts and
      tool-result files.  Each SDK turn uses a unique cwd, so this directory
      is safe to remove entirely.
    - ``/tmp/copilot-<session>/`` — the ephemeral working directory.

    Security: *cwd* MUST be created by ``_make_sdk_cwd()`` which sanitizes
    the session_id.
    """
    import shutil

    normalized = os.path.normpath(cwd)
    if not normalized.startswith(_SDK_CWD_PREFIX):
        logger.warning(f"[SDK] Rejecting cleanup for path outside workspace: {cwd}")
        return

    # Clean the CLI's project directory (transcripts + tool-results).
    cleanup_cli_project_dir(cwd)

    # Clean up the temp cwd directory itself.
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
    messages = session.messages[:-1]
    if len(messages) < 2:
        return messages

    from backend.util.prompt import compress_context

    # Convert ChatMessages to dicts for compress_context
    messages_dict = []
    for msg in messages:
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

    return messages


def _format_conversation_context(messages: list[ChatMessage]) -> str | None:
    """Format conversation messages into a context prefix for the user message.

    Includes user messages, assistant text, tool call summaries, and
    tool result summaries so the agent retains full context about what
    tools were invoked and their outcomes.

    Returns None if there are no messages to format.
    """
    if not messages:
        return None

    lines: list[str] = []
    for msg in messages:
        if msg.role == "user":
            if msg.content:
                lines.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            if msg.content:
                lines.append(f"You responded: {msg.content}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    func = tc.get("function", {})
                    tool_name = func.get("name", "unknown")
                    tool_args = func.get("arguments", "")
                    lines.append(f"You called tool: {tool_name}({tool_args})")
        elif msg.role == "tool":
            content = msg.content or ""
            lines.append(f"Tool result: {content}")

    if not lines:
        return None

    return "<conversation_history>\n" + "\n".join(lines) + "\n</conversation_history>"


def _is_tool_error_or_denial(content: str | None) -> bool:
    """Check if a tool message content indicates an error or denial.

    Currently unused — ``_format_conversation_context`` includes all tool
    results.  Kept as a utility for future selective filtering.
    """
    if not content:
        return False
    lower = content.lower()
    return any(
        marker in lower
        for marker in (
            "[security]",
            "cannot be bypassed",
            "not allowed",
            "not supported",  # background-task denial
            "maximum",  # subtask-limit denial
            "denied",
            "blocked",
            "failed to",  # internal tool execution failures
            '"iserror": true',  # MCP protocol error flag
        )
    )


async def _build_query_message(
    current_message: str,
    session: ChatSession,
    use_resume: bool,
    transcript_msg_count: int,
    session_id: str,
) -> str:
    """Build the query message with appropriate context.

    With --resume the CLI already has full context, so only the new message
    is needed.  Without resume, compress history into a context prefix.
    Hybrid mode: if the transcript is stale, compress only the gap.
    """
    msg_count = len(session.messages)

    if use_resume and transcript_msg_count > 0:
        if transcript_msg_count < msg_count - 1:
            gap = session.messages[transcript_msg_count:-1]
            gap_context = _format_conversation_context(gap)
            if gap_context:
                logger.info(
                    f"[SDK] Transcript stale: covers {transcript_msg_count} "
                    f"of {msg_count} messages, compressing {len(gap)} missed"
                )
                return f"{gap_context}\n\nNow, the user says:\n{current_message}"
    elif not use_resume and msg_count > 1:
        logger.warning(
            f"[SDK] Using compression fallback for session "
            f"{session_id} ({msg_count} messages) — no transcript for --resume"
        )
        compressed = await _compress_conversation_history(session)
        history_context = _format_conversation_context(compressed)
        if history_context:
            return f"{history_context}\n\nNow, the user says:\n{current_message}"

    return current_message


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
    # Initialise variables before the try so the finally block can
    # always attempt transcript upload regardless of errors.
    sdk_cwd = ""
    use_resume = False
    resume_file: str | None = None
    captured_transcript = CapturedTranscript()

    try:
        # Use a session-specific temp dir to avoid cleanup race conditions
        # between concurrent sessions.
        sdk_cwd = _make_sdk_cwd(session_id)
        os.makedirs(sdk_cwd, exist_ok=True)

        # Initialize saved message counter as mutable list so long-running
        # callback and streaming loop can coordinate (Layer 3: defense-in-depth)
        saved_msg_count_ref: list[int] = [len(session.messages)]

        set_execution_context(
            user_id,
            session,
            long_running_callback=_build_long_running_callback(
                user_id, saved_msg_count_ref
            ),
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
            # Read the file content immediately — the SDK may clean up
            # the file before our finally block runs.
            def _on_stop(transcript_path: str, sdk_session_id: str) -> None:
                captured_transcript.path = transcript_path
                captured_transcript.sdk_session_id = sdk_session_id
                content = read_transcript_file(transcript_path)
                if content:
                    captured_transcript.raw_content = content
                    logger.info(
                        f"[SDK] Stop hook: captured {len(content)}B from "
                        f"{transcript_path}"
                    )
                else:
                    logger.warning(
                        f"[SDK] Stop hook: transcript file empty/missing at "
                        f"{transcript_path}"
                    )

            security_hooks = create_security_hooks(
                user_id,
                sdk_cwd=sdk_cwd,
                max_subtasks=config.claude_agent_max_subtasks,
                on_stop=_on_stop if config.claude_agent_use_resume else None,
            )

            # --- Resume strategy: download transcript from bucket ---
            transcript_msg_count = 0  # watermark: session.messages length at upload

            if config.claude_agent_use_resume and user_id and len(session.messages) > 1:
                dl = await download_transcript(user_id, session_id)
                is_valid = bool(dl and validate_transcript(dl.content))
                if dl and is_valid:
                    logger.info(
                        f"[SDK] Transcript available for session {session_id}: "
                        f"{len(dl.content)}B, msg_count={dl.message_count}"
                    )
                    resume_file = write_transcript_to_tempfile(
                        dl.content, session_id, sdk_cwd
                    )
                    if resume_file:
                        use_resume = True
                        transcript_msg_count = dl.message_count
                        logger.debug(
                            f"[SDK] Using --resume ({len(dl.content)}B, "
                            f"msg_count={transcript_msg_count})"
                        )
                elif dl:
                    logger.warning(
                        f"[SDK] Transcript downloaded but invalid for {session_id}"
                    )
                else:
                    logger.warning(
                        f"[SDK] No transcript available for {session_id} "
                        f"({len(session.messages)} messages in session)"
                    )

            sdk_options_kwargs: dict[str, Any] = {
                "system_prompt": system_prompt,
                "mcp_servers": {"copilot": mcp_server},
                "allowed_tools": COPILOT_TOOL_NAMES,
                "disallowed_tools": SDK_DISALLOWED_TOOLS,
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

            adapter = SDKResponseAdapter(message_id=message_id, session_id=session_id)
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

                query_message = await _build_query_message(
                    current_message,
                    session,
                    use_resume,
                    transcript_msg_count,
                    session_id,
                )
                logger.info(
                    "[SDK] [%s] Sending query — resume=%s, "
                    "total_msgs=%d, query_len=%d",
                    session_id[:12],
                    use_resume,
                    len(session.messages),
                    len(query_message),
                )
                await client.query(query_message, session_id=session_id)

                assistant_response = ChatMessage(role="assistant", content="")
                accumulated_tool_calls: list[dict[str, Any]] = []
                has_appended_assistant = False
                has_tool_results = False
                # Track persisted message count. Uses shared ref so long-running
                # callback can update it (Layer 3: defense-in-depth)

                # Use an explicit async iterator with non-cancelling heartbeats.
                # CRITICAL: we must NOT cancel __anext__() mid-flight — doing so
                # (via asyncio.timeout or wait_for) corrupts the SDK's internal
                # anyio memory stream, causing StopAsyncIteration on the next
                # call and silently dropping all in-flight tool results.
                # Instead, wrap __anext__() in a Task and use asyncio.wait()
                # with a timeout.  On timeout we emit a heartbeat but keep the
                # Task alive so it can deliver the next message.
                msg_iter = client.receive_messages().__aiter__()
                pending_task: asyncio.Task[Any] | None = None
                try:
                    while not stream_completed:
                        if pending_task is None:

                            async def _next_msg() -> Any:
                                return await msg_iter.__anext__()

                            pending_task = asyncio.create_task(_next_msg())

                        done, _ = await asyncio.wait(
                            {pending_task}, timeout=_HEARTBEAT_INTERVAL
                        )

                        if not done:
                            # Timeout — emit heartbeat but keep the task alive
                            yield StreamHeartbeat()
                            continue

                        # Task completed — get result
                        pending_task = None
                        try:
                            sdk_msg = done.pop().result()
                        except StopAsyncIteration:
                            logger.info(
                                "[SDK] [%s] Stream ended normally "
                                "(StopAsyncIteration)",
                                session_id[:12],
                            )
                            break
                        except Exception as stream_err:
                            # SDK sends {"type": "error"} which raises
                            # Exception in receive_messages() — capture it
                            # so the session can still be saved and the
                            # frontend gets a clean finish.
                            logger.error(
                                "[SDK] [%s] Stream error from SDK: %s",
                                session_id[:12],
                                stream_err,
                                exc_info=True,
                            )
                            yield StreamError(
                                errorText=f"SDK stream error: {stream_err}",
                                code="sdk_stream_error",
                            )
                            break

                        logger.info(
                            "[SDK] [%s] Received: %s %s "
                            "(unresolved=%d, current=%d, resolved=%d)",
                            session_id[:12],
                            type(sdk_msg).__name__,
                            getattr(sdk_msg, "subtype", ""),
                            len(adapter.current_tool_calls)
                            - len(adapter.resolved_tool_calls),
                            len(adapter.current_tool_calls),
                            len(adapter.resolved_tool_calls),
                        )

                        # Race-condition fix: SDK hooks (PostToolUse) are
                        # executed asynchronously via start_soon() — the next
                        # message can arrive before the hook stashes output.
                        # wait_for_stash() awaits an asyncio.Event signaled by
                        # stash_pending_tool_output(), completing as soon as
                        # the hook finishes (typically <1ms).  The sleep(0)
                        # after lets any remaining concurrent hooks complete.
                        #
                        # Skip for parallel tool continuations: when the SDK
                        # sends parallel tool calls as separate
                        # AssistantMessages (each containing only
                        # ToolUseBlocks), we must NOT wait/flush — the prior
                        # tools are still executing concurrently.
                        from claude_agent_sdk import (
                            AssistantMessage,
                            ResultMessage,
                            ToolUseBlock,
                        )

                        is_parallel_continuation = isinstance(
                            sdk_msg, AssistantMessage
                        ) and all(isinstance(b, ToolUseBlock) for b in sdk_msg.content)

                        if (
                            adapter.has_unresolved_tool_calls
                            and isinstance(sdk_msg, (AssistantMessage, ResultMessage))
                            and not is_parallel_continuation
                        ):
                            if await wait_for_stash(timeout=0.5):
                                await asyncio.sleep(0)
                            else:
                                logger.warning(
                                    "[SDK] [%s] Timed out waiting for "
                                    "PostToolUse hook stash "
                                    "(%d unresolved tool calls)",
                                    session_id[:12],
                                    len(adapter.current_tool_calls)
                                    - len(adapter.resolved_tool_calls),
                                )

                        for response in adapter.convert_message(sdk_msg):
                            if isinstance(response, StreamStart):
                                continue

                            # Log tool events for debugging
                            if isinstance(
                                response,
                                (
                                    StreamToolInputAvailable,
                                    StreamToolOutputAvailable,
                                ),
                            ):
                                extra = ""
                                if isinstance(response, StreamToolOutputAvailable):
                                    out_len = len(str(response.output))
                                    extra = f", output_len={out_len}"
                                logger.info(
                                    "[SDK] [%s] Tool event: %s, tool=%s%s",
                                    session_id[:12],
                                    type(response).__name__,
                                    getattr(response, "toolName", "N/A"),
                                    extra,
                                )

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
                                            "arguments": json.dumps(
                                                response.input or {}
                                            ),
                                        },
                                    }
                                )
                                assistant_response.tool_calls = accumulated_tool_calls
                                if not has_appended_assistant:
                                    session.messages.append(assistant_response)
                                    has_appended_assistant = True
                                # Save before tool execution starts so the
                                # pending tool call is visible on refresh /
                                # other devices.
                                try:
                                    # Layer 2: Query DB for latest count (defense against stale counter)
                                    db_count = (
                                        await chat_db.get_chat_session_message_count(
                                            session_id
                                        )
                                    )
                                    await upsert_chat_session(
                                        session,
                                        existing_message_count=db_count,
                                    )
                                    # Layer 3: Update shared ref so callback stays in sync
                                    saved_msg_count_ref[0] = len(session.messages)
                                except Exception as save_err:
                                    logger.warning(
                                        "[SDK] [%s] Incremental save " "failed: %s",
                                        session_id[:12],
                                        save_err,
                                    )

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
                                # Save after tool completes so the result is
                                # visible on refresh / other devices.
                                try:
                                    # Layer 2: Query DB for latest count (defense against stale counter)
                                    db_count = (
                                        await chat_db.get_chat_session_message_count(
                                            session_id
                                        )
                                    )
                                    await upsert_chat_session(
                                        session,
                                        existing_message_count=db_count,
                                    )
                                    # Layer 3: Update shared ref so callback stays in sync
                                    saved_msg_count_ref[0] = len(session.messages)
                                except Exception as save_err:
                                    logger.warning(
                                        "[SDK] [%s] Incremental save " "failed: %s",
                                        session_id[:12],
                                        save_err,
                                    )

                            elif isinstance(response, StreamFinish):
                                stream_completed = True

                except asyncio.CancelledError:
                    # Task/generator was cancelled (e.g. client disconnect,
                    # server shutdown).  Log and let the safety-net / finally
                    # blocks handle cleanup.
                    logger.warning(
                        "[SDK] [%s] Streaming loop cancelled "
                        "(asyncio.CancelledError)",
                        session_id[:12],
                    )
                    raise
                finally:
                    # Cancel the pending __anext__ task to avoid a leaked
                    # coroutine.  This is safe even if the task already
                    # completed.
                    if pending_task is not None and not pending_task.done():
                        pending_task.cancel()
                        try:
                            await pending_task
                        except (asyncio.CancelledError, StopAsyncIteration):
                            pass

                # Safety net: if tools are still unresolved after the
                # streaming loop (e.g. StopAsyncIteration before ResultMessage,
                # or SDK not sending UserMessages for built-in tools), flush
                # them now so the frontend stops showing spinners.
                if adapter.has_unresolved_tool_calls:
                    logger.warning(
                        "[SDK] [%s] %d unresolved tool(s) after stream loop — "
                        "flushing as safety net",
                        session_id[:12],
                        len(adapter.current_tool_calls)
                        - len(adapter.resolved_tool_calls),
                    )
                    safety_responses: list[StreamBaseResponse] = []
                    adapter._flush_unresolved_tool_calls(safety_responses)
                    for response in safety_responses:
                        if isinstance(
                            response,
                            (StreamToolInputAvailable, StreamToolOutputAvailable),
                        ):
                            logger.info(
                                "[SDK] [%s] Safety flush: %s, tool=%s",
                                session_id[:12],
                                type(response).__name__,
                                getattr(response, "toolName", "N/A"),
                            )
                        yield response

                # If the stream ended without a ResultMessage (no
                # StreamFinish), the SDK CLI exited unexpectedly.  Close
                # the open step and emit StreamFinish so the frontend
                # transitions to the "ready" state.
                if not stream_completed:
                    logger.warning(
                        "[SDK] [%s] Stream ended without ResultMessage "
                        "(StopAsyncIteration) — emitting StreamFinish",
                        session_id[:12],
                    )
                    if adapter.step_open:
                        yield StreamFinishStep()
                        adapter.step_open = False
                    closing_responses: list[StreamBaseResponse] = []
                    adapter._end_text_if_open(closing_responses)
                    for r in closing_responses:
                        yield r
                    yield StreamFinish()
                    stream_completed = True

                if (
                    assistant_response.content or assistant_response.tool_calls
                ) and not has_appended_assistant:
                    session.messages.append(assistant_response)

            # --- Upload transcript for next-turn --resume ---
            # After async with the SDK task group has exited, so the Stop
            # hook has already fired and the CLI has been SIGTERMed.  The
            # CLI uses appendFileSync, so all writes are safely on disk.
            if config.claude_agent_use_resume and user_id:
                # With --resume the CLI appends to the resume file (most
                # complete).  Otherwise use the Stop hook path.
                if use_resume and resume_file:
                    raw_transcript = read_transcript_file(resume_file)
                    logger.debug("[SDK] Transcript source: resume file")
                elif captured_transcript.path:
                    raw_transcript = read_transcript_file(captured_transcript.path)
                    logger.debug(
                        "[SDK] Transcript source: stop hook (%s), " "read result: %s",
                        captured_transcript.path,
                        f"{len(raw_transcript)}B" if raw_transcript else "None",
                    )
                else:
                    raw_transcript = None

                if not raw_transcript:
                    logger.debug(
                        "[SDK] No usable transcript — CLI file had no "
                        "conversation entries (expected for first turn "
                        "without --resume)"
                    )

                if raw_transcript:
                    # Shield the upload from generator cancellation so a
                    # client disconnect / page refresh doesn't lose the
                    # transcript.  The upload must finish even if the SSE
                    # connection is torn down.
                    await asyncio.shield(
                        _try_upload_transcript(
                            user_id,
                            session_id,
                            raw_transcript,
                            message_count=len(session.messages),
                        )
                    )

        except ImportError:
            raise RuntimeError(
                "claude-agent-sdk is not installed. "
                "Disable SDK mode (CHAT_USE_CLAUDE_AGENT_SDK=false) "
                "to use the OpenAI-compatible fallback."
            )

        await asyncio.shield(upsert_chat_session(session))
        logger.info(
            "[SDK] [%s] Session saved with %d messages",
            session_id[:12],
            len(session.messages),
        )
        if not stream_completed:
            yield StreamFinish()

    except asyncio.CancelledError:
        # Client disconnect / server shutdown — log but re-raise so
        # the framework can clean up.  The finally block still runs
        # for transcript upload.
        logger.warning("[SDK] [%s] Session cancelled (CancelledError)", session_id[:12])
        raise
    except Exception as e:
        logger.error(f"[SDK] Error: {e}", exc_info=True)
        try:
            await asyncio.shield(upsert_chat_session(session))
        except Exception as save_err:
            logger.error(f"[SDK] Failed to save session on error: {save_err}")
        yield StreamError(
            errorText="An error occurred. Please try again.",
            code="sdk_error",
        )
        yield StreamFinish()
    finally:
        # --- Upload transcript for next-turn --resume ---
        # This MUST run in finally so the transcript is uploaded even when
        # the streaming loop raises an exception.  The CLI uses
        # appendFileSync, so whatever was written before the error/SIGTERM
        # is safely on disk and still useful for the next turn.
        if config.claude_agent_use_resume and user_id:
            try:
                # Prefer content captured in the Stop hook (read before
                # cleanup removes the file).  Fall back to the resume
                # file when the stop hook didn't fire (e.g. error before
                # completion) so we don't lose the prior transcript.
                raw_transcript = captured_transcript.raw_content or None
                if not raw_transcript and use_resume and resume_file:
                    raw_transcript = read_transcript_file(resume_file)

                if raw_transcript:
                    await asyncio.shield(
                        _try_upload_transcript(
                            user_id,
                            session_id,
                            raw_transcript,
                            message_count=len(session.messages),
                        )
                    )
                else:
                    logger.warning(f"[SDK] No transcript to upload for {session_id}")
            except Exception as upload_err:
                logger.error(
                    f"[SDK] Transcript upload failed in finally: {upload_err}",
                    exc_info=True,
                )

        if sdk_cwd:
            _cleanup_sdk_tool_results(sdk_cwd)


async def _try_upload_transcript(
    user_id: str,
    session_id: str,
    raw_content: str,
    message_count: int = 0,
) -> bool:
    """Strip progress entries and upload transcript (with timeout).

    Returns True if the upload completed without error.
    """
    try:
        async with asyncio.timeout(30):
            await upload_transcript(
                user_id, session_id, raw_content, message_count=message_count
            )
        return True
    except asyncio.TimeoutError:
        logger.warning(f"[SDK] Transcript upload timed out for {session_id}")
        return False
    except Exception as e:
        logger.error(
            f"[SDK] Failed to upload transcript for {session_id}: {e}",
            exc_info=True,
        )
        return False


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
