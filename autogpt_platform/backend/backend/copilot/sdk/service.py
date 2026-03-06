"""Claude Agent SDK service layer for CoPilot chat completions."""

import asyncio
import base64
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

import openai
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from langfuse import propagate_attributes
from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk
from pydantic import BaseModel

from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock
from backend.util.exceptions import NotFoundError
from backend.util.prompt import compress_context
from backend.util.settings import Settings

from ..config import ChatConfig
from ..constants import COPILOT_ERROR_PREFIX, COPILOT_SYSTEM_PREFIX
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
    StreamHeartbeat,
    StreamStart,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
)
from ..service import (
    _build_system_prompt,
    _generate_session_title,
    _is_langfuse_configured,
)
from ..tools.e2b_sandbox import get_or_create_sandbox
from ..tools.sandbox import WORKSPACE_PREFIX, make_session_path
from ..tools.workspace_files import get_manager
from ..tracking import track_user_message
from .compaction import CompactionTracker, filter_compaction_messages
from .response_adapter import SDKResponseAdapter
from .security_hooks import create_security_hooks
from .tool_adapter import (
    create_copilot_mcp_server,
    get_copilot_tool_names,
    get_sdk_disallowed_tools,
    set_execution_context,
    wait_for_stash,
)
from .transcript import (
    cleanup_cli_project_dir,
    download_transcript,
    upload_transcript,
    validate_transcript,
    write_transcript_to_tempfile,
)
from .transcript_builder import TranscriptBuilder

logger = logging.getLogger(__name__)
config = ChatConfig()


def _setup_langfuse_otel() -> None:
    """Configure OTEL tracing for the Claude Agent SDK → Langfuse.

    This uses LangSmith's built-in Claude Agent SDK integration to monkey-patch
    ``ClaudeSDKClient``, capturing every tool call and model turn as OTEL spans.
    Spans are exported via OTLP to Langfuse (or any OTEL-compatible backend).

    To route traces elsewhere, override ``OTEL_EXPORTER_OTLP_ENDPOINT`` and
    ``OTEL_EXPORTER_OTLP_HEADERS`` environment variables — no code changes needed.
    """
    if not _is_langfuse_configured():
        return

    try:
        settings = Settings()
        pk = settings.secrets.langfuse_public_key
        sk = settings.secrets.langfuse_secret_key
        host = settings.secrets.langfuse_host

        # OTEL exporter config — these are only set if not already present,
        # so explicit env-var overrides always win.
        creds = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        os.environ.setdefault("LANGSMITH_OTEL_ENABLED", "true")
        os.environ.setdefault("LANGSMITH_OTEL_ONLY", "true")
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", f"{host}/api/public/otel")
        os.environ.setdefault(
            "OTEL_EXPORTER_OTLP_HEADERS", f"Authorization=Basic {creds}"
        )

        # Set the Langfuse environment via OTEL resource attributes so the
        # Langfuse server maps it to the first-class environment field.
        tracing_env = settings.secrets.langfuse_tracing_environment
        os.environ.setdefault(
            "OTEL_RESOURCE_ATTRIBUTES",
            f"langfuse.environment={tracing_env}",
        )

        configure_claude_agent_sdk(tags=["sdk"])
        logger.info(
            "OTEL tracing configured for Claude Agent SDK → %s [%s]", host, tracing_env
        )
    except Exception:
        logger.warning("OTEL setup skipped — failed to configure", exc_info=True)


_setup_langfuse_otel()


# Set to hold background tasks to prevent garbage collection
_background_tasks: set[asyncio.Task[Any]] = set()


_SDK_CWD_PREFIX = WORKSPACE_PREFIX

# Heartbeat interval — keep SSE alive through proxies/LBs during tool execution.
# IMPORTANT: Must be less than frontend timeout (12s in useCopilotPage.ts)
_HEARTBEAT_INTERVAL = 10.0  # seconds


# Appended to the system prompt to inform the agent about available tools.
# The SDK built-in Bash is NOT available — use mcp__copilot__bash_exec instead,
# which has kernel-level network isolation (unshare --net).
_SHARED_TOOL_NOTES = """\

### Sharing files with the user
After saving a file to the persistent workspace with `write_workspace_file`,
share it with the user by embedding the `download_url` from the response in
your message as a Markdown link or image:

- **Any file** — shows as a clickable download link:
  `[report.csv](workspace://file_id#text/csv)`
- **Image** — renders inline in chat:
  `![chart](workspace://file_id#image/png)`
- **Video** — renders inline in chat with player controls:
  `![recording](workspace://file_id#video/mp4)`

The `download_url` field in the `write_workspace_file` response is already
in the correct format — paste it directly after the `(` in the Markdown.

### Long-running tools
Long-running tools (create_agent, edit_agent, etc.) are handled
asynchronously.  You will receive an immediate response; the actual result
is delivered to the user via a background stream.

### Large tool outputs
When a tool output exceeds the display limit, it is automatically saved to
the persistent workspace.  The truncated output includes a
`<tool-output-truncated>` tag with the workspace path.  Use
`read_workspace_file(path="...", offset=N, length=50000)` to retrieve
additional sections.

### Sub-agent tasks
- When using the Task tool, NEVER set `run_in_background` to true.
  All tasks must run in the foreground.
"""


_LOCAL_TOOL_SUPPLEMENT = (
    """

## Tool notes

### Shell commands
- The SDK built-in Bash tool is NOT available.  Use the `bash_exec` MCP tool
  for shell commands — it runs in a network-isolated sandbox.

### Working directory
- Your working directory is: `{cwd}`
- All SDK Read/Write/Edit/Glob/Grep tools AND `bash_exec` operate inside this
  directory.  This is the ONLY writable path — do not attempt to read or write
  anywhere else on the filesystem.
- Use relative paths or absolute paths under `{cwd}` for all file operations.

### Two storage systems — CRITICAL to understand

1. **Ephemeral working directory** (`{cwd}`):
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
"""
    + _SHARED_TOOL_NOTES
)


_E2B_TOOL_SUPPLEMENT = (
    """

## Tool notes

### Shell commands
- The SDK built-in Bash tool is NOT available.  Use the `bash_exec` MCP tool
  for shell commands — it runs in a cloud sandbox with full internet access.

### Working directory
- Your working directory is: `/home/user` (cloud sandbox)
- All file tools (`read_file`, `write_file`, `edit_file`, `glob`, `grep`)
  AND `bash_exec` operate on the **same cloud sandbox filesystem**.
- Files created by `bash_exec` are immediately visible to `read_file` and
  vice-versa — they share one filesystem.
- Use relative paths (resolved from `/home/user`) or absolute paths.

### Two storage systems — CRITICAL to understand

1. **Cloud sandbox** (`/home/user`):
   - Shared by all file tools AND `bash_exec` — same filesystem
   - Files **persist across turns** within the current session
   - Full Linux environment with internet access
   - Lost when the session expires (12 h inactivity)

2. **Persistent workspace** (cloud storage):
   - Files here **survive across sessions indefinitely**
   - Use `write_workspace_file` to save important files permanently
   - Use `read_workspace_file` to retrieve previously saved files
   - Use `list_workspace_files` to see what files you've saved before
   - Call `list_workspace_files(include_all_sessions=True)` to see files from
     all sessions

### Moving files between sandbox and persistent storage
- **Sandbox → Persistent**: Use `write_workspace_file` with `source_path`
  to copy from the sandbox to permanent storage
- **Persistent → Sandbox**: Use `read_workspace_file` with `save_to_path`
  to download into the sandbox for processing

### File persistence workflow
Important files that must survive beyond this session should be saved with
`write_workspace_file`.  Sandbox files persist across turns but are lost
when the session expires.
"""
    + _SHARED_TOOL_NOTES
)


STREAM_LOCK_PREFIX = "copilot:stream:lock:"


def _resolve_sdk_model() -> str | None:
    """Resolve the model name for the Claude Agent SDK CLI.

    Uses ``config.claude_agent_model`` if set, otherwise derives from
    ``config.model`` by stripping the OpenRouter provider prefix (e.g.,
    ``"anthropic/claude-opus-4.6"`` → ``"claude-opus-4.6"``).

    When ``use_claude_code_subscription`` is enabled and no explicit
    ``claude_agent_model`` is set, returns ``None`` so the CLI uses the
    default model for the user's subscription plan.
    """
    if config.claude_agent_model:
        return config.claude_agent_model
    if config.use_claude_code_subscription:
        return None
    model = config.model
    if "/" in model:
        return model.split("/", 1)[1]
    return model


@functools.cache
def _validate_claude_code_subscription() -> None:
    """Validate Claude CLI is installed and responds to ``--version``.

    Cached so the blocking subprocess check runs at most once per process
    lifetime.  A failure (CLI not installed) is a config error that requires
    a process restart anyway.
    """
    claude_path = shutil.which("claude")
    if not claude_path:
        raise RuntimeError(
            "Claude Code CLI not found. Install it with: "
            "npm install -g @anthropic-ai/claude-code"
        )
    result = subprocess.run(
        [claude_path, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Claude CLI check failed (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    logger.info(
        "Claude Code subscription mode: CLI version %s",
        result.stdout.strip(),
    )


def _build_sdk_env(
    session_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, str]:
    """Build env vars for the SDK CLI process.

    Routes API calls through OpenRouter (or a custom base_url) using
    the same ``config.api_key`` / ``config.base_url`` as the non-SDK path.
    This gives per-call token and cost tracking on the OpenRouter dashboard.

    When *session_id* is provided, an ``x-session-id`` custom header is
    injected via ``ANTHROPIC_CUSTOM_HEADERS`` so that OpenRouter Broadcast
    forwards traces (including cost/usage) to Langfuse for the
    ``/api/v1/messages`` endpoint.

    Only overrides ``ANTHROPIC_API_KEY`` when a valid proxy URL and auth
    token are both present — otherwise returns an empty dict so the SDK
    falls back to its default credentials.
    """
    env: dict[str, str] = {}

    if config.use_claude_code_subscription:
        # Claude Code subscription: let the CLI use its own logged-in auth.
        # Explicitly clear API key env vars so the subprocess doesn't pick
        # them up from the parent process and bypass subscription auth.
        _validate_claude_code_subscription()
        env["ANTHROPIC_API_KEY"] = ""
        env["ANTHROPIC_AUTH_TOKEN"] = ""
        env["ANTHROPIC_BASE_URL"] = ""
    elif config.api_key and config.base_url:
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

    # Inject broadcast headers so OpenRouter forwards traces to Langfuse.
    # The ``x-session-id`` header is *required* for the Anthropic-native
    # ``/messages`` endpoint — without it broadcast silently drops the
    # trace even when org-level Langfuse integration is configured.
    def _safe(value: str) -> str:
        """Strip CR/LF to prevent header injection, then truncate."""
        return value.replace("\r", "").replace("\n", "").strip()[:128]

    headers: list[str] = []
    if session_id:
        headers.append(f"x-session-id: {_safe(session_id)}")
    if user_id:
        headers.append(f"x-user-id: {_safe(user_id)}")
    # Only inject headers when routing through OpenRouter/proxy — they're
    # meaningless (and leak internal IDs) when using subscription mode.
    if headers and env.get("ANTHROPIC_BASE_URL"):
        env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(headers)

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


def _format_sdk_content_blocks(blocks: list) -> list[dict[str, Any]]:
    """Convert SDK content blocks to transcript format.

    Handles TextBlock, ToolUseBlock, ToolResultBlock, and ThinkingBlock.
    Unknown block types are logged and skipped.
    """
    result: list[dict[str, Any]] = []
    for block in blocks or []:
        if isinstance(block, TextBlock):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        elif isinstance(block, ToolResultBlock):
            result.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                }
            )
        elif isinstance(block, ThinkingBlock):
            result.append(
                {
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                }
            )
        else:
            logger.warning(
                f"[SDK] Unknown content block type: {type(block).__name__}. "
                f"This may indicate a new SDK version with additional block types."
            )
    return result


async def _compress_messages(
    messages: list[ChatMessage],
) -> tuple[list[ChatMessage], bool]:
    """Compress a list of messages if they exceed the token threshold.

    Uses the shared compress_context() from prompt.py which supports:
    - LLM summarization of old messages (keeps recent ones intact)
    - Progressive content truncation as fallback
    - Middle-out deletion as last resort
    """
    messages = filter_compaction_messages(messages)

    if len(messages) < 2:
        return messages, False

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
        ], True

    return messages, False


def _format_conversation_context(messages: list[ChatMessage]) -> str | None:
    """Format conversation messages into a context prefix for the user message.

    Includes user messages, assistant text, tool call summaries, and
    tool result summaries so the agent retains full context about what
    tools were invoked and their outcomes.

    Returns None if there are no messages to format.
    """
    if not messages:
        return None

    # Filter out compaction messages first, then format
    messages = filter_compaction_messages(messages)

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
) -> tuple[str, bool]:
    """Build the query message with appropriate context.

    Returns:
        Tuple of (query_message, was_compacted).
    """
    msg_count = len(session.messages)

    if use_resume and transcript_msg_count > 0:
        if transcript_msg_count < msg_count - 1:
            gap = session.messages[transcript_msg_count:-1]
            compressed, was_compressed = await _compress_messages(gap)
            gap_context = _format_conversation_context(compressed)
            if gap_context:
                logger.info(
                    "[SDK] Transcript stale: covers %d of %d messages, "
                    "gap=%d (compressed=%s)",
                    transcript_msg_count,
                    msg_count,
                    len(gap),
                    was_compressed,
                )
                return (
                    f"{gap_context}\n\nNow, the user says:\n{current_message}",
                    was_compressed,
                )
    elif not use_resume and msg_count > 1:
        logger.warning(
            f"[SDK] Using compression fallback for session "
            f"{session_id} ({msg_count} messages) — no transcript for --resume"
        )
        compressed, was_compressed = await _compress_messages(session.messages[:-1])
        history_context = _format_conversation_context(compressed)
        if history_context:
            return (
                f"{history_context}\n\nNow, the user says:\n{current_message}",
                was_compressed,
            )

    return current_message, False


# Claude API vision-supported image types.
_VISION_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

# Max size for embedding images directly in the user message (20 MiB raw).
_MAX_INLINE_IMAGE_BYTES = 20 * 1024 * 1024

# Matches characters unsafe for filenames.
_UNSAFE_FILENAME = re.compile(r"[^\w.\-]")


def _save_to_sdk_cwd(sdk_cwd: str, filename: str, content: bytes) -> str:
    """Write file content to the SDK ephemeral directory.

    Returns the absolute path.  Adds a numeric suffix on name collisions.
    """
    safe = _UNSAFE_FILENAME.sub("_", filename) or "file"
    candidate = os.path.join(sdk_cwd, safe)
    if os.path.exists(candidate):
        stem, ext = os.path.splitext(safe)
        idx = 1
        while os.path.exists(candidate):
            candidate = os.path.join(sdk_cwd, f"{stem}_{idx}{ext}")
            idx += 1
    with open(candidate, "wb") as f:
        f.write(content)
    return candidate


class PreparedAttachments(BaseModel):
    """Result of preparing file attachments for a query."""

    hint: str = ""
    """Text hint describing the files (appended to the user message)."""

    image_blocks: list[dict[str, Any]] = []
    """Claude API image content blocks to embed in the user message."""


async def _prepare_file_attachments(
    file_ids: list[str],
    user_id: str,
    session_id: str,
    sdk_cwd: str,
) -> PreparedAttachments:
    """Download workspace files and prepare them for Claude.

    Images (PNG/JPEG/GIF/WebP) are embedded directly as vision content blocks
    in the user message so Claude can see them without tool calls.

    Non-image files (PDFs, text, etc.) are saved to *sdk_cwd* so the CLI's
    built-in Read tool can access them.

    Returns a :class:`PreparedAttachments` with a text hint and any image
    content blocks.
    """
    empty = PreparedAttachments(hint="", image_blocks=[])
    if not file_ids or not user_id:
        return empty

    try:
        manager = await get_manager(user_id, session_id)
    except Exception:
        logger.warning(
            "Failed to create workspace manager for file attachments",
            exc_info=True,
        )
        return empty

    image_blocks: list[dict[str, Any]] = []
    file_descriptions: list[str] = []

    for fid in file_ids:
        try:
            file_info = await manager.get_file_info(fid)
            if file_info is None:
                continue
            content = await manager.read_file_by_id(fid)
            mime = (file_info.mime_type or "").split(";")[0].strip().lower()

            # Images: embed directly in the user message as vision blocks
            if mime in _VISION_MIME_TYPES and len(content) <= _MAX_INLINE_IMAGE_BYTES:
                b64 = base64.b64encode(content).decode("ascii")
                image_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    }
                )
                file_descriptions.append(
                    f"- {file_info.name} ({mime}, "
                    f"{file_info.size_bytes:,} bytes) [embedded as image]"
                )
            else:
                # Non-image files: save to sdk_cwd for Read tool access
                local_path = _save_to_sdk_cwd(sdk_cwd, file_info.name, content)
                file_descriptions.append(
                    f"- {file_info.name} ({mime}, "
                    f"{file_info.size_bytes:,} bytes) saved to {local_path}"
                )
        except Exception:
            logger.warning("Failed to prepare file %s", fid[:12], exc_info=True)

    if not file_descriptions:
        return empty

    noun = "file" if len(file_descriptions) == 1 else "files"
    has_non_images = len(file_descriptions) > len(image_blocks)
    read_hint = " Use the Read tool to view non-image files." if has_non_images else ""
    hint = (
        f"[The user attached {len(file_descriptions)} {noun}.{read_hint}\n"
        + "\n".join(file_descriptions)
        + "]"
    )
    return PreparedAttachments(hint=hint, image_blocks=image_blocks)


async def stream_chat_completion_sdk(
    session_id: str,
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    session: ChatSession | None = None,
    file_ids: list[str] | None = None,
    **_kwargs: Any,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Stream chat completion using Claude Agent SDK.

    Args:
        file_ids: Optional workspace file IDs attached to the user's message.
            Images are embedded as vision content blocks; other files are
            saved to the SDK working directory for the Read tool.
    """

    if session is None:
        session = await get_chat_session(session_id, user_id)

    if not session:
        raise NotFoundError(
            f"Session {session_id} not found. Please create a new session first."
        )

    # Type narrowing: session is guaranteed ChatSession after the check above
    session = cast(ChatSession, session)

    # Clean up stale error markers from previous turn before starting new turn
    # If the last message contains an error marker, remove it (user is retrying)
    if (
        len(session.messages) > 0
        and session.messages[-1].role == "assistant"
        and session.messages[-1].content
        and COPILOT_ERROR_PREFIX in session.messages[-1].content
    ):
        logger.info(
            "[SDK] [%s] Removing stale error marker from previous turn",
            session_id[:12],
        )
        session.messages.pop()

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

    # Structured log prefix: [SDK][<session>][T<turn>]
    # Turn = number of user messages (1-based), computed AFTER appending the new message.
    turn = sum(1 for m in session.messages if m.role == "user")
    log_prefix = f"[SDK][{session_id[:12]}][T{turn}]"

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

    message_id = str(uuid.uuid4())
    stream_id = str(uuid.uuid4())
    stream_completed = False
    ended_with_stream_error = False
    e2b_sandbox = None
    use_resume = False
    resume_file: str | None = None
    transcript_builder = TranscriptBuilder()
    sdk_cwd = ""

    # Acquire stream lock to prevent concurrent streams to the same session
    lock = AsyncClusterLock(
        redis=await get_redis_async(),
        key=f"{STREAM_LOCK_PREFIX}{session_id}",
        owner_id=stream_id,
        timeout=config.stream_lock_ttl,
    )

    lock_owner = await lock.try_acquire()
    if lock_owner != stream_id:
        # Another stream is active
        logger.warning(
            f"{log_prefix} Session already has an active stream: {lock_owner}"
        )
        yield StreamError(
            errorText="Another stream is already active for this session. "
            "Please wait or stop it.",
            code="stream_already_active",
        )
        return

    # OTEL context manager — initialized inside the try and cleaned up in finally.
    _otel_ctx: Any = None

    # Make sure there is no more code between the lock acquisition and try-block.
    try:
        # Build system prompt (reuses non-SDK path with Langfuse support).
        # Pre-compute the cwd here so the exact working directory path can be
        # injected into the supplement instead of the generic placeholder.
        # Catch ValueError early so the failure yields a clean StreamError rather
        # than propagating outside the stream error-handling path.
        has_history = len(session.messages) > 1
        try:
            sdk_cwd = _make_sdk_cwd(session_id)
            os.makedirs(sdk_cwd, exist_ok=True)
        except (ValueError, OSError) as e:
            logger.error("%s Invalid SDK cwd: %s", log_prefix, e)
            yield StreamError(
                errorText="Unable to initialize working directory.",
                code="sdk_cwd_error",
            )
            return
        # --- Run independent async I/O operations in parallel ---
        # E2B sandbox setup, system prompt build (Langfuse + DB), and transcript
        # download are independent network calls.  Running them concurrently
        # saves ~200-500ms compared to sequential execution.

        async def _setup_e2b():
            """Set up E2B sandbox if configured, return sandbox or None."""
            if config.use_e2b_sandbox and not config.e2b_api_key:
                logger.warning(
                    "[E2B] [%s] E2B sandbox enabled but no API key configured "
                    "(CHAT_E2B_API_KEY / E2B_API_KEY) — falling back to bubblewrap",
                    session_id[:12],
                )
                return None
            if config.use_e2b_sandbox and config.e2b_api_key:
                try:
                    return await get_or_create_sandbox(
                        session_id,
                        api_key=config.e2b_api_key,
                        template=config.e2b_sandbox_template,
                        timeout=config.e2b_sandbox_timeout,
                    )
                except Exception as e2b_err:
                    logger.error(
                        "[E2B] [%s] Setup failed: %s",
                        session_id[:12],
                        e2b_err,
                        exc_info=True,
                    )
            return None

        async def _fetch_transcript():
            """Download transcript for --resume if applicable."""
            if not (
                config.claude_agent_use_resume and user_id and len(session.messages) > 1
            ):
                return None
            try:
                return await download_transcript(
                    user_id, session_id, log_prefix=log_prefix
                )
            except Exception as transcript_err:
                logger.warning(
                    "%s Transcript download failed, continuing without " "--resume: %s",
                    log_prefix,
                    transcript_err,
                )
                return None

        e2b_sandbox, (base_system_prompt, _), dl = await asyncio.gather(
            _setup_e2b(),
            _build_system_prompt(user_id, has_conversation_history=has_history),
            _fetch_transcript(),
        )

        use_e2b = e2b_sandbox is not None
        system_prompt = base_system_prompt + (
            _E2B_TOOL_SUPPLEMENT
            if use_e2b
            else _LOCAL_TOOL_SUPPLEMENT.format(cwd=sdk_cwd)
        )

        # Process transcript download result
        transcript_msg_count = 0
        if dl:
            is_valid = validate_transcript(dl.content)
            dl_lines = dl.content.strip().split("\n") if dl.content else []
            logger.info(
                "%s Downloaded transcript: %dB, %d lines, " "msg_count=%d, valid=%s",
                log_prefix,
                len(dl.content),
                len(dl_lines),
                dl.message_count,
                is_valid,
            )
            if is_valid:
                # Load previous FULL context into builder
                transcript_builder.load_previous(dl.content, log_prefix=log_prefix)
                resume_file = write_transcript_to_tempfile(
                    dl.content, session_id, sdk_cwd
                )
                if resume_file:
                    use_resume = True
                    transcript_msg_count = dl.message_count
                    logger.debug(
                        f"{log_prefix} Using --resume ({len(dl.content)}B, "
                        f"msg_count={transcript_msg_count})"
                    )
            else:
                logger.warning(f"{log_prefix} Transcript downloaded but invalid")
        elif config.claude_agent_use_resume and user_id and len(session.messages) > 1:
            logger.warning(
                f"{log_prefix} No transcript available "
                f"({len(session.messages)} messages in session)"
            )

        yield StreamStart(messageId=message_id, sessionId=session_id)

        set_execution_context(user_id, session, sandbox=e2b_sandbox, sdk_cwd=sdk_cwd)

        # Fail fast when no API credentials are available at all.
        sdk_env = _build_sdk_env(session_id=session_id, user_id=user_id)
        if not config.api_key and not config.use_claude_code_subscription:
            raise RuntimeError(
                "No API key configured. Set OPEN_ROUTER_API_KEY, "
                "CHAT_API_KEY, or ANTHROPIC_API_KEY for API access, "
                "or CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=true to use "
                "Claude Code CLI subscription (requires `claude login`)."
            )

        mcp_server = create_copilot_mcp_server(use_e2b=use_e2b)

        sdk_model = _resolve_sdk_model()

        # Track SDK-internal compaction (PreCompact hook → start, next msg → end)
        compaction = CompactionTracker()

        security_hooks = create_security_hooks(
            user_id,
            sdk_cwd=sdk_cwd,
            max_subtasks=config.claude_agent_max_subtasks,
            on_compact=compaction.on_compact,
        )

        allowed = get_copilot_tool_names(use_e2b=use_e2b)
        disallowed = get_sdk_disallowed_tools(use_e2b=use_e2b)
        sdk_options_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "mcp_servers": {"copilot": mcp_server},
            "allowed_tools": allowed,
            "disallowed_tools": disallowed,
            "hooks": security_hooks,
            "cwd": sdk_cwd,
            "max_buffer_size": config.claude_agent_max_buffer_size,
        }
        if sdk_model:
            sdk_options_kwargs["model"] = sdk_model
        if sdk_env:
            sdk_options_kwargs["env"] = sdk_env
        if use_resume and resume_file:
            sdk_options_kwargs["resume"] = resume_file

        options = ClaudeAgentOptions(**sdk_options_kwargs)  # type: ignore[arg-type]  # dynamic kwargs

        adapter = SDKResponseAdapter(message_id=message_id, session_id=session_id)

        # Propagate user_id/session_id as OTEL context attributes so the
        # langsmith tracing integration attaches them to every span.  This
        # is what Langfuse (or any OTEL backend) maps to its native
        # user/session fields.
        _otel_ctx = propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            trace_name="copilot-sdk",
            tags=["sdk"],
            metadata={
                "resume": str(use_resume),
                "conversation_turn": str(turn),
            },
        )
        _otel_ctx.__enter__()

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
                return

            query_message, was_compacted = await _build_query_message(
                current_message,
                session,
                use_resume,
                transcript_msg_count,
                session_id,
            )
            # If files are attached, prepare them: images become vision
            # content blocks in the user message, other files go to sdk_cwd.
            attachments = await _prepare_file_attachments(
                file_ids or [], user_id or "", session_id, sdk_cwd
            )
            if attachments.hint:
                query_message = f"{query_message}\n\n{attachments.hint}"

            logger.info(
                "%s Sending query — resume=%s, total_msgs=%d, "
                "query_len=%d, attached_files=%d, image_blocks=%d",
                log_prefix,
                use_resume,
                len(session.messages),
                len(query_message),
                len(file_ids) if file_ids else 0,
                len(attachments.image_blocks),
            )

            compaction.reset_for_query()
            if was_compacted:
                for ev in compaction.emit_pre_query(session):
                    yield ev

            if attachments.image_blocks:
                # Build multimodal content: image blocks + text
                content_blocks: list[dict[str, Any]] = [
                    *attachments.image_blocks,
                    {"type": "text", "text": query_message},
                ]
                user_msg = {
                    "type": "user",
                    "message": {"role": "user", "content": content_blocks},
                    "parent_tool_use_id": None,
                    "session_id": session_id,
                }
                assert client._transport is not None  # noqa: SLF001
                await client._transport.write(  # noqa: SLF001
                    json.dumps(user_msg) + "\n"
                )
                # Capture user message in transcript (multimodal)
                transcript_builder.add_user_message(content=content_blocks)
            else:
                await client.query(query_message, session_id=session_id)
                # Capture actual user message in transcript (not the engineered query)
                # query_message may include context wrappers, but transcript needs raw input
                transcript_builder.add_user_message(content=current_message)

            assistant_response = ChatMessage(role="assistant", content="")
            accumulated_tool_calls: list[dict[str, Any]] = []
            has_appended_assistant = False
            has_tool_results = False
            ended_with_stream_error = False

            # Use an explicit async iterator with non-cancelling heartbeats.
            # CRITICAL: we must NOT cancel __anext__() mid-flight — doing so
            # (via asyncio.timeout or wait_for) corrupts the SDK's internal
            # anyio memory stream, causing StopAsyncIteration on the next
            # call and silently dropping all in-flight tool results.
            # Instead, wrap __anext__() in a Task and use asyncio.wait()
            # with a timeout.  On timeout we emit a heartbeat but keep the
            # Task alive so it can deliver the next message.
            msg_iter = client.receive_response().__aiter__()
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
                        await lock.refresh()
                        for ev in compaction.emit_start_if_ready():
                            yield ev
                        yield StreamHeartbeat()
                        continue

                    # Task completed — get result
                    pending_task = None
                    try:
                        sdk_msg = done.pop().result()
                    except StopAsyncIteration:
                        logger.info(
                            "%s Stream ended normally (StopAsyncIteration)",
                            log_prefix,
                        )
                        break
                    except Exception as stream_err:
                        # SDK sends {"type": "error"} which raises
                        # Exception in receive_response() — capture it
                        # so the session can still be saved and the
                        # frontend gets a clean finish.
                        logger.error(
                            "%s Stream error from SDK: %s",
                            log_prefix,
                            stream_err,
                            exc_info=True,
                        )
                        ended_with_stream_error = True
                        yield StreamError(
                            errorText=f"SDK stream error: {stream_err}",
                            code="sdk_stream_error",
                        )
                        break

                    logger.info(
                        "%s Received: %s %s "
                        "(unresolved=%d, current=%d, resolved=%d)",
                        log_prefix,
                        type(sdk_msg).__name__,
                        getattr(sdk_msg, "subtype", ""),
                        len(adapter.current_tool_calls)
                        - len(adapter.resolved_tool_calls),
                        len(adapter.current_tool_calls),
                        len(adapter.resolved_tool_calls),
                    )

                    # Capture SDK messages in transcript
                    if isinstance(sdk_msg, AssistantMessage):
                        content_blocks = _format_sdk_content_blocks(sdk_msg.content)
                        model_name = getattr(sdk_msg, "model", "")
                        transcript_builder.add_assistant_message(
                            content_blocks=content_blocks,
                            model=model_name,
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
                                "%s Timed out waiting for "
                                "PostToolUse hook stash "
                                "(%d unresolved tool calls)",
                                log_prefix,
                                len(adapter.current_tool_calls)
                                - len(adapter.resolved_tool_calls),
                            )

                    # Log ResultMessage details for debugging
                    if isinstance(sdk_msg, ResultMessage):
                        logger.info(
                            "%s Received: ResultMessage %s "
                            "(unresolved=%d, current=%d, resolved=%d)",
                            log_prefix,
                            sdk_msg.subtype,
                            len(adapter.current_tool_calls)
                            - len(adapter.resolved_tool_calls),
                            len(adapter.current_tool_calls),
                            len(adapter.resolved_tool_calls),
                        )
                        if sdk_msg.subtype in ("error", "error_during_execution"):
                            logger.error(
                                "%s SDK execution failed with error: %s",
                                log_prefix,
                                sdk_msg.result or "(no error message provided)",
                            )

                    # Emit compaction end if SDK finished compacting
                    for ev in await compaction.emit_end_if_ready(session):
                        yield ev

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
                                "%s Tool event: %s, tool=%s%s",
                                log_prefix,
                                type(response).__name__,
                                getattr(response, "toolName", "N/A"),
                                extra,
                            )

                        # Log errors being sent to frontend
                        if isinstance(response, StreamError):
                            logger.error(
                                "%s Sending error to frontend: %s (code=%s)",
                                log_prefix,
                                response.errorText,
                                response.code,
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

            except asyncio.CancelledError:
                # Task/generator was cancelled (e.g. client disconnect,
                # server shutdown).  Log and let the safety-net / finally
                # blocks handle cleanup.
                logger.warning(
                    "%s Streaming loop cancelled (asyncio.CancelledError)",
                    log_prefix,
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
                        # Expected: task was cancelled or exhausted during cleanup
                        logger.info(
                            "%s Pending __anext__ task completed during cleanup",
                            log_prefix,
                        )

            # Safety net: if tools are still unresolved after the
            # streaming loop (e.g. StopAsyncIteration before ResultMessage,
            # or SDK not sending UserMessages for built-in tools), flush
            # them now so the frontend stops showing spinners.
            if adapter.has_unresolved_tool_calls:
                logger.warning(
                    "%s %d unresolved tool(s) after stream loop — "
                    "flushing as safety net",
                    log_prefix,
                    len(adapter.current_tool_calls) - len(adapter.resolved_tool_calls),
                )
                safety_responses: list[StreamBaseResponse] = []
                adapter._flush_unresolved_tool_calls(safety_responses)
                for response in safety_responses:
                    if isinstance(
                        response,
                        (StreamToolInputAvailable, StreamToolOutputAvailable),
                    ):
                        logger.info(
                            "%s Safety flush: %s, tool=%s",
                            log_prefix,
                            type(response).__name__,
                            getattr(response, "toolName", "N/A"),
                        )
                    yield response

            # If the stream ended without a ResultMessage, the SDK
            # CLI exited unexpectedly or the user stopped execution.
            # Close any open text/step so chunks are well-formed, and
            # append a cancellation message so users see feedback.
            # StreamFinish is published by mark_session_completed in the processor.
            if not stream_completed and not ended_with_stream_error:
                logger.info(
                    "%s Stream ended without ResultMessage (stopped by user)",
                    log_prefix,
                )
                closing_responses: list[StreamBaseResponse] = []
                adapter._end_text_if_open(closing_responses)
                for r in closing_responses:
                    yield r

                # Add "Stopped by user" message so it persists after refresh
                # Use COPILOT_SYSTEM_PREFIX so frontend renders it as system message, not assistant
                session.messages.append(
                    ChatMessage(
                        role="assistant",
                        content=f"{COPILOT_SYSTEM_PREFIX} Execution stopped by user",
                    )
                )

            if (
                assistant_response.content or assistant_response.tool_calls
            ) and not has_appended_assistant:
                session.messages.append(assistant_response)

        # Transcript upload is handled exclusively in the finally block
        # to avoid double-uploads (the success path used to upload the
        # old resume file, then the finally block overwrote it with the
        # stop hook content — which could be smaller after compaction).

        if ended_with_stream_error:
            logger.warning(
                "%s Stream ended with SDK error after %d messages",
                log_prefix,
                len(session.messages),
            )
        else:
            logger.info(
                "%s Stream completed successfully with %d messages",
                log_prefix,
                len(session.messages),
            )
    except BaseException as e:
        # Catch BaseException to handle both Exception and CancelledError
        # (CancelledError inherits from BaseException in Python 3.8+)
        if isinstance(e, asyncio.CancelledError):
            logger.warning("%s Session cancelled", log_prefix)
            error_msg = "Operation cancelled"
        else:
            error_msg = str(e) or type(e).__name__
            # SDK cleanup RuntimeError is expected during cancellation, log as warning
            if isinstance(e, RuntimeError) and "cancel scope" in str(e):
                logger.warning("%s SDK cleanup error: %s", log_prefix, error_msg)
            else:
                logger.error("%s Error: %s", log_prefix, error_msg, exc_info=True)

        # Append error marker to session (non-invasive text parsing approach)
        # The finally block will persist the session with this error marker
        if session:
            session.messages.append(
                ChatMessage(
                    role="assistant", content=f"{COPILOT_ERROR_PREFIX} {error_msg}"
                )
            )
            logger.debug(
                "%s Appended error marker, will be persisted in finally",
                log_prefix,
            )

        # Yield StreamError for immediate feedback (only for non-cancellation errors)
        # Skip for CancelledError and RuntimeError cleanup issues (both are cancellations)
        is_cancellation = isinstance(e, asyncio.CancelledError) or (
            isinstance(e, RuntimeError) and "cancel scope" in str(e)
        )
        if not is_cancellation:
            yield StreamError(
                errorText=error_msg,
                code="sdk_error",
            )

        raise
    finally:
        # --- Close OTEL context ---
        if _otel_ctx is not None:
            try:
                _otel_ctx.__exit__(*sys.exc_info())
            except Exception:
                logger.warning("OTEL context teardown failed", exc_info=True)

        # --- Persist session messages ---
        # This MUST run in finally to persist messages even when the generator
        # is stopped early (e.g., user clicks stop, processor breaks stream loop).
        # Without this, messages disappear after refresh because they were never
        # saved to the database.
        if session is not None:
            try:
                await asyncio.shield(upsert_chat_session(session))
                logger.info(
                    "%s Session persisted in finally with %d messages",
                    log_prefix,
                    len(session.messages),
                )
            except Exception as persist_err:
                logger.error(
                    "%s Failed to persist session in finally: %s",
                    log_prefix,
                    persist_err,
                    exc_info=True,
                )

        # --- Upload transcript for next-turn --resume ---
        # This MUST run in finally so the transcript is uploaded even when
        # the streaming loop raises an exception.
        # The transcript represents the COMPLETE active context (atomic).
        if config.claude_agent_use_resume and user_id and session is not None:
            try:
                # Build complete transcript from captured SDK messages
                transcript_content = transcript_builder.to_jsonl()

                if not transcript_content:
                    logger.warning(
                        "%s No transcript to upload (builder empty)", log_prefix
                    )
                elif not validate_transcript(transcript_content):
                    logger.warning(
                        "%s Transcript invalid, skipping upload (entries=%d)",
                        log_prefix,
                        transcript_builder.entry_count,
                    )
                else:
                    logger.info(
                        "%s Uploading complete transcript (entries=%d, bytes=%d)",
                        log_prefix,
                        transcript_builder.entry_count,
                        len(transcript_content),
                    )
                    # Shield upload from cancellation - let it complete even if
                    # the finally block is interrupted. No timeout to avoid race
                    # conditions where backgrounded uploads overwrite newer transcripts.
                    await asyncio.shield(
                        upload_transcript(
                            user_id=user_id,
                            session_id=session_id,
                            content=transcript_content,
                            message_count=len(session.messages),
                            log_prefix=log_prefix,
                        )
                    )
            except Exception as upload_err:
                logger.error(
                    "%s Transcript upload failed in finally: %s",
                    log_prefix,
                    upload_err,
                    exc_info=True,
                )

        if sdk_cwd:
            _cleanup_sdk_tool_results(sdk_cwd)

        # Release stream lock to allow new streams for this session
        await lock.release()


async def _update_title_async(
    session_id: str, message: str, user_id: str | None = None
) -> None:
    """Background task to update session title."""
    try:
        title = await _generate_session_title(
            message, user_id=user_id, session_id=session_id
        )
        if title and user_id:
            await update_session_title(session_id, title)
            logger.debug(f"[SDK] Generated title for {session_id}: {title}")
    except Exception as e:
        logger.warning(f"[SDK] Failed to update session title: {e}")
