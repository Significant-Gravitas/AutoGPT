"""Claude Agent SDK service layer for CoPilot chat completions."""

import asyncio
import base64
import json
import logging
import os
import random
import re
import shutil
import sys
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

if TYPE_CHECKING:
    from backend.copilot.permissions import CopilotPermissions

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
from opentelemetry import trace as otel_trace
from pydantic import BaseModel

from backend.copilot.context import get_workspace_manager
from backend.copilot.pending_messages import (
    drain_pending_messages,
    format_pending_as_user_message,
)
from backend.copilot.permissions import apply_tool_permissions
from backend.copilot.rate_limit import get_user_tier
from backend.copilot.transcript import (
    _run_compression,
    cleanup_stale_project_dirs,
    compact_transcript,
    download_transcript,
    read_compacted_entries,
    upload_transcript,
    validate_transcript,
    write_transcript_to_tempfile,
)
from backend.copilot.transcript_builder import TranscriptBuilder
from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock
from backend.util.exceptions import NotFoundError
from backend.util.settings import Settings

from ..config import ChatConfig, CopilotMode
from ..constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
    COPILOT_SYSTEM_PREFIX,
    FRIENDLY_TRANSIENT_MSG,
    is_transient_api_error,
)
from ..context import encode_cwd_for_cli
from ..graphiti.config import is_enabled_for_user
from ..model import (
    ChatMessage,
    ChatSession,
    get_chat_session,
    maybe_append_user_message,
    upsert_chat_session,
)
from ..prompting import get_graphiti_supplement, get_sdk_supplement
from ..response_model import (
    StreamBaseResponse,
    StreamError,
    StreamFinish,
    StreamFinishStep,
    StreamHeartbeat,
    StreamStart,
    StreamStartStep,
    StreamStatus,
    StreamTextDelta,
    StreamToolInputAvailable,
    StreamToolInputStart,
    StreamToolOutputAvailable,
    StreamUsage,
)
from ..service import (
    _build_system_prompt,
    _is_langfuse_configured,
    _update_title_async,
    inject_user_context,
    strip_user_context_tags,
)
from ..token_tracking import persist_and_record_usage
from ..tools.e2b_sandbox import get_or_create_sandbox, pause_sandbox_direct
from ..tools.sandbox import WORKSPACE_PREFIX, make_session_path
from ..tracking import track_user_message
from .compaction import CompactionTracker, filter_compaction_messages
from .env import build_sdk_env  # noqa: F401 — re-export for backward compat
from .response_adapter import SDKResponseAdapter
from .security_hooks import create_security_hooks
from .tool_adapter import (
    create_copilot_mcp_server,
    get_copilot_tool_names,
    get_sdk_disallowed_tools,
    reset_stash_event,
    reset_tool_failure_counters,
    set_execution_context,
    wait_for_stash,
)

logger = logging.getLogger(__name__)
config = ChatConfig()


# On context-size errors the SDK query is retried with progressively
# less context: (1) original transcript → (2) compacted transcript →
# (3) no transcript (DB messages only).
# Non-context errors (network, auth, rate-limit) are NOT retried.
_MAX_STREAM_ATTEMPTS = 3

# Hard circuit breaker: abort the stream if the model sends this many
# consecutive tool calls with empty parameters (a sign of context
# saturation or serialization failure).  The MCP wrapper now returns
# guidance on the first empty call, giving the model a chance to
# self-correct.  The limit is generous to allow recovery attempts.
_EMPTY_TOOL_CALL_LIMIT = 5

# User-facing error shown when the empty-tool-call circuit breaker trips.
_CIRCUIT_BREAKER_ERROR_MSG = (
    "AutoPilot was unable to complete the tool call "
    "— this usually happens when the response is "
    "too large to fit in a single tool call. "
    "Try breaking your request into smaller parts."
)

# Idle timeout: abort the stream if no meaningful SDK message (only heartbeats)
# arrives for this many seconds. This catches hung tool calls (e.g. WebSearch
# hanging on a search provider that never responds).
_IDLE_TIMEOUT_SECONDS = 10 * 60  # 10 minutes

# Event types that are ephemeral / cosmetic and must NOT be counted toward
# ``events_yielded`` in the transient-retry loop.  Counting them would prevent
# the backoff retry from firing because ``_next_transient_backoff`` returns
# ``None`` when ``events_yielded > 0``.
_EPHEMERAL_EVENT_TYPES = (
    StreamHeartbeat,
    # Compaction UI events are cosmetic and must not block retry — they're
    # emitted before the SDK query on compacted attempts.
    StreamStartStep,
    StreamFinishStep,
    StreamToolInputStart,
    StreamToolInputAvailable,
    StreamToolOutputAvailable,
    # Transient StreamError and StreamStatus are ephemeral notifications,
    # not content.  Counting them would prevent the backoff retry from
    # firing because _next_transient_backoff() returns None when
    # events_yielded > 0.
    StreamError,
    StreamStatus,
)

# Patterns that indicate the prompt/request exceeds the model's context limit.
# Matched case-insensitively against the full exception chain.
_PROMPT_TOO_LONG_PATTERNS: tuple[str, ...] = (
    "prompt is too long",
    "request too large",
    "maximum context length",
    "context_length_exceeded",
    "input tokens exceed",
    "input is too long",
    "content length exceeds",
)


# Map raw SDK error patterns to user-friendly messages.
# Matched case-insensitively; first match wins.
_FRIENDLY_ERROR_MAP: tuple[tuple[str, str], ...] = (
    ("authentication", "Authentication failed. Please check your API key."),
    ("invalid api key", "Authentication failed. Please check your API key."),
    ("unauthorized", "Authentication failed. Please check your API key."),
    ("rate limit", "Rate limit exceeded. Please wait a moment and try again."),
    ("overloaded", "The AI service is currently overloaded. Please try again shortly."),
    ("server error", "The AI service encountered an internal error. Please retry."),
    ("timeout", "The request timed out. Please try again."),
    ("connection", "Connection error. Please check your network and retry."),
)


def _friendly_error_text(raw: str) -> str:
    """Map a raw SDK error string to a user-friendly message.

    Returns the mapped message if a known pattern is found, otherwise
    returns a generic sanitized version of the raw error.
    """
    lower = raw.lower()
    for pattern, friendly in _FRIENDLY_ERROR_MAP:
        if pattern in lower:
            return friendly
    # Fallback: sanitize but keep the original text for debugging
    return f"SDK stream error: {raw}"


def _is_prompt_too_long(err: BaseException) -> bool:
    """Return True if *err* indicates the prompt exceeds the model's limit.

    Walks the exception chain (`__cause__` / `__context__`) so that
    wrapped errors are detected too.
    """
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        msg = str(current).lower()
        if any(p in msg for p in _PROMPT_TOO_LONG_PATTERNS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_sdk_disconnect_error(exc: BaseException) -> bool:
    """Return True if *exc* is an expected SDK cleanup error from client disconnect.

    Two known patterns occur when ``GeneratorExit`` tears down the async
    generator and the SDK's ``__aexit__`` runs in a different context/task:

    * ``RuntimeError``: cancel scope exited in wrong task (anyio)
    * ``ValueError``: ContextVar token created in a different Context (OTEL)

    These are suppressed to avoid polluting Sentry with non-actionable noise.
    """
    if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
        return True
    if isinstance(exc, ValueError) and "was created in a different Context" in str(exc):
        return True
    return False


def _is_tool_only_message(sdk_msg: object) -> bool:
    """Return True if *sdk_msg* is an AssistantMessage containing only ToolUseBlocks.

    Such a message represents a parallel tool-call batch (no text output yet).
    The ``bool(…content)`` guard prevents vacuous-truth evaluation on an empty list.
    """
    return (
        isinstance(sdk_msg, AssistantMessage)
        and bool(sdk_msg.content)
        and all(isinstance(b, ToolUseBlock) for b in sdk_msg.content)
    )


class ReducedContext(NamedTuple):
    builder: TranscriptBuilder
    use_resume: bool
    resume_file: str | None
    transcript_lost: bool
    tried_compaction: bool


@dataclass
class _TokenUsage:
    """Token usage accumulators for a single turn.

    Separated from `_RetryState` because usage is reset between retry
    attempts independently of the retry-control fields, and is read by
    the outer `stream_chat_completion_sdk` scope after the retry loop
    completes.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float | None = None

    def reset(self) -> None:
        """Reset all accumulators for a new attempt."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cache_read_tokens = 0
        self.cache_creation_tokens = 0
        self.cost_usd = None


@dataclass
class _RetryState:
    """Mutable state passed to `_run_stream_attempt` instead of closures.

    Holds values that the retry loop modifies between attempts so the inner
    generator does not rely on reassigning closure variables.
    """

    options: ClaudeAgentOptions
    query_message: str
    was_compacted: bool
    use_resume: bool
    resume_file: str | None
    transcript_msg_count: int
    adapter: SDKResponseAdapter
    transcript_builder: TranscriptBuilder
    usage: _TokenUsage


@dataclass
class _StreamContext:
    """Per-request variables shared across all retry attempts.

    Extracted so that `_run_stream_attempt` can be a module-level function
    rather than a closure, making it independently testable and reducing the
    cognitive load of the 970-line `stream_chat_completion_sdk` function.

    Scalar fields (IDs, paths, the message string) are set once before the
    retry loop and never reassigned.  `session`, `compaction`, and
    `lock` are **shared mutable references** whose internal state is
    modified by both the retry loop and `_run_stream_attempt` (e.g.
    `session.messages` is rolled back on retry, `compaction` tracks
    mid-stream compaction events).  Their *references* are constant even
    though the objects they point to are mutated.
    """

    session: ChatSession
    session_id: str
    log_prefix: str
    sdk_cwd: str
    current_message: str
    file_ids: list[str] | None
    message_id: str
    attachments: "PreparedAttachments"
    compaction: CompactionTracker
    lock: AsyncClusterLock


async def _reduce_context(
    transcript_content: str,
    tried_compaction: bool,
    session_id: str,
    sdk_cwd: str,
    log_prefix: str,
) -> ReducedContext:
    """Prepare reduced context for a retry attempt.

    On the first retry, compacts the transcript via LLM summarization.
    On subsequent retries (or if compaction fails), drops the transcript
    entirely so the query is rebuilt from DB messages only.

    `transcript_lost` is True when the transcript was dropped (caller
    should set `skip_transcript_upload`).
    """
    # First retry: try compacting
    if transcript_content and not tried_compaction:
        compacted = await compact_transcript(
            transcript_content, model=config.model, log_prefix=log_prefix
        )
        if (
            compacted
            and compacted != transcript_content
            and validate_transcript(compacted)
        ):
            logger.info("%s Using compacted transcript for retry", log_prefix)
            tb = TranscriptBuilder()
            tb.load_previous(compacted, log_prefix=log_prefix)
            resume_file = await asyncio.to_thread(
                write_transcript_to_tempfile, compacted, session_id, sdk_cwd
            )
            if resume_file:
                return ReducedContext(tb, True, resume_file, False, True)
            logger.warning("%s Failed to write compacted transcript", log_prefix)
        logger.warning("%s Compaction failed, dropping transcript", log_prefix)

    # Subsequent retry or compaction failed: drop transcript entirely
    logger.warning("%s Dropping transcript, rebuilding from DB messages", log_prefix)
    return ReducedContext(TranscriptBuilder(), False, None, True, True)


def _append_error_marker(
    session: ChatSession | None,
    display_msg: str,
    *,
    retryable: bool = False,
) -> None:
    """Append a copilot error marker to *session* so it persists across refresh.

    Args:
        session: The chat session to append to (no-op if `None`).
        display_msg: User-visible error text.
        retryable: If `True`, use the retryable prefix so the frontend
            shows a "Try Again" button.
    """
    if session is None:
        return
    prefix = COPILOT_RETRYABLE_ERROR_PREFIX if retryable else COPILOT_ERROR_PREFIX
    session.messages.append(
        ChatMessage(role="assistant", content=f"{prefix} {display_msg}")
    )


def _setup_langfuse_otel() -> None:
    """Configure OTEL tracing for the Claude Agent SDK → Langfuse.

    This uses LangSmith's built-in Claude Agent SDK integration to monkey-patch
    `ClaudeSDKClient`, capturing every tool call and model turn as OTEL spans.
    Spans are exported via OTLP to Langfuse (or any OTEL-compatible backend).

    To route traces elsewhere, override `OTEL_EXPORTER_OTLP_ENDPOINT` and
    `OTEL_EXPORTER_OTLP_HEADERS` environment variables — no code changes needed.
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

_last_sweep_time: float = 0.0
_SWEEP_INTERVAL_SECONDS = 300  # 5 minutes

# Heartbeat interval — keep SSE alive through proxies/LBs during tool execution.
# IMPORTANT: Must be less than frontend timeout (12s in useCopilotPage.ts)
_HEARTBEAT_INTERVAL = 10.0  # seconds


STREAM_LOCK_PREFIX = "copilot:stream:lock:"


async def _safe_close_sdk_client(
    sdk_client: ClaudeSDKClient,
    log_prefix: str,
) -> None:
    """Close a ClaudeSDKClient, suppressing errors from client disconnect.

    When the SSE client disconnects mid-stream, ``GeneratorExit`` propagates
    through the async generator stack and causes ``ClaudeSDKClient.__aexit__``
    to run in a different async context or task than where the client was
    opened.  This triggers two known error classes:

    * ``ValueError``: ``<Token var=<ContextVar name='current_context'>>
      was created in a different Context`` — OpenTelemetry's
      ``context.detach()`` fails because the OTEL context token was
      created in the original generator coroutine but detach runs in
      the GC / cleanup coroutine (Sentry: AUTOGPT-SERVER-8BT).

    * ``RuntimeError``: ``Attempted to exit cancel scope in a different
      task than it was entered in`` — anyio's ``TaskGroup.__aexit__``
      detects that the cancel scope was entered in one task but is
      being exited in another (Sentry: AUTOGPT-SERVER-8BW).

    Both are harmless — the TCP connection is already dead and no
    resources leak.  Logging them at ``debug`` level keeps observability
    without polluting Sentry.
    """
    try:
        await sdk_client.__aexit__(None, None, None)
    except (ValueError, RuntimeError) as exc:
        if _is_sdk_disconnect_error(exc):
            # Expected during client disconnect — suppress to avoid Sentry noise.
            logger.debug(
                "%s SDK client cleanup error suppressed (client disconnect): %s: %s",
                log_prefix,
                type(exc).__name__,
                exc,
            )
        else:
            raise
    except GeneratorExit:
        # GeneratorExit can propagate through __aexit__ — suppress it here
        # since the generator is already being torn down.
        logger.debug(
            "%s SDK client cleanup GeneratorExit suppressed (client disconnect)",
            log_prefix,
        )
    except Exception:
        # Unexpected cleanup error — log at error level so Sentry captures it
        # (via its logging integration), but don't propagate since we're in
        # teardown and the caller cannot meaningfully handle this.
        logger.error(
            "%s Unexpected SDK client cleanup error",
            log_prefix,
            exc_info=True,
        )


async def _iter_sdk_messages(
    client: ClaudeSDKClient,
) -> AsyncGenerator[Any, None]:
    """Yield SDK messages with heartbeat-based timeouts.

    Uses an explicit async iterator with non-cancelling heartbeats.

    CRITICAL: we must NOT cancel `__anext__()` mid-flight — doing so
    (via `asyncio.timeout` or `wait_for`) corrupts the SDK's internal
    anyio memory stream, causing `StopAsyncIteration` on the next call
    and silently dropping all in-flight tool results.  Instead, wrap
    `__anext__()` in a `Task` and use `asyncio.wait()` with a
    timeout.  On timeout we yield a heartbeat sentinel but keep the Task
    alive so it can deliver the next message.

    Yields `None` on heartbeat timeout (caller should refresh locks and
    emit heartbeat events).  Yields the raw SDK message otherwise.
    On stream end (`StopAsyncIteration`), the generator returns normally.
    Any other exception from the SDK propagates to the caller.
    """
    msg_iter = client.receive_response().__aiter__()
    pending_task: asyncio.Task[Any] | None = None

    async def _next_msg() -> Any:
        """Await the next SDK message, wrapped for use with `asyncio.Task`."""
        return await msg_iter.__anext__()

    try:
        while True:
            if pending_task is None:
                pending_task = asyncio.create_task(_next_msg())

            done, _ = await asyncio.wait({pending_task}, timeout=_HEARTBEAT_INTERVAL)

            if not done:
                yield None  # heartbeat sentinel
                continue

            pending_task = None
            try:
                yield done.pop().result()
            except StopAsyncIteration:
                return
    finally:
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
            try:
                await pending_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass


def _normalize_model_name(raw_model: str) -> str:
    """Normalize a model name for the current routing configuration.

    Applies two transformations shared by both the primary and fallback
    model resolution paths:

    1. **Strip provider prefix** — OpenRouter-style names like
       ``"anthropic/claude-opus-4.6"`` are reduced to ``"claude-opus-4.6"``.
    2. **Dot-to-hyphen conversion** — when *not* routing through OpenRouter
       the direct Anthropic API requires hyphen-separated versions
       (``"claude-opus-4-6"``), so dots are replaced with hyphens.
    """
    model = raw_model
    if "/" in model:
        model = model.split("/", 1)[1]
    # OpenRouter uses dots in versions (claude-opus-4.6) but the direct
    # Anthropic API requires hyphens (claude-opus-4-6).  Only normalise
    # when NOT routing through OpenRouter.
    if not config.openrouter_active:
        model = model.replace(".", "-")
    return model


def _resolve_sdk_model() -> str | None:
    """Resolve the model name for the Claude Agent SDK CLI.

    Uses `config.claude_agent_model` if set, otherwise derives from
    `config.model` via :func:`_normalize_model_name`.

    When `use_claude_code_subscription` is enabled and no explicit
    `claude_agent_model` is set, returns `None` so the CLI uses the
    default model for the user's subscription plan.
    """
    if config.claude_agent_model:
        return config.claude_agent_model
    if config.use_claude_code_subscription:
        return None
    return _normalize_model_name(config.model)


def _resolve_fallback_model() -> str | None:
    """Resolve the fallback model name via :func:`_normalize_model_name`.

    Returns ``None`` when no fallback is configured (empty string).
    """
    raw = config.claude_agent_fallback_model
    if not raw:
        return None
    return _normalize_model_name(raw)


_MAX_TRANSIENT_BACKOFF_SECONDS = 30


def _compute_transient_backoff(attempt: int) -> int:
    """Return the exponential backoff delay (seconds) for a transient retry.

    ``attempt`` is 1-based: first retry → ~1 s, second → ~2 s, third → ~4 s,
    …, capped at :data:`_MAX_TRANSIENT_BACKOFF_SECONDS`.

    Full-jitter (``0.5 … 1.0 × base``) is applied to spread retries from
    multiple tenants hitting the same 429/529 simultaneously (thundering herd).
    The result is rounded to the nearest integer so the ``StreamStatus``
    message is human-readable.

    Extracted as a module-level pure function so it can be unit-tested
    independently of the closure that wraps it inside the retry loop.
    """
    base = min(_MAX_TRANSIENT_BACKOFF_SECONDS, 2 ** (attempt - 1))
    return max(1, round(base * (0.5 + random.random() * 0.5)))


def _next_transient_backoff(
    events_yielded: int,
    transient_retries: int,
    max_transient_retries: int,
) -> tuple[int | None, int]:
    """Decide whether to retry after a transient error.

    Returns ``(backoff_seconds, updated_retries)``.  ``backoff_seconds`` is
    ``None`` when no retry should be attempted — either content has already
    been streamed to the frontend (``events_yielded > 0``, retrying would
    produce duplicates) or the retry budget is exhausted.

    Extracted as a module-level pure function so it can be unit-tested
    independently of the retry loop.
    """
    if events_yielded > 0:
        return None, transient_retries
    new_retries = transient_retries + 1
    if new_retries > max_transient_retries:
        return None, new_retries
    return _compute_transient_backoff(new_retries), new_retries


async def _do_transient_backoff(
    backoff: int,
    state: _RetryState,
    message_id: str,
    session_id: str,
) -> AsyncIterator[StreamStatus]:
    """Emit a retry notification, sleep, and reset the SDK adapter.

    Yields a single :class:`StreamStatus` so the caller can forward it to
    the client, then sleeps for *backoff* seconds and resets ``state.adapter``
    and ``state.usage`` so the next attempt starts clean.

    Extracted from both exception handlers in the retry loop to remove
    near-identical code duplication.
    """
    yield StreamStatus(message=f"Connection interrupted, retrying in {backoff}s…")
    await asyncio.sleep(backoff)
    state.adapter = SDKResponseAdapter(message_id=message_id, session_id=session_id)
    state.usage.reset()


def _is_fallback_stderr(line: str) -> bool:
    """Return True if a CLI stderr line signals fallback-model activation.

    Matches ``"fallback model"`` case-insensitively.  Uses the specific
    two-word phrase rather than just ``"fallback"`` to avoid false positives
    from unrelated CLI stderr lines (tool retries, cached-result fallbacks).

    Extracted as a module-level pure function so it can be unit-tested
    without wiring up the full ``_on_stderr`` closure.
    """
    return "fallback model" in line.lower()


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


async def _cleanup_sdk_tool_results(cwd: str) -> None:
    """Remove SDK session artifacts for a specific working directory.

    Cleans up the ephemeral working directory `/tmp/copilot-<session>/`.

    Also sweeps stale CLI project directories (older than 12 h) to prevent
    unbounded disk growth.  The sweep is best-effort, rate-limited to once
    every 5 minutes, and capped at 50 directories per sweep.

    Security: *cwd* MUST be created by `_make_sdk_cwd()` which sanitizes
    the session_id.
    """
    normalized = os.path.normpath(cwd)
    if not normalized.startswith(_SDK_CWD_PREFIX):
        logger.warning("[SDK] Rejecting cleanup for path outside workspace: %s", cwd)
        return

    await asyncio.to_thread(shutil.rmtree, normalized, True)

    # Best-effort sweep of old project dirs to prevent disk leak.
    # Pass the encoded cwd so only this session's project directory is swept,
    # which is safe in multi-tenant environments.
    global _last_sweep_time
    now = time.time()
    if now - _last_sweep_time >= _SWEEP_INTERVAL_SECONDS:
        _last_sweep_time = now
        encoded = encode_cwd_for_cli(normalized)
        await asyncio.to_thread(cleanup_stale_project_dirs, encoded)


def _format_sdk_content_blocks(blocks: list) -> list[dict[str, Any]]:
    """Convert SDK content blocks to transcript format.

    Handles TextBlock, ToolUseBlock, ToolResultBlock, and ThinkingBlock.
    Raw dicts (e.g. ``redacted_thinking`` blocks that the SDK may not have
    a typed class for) are passed through verbatim to preserve them in the
    transcript.  Unknown typed block objects are logged and skipped.
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
            tool_result_entry: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": block.tool_use_id,
                "content": block.content,
            }
            if block.is_error:
                tool_result_entry["is_error"] = True
            result.append(tool_result_entry)
        elif isinstance(block, ThinkingBlock):
            result.append(
                {
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                }
            )
        elif isinstance(block, dict) and "type" in block:
            # Preserve raw dict blocks (e.g. redacted_thinking) verbatim.
            result.append(block)
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

    Delegates to `_run_compression` (`transcript.py`) which centralizes
    the "try LLM, fallback to truncation" pattern with timeouts.  Both
    `_compress_messages` and `compact_transcript` share this helper so
    client acquisition and error handling are consistent.

    See also:
        `_run_compression` — shared compression with timeout guards.
        `compact_transcript` — compresses JSONL transcript entries.
        `CompactionTracker` — emits UI events for mid-stream compaction.
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
        result = await _run_compression(messages_dict, config.model, "[SDK]")
    except Exception as exc:
        # Guard against timeouts or unexpected errors in compression —
        # return the original messages so the caller can proceed without
        # compaction rather than propagating the error to the retry loop.
        logger.warning("[SDK] _compress_messages failed, returning originals: %s", exc)
        return messages, False

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


def _session_messages_to_transcript(messages: list[ChatMessage]) -> str:
    """Convert session ChatMessages to JSONL transcript for ``--resume``.

    Reconstructs proper ``tool_use`` and ``tool_result`` content blocks from
    :attr:`ChatMessage.tool_calls` and :attr:`ChatMessage.tool_call_id` so the
    Claude CLI receives full structural context when no previous transcript file
    is available (e.g. first turn after a storage failure or compaction drop).

    This gives the model the same fidelity as an on-disk session JSONL file —
    preserving tool call names, IDs, inputs, and *complete* (un-truncated)
    tool results — rather than the lossy plain-text injection produced by
    :func:`_format_conversation_context` (which caps tool results at 500 chars
    and discards structural linkage).

    Args:
        messages: Prior session messages, typically ``session.messages[:-1]``
            (all turns except the current user query).

    Returns:
        A JSONL string suitable for writing to a temp file and passing as
        ``ClaudeAgentOptions.resume``.  Returns an empty string if the input
        list is empty after filtering compaction entries.
    """
    filtered = filter_compaction_messages(messages)
    if not filtered:
        return ""
    builder = TranscriptBuilder()
    for msg in filtered:
        if msg.role == "user" and msg.content:
            builder.append_user(msg.content)
        elif msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls or []:
                try:
                    tc_input: dict[str, Any] = json.loads(
                        tc.get("function", {}).get("arguments", "{}")
                    )
                except (json.JSONDecodeError, ValueError):
                    tc_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": tc.get("function", {}).get("name", ""),
                        "input": tc_input,
                    }
                )
            if blocks:
                builder.append_assistant(blocks)
        elif msg.role == "tool" and msg.tool_call_id:
            builder.append_tool_result(
                tool_use_id=msg.tool_call_id,
                content=msg.content or "",
            )
    return builder.to_jsonl()


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
            # Omit tool_calls — any text representation gets mimicked
            # by the model. Tool results below provide the context.
        elif msg.role == "tool":
            content = msg.content or ""
            lines.append(f"Tool output: {content[:500]}")

    if not lines:
        return None

    return "<conversation_history>\n" + "\n".join(lines) + "\n</conversation_history>"


async def _build_query_message(
    current_message: str,
    session: ChatSession,
    use_resume: bool,
    transcript_msg_count: int,
    session_id: str,
    *,
    session_msg_ceiling: int | None = None,
) -> tuple[str, bool]:
    """Build the query message with appropriate context.

    Args:
        session_msg_ceiling: If provided, treat ``session.messages`` as if it
            only has this many entries when computing the gap slice.  Pass
            ``len(session.messages)`` captured *before* appending any pending
            messages so that mid-turn drains do not skew the gap calculation
            and cause pending messages to be duplicated in both the gap context
            and ``current_message``.

    Returns:
        Tuple of (query_message, was_compacted).
    """
    msg_count = len(session.messages)
    # Use the ceiling if supplied (prevents pending-message duplication when
    # messages were appended to session.messages after the drain but before
    # this function is called).
    effective_count = (
        session_msg_ceiling if session_msg_ceiling is not None else msg_count
    )

    if use_resume and transcript_msg_count > 0:
        if transcript_msg_count < effective_count - 1:
            gap = session.messages[transcript_msg_count : effective_count - 1]
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
    elif not use_resume and effective_count > 1:
        logger.warning(
            f"[SDK] Using compression fallback for session "
            f"{session_id} ({effective_count} messages) — no transcript for --resume"
        )
        compressed, was_compressed = await _compress_messages(
            session.messages[: effective_count - 1]
        )
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
        manager = await get_workspace_manager(user_id, session_id)
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


@dataclass
class _StreamAccumulator:
    """Mutable state accumulated during a single streaming attempt.

    Tracks the assistant message being built, tool calls, and flags that
    control session-message bookkeeping in the dispatch loop.
    """

    assistant_response: ChatMessage
    accumulated_tool_calls: list[dict[str, Any]]
    has_appended_assistant: bool = False
    has_tool_results: bool = False
    stream_completed: bool = False


def _dispatch_response(
    response: StreamBaseResponse,
    acc: _StreamAccumulator,
    ctx: "_StreamContext",
    state: "_RetryState",
    entries_replaced: bool,
    log_prefix: str,
) -> StreamBaseResponse | None:
    """Process a single adapter response and update session/accumulator state.

    Returns the response to yield to the client, or `None` if the response
    should be suppressed (e.g. `StreamStart` duplicates).

    Handles:
    - Logging tool events and errors
    - Persisting error markers
    - Accumulating text deltas into `assistant_response`
    - Appending tool input/output to session messages and transcript
    - Detecting `StreamFinish`
    """
    if isinstance(response, StreamStart):
        return None

    if isinstance(
        response,
        (StreamToolInputAvailable, StreamToolOutputAvailable),
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

    # Persist error markers so they survive page refresh
    if isinstance(response, StreamError):
        logger.error(
            "%s Sending error to frontend: %s (code=%s)",
            log_prefix,
            response.errorText,
            response.code,
        )
        _append_error_marker(
            ctx.session,
            response.errorText,
            retryable=(response.code == "transient_api_error"),
        )

    if isinstance(response, StreamTextDelta):
        delta = response.delta or ""
        if acc.has_tool_results and acc.has_appended_assistant:
            acc.assistant_response = ChatMessage(role="assistant", content=delta)
            acc.accumulated_tool_calls = []
            acc.has_appended_assistant = False
            acc.has_tool_results = False
            ctx.session.messages.append(acc.assistant_response)
            acc.has_appended_assistant = True
        else:
            acc.assistant_response.content = (
                acc.assistant_response.content or ""
            ) + delta
            if not acc.has_appended_assistant:
                ctx.session.messages.append(acc.assistant_response)
                acc.has_appended_assistant = True

    elif isinstance(response, StreamToolInputAvailable):
        acc.accumulated_tool_calls.append(
            {
                "id": response.toolCallId,
                "type": "function",
                "function": {
                    "name": response.toolName,
                    "arguments": json.dumps(response.input or {}),
                },
            }
        )
        acc.assistant_response.tool_calls = acc.accumulated_tool_calls
        if not acc.has_appended_assistant:
            ctx.session.messages.append(acc.assistant_response)
            acc.has_appended_assistant = True

    elif isinstance(response, StreamToolOutputAvailable):
        content = (
            response.output
            if isinstance(response.output, str)
            else json.dumps(response.output, ensure_ascii=False)
        )
        ctx.session.messages.append(
            ChatMessage(
                role="tool",
                content=content,
                tool_call_id=response.toolCallId,
            )
        )
        if not entries_replaced:
            state.transcript_builder.append_tool_result(
                tool_use_id=response.toolCallId,
                content=content,
            )
        acc.has_tool_results = True

    elif isinstance(response, StreamFinish):
        acc.stream_completed = True

    return response


class _HandledStreamError(Exception):
    """Raised by `_run_stream_attempt` when an attempt fails and the outer
    retry loop must roll back session state.

    Two sub-cases:

    * ``already_yielded=True`` (default) — a ``StreamError`` was already sent
      to the client inside ``_run_stream_attempt`` (circuit-breaker, idle
      timeout, etc.).  The outer loop must **not** yield another one.
    * ``already_yielded=False`` — the error is transient and the outer loop
      will decide whether to retry or surface the error.  If retrying it
      yields a ``StreamStatus("retrying…")``; if exhausted it yields the
      ``StreamError`` itself so the client sees it only once.

    Attributes:
        error_msg: The user-facing error message to persist.
        code: Machine-readable error code (e.g. ``circuit_breaker_empty_tool_calls``).
        retryable: Whether the frontend should offer a retry button.
        already_yielded: ``True`` when ``StreamError`` was already sent to the
            client before this exception was raised.
    """

    def __init__(
        self,
        message: str,
        error_msg: str | None = None,
        code: str | None = None,
        retryable: bool = True,
        already_yielded: bool = True,
    ):
        super().__init__(message)
        self.error_msg = error_msg
        self.code = code
        self.retryable = retryable
        self.already_yielded = already_yielded


@dataclass
class _EmptyToolBreakResult:
    """Result of checking for empty tool calls in a single AssistantMessage."""

    count: int  # Updated consecutive counter
    tripped: bool  # Whether the circuit breaker fired
    error: StreamError | None  # StreamError to yield (if tripped)
    error_msg: str | None  # Error message (if tripped)
    error_code: str | None  # Error code (if tripped)


def _check_empty_tool_breaker(
    sdk_msg: object,
    consecutive: int,
    ctx: _StreamContext,
    state: _RetryState,
) -> _EmptyToolBreakResult:
    """Detect consecutive empty tool calls and trip the circuit breaker.

    Returns an ``_EmptyToolBreakResult`` with the updated counter and, if the
    breaker tripped, the ``StreamError`` to yield plus the error metadata.
    """
    if not isinstance(sdk_msg, AssistantMessage):
        return _EmptyToolBreakResult(consecutive, False, None, None, None)

    empty_tools = [
        b.name for b in sdk_msg.content if isinstance(b, ToolUseBlock) and not b.input
    ]
    if not empty_tools:
        # Reset on any non-empty-tool AssistantMessage (including text-only
        # messages — any() over empty content is False).
        return _EmptyToolBreakResult(0, False, None, None, None)

    consecutive += 1

    # Log full diagnostics on first occurrence only; subsequent hits just
    # log the counter to reduce noise.
    if consecutive == 1:
        logger.warning(
            "%s Empty tool call detected (%d/%d): "
            "tools=%s, model=%s, error=%s, "
            "block_types=%s, cumulative_usage=%s",
            ctx.log_prefix,
            consecutive,
            _EMPTY_TOOL_CALL_LIMIT,
            empty_tools,
            sdk_msg.model,
            sdk_msg.error,
            [type(b).__name__ for b in sdk_msg.content],
            {
                "prompt": state.usage.prompt_tokens,
                "completion": state.usage.completion_tokens,
                "cache_read": state.usage.cache_read_tokens,
            },
        )
    else:
        logger.warning(
            "%s Empty tool call detected (%d/%d): tools=%s",
            ctx.log_prefix,
            consecutive,
            _EMPTY_TOOL_CALL_LIMIT,
            empty_tools,
        )

    if consecutive < _EMPTY_TOOL_CALL_LIMIT:
        return _EmptyToolBreakResult(consecutive, False, None, None, None)

    logger.error(
        "%s Circuit breaker: aborting stream after %d "
        "consecutive empty tool calls. "
        "This is likely caused by the model attempting "
        "to write content too large for a single tool "
        "call's output token limit. The model should "
        "write large files in chunks using bash_exec "
        "with cat >> (append).",
        ctx.log_prefix,
        consecutive,
    )
    error_msg = _CIRCUIT_BREAKER_ERROR_MSG
    error_code = "circuit_breaker_empty_tool_calls"
    _append_error_marker(ctx.session, error_msg, retryable=True)
    return _EmptyToolBreakResult(
        count=consecutive,
        tripped=True,
        error=StreamError(errorText=error_msg, code=error_code),
        error_msg=error_msg,
        error_code=error_code,
    )


async def _run_stream_attempt(
    ctx: _StreamContext,
    state: _RetryState,
) -> AsyncIterator[StreamBaseResponse]:
    """Run one SDK streaming attempt.

    Opens a `ClaudeSDKClient`, sends the query, iterates SDK messages with
    heartbeat timeouts, dispatches adapter responses, and performs post-stream
    cleanup (safety-net flush, stopped-by-user handling).

    Yields stream events.  On stream error the exception propagates to the
    caller so the retry loop can rollback and retry.

    Args:
        ctx: Per-request context shared across retry attempts.  Scalar
            fields (IDs, paths, message string) are set once and never
            reassigned.  `session`, `compaction`, and `lock` are
            shared mutable references: `session.messages` is rolled back
            on retry, `compaction` tracks mid-stream compaction events,
            and `lock` is refreshed during heartbeats.  Their references
            are constant even though the objects they point to are mutated.
        state: Mutable retry state — holds values that the retry loop
            modifies between attempts (options, query, adapter, etc.).

    See also:
        `stream_chat_completion_sdk` — owns the retry loop that calls this
        function up to `_MAX_STREAM_ATTEMPTS` times with reduced context.
    """
    acc = _StreamAccumulator(
        assistant_response=ChatMessage(role="assistant", content=""),
        accumulated_tool_calls=[],
    )
    ended_with_stream_error = False
    # Stores the error message used by _append_error_marker so the outer
    # retry loop can re-append the correct message after session rollback.
    stream_error_msg: str | None = None
    stream_error_code: str | None = None

    consecutive_empty_tool_calls = 0

    # --- Intermediate persistence tracking ---
    # Flush session messages to DB periodically so page reloads show progress
    # during long-running turns (see incident d2f7cba3: 82-min turn lost on refresh).
    _last_flush_time = time.monotonic()
    _msgs_since_flush = 0
    _FLUSH_INTERVAL_SECONDS = 30.0
    _FLUSH_MESSAGE_THRESHOLD = 10

    # Use manual __aenter__/__aexit__ instead of ``async with`` so we can
    # suppress SDK cleanup errors that occur when the SSE client disconnects
    # mid-stream.  GeneratorExit causes the SDK's ``__aexit__`` to run in a
    # different async context/task than where the client was opened, which
    # triggers:
    #   - ValueError: ContextVar token mismatch (AUTOGPT-SERVER-8BT)
    #   - RuntimeError: cancel scope in wrong task  (AUTOGPT-SERVER-8BW)
    # Both are harmless — the TCP connection is already dead.
    sdk_client = ClaudeSDKClient(options=state.options)
    client = await sdk_client.__aenter__()
    try:
        logger.info(
            "%s Sending query — resume=%s, total_msgs=%d, "
            "query_len=%d, attached_files=%d, image_blocks=%d",
            ctx.log_prefix,
            state.use_resume,
            len(ctx.session.messages),
            len(state.query_message),
            len(ctx.file_ids) if ctx.file_ids else 0,
            len(ctx.attachments.image_blocks),
        )

        ctx.compaction.reset_for_query()
        if state.was_compacted:
            for ev in ctx.compaction.emit_pre_query(ctx.session):
                yield ev

        if ctx.attachments.image_blocks:
            content_blocks: list[dict[str, Any]] = [
                *ctx.attachments.image_blocks,
                {"type": "text", "text": state.query_message},
            ]
            user_msg = {
                "type": "user",
                "message": {"role": "user", "content": content_blocks},
                "parent_tool_use_id": None,
                "session_id": ctx.session_id,
            }
            if client._transport is None:  # noqa: SLF001
                raise RuntimeError("ClaudeSDKClient transport is not initialized")
            await client._transport.write(json.dumps(user_msg) + "\n")  # noqa: SLF001
            state.transcript_builder.append_user(
                content=[
                    *ctx.attachments.image_blocks,
                    {"type": "text", "text": ctx.current_message},
                ]
            )
        else:
            await client.query(state.query_message, session_id=ctx.session_id)
            state.transcript_builder.append_user(content=ctx.current_message)

        _last_real_msg_time = time.monotonic()

        async for sdk_msg in _iter_sdk_messages(client):
            # Heartbeat sentinel — refresh lock and keep SSE alive
            if sdk_msg is None:
                await ctx.lock.refresh()
                for ev in ctx.compaction.emit_start_if_ready():
                    yield ev
                yield StreamHeartbeat()

                # Idle timeout: if no real SDK message for too long, a tool
                # call is likely hung (e.g. WebSearch provider not responding).
                idle_seconds = time.monotonic() - _last_real_msg_time
                if idle_seconds >= _IDLE_TIMEOUT_SECONDS:
                    logger.error(
                        "%s Idle timeout after %.0fs with no SDK message — "
                        "aborting stream (likely hung tool call)",
                        ctx.log_prefix,
                        idle_seconds,
                    )
                    stream_error_msg = (
                        "A tool call appears to be stuck "
                        "(no response for 10 minutes). "
                        "Please try again."
                    )
                    stream_error_code = "idle_timeout"
                    _append_error_marker(ctx.session, stream_error_msg, retryable=True)
                    yield StreamError(
                        errorText=stream_error_msg,
                        code=stream_error_code,
                    )
                    ended_with_stream_error = True
                    break
                continue

            _last_real_msg_time = time.monotonic()

            logger.info(
                "%s Received: %s %s (unresolved=%d, current=%d, resolved=%d)",
                ctx.log_prefix,
                type(sdk_msg).__name__,
                getattr(sdk_msg, "subtype", ""),
                len(state.adapter.current_tool_calls)
                - len(state.adapter.resolved_tool_calls),
                len(state.adapter.current_tool_calls),
                len(state.adapter.resolved_tool_calls),
            )

            # Log AssistantMessage API errors (e.g. invalid_request)
            # so we can debug Anthropic API 400s surfaced by the CLI.
            sdk_error = getattr(sdk_msg, "error", None)
            if isinstance(sdk_msg, AssistantMessage) and sdk_error:
                error_text = str(sdk_error)
                error_preview = str(sdk_msg.content)[:500]
                logger.error(
                    "[SDK] [%s] AssistantMessage has error=%s, "
                    "content_blocks=%d, content_preview=%s",
                    ctx.session_id[:12],
                    sdk_error,
                    len(sdk_msg.content),
                    error_preview,
                )

                # Intercept prompt-too-long errors surfaced as
                # AssistantMessage.error (not as a Python exception).
                # Re-raise so the outer retry loop can compact the
                # transcript and retry with reduced context.
                # Check both error_text and error_preview: sdk_error
                # being set confirms this is an error message (not user
                # content), so checking content is safe. The actual
                # error description (e.g. "Prompt is too long") may be
                # in the content, not the error type field
                # (e.g. error="invalid_request", content="Prompt is
                # too long").
                if _is_prompt_too_long(Exception(error_text)) or _is_prompt_too_long(
                    Exception(error_preview)
                ):
                    logger.warning(
                        "%s Prompt-too-long detected via AssistantMessage "
                        "error — raising for retry",
                        ctx.log_prefix,
                    )
                    raise RuntimeError("Prompt is too long")

                # Intercept transient API errors (socket closed,
                # ECONNRESET) — replace the raw message with a
                # user-friendly error text and use the retryable
                # error prefix so the frontend shows a retry button.
                # Check both the error field and content for patterns.
                if is_transient_api_error(error_text) or is_transient_api_error(
                    error_preview
                ):
                    logger.warning(
                        "%s Transient Anthropic API error detected, "
                        "suppressing raw error text",
                        ctx.log_prefix,
                    )
                    stream_error_msg = FRIENDLY_TRANSIENT_MSG
                    stream_error_code = "transient_api_error"
                    # Do NOT yield StreamError or append error marker here.
                    # The outer retry loop decides: if a retry is available it
                    # yields StreamStatus("retrying…"); if retries are exhausted
                    # it appends the marker and yields StreamError exactly once.
                    # Yielding StreamError before the retry decision causes the
                    # client to display an error that is immediately superseded.
                    ended_with_stream_error = True
                    break

            # Determine if the message is a tool-only batch (all content
            # items are ToolUseBlocks) — such messages have no text output yet,
            # so we skip the wait_for_stash flush below.
            #
            # Note: parallel execution of tools is handled natively by the
            # SDK CLI via readOnlyHint annotations on tool definitions.
            is_tool_only = False
            if isinstance(sdk_msg, AssistantMessage) and sdk_msg.content:
                is_tool_only = all(
                    isinstance(item, ToolUseBlock) for item in sdk_msg.content
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
            if (
                state.adapter.has_unresolved_tool_calls
                and isinstance(sdk_msg, (AssistantMessage, ResultMessage))
                and not is_tool_only
            ):
                if await wait_for_stash():
                    await asyncio.sleep(0)
                else:
                    logger.warning(
                        "%s Timed out waiting for PostToolUse "
                        "hook stash (%d unresolved tool calls)",
                        ctx.log_prefix,
                        len(state.adapter.current_tool_calls)
                        - len(state.adapter.resolved_tool_calls),
                    )

            # Log ResultMessage details and capture token usage
            if isinstance(sdk_msg, ResultMessage):
                logger.info(
                    "%s Received: ResultMessage %s "
                    "(unresolved=%d, current=%d, resolved=%d, "
                    "num_turns=%d, cost_usd=%s, result=%s)",
                    ctx.log_prefix,
                    sdk_msg.subtype,
                    len(state.adapter.current_tool_calls)
                    - len(state.adapter.resolved_tool_calls),
                    len(state.adapter.current_tool_calls),
                    len(state.adapter.resolved_tool_calls),
                    sdk_msg.num_turns,
                    sdk_msg.total_cost_usd,
                    (sdk_msg.result or "")[:200],
                )
                if sdk_msg.subtype in (
                    "error",
                    "error_during_execution",
                ):
                    logger.error(
                        "%s SDK execution failed with error: %s",
                        ctx.log_prefix,
                        sdk_msg.result or "(no error message provided)",
                    )

                # Check for prompt-too-long regardless of subtype — the
                # SDK may return subtype="success" with result="Prompt is
                # too long" when the CLI rejects the prompt before calling
                # the API (cost_usd=0, no tokens consumed).  If we only
                # check the "error" subtype path, the stream appears to
                # complete normally, the synthetic error text is stored
                # in the transcript, and the session grows without bound.
                if _is_prompt_too_long(RuntimeError(sdk_msg.result or "")):
                    raise RuntimeError("Prompt is too long")

                # Capture token usage from ResultMessage.
                # Anthropic reports cached tokens separately:
                #   input_tokens = uncached only
                #   cache_read_input_tokens = served from cache
                #   cache_creation_input_tokens = written to cache
                if sdk_msg.usage:
                    state.usage.prompt_tokens += sdk_msg.usage.get("input_tokens", 0)
                    state.usage.cache_read_tokens += sdk_msg.usage.get(
                        "cache_read_input_tokens", 0
                    )
                    state.usage.cache_creation_tokens += sdk_msg.usage.get(
                        "cache_creation_input_tokens", 0
                    )
                    state.usage.completion_tokens += sdk_msg.usage.get(
                        "output_tokens", 0
                    )
                    logger.info(
                        "%s Token usage: uncached=%d, cache_read=%d, "
                        "cache_create=%d, output=%d",
                        ctx.log_prefix,
                        state.usage.prompt_tokens,
                        state.usage.cache_read_tokens,
                        state.usage.cache_creation_tokens,
                        state.usage.completion_tokens,
                    )
                if sdk_msg.total_cost_usd is not None:
                    state.usage.cost_usd = sdk_msg.total_cost_usd

            # Emit compaction end if SDK finished compacting.
            # Sync TranscriptBuilder with the CLI's active context.
            compact_result = await ctx.compaction.emit_end_if_ready(ctx.session)
            if compact_result.events:
                # Compaction events end with StreamFinishStep, which maps to
                # Vercel AI SDK's "finish-step" — that clears activeTextParts.
                # Close any open text block BEFORE the compaction events so
                # the text-end arrives before finish-step, preventing
                # "text-end for missing text part" errors on the frontend.
                pre_close: list[StreamBaseResponse] = []
                state.adapter._end_text_if_open(pre_close)
                # Compaction events bypass the adapter, so sync step state
                # when a StreamFinishStep is present — otherwise the adapter
                # will skip StreamStartStep on the next AssistantMessage.
                if any(
                    isinstance(ev, StreamFinishStep) for ev in compact_result.events
                ):
                    state.adapter.step_open = False
                for r in pre_close:
                    yield r
            for ev in compact_result.events:
                yield ev
            entries_replaced = False
            if compact_result.just_ended:
                compacted = await asyncio.to_thread(
                    read_compacted_entries,
                    compact_result.transcript_path,
                )
                if compacted is not None:
                    state.transcript_builder.replace_entries(
                        compacted, log_prefix=ctx.log_prefix
                    )
                    entries_replaced = True

            # --- Hard circuit breaker for empty tool calls ---
            breaker = _check_empty_tool_breaker(
                sdk_msg, consecutive_empty_tool_calls, ctx, state
            )
            consecutive_empty_tool_calls = breaker.count
            if breaker.tripped and breaker.error is not None:
                stream_error_msg = breaker.error_msg
                stream_error_code = breaker.error_code
                yield breaker.error
                ended_with_stream_error = True
                break

            # --- Dispatch adapter responses ---
            for response in state.adapter.convert_message(sdk_msg):
                dispatched = _dispatch_response(
                    response, acc, ctx, state, entries_replaced, ctx.log_prefix
                )
                if dispatched is not None:
                    yield dispatched

            # Append assistant entry AFTER convert_message so that
            # any stashed tool results from the previous turn are
            # recorded first, preserving the required API order:
            # assistant(tool_use) → tool_result → assistant(text).
            # Skip if replace_entries just ran — the CLI session
            # file already contains this message.
            if isinstance(sdk_msg, AssistantMessage) and not entries_replaced:
                state.transcript_builder.append_assistant(
                    content_blocks=_format_sdk_content_blocks(sdk_msg.content),
                    model=sdk_msg.model,
                )

            # --- Intermediate persistence ---
            # Flush session messages to DB periodically so page reloads
            # show progress during long-running turns.
            #
            # IMPORTANT: Skip the flush while tool calls are pending
            # (tool_calls set on assistant but results not yet received).
            # The DB save is append-only (uses start_sequence), so if we
            # flush the assistant message before tool_calls are set on it
            # (text and tool_use arrive as separate SDK events), the
            # tool_calls update is lost — the next flush starts past it.
            _msgs_since_flush += 1
            now = time.monotonic()
            has_pending_tools = (
                acc.has_appended_assistant
                and acc.accumulated_tool_calls
                and not acc.has_tool_results
            )
            if not has_pending_tools and (
                _msgs_since_flush >= _FLUSH_MESSAGE_THRESHOLD
                or (now - _last_flush_time) >= _FLUSH_INTERVAL_SECONDS
            ):
                try:
                    await asyncio.shield(upsert_chat_session(ctx.session))
                    logger.debug(
                        "%s Intermediate flush: %d messages "
                        "(msgs_since=%d, elapsed=%.1fs)",
                        ctx.log_prefix,
                        len(ctx.session.messages),
                        _msgs_since_flush,
                        now - _last_flush_time,
                    )
                except Exception as flush_err:
                    logger.warning(
                        "%s Intermediate flush failed: %s",
                        ctx.log_prefix,
                        flush_err,
                    )
                _last_flush_time = now
                _msgs_since_flush = 0

            if acc.stream_completed:
                break
    finally:
        await _safe_close_sdk_client(sdk_client, ctx.log_prefix)

    # --- Post-stream processing (only on success) ---
    if state.adapter.has_unresolved_tool_calls:
        logger.warning(
            "%s %d unresolved tool(s) after stream — flushing",
            ctx.log_prefix,
            len(state.adapter.current_tool_calls)
            - len(state.adapter.resolved_tool_calls),
        )
        safety_responses: list[StreamBaseResponse] = []
        state.adapter._flush_unresolved_tool_calls(safety_responses)
        for response in safety_responses:
            if isinstance(
                response,
                (StreamToolInputAvailable, StreamToolOutputAvailable),
            ):
                logger.info(
                    "%s Safety flush: %s, tool=%s",
                    ctx.log_prefix,
                    type(response).__name__,
                    getattr(response, "toolName", "N/A"),
                )
            if isinstance(response, StreamToolOutputAvailable):
                state.transcript_builder.append_tool_result(
                    tool_use_id=response.toolCallId,
                    content=(
                        response.output
                        if isinstance(response.output, str)
                        else json.dumps(response.output, ensure_ascii=False)
                    ),
                )
            yield response

    if not acc.stream_completed and not ended_with_stream_error:
        logger.info(
            "%s Stream ended without ResultMessage (stopped by user)",
            ctx.log_prefix,
        )
        closing_responses: list[StreamBaseResponse] = []
        state.adapter._end_text_if_open(closing_responses)
        for r in closing_responses:
            yield r
        ctx.session.messages.append(
            ChatMessage(
                role="assistant",
                content=f"{COPILOT_SYSTEM_PREFIX} Execution stopped by user",
            )
        )

    if (
        acc.assistant_response.content or acc.assistant_response.tool_calls
    ) and not acc.has_appended_assistant:
        ctx.session.messages.append(acc.assistant_response)

    # Raise so the outer retry loop can rollback session messages.
    # already_yielded=False for transient_api_error: StreamError was NOT
    # sent to the client yet (the outer loop does it when retries are
    # exhausted, avoiding a premature error flash before the retry).
    if ended_with_stream_error:
        raise _HandledStreamError(
            "Stream error handled",
            error_msg=stream_error_msg,
            code=stream_error_code,
            already_yielded=(stream_error_code != "transient_api_error"),
        )


async def stream_chat_completion_sdk(
    session_id: str,
    message: str | None = None,
    is_user_message: bool = True,
    user_id: str | None = None,
    session: ChatSession | None = None,
    file_ids: list[str] | None = None,
    permissions: "CopilotPermissions | None" = None,
    mode: CopilotMode | None = None,
    **_kwargs: Any,
) -> AsyncIterator[StreamBaseResponse]:
    """Stream chat completion using Claude Agent SDK.

    Args:
        file_ids: Optional workspace file IDs attached to the user's message.
            Images are embedded as vision content blocks; other files are
            saved to the SDK working directory for the Read tool.
        mode: Accepted for signature compatibility with the baseline path.
            The SDK path does not currently branch on this value.
    """
    _ = mode  # SDK path ignores the requested mode.

    if session is None:
        session = await get_chat_session(session_id, user_id)

    if not session:
        raise NotFoundError(
            f"Session {session_id} not found. Please create a new session first."
        )

    # Type narrowing: session is guaranteed ChatSession after the check above
    session = cast(ChatSession, session)

    # Clean up ALL trailing error markers from previous turn before starting
    # a new turn.  Multiple markers can accumulate when a mid-stream error is
    # followed by a cleanup error in __aexit__ (both append a marker).
    while (
        len(session.messages) > 0
        and session.messages[-1].role == "assistant"
        and session.messages[-1].content
        and (
            COPILOT_ERROR_PREFIX in session.messages[-1].content
            or COPILOT_RETRYABLE_ERROR_PREFIX in session.messages[-1].content
        )
    ):
        logger.info(
            "[SDK] [%s] Removing stale error marker from previous turn",
            session_id[:12],
        )
        session.messages.pop()

    # Strip any user-injected <user_context> tags on every turn.
    # Only the server-injected prefix on the first message is trusted.
    if message:
        message = strip_user_context_tags(message)

    if maybe_append_user_message(session, message, is_user_message):
        if is_user_message:
            track_user_message(
                user_id=user_id,
                session_id=session_id,
                message_length=len(message or ""),
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
    ended_with_stream_error = False
    e2b_sandbox = None
    use_resume = False
    resume_file: str | None = None
    transcript_builder = TranscriptBuilder()
    sdk_cwd = ""
    # True when transcript_builder represents a full prefix of session history.
    # First turn (<=1 prior message) is fully covered even without a download.
    # Set to True when load_previous succeeds; stays False when download fails
    # on a session with prior messages, preventing a partial upload that would
    # mislead _build_query_message into skipping gap reconstruction next turn.
    transcript_covers_prefix = True

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
            "%s Session already has an active stream: %s", log_prefix, lock_owner
        )
        yield StreamError(
            errorText="Another stream is already active for this session. "
            "Please wait or stop it.",
            code="stream_already_active",
        )
        return

    # OTEL context manager — initialized inside the try and cleaned up in finally.
    _otel_ctx: Any = None
    skip_transcript_upload = False
    transcript_content: str = ""
    state: _RetryState | None = None

    # Token usage accumulators — populated from ResultMessage at end of turn
    turn_prompt_tokens = 0  # uncached input tokens only
    turn_completion_tokens = 0
    turn_cache_read_tokens = 0
    turn_cache_creation_tokens = 0
    turn_cost_usd: float | None = None
    graphiti_enabled = False

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
            if not (e2b_api_key := config.active_e2b_api_key):
                if config.use_e2b_sandbox:
                    logger.warning(
                        "[E2B] [%s] E2B sandbox enabled but no API key configured "
                        "(CHAT_E2B_API_KEY / E2B_API_KEY) — falling back to bubblewrap",
                        session_id[:12],
                    )
                return None
            try:
                sandbox = await get_or_create_sandbox(
                    session_id,
                    api_key=e2b_api_key,
                    template=config.e2b_sandbox_template,
                    timeout=config.e2b_sandbox_timeout,
                    on_timeout=config.e2b_sandbox_on_timeout,
                )
            except Exception as e2b_err:
                logger.error(
                    "[E2B] [%s] Setup failed: %s",
                    session_id[:12],
                    e2b_err,
                    exc_info=True,
                )
                return None

            return sandbox

        async def _fetch_transcript():
            """Download transcript for --resume if applicable."""
            assert session is not None  # narrowed at line 1898
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
                    "%s Transcript download failed, continuing without --resume: %s",
                    log_prefix,
                    transcript_err,
                )
                return None

        e2b_sandbox, (base_system_prompt, understanding), dl = await asyncio.gather(
            _setup_e2b(),
            _build_system_prompt(user_id if not has_history else None),
            _fetch_transcript(),
        )

        use_e2b = e2b_sandbox is not None
        # Append appropriate supplement (Claude gets tool schemas automatically)

        graphiti_enabled = await is_enabled_for_user(user_id)

        graphiti_supplement = get_graphiti_supplement() if graphiti_enabled else ""
        system_prompt = (
            base_system_prompt
            + get_sdk_supplement(use_e2b=use_e2b, cwd=sdk_cwd)
            + graphiti_supplement
        )

        # Warm context: pre-load relevant facts from Graphiti on first turn
        if graphiti_enabled and user_id and len(session.messages) <= 1:
            from backend.copilot.graphiti.context import fetch_warm_context

            warm_ctx = await fetch_warm_context(user_id, message or "")
            if warm_ctx:
                system_prompt += f"\n\n{warm_ctx}"

        # Process transcript download result
        transcript_msg_count = 0
        if dl:
            is_valid = validate_transcript(dl.content)
            dl_lines = dl.content.strip().split("\n") if dl.content else []
            logger.info(
                "%s Downloaded transcript: %dB, %d lines, msg_count=%d, valid=%s",
                log_prefix,
                len(dl.content),
                len(dl_lines),
                dl.message_count,
                is_valid,
            )
            if is_valid:
                # Load previous FULL context into builder
                transcript_content = dl.content
                transcript_builder.load_previous(dl.content, log_prefix=log_prefix)
                resume_file = await asyncio.to_thread(
                    write_transcript_to_tempfile, dl.content, session_id, sdk_cwd
                )
                if resume_file:
                    use_resume = True
                    transcript_msg_count = dl.message_count
                    logger.debug(
                        "%s Using --resume (%dB, msg_count=%d)",
                        log_prefix,
                        len(dl.content),
                        transcript_msg_count,
                    )
            else:
                logger.warning("%s Transcript downloaded but invalid", log_prefix)
                transcript_covers_prefix = False
        elif config.claude_agent_use_resume and user_id and len(session.messages) > 1:
            # No transcript on disk — try to reconstruct a full JSONL from the
            # session.messages stored in the DB.  This gives the Claude CLI
            # proper tool_use/tool_result structural context via --resume
            # instead of the lossy plain-text injection in _build_query_message
            # (which caps tool results at 500 chars and drops call/result IDs).
            prior = session.messages[:-1]
            reconstructed = _session_messages_to_transcript(prior)
            if reconstructed:
                rebuilt_resume = await asyncio.to_thread(
                    write_transcript_to_tempfile, reconstructed, session_id, sdk_cwd
                )
                if rebuilt_resume:
                    use_resume = True
                    resume_file = rebuilt_resume
                    transcript_msg_count = len(prior)
                    transcript_content = reconstructed
                    transcript_builder.load_previous(
                        reconstructed, log_prefix=log_prefix
                    )
                    transcript_covers_prefix = True
                    logger.info(
                        "%s Reconstructed transcript from %d session messages "
                        "for --resume (no previous transcript file)",
                        log_prefix,
                        len(prior),
                    )
                else:
                    logger.warning(
                        "%s Transcript reconstruction failed — write_transcript_to_tempfile"
                        " returned None (%d messages)",
                        log_prefix,
                        len(prior),
                    )
                    transcript_covers_prefix = False
            else:
                logger.warning(
                    "%s No transcript available and reconstruction produced empty"
                    " output (%d messages in session)",
                    log_prefix,
                    len(session.messages),
                )
                transcript_covers_prefix = False

        yield StreamStart(messageId=message_id, sessionId=session_id)

        set_execution_context(
            user_id,
            session,
            sandbox=e2b_sandbox,
            sdk_cwd=sdk_cwd,
            permissions=permissions,
        )

        # Fail fast when no API credentials are available at all.
        # sdk_cwd routes the CLI's temp dir into the per-session workspace
        # so sub-agent output files land inside sdk_cwd (see build_sdk_env).
        sdk_env = build_sdk_env(session_id=session_id, user_id=user_id, sdk_cwd=sdk_cwd)

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

        if permissions is not None:
            allowed, disallowed = apply_tool_permissions(permissions, use_e2b=use_e2b)
        else:
            allowed = get_copilot_tool_names(use_e2b=use_e2b)
            disallowed = get_sdk_disallowed_tools(use_e2b=use_e2b)

        def _on_stderr(line: str) -> None:
            """Log a stderr line emitted by the Claude CLI subprocess."""
            nonlocal fallback_model_activated_per_attempt
            sid = session_id[:12] if session_id else "?"
            logger.info("[SDK] [%s] CLI stderr: %s", sid, line.rstrip())
            # Detect SDK fallback-model activation via the module-level pure
            # helper so the detection logic can be unit-tested independently.
            # Sets the per-attempt flag which is preserved across transient
            # retries so the user notification is never lost.
            if not fallback_model_activated_per_attempt and _is_fallback_stderr(line):
                fallback_model_activated_per_attempt = True
                logger.warning(
                    "[SDK] [%s] Fallback model activated — primary model "
                    "overloaded, switching to fallback",
                    sid,
                )

        sdk_options_kwargs: dict[str, Any] = {
            "system_prompt": system_prompt,
            "mcp_servers": {"copilot": mcp_server},
            "allowed_tools": allowed,
            "disallowed_tools": disallowed,
            "hooks": security_hooks,
            "cwd": sdk_cwd,
            "max_buffer_size": config.claude_agent_max_buffer_size,
            "stderr": _on_stderr,
            # --- P0 guardrails ---
            # fallback_model: SDK auto-retries with this cheaper model on
            # 529 (overloaded) errors, avoiding user-visible failures.
            "fallback_model": _resolve_fallback_model(),
            # max_turns: hard cap on agentic tool-use loops per query to
            # prevent runaway execution from burning budget.
            "max_turns": config.claude_agent_max_turns,
            # max_budget_usd: per-query spend ceiling enforced by the CLI.
            "max_budget_usd": config.claude_agent_max_budget_usd,
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
        _user_tier = await get_user_tier(user_id) if user_id else None
        _otel_metadata: dict[str, str] = {
            "resume": str(use_resume),
            "conversation_turn": str(turn),
        }
        if _user_tier:
            _otel_metadata["subscription_tier"] = _user_tier.value

        _otel_ctx = propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            trace_name="copilot-sdk",
            tags=["sdk"],
            metadata=_otel_metadata,
        )
        _otel_ctx.__enter__()

        current_message = message or ""
        if not current_message and session.messages:
            last_user = [m for m in session.messages if m.role == "user"]
            if last_user:
                current_message = last_user[-1].content or ""

        # Capture the message count *before* draining so _build_query_message
        # can compute the gap slice without including the newly-drained pending
        # messages.  Pending messages are both appended to session.messages AND
        # concatenated into current_message; without the ceiling the gap slice
        # would extend into the pending messages and duplicate them in the
        # model's input context (gap_context + current_message both containing
        # them).
        _pre_drain_msg_count = len(session.messages)

        # Drain any messages the user queued via POST /messages/pending
        # while the previous turn was running (or since the session was
        # idle).  Messages are drained ATOMICALLY — one LPOP with count
        # removes them all at once, so a concurrent push lands *after*
        # the drain and stays queued for the next turn instead of being
        # lost between LPOP and clear.  File IDs and context are
        # preserved via format_pending_as_user_message.
        #
        # The drained content is concatenated into ``current_message``
        # so the SDK CLI sees it in the new user message, AND appended
        # directly to ``session.messages`` (no dedup — pending messages are
        # atomically-popped from Redis and are never stale-cache duplicates)
        # so the durable transcript records it too.  Session is persisted
        # immediately after the drain so a crash doesn't lose the messages.
        # The endpoint deliberately does NOT persist to session.messages —
        # Redis is the single source of truth until this drain runs.
        try:
            pending_at_start = await drain_pending_messages(session_id)
        except Exception:
            logger.warning(
                "%s drain_pending_messages failed at turn start, skipping",
                log_prefix,
                exc_info=True,
            )
            pending_at_start = []
        if pending_at_start:
            logger.info(
                "%s Draining %d pending message(s) at turn start",
                log_prefix,
                len(pending_at_start),
            )
            pending_texts: list[str] = [
                format_pending_as_user_message(pm)["content"] for pm in pending_at_start
            ]
            for pt in pending_texts:
                # Append directly — pending messages are atomically-popped from
                # Redis and are never stale-cache duplicates, so the
                # maybe_append_user_message dedup is wrong here.
                session.messages.append(ChatMessage(role="user", content=pt))
            if current_message.strip():
                current_message = current_message + "\n\n" + "\n\n".join(pending_texts)
            else:
                current_message = "\n\n".join(pending_texts)
            # Persist immediately so a crash between here and the finally block
            # doesn't lose messages that were already drained from Redis.
            try:
                session = await upsert_chat_session(session)
            except Exception as _persist_err:
                logger.warning(
                    "%s Failed to persist drained pending messages: %s",
                    log_prefix,
                    _persist_err,
                )

        if not current_message.strip():
            yield StreamError(
                errorText="Message cannot be empty.",
                code="empty_prompt",
            )
            return

        # Strip any user-injected <user_context> tags from current_message.
        # On --resume, current_message may come from session history which was
        # already sanitized on the original turn; strip again as defence-in-depth.
        current_message = strip_user_context_tags(current_message)

        # On the first turn inject user context into the message before building
        # the query so that _build_query_message sees the full prefixed content.
        # The system prompt is now static (same for all users) so the LLM can
        # cache it across sessions.
        #
        # On resume (has_history=True) we intentionally skip re-injection: the
        # transcript already contains the <user_context> prefix from the original
        # turn (persisted to the DB in inject_user_context), so the SDK replay
        # carries context continuity without us prepending it again.  Adding it
        # a second time would duplicate the block and inflate tokens.
        if not has_history:
            prefixed_message = await inject_user_context(
                understanding, current_message, session_id, session.messages
            )
            if prefixed_message is not None:
                current_message = prefixed_message

        query_message, was_compacted = await _build_query_message(
            current_message,
            session,
            use_resume,
            transcript_msg_count,
            session_id,
            session_msg_ceiling=_pre_drain_msg_count,
        )
        # If files are attached, prepare them: images become vision
        # content blocks in the user message, other files go to sdk_cwd.
        attachments = await _prepare_file_attachments(
            file_ids or [], user_id or "", session_id, sdk_cwd
        )
        if attachments.hint:
            query_message = f"{query_message}\n\n{attachments.hint}"

        tried_compaction = False

        # Build the per-request context carrier (shared across attempts).
        # Scalar fields are immutable; session/compaction/lock are shared
        # mutable references (see `_StreamContext` docstring for details).
        stream_ctx = _StreamContext(
            session=session,
            session_id=session_id,
            log_prefix=log_prefix,
            sdk_cwd=sdk_cwd,
            current_message=current_message,
            file_ids=file_ids,
            message_id=message_id,
            attachments=attachments,
            compaction=compaction,
            lock=lock,
        )

        # ---------------------------------------------------------------
        # Retry loop: original → compacted → no transcript
        # ---------------------------------------------------------------
        ended_with_stream_error = False
        attempts_exhausted = False
        transient_exhausted = False
        stream_err: Exception | None = None

        transient_retries = 0
        max_transient_retries = config.claude_agent_max_transient_retries
        # Preserved across transient retries so the fallback-model notification
        # is not lost when a retry resets local per-attempt variables.  Reset
        # only on context-level attempt changes (same guard as transient_retries).
        fallback_model_activated_per_attempt = False
        fallback_notified_per_attempt = False

        state = _RetryState(
            options=options,
            query_message=query_message,
            was_compacted=was_compacted,
            use_resume=use_resume,
            resume_file=resume_file,
            transcript_msg_count=transcript_msg_count,
            adapter=adapter,
            transcript_builder=transcript_builder,
            usage=_TokenUsage(),
        )

        attempt = 0
        _last_reset_attempt = -1
        while attempt < _MAX_STREAM_ATTEMPTS:
            # Reset transient retry counter per context-level attempt so
            # each attempt (original, compacted, no-transcript) gets the
            # full retry budget for transient errors.
            # Only reset when the attempt number actually changes —
            # transient retries `continue` back to the loop top without
            # incrementing `attempt`, so resetting unconditionally would
            # create an infinite retry loop.
            if attempt != _last_reset_attempt:
                transient_retries = 0
                fallback_model_activated_per_attempt = False
                fallback_notified_per_attempt = False
                _last_reset_attempt = attempt
            # Clear any stale stash signal from the previous attempt so
            # wait_for_stash() doesn't fire prematurely on a leftover event.
            reset_stash_event()
            # Reset tool-level circuit breaker so failures from a previous
            # (rolled-back) attempt don't carry over to the fresh attempt.
            reset_tool_failure_counters()
            if attempt > 0:
                logger.info(
                    "%s Retrying with reduced context (%d/%d)",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                )
                yield StreamStatus(message="Optimizing conversation context\u2026")

                ctx = await _reduce_context(
                    transcript_content,
                    tried_compaction,
                    session_id,
                    sdk_cwd,
                    log_prefix,
                )
                state.transcript_builder = ctx.builder
                state.use_resume = ctx.use_resume
                state.resume_file = ctx.resume_file
                tried_compaction = ctx.tried_compaction
                state.transcript_msg_count = 0
                if ctx.transcript_lost:
                    skip_transcript_upload = True

                # Rebuild SDK options and query for the reduced context
                sdk_options_kwargs_retry = dict(sdk_options_kwargs)
                if ctx.use_resume and ctx.resume_file:
                    sdk_options_kwargs_retry["resume"] = ctx.resume_file
                elif "resume" in sdk_options_kwargs_retry:
                    del sdk_options_kwargs_retry["resume"]
                state.options = ClaudeAgentOptions(**sdk_options_kwargs_retry)  # type: ignore[arg-type]  # dynamic kwargs
                state.query_message, state.was_compacted = await _build_query_message(
                    current_message,
                    session,
                    state.use_resume,
                    state.transcript_msg_count,
                    session_id,
                    session_msg_ceiling=_pre_drain_msg_count,
                )
                if attachments.hint:
                    state.query_message = f"{state.query_message}\n\n{attachments.hint}"
                state.adapter = SDKResponseAdapter(
                    message_id=message_id, session_id=session_id
                )
                # Reset token accumulators so a failed attempt's partial
                # usage is not double-counted in the successful attempt.
                state.usage.reset()

            pre_attempt_msg_count = len(session.messages)
            # Snapshot transcript builder state — it maintains an
            # independent _entries list from session.messages, so rolling
            # back session.messages alone would leave duplicate entries
            # from the failed attempt in the uploaded transcript.
            transcript_snap = state.transcript_builder.snapshot()
            events_yielded = 0

            try:
                async for event in _run_stream_attempt(stream_ctx, state):
                    if not isinstance(event, _EPHEMERAL_EVENT_TYPES):
                        events_yielded += 1
                    # Emit a one-time StreamStatus when the SDK switches
                    # to the fallback model (detected via stderr).  The flag
                    # is preserved across transient retries (reset only on
                    # context-level attempt change) so the notification is
                    # not lost if the activation occurs during a failed sub-
                    # attempt that later retries successfully.
                    if (
                        fallback_model_activated_per_attempt
                        and not fallback_notified_per_attempt
                    ):
                        fallback_notified_per_attempt = True
                        yield StreamStatus(
                            message="Primary model overloaded — "
                            "using fallback model for this request"
                        )
                    yield event
                break  # Stream completed — exit retry loop
            except asyncio.CancelledError:
                logger.warning(
                    "%s Streaming cancelled (attempt %d/%d)",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                )
                raise
            except _HandledStreamError as exc:
                # _run_stream_attempt already yielded a StreamError and
                # appended an error marker.  We only need to rollback
                # session messages and set the error flag — do NOT set
                # stream_err so the post-loop code won't emit a
                # duplicate StreamError.
                session.messages = session.messages[:pre_attempt_msg_count]
                state.transcript_builder.restore(transcript_snap)
                # Check if this is a transient error we can retry with backoff.
                # exc.code is the only reliable signal — str(exc) is always the
                # static "Stream error handled — StreamError already yielded" message.
                if exc.code == "transient_api_error":
                    backoff, transient_retries = _next_transient_backoff(
                        events_yielded, transient_retries, max_transient_retries
                    )
                    if backoff is not None:
                        logger.warning(
                            "%s Transient error — retrying in %ds (%d/%d)",
                            log_prefix,
                            backoff,
                            transient_retries,
                            max_transient_retries,
                        )
                        async for evt in _do_transient_backoff(
                            backoff, state, message_id, session_id
                        ):
                            yield evt
                        continue  # retry the same context-level attempt
                logger.warning(
                    "%s Stream error handled in attempt "
                    "(attempt %d/%d, code=%s, events_yielded=%d)",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                    exc.code or "transient",
                    events_yielded,
                )
                # transcript_builder still contains entries from the aborted
                # attempt that no longer match session.messages.  Skip upload
                # so a future --resume doesn't replay rolled-back content.
                skip_transcript_upload = True
                # Re-append the error marker so it survives the rollback
                # and is persisted by the finally block (see #2947655365).
                # Use the specific error message from the attempt (e.g.
                # circuit breaker msg) rather than always the generic one.
                _append_error_marker(
                    session,
                    exc.error_msg or FRIENDLY_TRANSIENT_MSG,
                    retryable=True,
                )
                ended_with_stream_error = True
                # For transient errors the StreamError was deliberately NOT
                # yielded inside _run_stream_attempt (already_yielded=False)
                # so the client didn't see a premature error flash.  Yield it
                # now that we know retries are exhausted.
                # For non-transient errors (circuit breaker, idle timeout)
                # already_yielded=True — do NOT yield again.
                if not exc.already_yielded:
                    yield StreamError(
                        errorText=exc.error_msg or FRIENDLY_TRANSIENT_MSG,
                        code=exc.code or "transient_api_error",
                    )
                break
            except Exception as e:
                stream_err = e
                is_context_error = _is_prompt_too_long(e)
                is_transient = is_transient_api_error(str(e))
                logger.warning(
                    "%s Stream error (attempt %d/%d, context_error=%s, "
                    "transient=%s, events_yielded=%d): %s",
                    log_prefix,
                    attempt + 1,
                    _MAX_STREAM_ATTEMPTS,
                    is_context_error,
                    is_transient,
                    events_yielded,
                    stream_err,
                    exc_info=True,
                )
                session.messages = session.messages[:pre_attempt_msg_count]
                state.transcript_builder.restore(transcript_snap)
                if events_yielded > 0:
                    # Events were already sent to the frontend and cannot be
                    # unsent.  Retrying would produce duplicate/inconsistent
                    # output, so treat this as a final error.
                    logger.warning(
                        "%s Not retrying — %d events already yielded",
                        log_prefix,
                        events_yielded,
                    )
                    skip_transcript_upload = True
                    ended_with_stream_error = True
                    break
                # Transient API errors (ECONNRESET, 429, 5xx) — retry
                # with exponential backoff via the shared helper.
                if is_transient:
                    backoff, transient_retries = _next_transient_backoff(
                        events_yielded, transient_retries, max_transient_retries
                    )
                    if backoff is not None:
                        logger.warning(
                            "%s Transient exception — retrying in %ds (%d/%d)",
                            log_prefix,
                            backoff,
                            transient_retries,
                            max_transient_retries,
                        )
                        async for evt in _do_transient_backoff(
                            backoff, state, message_id, session_id
                        ):
                            yield evt
                        continue  # retry same context-level attempt
                    # Retries exhausted — persist retryable marker so the
                    # frontend shows "Try again" after refresh.
                    # Mirrors the _HandledStreamError exhausted-retry path
                    # at line ~2310.
                    transient_exhausted = True
                    skip_transcript_upload = True
                    _append_error_marker(
                        session, FRIENDLY_TRANSIENT_MSG, retryable=True
                    )
                    ended_with_stream_error = True
                    break

                if not is_context_error:
                    # Non-context, non-transient errors (auth, fatal)
                    # should not trigger compaction — surface immediately.
                    skip_transcript_upload = True
                    ended_with_stream_error = True
                    break
                attempt += 1  # advance to next context-level attempt
                continue
        else:
            # while condition became False — all attempts exhausted without
            # break.  skip_transcript_upload is already set by _reduce_context
            # when the transcript was dropped (transcript_lost=True).
            ended_with_stream_error = True
            attempts_exhausted = True
            logger.error(
                "%s All %d query attempts exhausted: %s",
                log_prefix,
                _MAX_STREAM_ATTEMPTS,
                stream_err,
            )

        if ended_with_stream_error and state is not None:
            # Flush any unresolved tool calls so the frontend can close
            # stale UI elements (e.g. spinners) that were started before
            # the exception interrupted the stream.
            error_flush: list[StreamBaseResponse] = []
            state.adapter._end_text_if_open(error_flush)
            if state.adapter.has_unresolved_tool_calls:
                logger.warning(
                    "%s Flushing %d unresolved tool(s) after stream error",
                    log_prefix,
                    len(state.adapter.current_tool_calls)
                    - len(state.adapter.resolved_tool_calls),
                )
                state.adapter._flush_unresolved_tool_calls(error_flush)
            for response in error_flush:
                yield response

        if ended_with_stream_error and stream_err is not None:
            # Use distinct error codes depending on how the loop ended:
            # • "all_attempts_exhausted" — context compaction ran out of room
            # • "transient_api_error" — 429/5xx/ECONNRESET retries exhausted
            # • "sdk_stream_error" — non-context, non-transient fatal error
            safe_err = str(stream_err).replace("\n", " ").replace("\r", "")[:500]
            if attempts_exhausted:
                error_text = (
                    "Your conversation is too long. "
                    "Please start a new chat or clear some history."
                )
                error_code = "all_attempts_exhausted"
            elif transient_exhausted:
                error_text = FRIENDLY_TRANSIENT_MSG
                error_code = "transient_api_error"
            else:
                error_text = _friendly_error_text(safe_err)
                error_code = "sdk_stream_error"
            yield StreamError(errorText=error_text, code=error_code)

        # Copy token usage from retry state to outer-scope accumulators
        # so the finally block can persist them.
        if state is not None:
            turn_prompt_tokens = state.usage.prompt_tokens
            turn_completion_tokens = state.usage.completion_tokens
            turn_cache_read_tokens = state.usage.cache_read_tokens
            turn_cache_creation_tokens = state.usage.cache_creation_tokens
            turn_cost_usd = state.usage.cost_usd

        # Emit token usage to the client (must be in try to reach SSE stream).
        # Session persistence of usage is in finally to stay consistent with
        # rate-limit recording even if an exception interrupts between here
        # and the finally block.
        if turn_prompt_tokens > 0 or turn_completion_tokens > 0:
            # total_tokens = prompt (uncached input) + completion (output).
            # Cache tokens are tracked separately and excluded from total
            # so that the semantics match the baseline path (OpenRouter)
            # which folds cache into prompt_tokens. Keeping total_tokens
            # = prompt + completion everywhere makes cross-path comparisons
            # and session-level aggregation consistent.
            total_tokens = turn_prompt_tokens + turn_completion_tokens
            yield StreamUsage(
                prompt_tokens=turn_prompt_tokens,
                completion_tokens=turn_completion_tokens,
                total_tokens=total_tokens,
                cache_read_tokens=turn_cache_read_tokens,
                cache_creation_tokens=turn_cache_creation_tokens,
            )

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
    except GeneratorExit:
        # GeneratorExit is raised when the async generator is closed by the
        # caller (e.g. client disconnect, page refresh).  We MUST release the
        # stream lock here because the ``finally`` block at the end of this
        # function may not execute when GeneratorExit propagates through nested
        # async generators.  Without this, the lock stays held for its full TTL
        # and the user sees "Another stream is already active" on every retry.
        logger.warning("%s GeneratorExit — releasing stream lock", log_prefix)
        await lock.release()
        raise
    except BaseException as e:
        # Catch BaseException to handle both Exception and CancelledError
        # (CancelledError inherits from BaseException in Python 3.8+)
        if isinstance(e, asyncio.CancelledError):
            logger.warning("%s Session cancelled", log_prefix)
            error_msg = "Operation cancelled"
        else:
            error_msg = str(e) or type(e).__name__
            # SDK cleanup errors are expected during client disconnect —
            # log as warning rather than error to reduce Sentry noise.
            # These are normally caught by _safe_close_sdk_client but
            # can escape in edge cases (e.g. GeneratorExit timing).
            if _is_sdk_disconnect_error(e):
                logger.warning(
                    "%s SDK cleanup error (client disconnect): %s",
                    log_prefix,
                    error_msg,
                )
            else:
                logger.error("%s Error: %s", log_prefix, error_msg, exc_info=True)

        is_transient = is_transient_api_error(error_msg)
        if is_transient:
            display_msg, code = FRIENDLY_TRANSIENT_MSG, "transient_api_error"
        else:
            display_msg, code = error_msg, "sdk_error"

        # Append error marker to session (non-invasive text parsing approach).
        # The finally block will persist the session with this error marker.
        # Skip if a marker was already appended inside the stream loop
        # (ended_with_stream_error) to avoid duplicate stale markers.
        if not ended_with_stream_error:
            _append_error_marker(session, display_msg, retryable=is_transient)
            logger.debug(
                "%s Appended error marker, will be persisted in finally",
                log_prefix,
            )

        # Yield StreamError for immediate feedback (only for non-cancellation errors)
        # Skip for CancelledError and SDK disconnect cleanup errors — these
        # are not actionable by the user and the SSE connection is already dead.
        is_cancellation = isinstance(
            e, asyncio.CancelledError
        ) or _is_sdk_disconnect_error(e)
        if not is_cancellation:
            yield StreamError(errorText=display_msg, code=code)

        raise
    finally:
        # Pending messages are drained atomically at the start of each
        # turn (see drain_pending_messages call above), so there's
        # nothing to clean up here — any message pushed after that
        # point belongs to the next turn.

        # --- Close OTEL context (with cost attributes) ---
        if _otel_ctx is not None:
            try:
                span = otel_trace.get_current_span()
                if span and span.is_recording():
                    span.set_attribute("gen_ai.usage.prompt_tokens", turn_prompt_tokens)
                    span.set_attribute(
                        "gen_ai.usage.completion_tokens", turn_completion_tokens
                    )
                    span.set_attribute(
                        "gen_ai.usage.cache_read_tokens", turn_cache_read_tokens
                    )
                    span.set_attribute(
                        "gen_ai.usage.cache_creation_tokens",
                        turn_cache_creation_tokens,
                    )
                    if turn_cost_usd is not None:
                        span.set_attribute("gen_ai.usage.cost_usd", turn_cost_usd)
            except Exception:
                logger.debug("Failed to set OTEL cost attributes", exc_info=True)
            try:
                _otel_ctx.__exit__(*sys.exc_info())
            except Exception:
                logger.warning("OTEL context teardown failed", exc_info=True)

        # --- Persist token usage to session + rate-limit counters ---
        # Both must live in finally so they stay consistent even when an
        # exception interrupts the try block after StreamUsage was yielded.
        await persist_and_record_usage(
            session=session,
            user_id=user_id,
            prompt_tokens=turn_prompt_tokens,
            completion_tokens=turn_completion_tokens,
            cache_read_tokens=turn_cache_read_tokens,
            cache_creation_tokens=turn_cache_creation_tokens,
            log_prefix=log_prefix,
            cost_usd=turn_cost_usd,
            model=config.model,
            provider="anthropic",
        )

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

        # --- Pause E2B sandbox to stop billing between turns ---
        # Fire-and-forget: pausing is best-effort and must not block the
        # response or the transcript upload.  The task is anchored to
        # _background_tasks to prevent garbage collection.
        # Use pause_sandbox_direct to skip the Redis lookup and reconnect
        # round-trip — e2b_sandbox is the live object from this turn.
        if e2b_sandbox is not None:
            task = asyncio.create_task(pause_sandbox_direct(e2b_sandbox, session_id))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)

        # --- Graphiti: ingest conversation turn for temporal memory ---
        if graphiti_enabled and user_id and message and is_user_message:
            from backend.copilot.graphiti.ingest import enqueue_conversation_turn

            _ingest_task = asyncio.create_task(
                enqueue_conversation_turn(user_id, session_id, message)
            )
            _background_tasks.add(_ingest_task)
            _ingest_task.add_done_callback(_background_tasks.discard)

        # --- Upload transcript for next-turn --resume ---
        # TranscriptBuilder is the single source of truth.  It mirrors the
        # CLI's active context: on compaction, replace_entries() syncs it
        # with the compacted session file.  No CLI file read needed here.
        if skip_transcript_upload:
            logger.warning(
                "%s Skipping transcript upload — transcript was dropped "
                "during prompt-too-long recovery",
                log_prefix,
            )
        elif (
            config.claude_agent_use_resume
            and user_id
            and session is not None
            and state is not None
        ):
            try:
                transcript_upload_content = state.transcript_builder.to_jsonl()
                entry_count = state.transcript_builder.entry_count

                if not transcript_upload_content:
                    logger.warning(
                        "%s No transcript to upload (builder empty)", log_prefix
                    )
                elif not validate_transcript(transcript_upload_content):
                    logger.warning(
                        "%s Transcript invalid, skipping upload (entries=%d)",
                        log_prefix,
                        entry_count,
                    )
                elif not transcript_covers_prefix:
                    logger.warning(
                        "%s Skipping transcript upload — builder does not "
                        "cover full session prefix (entries=%d, session=%d)",
                        log_prefix,
                        entry_count,
                        len(session.messages),
                    )
                else:
                    logger.info(
                        "%s Uploading transcript (entries=%d, bytes=%d)",
                        log_prefix,
                        entry_count,
                        len(transcript_upload_content),
                    )
                    await asyncio.shield(
                        upload_transcript(
                            user_id=user_id,
                            session_id=session_id,
                            content=transcript_upload_content,
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

        try:
            if sdk_cwd:
                await _cleanup_sdk_tool_results(sdk_cwd)
        except Exception:
            logger.warning("%s SDK cleanup failed", log_prefix, exc_info=True)
        finally:
            # Release stream lock to allow new streams for this session
            await lock.release()
