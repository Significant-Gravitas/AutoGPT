"""Tool adapter for wrapping existing CoPilot tools as Claude Agent SDK MCP tools.

This module provides the adapter layer that converts existing BaseTool implementations
into in-process MCP tools that can be used with the Claude Agent SDK.
"""

import asyncio
import itertools
import json
import logging
import os
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import create_sdk_mcp_server, tool
from mcp.types import ToolAnnotations

from backend.copilot.context import (
    _current_permissions,
    _current_project_dir,
    _current_sandbox,
    _current_sdk_cwd,
    _current_session,
    _current_user_id,
    _encode_cwd_for_cli,
    get_execution_context,
    is_sdk_tool_path,
)
from backend.copilot.model import ChatSession
from backend.copilot.sdk.file_ref import (
    FileRefExpansionError,
    expand_file_refs_in_args,
    read_file_bytes,
)
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools.base import BaseTool
from backend.util.truncate import truncate

from .e2b_file_tools import (
    E2B_FILE_TOOL_NAMES,
    E2B_FILE_TOOLS,
    EDIT_TOOL_DESCRIPTION,
    EDIT_TOOL_NAME,
    EDIT_TOOL_SCHEMA,
    READ_TOOL_DESCRIPTION,
    READ_TOOL_NAME,
    READ_TOOL_SCHEMA,
    WRITE_TOOL_DESCRIPTION,
    WRITE_TOOL_NAME,
    WRITE_TOOL_SCHEMA,
    bridge_and_annotate,
    get_edit_tool_handler,
    get_read_tool_handler,
    get_write_tool_handler,
)

if TYPE_CHECKING:
    from e2b import AsyncSandbox

    from backend.copilot.permissions import CopilotPermissions

logger = logging.getLogger(__name__)

# Max MCP response size in chars — sized to the Claude CLI's internal cap.
#
# The CLI has a default ``maxResultSizeChars = 1e5`` (100K chars) annotation
# for MCP tool results, but the actual trigger is TOKEN-based (see
# ``sizeEstimateTokens`` in the bundled CLI at ``tengu_mcp_large_result_handled``)
# and fires around 20–25K tokens.  For JSON-heavy tool output (~3–4 chars/token)
# that lands anywhere from ~60K to ~100K chars in practice; we've observed the
# error path at 81K chars in production.  When it fires, the CLI persists the
# full output to disk and REPLACES the returned content with a synthetic
# ``"Error: result (N characters) exceeds maximum allowed tokens. Output has
# been saved to …"`` message — which destroys any `<user_follow_up>` block
# we injected.
#
# 70K gives us headroom below the observed 81K trigger and leaves ~6K for the
# follow-up injection plus CLI wire overhead.  Oversized content is still
# reachable via ``read_tool_result`` against the persisted disk file; only
# the inline reply to this specific call is truncated.
_MCP_MAX_CHARS = 70_000

# MCP server naming - the SDK prefixes tool names as "mcp__{server_name}__{tool}"
MCP_SERVER_NAME = "copilot"
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

# Fields stripped from the MCP tool result JSON before it is forwarded to the LLM.
# These fields would reveal execution mode (e.g. dry_run) to the model.
# Stripping happens AFTER the tool output is stashed for the frontend SSE stream,
# so StreamToolOutputAvailable still receives the full output including these fields.
_STRIP_FROM_LLM: frozenset[str] = frozenset(["is_dry_run"])


# Stash for MCP tool outputs before the SDK potentially truncates them.
# Keyed by tool_name → full output string. Consumed (popped) by the
# response adapter when it builds StreamToolOutputAvailable.
_pending_tool_outputs: ContextVar[dict[str, list[str]]] = ContextVar(
    "pending_tool_outputs",
    default=None,  # type: ignore[arg-type]
)
# Event signaled whenever stash_pending_tool_output() adds a new entry.
# Used by the streaming loop to wait for PostToolUse hooks to complete
# instead of sleeping an arbitrary duration.  The SDK fires hooks via
# start_soon (fire-and-forget) so the next message can arrive before
# the hook stashes its output — this event bridges that gap.
_stash_event: ContextVar[asyncio.Event | None] = ContextVar(
    "_stash_event", default=None
)

# Circuit breaker: tracks consecutive tool failures to detect infinite retry loops.
# When a tool is called repeatedly with empty/identical args and keeps failing,
# this counter is incremented.  After _MAX_CONSECUTIVE_TOOL_FAILURES identical
# failures the tool handler returns a hard-stop message instead of the raw error.
_MAX_CONSECUTIVE_TOOL_FAILURES = 3
_consecutive_tool_failures: ContextVar[dict[str, int]] = ContextVar(
    "_consecutive_tool_failures",
    default=None,  # type: ignore[arg-type]
)


def set_execution_context(
    user_id: str | None,
    session: ChatSession,
    sandbox: "AsyncSandbox | None" = None,
    sdk_cwd: str | None = None,
    permissions: "CopilotPermissions | None" = None,
) -> None:
    """Set the execution context for tool calls.

    This must be called before streaming begins to ensure tools have access
    to user_id, session, and (optionally) an E2B sandbox for bash execution.

    Args:
        user_id: Current user's ID.
        session: Current chat session.
        sandbox: Optional E2B sandbox; when set, bash_exec routes commands there.
        sdk_cwd: SDK working directory; used to scope tool-results reads.
        permissions: Optional capability filter restricting tools/blocks.
    """
    _current_user_id.set(user_id)
    _current_session.set(session)
    _current_sandbox.set(sandbox)
    _current_sdk_cwd.set(sdk_cwd or "")
    _current_project_dir.set(_encode_cwd_for_cli(sdk_cwd) if sdk_cwd else "")
    _current_permissions.set(permissions)
    _pending_tool_outputs.set({})
    _stash_event.set(asyncio.Event())
    _consecutive_tool_failures.set({})


def reset_stash_event() -> None:
    """Clear any stale stash signal left over from a previous stream attempt.

    ``_stash_event`` is set once per session in ``set_execution_context`` and
    reused across retry attempts.  A PostToolUse hook from a failed attempt may
    leave the event set; calling this at the start of each retry prevents
    ``wait_for_stash`` from returning prematurely on a stale signal.
    """
    event = _stash_event.get(None)
    if event is not None:
        event.clear()


def reset_tool_failure_counters() -> None:
    """Reset all tool-level circuit breaker counters.

    Called at the start of each SDK retry attempt so that failure counts
    from a previous (rolled-back) attempt do not carry over and prematurely
    trip the breaker on a fresh attempt with different context.
    """
    _consecutive_tool_failures.set({})


def pop_pending_tool_output(tool_name: str) -> str | None:
    """Pop and return the oldest stashed output for *tool_name*.

    The SDK CLI may truncate large tool results (writing them to disk and
    replacing the content with a file reference). This stash keeps the
    original MCP output so the response adapter can forward it to the
    frontend for proper widget rendering.

    Uses a FIFO queue per tool name so duplicate calls to the same tool
    in one turn each get their own output.

    Returns ``None`` if nothing was stashed for *tool_name*.
    """
    pending = _pending_tool_outputs.get(None)
    if pending is None:
        return None
    queue = pending.get(tool_name)
    if not queue:
        pending.pop(tool_name, None)
        return None
    value = queue.pop(0)
    if not queue:
        del pending[tool_name]
    return value


def stash_pending_tool_output(tool_name: str, output: Any) -> None:
    """Stash tool output for later retrieval by the response adapter.

    Used by the PostToolUse hook to capture SDK built-in tool outputs
    (WebSearch, Read, etc.) that aren't available through the MCP stash
    mechanism in ``_execute_tool_sync``.

    Appends to a FIFO queue per tool name so multiple calls to the same
    tool in one turn are all preserved.
    """
    pending = _pending_tool_outputs.get(None)
    if pending is None:
        return
    if isinstance(output, str):
        text = output
    else:
        try:
            text = json.dumps(output)
        except (TypeError, ValueError):
            text = str(output)
    pending.setdefault(tool_name, []).append(text)
    # Signal any waiters that new output is available.
    event = _stash_event.get(None)
    if event is not None:
        event.set()


async def wait_for_stash(timeout: float = 2.0) -> bool:
    """Wait for a PostToolUse hook to stash tool output.

    The SDK fires PostToolUse hooks asynchronously via ``start_soon()`` —
    the next message (AssistantMessage/ResultMessage) can arrive before the
    hook completes and stashes its output.  This function bridges that gap
    by waiting on the ``_stash_event``, which is signaled by
    :func:`stash_pending_tool_output`.

    Uses ``asyncio.Event.wait()`` so it returns the instant the hook signals —
    the timeout is purely a safety net for the case where the hook never fires.
    Returns ``True`` if the stash signal was received, ``False`` on timeout.
    """
    event = _stash_event.get(None)
    if event is None:
        return False
    # Fast path: hook already completed before we got here.
    if event.is_set():
        event.clear()
        return True
    # Slow path: block until the hook signals or the safety timeout expires.
    try:
        async with asyncio.timeout(timeout):
            await event.wait()
        event.clear()
        return True
    except TimeoutError:
        return False


async def _execute_tool_sync(
    base_tool: BaseTool,
    user_id: str | None,
    session: ChatSession,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Execute a tool inline and return an MCP-formatted response.

    The call runs to completion — no per-handler timeout, no parking. The
    stream-level idle timer in ``_run_stream_attempt`` pauses while a tool
    is pending, so a long sub-AutoPilot / graph execution doesn't trip the
    30-min idle safety net (SECRT-2247). A genuine hang is handled by the
    broader session lifecycle (user closes the tab / cancel endpoint).
    """
    effective_id = f"sdk-{uuid.uuid4().hex[:12]}"
    result = await base_tool.execute(
        user_id=user_id,
        session=session,
        tool_call_id=effective_id,
        **args,
    )

    text = (
        result.output if isinstance(result.output, str) else json.dumps(result.output)
    )

    return {
        "content": [{"type": "text", "text": text}],
        "isError": not result.success,
    }


def _mcp_error(message: str) -> dict[str, Any]:
    return {
        "content": [
            {"type": "text", "text": json.dumps({"error": message, "type": "error"})}
        ],
        "isError": True,
    }


def _failure_key(tool_name: str, args: dict[str, Any]) -> str:
    """Compute a stable fingerprint for (tool_name, args) used by the circuit breaker."""
    args_key = json.dumps(args, sort_keys=True, default=str)
    return f"{tool_name}:{args_key}"


def _check_circuit_breaker(tool_name: str, args: dict[str, Any]) -> str | None:
    """Check if a tool has hit the consecutive failure limit.

    Tracks failures keyed by (tool_name, args_fingerprint). Returns an error
    message if the circuit breaker has tripped, or None if the call should proceed.
    """
    tracker = _consecutive_tool_failures.get(None)
    if tracker is None:
        return None

    key = _failure_key(tool_name, args)
    count = tracker.get(key, 0)
    if count >= _MAX_CONSECUTIVE_TOOL_FAILURES:
        logger.warning(
            "Circuit breaker tripped for tool %s after %d consecutive "
            "identical failures (args=%s)",
            tool_name,
            count,
            key[len(tool_name) + 1 :][:200],
        )
        return (
            f"STOP: Tool '{tool_name}' has failed {count} consecutive times with "
            f"the same arguments. Do NOT retry this tool call. "
            f"If you were trying to write content to a file, instead respond with "
            f"the content directly as a text message to the user."
        )
    return None


def _record_tool_failure(tool_name: str, args: dict[str, Any]) -> None:
    """Record a tool failure for circuit breaker tracking."""
    tracker = _consecutive_tool_failures.get(None)
    if tracker is None:
        return
    key = _failure_key(tool_name, args)
    tracker[key] = tracker.get(key, 0) + 1


def _clear_tool_failures(tool_name: str) -> None:
    """Clear failure tracking for a tool on success.

    Clears ALL args variants for the tool, not just the successful call's args.
    This gives the tool a "fresh start" on any success, which is appropriate for
    the primary use case (detecting infinite loops with identical failing args).
    """
    tracker = _consecutive_tool_failures.get(None)
    if tracker is None:
        return
    # Clear all entries for this tool name
    keys_to_remove = [k for k in tracker if k.startswith(f"{tool_name}:")]
    for k in keys_to_remove:
        del tracker[k]


def create_tool_handler(base_tool: BaseTool):
    """Create an async handler function for a BaseTool.

    This wraps the existing BaseTool._execute method to be compatible
    with the Claude Agent SDK MCP tool format.
    """

    async def tool_handler(args: dict[str, Any]) -> dict[str, Any]:
        """Execute the wrapped tool and return MCP-formatted response."""
        user_id, session = get_execution_context()

        if session is None:
            return _mcp_error("No session context available")

        try:
            return await _execute_tool_sync(base_tool, user_id, session, args)
        except Exception as e:
            logger.error(
                "Error executing tool %s: %s", base_tool.name, e, exc_info=True
            )
            return _mcp_error(
                f"Failed to execute {base_tool.name}. Check server logs for details."
            )

    return tool_handler


def _build_input_schema(base_tool: BaseTool) -> dict[str, Any]:
    """Build a JSON Schema input schema for a tool.

    ``required`` is intentionally omitted from the schema sent to the MCP SDK.
    The SDK validates ``required`` fields BEFORE calling the Python handler \u2014
    when the LLM's output tokens are truncated the tool call arrives as ``{}``
    and the SDK rejects it with an opaque ``'X' is a required property`` error.
    By omitting ``required`` the empty-args case reaches our Python handler
    where ``_make_truncating_wrapper`` returns actionable chunking guidance.
    """
    return {
        "type": "object",
        "properties": base_tool.parameters.get("properties", {}),
    }


async def _read_file_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Read a file with optional offset/limit.

    Supports ``workspace://`` URIs (delegated to the workspace manager) and
    local paths within the session's allowed directories (sdk_cwd + tool-results).
    """

    def _mcp_err(text: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": text}], "isError": True}

    def _mcp_ok(text: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": text}], "isError": False}

    if not args:
        return _mcp_err(
            "Your Read call had empty arguments \u2014 this means your previous "
            "response was too long and the tool call was truncated by the API. "
            "Break your work into smaller steps."
        )

    file_path = args.get("file_path", "")
    try:
        offset = max(0, int(args.get("offset", 0)))
        limit = max(1, int(args.get("limit", 2000)))
    except (ValueError, TypeError):
        return _mcp_err("Invalid offset/limit \u2014 must be integers.")

    if not file_path:
        if "offset" in args or "limit" in args:
            return _mcp_err(
                "Your Read call was truncated (file_path missing but "
                "offset/limit were present). Resend with the full file_path."
            )
        return _mcp_err("file_path is required")

    if file_path.startswith("workspace://"):
        user_id, session = get_execution_context()
        if session is None:
            return _mcp_err("workspace:// file references require an active session")
        try:
            raw = await read_file_bytes(file_path, user_id, session)
        except ValueError as exc:
            return _mcp_err(str(exc))
        lines = raw.decode("utf-8", errors="replace").splitlines(keepends=True)
        selected = list(itertools.islice(lines, offset, offset + limit))
        numbered = "".join(
            f"{i + offset + 1:>6}\t{line}" for i, line in enumerate(selected)
        )
        return _mcp_ok(numbered)

    # Use is_sdk_tool_path (not is_allowed_local_path) to restrict this tool
    # to only SDK-internal tool-results/tool-outputs paths.  is_sdk_tool_path
    # validates session membership via _current_project_dir, preventing
    # cross-session reads.  sdk_cwd files (workspace outputs) are NOT allowed
    # here — they are served by the e2b_file_tools Read handler instead.
    if not is_sdk_tool_path(file_path):
        return _mcp_err(f"Path not allowed: {os.path.basename(file_path)}")

    resolved = os.path.realpath(os.path.expanduser(file_path))
    try:
        with open(resolved, encoding="utf-8", errors="replace") as f:
            selected = list(itertools.islice(f, offset, offset + limit))
        # Cleanup happens in _cleanup_sdk_tool_results after session ends;
        # don't delete here — the SDK may read in multiple chunks.
        #
        # When E2B is active, also copy the file into the sandbox so
        # bash_exec can process it (the model often uses Read then bash).
        text = "".join(selected)
        sandbox = _current_sandbox.get(None)
        if sandbox is not None:
            annotation = await bridge_and_annotate(sandbox, resolved, offset, limit)
            if annotation:
                text += annotation
        return _mcp_ok(text)
    except FileNotFoundError:
        return _mcp_err(f"File not found: {file_path}")
    except Exception as e:
        return _mcp_err(f"Error reading file: {e}")


_READ_TOOL_NAME = "read_tool_result"
_READ_TOOL_DESCRIPTION = (
    "Read an SDK-internal tool-result file or a workspace:// URI. "
    "Use this tool only for paths under ~/.claude/projects/.../tool-results/ "
    "or tool-outputs/, and for workspace:// URIs returned by other tools. "
    "For files in the working directory use read_file instead. "
    "Use offset and limit to read specific line ranges for large files."
)
_READ_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "The absolute path to the file to read",
        },
        "offset": {
            "type": "integer",
            "description": "Line number to start reading from (0-indexed). Default: 0",
        },
        "limit": {
            "type": "integer",
            "description": "Number of lines to read. Default: 2000",
        },
    },
}


# ---------------------------------------------------------------------------
# MCP result helpers
# ---------------------------------------------------------------------------


def _text_from_mcp_result(result: dict[str, Any]) -> str:
    """Extract concatenated text from an MCP response's content blocks."""
    content = result.get("content", [])
    if not isinstance(content, list):
        return ""
    return "".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") == "text"
    )


_PARALLEL_ANNOTATION = ToolAnnotations(readOnlyHint=True)
_MUTATING_ANNOTATION = ToolAnnotations(readOnlyHint=False)


def _strip_llm_fields(result: dict[str, Any]) -> dict[str, Any]:
    """Strip fields in *_STRIP_FROM_LLM* from every JSON text block in *result*.

    Called by *_truncating* AFTER the output has been stashed for the frontend
    SSE stream, so StreamToolOutputAvailable still receives the full payload
    (including ``is_dry_run``).  The returned dict is what the LLM sees.

    Non-JSON blocks, non-dict JSON values, and error results are returned unchanged.

    Note: only top-level keys are stripped. Nested occurrences of _STRIP_FROM_LLM
    fields (e.g. inside an ``outputs`` sub-dict) are not removed. Current tool
    responses only set these fields at the top level.
    """
    if result.get("isError"):
        return result
    content = result.get("content", [])
    new_content = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            raw = block.get("text", "")
            # Skip JSON parse/re-serialise round-trip when no stripped field
            # appears in the raw text — fast path for the common non-dry-run case.
            if not any(field in raw for field in _STRIP_FROM_LLM):
                new_content.append(block)
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                logger.debug("_strip_llm_fields: skipping non-JSON block: %s", exc)
                new_content.append(block)
                continue
            if isinstance(parsed, dict):
                for field in _STRIP_FROM_LLM:
                    parsed.pop(field, None)
                block = {**block, "text": json.dumps(parsed)}
        new_content.append(block)
    return {**result, "content": new_content}


def _make_truncating_wrapper(
    fn, tool_name: str, input_schema: dict[str, Any] | None = None
):
    """Return a wrapper around *fn* that truncates output, stashes it for the
    frontend SSE stream, and strips LLM-revealing fields before returning.

    Extracted from ``create_copilot_mcp_server`` so it can be tested directly.

    WARNING: ``stash_pending_tool_output`` must be called BEFORE
    ``_strip_llm_fields`` so the frontend SSE stream receives the full payload
    (including ``is_dry_run``) while the LLM sees a cleaned version.
    Swapping this order would cause the frontend to lose ``is_dry_run``.
    """

    async def wrapper(args: dict[str, Any]) -> dict[str, Any]:
        # Detect empty-args truncation: args is empty AND the schema declares
        # at least one property (so a non-empty call was expected).
        # NOTE: _build_input_schema intentionally omits "required" to avoid
        # SDK-side validation rejecting truncated calls before reaching this
        # handler.  We detect truncation via "properties" instead.
        schema_has_params = bool(input_schema and input_schema.get("properties"))
        if not args and schema_has_params:
            logger.warning(
                "[MCP] %s called with empty args (likely output "
                "token truncation) — returning guidance",
                tool_name,
            )
            return _mcp_error(
                f"Your call to {tool_name} had empty arguments — "
                f"this means your previous response was too long and "
                f"the tool call input was truncated by the API. "
                f"To fix this: break your work into smaller steps. "
                f"For large content, first write it to a file using "
                f"bash_exec with cat >> (append section by section), "
                f"then pass it via @@agptfile:filename reference. "
                f"Do NOT retry with the same approach — it will "
                f"be truncated again."
            )

        original_args = args
        stop_msg = _check_circuit_breaker(tool_name, original_args)
        if stop_msg:
            return _mcp_error(stop_msg)

        user_id, session = get_execution_context()
        if session is not None:
            try:
                args = await expand_file_refs_in_args(
                    args, user_id, session, input_schema=input_schema
                )
            except FileRefExpansionError as exc:
                _record_tool_failure(tool_name, original_args)
                return _mcp_error(
                    f"@@agptfile: reference could not be resolved: {exc}. "
                    "Ensure the file exists before referencing it. "
                    "For sandbox paths use bash_exec to verify the file exists first; "
                    "for workspace files use a workspace:// URI."
                )
        result = await fn(args)
        truncated = truncate(result, _MCP_MAX_CHARS)

        if truncated.get("isError"):
            _record_tool_failure(tool_name, original_args)
        else:
            _clear_tool_failures(tool_name)

        # Stash the raw tool output for the frontend SSE stream so widgets
        # (bash, tool viewers) receive clean JSON.  Mid-turn user follow-up
        # injection for MCP + built-in tools is now handled uniformly by
        # the ``PostToolUse`` hook via ``additionalContext`` so Claude sees
        # the follow-up attached to the tool_result without mutating the
        # frontend-facing payload.
        if not truncated.get("isError"):
            text = _text_from_mcp_result(truncated)
            if text:
                stash_pending_tool_output(tool_name, text)

        # Strip is_dry_run only when the session itself is in dry_run mode.
        # In that case the LLM must not know it is simulating — it should act
        # as if every tool call produced real results.
        # In normal (non-session-dry_run) mode, is_dry_run=True is intentionally
        # left visible to the LLM so it knows a specific tool was simulated and
        # can reason about the reliability of that output.
        if session is not None and session.dry_run:
            truncated = _strip_llm_fields(truncated)

        return truncated

    return wrapper


def create_copilot_mcp_server(*, use_e2b: bool = False):
    """Create an in-process MCP server configuration for CoPilot tools.

    All tools are annotated with ``readOnlyHint=True`` so the SDK CLI
    dispatches concurrent tool calls in parallel rather than sequentially.
    This is a deliberate override: even side-effect tools use the hint
    because the MCP tools are already individually sandboxed and the
    pre-launch duplicate-execution bug (SECRT-2204) is worse than
    sequential dispatch.

    When *use_e2b* is True, five additional MCP file tools are registered
    that route directly to the E2B sandbox filesystem, and the caller should
    disable the corresponding SDK built-in tools via
    :func:`get_sdk_disallowed_tools`.
    """

    sdk_tools = []

    for tool_name, base_tool in TOOL_REGISTRY.items():
        handler = create_tool_handler(base_tool)
        schema = _build_input_schema(base_tool)
        # All tools annotated readOnlyHint=True to enable parallel dispatch.
        # The SDK CLI uses this hint to dispatch concurrent tool calls in
        # parallel rather than sequentially.  Side-effect safety is ensured
        # by the tool implementations themselves (idempotency, credentials).
        decorated = tool(
            tool_name,
            base_tool.description,
            schema,
            annotations=_PARALLEL_ANNOTATION,
        )(_make_truncating_wrapper(handler, tool_name, input_schema=schema))
        sdk_tools.append(decorated)

    # E2B file tools replace SDK built-in Read/Write/Edit/Glob/Grep.
    _MUTATING_E2B_TOOLS = {"write_file", "edit_file"}
    if use_e2b:
        for name, desc, schema, handler in E2B_FILE_TOOLS:
            ann = (
                _MUTATING_ANNOTATION
                if name in _MUTATING_E2B_TOOLS
                else _PARALLEL_ANNOTATION
            )
            decorated = tool(
                name,
                desc,
                schema,
                annotations=ann,
            )(_make_truncating_wrapper(handler, name))
            sdk_tools.append(decorated)

    # Unified Write/Read/Edit tools — replace the CLI's built-in versions
    # which have no defence against output-token truncation.
    # Skip in E2B mode: E2B_FILE_TOOLS already registers "write_file",
    # "read_file", and "edit_file".  Registering both would give the LLM
    # duplicate tools per operation.
    if not use_e2b:
        write_handler = get_write_tool_handler()
        write_tool = tool(
            WRITE_TOOL_NAME,
            WRITE_TOOL_DESCRIPTION,
            WRITE_TOOL_SCHEMA,
            annotations=_MUTATING_ANNOTATION,
        )(
            _make_truncating_wrapper(
                write_handler, WRITE_TOOL_NAME, input_schema=WRITE_TOOL_SCHEMA
            )
        )
        sdk_tools.append(write_tool)

        read_file_handler = get_read_tool_handler()
        read_file_tool = tool(
            READ_TOOL_NAME,
            READ_TOOL_DESCRIPTION,
            READ_TOOL_SCHEMA,
            annotations=_PARALLEL_ANNOTATION,
        )(
            _make_truncating_wrapper(
                read_file_handler, READ_TOOL_NAME, input_schema=READ_TOOL_SCHEMA
            )
        )
        sdk_tools.append(read_file_tool)

        edit_handler = get_edit_tool_handler()
        edit_tool = tool(
            EDIT_TOOL_NAME,
            EDIT_TOOL_DESCRIPTION,
            EDIT_TOOL_SCHEMA,
            annotations=_MUTATING_ANNOTATION,
        )(
            _make_truncating_wrapper(
                edit_handler, EDIT_TOOL_NAME, input_schema=EDIT_TOOL_SCHEMA
            )
        )
        sdk_tools.append(edit_tool)

    # Read tool for SDK-truncated tool results (always needed, read-only).
    read_tool = tool(
        _READ_TOOL_NAME,
        _READ_TOOL_DESCRIPTION,
        _READ_TOOL_SCHEMA,
        annotations=_PARALLEL_ANNOTATION,
    )(_make_truncating_wrapper(_read_file_handler, _READ_TOOL_NAME))
    sdk_tools.append(read_tool)

    return create_sdk_mcp_server(
        name=MCP_SERVER_NAME,
        version="1.0.0",
        tools=sdk_tools,
    )


# SDK built-in tools allowed within the workspace directory.
# Security hooks validate that file paths stay within sdk_cwd.
# Bash is NOT included — use the sandboxed MCP bash_exec tool instead,
# which provides kernel-level network isolation via unshare --net.
# Task/Agent allows spawning sub-agents (rate-limited by security hooks).
#   The CLI renamed "Task" → "Agent" in v2.x; both are listed for compat.
# WebSearch uses Brave Search via Anthropic's API — safe, no SSRF risk.
# TodoWrite manages the task checklist shown in the UI — no security concern.
# In E2B mode, all five are disabled — MCP equivalents provide direct sandbox
# access.  read_file also handles local tool-results and ephemeral reads.
_SDK_BUILTIN_FILE_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep"]
_SDK_BUILTIN_ALWAYS = ["Task", "Agent", "WebSearch", "TodoWrite"]
_SDK_BUILTIN_TOOLS = [*_SDK_BUILTIN_FILE_TOOLS, *_SDK_BUILTIN_ALWAYS]

# SDK built-in tools that must be explicitly blocked.
# Bash: dangerous — agent uses mcp__copilot__bash_exec with kernel-level
#   network isolation (unshare --net) instead.
# WebFetch: SSRF risk — can reach internal network (localhost, 10.x, etc.).
#   Agent uses the SSRF-protected mcp__copilot__web_fetch tool instead.
# AskUserQuestion: interactive CLI tool — no terminal in copilot context.
# Write: the CLI's built-in Write tool has no defence against output-token
#   truncation.  When the LLM generates a very large `content` argument the
#   API truncates the response mid-JSON and Ajv rejects it with the opaque
#   "'file_path' is a required property" error, losing the user's work.
#   All writes go through our MCP Write tool (e2b_file_tools.py) where we
#   control validation and return actionable guidance.
# Edit: same truncation risk as Write — the CLI's built-in Edit has no
#   defence against output-token truncation.  All edits go through our
#   MCP Edit tool (e2b_file_tools.py).
# Read: already disallowed in E2B mode (prod/dev) via
#   _SDK_BUILTIN_FILE_TOOLS.  Disallow in non-E2B too for consistency
#   — our MCP read_file handles tool-results paths via
#   is_allowed_local_path() and has been the only Read available in
#   prod without issues.
SDK_DISALLOWED_TOOLS = [
    "Bash",
    "WebFetch",
    "AskUserQuestion",
    "Write",
    "Edit",
    "Read",
]

# Tools that are blocked entirely in security hooks (defence-in-depth).
# Includes SDK_DISALLOWED_TOOLS plus common aliases/synonyms.
BLOCKED_TOOLS = {
    *SDK_DISALLOWED_TOOLS,
    "bash",
    "shell",
    "exec",
    "terminal",
    "command",
}

# Tools allowed only when their path argument stays within the SDK workspace.
# The SDK uses these to handle oversized tool results (writes to tool-results/
# files, then reads them back) and for workspace file operations.
# Read is included because the SDK reads back oversized tool results from
# tool-results/ and tool-outputs/ directories.  It is also in
# SDK_DISALLOWED_TOOLS (which controls the SDK's disallowed_tools config),
# but the security hooks check workspace scope BEFORE the blocked list
# so that these internal reads are permitted.
# Write and Edit are NOT included: they are fully replaced by MCP equivalents.
WORKSPACE_SCOPED_TOOLS = {"Glob", "Grep", "Read"}

# Dangerous patterns in tool inputs
DANGEROUS_PATTERNS = [
    r"sudo",
    r"rm\s+-rf",
    r"dd\s+if=",
    r"/etc/passwd",
    r"/etc/shadow",
    r"chmod\s+777",
    r"curl\s+.*\|.*sh",
    r"wget\s+.*\|.*sh",
    r"eval\s*\(",
    r"exec\s*\(",
    r"__import__",
    r"os\.system",
    r"subprocess",
]

# Static tool name list for the non-E2B case (backward compatibility).
COPILOT_TOOL_NAMES = [
    *[f"{MCP_TOOL_PREFIX}{name}" for name in TOOL_REGISTRY.keys()],
    f"{MCP_TOOL_PREFIX}{WRITE_TOOL_NAME}",
    f"{MCP_TOOL_PREFIX}{READ_TOOL_NAME}",
    f"{MCP_TOOL_PREFIX}{EDIT_TOOL_NAME}",
    f"{MCP_TOOL_PREFIX}{_READ_TOOL_NAME}",
    *_SDK_BUILTIN_TOOLS,
]


def get_copilot_tool_names(*, use_e2b: bool = False) -> list[str]:
    """Build the ``allowed_tools`` list for :class:`ClaudeAgentOptions`.

    When *use_e2b* is True the SDK built-in file tools are replaced by MCP
    equivalents that route to the E2B sandbox.
    """
    if not use_e2b:
        return list(COPILOT_TOOL_NAMES)

    # In E2B mode, Write/Edit are NOT registered (E2B uses write_file/edit_file
    # from E2B_FILE_TOOLS instead), so don't include them here.
    # _READ_TOOL_NAME is still needed for SDK tool-result reads.
    return [
        *[f"{MCP_TOOL_PREFIX}{name}" for name in TOOL_REGISTRY.keys()],
        f"{MCP_TOOL_PREFIX}{_READ_TOOL_NAME}",
        *[f"{MCP_TOOL_PREFIX}{name}" for name in E2B_FILE_TOOL_NAMES],
        *_SDK_BUILTIN_ALWAYS,
    ]


def get_sdk_disallowed_tools(*, use_e2b: bool = False) -> list[str]:
    """Build the ``disallowed_tools`` list for :class:`ClaudeAgentOptions`.

    When *use_e2b* is True the SDK built-in file tools are also disabled
    because MCP equivalents provide direct sandbox access.
    """
    if not use_e2b:
        return list(SDK_DISALLOWED_TOOLS)
    return [*SDK_DISALLOWED_TOOLS, *_SDK_BUILTIN_FILE_TOOLS]
