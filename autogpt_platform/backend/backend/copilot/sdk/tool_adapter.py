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
from collections.abc import Iterable
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
from backend.copilot.tools import TOOL_REGISTRY, ToolGroup, tool_names_in_groups
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
# Keyed by a (tool_name + canonical input) composite — see ``_output_key`` —
# → FIFO of full output strings. Consumed (popped) by the response adapter
# when it builds StreamToolOutputAvailable.
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


def reset_pending_tool_outputs() -> None:
    """Drop stashed tool outputs left over from a previous stream attempt.

    The stash is a per-call FIFO (see ``_output_key``), not keyed by
    tool_call_id. A rolled-back attempt that executed tools but never consumed
    their results leaves orphaned entries, so on the retry every
    ``pop_pending_tool_output`` for that key returns the stale first-attempt
    output — shifting all
    subsequent pops off-by-one and attaching wrong payloads to the frontend's
    ``StreamToolOutputAvailable`` events (e.g. a ``setup_requirements`` card
    silently replaced by an older result). Called at the top of each retry
    attempt, where no tool call can be in flight.
    """
    _pending_tool_outputs.set({})


def _output_key(tool_name: str, tool_input: Any = None) -> str:
    """Build the stash key correlating a tool call to its output.

    Tool *name* alone is insufficient (OPEN-3158): the model can issue two
    parallel calls to the same tool in one turn (e.g. two ``web_search``
    queries).  Their outputs are stashed in completion order but consumed in
    tool-result order — with a name-only key those orders diverge and the
    outputs attach to the wrong ``tool_call_id``, swapping the two cards in
    the UI.  Including a canonical serialization of the call's input
    disambiguates the common case where the two calls differ.

    The in-process MCP handler never sees the SDK ``tool_use_id`` (only the
    arguments), so the input is the most specific key available on the stash
    side.  Empty/falsy input falls back to the name-only key: such calls
    can't be disambiguated, and two identical calls produce interchangeable
    outputs so a swap between them is not user-visible.
    """
    if not tool_input:
        return tool_name
    try:
        canonical = json.dumps(tool_input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        canonical = str(tool_input)
    return f"{tool_name}\x00{canonical}"


def pop_pending_tool_output(tool_name: str, tool_input: Any = None) -> str | None:
    """Pop and return the oldest stashed output for a tool call.

    The SDK CLI may truncate large tool results (writing them to disk and
    replacing the content with a file reference). This stash keeps the
    original MCP output so the response adapter can forward it to the
    frontend for proper widget rendering.

    Matched by ``_output_key(tool_name, tool_input)`` so parallel calls to
    the same tool with different inputs each get their own output instead of
    being swapped.  Falls back to a FIFO queue when several calls share the
    same key.

    Returns ``None`` if nothing was stashed for this call.
    """
    pending = _pending_tool_outputs.get(None)
    if pending is None:
        return None
    key = _output_key(tool_name, tool_input)
    queue = pending.get(key)
    if not queue:
        pending.pop(key, None)
        return None
    value = queue.pop(0)
    if not queue:
        del pending[key]
    return value


def stash_pending_tool_output(
    tool_name: str, output: Any, tool_input: Any = None
) -> None:
    """Stash tool output for later retrieval by the response adapter.

    Used by the MCP truncating wrapper and the PostToolUse hook (for SDK
    built-in tools like WebSearch/Read that aren't available through the MCP
    stash mechanism in ``_execute_tool_sync``).

    Keyed by ``_output_key(tool_name, tool_input)`` so the response adapter
    can pop the output belonging to a *specific* tool call rather than the
    next one for that tool name — see ``_output_key`` for why this matters.
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
    pending.setdefault(_output_key(tool_name, tool_input), []).append(text)
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
    char_offset_arg = args.get("char_offset")
    char_limit_arg = args.get("char_limit")
    try:
        offset = max(0, int(args.get("offset", 0)))
        limit = max(1, int(args.get("limit", 2000)))
        char_offset = (
            max(0, int(char_offset_arg)) if char_offset_arg is not None else None
        )
        char_limit = max(1, int(char_limit_arg)) if char_limit_arg is not None else None
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
        # Read the whole file: tool-result envelopes are usually small JSON
        # but the payload inside is often one massive minified line, so
        # line-based offset/limit applied to the raw bytes is useless.
        # _navigable_tool_result_text() unwraps the envelope and pretty-
        # prints inner JSON so line offsets actually slice the payload.
        with open(resolved, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        navigable = _navigable_tool_result_text(raw)
        if char_offset is not None or char_limit is not None:
            # Character-mode slicing: precise control for huge payloads
            # where even the pretty-printed inner JSON has multi-KB lines
            # (e.g. base64 blobs in tool results).
            start = char_offset or 0
            end = start + char_limit if char_limit is not None else len(navigable)
            text = navigable[start:end]
        else:
            lines = navigable.splitlines(keepends=True)
            text = "".join(lines[offset : offset + limit])
        # Cleanup happens in _cleanup_sdk_tool_results after session ends;
        # don't delete here — the SDK may read in multiple chunks.
        #
        # When E2B is active, also copy the file into the sandbox so
        # bash_exec can process it (the model often uses Read then bash).
        # CAVEAT: only bridge when the on-disk bytes and what the model
        # just read are the same — i.e. when ``_navigable_tool_result_text``
        # did *not* transform the content. If we pretty-printed an MCP
        # envelope, the model sees a pretty-printed slice while the
        # bridged sandbox file would hold the raw envelope; bash commands
        # operating on the bridged copy would then see different content
        # than the model just read, leading to silent format-mismatch
        # bugs. Same constraint applies to char-mode slices. The new
        # bash_exec SDK-path redirect (added in this PR) covers the
        # alternative workflow when the bridge is skipped — the model
        # can use ``read_tool_result`` again with offsets, or pipe a
        # slice via ``@@agptfile:<path>[<start>-<end>]``.
        sandbox = _current_sandbox.get(None)
        if (
            sandbox is not None
            and navigable == raw
            and char_offset is None
            and char_limit is None
        ):
            annotation = await bridge_and_annotate(sandbox, resolved, offset, limit)
            if annotation:
                text += annotation
        return _mcp_ok(text)
    except FileNotFoundError:
        return _mcp_err(f"File not found: {file_path}")
    except Exception as e:
        return _mcp_err(f"Error reading file: {e}")


def _navigable_tool_result_text(raw: str) -> str:
    """Return *raw* in a form the model can actually slice with offset/limit.

    Tool-result files are stored as the MCP envelope ``[{"type":"text",
    "text": "<payload>"}]``. The outer list is pretty-printed but the
    inner ``text`` field is one giant minified string, so line-based
    offset/limit on the raw file content slices the *envelope* — useless.

    This helper unwraps the envelope (when the shape matches) and
    pretty-prints the inner payload (when it parses as JSON) so the model
    can navigate ``execution.node_executions[…].error`` etc. with normal
    line offsets. Falls back to the raw text on any mismatch.
    """
    try:
        outer = json.loads(raw)
    except (ValueError, TypeError):
        return raw
    inner = _extract_single_text_block(outer)
    if inner is None:
        return raw
    try:
        return json.dumps(json.loads(inner), indent=2, ensure_ascii=False)
    except (ValueError, TypeError):
        # Inner text wasn't JSON (e.g. a bash command's stdout) — return
        # it raw so the model sees the payload without the envelope noise.
        return inner


def _extract_single_text_block(envelope: object) -> str | None:
    """Return the text of a single-text-block MCP envelope, else None.

    Matches ``[{"type": "text", "text": <str>}]`` exactly. Returns None
    for envelopes with images, multiple blocks, or any other shape.
    """
    if not isinstance(envelope, list) or len(envelope) != 1:
        return None
    block = envelope[0]
    if not isinstance(block, dict) or block.get("type") != "text":
        return None
    text = block.get("text")
    return text if isinstance(text, str) else None


_READ_TOOL_NAME = "read_tool_result"
_READ_TOOL_DESCRIPTION = (
    "Read an SDK-internal tool-result file or a workspace:// URI. "
    "Use this tool only for paths under ~/.claude/projects/.../tool-results/ "
    "or tool-outputs/, and for workspace:// URIs returned by other tools. "
    "For files in the working directory use read_file instead. "
    "MCP envelopes are auto-unwrapped and JSON payloads pretty-printed, "
    "so offset/limit (line-based) slice into the actual payload, not the "
    "envelope wrapper. For piping a slice into another tool's command, "
    "use `@@agptfile:<absolute-path>[<start>-<end>]` in that tool's "
    "argument instead — it works in bash_exec and avoids a round-trip."
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
        "char_offset": {
            "type": "integer",
            "description": (
                "Character offset to start reading from (0-indexed). "
                "Overrides `offset`. Use when even pretty-printed lines "
                "are too long to slice with line offsets (e.g. base64 "
                "blobs in a tool result)."
            ),
        },
        "char_limit": {
            "type": "integer",
            "description": (
                "Number of characters to read. Pairs with `char_offset` "
                "and overrides `limit`."
            ),
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
    fn,
    tool_name: str,
    input_schema: dict[str, Any] | None = None,
    required_args: list[str] | None = None,
):
    """Return a wrapper around *fn* that truncates output, stashes it for the
    frontend SSE stream, and strips LLM-revealing fields before returning.

    Extracted from ``create_copilot_mcp_server`` so it can be tested directly.

    ``required_args`` is the tool's original list of required parameter names
    (before ``_build_input_schema`` stripped ``required`` from the SDK-visible
    schema). It's used to distinguish a truncated empty call from a legitimate
    no-arg invocation on a tool whose params are all optional (e.g.
    ``list_schedules``). When the list is empty/None the empty-args truncation
    guard is skipped — the model is free to call the tool with no arguments.

    WARNING: ``stash_pending_tool_output`` must be called BEFORE
    ``_strip_llm_fields`` so the frontend SSE stream receives the full payload
    (including ``is_dry_run``) while the LLM sees a cleaned version.
    Swapping this order would cause the frontend to lose ``is_dry_run``.
    """

    async def wrapper(args: dict[str, Any]) -> dict[str, Any]:
        # Detect empty-args truncation: args is empty AND the original tool
        # declared at least one *required* property. Tools whose params are all
        # optional (filters-only tools like list_schedules) legitimately accept
        # no args, so we skip the guard for them.
        # NOTE: _build_input_schema intentionally omits "required" from the
        # SDK-visible schema to avoid SDK-side validation rejecting truncated
        # calls before reaching this handler. We carry required_args through
        # the wrapper instead.
        if not args and required_args:
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
                # Key by the model's ORIGINAL args (pre file-ref expansion) so
                # it matches the ToolUseBlock.input the response adapter pops
                # with — see ``_output_key`` (OPEN-3158).
                stash_pending_tool_output(tool_name, text, original_args)

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


def create_copilot_mcp_server(
    *,
    use_e2b: bool = False,
    hidden_tool_names: Iterable[str] = (),
):
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

    Short tool names in *hidden_tool_names* are not registered at all — the
    model never sees them.  ``allowed_tools``/``disallowed_tools`` alone are
    insufficient because the CLI auto-rejects calls to denied tools with a
    canned "Permission to use ... has been denied" string that the model
    then narrates as a Claude-Code-style approval prompt (no such UI exists
    in copilot).  Hiding the tool removes the temptation entirely.
    """

    hidden = frozenset(hidden_tool_names)
    sdk_tools = []

    for tool_name, base_tool in TOOL_REGISTRY.items():
        if tool_name in hidden:
            continue
        handler = create_tool_handler(base_tool)
        schema = _build_input_schema(base_tool)
        required = list(base_tool.parameters.get("required", []))
        # All tools annotated readOnlyHint=True to enable parallel dispatch.
        # The SDK CLI uses this hint to dispatch concurrent tool calls in
        # parallel rather than sequentially.  Side-effect safety is ensured
        # by the tool implementations themselves (idempotency, credentials).
        decorated = tool(
            tool_name,
            base_tool.description,
            schema,
            annotations=_PARALLEL_ANNOTATION,
        )(
            _make_truncating_wrapper(
                handler, tool_name, input_schema=schema, required_args=required
            )
        )
        sdk_tools.append(decorated)

    # E2B file tools replace SDK built-in Read/Write/Edit/Glob/Grep.
    # All E2B file tools have at least one required arg (the path), so empty
    # args is always truncation — passing a sentinel non-empty required_args
    # list keeps the truncation guard active for them.
    _MUTATING_E2B_TOOLS = {"write_file", "edit_file"}
    if use_e2b:
        for name, desc, schema, handler in E2B_FILE_TOOLS:
            if name in hidden:
                continue
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
            )(_make_truncating_wrapper(handler, name, required_args=["path"]))
            sdk_tools.append(decorated)

    # Unified Write/Read/Edit tools — replace the CLI's built-in versions
    # which have no defence against output-token truncation.
    # Skip in E2B mode: E2B_FILE_TOOLS already registers "write_file",
    # "read_file", and "edit_file".  Registering both would give the LLM
    # duplicate tools per operation.
    if not use_e2b:
        if WRITE_TOOL_NAME not in hidden:
            write_handler = get_write_tool_handler()
            write_tool = tool(
                WRITE_TOOL_NAME,
                WRITE_TOOL_DESCRIPTION,
                WRITE_TOOL_SCHEMA,
                annotations=_MUTATING_ANNOTATION,
            )(
                _make_truncating_wrapper(
                    write_handler,
                    WRITE_TOOL_NAME,
                    input_schema=WRITE_TOOL_SCHEMA,
                    required_args=["file_path", "content"],
                )
            )
            sdk_tools.append(write_tool)

        if READ_TOOL_NAME not in hidden:
            read_file_handler = get_read_tool_handler()
            read_file_tool = tool(
                READ_TOOL_NAME,
                READ_TOOL_DESCRIPTION,
                READ_TOOL_SCHEMA,
                annotations=_PARALLEL_ANNOTATION,
            )(
                _make_truncating_wrapper(
                    read_file_handler,
                    READ_TOOL_NAME,
                    input_schema=READ_TOOL_SCHEMA,
                    required_args=["file_path"],
                )
            )
            sdk_tools.append(read_file_tool)

        if EDIT_TOOL_NAME not in hidden:
            edit_handler = get_edit_tool_handler()
            edit_tool = tool(
                EDIT_TOOL_NAME,
                EDIT_TOOL_DESCRIPTION,
                EDIT_TOOL_SCHEMA,
                annotations=_MUTATING_ANNOTATION,
            )(
                _make_truncating_wrapper(
                    edit_handler,
                    EDIT_TOOL_NAME,
                    input_schema=EDIT_TOOL_SCHEMA,
                    required_args=["file_path", "old_string", "new_string"],
                )
            )
            sdk_tools.append(edit_tool)

    # Read tool for SDK-truncated tool results (always needed, read-only).
    if _READ_TOOL_NAME not in hidden:
        read_tool = tool(
            _READ_TOOL_NAME,
            _READ_TOOL_DESCRIPTION,
            _READ_TOOL_SCHEMA,
            annotations=_PARALLEL_ANNOTATION,
        )(
            _make_truncating_wrapper(
                _read_file_handler, _READ_TOOL_NAME, required_args=["file_path"]
            )
        )
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
# WebSearch moved to ``SDK_DISALLOWED_TOOLS`` — routed through
# ``mcp__copilot__web_search`` so cost tracking is unified across paths.
_SDK_BUILTIN_ALWAYS = ["Task", "Agent", "TodoWrite"]
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
# ScheduleWakeup: no /loop runtime in copilot turns; the handler returns
#   {"scheduledFor": 0} and nothing is scheduled.
SDK_DISALLOWED_TOOLS = [
    "Bash",
    "WebFetch",
    "WebSearch",
    "AskUserQuestion",
    "Write",
    "Edit",
    "Read",
    "ScheduleWakeup",
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

# Platform-tool names whose MCP wrappers must NOT be exposed to SDK mode.
# Baseline ships an MCP ``TodoWrite`` for model-flexibility parity; SDK mode
# keeps using the CLI-native built-in listed in ``_SDK_BUILTIN_ALWAYS`` so
# there is no double exposure.  Public (no leading underscore) so a future
# refactor renaming it is visible at both call sites —
# ``permissions.apply_tool_permissions`` maps short tool names back to the
# CLI built-in form for SDK mode.
BASELINE_ONLY_MCP_TOOLS: frozenset[str] = frozenset({"TodoWrite"})


def _registry_mcp_tools(*, hidden: frozenset[str] = frozenset()) -> list[str]:
    return [
        f"{MCP_TOOL_PREFIX}{name}"
        for name in TOOL_REGISTRY.keys()
        if name not in BASELINE_ONLY_MCP_TOOLS and name not in hidden
    ]


# Static tool name list for the non-E2B case (backward compatibility).
# Includes all capability-gated tools; per-user filtering happens in
# ``get_copilot_tool_names`` when the caller passes ``disabled_groups``.
COPILOT_TOOL_NAMES = [
    *_registry_mcp_tools(),
    f"{MCP_TOOL_PREFIX}{WRITE_TOOL_NAME}",
    f"{MCP_TOOL_PREFIX}{READ_TOOL_NAME}",
    f"{MCP_TOOL_PREFIX}{EDIT_TOOL_NAME}",
    f"{MCP_TOOL_PREFIX}{_READ_TOOL_NAME}",
    *_SDK_BUILTIN_TOOLS,
]


def get_copilot_tool_names(
    *,
    use_e2b: bool = False,
    disabled_groups: Iterable[ToolGroup] = (),
) -> list[str]:
    """Build the ``allowed_tools`` list for :class:`ClaudeAgentOptions`.

    When *use_e2b* is True the SDK built-in file tools are replaced by MCP
    equivalents that route to the E2B sandbox.  Tools belonging to any of
    *disabled_groups* are filtered out — see ``ToolGroup`` / ``TOOL_GROUPS``
    in ``backend.copilot.tools`` for the full list.
    """
    hidden_short_names = tool_names_in_groups(disabled_groups)
    hidden_mcp_names = {f"{MCP_TOOL_PREFIX}{n}" for n in hidden_short_names}

    if not use_e2b:
        if not hidden_mcp_names:
            return list(COPILOT_TOOL_NAMES)
        return [n for n in COPILOT_TOOL_NAMES if n not in hidden_mcp_names]

    # In E2B mode, Write/Edit are NOT registered (E2B uses write_file/edit_file
    # from E2B_FILE_TOOLS instead), so don't include them here.
    # _READ_TOOL_NAME is still needed for SDK tool-result reads.
    return [
        *_registry_mcp_tools(hidden=hidden_short_names),
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
