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

from backend.copilot.context import (
    _current_project_dir,
    _current_sandbox,
    _current_sdk_cwd,
    _current_session,
    _current_user_id,
    _encode_cwd_for_cli,
    get_execution_context,
    get_sdk_cwd,
    is_allowed_local_path,
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

from .e2b_file_tools import E2B_FILE_TOOL_NAMES, E2B_FILE_TOOLS

if TYPE_CHECKING:
    from e2b import AsyncSandbox

logger = logging.getLogger(__name__)

# Max MCP response size in chars — keeps tool output under the SDK's 10 MB JSON buffer.
_MCP_MAX_CHARS = 500_000

# MCP server naming - the SDK prefixes tool names as "mcp__{server_name}__{tool}"
MCP_SERVER_NAME = "copilot"
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

# Map from tool_name -> Queue of pre-launched (task, args) pairs.
# Initialised per-session in set_execution_context() so concurrent sessions
# never share the same dict.
_TaskQueueItem = tuple[asyncio.Task[dict[str, Any]], dict[str, Any]]
_tool_task_queues: ContextVar[dict[str, asyncio.Queue[_TaskQueueItem]] | None] = (
    ContextVar("_tool_task_queues", default=None)
)

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


def set_execution_context(
    user_id: str | None,
    session: ChatSession,
    sandbox: "AsyncSandbox | None" = None,
    sdk_cwd: str | None = None,
) -> None:
    """Set the execution context for tool calls.

    This must be called before streaming begins to ensure tools have access
    to user_id, session, and (optionally) an E2B sandbox for bash execution.

    Args:
        user_id: Current user's ID.
        session: Current chat session.
        sandbox: Optional E2B sandbox; when set, bash_exec routes commands there.
        sdk_cwd: SDK working directory; used to scope tool-results reads.
    """
    _current_user_id.set(user_id)
    _current_session.set(session)
    _current_sandbox.set(sandbox)
    _current_sdk_cwd.set(sdk_cwd or "")
    _current_project_dir.set(_encode_cwd_for_cli(sdk_cwd) if sdk_cwd else "")
    _pending_tool_outputs.set({})
    _stash_event.set(asyncio.Event())
    _tool_task_queues.set({})


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


def cancel_pending_tool_tasks() -> None:
    """Cancel all queued pre-launched tasks for the current execution context.

    Call this when a stream attempt aborts (error, cancellation) to prevent
    pre-launched tasks from continuing to execute against a rolled-back session.
    Tasks that are already done are skipped; in-flight tasks are cancelled.
    """
    queues = _tool_task_queues.get()
    if not queues:
        return
    for tool_name, queue in list(queues.items()):
        cancelled = 0
        while not queue.empty():
            task, _args = queue.get_nowait()
            if not task.done():
                task.cancel()
                cancelled += 1
        if cancelled:
            logger.debug(
                "Cancelled %d pre-launched task(s) for tool '%s'", cancelled, tool_name
            )
    queues.clear()


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

    The 2.0 s default was chosen to accommodate slower tool startup in cloud
    sandboxes while still failing fast when the hook genuinely will not fire.
    With the parallel pre-launch path, hooks typically fire well under 1 ms.
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


async def pre_launch_tool_call(tool_name: str, args: dict[str, Any]) -> None:
    """Pre-launch a tool as a background task so parallel calls run concurrently.

    Called when an AssistantMessage with ToolUseBlocks is received, before the
    SDK dispatches the MCP tool/call requests. The tool_handler will await the
    pre-launched task instead of executing fresh.

    The tool_name may include an MCP prefix (e.g. ``mcp__copilot__run_block``);
    the prefix is stripped automatically before looking up the tool.

    Ordering guarantee: the Claude Agent SDK dispatches MCP ``tools/call`` requests
    in the same order as the ToolUseBlocks appear in the AssistantMessage.
    Pre-launched tasks are queued FIFO per tool name, so the N-th handler for a
    given tool name dequeues the N-th pre-launched task — result and args always
    correspond when the SDK preserves order (which it does in the current SDK).
    """
    queues = _tool_task_queues.get()
    if queues is None:
        return

    # Strip the MCP server prefix (e.g. "mcp__copilot__") to get the bare tool name.
    # Use removeprefix so tool names that themselves contain "__" are handled correctly.
    bare_name = tool_name.removeprefix(MCP_TOOL_PREFIX)

    base_tool = TOOL_REGISTRY.get(bare_name)
    if base_tool is None:
        return

    user_id, session = get_execution_context()
    if session is None:
        return

    # Expand @@agptfile: references before launching the task.
    # The _truncating wrapper (which normally handles expansion) runs AFTER
    # pre_launch_tool_call — the pre-launched task would otherwise receive raw
    # @@agptfile: tokens and fail to resolve them inside _execute_tool_sync.
    # Use _build_input_schema (same path as _truncating) for schema-aware expansion.
    input_schema: dict[str, Any] | None
    try:
        input_schema = _build_input_schema(base_tool)
    except Exception:
        input_schema = None  # schema unavailable — skip schema-aware expansion
    try:
        args = await expand_file_refs_in_args(
            args, user_id, session, input_schema=input_schema
        )
    except FileRefExpansionError as exc:
        logger.warning(
            "pre_launch_tool_call: @@agptfile expansion failed for %s: %s — skipping pre-launch",
            bare_name,
            exc,
        )
        return

    task = asyncio.create_task(_execute_tool_sync(base_tool, user_id, session, args))
    # Log unhandled exceptions so "Task exception was never retrieved" warnings
    # do not pollute stderr when a task is pre-launched but never dequeued.
    task.add_done_callback(
        lambda t, name=bare_name: (
            logger.warning(
                "Pre-launched task for %s raised unhandled: %s",
                name,
                t.exception(),
            )
            if not t.cancelled() and t.exception()
            else None
        )
    )

    if bare_name not in queues:
        queues[bare_name] = asyncio.Queue[_TaskQueueItem]()
    # Store (task, args) so the handler can log a warning if the SDK dispatches
    # calls in a different order than the ToolUseBlocks appeared in the message.
    queues[bare_name].put_nowait((task, args))


async def _execute_tool_sync(
    base_tool: BaseTool,
    user_id: str | None,
    session: ChatSession,
    args: dict[str, Any],
) -> dict[str, Any]:
    """Execute a tool synchronously and return MCP-formatted response.

    Note: ``@@agptfile:`` expansion should be performed by the caller before
    invoking this function.  For the normal (non-parallel) path it is handled
    by the ``_truncating`` wrapper; for the pre-launched parallel path it is
    handled in :func:`pre_launch_tool_call` before the task is created.
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


def create_tool_handler(base_tool: BaseTool):
    """Create an async handler function for a BaseTool.

    This wraps the existing BaseTool._execute method to be compatible
    with the Claude Agent SDK MCP tool format.
    """

    async def tool_handler(args: dict[str, Any]) -> dict[str, Any]:
        """Execute the wrapped tool and return MCP-formatted response.

        If a pre-launched task exists (from parallel tool pre-launch in the
        message loop), await it instead of executing fresh.
        """
        queues = _tool_task_queues.get()
        if queues and base_tool.name in queues:
            queue = queues[base_tool.name]
            if not queue.empty():
                task, launch_args = queue.get_nowait()
                # Sanity-check: warn if the args don't match — this can happen
                # if the SDK dispatches tool calls in a different order than the
                # ToolUseBlocks appeared in the AssistantMessage (unlikely but
                # could occur in future SDK versions or with SDK bugs).
                # We compare full values (not just keys) so that two run_block
                # calls with different block_id values are caught even though
                # both have the same key set.
                if launch_args != args:
                    logger.warning(
                        "Pre-launched task for %s: arg mismatch "
                        "(launch=%s, call=%s) — may return incorrect result",
                        base_tool.name,
                        launch_args,
                        args,
                    )
                try:
                    result = await task
                except asyncio.CancelledError:
                    # Re-raise: CancelledError may be propagating from the outer
                    # streaming loop being cancelled — swallowing it would mask
                    # the cancellation and prevent proper cleanup.
                    logger.warning(
                        "Pre-launched tool %s was cancelled — re-raising",
                        base_tool.name,
                    )
                    raise
                except Exception as e:
                    logger.error(
                        "Pre-launched tool %s failed: %s",
                        base_tool.name,
                        e,
                        exc_info=True,
                    )
                    return _mcp_error(f"Failed to execute {base_tool.name}: {e}")

                # Pre-truncate the result so the _truncating wrapper (which
                # wraps this handler) receives an already-within-budget value.
                # _truncating handles stashing — we must NOT stash here or the
                # output will be appended twice to the FIFO queue and
                # pop_pending_tool_output would return a duplicate entry on the
                # second call for the same tool.
                return truncate(result, _MCP_MAX_CHARS)

        # No pre-launched task — execute directly (fallback for non-parallel calls).
        user_id, session = get_execution_context()

        if session is None:
            return _mcp_error("No session context available")

        try:
            return await _execute_tool_sync(base_tool, user_id, session, args)
        except Exception as e:
            logger.error(
                "Error executing tool %s: %s", base_tool.name, e, exc_info=True
            )
            return _mcp_error(f"Failed to execute {base_tool.name}: {e}")

    return tool_handler


def _build_input_schema(base_tool: BaseTool) -> dict[str, Any]:
    """Build a JSON Schema input schema for a tool."""
    return {
        "type": "object",
        "properties": base_tool.parameters.get("properties", {}),
        "required": base_tool.parameters.get("required", []),
    }


async def _read_file_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Read a file with optional offset/limit.

    Supports ``workspace://`` URIs (delegated to the workspace manager) and
    local paths within the session's allowed directories (sdk_cwd + tool-results).
    """
    file_path = args.get("file_path", "")
    offset = max(0, int(args.get("offset", 0)))
    limit = max(1, int(args.get("limit", 2000)))

    def _mcp_err(text: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": text}], "isError": True}

    def _mcp_ok(text: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": text}], "isError": False}

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

    if not is_allowed_local_path(file_path, get_sdk_cwd()):
        return _mcp_err(f"Path not allowed: {file_path}")

    resolved = os.path.realpath(os.path.expanduser(file_path))
    try:
        with open(resolved, encoding="utf-8", errors="replace") as f:
            selected = list(itertools.islice(f, offset, offset + limit))
        # Cleanup happens in _cleanup_sdk_tool_results after session ends;
        # don't delete here — the SDK may read in multiple chunks.
        return _mcp_ok("".join(selected))
    except FileNotFoundError:
        return _mcp_err(f"File not found: {file_path}")
    except Exception as e:
        return _mcp_err(f"Error reading file: {e}")


_READ_TOOL_NAME = "Read"
_READ_TOOL_DESCRIPTION = (
    "Read a file from the local filesystem. "
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
    "required": ["file_path"],
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


def create_copilot_mcp_server(*, use_e2b: bool = False):
    """Create an in-process MCP server configuration for CoPilot tools.

    When *use_e2b* is True, five additional MCP file tools are registered
    that route directly to the E2B sandbox filesystem, and the caller should
    disable the corresponding SDK built-in tools via
    :func:`get_sdk_disallowed_tools`.
    """

    def _truncating(fn, tool_name: str, input_schema: dict[str, Any] | None = None):
        """Wrap a tool handler so its response is truncated to stay under the
        SDK's 10 MB JSON buffer, and stash the (truncated) output for the
        response adapter before the SDK can apply its own head-truncation.

        Also expands ``@@agptfile:`` references in args so every registered tool
        (BaseTool, E2B file tools, Read) receives resolved content uniformly.

        Applied once to every registered tool."""

        async def wrapper(args: dict[str, Any]) -> dict[str, Any]:
            user_id, session = get_execution_context()
            if session is not None:
                try:
                    args = await expand_file_refs_in_args(
                        args, user_id, session, input_schema=input_schema
                    )
                except FileRefExpansionError as exc:
                    return _mcp_error(
                        f"@@agptfile: reference could not be resolved: {exc}. "
                        "Ensure the file exists before referencing it. "
                        "For sandbox paths use bash_exec to verify the file exists first; "
                        "for workspace files use a workspace:// URI."
                    )
            result = await fn(args)
            truncated = truncate(result, _MCP_MAX_CHARS)

            # Stash the text so the response adapter can forward our
            # middle-out truncated version to the frontend instead of the
            # SDK's head-truncated version (for outputs >~100 KB the SDK
            # persists to tool-results/ with a 2 KB head-only preview).
            if not truncated.get("isError"):
                text = _text_from_mcp_result(truncated)
                if text:
                    stash_pending_tool_output(tool_name, text)

            return truncated

        return wrapper

    sdk_tools = []

    for tool_name, base_tool in TOOL_REGISTRY.items():
        handler = create_tool_handler(base_tool)
        schema = _build_input_schema(base_tool)
        decorated = tool(
            tool_name,
            base_tool.description,
            schema,
        )(_truncating(handler, tool_name, input_schema=schema))
        sdk_tools.append(decorated)

    # E2B file tools replace SDK built-in Read/Write/Edit/Glob/Grep.
    if use_e2b:
        for name, desc, schema, handler in E2B_FILE_TOOLS:
            decorated = tool(name, desc, schema)(_truncating(handler, name))
            sdk_tools.append(decorated)

    # Read tool for SDK-truncated tool results (always needed).
    read_tool = tool(
        _READ_TOOL_NAME,
        _READ_TOOL_DESCRIPTION,
        _READ_TOOL_SCHEMA,
    )(_truncating(_read_file_handler, _READ_TOOL_NAME))
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
# Task allows spawning sub-agents (rate-limited by security hooks).
# WebSearch uses Brave Search via Anthropic's API — safe, no SSRF risk.
# TodoWrite manages the task checklist shown in the UI — no security concern.
# In E2B mode, all five are disabled — MCP equivalents provide direct sandbox
# access.  read_file also handles local tool-results and ephemeral reads.
_SDK_BUILTIN_FILE_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep"]
_SDK_BUILTIN_ALWAYS = ["Task", "WebSearch", "TodoWrite"]
_SDK_BUILTIN_TOOLS = [*_SDK_BUILTIN_FILE_TOOLS, *_SDK_BUILTIN_ALWAYS]

# SDK built-in tools that must be explicitly blocked.
# Bash: dangerous — agent uses mcp__copilot__bash_exec with kernel-level
#   network isolation (unshare --net) instead.
# WebFetch: SSRF risk — can reach internal network (localhost, 10.x, etc.).
#   Agent uses the SSRF-protected mcp__copilot__web_fetch tool instead.
# AskUserQuestion: interactive CLI tool — no terminal in copilot context.
SDK_DISALLOWED_TOOLS = [
    "Bash",
    "WebFetch",
    "AskUserQuestion",
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
WORKSPACE_SCOPED_TOOLS = {"Read", "Write", "Edit", "Glob", "Grep"}

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
