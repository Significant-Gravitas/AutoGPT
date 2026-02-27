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

from backend.copilot.model import ChatSession
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools.base import BaseTool

if TYPE_CHECKING:
    from e2b import AsyncSandbox

logger = logging.getLogger(__name__)

# Allowed base directory for the Read tool (SDK saves oversized tool results here).
# Restricted to ~/.claude/projects/ and further validated to require "tool-results"
# in the path — prevents reading settings, credentials, or other sensitive files.
_SDK_PROJECTS_DIR = os.path.expanduser("~/.claude/projects/")

# MCP server naming - the SDK prefixes tool names as "mcp__{server_name}__{tool}"
MCP_SERVER_NAME = "copilot"
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

# Context variables to pass user/session info to tool execution
_current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)
_current_session: ContextVar[ChatSession | None] = ContextVar(
    "current_session", default=None
)
# E2B cloud sandbox for the current turn (None when E2B is not configured).
# Passed to bash_exec so commands run on E2B instead of the local bwrap sandbox.
_current_sandbox: ContextVar["AsyncSandbox | None"] = ContextVar(
    "_current_sandbox", default=None
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
) -> None:
    """Set the execution context for tool calls.

    This must be called before streaming begins to ensure tools have access
    to user_id, session, and (optionally) an E2B sandbox for bash execution.

    Args:
        user_id: Current user's ID.
        session: Current chat session.
        sandbox: Optional E2B sandbox; when set, bash_exec routes commands there.
    """
    _current_user_id.set(user_id)
    _current_session.set(session)
    _current_sandbox.set(sandbox)
    _pending_tool_outputs.set({})
    _stash_event.set(asyncio.Event())


def get_current_sandbox() -> "AsyncSandbox | None":
    """Return the E2B sandbox for the current turn, or None."""
    return _current_sandbox.get()


def get_execution_context() -> tuple[str | None, ChatSession | None]:
    """Get the current execution context."""
    return (
        _current_user_id.get(),
        _current_session.get(),
    )


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


async def wait_for_stash(timeout: float = 0.5) -> bool:
    """Wait for a PostToolUse hook to stash tool output.

    The SDK fires PostToolUse hooks asynchronously via ``start_soon()`` —
    the next message (AssistantMessage/ResultMessage) can arrive before the
    hook completes and stashes its output.  This function bridges that gap
    by waiting on the ``_stash_event``, which is signaled by
    :func:`stash_pending_tool_output`.

    After the event fires, callers should ``await asyncio.sleep(0)`` to
    give any remaining concurrent hooks a chance to complete.

    Returns ``True`` if a stash signal was received, ``False`` on timeout.
    The timeout is a safety net — normally the stash happens within
    microseconds of yielding to the event loop.
    """
    event = _stash_event.get(None)
    if event is None:
        return False
    # Fast path: hook already completed before we got here.
    if event.is_set():
        event.clear()
        return True
    # Slow path: wait for the hook to signal.
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
    """Execute a tool synchronously and return MCP-formatted response."""
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

    # Stash the full output before the SDK potentially truncates it.
    pending = _pending_tool_outputs.get(None)
    if pending is not None:
        pending.setdefault(base_tool.name, []).append(text)

    content_blocks: list[dict[str, str]] = [{"type": "text", "text": text}]

    # If the tool result contains inline image data, add an MCP image block
    # so Claude can "see" the image (e.g. read_workspace_file on a small PNG).
    image_block = _extract_image_block(text)
    if image_block:
        content_blocks.append(image_block)

    return {
        "content": content_blocks,
        "isError": not result.success,
    }


# MIME types that Claude can process as image content blocks.
_SUPPORTED_IMAGE_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/gif", "image/webp"}
)


def _extract_image_block(text: str) -> dict[str, str] | None:
    """Extract an MCP image content block from a tool result JSON string.

    Detects workspace file responses with ``content_base64`` and an image
    MIME type, returning an MCP-format image block that allows Claude to
    "see" the image.  Returns ``None`` if the result is not an inline image.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    mime_type = data.get("mime_type", "")
    base64_content = data.get("content_base64", "")

    # Only inline small images — large ones would exceed Claude's limits.
    # 32 KB raw ≈ ~43 KB base64.
    _MAX_IMAGE_BASE64_BYTES = 43_000
    if (
        mime_type in _SUPPORTED_IMAGE_TYPES
        and base64_content
        and len(base64_content) <= _MAX_IMAGE_BASE64_BYTES
    ):
        return {
            "type": "image",
            "data": base64_content,
            "mimeType": mime_type,
        }

    return None


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
        """Execute the wrapped tool and return MCP-formatted response."""
        user_id, session = get_execution_context()

        if session is None:
            return _mcp_error("No session context available")

        try:
            return await _execute_tool_sync(base_tool, user_id, session, args)
        except Exception as e:
            logger.error(f"Error executing tool {base_tool.name}: {e}", exc_info=True)
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
    """Read a file with optional offset/limit. Restricted to SDK working directory.

    After reading, the file is deleted to prevent accumulation in long-running pods.
    """
    file_path = args.get("file_path", "")
    offset = args.get("offset", 0)
    limit = args.get("limit", 2000)

    # Security: only allow reads under ~/.claude/projects/**/tool-results/
    real_path = os.path.realpath(file_path)
    if not real_path.startswith(_SDK_PROJECTS_DIR) or "tool-results" not in real_path:
        return {
            "content": [{"type": "text", "text": f"Access denied: {file_path}"}],
            "isError": True,
        }

    try:
        with open(real_path) as f:
            selected = list(itertools.islice(f, offset, offset + limit))
        content = "".join(selected)
        # Cleanup happens in _cleanup_sdk_tool_results after session ends;
        # don't delete here — the SDK may read in multiple chunks.
        return {"content": [{"type": "text", "text": content}], "isError": False}
    except FileNotFoundError:
        return {
            "content": [{"type": "text", "text": f"File not found: {file_path}"}],
            "isError": True,
        }
    except Exception as e:
        return {
            "content": [{"type": "text", "text": f"Error reading file: {e}"}],
            "isError": True,
        }


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


# Create the MCP server configuration
def create_copilot_mcp_server(*, use_e2b: bool = False):
    """Create an in-process MCP server configuration for CoPilot tools.

    When *use_e2b* is True, five additional MCP file tools are registered
    that route directly to the E2B sandbox filesystem, and the caller should
    disable the corresponding SDK built-in tools via
    :func:`get_sdk_disallowed_tools`.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool

    sdk_tools = []

    for tool_name, base_tool in TOOL_REGISTRY.items():
        handler = create_tool_handler(base_tool)
        decorated = tool(
            tool_name,
            base_tool.description,
            _build_input_schema(base_tool),
        )(handler)
        sdk_tools.append(decorated)

    # E2B file tools replace SDK built-in Read/Write/Edit/Glob/Grep.
    if use_e2b:
        from .e2b_file_tools import E2B_FILE_TOOLS

        for name, desc, schema, handler in E2B_FILE_TOOLS:
            decorated = tool(name, desc, schema)(handler)
            sdk_tools.append(decorated)

    # Read tool for SDK-truncated tool results (always needed).
    read_tool = tool(
        _READ_TOOL_NAME,
        _READ_TOOL_DESCRIPTION,
        _READ_TOOL_SCHEMA,
    )(_read_file_handler)
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

    from .e2b_file_tools import E2B_FILE_TOOL_NAMES

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
