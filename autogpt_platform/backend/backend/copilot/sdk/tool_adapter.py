"""Tool adapter for wrapping existing CoPilot tools as Claude Agent SDK MCP tools.

This module provides the adapter layer that converts existing BaseTool implementations
into in-process MCP tools that can be used with the Claude Agent SDK.
"""

import asyncio
import itertools
import json
import logging
import os
import re
import uuid
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from claude_agent_sdk import create_sdk_mcp_server, tool

from backend.copilot.model import ChatSession
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools.base import BaseTool
from backend.util.truncate import truncate

from .e2b_file_tools import E2B_FILE_TOOL_NAMES, E2B_FILE_TOOLS

if TYPE_CHECKING:
    from e2b import AsyncSandbox

logger = logging.getLogger(__name__)

# Allowed base directory for the Read tool (SDK saves oversized tool results here).
# Restricted to ~/.claude/projects/ and further validated to require "tool-results"
# in the path — prevents reading settings, credentials, or other sensitive files.
_SDK_PROJECTS_DIR = os.path.realpath(os.path.expanduser("~/.claude/projects"))

# Max MCP response size in chars — keeps tool output under the SDK's 10 MB JSON buffer.
_MCP_MAX_CHARS = 500_000

# Context variable holding the encoded project directory name for the current
# session (e.g. "-private-tmp-copilot-<uuid>").  Set by set_execution_context()
# so that path validation can scope tool-results reads to the current session.
_current_project_dir: ContextVar[str] = ContextVar("_current_project_dir", default="")


def _encode_cwd_for_cli(cwd: str) -> str:
    """Encode a working directory path the same way the Claude CLI does.

    The CLI replaces all non-alphanumeric characters with ``-``.
    """
    return re.sub(r"[^a-zA-Z0-9]", "-", os.path.realpath(cwd))


def is_allowed_local_path(path: str, sdk_cwd: str | None = None) -> bool:
    """Check whether *path* is an allowed host-filesystem path.

    Allowed:
    - Files under *sdk_cwd* (``/tmp/copilot-<session>/``)
    - Files under ``~/.claude/projects/<encoded-cwd>/`` — the SDK's
      project directory for this session (tool-results, transcripts, etc.)

    Both checks are scoped to the **current session** so sessions cannot
    read each other's data.
    """
    if not path:
        return False

    if path.startswith("~"):
        resolved = os.path.realpath(os.path.expanduser(path))
    elif not os.path.isabs(path) and sdk_cwd:
        resolved = os.path.realpath(os.path.join(sdk_cwd, path))
    else:
        resolved = os.path.realpath(path)

    # Allow access within the SDK working directory
    if sdk_cwd:
        norm_cwd = os.path.realpath(sdk_cwd)
        if resolved == norm_cwd or resolved.startswith(norm_cwd + os.sep):
            return True

    # Allow access within the current session's CLI project directory
    # (~/.claude/projects/<encoded-cwd>/).
    encoded = _current_project_dir.get("")
    if encoded:
        session_project = os.path.join(_SDK_PROJECTS_DIR, encoded)
        if resolved == session_project or resolved.startswith(session_project + os.sep):
            return True

    return False


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
    _current_project_dir.set(_encode_cwd_for_cli(sdk_cwd) if sdk_cwd else "")
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

    # If the tool result contains inline multimodal data, add a content block
    # so Claude can "see" images or read documents (e.g. read_workspace_file).
    # When successful, strip the (potentially huge) base64 from the text block
    # to avoid duplicating the payload — the multimodal block carries it.
    content_block = _extract_content_block(text)
    if content_block:
        text = _strip_base64_from_text(text)

    content_blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if content_block:
        content_blocks.append(content_block)

    return {
        "content": content_blocks,
        "isError": not result.success,
    }


# ---------------------------------------------------------------------------
# Multimodal content block support
# ---------------------------------------------------------------------------
# Each entry maps a MIME type to a ``(block_type, max_base64_bytes)`` tuple.
#   • "image"    → MCP image block (Claude vision, ≤20 MB raw / ~27 MB b64)
#   • "document" → Claude document block (PDF, ≤32 MB raw / ~43 MB b64)
#
# To support a new file type, add a single entry here.
# ---------------------------------------------------------------------------

_IMAGE_MAX_B64 = 28_000_000  # ~20 MB raw → ceil(20*1024*1024 * 4/3) ≈ 27_962_028
_DOCUMENT_MAX_B64 = 43_000_000  # ~32 MB raw

_MULTIMODAL_TYPES: dict[str, tuple[str, int]] = {
    # Images
    "image/png": ("image", _IMAGE_MAX_B64),
    "image/jpeg": ("image", _IMAGE_MAX_B64),
    "image/gif": ("image", _IMAGE_MAX_B64),
    "image/webp": ("image", _IMAGE_MAX_B64),
    # Documents
    "application/pdf": ("document", _DOCUMENT_MAX_B64),
}

# Block-type → builder function.  Keeps _extract_content_block flat.
_BLOCK_BUILDERS: dict[str, Callable[[str, str], dict[str, Any]]] = {
    "image": lambda mime, b64: {
        "type": "image",
        "data": b64,
        "mimeType": mime,
    },
    "document": lambda mime, b64: {
        "type": "document",
        "source": {"type": "base64", "media_type": mime, "data": b64},
    },
}


def _extract_content_block(text: str) -> dict[str, Any] | None:
    """Extract a multimodal content block from a tool result JSON string.

    Detects workspace file responses with ``content_base64`` and a supported
    MIME type, returning the appropriate content block so Claude can process
    the file (images via vision, PDFs via document support, etc.).

    Returns ``None`` if the result is not a supported multimodal type or
    exceeds size limits.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    raw_mime = data.get("mime_type", "")
    base64_content = data.get("content_base64", "")
    if not isinstance(raw_mime, str) or not isinstance(base64_content, str):
        return None
    if not raw_mime or not base64_content:
        return None

    # Normalise: strip parameters (e.g. "application/pdf; charset=binary").
    mime_type = raw_mime.split(";", 1)[0].strip().lower()
    entry = _MULTIMODAL_TYPES.get(mime_type)
    if entry is None:
        return None

    block_type, max_b64 = entry
    if len(base64_content) > max_b64:
        return None

    builder = _BLOCK_BUILDERS.get(block_type)
    if builder is None:
        return None

    return builder(mime_type, base64_content)


def _strip_base64_from_text(text: str) -> str:
    """Replace ``content_base64`` in a JSON string with a short placeholder.

    Called when a multimodal content block has already been created from the
    base64 data, so the text block only needs the metadata (file name, MIME
    type, etc.) for Claude to reference.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(data, dict) and "content_base64" in data:
        data["content_base64"] = "(see attached content block)"
        return json.dumps(data)
    return text


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
    """Read a local file with optional offset/limit.

    Only allows paths that pass :func:`is_allowed_local_path` — the current
    session's tool-results directory and ephemeral working directory.
    """
    file_path = args.get("file_path", "")
    offset = args.get("offset", 0)
    limit = args.get("limit", 2000)

    if not is_allowed_local_path(file_path):
        return {
            "content": [{"type": "text", "text": f"Access denied: {file_path}"}],
            "isError": True,
        }

    resolved = os.path.realpath(os.path.expanduser(file_path))
    try:
        with open(resolved) as f:
            selected = list(itertools.islice(f, offset, offset + limit))
        content = "".join(selected)
        # Cleanup happens in _cleanup_sdk_tool_results after session ends;
        # don't delete here — the SDK may read in multiple chunks.
        return {
            "content": [{"type": "text", "text": content}],
            "isError": False,
        }
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


# ---------------------------------------------------------------------------
# MCP result helpers
# ---------------------------------------------------------------------------


def _split_content_blocks(
    result: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split an MCP result's content into text blocks and non-text blocks.

    Returns ``(text_blocks, non_text_blocks)`` so callers can truncate only
    the text portion without corrupting binary data (base64 images/documents).
    """
    content = result.get("content", [])
    text_blocks: list[dict[str, Any]] = []
    non_text_blocks: list[dict[str, Any]] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") != "text":
                non_text_blocks.append(block)
            else:
                text_blocks.append(block)
    else:
        # Unexpected shape — treat the whole thing as text-like.
        text_blocks = content  # type: ignore[assignment]
    return text_blocks, non_text_blocks


def _text_from_mcp_result(result: dict[str, Any]) -> str:
    """Extract concatenated text from an MCP response's content blocks."""
    text_blocks, _ = _split_content_blocks(result)
    return "".join(
        b.get("text", "")
        for b in text_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )


def create_copilot_mcp_server(*, use_e2b: bool = False):
    """Create an in-process MCP server configuration for CoPilot tools.

    When *use_e2b* is True, five additional MCP file tools are registered
    that route directly to the E2B sandbox filesystem, and the caller should
    disable the corresponding SDK built-in tools via
    :func:`get_sdk_disallowed_tools`.
    """

    def _truncating(fn, tool_name: str):
        """Wrap a tool handler so its response is truncated to stay under the
        SDK's 10 MB JSON buffer, and stash the (truncated) output for the
        response adapter before the SDK can apply its own head-truncation.

        Applied once to every registered tool."""

        async def wrapper(args: dict[str, Any]) -> dict[str, Any]:
            result = await fn(args)

            # Separate non-text content blocks (images, documents) before
            # truncation — truncate() recursively shortens ALL strings,
            # which would corrupt base64 data in multimodal blocks.
            text_only_content, non_text_blocks = _split_content_blocks(result)

            # Truncate only the text portion of the result.
            truncated = truncate(
                {**result, "content": text_only_content}, _MCP_MAX_CHARS
            )

            # Re-attach non-text blocks (images, documents) intact.
            if non_text_blocks:
                truncated_content = truncated.get("content", [])
                if isinstance(truncated_content, list):
                    truncated["content"] = truncated_content + non_text_blocks
                else:
                    truncated["content"] = non_text_blocks

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
        decorated = tool(
            tool_name,
            base_tool.description,
            _build_input_schema(base_tool),
        )(_truncating(handler, tool_name))
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
