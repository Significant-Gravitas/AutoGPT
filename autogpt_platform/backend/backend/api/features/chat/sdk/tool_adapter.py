"""Tool adapter for wrapping existing CoPilot tools as Claude Agent SDK MCP tools.

This module provides the adapter layer that converts existing BaseTool implementations
into in-process MCP tools that can be used with the Claude Agent SDK.
"""

import asyncio
import json
import logging
import os
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from backend.api.features.chat.model import ChatSession
from backend.api.features.chat.tools import TOOL_REGISTRY
from backend.api.features.chat.tools.base import BaseTool

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
# Stash for MCP tool outputs before the SDK potentially truncates them.
# Keyed by tool_name → full output string. Consumed (popped) by the
# response adapter when it builds StreamToolOutputAvailable.
_pending_tool_outputs: ContextVar[dict[str, str]] = ContextVar(
    "pending_tool_outputs", default=None  # type: ignore[arg-type]
)


@dataclass
class _BackgroundOp:
    """Tracks a background tool operation."""

    tool_name: str
    task: asyncio.Task[Any]
    result: dict[str, Any] | None = None
    done: bool = False


# Module-level registry for background long-running operations.
# Keyed by operation_id.  Cleaned up after result is consumed.
_background_ops: dict[str, _BackgroundOp] = {}
_background_ops_lock = asyncio.Lock()

_CHECK_OP_TOOL_NAME = "check_operation"


def set_execution_context(
    user_id: str | None,
    session: ChatSession,
) -> None:
    """Set the execution context for tool calls.

    This must be called before streaming begins to ensure tools have access
    to user_id and session information.
    """
    _current_user_id.set(user_id)
    _current_session.set(session)
    _pending_tool_outputs.set({})


def get_execution_context() -> tuple[str | None, ChatSession | None]:
    """Get the current execution context."""
    return (
        _current_user_id.get(),
        _current_session.get(),
    )


def pop_pending_tool_output(tool_name: str) -> str | None:
    """Pop and return the stashed full output for *tool_name*.

    The SDK CLI may truncate large tool results (writing them to disk and
    replacing the content with a file reference). This stash keeps the
    original MCP output so the response adapter can forward it to the
    frontend for proper widget rendering.

    Returns ``None`` if nothing was stashed for *tool_name*.
    """
    pending = _pending_tool_outputs.get(None)
    if pending is None:
        return None
    return pending.pop(tool_name, None)


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
        pending[base_tool.name] = text

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

    Long-running tools (``is_long_running=True``) are spawned as background
    tasks and return immediately with an ``operation_id``.  The SDK should
    then poll ``check_operation`` to retrieve the result.
    """

    async def tool_handler(args: dict[str, Any]) -> dict[str, Any]:
        """Execute the wrapped tool and return MCP-formatted response."""
        user_id, session = get_execution_context()

        if session is None:
            return _mcp_error("No session context available")

        # --- Long-running: fire-and-forget, return operation_id ---
        if base_tool.is_long_running:
            op_id = f"op-{uuid.uuid4().hex[:12]}"

            async def _bg_run() -> None:
                try:
                    result = await _execute_tool_sync(base_tool, user_id, session, args)
                    op = _background_ops.get(op_id)
                    if op:
                        op.result = result
                        op.done = True
                except Exception as exc:
                    op = _background_ops.get(op_id)
                    if op:
                        op.result = _mcp_error(str(exc))
                        op.done = True
                    logger.error(
                        f"Background tool {base_tool.name} failed: {exc}",
                        exc_info=True,
                    )

            task = asyncio.create_task(_bg_run())
            _background_ops[op_id] = _BackgroundOp(tool_name=base_tool.name, task=task)
            logger.info(
                f"[SDK] Long-running tool {base_tool.name} started "
                f"(operation_id={op_id})"
            )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "started",
                                "operation_id": op_id,
                                "message": (
                                    f"{base_tool.name} is running in the background. "
                                    f"Call check_operation with "
                                    f"operation_id='{op_id}' to get the result."
                                ),
                            }
                        ),
                    }
                ],
                "isError": False,
            }

        # --- Normal (fast) tool: execute synchronously ---
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
            lines = f.readlines()
        selected = lines[offset : offset + limit]
        content = "".join(selected)
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


async def _check_operation_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Check the status of a background long-running operation."""
    op_id = args.get("operation_id", "")
    if not op_id or op_id not in _background_ops:
        return _mcp_error(f"Operation '{op_id}' not found.")

    op = _background_ops[op_id]
    if not op.done:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "status": "in_progress",
                            "operation_id": op_id,
                            "tool_name": op.tool_name,
                            "message": (
                                f"{op.tool_name} is still running. "
                                "Check again in a few seconds."
                            ),
                        }
                    ),
                }
            ],
            "isError": False,
        }

    # Done — return result and clean up
    result = op.result or _mcp_error("Operation completed but no result available.")
    del _background_ops[op_id]
    logger.info(f"[SDK] Background operation {op_id} ({op.tool_name}) collected")
    return result


_CHECK_OP_DESCRIPTION = (
    "Check the status of a background operation started by a long-running tool "
    "(like create_agent). Returns the result when done, or 'in_progress' if still "
    "running. Call this periodically (every few seconds) after starting an operation."
)
_CHECK_OP_SCHEMA = {
    "type": "object",
    "properties": {
        "operation_id": {
            "type": "string",
            "description": "The operation_id returned by the long-running tool.",
        },
    },
    "required": ["operation_id"],
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
def create_copilot_mcp_server():
    """Create an in-process MCP server configuration for CoPilot tools.

    This can be passed to ClaudeAgentOptions.mcp_servers.

    Note: The actual SDK MCP server creation depends on the claude-agent-sdk
    package being available. This function returns the configuration that
    can be used with the SDK.
    """
    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool

        # Create decorated tool functions
        sdk_tools = []

        for tool_name, base_tool in TOOL_REGISTRY.items():
            handler = create_tool_handler(base_tool)
            decorated = tool(
                tool_name,
                base_tool.description,
                _build_input_schema(base_tool),
            )(handler)
            sdk_tools.append(decorated)

        # Add the Read tool so the SDK can read back oversized tool results
        read_tool = tool(
            _READ_TOOL_NAME,
            _READ_TOOL_DESCRIPTION,
            _READ_TOOL_SCHEMA,
        )(_read_file_handler)
        sdk_tools.append(read_tool)

        # Add the check_operation tool for polling background operations
        check_op_tool = tool(
            _CHECK_OP_TOOL_NAME,
            _CHECK_OP_DESCRIPTION,
            _CHECK_OP_SCHEMA,
        )(_check_operation_handler)
        sdk_tools.append(check_op_tool)

        server = create_sdk_mcp_server(
            name=MCP_SERVER_NAME,
            version="1.0.0",
            tools=sdk_tools,
        )

        return server

    except ImportError:
        # Let ImportError propagate so service.py handles the fallback
        raise


# SDK built-in tools allowed within the workspace directory.
# Security hooks validate that file paths stay within sdk_cwd.
# Bash is NOT included — use the sandboxed MCP bash_exec tool instead,
# which provides kernel-level network isolation via unshare --net.
# Task allows spawning sub-agents (rate-limited by security hooks).
_SDK_BUILTIN_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Task"]

# List of tool names for allowed_tools configuration
# Include MCP tools, the MCP Read tool for oversized results,
# and SDK built-in file tools for workspace operations.
COPILOT_TOOL_NAMES = [
    *[f"{MCP_TOOL_PREFIX}{name}" for name in TOOL_REGISTRY.keys()],
    f"{MCP_TOOL_PREFIX}{_READ_TOOL_NAME}",
    f"{MCP_TOOL_PREFIX}{_CHECK_OP_TOOL_NAME}",
    *_SDK_BUILTIN_TOOLS,
]
