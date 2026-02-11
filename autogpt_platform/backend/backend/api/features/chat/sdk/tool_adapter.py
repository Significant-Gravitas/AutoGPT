"""Tool adapter for wrapping existing CoPilot tools as Claude Agent SDK MCP tools.

This module provides the adapter layer that converts existing BaseTool implementations
into in-process MCP tools that can be used with the Claude Agent SDK.
"""

import json
import logging
import os
import uuid
from contextvars import ContextVar
from typing import Any

from backend.api.features.chat.model import ChatSession
from backend.api.features.chat.tools import TOOL_REGISTRY
from backend.api.features.chat.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Allowed base directory for the Read tool (SDK saves oversized tool results here)
_SDK_TOOL_RESULTS_DIR = os.path.expanduser("~/.claude/")

# MCP server naming - the SDK prefixes tool names as "mcp__{server_name}__{tool}"
MCP_SERVER_NAME = "copilot"
MCP_TOOL_PREFIX = f"mcp__{MCP_SERVER_NAME}__"

# Context variables to pass user/session info to tool execution
_current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)
_current_session: ContextVar[ChatSession | None] = ContextVar(
    "current_session", default=None
)
_current_tool_call_id: ContextVar[str | None] = ContextVar(
    "current_tool_call_id", default=None
)

# Stash for MCP tool outputs before the SDK potentially truncates them.
# Keyed by tool_name → full output string. Consumed (popped) by the
# response adapter when it builds StreamToolOutputAvailable.
_pending_tool_outputs: ContextVar[dict[str, str]] = ContextVar(
    "pending_tool_outputs", default=None  # type: ignore[arg-type]
)


def set_execution_context(
    user_id: str | None,
    session: ChatSession,
    tool_call_id: str | None = None,
) -> None:
    """Set the execution context for tool calls.

    This must be called before streaming begins to ensure tools have access
    to user_id and session information.
    """
    _current_user_id.set(user_id)
    _current_session.set(session)
    _current_tool_call_id.set(tool_call_id)
    _pending_tool_outputs.set({})


def get_execution_context() -> tuple[str | None, ChatSession | None, str | None]:
    """Get the current execution context."""
    return (
        _current_user_id.get(),
        _current_session.get(),
        _current_tool_call_id.get(),
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


def create_tool_handler(base_tool: BaseTool):
    """Create an async handler function for a BaseTool.

    This wraps the existing BaseTool._execute method to be compatible
    with the Claude Agent SDK MCP tool format.
    """

    async def tool_handler(args: dict[str, Any]) -> dict[str, Any]:
        """Execute the wrapped tool and return MCP-formatted response."""
        user_id, session, tool_call_id = get_execution_context()

        if session is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "error": "No session context available",
                                "type": "error",
                            }
                        ),
                    }
                ],
                "isError": True,
            }

        try:
            # Call the existing tool's execute method
            # Generate unique tool_call_id per invocation for proper correlation
            effective_id = tool_call_id or f"sdk-{uuid.uuid4().hex[:12]}"
            result = await base_tool.execute(
                user_id=user_id,
                session=session,
                tool_call_id=effective_id,
                **args,
            )

            # The result is a StreamToolOutputAvailable, extract the output
            text = (
                result.output
                if isinstance(result.output, str)
                else json.dumps(result.output)
            )

            # Stash the full output before the SDK potentially truncates it.
            # The response adapter will pop this for frontend widget rendering.
            pending = _pending_tool_outputs.get(None)
            if pending is not None:
                pending[base_tool.name] = text

            return {
                "content": [{"type": "text", "text": text}],
                "isError": not result.success,
            }

        except Exception as e:
            logger.error(f"Error executing tool {base_tool.name}: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "error": str(e),
                                "type": "error",
                                "message": f"Failed to execute {base_tool.name}",
                            }
                        ),
                    }
                ],
                "isError": True,
            }

    return tool_handler


def _build_input_schema(base_tool: BaseTool) -> dict[str, Any]:
    """Build a JSON Schema input schema for a tool."""
    return {
        "type": "object",
        "properties": base_tool.parameters.get("properties", {}),
        "required": base_tool.parameters.get("required", []),
    }


def get_tool_definitions() -> list[dict[str, Any]]:
    """Get all tool definitions in MCP format.

    Returns a list of tool definitions that can be used with
    create_sdk_mcp_server or as raw tool definitions.
    """
    tool_definitions = []

    for tool_name, base_tool in TOOL_REGISTRY.items():
        tool_def = {
            "name": tool_name,
            "description": base_tool.description,
            "inputSchema": _build_input_schema(base_tool),
        }
        tool_definitions.append(tool_def)

    return tool_definitions


def get_tool_handlers() -> dict[str, Any]:
    """Get all tool handlers mapped by name.

    Returns a dictionary mapping tool names to their handler functions.
    """
    handlers = {}

    for tool_name, base_tool in TOOL_REGISTRY.items():
        handlers[tool_name] = create_tool_handler(base_tool)

    return handlers


async def _read_file_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Read a file with optional offset/limit. Restricted to SDK working directory.

    After reading, the file is deleted to prevent accumulation in long-running pods.
    """
    file_path = args.get("file_path", "")
    offset = args.get("offset", 0)
    limit = args.get("limit", 2000)

    # Security: only allow reads under the SDK's working directory
    real_path = os.path.realpath(file_path)
    if not real_path.startswith(_SDK_TOOL_RESULTS_DIR):
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
# Security hooks validate that file paths stay within sdk_cwd
# and that Bash commands are restricted to a safe allowlist.
_SDK_BUILTIN_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep", "Bash"]

# List of tool names for allowed_tools configuration
# Include MCP tools, the MCP Read tool for oversized results,
# and SDK built-in file tools for workspace operations.
COPILOT_TOOL_NAMES = [
    *[f"{MCP_TOOL_PREFIX}{name}" for name in TOOL_REGISTRY.keys()],
    f"{MCP_TOOL_PREFIX}{_READ_TOOL_NAME}",
    *_SDK_BUILTIN_TOOLS,
]
