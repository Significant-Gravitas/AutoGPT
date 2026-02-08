"""Tool adapter for wrapping existing CoPilot tools as Claude Agent SDK MCP tools.

This module provides the adapter layer that converts existing BaseTool implementations
into in-process MCP tools that can be used with the Claude Agent SDK.
"""

import json
import logging
from contextvars import ContextVar
from typing import Any

from backend.api.features.chat.model import ChatSession
from backend.api.features.chat.tools import TOOL_REGISTRY
from backend.api.features.chat.tools.base import BaseTool

logger = logging.getLogger(__name__)

# Context variables to pass user/session info to tool execution
_current_user_id: ContextVar[str | None] = ContextVar("current_user_id", default=None)
_current_session: ContextVar[ChatSession | None] = ContextVar(
    "current_session", default=None
)
_current_tool_call_id: ContextVar[str | None] = ContextVar(
    "current_tool_call_id", default=None
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


def get_execution_context() -> tuple[str | None, ChatSession | None, str | None]:
    """Get the current execution context."""
    return (
        _current_user_id.get(),
        _current_session.get(),
        _current_tool_call_id.get(),
    )


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
            result = await base_tool.execute(
                user_id=user_id,
                session=session,
                tool_call_id=tool_call_id or "sdk-call",
                **args,
            )

            # The result is a StreamToolOutputAvailable, extract the output
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            result.output
                            if isinstance(result.output, str)
                            else json.dumps(result.output)
                        ),
                    }
                ],
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
            "inputSchema": {
                "type": "object",
                "properties": base_tool.parameters.get("properties", {}),
                "required": base_tool.parameters.get("required", []),
            },
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
            # Get the handler
            handler = create_tool_handler(base_tool)

            # Create the decorated tool
            # The @tool decorator expects (name, description, schema)
            # Pass full JSON schema with type, properties, and required
            decorated = tool(
                tool_name,
                base_tool.description,
                {
                    "type": "object",
                    "properties": base_tool.parameters.get("properties", {}),
                    "required": base_tool.parameters.get("required", []),
                },
            )(handler)

            sdk_tools.append(decorated)

        # Create the MCP server
        server = create_sdk_mcp_server(
            name="copilot",
            version="1.0.0",
            tools=sdk_tools,
        )

        return server

    except ImportError:
        logger.warning(
            "claude-agent-sdk not available, returning tool definitions only"
        )
        return {
            "tools": get_tool_definitions(),
            "handlers": get_tool_handlers(),
        }


# List of tool names for allowed_tools configuration
COPILOT_TOOL_NAMES = [f"mcp__copilot__{name}" for name in TOOL_REGISTRY.keys()]

# Also export the raw tool names for flexibility
RAW_TOOL_NAMES = list(TOOL_REGISTRY.keys())
