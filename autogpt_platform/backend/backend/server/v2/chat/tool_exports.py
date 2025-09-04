"""Chat tools for OpenAI function calling - main exports."""

import json
import logging
from typing import Any

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel

from backend.server.v2.chat.tools import (
    FindAgentTool,
    GetAgentDetailsTool,
    GetRequiredSetupInfoTool,
    RunAgentTool,
    SetupAgentTool,
)

logger = logging.getLogger(__name__)

# Initialize tool instances
find_agent_tool = FindAgentTool()
get_agent_details_tool = GetAgentDetailsTool()
get_required_setup_info_tool = GetRequiredSetupInfoTool()
setup_agent_tool = SetupAgentTool()
run_agent_tool = RunAgentTool()

# Export tools as OpenAI format
tools: list[ChatCompletionToolParam] = [
    find_agent_tool.as_openai_tool(),
    get_agent_details_tool.as_openai_tool(),
    get_required_setup_info_tool.as_openai_tool(),
    setup_agent_tool.as_openai_tool(),
    run_agent_tool.as_openai_tool(),
]


# Tool execution dispatcher
async def execute_tool(
    tool_name: str,
    parameters: dict[str, Any],
    user_id: str | None,
    session_id: str,
) -> str:
    """Execute a tool by name with the given parameters.

    Args:
        tool_name: Name of the tool to execute
        parameters: Tool parameters
        user_id: User ID (may be anonymous)
        session_id: Chat session ID

    Returns:
        JSON string result from the tool

    """
    # Map tool names to instances
    tool_map = {
        "find_agent": find_agent_tool,
        "get_agent_details": get_agent_details_tool,
        "get_required_setup_info": get_required_setup_info_tool,
        "setup_agent": setup_agent_tool,
        "run_agent": run_agent_tool,
    }

    tool = tool_map.get(tool_name)
    if not tool:
        return json.dumps(
            {
                "type": "error",
                "message": f"Unknown tool: {tool_name}",
            }
        )

    try:
        # Execute tool - returns Pydantic model
        result = await tool.execute(user_id, session_id, **parameters)

        # Convert Pydantic model to JSON string
        if isinstance(result, BaseModel):
            return result.model_dump_json(indent=2)
        # Fallback for non-Pydantic responses
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
        return json.dumps(
            {
                "type": "error",
                "message": f"Tool execution failed: {e!s}",
            }
        )
