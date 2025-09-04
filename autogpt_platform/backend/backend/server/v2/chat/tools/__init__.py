"""Chat tools for agent discovery, setup, and execution."""

from .base import BaseTool
from .find_agent import FindAgentTool
from .get_agent_details import GetAgentDetailsTool
from .get_required_setup_info import GetRequiredSetupInfoTool
from .run_agent import RunAgentTool
from .setup_agent import SetupAgentTool

__all__ = [
    "CHAT_TOOLS",
    "BaseTool",
    "FindAgentTool",
    "GetAgentDetailsTool",
    "GetRequiredSetupInfoTool",
    "RunAgentTool",
    "SetupAgentTool",
    "find_agent_tool",
    "get_agent_details_tool",
    "get_required_setup_info_tool",
    "run_agent_tool",
    "setup_agent_tool",
]

# Initialize all tools
find_agent_tool = FindAgentTool()
get_agent_details_tool = GetAgentDetailsTool()
get_required_setup_info_tool = GetRequiredSetupInfoTool()
setup_agent_tool = SetupAgentTool()
run_agent_tool = RunAgentTool()

# Export tool instances
CHAT_TOOLS = [
    find_agent_tool,
    get_agent_details_tool,
    get_required_setup_info_tool,
    setup_agent_tool,
    run_agent_tool,
]
