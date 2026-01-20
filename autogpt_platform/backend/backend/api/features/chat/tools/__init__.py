from typing import TYPE_CHECKING, Any

from openai.types.chat import ChatCompletionToolParam

from backend.api.features.chat.model import ChatSession

from .add_understanding import AddUnderstandingTool
from .agent_output import AgentOutputTool
from .base import BaseTool
from .create_agent import CreateAgentTool
from .edit_agent import EditAgentTool
from .find_agent import FindAgentTool
from .find_block import FindBlockTool
from .find_library_agent import FindLibraryAgentTool
from .get_doc_page import GetDocPageTool
from .run_agent import RunAgentTool
from .run_block import RunBlockTool
from .search_docs import SearchDocsTool

if TYPE_CHECKING:
    from backend.api.features.chat.response_model import StreamToolOutputAvailable

# Single source of truth for all tools
TOOL_REGISTRY: dict[str, BaseTool] = {
    "add_understanding": AddUnderstandingTool(),
    "create_agent": CreateAgentTool(),
    "edit_agent": EditAgentTool(),
    "find_agent": FindAgentTool(),
    "find_block": FindBlockTool(),
    "find_library_agent": FindLibraryAgentTool(),
    "run_agent": RunAgentTool(),
    "run_block": RunBlockTool(),
    "view_agent_output": AgentOutputTool(),
    "search_docs": SearchDocsTool(),
    "get_doc_page": GetDocPageTool(),
}

# Export individual tool instances for backwards compatibility
find_agent_tool = TOOL_REGISTRY["find_agent"]
run_agent_tool = TOOL_REGISTRY["run_agent"]

# Generated from registry for OpenAI API
tools: list[ChatCompletionToolParam] = [
    tool.as_openai_tool() for tool in TOOL_REGISTRY.values()
]


async def execute_tool(
    tool_name: str,
    parameters: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
    tool_call_id: str,
) -> "StreamToolOutputAvailable":
    """Execute a tool by name."""
    tool = TOOL_REGISTRY.get(tool_name)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found")
    return await tool.execute(user_id, session, tool_call_id, **parameters)
