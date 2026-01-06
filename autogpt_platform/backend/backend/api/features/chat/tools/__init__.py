from typing import TYPE_CHECKING, Any

from openai.types.chat import ChatCompletionToolParam

from backend.api.features.chat.model import ChatSession

from .base import BaseTool
from .find_agent import FindAgentTool
from .run_agent import RunAgentTool

if TYPE_CHECKING:
    from backend.api.features.chat.response_model import StreamToolExecutionResult

# Initialize tool instances
find_agent_tool = FindAgentTool()
run_agent_tool = RunAgentTool()

# Export tools as OpenAI format
tools: list[ChatCompletionToolParam] = [
    find_agent_tool.as_openai_tool(),
    run_agent_tool.as_openai_tool(),
]


async def execute_tool(
    tool_name: str,
    parameters: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
    tool_call_id: str,
) -> "StreamToolExecutionResult":

    tool_map: dict[str, BaseTool] = {
        "find_agent": find_agent_tool,
        "run_agent": run_agent_tool,
    }
    if tool_name not in tool_map:
        raise ValueError(f"Tool {tool_name} not found")
    return await tool_map[tool_name].execute(
        user_id, session, tool_call_id, **parameters
    )
