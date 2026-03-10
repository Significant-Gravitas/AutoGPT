from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from openai.types.chat import ChatCompletionToolParam

from backend.copilot.tracking import track_tool_called

from .add_understanding import AddUnderstandingTool
from .agent_browser import BrowserActTool, BrowserNavigateTool, BrowserScreenshotTool
from .agent_output import AgentOutputTool
from .base import BaseTool
from .bash_exec import BashExecTool
from .create_agent import CreateAgentTool
from .customize_agent import CustomizeAgentTool
from .edit_agent import EditAgentTool
from .feature_requests import CreateFeatureRequestTool, SearchFeatureRequestsTool
from .find_agent import FindAgentTool
from .find_block import FindBlockTool
from .find_library_agent import FindLibraryAgentTool
from .fix_agent import FixAgentGraphTool
from .get_agent_building_guide import GetAgentBuildingGuideTool
from .get_doc_page import GetDocPageTool
from .get_mcp_guide import GetMCPGuideTool
from .manage_folders import (
    CreateFolderTool,
    DeleteFolderTool,
    ListFoldersTool,
    MoveAgentsToFolderTool,
    MoveFolderTool,
    UpdateFolderTool,
)
from .run_agent import RunAgentTool
from .run_block import RunBlockTool
from .run_mcp_tool import RunMCPToolTool
from .search_docs import SearchDocsTool
from .validate_agent import ValidateAgentGraphTool
from .web_fetch import WebFetchTool
from .workspace_files import (
    DeleteWorkspaceFileTool,
    ListWorkspaceFilesTool,
    ReadWorkspaceFileTool,
    WriteWorkspaceFileTool,
)

if TYPE_CHECKING:
    from backend.copilot.model import ChatSession
    from backend.copilot.response_model import StreamToolOutputAvailable

logger = logging.getLogger(__name__)

# Single source of truth for all tools
TOOL_REGISTRY: dict[str, BaseTool] = {
    tool.name: tool
    for tool in [
        AddUnderstandingTool(),
        CreateAgentTool(),
        CustomizeAgentTool(),
        EditAgentTool(),
        FindAgentTool(),
        FindBlockTool(),
        FindLibraryAgentTool(),
        # Folder management tools
        CreateFolderTool(),
        ListFoldersTool(),
        UpdateFolderTool(),
        MoveFolderTool(),
        DeleteFolderTool(),
        MoveAgentsToFolderTool(),
        RunAgentTool(),
        RunBlockTool(),
        RunMCPToolTool(),
        GetMCPGuideTool(),
        AgentOutputTool(),
        SearchDocsTool(),
        GetDocPageTool(),
        GetAgentBuildingGuideTool(),
        # Web fetch for safe URL retrieval
        WebFetchTool(),
        # Agent-browser multi-step automation (navigate, act, screenshot)
        BrowserNavigateTool(),
        BrowserActTool(),
        BrowserScreenshotTool(),
        # Sandboxed code execution (bubblewrap)
        BashExecTool(),
        # Persistent workspace tools (cloud storage, survives across sessions)
        # Feature request tools
        SearchFeatureRequestsTool(),
        CreateFeatureRequestTool(),
        # Agent generation tools (local validation/fixing)
        ValidateAgentGraphTool(),
        FixAgentGraphTool(),
        # Workspace tools for CoPilot file operations
        ListWorkspaceFilesTool(),
        ReadWorkspaceFileTool(),
        WriteWorkspaceFileTool(),
        DeleteWorkspaceFileTool(),
    ]
}

# Export individual tool instances for backwards compatibility
find_agent_tool = TOOL_REGISTRY["find_agent"]
run_agent_tool = TOOL_REGISTRY["run_agent"]


def get_available_tools() -> list[ChatCompletionToolParam]:
    """Return OpenAI tool schemas for tools available in the current environment.

    Called per-request so that env-var or binary availability is evaluated
    fresh each time (e.g. browser_* tools are excluded when agent-browser
    CLI is not installed).
    """
    return [
        tool.as_openai_tool() for tool in TOOL_REGISTRY.values() if tool.is_available
    ]


def get_tool(tool_name: str) -> BaseTool | None:
    """Get a tool instance by name."""
    return TOOL_REGISTRY.get(tool_name)


async def execute_tool(
    tool_name: str,
    parameters: dict[str, Any],
    user_id: str | None,
    session: ChatSession,
    tool_call_id: str,
) -> "StreamToolOutputAvailable":
    """Execute a tool by name."""
    tool = get_tool(tool_name)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found")

    # Track tool call in PostHog
    logger.info(
        f"Tracking tool call: tool={tool_name}, user={user_id}, "
        f"session={session.session_id}, call_id={tool_call_id}"
    )
    track_tool_called(
        user_id=user_id,
        session_id=session.session_id,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
    )

    return await tool.execute(user_id, session, tool_call_id, **parameters)
