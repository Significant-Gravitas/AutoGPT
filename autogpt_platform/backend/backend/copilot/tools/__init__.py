from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from openai.types.chat import ChatCompletionToolParam

from backend.copilot.tracking import track_tool_called

from .add_understanding import AddUnderstandingTool
from .agent_browser import BrowserActTool, BrowserNavigateTool, BrowserScreenshotTool
from .agent_output import AgentOutputTool
from .base import BaseTool
from .bash_exec import BashExecTool
from .connect_integration import ConnectIntegrationTool
from .continue_run_block import ContinueRunBlockTool
from .create_agent import CreateAgentTool
from .customize_agent import CustomizeAgentTool
from .decompose_goal import DecomposeGoalTool
from .edit_agent import EditAgentTool
from .feature_requests import CreateFeatureRequestTool, SearchFeatureRequestsTool
from .find_agent import FindAgentTool
from .find_block import FindBlockTool
from .find_library_agent import FindLibraryAgentTool
from .fix_agent import FixAgentGraphTool
from .get_agent_building_guide import GetAgentBuildingGuideTool
from .get_doc_page import GetDocPageTool
from .get_mcp_guide import GetMCPGuideTool
from .get_sub_session_result import GetSubSessionResultTool
from .graphiti_forget import MemoryForgetConfirmTool, MemoryForgetSearchTool
from .graphiti_search import MemorySearchTool
from .graphiti_store import MemoryStoreTool
from .list_agent_triggers import ListAgentTriggersTool
from .manage_folders import (
    CreateFolderTool,
    DeleteFolderTool,
    ListFoldersTool,
    MoveAgentsToFolderTool,
    MoveFolderTool,
    UpdateFolderTool,
)
from .manage_schedules import DeleteScheduleTool, ListSchedulesTool
from .platform_info import PlatformInfoTool
from .run_agent import RunAgentTool
from .run_block import RunBlockTool
from .run_mcp_tool import RunMCPToolTool
from .run_sub_session import RunSubSessionTool
from .schedule_followup import ScheduleFollowupTool
from .search_docs import SearchDocsTool
from .skills import DeleteSkillTool, ListSkillsTool, ReadSkillTool, StoreSkillTool
from .todo_write import TodoWriteTool
from .validate_agent import ValidateAgentGraphTool
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool
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
        # Workflow tools
        TodoWriteTool(),
        RunMCPToolTool(),
        GetMCPGuideTool(),
        SearchDocsTool(),
        GetDocPageTool(),
        AddUnderstandingTool(),
        # Graphiti memory tools
        MemoryForgetConfirmTool(),
        MemoryForgetSearchTool(),
        MemorySearchTool(),
        MemoryStoreTool(),
        # Skills (self-distilled procedure registry; see tools/skills.py).
        # Defaults seed the agent-building / MCP guides so the registry is
        # the single discovery surface for both built-in and user knowledge.
        StoreSkillTool(),
        ReadSkillTool(),
        DeleteSkillTool(),
        ListSkillsTool(),
        # Platform info (subscription, billing)
        PlatformInfoTool(),
        # Building and managing agents
        GetAgentBuildingGuideTool(),
        DecomposeGoalTool(),
        CreateAgentTool(),
        CustomizeAgentTool(),
        EditAgentTool(),
        FindAgentTool(),
        FindBlockTool(),
        FindLibraryAgentTool(),
        ValidateAgentGraphTool(),
        FixAgentGraphTool(),
        # Library folder management tools
        CreateFolderTool(),
        ListFoldersTool(),
        UpdateFolderTool(),
        MoveFolderTool(),
        DeleteFolderTool(),
        MoveAgentsToFolderTool(),
        # Running agents and blocks
        RunAgentTool(),
        RunBlockTool(),
        ContinueRunBlockTool(),
        RunSubSessionTool(),
        GetSubSessionResultTool(),
        AgentOutputTool(),
        # Schedule management
        ListSchedulesTool(),
        DeleteScheduleTool(),
        ScheduleFollowupTool(),
        # Trigger discovery (parent agent → its triggers)
        ListAgentTriggersTool(),
        # Web fetch for safe URL retrieval
        WebFetchTool(),
        WebSearchTool(),
        # Agent-browser multi-step automation (navigate, act, screenshot)
        BrowserNavigateTool(),
        BrowserActTool(),
        BrowserScreenshotTool(),
        # Sandboxed code execution (bubblewrap)
        BashExecTool(),
        ConnectIntegrationTool(),
        # Persistent workspace tools (cloud storage, survives across sessions)
        # Feature request tools
        SearchFeatureRequestsTool(),
        CreateFeatureRequestTool(),
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


# Capability groups a tool may belong to.  The service layer can hide all
# tools in a group when the backing capability isn't available to this user
# (e.g. Graphiti memory behind a feature flag), so the model doesn't reach
# for tools whose backend is off and then hit opaque runtime errors.  Add
# a new group by extending ``ToolGroup`` and registering its members in
# ``TOOL_GROUPS`` below.
ToolGroup = Literal["graphiti"]

TOOL_GROUPS: dict[str, ToolGroup] = {
    "memory_store": "graphiti",
    "memory_search": "graphiti",
    "memory_forget_search": "graphiti",
    "memory_forget_confirm": "graphiti",
}


def tool_names_in_groups(groups: Iterable[ToolGroup]) -> frozenset[str]:
    """Return the set of tool short-names belonging to any of *groups*."""
    group_set = frozenset(groups)
    return frozenset(name for name, g in TOOL_GROUPS.items() if g in group_set)


def get_available_tools(
    *,
    disabled_groups: Iterable[ToolGroup] = (),
) -> list[ChatCompletionToolParam]:
    """Return OpenAI tool schemas for tools available in the current environment.

    Called per-request so that env-var or binary availability is evaluated
    fresh each time (e.g. browser_* tools are excluded when agent-browser
    CLI is not installed).  Tools belonging to any *disabled_groups* are
    also filtered out — use this to hide capability-gated tools (e.g.
    ``graphiti`` when the memory backend is off for the current user).
    """
    hidden = tool_names_in_groups(disabled_groups)
    return [
        tool.as_openai_tool()
        for name, tool in TOOL_REGISTRY.items()
        if tool.is_available and name not in hidden
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
