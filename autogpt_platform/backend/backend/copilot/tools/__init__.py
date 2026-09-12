from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from openai.types.chat import ChatCompletionToolParam

from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.tracking import track_tool_called

from .add_understanding import AddUnderstandingTool
from .agent_browser import BrowserActTool, BrowserNavigateTool, BrowserScreenshotTool
from .agent_output import AgentOutputTool
from .ask_question import AskQuestionTool
from .base import BaseTool
from .bash_exec import BashExecTool
from .chat_platform import (
    EditChatPlatformMessageTool,
    ListChatPlatformChannelsTool,
    PostToChatPlatformTool,
)
from .confirm_expert_change import ConfirmExpertChangeTool
from .connect_integration import ConnectIntegrationTool
from .continue_run_block import ContinueRunBlockTool
from .create_agent import CreateAgentTool
from .customize_agent import CustomizeAgentTool
from .decompose_goal import DecomposeGoalTool
from .delegate_to_expert import DelegateToExpertTool
from .edit_agent import EditAgentTool
from .enter_building_mode import EnterAgentBuildingModeTool
from .expert_chats import ListExpertChatsTool, ReadExpertChatTool
from .expert_onboarding import ExpertOnboardingTool
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
from .handoff_to_expert import HandoffToExpertTool
from .hire_expert import HireExpertTool
from .list_agent_triggers import ListAgentTriggersTool
from .list_team import ListTeamTool
from .manage_folders import (
    CreateFolderTool,
    DeleteFolderTool,
    ListFoldersTool,
    MoveAgentsToFolderTool,
    MoveFolderTool,
    UpdateFolderTool,
)
from .manage_presets import DeletePresetTool, ListPresetsTool, UpdatePresetTool
from .manage_schedules import DeleteScheduleTool, ListSchedulesTool
from .models import ErrorResponse
from .platform_info import PlatformInfoTool
from .raise_expert import RaiseExpertTool
from .run_agent import RunAgentTool
from .run_block import RunBlockTool
from .run_mcp_tool import RunMCPToolTool
from .run_sub_session import RunSubSessionTool
from .schedule_followup import ScheduleFollowupTool
from .search_docs import SearchDocsTool
from .setup_agent_webhook_trigger import SetupAgentWebhookTriggerTool
from .skills import DeleteSkillTool, ListSkillsTool, ReadSkillTool, StoreSkillTool
from .todo_write import TodoWriteTool
from .update_expert import UpdateExpertTool
from .update_soul import ConfirmExpertSoulUpdateTool, UpdateExpertSoulTool
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

logger = logging.getLogger(__name__)

# Single source of truth for all tools
TOOL_REGISTRY: dict[str, BaseTool] = {
    "add_understanding": AddUnderstandingTool(),
    "ask_question": AskQuestionTool(),
    "create_agent": CreateAgentTool(),
    "customize_agent": CustomizeAgentTool(),
    "decompose_goal": DecomposeGoalTool(),
    "edit_agent": EditAgentTool(),
    "find_agent": FindAgentTool(),
    "find_block": FindBlockTool(),
    "find_library_agent": FindLibraryAgentTool(),
    # Graphiti memory tools
    "memory_forget_confirm": MemoryForgetConfirmTool(),
    "memory_forget_search": MemoryForgetSearchTool(),
    "memory_search": MemorySearchTool(),
    "memory_store": MemoryStoreTool(),
    # Folder management tools
    "create_folder": CreateFolderTool(),
    "list_folders": ListFoldersTool(),
    "update_folder": UpdateFolderTool(),
    "move_folder": MoveFolderTool(),
    "delete_folder": DeleteFolderTool(),
    "move_agents_to_folder": MoveAgentsToFolderTool(),
    "run_agent": RunAgentTool(),
    # Schedule management
    "list_schedules": ListSchedulesTool(),
    "delete_schedule": DeleteScheduleTool(),
    "schedule_followup": ScheduleFollowupTool(),
    # Proactive chat-platform output (post message / open thread on user's behalf)
    "post_to_chat_platform": PostToChatPlatformTool(),
    "edit_chat_platform_message": EditChatPlatformMessageTool(),
    "list_chat_platform_channels": ListChatPlatformChannelsTool(),
    # Trigger management (parent agent → its triggers)
    "list_agent_triggers": ListAgentTriggersTool(),
    # Webhook-trigger setup (create triggered preset + return ingress URL)
    "setup_agent_webhook_trigger": SetupAgentWebhookTriggerTool(),
    # Preset management (list / update / delete; works for triggers too)
    "list_presets": ListPresetsTool(),
    "update_preset": UpdatePresetTool(),
    "delete_preset": DeletePresetTool(),
    "run_block": RunBlockTool(),
    "continue_run_block": ContinueRunBlockTool(),
    "run_sub_session": RunSubSessionTool(),
    "get_sub_session_result": GetSubSessionResultTool(),
    "delegate_to_expert": DelegateToExpertTool(),
    "list_team": ListTeamTool(),
    "TodoWrite": TodoWriteTool(),
    "run_mcp_tool": RunMCPToolTool(),
    "get_mcp_guide": GetMCPGuideTool(),
    "view_agent_output": AgentOutputTool(),
    "search_docs": SearchDocsTool(),
    "get_doc_page": GetDocPageTool(),
    "enter_agent_building_mode": EnterAgentBuildingModeTool(),
    "get_agent_building_guide": GetAgentBuildingGuideTool(),
    # Skills (self-distilled procedure registry; see tools/skills.py).
    # Defaults seed the agent-building / MCP guides so the registry is
    # the single discovery surface for both built-in and user knowledge.
    "store_skill": StoreSkillTool(),
    "read_skill": ReadSkillTool(),
    "delete_skill": DeleteSkillTool(),
    "list_skills": ListSkillsTool(),
    # Web fetch for safe URL retrieval
    "web_fetch": WebFetchTool(),
    "web_search": WebSearchTool(),
    # Agent-browser multi-step automation (navigate, act, screenshot)
    "browser_navigate": BrowserNavigateTool(),
    "browser_act": BrowserActTool(),
    "browser_screenshot": BrowserScreenshotTool(),
    # Sandboxed code execution (bubblewrap)
    "bash_exec": BashExecTool(),
    "connect_integration": ConnectIntegrationTool(),
    # Persistent workspace tools (cloud storage, survives across sessions)
    # Feature request tools
    "search_feature_requests": SearchFeatureRequestsTool(),
    "create_feature_request": CreateFeatureRequestTool(),
    # Platform info (subscription, billing)
    "get_platform_info": PlatformInfoTool(),
    # Agent generation tools (local validation/fixing)
    "validate_agent_graph": ValidateAgentGraphTool(),
    "fix_agent_graph": FixAgentGraphTool(),
    # Workspace tools for CoPilot file operations
    "list_workspace_files": ListWorkspaceFilesTool(),
    "read_workspace_file": ReadWorkspaceFileTool(),
    "write_workspace_file": WriteWorkspaceFileTool(),
    "delete_workspace_file": DeleteWorkspaceFileTool(),
    # The hire's first turn (expert sessions only): greeting + intake card.
    "expert_onboarding": ExpertOnboardingTool(),
    # Expert Soul edits from chat (expert sessions only): preview + confirm
    "update_expert_soul": UpdateExpertSoulTool(),
    "confirm_expert_soul_update": ConfirmExpertSoulUpdateTool(),
    # Team changes from chat (Otto sessions only): preview + one
    # shared confirm.  Handoff is the expert-session counterpart.
    "hire_expert": HireExpertTool(),
    "raise_expert": RaiseExpertTool(),
    "update_expert": UpdateExpertTool(),
    "confirm_expert_change": ConfirmExpertChangeTool(),
    "handoff_to_expert": HandoffToExpertTool(),
    # Reading a teammate's chats (Otto sessions only): the user's
    # own data, read with the query the chat API uses.
    "list_expert_chats": ListExpertChatsTool(),
    "read_expert_chat": ReadExpertChatTool(),
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
ToolGroup = Literal["graphiti", "experts", "expert_admin", "delegation"]

TOOL_GROUPS: dict[str, ToolGroup] = {
    "memory_store": "graphiti",
    "memory_search": "graphiti",
    "memory_forget_search": "graphiti",
    "memory_forget_confirm": "graphiti",
    # Soul edits only make sense in an expert-scoped session; the engines
    # disable this group when the session has no expert_id.
    "update_expert_soul": "experts",
    # The intake card names the expert it belongs to, so it needs one.
    "expert_onboarding": "experts",
    "confirm_expert_soul_update": "experts",
    # A handoff transfers a task between experts, so it needs a caller with
    # an expert identity to hand it off from.
    "handoff_to_expert": "experts",
    # Oversight of the team is the user's, exercised in the Otto chat:
    # an expert must not hire its own teammates, and must not read another
    # expert's chats.  The engines disable this group whenever the session
    # HAS an expert_id (the opposite gate to ``experts`` above).
    "hire_expert": "expert_admin",
    "raise_expert": "expert_admin",
    "update_expert": "expert_admin",
    "confirm_expert_change": "expert_admin",
    "list_expert_chats": "expert_admin",
    "read_expert_chat": "expert_admin",
    # Delegation works from either side of ``session.expert_id`` (Otto
    # and expert sessions alike), so it has its own group: the engines
    # disable it only when the user's hire-experts flag is off.
    "delegate_to_expert": "delegation",
    # Read-only, but it shares the same gate: with the flag off there is no
    # team to list.
    "list_team": "delegation",
}


def expert_tool_disabled_groups(
    *, experts_enabled: bool, expert_id: str | None
) -> list[ToolGroup]:
    """Expert-team groups to disable for a turn — shared by both engines.

    Without the hire-experts flag every team tool is hidden. With it, the
    split follows the session role: an expert session loses the staffing
    tools (``expert_admin``), a plain Otto session loses the
    expert-session tools (``experts``).
    """
    if not experts_enabled:
        return ["experts", "expert_admin", "delegation"]
    return ["expert_admin"] if expert_id else ["experts"]


def tool_names_in_groups(groups: Iterable[ToolGroup]) -> frozenset[str]:
    """Return the set of tool short-names belonging to any of *groups*."""
    group_set = frozenset(groups)
    return frozenset(name for name, g in TOOL_GROUPS.items() if g in group_set)


# The one tool a freshly hired expert may reach on its kickoff turn.  That
# turn is a control message the server sends on the user's behalf, so
# nothing on it was asked for — one click on Hire once ran a Gmail send
# (SECRT-2622).  The kickoff prompt already says "call expert_onboarding
# once, and nothing else"; ``kickoff_turn_disabled_tools`` is that sentence
# as an enforcement boundary, for the replayed-transcript, prompt-injection
# and plain-disobedience cases a prompt cannot cover.
KICKOFF_TURN_TOOL = "expert_onboarding"


def kickoff_turn_disabled_tools() -> frozenset[str]:
    """Copilot tools to refuse on an expert's kickoff turn.

    Scoped to this registry on purpose: the SDK built-ins it leaves alone
    (file and shell tools) are confined to the session workspace by the
    security hooks, so they reach nothing the user would have to undo.
    Everything that can touch the world outside — ``run_agent``,
    ``schedule_followup``, ``post_to_chat_platform`` — lives here.
    """
    return frozenset(TOOL_REGISTRY) - {KICKOFF_TURN_TOOL}


def get_available_tools(
    *,
    disabled_groups: Iterable[ToolGroup] = (),
    disabled_tools: Iterable[str] = (),
) -> list[ChatCompletionToolParam]:
    """Return OpenAI tool schemas for tools available in the current environment.

    Called per-request so that env-var or binary availability is evaluated
    fresh each time (e.g. browser_* tools are excluded when agent-browser
    CLI is not installed).  Tools belonging to any *disabled_groups* are
    also filtered out — use this to hide capability-gated tools (e.g.
    ``graphiti`` when the memory backend is off for the current user).
    *disabled_tools* hides individual tools for gates that don't follow the
    group split, e.g. ``kickoff_turn_disabled_tools`` on a hire's first turn.
    """
    hidden = tool_names_in_groups(disabled_groups) | frozenset(disabled_tools)
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
    *,
    disabled_groups: Iterable[ToolGroup],
    disabled_tools: Iterable[str],
) -> StreamToolOutputAvailable:
    """Execute a tool by name, refusing anything the turn disabled.

    ``get_available_tools`` only hides disabled tools from the schema list it
    hands the model, which is a presentation filter: a model that names a
    hidden tool anyway (replayed transcript, prompt injection, a flag flipped
    mid-session) would still reach ``tool.execute``.  Re-checking the group
    here makes the capability gate an enforcement boundary, matching the SDK
    engine where hidden tools are never registered with the MCP server at all.

    ``disabled_groups`` and ``disabled_tools`` are keyword-only and have no
    default on purpose: they are an enforcement boundary, so a new call site
    must state its gate rather than silently inherit "nothing is disabled"
    and drop back to the presentation-only behaviour this function exists to
    close.  Both must be the same values used to build the turn's schema
    list.
    """
    tool = get_tool(tool_name)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found")

    if tool_name in tool_names_in_groups(disabled_groups) or tool_name in frozenset(
        disabled_tools
    ):
        logger.warning(
            "Refusing disabled tool: tool=%s user=%s session=%s",
            tool_name,
            user_id,
            session.session_id,
        )
        return StreamToolOutputAvailable(
            toolCallId=tool_call_id,
            toolName=tool_name,
            output=ErrorResponse(
                message=f"{tool_name} is not available in this session.",
                error="tool_disabled",
                session_id=session.session_id,
            ).model_dump_json(),
            success=False,
        )

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
