from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from openai.types.chat import ChatCompletionToolParam

from backend.copilot.capabilities.dispatch import resolve_tool_dispatch
from backend.copilot.capabilities.registry import configure_tools
from backend.copilot.capabilities.sources import EAGER_CORE
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
from .consult_teammate import ConsultTeammateTool
from .create_agent import CreateAgentTool
from .customize_agent import CustomizeAgentTool
from .decompose_goal import DecomposeGoalTool
from .delegate_to_expert import DelegateToExpertTool
from .describe_capability import DescribeCapabilityTool
from .edit_agent import EditAgentTool
from .enter_building_mode import EnterAgentBuildingModeTool
from .expert_chats import ListExpertChatsTool, ReadExpertChatTool
from .expert_onboarding import ExpertOnboardingTool
from .expert_resources import (
    GrantExpertCredentialTool,
    InstallExpertWorkflowTool,
    ListExpertCredentialsTool,
    ListExpertWorkflowsTool,
    RemoveExpertWorkflowTool,
    RequestCredentialGrantTool,
    RevokeExpertCredentialTool,
)
from .feature_requests import CreateFeatureRequestTool, SearchFeatureRequestsTool
from .find_agent import FindAgentTool
from .find_capability import FindCapabilityTool
from .find_library_agent import FindLibraryAgentTool
from .find_session import FindSessionTool
from .fix_agent import FixAgentGraphTool
from .get_agent_building_guide import GetAgentBuildingGuideTool
from .get_doc_page import GetDocPageTool
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
from .manage_schedules import (
    DeleteScheduleTool,
    ListSchedulesTool,
    PauseScheduleTool,
    ResumeScheduleTool,
)
from .message_session import MessageSessionTool
from .models import ErrorResponse
from .platform_info import PlatformInfoTool
from .raise_expert import RaiseExpertTool
from .resume_capability import ResumeCapabilityTool
from .routines import ListRoutinesTool, ScheduleRoutineTool
from .run_agent import RunAgentTool
from .run_capability import RunCapabilityTool
from .run_sub_session import RunSubSessionTool
from .schedule_followup import ScheduleFollowupTool
from .search_docs import SearchDocsTool
from .setup_agent_webhook_trigger import SetupAgentWebhookTriggerTool
from .skills import DeleteSkillTool, ListSkillsTool, ReadSkillTool, StoreSkillTool
from .start_desktop import StartDesktopTool
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
    from backend.copilot.model import ChatSession, ChatSessionOrigin

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
    "find_library_agent": FindLibraryAgentTool(),
    # Capability registry: one discovery/execution surface for blocks, MCP
    # servers and the deferred platform tools (see EAGER_CORE).
    "find_capability": FindCapabilityTool(),
    "describe_capability": DescribeCapabilityTool(),
    "run_capability": RunCapabilityTool(),
    "resume_capability": ResumeCapabilityTool(),
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
    "pause_schedule": PauseScheduleTool(),
    "resume_schedule": ResumeScheduleTool(),
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
    "run_sub_session": RunSubSessionTool(),
    "get_sub_session_result": GetSubSessionResultTool(),
    "consult_teammate": ConsultTeammateTool(),
    "find_session": FindSessionTool(),
    "message_session": MessageSessionTool(),
    "delegate_to_expert": DelegateToExpertTool(),
    "list_team": ListTeamTool(),
    "TodoWrite": TodoWriteTool(),
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
    "start_desktop": StartDesktopTool(),
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
    # Expert resources: an expert installs onto itself, Otto names the
    # expert. Credential grants are the owner's call, so Otto only.
    "install_expert_workflow": InstallExpertWorkflowTool(),
    "remove_expert_workflow": RemoveExpertWorkflowTool(),
    "list_expert_workflows": ListExpertWorkflowsTool(),
    # Standing work: what the expert offers to do unattended, and the round
    # trip that turns one of those offers into a real cadence.
    "list_routines": ListRoutinesTool(),
    "schedule_routine": ScheduleRoutineTool(),
    "list_expert_credentials": ListExpertCredentialsTool(),
    "grant_expert_credential": GrantExpertCredentialTool(),
    "revoke_expert_credential": RevokeExpertCredentialTool(),
    "request_credential_grant": RequestCredentialGrantTool(),
}

# Export individual tool instances for backwards compatibility
find_agent_tool = TOOL_REGISTRY["find_agent"]
run_agent_tool = TOOL_REGISTRY["run_agent"]

# Tools the model does not see in its tool list; they are reached by id
# through ``run_capability`` (their schema arrives via ``describe_capability``).
# Keeping the prefix to the eager core is what makes the cold prompt cheap.
DEFERRED_TOOL_NAMES: frozenset[str] = frozenset(TOOL_REGISTRY) - EAGER_CORE


# Capability groups a tool may belong to.  The service layer can hide all
# tools in a group when the backing capability isn't available to this user
# (e.g. Graphiti memory behind a feature flag), so the model doesn't reach
# for tools whose backend is off and then hit opaque runtime errors.  Add
# a new group by extending ``ToolGroup`` and registering its members in
# ``TOOL_GROUPS`` below.
ToolGroup = Literal[
    "graphiti", "experts", "expert_admin", "delegation", "expert_resources"
]

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
    # A consult is read-only and costs one bounded completion, but it is
    # meaningless without teammates to ask, so it rides the same gate.
    "consult_teammate": "delegation",
    # Reaching an existing session is a teammate action: it needs someone to
    # reach, and works from either side of session.expert_id.
    "find_session": "delegation",
    "message_session": "delegation",
    # Workflow installs work from either side; credential grants are
    # owner-only, so they ride the staffing gate.
    "install_expert_workflow": "expert_resources",
    "remove_expert_workflow": "expert_resources",
    "list_expert_workflows": "expert_resources",
    # Routines ride the same gate as workflow installs: an expert manages its
    # own standing work, and personal AutoPilot manages any expert's — and,
    # with no expert named, the account's own.
    "list_routines": "expert_resources",
    "schedule_routine": "expert_resources",
    "list_expert_credentials": "expert_resources",
    "grant_expert_credential": "expert_admin",
    "revoke_expert_credential": "expert_admin",
    # Only an expert has someone to ask.
    "request_credential_grant": "experts",
    # Read-only, but it shares the same gate: with the flag off there is no
    # team to list.
    "list_team": "delegation",
}


# The capability registry indexes this registry; hand it over now that both
# exist (the registry package cannot import them without a cycle).
configure_tools(TOOL_REGISTRY, TOOL_GROUPS)


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
        return ["experts", "expert_admin", "delegation", "expert_resources"]
    return ["expert_admin"] if expert_id else ["experts"]


# The tools ``autopilot_session_guard`` refuses off an interactive origin:
# hidden there rather than declared and then refused.  Not a ``ToolGroup`` —
# that says what a tool does, and each of these already holds ``expert_admin``
# — and not a wider "needs a person" set either, because ``automation`` marks
# a machine-authored PROMPT, not an empty chat: a dream pass and a scheduled
# brief both carry it and both expect the user to read and reply.  So what is
# safe to withhold on this seam is what the runtime already withholds, no
# more.  ``tool_schema_test`` asserts the two stay equal.
INTERACTIVE_ORIGIN_TOOLS: frozenset[str] = frozenset(
    {
        "hire_expert",
        "raise_expert",
        "update_expert",
        "confirm_expert_change",
    }
)


def origin_disabled_tools(origin: "ChatSessionOrigin | None") -> frozenset[str]:
    """Tools to hide from a session *origin* no person is driving.

    Positive match, so a legacy ``None`` is hidden from too — an unknown
    origin cannot prove a human is here, and ``autopilot_session_guard``
    takes the same side of the same unknown, which is the point: whatever
    it would refuse, this stops us declaring.
    """
    return frozenset() if origin == "interactive" else INTERACTIVE_ORIGIN_TOOLS


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
    include_deferred: bool = False,
) -> list[ChatCompletionToolParam]:
    """Return OpenAI tool schemas for tools available in the current environment.

    Called per-request so that env-var or binary availability is evaluated
    fresh each time (e.g. browser_* tools are excluded when agent-browser
    CLI is not installed).  Tools belonging to any *disabled_groups* are
    also filtered out — use this to hide capability-gated tools (e.g.
    ``graphiti`` when the memory backend is off for the current user).
    *disabled_tools* hides individual tools for gates that don't follow the
    group split, e.g. ``kickoff_turn_disabled_tools`` on a hire's first turn.
    ``DEFERRED_TOOL_NAMES`` are left out unless *include_deferred*: the model
    reaches them through ``run_capability``.
    """
    hidden = tool_names_in_groups(disabled_groups) | frozenset(disabled_tools)
    if not include_deferred:
        hidden |= DEFERRED_TOOL_NAMES
    return [
        tool.as_openai_tool()
        for name, tool in TOOL_REGISTRY.items()
        if tool.is_available and name not in hidden
    ]


def get_tool(tool_name: str) -> BaseTool | None:
    """Get a tool instance by name."""
    return TOOL_REGISTRY.get(tool_name)


def reachable_tool_names(
    *,
    disabled_groups: Iterable[ToolGroup] = (),
    disabled_tools: Iterable[str] = (),
) -> frozenset[str]:
    """Names this turn can actually run, declared or reached by id.

    Since the swap, a tool being absent from the schema list no longer means
    the turn cannot run it: 57 of them are reached through ``run_capability``
    instead, bounded by the same hidden set. Gate tests that ask "can this
    session still do X" want this, not ``get_available_tools``; asking the
    schema list alone reads every deferred tool as removed.
    """
    hidden = tool_names_in_groups(disabled_groups) | frozenset(disabled_tools)
    declared = {
        name
        for name, tool in TOOL_REGISTRY.items()
        if tool.is_available and name not in hidden and name not in DEFERRED_TOOL_NAMES
    }
    if "run_capability" not in declared:
        # Nothing reaches a deferred tool without the tool that runs them.
        return frozenset(declared)
    return frozenset(declared) | frozenset(
        name
        for name in DEFERRED_TOOL_NAMES
        if name not in hidden and TOOL_REGISTRY[name].is_available
    )


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

    ``DEFERRED_TOOL_NAMES`` are refused when the model names one directly:
    they are reached by id through ``run_capability``, whose dispatch this
    function resolves back into a call to the tool itself.

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

    # A deferred tool is absent from every schema list, but a model that
    # names one anyway reached it here and ran it -- routing around
    # ``run_capability`` and the permission and envelope gates it applies.
    # Deferred-ness is a property of the tool, not of the turn, so it is
    # refused here rather than left to each caller's gate.
    if tool_name in DEFERRED_TOOL_NAMES or (
        tool_name in tool_names_in_groups(disabled_groups)
        or tool_name in frozenset(disabled_tools)
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

    # A dispatch of a platform tool IS a call to that tool, so it runs the rest
    # of this path under its own name: the refusal above still answers the model
    # that named a deferred tool directly, because it ran before the resolve.
    if dispatch := resolve_tool_dispatch(tool_name, parameters):
        tool, tool_name, parameters = dispatch

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
