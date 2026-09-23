"""What every tool does, and what each mode does about it — the part no model decides.

The supervisor (``gate/classifier.py``) only ever adds a question: every
decision here must hold even if it always answered "allow".
"""

from enum import Enum

from backend.copilot.model import AutopilotMode

DEFAULT_MODE: AutopilotMode = "auto"


class Effect(str, Enum):
    READ = "read"
    WORKSPACE = "workspace"
    SHELL = "shell"
    PLATFORM = "platform"
    EXTERNAL = "external"
    # Never gated: the call is itself a question to the user, finishes a
    # review that is already open, or answers to L1's irreversible-action pause
    # until it can name its subject.
    UNGATED = "ungated"


class Verdict(str, Enum):
    RUN = "run"
    JUDGE = "judge"
    ASK = "ask"


_READ = frozenset(
    {
        "ask_question",
        # One bounded model call with no tools; a paid read.
        "consult_teammate",
        "decompose_goal",
        "describe_capability",
        "expert_onboarding",
        "find_agent",
        "find_capability",
        "find_library_agent",
        "find_session",
        "get_agent_building_guide",
        "get_doc_page",
        "get_platform_info",
        "get_sub_session_result",
        "list_agent_triggers",
        "list_chat_platform_channels",
        "list_expert_chats",
        "list_expert_credentials",
        "list_expert_workflows",
        "list_folders",
        "list_presets",
        "list_routines",
        "list_schedules",
        "list_skills",
        "list_team",
        "list_workspace_files",
        "memory_forget_search",
        "memory_search",
        "read_expert_chat",
        "read_skill",
        "read_workspace_file",
        "search_docs",
        "search_feature_requests",
        "validate_agent_graph",
        "view_agent_output",
        "web_fetch",
        "web_search",
        "browser_navigate",
        "browser_screenshot",
    }
)

_WORKSPACE = frozenset(
    {
        "TodoWrite",
        "add_understanding",
        "memory_store",
        "start_desktop",
        "store_skill",
        "write_workspace_file",
    }
)

_PLATFORM = frozenset(
    {
        "create_agent",
        "customize_agent",
        "edit_agent",
        "enter_agent_building_mode",
        "fix_agent_graph",
        "create_folder",
        "delete_folder",
        "move_agents_to_folder",
        "move_folder",
        "update_folder",
        "delete_preset",
        "update_preset",
        "delete_schedule",
        "pause_schedule",
        "resume_schedule",
        "schedule_followup",
        "schedule_routine",
        "setup_agent_webhook_trigger",
        "create_feature_request",
        "confirm_expert_change",
        "confirm_expert_soul_update",
        "grant_expert_credential",
        "hire_expert",
        "install_expert_workflow",
        "raise_expert",
        "remove_expert_workflow",
        "revoke_expert_credential",
        "update_expert",
        "update_expert_soul",
        "delegate_to_expert",
        "handoff_to_expert",
        "message_session",
        "run_sub_session",
        # Deletes inside the workspace: the user may have put the thing there.
        "delete_skill",
        "delete_workspace_file",
        "memory_forget_confirm",
    }
)

_EXTERNAL = frozenset(
    {"browser_act", "edit_chat_platform_message", "post_to_chat_platform"}
)

_UNGATED = frozenset(
    {
        "connect_integration",
        "request_credential_grant",
        "resume_capability",
        "run_agent",
        "run_capability",
    }
)

# Registered straight onto the MCP server by ``create_copilot_mcp_server``, so
# the second seam in ``sdk/tool_adapter.py`` is the only gate they reach.
MCP_FILE_WRITE_TOOLS = frozenset({"Edit", "Write", "edit_file", "write_file"})
MCP_FILE_READ_TOOLS = frozenset(
    {"Read", "glob", "grep", "read_file", "read_tool_result"}
)

_EFFECTS: dict[str, Effect] = {
    **{name: Effect.READ for name in _READ | MCP_FILE_READ_TOOLS},
    **{name: Effect.WORKSPACE for name in _WORKSPACE | MCP_FILE_WRITE_TOOLS},
    "bash_exec": Effect.SHELL,
    **{name: Effect.PLATFORM for name in _PLATFORM},
    **{name: Effect.EXTERNAL for name in _EXTERNAL},
    **{name: Effect.UNGATED for name in _UNGATED},
}

_MODE_VERDICTS: dict[AutopilotMode, dict[Effect, Verdict]] = {
    "ask_first": {
        Effect.READ: Verdict.RUN,
        Effect.WORKSPACE: Verdict.RUN,
        Effect.SHELL: Verdict.ASK,
        Effect.PLATFORM: Verdict.ASK,
        Effect.EXTERNAL: Verdict.ASK,
    },
    "auto": {
        Effect.READ: Verdict.RUN,
        Effect.WORKSPACE: Verdict.RUN,
        Effect.SHELL: Verdict.JUDGE,
        Effect.PLATFORM: Verdict.JUDGE,
        Effect.EXTERNAL: Verdict.ASK,
    },
    "unsupervised": {effect: Verdict.RUN for effect in Effect},
}


def effect_for(tool_name: str) -> Effect:
    # A tool nobody classified is a platform edit: judged in Auto, asked in Ask First.
    return _EFFECTS.get(tool_name, Effect.PLATFORM)


def verdict_for(mode: AutopilotMode, tool_name: str) -> Verdict:
    effect = effect_for(tool_name)
    if effect is Effect.UNGATED:
        return Verdict.RUN
    return _MODE_VERDICTS[mode][effect]


def classified_tools() -> frozenset[str]:
    return frozenset(_EFFECTS)
