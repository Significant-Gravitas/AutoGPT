"""How a held call is named to the user: the one table the card, the chain row,
Home and the channels all read, so none of them can word it differently."""

from typing import Any

from pydantic import BaseModel

from .references import Reference

# Tool -> (the action, the arguments that name its object, first match wins).
# Every tool the gate can hold; built from our registry, never from the reason.
_ASK: dict[str, tuple[str, tuple[str, ...]]] = {
    "bash_exec": ("Run a command in the sandbox", ()),
    "browser_act": ("Act on the open page", ()),
    "post_to_chat_platform": ("Post a message", ()),
    "edit_chat_platform_message": ("Edit a posted message", ()),
    "create_agent": ("Create an agent", ()),
    "customize_agent": ("Customize an agent", ()),
    "edit_agent": ("Edit an agent", ()),
    "fix_agent_graph": ("Fix an agent", ()),
    "enter_agent_building_mode": ("Start building an agent", ()),
    "create_folder": ("Create folder", ("name",)),
    "update_folder": ("Update folder", ("name",)),
    "move_folder": ("Move a folder", ()),
    "delete_folder": ("Delete a folder", ()),
    "move_agents_to_folder": ("Move agents into a folder", ()),
    "update_preset": ("Update preset", ("name",)),
    "delete_preset": ("Delete a preset", ()),
    "schedule_followup": ("Schedule a follow-up", ("name",)),
    "schedule_routine": ("Schedule a routine", ("title",)),
    "pause_schedule": ("Pause a schedule", ()),
    "resume_schedule": ("Resume a schedule", ()),
    "delete_schedule": ("Delete a schedule", ()),
    "setup_agent_webhook_trigger": ("Set up the trigger", ("name",)),
    "create_feature_request": ("File a feature request", ("title",)),
    "hire_expert": ("Hire", ("name",)),
    "raise_expert": ("Create teammate", ("name",)),
    "update_expert": ("Update teammate", ("name",)),
    "update_expert_soul": ("Change how this teammate works", ()),
    "confirm_expert_change": ("Confirm the team change", ()),
    "confirm_expert_soul_update": ("Confirm the teammate change", ()),
    "install_expert_workflow": ("Give a teammate a workflow", ()),
    "remove_expert_workflow": ("Take a workflow from a teammate", ()),
    "grant_expert_credential": ("Give a teammate an account", ()),
    "revoke_expert_credential": ("Take an account from a teammate", ()),
    "delegate_to_expert": ("Hand a task to a teammate", ()),
    "handoff_to_expert": ("Hand this chat to a teammate", ()),
    "message_session": ("Message another chat", ()),
    "run_sub_session": ("Start a subtask", ()),
    "delete_skill": ("Delete skill", ("name",)),
    "delete_workspace_file": ("Delete file", ("path",)),
    "memory_forget_confirm": ("Forget memories", ()),
}
# Tool -> the id whose resolved name is the object when no argument names one.
_OBJECT_ID: dict[str, str] = {
    "edit_agent": "agent_id",
    "update_folder": "folder_id",
    "move_folder": "folder_id",
    "delete_folder": "folder_id",
    "move_agents_to_folder": "folder_id",
    "update_preset": "preset_id",
    "delete_preset": "preset_id",
    "schedule_routine": "routine_id",
    "pause_schedule": "schedule_id",
    "resume_schedule": "schedule_id",
    "delete_schedule": "schedule_id",
    "update_expert": "expert_id",
    "confirm_expert_change": "confirmation_id",
    "confirm_expert_soul_update": "confirmation_id",
    "install_expert_workflow": "library_agent_id",
    "remove_expert_workflow": "workflow_id",
    "grant_expert_credential": "credential_id",
    "revoke_expert_credential": "credential_id",
    "delegate_to_expert": "expert_id",
    "handoff_to_expert": "expert_id",
    "message_session": "session_id",
    "run_sub_session": "sub_autopilot_session_id",
}
_MAX_OBJECT_CHARS = 60


class Headline(BaseModel):
    ask: str
    object: str | None = None
    # The argument the object came from, which the card does not list again.
    object_key: str | None = None

    @property
    def text(self) -> str:
        return f"{self.ask} “{self.object}”" if self.object else self.ask


def headline_for(
    tool_name: str,
    args: dict[str, Any],
    references: list[Reference] | None = None,
) -> Headline:
    ask, keys = _ASK.get(tool_name, (f"Run {tool_name.replace('_', ' ')}", ()))
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return _named(ask, key, value)
    id_key = _OBJECT_ID.get(tool_name)
    for ref in references or []:
        if ref.key == id_key and ref.name:
            return _named(ask, ref.key, ref.name)
    return Headline(ask=ask)


def gated_tools() -> frozenset[str]:
    return frozenset(_ASK)


def _named(ask: str, key: str, value: str) -> Headline:
    name = " ".join(value.split())
    if len(name) > _MAX_OBJECT_CHARS:
        name = name[: _MAX_OBJECT_CHARS - 1] + "…"
    # The card hides the argument only when the headline shows all of it.
    return Headline(ask=ask, object=name, object_key=key if name == value else None)
