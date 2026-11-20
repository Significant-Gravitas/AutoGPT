"""The per-kind gates that guard capability execution.

``run_capability`` and ``resume_capability`` both run capabilities, so both
answer to the same gates. They live here rather than in either tool so
neither has to reach into the other's privates for them.
"""

from backend.copilot.context import (
    get_current_envelope,
    get_current_hidden_tools,
    get_current_permissions,
    is_unattended_turn,
)
from backend.copilot.permissions import ALL_TOOL_NAMES, ROUTINE_SELF_ESCALATION_TOOLS

from .models import ErrorResponse


def gate_denied(name: str) -> bool:
    """True when this turn may not use *name* (a tool or a kind gate)."""
    if name in get_current_hidden_tools():
        return True
    envelope = get_current_envelope()
    if envelope is not None and not envelope.permits(name):
        return True
    permissions = get_current_permissions()
    return permissions is not None and name not in permissions.effective_allowed_tools(
        ALL_TOOL_NAMES
    )


def gate_denied_error(
    name: str, session_id: str, *, describing: bool = False
) -> ErrorResponse:
    """The refusal for *name*. A *describing* lookup attempted nothing, so it
    is not told that something did not happen."""
    message = (
        f"'{name}' is not available in this session's permissions. "
        "If the task needs it, say so instead of working around it."
        if describing
        else tool_refused_message(name)
    )
    return ErrorResponse(
        message=message, error="tool_disabled", session_id=session_id
    ).from_outside()


def tool_refused_message(name: str) -> str:
    """What the model is told when a turn refuses a call to *name*: from this
    gate, from ``execute_tool``, and from the SDK engine's PreToolUse hook.

    The model reports what it did, and a refusal it reads as a hiccup turns
    into a claim: a routine run refused ``schedule_routine`` and Otto went on
    telling the user the monitoring was running (SECRT-3224). So the message
    says the call did not run, not only that the tool is missing.

    A run nobody is watching is refused anything that schedules more work, so
    it is also told where that work can still be set up: in a chat with the
    user, where the same tool is available.
    """
    message = (
        f"'{name}' is not available in this session, so the call did not run "
        "and nothing was created or changed. Tell the user it did not happen, "
        "and never say it did. If the task needs it, say so instead of working "
        "around it."
    )
    if name in ROUTINE_SELF_ESCALATION_TOOLS and is_unattended_turn():
        message += (
            " A run nobody is watching, such as a routine's, cannot schedule "
            "more work. Say that it can be set up from a chat with the user, "
            "where it is available."
        )
    return message
