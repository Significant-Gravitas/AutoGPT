"""The per-kind gates that guard capability execution.

``run_capability`` and ``resume_capability`` both run capabilities, so both
answer to the same gates. They live here rather than in either tool so
neither has to reach into the other's privates for them.
"""

from backend.copilot.context import (
    get_current_envelope,
    get_current_hidden_tools,
    get_current_permissions,
)
from backend.copilot.permissions import ALL_TOOL_NAMES

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


def gate_denied_error(name: str, session_id: str) -> ErrorResponse:
    return ErrorResponse(
        message=(
            f"'{name}' is not available in this session's permissions. "
            "If the task needs it, say so instead of working around it."
        ),
        error="tool_disabled",
        session_id=session_id,
    )
