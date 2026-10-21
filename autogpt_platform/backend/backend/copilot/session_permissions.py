"""What a session may call, derived from the session itself.

A leaf module on purpose: it depends on the models and ``permissions`` only,
never on ``copilot.tools``. Anything that dispatches a turn on a session has
to resolve that session's own permissions, and tools are among those callers —
so keeping this out of ``builder_context`` (which imports three tool modules)
is what lets ``copilot/tools/message_session.py`` reuse it without closing an
import cycle.
"""

from backend.copilot.model import ChatSessionInfo
from backend.copilot.permissions import CopilotPermissions

# Tools hidden from builder-bound sessions: ``create_agent`` /
# ``customize_agent`` would mint a new graph (panel is bound to one),
# and ``get_agent_building_guide`` duplicates bytes already in the
# system-prompt suffix. Everything else (find_capability, find_agent, …)
# stays available so the LLM can look up ids instead of hallucinating.
BUILDER_BLOCKED_TOOLS: tuple[str, ...] = (
    "create_agent",
    "customize_agent",
    "get_agent_building_guide",
)


def resolve_session_permissions(
    session: ChatSessionInfo | None,
) -> CopilotPermissions | None:
    """Blacklist :data:`BUILDER_BLOCKED_TOOLS` for builder-bound sessions,
    return ``None`` (unrestricted) otherwise.

    Reads ``metadata.builder_graph_id`` only — works on either the bare
    ``ChatSessionInfo`` (no messages) or the full ``ChatSession``.
    """
    if session is None or not session.metadata.builder_graph_id:
        return None
    return CopilotPermissions(
        tools=list(BUILDER_BLOCKED_TOOLS),
        tools_exclude=True,
    )
