"""What a session may call, derived from the session itself.

A leaf module on purpose: it depends on the models and ``permissions`` only,
never on ``copilot.tools``. Anything that dispatches a turn on a session has
to resolve that session's own permissions, and tools are among those callers —
so keeping this out of ``builder_context`` (which imports three tool modules)
is what lets ``copilot/tools/message_session.py`` reuse it without closing an
import cycle.
"""

from backend.copilot.model import ONBOARDING_INTERVIEW_KIND, ChatSessionInfo
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

# The only tools an onboarding interview may call: the inverse of the
# builder's blacklist, because the interviewer's whole job is the
# conversation, and every declared tool is schema the model reads before its
# first token. ``ask_question`` puts tappable choices in front of someone
# giving one-word answers, and ``end_interview`` is Otto's end signal, which
# closes the interview on the first task the person agreed their team starts
# on. Both are in the eager core, so neither needs ``run_capability``.
# Two tools are left out on purpose:
# - ``add_understanding``: the profile is written once the interview is over,
#   from what the user said (``onboarding_dump.interview``), not from fields
#   the model composes mid-call, before the user has confirmed anything.
# - ``memory_store``: both engines already ingest every turn into memory.
# Both are deferred tools as well, so allowing either would also hand the
# model ``run_capability`` (``permissions._with_implied_grants``).
ONBOARDING_INTERVIEW_ALLOWED_TOOLS: tuple[str, ...] = ("ask_question", "end_interview")


def resolve_session_permissions(
    session: ChatSessionInfo | None,
) -> CopilotPermissions | None:
    """The tool filter a session's own kind or binding implies, or ``None``
    (unrestricted) for an ordinary chat.

    An onboarding interview gets the :data:`ONBOARDING_INTERVIEW_ALLOWED_TOOLS`
    whitelist; a builder-bound session gets the :data:`BUILDER_BLOCKED_TOOLS`
    blacklist. Reads ``metadata`` only — works on either the bare
    ``ChatSessionInfo`` (no messages) or the full ``ChatSession``.
    """
    if session is None:
        return None
    if session.metadata.kind == ONBOARDING_INTERVIEW_KIND:
        return CopilotPermissions(
            tools=list(ONBOARDING_INTERVIEW_ALLOWED_TOOLS),
            tools_exclude=False,
        )
    if not session.metadata.builder_graph_id:
        return None
    return CopilotPermissions(
        tools=list(BUILDER_BLOCKED_TOOLS),
        tools_exclude=True,
    )
