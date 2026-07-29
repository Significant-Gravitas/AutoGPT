"""Expert context injection for copilot sessions.

Two layers with different prompt weights:

- ``build_expert_identity_suffix()`` → ``<expert_identity>`` (name, role,
  persona doc, precedence over the AutoPilot base identity).  Appended to the
  per-session SYSTEM prompt by both engines, exactly like the building-mode
  guide suffix: session-stable, so the cacheable base prefix stays
  byte-identical and cross-user cache hits are preserved for plain sessions
  (empty suffix).  Identity must live here — injected user-message context
  loses to the system prompt on direct identity questions.
- ``build_expert_context()`` → first-user-message context blocks:
  ``<expert_workflows>`` (expert session: installed workflows the model
  should prefer ``run_agent`` on) or ``<team_context>`` (plain session:
  hired experts the model can suggest, never silently delegate to).

Everything here degrades silently to ``""`` — chat must never hard-fail on
expert lookup (archived/missing expert, DB error, no hired experts).

Returned strings carry their own separators so callers can concatenate
directly (suffix: leading ``\\n\\n``; message blocks: trailing ``\\n\\n``).
"""

import logging

from backend.api.features.experts.models import Expert
from backend.data.db_accessors import experts_db

logger = logging.getLogger(__name__)


def _escape(value: str) -> str:
    """Escape angle brackets so user-supplied (expert name) or marketplace
    (workflow name/description) text cannot terminate the surrounding trusted
    block. Mirrors ``service._sanitize_user_context_field``; duplicated here
    because ``service`` imports this module.
    """
    return value.replace("<", "&lt;").replace(">", "&gt;")


async def build_expert_identity_suffix(
    user_id: str | None, expert_id: str | None
) -> str:
    """Build the ``<expert_identity>`` system-prompt suffix for an expert
    session.

    Returns ``""`` for plain sessions (keeps the system prompt byte-identical
    for cross-user caching) and on any lookup failure.

    Runs on every turn, so it skips the workflow joins — only the expert's
    own name/role/identity columns are read here.
    """
    if not user_id or not expert_id:
        return ""
    try:
        expert = await experts_db().get_expert(
            user_id, expert_id, include_workflows=False
        )
    except Exception as e:
        logger.warning(f"Failed to build expert identity suffix: {e}")
        return ""
    if expert is None or expert.is_archived:
        return ""

    name = _escape(expert.name)
    return (
        f"\n\n<expert_identity>\n"
        f"For this session you are {name} — {expert.role}, a hired "
        f"expert on the user's team.\n"
        f"{expert.identity}\n"
        f"The base instructions above describe AutoPilot, the platform "
        f"engine you run on. All platform capabilities and tools remain "
        f"available to you, but you always speak and act as {name}: "
        f"never present yourself as AutoPilot, and if asked who you are, "
        f"you are {name}.\n"
        f"</expert_identity>"
    )


async def build_expert_context(user_id: str | None, expert_id: str | None) -> str:
    """Build the expert/team context prefix for the first user message.

    Returns ``""`` when there is nothing to inject or any lookup fails.
    """
    if not user_id:
        return ""
    try:
        if expert_id:
            return await _expert_session_context(user_id, expert_id)
        return await _team_context(user_id)
    except Exception as e:
        logger.warning(f"Failed to build expert context: {e}")
        return ""


async def _expert_session_context(user_id: str, expert_id: str) -> str:
    expert = await experts_db().get_expert(user_id, expert_id)
    # Archived/missing expert at stream time → omit the block and let the
    # turn proceed as a plain Autopilot session.
    if expert is None or expert.is_archived:
        return ""

    if expert.workflows:
        workflow_lines = "\n".join(
            f"- {_escape(w.name or 'Unnamed workflow')} "
            f"(library_agent_id: {w.library_agent_id}, graph_id: {w.graph_id})"
            f": {_escape(w.description or 'No description')}"
            for w in expert.workflows
        )
    else:
        workflow_lines = "- No workflows installed yet."

    return (
        f"<expert_workflows>\n"
        f"Workflows installed on this expert. For requests that match a "
        f"workflow's purpose, prefer running it with `run_agent` using the "
        f"IDs below over building something new:\n"
        f"{workflow_lines}\n"
        f"</expert_workflows>\n\n"
    )


async def _team_context(user_id: str) -> str:
    experts = await experts_db().list_experts(user_id)
    if not experts:
        return ""

    lines = "\n".join(_team_line(e) for e in experts)
    return (
        f"<team_context>\n"
        f"The user has hired these experts:\n"
        f"{lines}\n"
        f"When a request clearly matches an expert's domain, suggest opening "
        f"that expert's thread (by expert id) instead of handling it here; "
        f"never silently delegate to an expert.\n"
        f"</team_context>\n\n"
    )


def _team_line(expert: Expert) -> str:
    workflow_names = ", ".join(
        _escape(w.name or "Unnamed workflow") for w in expert.workflows
    )
    if not workflow_names:
        workflow_names = "none installed"
    return (
        f"- {_escape(expert.name)} — {expert.role} (expert id: {expert.id}); "
        f"installed workflows: {workflow_names}"
    )
