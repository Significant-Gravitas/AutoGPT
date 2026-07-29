"""Expert context injection for the copilot first-turn message.

Builds the per-session expert/team context blocks that inject_user_context()
prepends to the first user message:

- Expert session → ``<expert_identity>`` (name, role, persona doc) +
  ``<expert_workflows>`` (installed workflows the model should prefer
  ``run_agent`` on).
- Plain session → ``<team_context>`` listing the user's hired experts so the
  model can suggest opening a matching expert's thread.

Everything here degrades silently to ``""`` — chat must never hard-fail on
expert lookup (archived/missing expert, DB error, no hired experts).

The returned string ends with a ``\\n\\n`` separator so callers can prepend it
directly in front of the message.
"""

import logging

from backend.api.features.experts import experts_db
from backend.api.features.experts.models import Expert

logger = logging.getLogger(__name__)


def _escape(value: str) -> str:
    """Escape angle brackets so user-supplied (expert name) or marketplace
    (workflow name/description) text cannot terminate the surrounding trusted
    block. Mirrors ``service._sanitize_user_context_field``; duplicated here
    because ``service`` imports this module.
    """
    return value.replace("<", "&lt;").replace(">", "&gt;")


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
    expert = await experts_db.get_expert(user_id, expert_id)
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

    name = _escape(expert.name)
    return (
        f"<expert_identity>\n"
        f"You are {name}, {expert.role}.\n"
        f"{expert.identity}\n"
        f"Stay in persona as {name} for the whole conversation.\n"
        f"</expert_identity>\n\n"
        f"<expert_workflows>\n"
        f"Workflows installed on this expert. For requests that match a "
        f"workflow's purpose, prefer running it with `run_agent` using the "
        f"IDs below over building something new:\n"
        f"{workflow_lines}\n"
        f"</expert_workflows>\n\n"
    )


async def _team_context(user_id: str) -> str:
    experts = await experts_db.list_experts(user_id)
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
