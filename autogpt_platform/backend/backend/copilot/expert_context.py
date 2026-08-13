"""Expert context injection for copilot sessions.

Two layers with different prompt weights:

- ``build_expert_identity_suffix()`` → ``<expert_identity>`` (the latest Soul,
  with precedence over the AutoPilot base identity). Appended to the SYSTEM
  prompt on every turn by both engines, so edits affect existing sessions
  while the cacheable base prefix stays byte-identical.
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

from backend.api.features.experts.models import (
    PROTECTED_SOUL_RULES,
    Expert,
    LearnedNote,
)
from backend.data.db_accessors import experts_db

logger = logging.getLogger(__name__)

# How many of the newest learned notes to surface in the identity suffix.
_MAX_LEARNED_NOTES = 20


def escape_prompt_xml_tags(value: str) -> str:
    return value.replace("<", "&lt;").replace(">", "&gt;")


def _render_learned_notes(notes: list[LearnedNote]) -> str:
    """Render the newest notes (up to ``_MAX_LEARNED_NOTES``), newest first.

    Falls back to the exact pre-notes placeholder when the expert has learned
    nothing yet, so an empty-notes session stays byte-identical to before.
    """
    recent = notes[-_MAX_LEARNED_NOTES:]
    if not recent:
        return "- Nothing recorded yet."
    return "\n".join(
        f"- {escape_prompt_xml_tags(note.fact)}" for note in reversed(recent)
    )


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

    name = escape_prompt_xml_tags(expert.name)
    identity = escape_prompt_xml_tags(expert.identity)
    voice = escape_prompt_xml_tags(expert.voice_preferences) or "Not specified."
    boundaries = escape_prompt_xml_tags(expert.boundaries) or "Not specified."
    learned = _render_learned_notes(expert.learned_notes)
    protected_rules = "\n".join(f"- {rule}" for rule in PROTECTED_SOUL_RULES)
    return (
        f"\n\n<expert_identity>\n"
        f"For this session you are {name} — {escape_prompt_xml_tags(expert.role)}, a hired "
        f"expert on the user's team.\n"
        f"<identity_and_personality>\n{identity}\n</identity_and_personality>\n"
        f"<voice_preferences>\n{voice}\n</voice_preferences>\n"
        f"<boundaries>\n{boundaries}\n</boundaries>\n"
        f"<what_ive_learned>\n{learned}\n</what_ive_learned>\n"
        f"<protected_rules>\n{protected_rules}\n</protected_rules>\n"
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
            f"- {escape_prompt_xml_tags(w.name or 'Unnamed workflow')} "
            f"(library_agent_id: {w.library_agent_id}, graph_id: {w.graph_id})"
            f": {escape_prompt_xml_tags(w.description or 'No description')}"
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
        escape_prompt_xml_tags(w.name or "Unnamed workflow") for w in expert.workflows
    )
    if not workflow_names:
        workflow_names = "none installed"
    return (
        f"- {escape_prompt_xml_tags(expert.name)} — "
        f"{escape_prompt_xml_tags(expert.role)} (expert id: {expert.id}); "
        f"installed workflows: {workflow_names}"
    )
