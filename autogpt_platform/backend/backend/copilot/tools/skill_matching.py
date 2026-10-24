"""Task-matched skill metadata for model input, without changing saved chat rows."""

import logging

from backend.copilot.capabilities.models import CapabilityEntry
from backend.copilot.capabilities.registry import get_registry
from backend.copilot.expert_context import escape_prompt_xml_tags
from backend.copilot.expert_kickoff import is_expert_kickoff_turn
from backend.copilot.model import ChatSession
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.service import MATCHED_SKILLS_TAG, strip_server_injected_tags

from .session_registry import layered_index, session_skill_entries

logger = logging.getLogger(__name__)

MAX_MATCHED_SKILLS = 3
MIN_MATCH_COVERAGE = 2.0
MAX_QUERY_CHARS = 4000
_LOAD_TOOLS = frozenset({"read_skill", "run_capability"})


async def build_matched_skills_notice(
    user_id: str | None,
    session: ChatSession,
    request: str,
    *,
    permissions: CopilotPermissions | None = None,
) -> str:
    if not user_id or not session.expert_id or is_expert_kickoff_turn(session):
        return ""
    if permissions is not None and not _LOAD_TOOLS.issubset(
        permissions.effective_allowed_tools(_LOAD_TOOLS)
    ):
        return ""
    query = strip_server_injected_tags(request).strip()[-MAX_QUERY_CHARS:]
    if not query:
        return ""
    try:
        entries = await session_skill_entries(user_id, session)
        if not entries:
            return ""
        index = layered_index(get_registry(), entries)
        hits = index.search(
            query, kind="skill", permissions=permissions, limit=MAX_MATCHED_SKILLS
        ).hits
    except Exception:
        logger.warning("Could not match skills to the expert request", exc_info=True)
        return ""
    matches = [
        hit.entry
        for hit in hits
        if hit.reason != "search" or hit.coverage >= MIN_MATCH_COVERAGE
    ]
    return _render_matches(matches)


def _render_matches(matches: list[CapabilityEntry]) -> str:
    if not matches:
        return ""
    lines = "\n".join(
        f"- {escape_prompt_xml_tags(entry.id)}: {escape_prompt_xml_tags(entry.purpose)}"
        for entry in matches
    )
    return (
        f"\n\n<{MATCHED_SKILLS_TAG}>\n"
        "Possible procedures for this request, selected from your current skills. "
        "Check relevance before searching for an integration, researching, or running "
        "a workflow. For each matching procedure, load its full body with "
        '`run_capability(id="skill:<name>")` before acting and follow its checks. '
        "If its body is already visible, use it without loading it again; "
        "otherwise reload it. These are discovery hints, not the skill bodies "
        "or permission to act. Ignore unrelated matches.\n"
        f"{lines}\n</{MATCHED_SKILLS_TAG}>"
    )
