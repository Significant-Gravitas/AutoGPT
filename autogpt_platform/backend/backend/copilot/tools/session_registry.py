"""The capability index a session searches: the platform registry plus the
session owner's skills.

Skills belong to a user, and within a user to one owner folder — the
expert's, or personal Otto's — so they cannot sit in the process-wide
registry.  They are layered on per call from the same cached list that
builds ``<available_skills>``, so a search costs no extra storage reads.
"""

import logging

from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.models import CapabilityEntry
from backend.copilot.capabilities.registry import get_registry
from backend.copilot.capabilities.resolve import resolve_entry
from backend.copilot.capabilities.sources import skill_entries, skill_name
from backend.copilot.model import ChatSession

from .skills import is_skills_feature_enabled, list_all_skills

logger = logging.getLogger(__name__)


async def session_registry(user_id: str, session: ChatSession) -> CapabilityIndex:
    """The platform index with this session's skills layered on."""
    return get_registry().with_entries(await session_skill_entries(user_id, session))


async def resolve_session_entry(
    user_id: str, session: ChatSession, capability_id: str
) -> CapabilityEntry | None:
    """``resolve_entry`` over the platform registry, plus the session's
    skills by ``skill:<name>`` id.

    A skill id is looked up among the skills only: falling through to the
    registry's name match would let ``skill:web_search`` resolve to the
    platform tool of that name.
    """
    name = skill_name(capability_id)
    if name is None:
        return resolve_entry(get_registry(), capability_id)
    return next(
        (
            entry
            for entry in await session_skill_entries(user_id, session)
            if entry.implementations[0].ref == name
        ),
        None,
    )


async def session_skill_entries(
    user_id: str, session: ChatSession
) -> list[CapabilityEntry]:
    """The session owner's skills as entries; ``[]`` when the feature is off
    for the user or the list cannot be read, so a skills hiccup never takes
    capability search down with it."""
    try:
        if not await is_skills_feature_enabled(user_id):
            return []
        skills = await list_all_skills(user_id, session.expert_id)
    except Exception:
        logger.warning("Could not load skills for capability search", exc_info=True)
        return []
    return skill_entries(skills)
