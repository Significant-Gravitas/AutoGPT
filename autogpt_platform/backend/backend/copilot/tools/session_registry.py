"""The capability index a session searches: the platform registry plus the
session owner's skills.

Skills belong to a user, and within a user to one owner folder — the
expert's, or personal Otto's — so they cannot sit in the process-wide
registry.  They are layered on per call from the same cached list that
builds ``<available_skills>``, so a search costs no extra storage reads.
"""

import logging
import time
from collections import OrderedDict

from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.models import CapabilityEntry
from backend.copilot.capabilities.registry import get_registry
from backend.copilot.capabilities.resolve import resolve_entry
from backend.copilot.capabilities.sources import skill_entries, skill_name
from backend.copilot.model import ChatSession

from .skills import SKILLS_INDEX_CACHE_TTL_S, is_skills_feature_enabled, list_all_skills

logger = logging.getLogger(__name__)

# Layering reuses the platform documents but BM25 re-derives its corpus
# statistics over all of them: some 40 ms of CPU on the event loop per
# search.  A skill list changes rarely and is itself cached for
# ``SKILLS_INDEX_CACHE_TTL_S``, so the layered index is kept for the same
# window, keyed by the platform index (a new one after a block reload) and
# by what the skills say, so a rewritten skill is re-indexed at once.
_LAYERED_MAX = 128
_layered: OrderedDict[
    tuple[int, int, tuple[tuple[str, str], ...]], tuple[float, CapabilityIndex]
] = OrderedDict()


async def session_registry(user_id: str, session: ChatSession) -> CapabilityIndex:
    """The platform index with this session's skills layered on."""
    return layered_index(get_registry(), await session_skill_entries(user_id, session))


def layered_index(
    base: CapabilityIndex, skills: list[CapabilityEntry]
) -> CapabilityIndex:
    """*base* with *skills* layered on, reused within the skill-cache window
    for the same platform index and the same skills."""
    if not skills:
        return base
    key = (id(base), len(base), tuple((e.id, e.description) for e in skills))
    now = time.monotonic()
    cached = _layered.get(key)
    if cached is not None and cached[0] > now:
        _layered.move_to_end(key)
        return cached[1]
    index = base.with_entries(skills)
    _layered[key] = (now + SKILLS_INDEX_CACHE_TTL_S, index)
    _layered.move_to_end(key)
    while len(_layered) > _LAYERED_MAX:
        _layered.popitem(last=False)
    return index


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
