"""The capability index a session searches: the platform registry plus the
session owner's skills and the experts the session can hire or delegate to.

Skills belong to a user, and within a user to one owner folder — the
expert's, or personal Otto's — so they cannot sit in the process-wide
registry.  They are layered on per call from the same cached list that
builds ``<available_skills>``, so a search costs no extra storage reads.
Experts depend on the user too: the roster minus who they hired, plus the
team itself.
"""

import asyncio
import logging
import time
from collections import OrderedDict

from backend.api.features.experts.models import ExpertTemplate
from backend.copilot.capabilities.index import CapabilityIndex
from backend.copilot.capabilities.mcp_connections import (
    catalog_mcp_entry,
    connected_mcp_entries,
    stored_mcp_urls,
)
from backend.copilot.capabilities.models import CapabilityEntry
from backend.copilot.capabilities.ranking import normalize_server_url
from backend.copilot.capabilities.registry import get_registry
from backend.copilot.capabilities.resolve import load_connection_state, resolve_entry
from backend.copilot.capabilities.sources import (
    expert_dispatch,
    expert_entries,
    skill_entries,
    skill_name,
)
from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.util.cache import cached
from backend.util.feature_flag import Flag, is_feature_enabled

from .skills import SKILLS_INDEX_CACHE_TTL_S, is_skills_feature_enabled, list_all_skills

logger = logging.getLogger(__name__)

# The roster changes only when a deploy publishes the skills catalog.
ROSTER_CACHE_TTL_S = 300

# Layering reuses the platform documents but BM25 re-derives its corpus
# statistics over all of them: some 40 ms of CPU on the event loop per
# search.  A skill list changes rarely and is itself cached for
# ``SKILLS_INDEX_CACHE_TTL_S``, so the layered index is kept for the same
# window, keyed by the platform index (a new one after a block reload) and
# by complete entries, so rewritten skills and changed MCP endpoints are
# re-indexed at once.
_LAYERED_MAX = 128
_layered: OrderedDict[
    tuple[int, int, tuple[tuple[str, str], ...]],
    tuple[float, CapabilityIndex, CapabilityIndex],
] = OrderedDict()


async def session_registry(user_id: str, session: ChatSession) -> CapabilityIndex:
    """The platform index with this session's skills and experts layered on."""
    skills, experts = await asyncio.gather(
        session_skill_entries(user_id, session),
        session_expert_entries(user_id, session),
    )
    return layered_index(get_registry(), skills + experts)


def layered_index(
    base: CapabilityIndex, entries: list[CapabilityEntry]
) -> CapabilityIndex:
    """Layer session entries, caching identical metadata for the skill-cache window."""
    if not entries:
        return base
    key = (id(base), len(base), tuple((e.id, e.model_dump_json()) for e in entries))
    now = time.monotonic()
    cached = _layered.get(key)
    if cached is not None and cached[0] > now and cached[2] is base:
        _layered.move_to_end(key)
        return cached[1]
    index = base.with_entries(entries)
    # Retain the base so an evicted skills layer's address cannot be reused.
    _layered[key] = (now + SKILLS_INDEX_CACHE_TTL_S, index, base)
    _layered.move_to_end(key)
    while len(_layered) > _LAYERED_MAX:
        _layered.popitem(last=False)
    return index


async def resolve_session_entry(
    user_id: str, session: ChatSession, capability_id: str
) -> CapabilityEntry | None:
    """Resolve platform entries, connected MCP options, session skills and experts.

    A skill or expert id is looked up among those only: falling through to
    the registry's name match would let ``skill:web_search`` resolve to the
    platform tool of that name.
    """
    if expert_dispatch(capability_id) is not None:
        key = capability_id.strip().lower()
        return next(
            (
                entry
                for entry in await session_expert_entries(user_id, session)
                if entry.id.lower() == key
            ),
            None,
        )
    name = skill_name(capability_id)
    if name is None:
        return await _resolve_platform_entry(user_id, session, capability_id)
    return next(
        (
            entry
            for entry in await session_skill_entries(user_id, session)
            if entry.implementations[0].ref == name
        ),
        None,
    )


async def _resolve_platform_entry(
    user_id: str, session: ChatSession, capability_id: str
) -> CapabilityEntry | None:
    """Bind stored catalog options and preserve connected custom URL identity."""
    index = get_registry()
    key = capability_id.strip()
    if "://" in key:
        catalog = catalog_mcp_entry(index, key)
        if catalog is not None:
            return catalog
    entry = resolve_entry(index, key)
    if entry is None or entry.kind != "mcp_server":
        return entry
    if entry.connection.key and (
        "://" not in key
        or normalize_server_url(key) == normalize_server_url(entry.connection.key)
    ):
        return entry
    connections = await load_connection_state(user_id, session.expert_id)
    if entry.connection.key:
        return (
            None
            if normalize_server_url(key)
            in {normalize_server_url(url) for url in stored_mcp_urls(connections)}
            else entry
        )
    return next(
        (
            bound
            for bound in connected_mcp_entries(index, connections)
            if bound.id == entry.id
        ),
        entry,
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


async def session_expert_entries(
    user_id: str, session: ChatSession
) -> list[CapabilityEntry]:
    """Roster templates this session may hire and the teammates it may hand
    work to; ``[]`` when hire-experts is off or a read fails, so an experts
    hiccup never takes capability search down with it."""
    try:
        if not await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False):
            return []
        team = await experts_db().list_experts(user_id, with_metrics=False)
        templates = await _hireable_roster(user_id) if may_hire(session) else []
    except Exception:
        logger.warning("Could not load experts for capability search", exc_info=True)
        return []
    return expert_entries(
        templates, [expert for expert in team if expert.id != session.expert_id]
    )


def may_hire(session: ChatSession) -> bool:
    """Whether ``hire_expert`` runs in this session: the engines hide it from
    expert sessions (``expert_admin``) and from origins nobody is driving
    (``origin_disabled_tools``); ``expert_capabilities_test`` holds them equal."""
    return session.expert_id is None and session.metadata.origin == "interactive"


@cached(ttl_seconds=ROSTER_CACHE_TTL_S, maxsize=1024)
async def _hireable_roster(user_id: str) -> list[ExpertTemplate]:
    db = experts_db()
    return await db.with_bundled_skills(await db.list_templates(), user_id)
