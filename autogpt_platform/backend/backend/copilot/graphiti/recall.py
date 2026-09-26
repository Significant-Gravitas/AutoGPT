"""The recall policy: the one place memory is read back for the assistant.

Warm context, ``memory_search``, ``memory_forget_search`` and the settings
fact list read facts and episodes through this module, and every forget goes
through ``retract`` (``recall_forget.py``), so a fact one path forgot cannot
come back through another.

A fact (a ``RELATES_TO`` edge) is live while ``expired_at`` is unset and its
``status`` is ``active`` or ``tentative``; an edge with no ``status`` predates
the ``MemoryFact`` edge type and counts as ``active``. An episode stays
recallable until a forget stamps ``redacted_at`` on it, which happens once
none of the facts extracted from it is live, so its raw text cannot bring a
forgotten fact back.
"""

import asyncio
from datetime import datetime, timezone

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode, get_episodic_node_from_record
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)
from pydantic import BaseModel, ValidationError

from backend.copilot.dream.ratification_hits import record_memory_hit

from .client import get_graphiti_client
from .falkordb_driver import open_driver
from .memory_model import MemoryStatus
from .scope import MemoryScope

# Scope of an episode that is not a ``MemoryEnvelope`` (plain conversation).
GLOBAL_SCOPE = "real:global"
# Episode bodies are cut to this many characters when rendered.
EPISODE_DISPLAY_CHARS = 500

_LIVE_STATUSES = (MemoryStatus.active, MemoryStatus.tentative)
_RETIRED_STATUSES = frozenset(
    status.value
    for status in (
        MemoryStatus.superseded,
        MemoryStatus.contradicted,
        MemoryStatus.retracted,
    )
)

# The ``expired_at`` half of the live test, applied inside every graphiti
# search method so a retired fact never takes one of the ``limit`` slots. The
# ``status`` half runs in Python (``is_live``): graphiti 0.30 builds no Cypher
# from ``SearchFilters.property_filters``.
_LIVE_SEARCH_FILTER = SearchFilters(
    expired_at=[[DateFilter(comparison_operator=ComparisonOperator.is_null)]]
)


def live_fact_predicate(alias: str = "e", *, include_tentative: bool = True) -> str:
    """The live-fact test as a Cypher ``WHERE`` fragment on edge ``alias``.

    For code that lists or counts facts with its own Cypher, so its idea of a
    live fact stays the one recall uses.
    """
    statuses = ", ".join(f"'{s.value}'" for s in _statuses(include_tentative))
    return (
        f"{alias}.expired_at IS NULL"
        f" AND ({alias}.status IS NULL OR {alias}.status IN [{statuses}])"
    )


def is_live(fact: EntityEdge, *, include_tentative: bool = True) -> bool:
    """The live-fact test for an edge graphiti returned."""
    status = fact_status(fact)
    allowed = {s.value for s in _statuses(include_tentative)}
    return fact.expired_at is None and (status is None or status in allowed)


def fact_status(fact: EntityEdge) -> str | None:
    """The edge's ``MemoryFact.status``; ``None`` on edges that predate it."""
    status = fact.attributes.get("status")
    return None if status is None else str(status)


async def search_facts(
    scope: MemoryScope,
    query: str,
    *,
    limit: int,
    recipe: SearchConfig | None = None,
    include_tentative: bool = True,
) -> list[EntityEdge]:
    """Live facts matching ``query``, best first, at most ``limit``.

    ``recipe`` defaults to the hybrid RRF config ``Graphiti.search`` uses. Its
    ``limit`` is replaced on a copy: ``Graphiti.search`` sets it on the shared
    recipe object, which graphiti's own ingestion dedup also reads.
    """
    client = await get_graphiti_client(scope.group_id)
    config = (recipe or EDGE_HYBRID_SEARCH_RRF).model_copy(update={"limit": limit})
    results = await client.search_(
        query=query,
        config=config,
        group_ids=[scope.group_id],
        search_filter=_LIVE_SEARCH_FILTER,
    )
    return [
        fact
        for fact in results.edges
        if is_live(fact, include_tentative=include_tentative)
    ]


async def recent_episodes(scope: MemoryScope, n: int) -> list[EpisodicNode]:
    """The ``n`` newest unredacted episodes, oldest first.

    graphiti's ``retrieve_episodes`` plus the ``redacted_at`` filter it has
    no way to express.
    """
    driver = open_driver(scope)
    try:
        result = await driver.execute_query(
            _RECENT_EPISODES_QUERY,
            group_id=scope.group_id,
            reference_time=datetime.now(timezone.utc),
            limit=n,
        )
    finally:
        await driver.close()
    records = result[0] if result else []
    return [get_episodic_node_from_record(record) for record in reversed(records)]


def render(fact: EntityEdge) -> str:
    """One recall line for ``fact``: its text and when it holds.

    A retired fact (expired, or in a status recall skips) is labelled with
    how and when it was retired, never as valid until "present". Recall never
    returns one; the label keeps any other caller honest.
    """
    text = fact_text(fact)
    if is_live(fact):
        valid_from, valid_to = fact_validity(fact)
        return f"{text} (valid: {valid_from} — {valid_to})"
    retired_at = str(fact.expired_at) if fact.expired_at else "at an unknown time"
    return f"{text} ({_retired_label(fact)} {retired_at})"


def fact_text(fact: EntityEdge) -> str:
    """The fact sentence, or the relation name when extraction left none."""
    return fact.fact or fact.name


def fact_validity(fact: EntityEdge) -> tuple[str, str]:
    """``(valid_from, valid_to)`` in valid time; "present" means no end yet.

    Only meaningful for a live fact; ``render`` labels a retired one instead.
    """
    valid_from = str(fact.valid_at) if fact.valid_at else "unknown"
    valid_to = str(fact.invalid_at) if fact.invalid_at else "present"
    return valid_from, valid_to


def render_episode(episode: EpisodicNode) -> str:
    """``[created_at] body``, the body cut to ``EPISODE_DISPLAY_CHARS``."""
    return f"[{episode.created_at}] {episode.content[:EPISODE_DISPLAY_CHARS]}"


def episode_scope(episode: EpisodicNode) -> str:
    """The ``MemoryEnvelope`` scope an episode was stored under.

    An episode that is not an envelope (plain conversation, or JSON that is
    not an object) belongs to ``GLOBAL_SCOPE``. Reads the full body: an
    envelope cut to display length is no longer valid JSON.
    """
    try:
        return _EnvelopeScope.model_validate_json(episode.content).scope
    except ValidationError:
        return GLOBAL_SCOPE


async def record_hit(scope: MemoryScope, edge_uuids: list[str]) -> None:
    """Count one recall hit on each edge, for the ratification sweep.

    Best-effort like ``record_memory_hit``, which never raises: a lost hit
    only delays the promotion of a tentative fact.
    """
    await asyncio.gather(
        *(
            record_memory_hit(scope, edge_uuid)
            for edge_uuid in dict.fromkeys(edge_uuids)
        )
    )


class _EnvelopeScope(BaseModel):
    scope: str = GLOBAL_SCOPE


def _statuses(include_tentative: bool) -> tuple[MemoryStatus, ...]:
    return _LIVE_STATUSES if include_tentative else (MemoryStatus.active,)


def _retired_label(fact: EntityEdge) -> str:
    status = fact_status(fact)
    return status if status in _RETIRED_STATUSES else "expired"


# graphiti's ``retrieve_episodes`` query with the redaction filter added.
# ``valid_at`` is stored as an ISO string, so the comparison is lexical, as
# in graphiti.
_RECENT_EPISODES_QUERY = """
MATCH (e:Episodic)
WHERE e.group_id = $group_id
  AND e.valid_at <= $reference_time
  AND e.redacted_at IS NULL
RETURN e.uuid AS uuid, e.name AS name, e.group_id AS group_id,
       e.created_at AS created_at, e.source AS source,
       e.source_description AS source_description, e.content AS content,
       e.valid_at AS valid_at, e.entity_edges AS entity_edges
ORDER BY e.valid_at DESC
LIMIT $limit
"""
