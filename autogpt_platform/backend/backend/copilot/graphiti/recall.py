"""The recall policy: the one place memory is read back for the assistant.

Warm context, ``memory_search``, ``memory_forget_search``, the settings fact
list, the dream gather and ingestion's extraction context read through this
module's functions or predicates, and every forget goes through ``retract``
(``recall_forget.py``), so a fact one path forgot cannot come back through
another. ``recall_render.py`` writes out what recall returns.

A fact (a ``RELATES_TO`` edge) is live while ``expired_at`` is unset and its
``status`` is ``active`` or ``tentative``; an edge with no ``status`` predates
the ``MemoryFact`` edge type and counts as ``active``. A forgotten fact
(``forgotten_fact_predicate``) is never live. An episode is recallable while
no forget has stamped ``redacted_at`` on it and none of the facts extracted
from it is forgotten: one forgotten fact hides the whole text, since nothing
records which sentence it came from. The episode's other facts stay live.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodeType, EpisodicNode, get_episodic_node_from_record
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT

from backend.copilot.dream.ratification_hits import record_memory_hit

from .client import get_graphiti_client
from .falkordb_driver import open_driver
from .memory_model import MemoryStatus
from .scope import MemoryScope

logger = logging.getLogger(__name__)

# The ``expiration_reason`` a user's forget records (``recall_forget.retract``).
USER_FORGET_REASON = "user_signal"

_LIVE_STATUSES = (MemoryStatus.active, MemoryStatus.tentative)

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


def forgotten_fact_predicate(alias: str = "e") -> str:
    """The forgotten-fact test as a Cypher ``WHERE`` fragment on edge ``alias``.

    A user forgot the fact: it is ``retracted``, its ``expiration_reason``
    is ``USER_FORGET_REASON`` (a dream demotion made on the user's word
    records that too), or it has the ``legacy_forget_predicate`` shape.
    """
    return (
        f"({alias}.status = '{MemoryStatus.retracted.value}'"
        f" OR {alias}.expiration_reason = '{USER_FORGET_REASON}'"
        f" OR ({legacy_forget_predicate(alias)}))"
    )


def legacy_forget_predicate(alias: str = "e") -> str:
    """What a forget left before this policy: ``expired_at`` and nothing else.

    graphiti sets ``invalid_at`` whenever it expires an edge and a dream
    demotion always records a reason, so no other writer leaves this shape.
    ``migrations/backfill_legacy_forgets.py`` restamps these edges as
    retractions, after which this clause can go.
    """
    return (
        f"{alias}.expired_at IS NOT NULL AND {alias}.invalid_at IS NULL"
        f" AND {alias}.expiration_reason IS NULL"
    )


def is_forgotten(fact: EntityEdge) -> bool:
    """The forgotten-fact test for an edge graphiti returned."""
    reason = fact.attributes.get("expiration_reason")
    legacy = fact.expired_at is not None and fact.invalid_at is None
    return (
        fact_status(fact) == MemoryStatus.retracted.value
        or reason == USER_FORGET_REASON
        or (legacy and reason is None)
    )


def forgotten_facts_clause(var: str = "forgotten") -> str:
    """Cypher that opens a query by binding ``var`` to every forgotten fact's
    uuid, for ``recallable_episode_predicate`` to test episodes against.

    One pass over the facts, not one per episode.
    """
    return (
        "OPTIONAL MATCH ()-[forgotten_fact:RELATES_TO]->()\n"
        f"WHERE {forgotten_fact_predicate('forgotten_fact')}\n"
        f"WITH collect(forgotten_fact.uuid) AS {var}\n"
    )


def recallable_episode_predicate(alias: str = "e", forgotten: str = "forgotten") -> str:
    """The recallable-episode test as a Cypher ``WHERE`` fragment on episode
    ``alias``; the query must open with ``forgotten_facts_clause(forgotten)``."""
    return (
        f"{alias}.redacted_at IS NULL"
        f" AND none(x IN coalesce({alias}.entity_edges, []) WHERE x IN {forgotten})"
    )


def is_recallable_episode(
    entity_edges: list[str], forgotten: set[str], *, redacted: bool
) -> bool:
    """The recallable-episode test in Python: not ``redacted`` and citing no
    uuid in ``forgotten`` among its ``entity_edges``."""
    return not redacted and forgotten.isdisjoint(entity_edges)


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
    """The ``n`` newest recallable episodes, oldest first.

    graphiti's ``retrieve_episodes`` plus the recallable-episode test it has
    no way to express.
    """
    driver = open_driver(scope)
    try:
        records = await _recallable_episodes(
            driver, scope.group_id, datetime.now(timezone.utc), n
        )
    finally:
        await driver.close()
    return [get_episodic_node_from_record(record) for record in reversed(records)]


async def previous_episode_uuids(
    driver: GraphDriver,
    group_id: str,
    reference_time: datetime,
    source: EpisodeType,
) -> list[str]:
    """The earlier episodes ``add_episode`` may show its extraction prompts.

    graphiti's own pick (``retrieve_episodes``: the ``RELEVANT_SCHEMA_LIMIT``
    newest of the same source up to ``reference_time``) cannot see a forget,
    so ingestion passes this one: the same pick of recallable episodes,
    oldest first. Never raises: on a failed read extraction gets no earlier
    episodes, not graphiti's unfiltered pick, and the write still happens.
    """
    try:
        records = await _recallable_episodes(
            driver, group_id, reference_time, RELEVANT_SCHEMA_LIMIT, source.value
        )
    except Exception:
        logger.warning(
            f"Prior-episode read failed for group {group_id[:12]}; "
            "extracting without earlier episodes",
            exc_info=True,
        )
        return []
    return [str(record["uuid"]) for record in reversed(records)]


async def _recallable_episodes(
    driver: GraphDriver,
    group_id: str,
    reference_time: datetime,
    limit: int,
    source: str | None = None,
) -> list[dict[str, Any]]:
    result = await driver.execute_query(
        _RECALLABLE_EPISODES_QUERY,
        group_id=group_id,
        reference_time=reference_time,
        source=source,
        limit=limit,
    )
    return result[0] if result else []


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


def _statuses(include_tentative: bool) -> tuple[MemoryStatus, ...]:
    return _LIVE_STATUSES if include_tentative else (MemoryStatus.active,)


# graphiti's ``retrieve_episodes`` query (group, ``reference_time`` cut-off,
# optional source, newest first) with the recallable-episode test added.
# ``valid_at`` is stored as an ISO string, so the comparison is lexical, as
# in graphiti.
_RECALLABLE_EPISODES_QUERY = (
    forgotten_facts_clause()
    + f"""
MATCH (e:Episodic)
WHERE e.group_id = $group_id
  AND e.valid_at <= $reference_time
  AND ($source IS NULL OR e.source = $source)
  AND {recallable_episode_predicate("e")}
RETURN e.uuid AS uuid, e.name AS name, e.group_id AS group_id,
       e.created_at AS created_at, e.source AS source,
       e.source_description AS source_description, e.content AS content,
       e.valid_at AS valid_at, e.entity_edges AS entity_edges
ORDER BY e.valid_at DESC
LIMIT $limit
"""
)
