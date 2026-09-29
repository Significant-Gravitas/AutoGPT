"""The recall policy: what memory may be read back, for the assistant, the
dream and ingestion.

Warm context, ``memory_search``, ``memory_forget_search``, the settings fact
list, the dream gather and ingestion's extraction context read through this
module's functions or predicates, and what the assistant is shown is read
again by uuid right before it is rendered (``live_now``,
``recall_recheck.py``). Every forget goes through ``retract``
(``recall_forget.py``); ``graphiti/AGENTS.md`` lists what a forget reaches
and what it does not. ``recall_render.py`` writes out what recall returns.

A fact (a ``RELATES_TO`` edge) is live while ``expired_at`` and
``forgotten_at`` are unset and its ``status`` is ``active`` or ``tentative``;
an edge with no ``status`` predates the ``MemoryFact`` edge type and counts
as ``active``. A forgotten fact (``forgotten_fact_predicate``) is never live:
a forget stamps ``forgotten_at``, which only a forget writes
(``recall_forget.py`` and the cascade it runs on what the dream derived
from the fact, or a backfill for older forgets), so no
other writer's status or reason can make it look remembered again. An
episode is recallable while no forget has stamped ``redacted_at`` on it and
none of the facts extracted from it is forgotten: one forgotten fact hides
the whole text, since nothing records which sentence it came from. The
episode's other facts stay live. Nor is an episode a dream write placed
and is still writing (``write_pending``, ``marked_write.py``): graphiti's
save of it clears that.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode, get_episodic_node_from_record
from graphiti_core.search.search_config import SearchConfig
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import (
    ComparisonOperator,
    DateFilter,
    SearchFilters,
)

from backend.copilot.dream.ratification_hits import record_memory_hit

from .client import get_graphiti_client
from .memory_model import MemoryStatus
from .scope import MemoryScope

# The ``expiration_reason`` a user's forget records (``recall_forget.retract``).
USER_FORGET_REASON = "user_signal"
# What a forgotten fact's ``fact`` and ``name`` read once a forget has moved
# them to ``fact_redacted`` / ``name_redacted`` (kept for audit), so graphiti's
# own prompts never see the sentence (``recall_hide.py``).
FORGOTTEN_FACT = "[forgotten]"

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
        f"{alias}.expired_at IS NULL AND {alias}.forgotten_at IS NULL"
        f" AND ({alias}.status IS NULL OR {alias}.status IN [{statuses}])"
    )


def is_live(fact: EntityEdge, *, include_tentative: bool = True) -> bool:
    """The live-fact test for an edge graphiti returned."""
    status = fact_status(fact)
    allowed = {s.value for s in _statuses(include_tentative)}
    unmarked = fact.expired_at is None and fact.attributes.get("forgotten_at") is None
    return unmarked and (status is None or status in allowed)


def fact_status(fact: EntityEdge) -> str | None:
    """The edge's ``MemoryFact.status``; ``None`` on edges that predate it."""
    status = fact.attributes.get("status")
    return None if status is None else str(status)


def forgotten_fact_predicate(alias: str = "e") -> str:
    """The forgotten-fact test as a Cypher ``WHERE`` fragment on edge ``alias``.

    A user forgot the fact: a forget stamped ``forgotten_at`` (the marker
    only a forget writes), or, from before that marker, it is ``retracted``,
    its ``expiration_reason`` is ``USER_FORGET_REASON`` (a dream demotion made
    on the user's word records that too), or it has the
    ``legacy_forget_predicate`` shape.

    It is never null: a ``status`` or ``expiration_reason`` left unset reads
    as matching neither value, so ``NOT`` it is the not-forgotten test on
    every edge, a live one included (a bare ``e.status = 'retracted'`` is
    null there, and so would the negation be).
    """
    return (
        f"({alias}.forgotten_at IS NOT NULL"
        f" OR coalesce({alias}.status, '') = '{MemoryStatus.retracted.value}'"
        f" OR coalesce({alias}.expiration_reason, '') = '{USER_FORGET_REASON}'"
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
        fact.attributes.get("forgotten_at") is not None
        or fact_status(fact) == MemoryStatus.retracted.value
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
        f" AND {alias}.write_pending IS NULL"
        f" AND none(x IN coalesce({alias}.entity_edges, []) WHERE x IN {forgotten})"
    )


def is_recallable_episode(
    entity_edges: list[str],
    forgotten: set[str],
    *,
    redacted: bool,
    write_pending: bool = False,
) -> bool:
    """The recallable-episode test in Python: not ``redacted``, not still
    being written, and citing no uuid in ``forgotten`` among its
    ``entity_edges``."""
    return not (redacted or write_pending) and forgotten.isdisjoint(entity_edges)


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
    recipe object, which graphiti's own ingestion dedup also reads. The search
    can take seconds (a cross-encoder rerank), so its last step reads the
    facts again by uuid (``live_now``): a forget that answered meanwhile is
    not returned.
    """
    client = await get_graphiti_client(scope.group_id)
    config = (recipe or EDGE_HYBRID_SEARCH_RRF).model_copy(update={"limit": limit})
    results = await client.search_(
        query=query,
        config=config,
        group_ids=[scope.group_id],
        search_filter=_LIVE_SEARCH_FILTER,
    )
    tentative = include_tentative
    found = [f for f in results.edges if is_live(f, include_tentative=tentative)]
    return await live_now(client.driver, found, include_tentative=tentative)


async def live_now(
    driver: GraphDriver, facts: list[EntityEdge], *, include_tentative: bool = True
) -> list[EntityEdge]:
    """``facts`` still live, in order, read again by uuid in one query. As
    the last graph read before they are shown, it bounds the stale window to
    the time between this read and the response."""
    if not facts:
        return facts
    live = live_fact_predicate("e", include_tentative=include_tentative)
    result = await driver.execute_query(
        f"MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $uuids AND {live}"
        " RETURN e.uuid AS uuid",
        uuids=[fact.uuid for fact in facts],
    )
    kept = {row["uuid"] for row in (result[0] if result else [])}
    return [fact for fact in facts if fact.uuid in kept]


async def recent_episodes(scope: MemoryScope, n: int) -> list[EpisodicNode]:
    """The ``n`` newest recallable episodes, oldest first.

    graphiti's ``retrieve_episodes`` plus the recallable-episode test it has
    no way to express. Read on the driver of the scope's cached client, as
    the fact search and the recheck are: warm context calls this on every
    qualifying chat turn, and a driver per read would build and connect a
    FalkorDB client each time (``falkordb_driver.open_driver``).
    """
    client = await get_graphiti_client(scope.group_id)
    records = await recallable_episodes(
        client.driver, scope.group_id, datetime.now(timezone.utc), n
    )
    return [get_episodic_node_from_record(record) for record in reversed(records)]


async def recallable_episodes(
    driver: GraphDriver,
    group_id: str,
    reference_time: datetime,
    limit: int,
    source: str | None = None,
) -> list[dict[str, Any]]:
    """The ``limit`` newest recallable episodes up to ``reference_time``
    (of ``source`` when given), newest first, as records."""
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
