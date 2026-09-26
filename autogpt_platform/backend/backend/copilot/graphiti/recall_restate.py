"""New live edges for facts an episode stated again after they were forgotten.

graphiti can resolve a restated fact into the forgotten edge between the
same entities: the statement then has no edge of its own. Its text is not in
graphiti's result, so it is read out of the episode again with graphiti's own
edge extraction (once per ingestion, whatever the number of such edges).
Every extracted statement between a pair of entities graphiti merged into a
forgotten edge becomes a new live edge, whichever forgotten edge took it,
unless a live fact between them already says it. Each live fact matches one
statement only, so nothing is created twice. A statement that graphiti merged
into a live fact worded differently, between the same pair in the same
episode, cannot be told apart and gets an edge of its own too.
"""

import logging
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EpisodicNode
from graphiti_core.utils.maintenance.edge_operations import extract_edges

from .recall import live_fact_predicate
from .types import EDGE_TYPE_MAP, EDGE_TYPES, MemoryFact

logger = logging.getLogger(__name__)

# A fact stated again starts live, like any fact graphiti extracts.
_LIVE_ATTRIBUTES = MemoryFact().model_dump(mode="json", exclude_none=True)

Pair = tuple[str, str]


async def restate(
    client: Graphiti,
    result: AddEpisodeResults,
    merged: set[Pair],
    previous: list[str],
    instructions: str | None,
) -> list[EntityEdge]:
    """A new live edge for each statement graphiti merged into a forgotten
    edge; ``merged`` holds the ``(source, target)`` pairs it merged into."""
    episode = result.episode
    context = (
        await EpisodicNode.get_by_uuids(client.driver, previous) if previous else []
    )
    extracted = await extract_edges(
        client.clients,
        episode,
        result.nodes,
        context,
        EDGE_TYPE_MAP,
        episode.group_id,
        EDGE_TYPES,
        instructions,
    )
    live = await _live_statements(client.driver, sorted(merged))
    stated = unmatched(extracted, merged, live)
    if not stated:
        logger.warning(f"Episode {episode.uuid}: no restated fact was re-extracted")
    return [await _live_edge(client, episode, statement) for statement in stated]


def unmatched(
    extracted: list[EntityEdge], merged: set[Pair], live: Counter[Any]
) -> list[EntityEdge]:
    """The extracted statements to give a new edge: between a pair graphiti
    merged into a forgotten edge, and not already a live fact there. Each
    live fact is matched once (``live`` is consumed)."""
    chosen: list[EntityEdge] = []
    for statement in extracted:
        pair = (statement.source_node_uuid, statement.target_node_uuid)
        key = (*pair, normalized(statement.fact))
        if pair not in merged:
            continue
        if live[key] > 0:
            live[key] -= 1
            continue
        chosen.append(statement)
    return chosen


def normalized(text: str | None) -> str:
    """How two statements are compared: case and spacing aside."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


async def _live_statements(driver: GraphDriver, pairs: list[Pair]) -> Counter[Any]:
    result = await driver.execute_query(
        _LIVE_STATEMENTS_QUERY, pairs=[list(pair) for pair in pairs]
    )
    rows = result[0] if result else []
    return Counter(
        (row["source"], row["target"], normalized(row["fact"])) for row in rows
    )


async def _live_edge(
    client: Graphiti, episode: EpisodicNode, stated: EntityEdge
) -> EntityEdge:
    edge = EntityEdge(
        group_id=episode.group_id,
        source_node_uuid=stated.source_node_uuid,
        target_node_uuid=stated.target_node_uuid,
        created_at=datetime.now(timezone.utc),
        name=stated.name,
        fact=stated.fact,
        episodes=[episode.uuid],
        valid_at=episode.valid_at,
        attributes=dict(_LIVE_ATTRIBUTES),
    )
    await edge.generate_embedding(client.embedder)
    await edge.save(client.driver)
    return edge


_LIVE_STATEMENTS_QUERY = f"""
UNWIND $pairs AS pair
MATCH (source:Entity {{uuid: pair[0]}})-[e:RELATES_TO]->(target:Entity {{uuid: pair[1]}})
WHERE {live_fact_predicate("e")}
RETURN source.uuid AS source, target.uuid AS target, e.fact AS fact
"""
