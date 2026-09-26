"""What ingestion does so graphiti's ``add_episode`` cannot undo a forget.

graphiti resolves each new fact against every edge between the same
entities, forgotten ones included, and offers any edge as a contradiction
candidate. Resolving into a forgotten edge appends the new episode to it
(and, on its LLM path, rewrites its attributes, dropping the forget's audit
fields); a contradiction stamps its ``invalid_at``. Either way the new
episode then cites a forgotten edge and is hidden with it, and a fact the
user states again never becomes live.

So ingestion snapshots every forgotten edge before ``add_episode`` and hands
the snapshot back here afterwards: each forgotten edge graphiti changed is
put back exactly, the new episode stops citing it, and a fact stated again
gets a new live edge, built from what graphiti's own edge extraction reads
out of the episode. Neither step ever fails the write: a failed read or
repair is logged, and the episode stays as graphiti wrote it.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EpisodicNode
from graphiti_core.utils.maintenance.edge_operations import extract_edges
from pydantic import BaseModel

from .recall import forgotten_fact_predicate
from .types import EDGE_TYPE_MAP, EDGE_TYPES, MemoryFact

logger = logging.getLogger(__name__)

# Every field a forget, the audit views or recall read off a forgotten edge.
_FIELDS = (
    *MemoryFact.model_fields,
    "forgotten_at",
    "fact",
    "fact_redacted",
    "name",
    "name_redacted",
    "expired_at",
    "invalid_at",
    "valid_at",
    "episodes",
)
# A fact stated again starts live, like any fact graphiti extracts.
_LIVE_ATTRIBUTES = MemoryFact().model_dump(mode="json", exclude_none=True)


class ForgottenEdge(BaseModel):
    uuid: str
    source: str
    target: str
    fields: dict[str, Any]


async def snapshot_forgotten(driver: GraphDriver) -> dict[str, ForgottenEdge]:
    """Every forgotten fact as it stands before an ``add_episode``."""
    try:
        rows = _rows(await driver.execute_query(_SNAPSHOT_QUERY))
    except Exception:
        logger.warning("Forgotten-fact snapshot failed; not guarded", exc_info=True)
        return {}
    return {row["uuid"]: _edge(row) for row in rows}


async def keep_forgotten(
    client: Graphiti,
    before: dict[str, ForgottenEdge],
    result: AddEpisodeResults,
    previous: list[str],
    instructions: str | None,
) -> None:
    """Put back each forgotten edge ``add_episode`` changed, and give each
    fact it resolved into one a new live edge the new episode cites."""
    if not before:
        return
    try:
        absorbed = await _put_back(client.driver, before, result.episode.uuid)
        new_edges = (
            await _restate(client, absorbed, result, previous, instructions)
            if absorbed
            else []
        )
        await _repoint(client.driver, result.episode, list(before), new_edges)
    except Exception:
        logger.warning(
            "Repairing forgotten facts after ingestion failed", exc_info=True
        )
        return
    result.edges = [e for e in result.edges if e.uuid not in before] + new_edges


async def _put_back(
    driver: GraphDriver, before: dict[str, ForgottenEdge], episode_uuid: str
) -> list[ForgottenEdge]:
    """Restore every forgotten edge ``add_episode`` changed; returns those it
    had merged the new episode into."""
    rows = _rows(await driver.execute_query(_READ_QUERY, uuids=list(before)))
    after = {row["uuid"]: _edge(row) for row in rows}
    changed = [edge for edge in before.values() if after.get(edge.uuid) != edge]
    for edge in changed:
        await driver.execute_query(_RESTORE_QUERY, uuid=edge.uuid, fields=edge.fields)
    return [
        edge
        for edge in changed
        if edge.uuid in after
        and episode_uuid in (after[edge.uuid].fields.get("episodes") or [])
    ]


async def _repoint(
    driver: GraphDriver,
    episode: EpisodicNode,
    forgotten: list[str],
    new_edges: list[EntityEdge],
) -> None:
    """Point the new episode at its new live edges and away from any
    forgotten one, which would otherwise hide it."""
    if not new_edges and not set(forgotten) & set(episode.entity_edges):
        return
    await driver.execute_query(
        _REPOINT_QUERY,
        uuid=episode.uuid,
        forgotten=forgotten,
        added=[edge.uuid for edge in new_edges],
    )


async def _restate(
    client: Graphiti,
    absorbed: list[ForgottenEdge],
    result: AddEpisodeResults,
    previous: list[str],
    instructions: str | None,
) -> list[EntityEdge]:
    """A new live edge for each forgotten fact the episode stated again, its
    sentence from graphiti's own extraction of the episode."""
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
    new_edges: list[EntityEdge] = []
    for edge in absorbed:
        stated = next(
            (
                x
                for x in extracted
                if (x.source_node_uuid, x.target_node_uuid)
                == (edge.source, edge.target)
            ),
            None,
        )
        if stated is None:
            logger.warning(f"No sentence re-extracted for forgotten fact {edge.uuid}")
            continue
        new_edges.append(await _live_edge(client, episode, stated))
    return new_edges


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


def _edge(row: dict[str, Any]) -> ForgottenEdge:
    return ForgottenEdge(
        uuid=row["uuid"],
        source=row["source"],
        target=row["target"],
        fields={field: row[field] for field in _FIELDS},
    )


def _rows(
    result: tuple[list[dict[str, Any]], list[str], None] | None,
) -> list[dict[str, Any]]:
    return result[0] if result else []


_RETURN = "e.uuid AS uuid, source.uuid AS source, target.uuid AS target, " + ", ".join(
    f"e.{field} AS {field}" for field in _FIELDS
)

_SNAPSHOT_QUERY = f"""
MATCH (source)-[e:RELATES_TO]->(target)
WHERE {forgotten_fact_predicate("e")}
RETURN {_RETURN}
"""

# By uuid, not by the forgotten test: graphiti's attribute rewrite can strip
# the markers that test reads.
_READ_QUERY = f"""
MATCH (source)-[e:RELATES_TO]->(target)
WHERE e.uuid IN $uuids
RETURN {_RETURN}
"""

# ``+=`` sets every snapshot field back and leaves the embedding alone; a
# null in the snapshot removes a property graphiti added.
_RESTORE_QUERY = """
MATCH ()-[e:RELATES_TO {uuid: $uuid}]->()
SET e += $fields
"""

_REPOINT_QUERY = """
MATCH (ep:Episodic {uuid: $uuid})
SET ep.entity_edges =
    [x IN coalesce(ep.entity_edges, []) WHERE NOT x IN $forgotten] + $added
"""
