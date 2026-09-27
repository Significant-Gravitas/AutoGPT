"""A hard forget's last step: empty and delete what the forgotten edges leave.

It runs after ``recall_forget.retract`` has retracted and hidden the facts
(``recall_hide.py``), so every step here works on facts recall already
hides, and every step is safe to repeat: the edges are deleted last, in one
query each, so forgetting again after any failure finds them and finishes.

This is ownership, not visibility. An episode that no remaining fact edge
cites, whatever that edge's status (a retracted edge kept for audit still
needs its source), is not deleted but emptied into a tombstone: ``content``
cleared, ``hard_deleted_at`` set, its name, source description and envelope
provenance kept, so the dream still knows which chat session to leave out
(``dream/hidden_sessions.py``). Its mentions go, and so does every entity
left with no fact and no mention.
"""

import logging
from typing import Any

from graphiti_core.driver.driver import GraphDriver

from .memory_model import ForgetResult, MemoryForgetFailure, envelope_provenance

logger = logging.getLogger(__name__)


async def purge(
    driver: GraphDriver,
    group_id: str,
    uuids: list[str],
    now: str,
    result: ForgetResult,
) -> None:
    """Tombstone the episodes only ``uuids`` cite, then delete each edge and
    the entities it alone kept; a failed step is each edge's
    ``cleanup_error``."""
    try:
        result.tombstoned_episodes = await _tombstone(driver, uuids, now)
    except Exception as exc:
        _report(group_id, uuids, exc, result)
        return
    for edge_uuid in uuids:
        await _delete_edge(driver, group_id, edge_uuid, result)
    tombstoned = set(result.tombstoned_episodes)
    result.redacted_episodes = [
        episode for episode in result.redacted_episodes if episode not in tombstoned
    ]


async def _tombstone(driver: GraphDriver, uuids: list[str], now: str) -> list[str]:
    """Empty every episode nothing outside ``uuids`` cites, keeping the
    envelope provenance its text carried (parsed here, not in Cypher)."""
    citing = _rows(await driver.execute_query(_CITING_EPISODES_QUERY, uuids=uuids))
    provenance = [
        [row["uuid"], found]
        for row in citing
        if (found := envelope_provenance(row["content"])) is not None
    ]
    rows = _rows(
        await driver.execute_query(
            _TOMBSTONE_QUERY, uuids=uuids, provenance=provenance, now=now
        )
    )
    return [row["uuid"] for row in rows]


async def _delete_edge(
    driver: GraphDriver,
    group_id: str,
    edge_uuid: str,
    result: ForgetResult,
) -> None:
    try:
        rows = _rows(
            await driver.execute_query(
                _DELETE_EDGE_QUERY, uuid=edge_uuid, group_id=group_id
            )
        )
    except Exception as exc:
        _report(group_id, [edge_uuid], exc, result)
        return
    if not rows:
        result.failures.append(MemoryForgetFailure.no_match(edge_uuid))
        return
    result.deleted.append(edge_uuid)
    result.deleted_entities.extend(rows[0]["deleted_entities"])


def _report(
    group_id: str, uuids: list[str], exc: Exception, result: ForgetResult
) -> None:
    logger.warning(
        f"Edges retracted and hidden but hard deletion failed in graph "
        f"{group_id[:20]}",
        exc_info=True,
    )
    result.failures.extend(
        MemoryForgetFailure.cleanup_error(edge_uuid, exc) for edge_uuid in uuids
    )


def _rows(result: Any) -> list[dict[str, Any]]:
    return result[0] if result else []


_CITING_EPISODES_QUERY = """
MATCH (ep:Episodic)
WHERE any(x IN coalesce(ep.entity_edges, []) WHERE x IN $uuids)
RETURN ep.uuid AS uuid, ep.content AS content
"""

# An episode is cited by an edge its ``entity_edges`` names or whose
# ``episodes`` names it (graphiti writes both). ``coalesce`` keeps a first
# tombstone's stamps and provenance on a repeat. The tombstone keeps
# ``entity_edges`` until its edges are deleted, so a retry still finds it.
_TOMBSTONE_QUERY = """
MATCH (ep:Episodic)
WHERE any(x IN coalesce(ep.entity_edges, []) WHERE x IN $uuids)
OPTIONAL MATCH ()-[ref:RELATES_TO]->()
WHERE NOT ref.uuid IN $uuids
  AND (ref.uuid IN ep.entity_edges OR ep.uuid IN coalesce(ref.episodes, []))
WITH ep, count(ref) AS refs
WHERE refs = 0
SET ep.provenance = coalesce(
        ep.provenance, [p IN $provenance WHERE p[0] = ep.uuid | p[1]][0]
    ),
    ep.content = '',
    ep.redacted_at = coalesce(ep.redacted_at, $now),
    ep.hard_deleted_at = coalesce(ep.hard_deleted_at, $now)
RETURN ep.uuid AS uuid
"""

# One edge, atomically: delete it, drop it from every episode's
# ``entity_edges``, drop every tombstone's mentions, then delete each entity
# among its endpoints and the entities those mentions named that is left
# with no fact and no mention. Ids are captured before any ``DELETE``:
# FalkorDB cannot read a deleted element (FalkorDB #1393). A community
# membership (``HAS_MEMBER``) does not keep an entity: communities are
# rebuilt from the facts.
_DELETE_EDGE_QUERY = """
MATCH (source)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(target)
WHERE e.group_id = $group_id OR e.group_id IS NULL
WITH e, e.uuid AS uuid, source.uuid AS source_uuid, target.uuid AS target_uuid
DELETE e
WITH uuid, source_uuid, target_uuid
OPTIONAL MATCH (ep:Episodic)
WHERE uuid IN coalesce(ep.entity_edges, [])
WITH uuid, source_uuid, target_uuid, collect(ep) AS citing
FOREACH (ep IN citing |
    SET ep.entity_edges = [x IN ep.entity_edges WHERE x <> uuid])
WITH uuid, source_uuid, target_uuid
OPTIONAL MATCH (tomb:Episodic)-[m:MENTIONS]->(mentioned:Entity)
WHERE tomb.hard_deleted_at IS NOT NULL
WITH uuid, source_uuid, target_uuid, collect(m) AS mentions,
     collect(DISTINCT mentioned.uuid) AS mentioned_uuids
FOREACH (m IN mentions | DELETE m)
WITH uuid, [source_uuid, target_uuid] + mentioned_uuids AS candidates
OPTIONAL MATCH (candidate:Entity)
WHERE candidate.uuid IN candidates
OPTIONAL MATCH (candidate)-[link:RELATES_TO|MENTIONS]-()
WITH uuid, candidate, count(link) AS links
WITH uuid,
     collect(CASE WHEN links = 0 THEN candidate END) AS orphans,
     collect(CASE WHEN links = 0 THEN candidate.uuid END) AS orphan_uuids
FOREACH (orphan IN orphans | DETACH DELETE orphan)
RETURN uuid, orphan_uuids AS deleted_entities
"""
