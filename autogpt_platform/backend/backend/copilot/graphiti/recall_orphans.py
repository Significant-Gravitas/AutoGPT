"""What a hard forget deletes after its edges: the episodes and entities that
nothing references any more.

This is ownership, not visibility. An episode goes only when no remaining
fact edge cites it, whatever that edge's status: a retracted edge kept for
audit still needs the episode it came from. An entity goes only when no fact
and no mention touches it. Hiding is the recall policy's job, and
``recall_forget.retract`` redacts before it deletes anything.
"""

from typing import Any

from .falkordb_driver import AutoGPTFalkorDriver
from .memory_model import ForgetResult


async def delete_orphans(
    driver: AutoGPTFalkorDriver, endpoints: set[str], result: ForgetResult
) -> None:
    """Delete what removing ``result.deleted`` orphaned, in dependency order.

    Episodes go first; they are found through their ``entity_edges``, which
    still name the deleted edges, and the query that deletes each one checks
    its references itself, so none added in between is lost. The surviving
    episodes then drop the deleted uuids, and last go the entities (endpoints
    of the deleted edges, or mentioned by a deleted episode) left with no fact
    and no mention.
    """
    rows = _rows(
        await driver.execute_query(
            _DELETE_ORPHANED_EPISODES_QUERY, uuids=result.deleted
        )
    )
    result.deleted_episodes = [row["uuid"] for row in rows]
    mentioned = {entity for row in rows for entity in row["mentioned"]}
    await driver.execute_query(_DROP_EDGE_BACKREFS_QUERY, uuids=result.deleted)
    rows = _rows(
        await driver.execute_query(
            _DELETE_ORPHANED_ENTITIES_QUERY, uuids=sorted(endpoints | mentioned)
        )
    )
    result.deleted_entities = [row["uuid"] for row in rows]


def _rows(
    result: tuple[list[dict[str, Any]], list[str], None] | None,
) -> list[dict[str, Any]]:
    return result[0] if result else []


# An episode is referenced by an edge its ``entity_edges`` names, or whose
# ``episodes`` names it (graphiti writes both). ``WITH`` captures what the
# caller needs before ``DETACH DELETE``, as FalkorDB cannot read a deleted
# node's properties.
_DELETE_ORPHANED_EPISODES_QUERY = """
MATCH (ep:Episodic)
WHERE any(x IN coalesce(ep.entity_edges, []) WHERE x IN $uuids)
OPTIONAL MATCH ()-[ref:RELATES_TO]->()
WHERE ref.uuid IN ep.entity_edges OR ep.uuid IN coalesce(ref.episodes, [])
WITH ep, count(ref) AS refs
WHERE refs = 0
OPTIONAL MATCH (ep)-[:MENTIONS]->(n:Entity)
WITH ep, ep.uuid AS uuid, collect(n.uuid) AS mentioned
DETACH DELETE ep
RETURN uuid, mentioned
"""

_DROP_EDGE_BACKREFS_QUERY = """
MATCH (ep:Episodic)
WHERE any(x IN ep.entity_edges WHERE x IN $uuids)
SET ep.entity_edges = [x IN ep.entity_edges WHERE NOT x IN $uuids]
"""

# A community membership (``HAS_MEMBER``) does not keep an entity: communities
# are rebuilt from the facts, and the entity's own summary may repeat the
# fact that was just deleted.
_DELETE_ORPHANED_ENTITIES_QUERY = """
MATCH (n:Entity)
WHERE n.uuid IN $uuids
OPTIONAL MATCH (n)-[r:RELATES_TO|MENTIONS]-()
WITH n, count(r) AS links
WHERE links = 0
WITH n, n.uuid AS uuid
DETACH DELETE n
RETURN uuid
"""
