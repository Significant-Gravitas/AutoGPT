"""Walk each of a graphiti BFS read's origins once.

graphiti-core 0.30.2's edge search hands its second-stage BFS the source node
of every fact its first stage found, repeats included (``search.edge_search``),
and the BFS read walks out from every entry of ``$bfs_origin_node_uuids`` before
its ``RETURN DISTINCT`` drops the repeated rows. A memory's facts mostly share a
few source nodes, so those nodes are walked many times over, and the walk's
server time grows with the repeats until it passes FalkorDB's query timeout,
which fails the whole recall.

The origins are de-duplicated in first-occurrence order. Repeated origins cannot
add another distinct row, so the result set is unchanged. Current FalkorDB has
also returned the same order in live comparisons, but Cypher makes no row-order
guarantee without ``ORDER BY``. Only graphiti's pinned edge BFS gets this:
graphiti's entity BFS returns every row its walks find, repeats included, before
its ``LIMIT``, so fewer origins would change which rows it returns.
"""

from typing import Any

BFS_ORIGINS = "bfs_origin_node_uuids"

_EDGE_BFS_PREFIX = (
    "UNWIND $bfs_origin_node_uuids AS origin_uuid "
    "MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1.."
)
_EDGE_BFS_AFTER_DEPTH = (
    "]->(:Entity) "
    "UNWIND relationships(path) AS rel "
    "WITH rel AS e, startNode(rel) AS n, endNode(rel) AS m "
    "WHERE type(e) = 'RELATES_TO' "
    "WITH e, n, m"
)
_EDGE_BFS_FILTERS = (
    "",
    " WHERE e.group_id IN $group_ids",
    " WHERE ((e.expired_at IS NULL))",
    " WHERE ((e.expired_at IS NULL)) AND e.group_id IN $group_ids",
)
# Match the projection after without_embeddings rewrites graphiti's attributes.
_EDGE_BFS_RETURN = (
    " RETURN DISTINCT "
    "e.uuid AS uuid, "
    "startNode(e).uuid AS source_node_uuid, "
    "endNode(e).uuid AS target_node_uuid, "
    "e.group_id AS group_id, "
    "e.created_at AS created_at, "
    "e.name AS name, "
    "e.fact AS fact, "
    "e.episodes AS episodes, "
    "e.expired_at AS expired_at, "
    "e.valid_at AS valid_at, "
    "e.invalid_at AS invalid_at, "
    "e.reference_time AS reference_time, "
    "e {.*, fact_embedding: NULL} AS attributes "
    "LIMIT $limit"
)


def distinct_origins(cypher_query: str, params: dict[str, Any]) -> dict[str, Any]:
    """List each origin once only for graphiti's pinned distinct edge BFS."""
    origins = params.get(BFS_ORIGINS)
    if not origins or not _is_edge_bfs(cypher_query):
        return params
    return {**params, BFS_ORIGINS: list(dict.fromkeys(origins))}


def _is_edge_bfs(cypher_query: str) -> bool:
    query = " ".join(cypher_query.split())
    if not query.startswith(_EDGE_BFS_PREFIX) or query.count(f"${BFS_ORIGINS}") != 1:
        return False

    depth_end = query.find(_EDGE_BFS_AFTER_DEPTH, len(_EDGE_BFS_PREFIX))
    if depth_end < 0:
        return False
    depth = query[len(_EDGE_BFS_PREFIX) : depth_end]
    if not depth.isascii() or not depth.isdecimal() or depth.startswith("0"):
        return False

    prefix = f"{_EDGE_BFS_PREFIX}{depth}{_EDGE_BFS_AFTER_DEPTH}"
    return any(
        query == f"{prefix}{filter_clause}{_EDGE_BFS_RETURN}"
        for filter_clause in _EDGE_BFS_FILTERS
    )
