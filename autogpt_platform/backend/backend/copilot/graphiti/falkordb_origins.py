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

_EDGE_BFS_MARKERS = (
    "UNWIND $bfs_origin_node_uuids AS origin_uuid",
    "MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..",
    "]->(:Entity)",
    "UNWIND relationships(path) AS rel",
    "WITH rel AS e, startNode(rel) AS n, endNode(rel) AS m",
    "WHERE type(e) =",
    "WITH e, n, m",
    "RETURN DISTINCT",
    "e.uuid AS uuid",
    "startNode(e).uuid AS source_node_uuid",
    "endNode(e).uuid AS target_node_uuid",
    "e.reference_time AS reference_time",
    "AS attributes",
    "LIMIT $limit",
)


def distinct_origins(cypher_code: str, params: dict[str, Any]) -> dict[str, Any]:
    """List each origin once only for graphiti's pinned distinct edge BFS."""
    origins = params.get(BFS_ORIGINS)
    if not origins or not _is_edge_bfs(cypher_code):
        return params
    return {**params, BFS_ORIGINS: list(dict.fromkeys(origins))}


def _is_edge_bfs(cypher_code: str) -> bool:
    search_from = 0
    for marker in _EDGE_BFS_MARKERS:
        marker_start = cypher_code.find(marker, search_from)
        if marker_start < 0:
            return False
        search_from = marker_start + len(marker)
    return True
