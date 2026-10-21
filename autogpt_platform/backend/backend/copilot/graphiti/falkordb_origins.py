"""Walk each of a graphiti BFS read's origins once.

graphiti-core 0.30.2's edge search hands its second-stage BFS the source node
of every fact its first stage found, repeats included (``search.edge_search``),
and the BFS read walks out from every entry of ``$bfs_origin_node_uuids`` before
its ``RETURN DISTINCT`` drops the repeated rows. A memory's facts mostly share a
few source nodes (the user's or the expert's own), so a few nodes are walked
many times over, and the walk's server time grows with the repeats until it
passes FalkorDB's query timeout, which fails the whole recall.

The origins are de-duplicated in their order: every origin is still walked
from, in the same order, so ``RETURN DISTINCT`` keeps the same first rows and
the read returns the same rows in the same order.
"""

from typing import Any

BFS_ORIGINS = "bfs_origin_node_uuids"


def distinct_origins(params: dict[str, Any]) -> dict[str, Any]:
    """``params`` with its BFS origins, if any, each listed once."""
    origins = params.get(BFS_ORIGINS)
    if not origins:
        return params
    return {**params, BFS_ORIGINS: list(dict.fromkeys(origins))}
