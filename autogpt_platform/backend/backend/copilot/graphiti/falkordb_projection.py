"""Leave the embeddings out of what graphiti's FalkorDB reads return.

graphiti-core 0.30.2 returns an edge's or an entity's ``attributes`` as every
property it has (``properties(e)`` / ``properties(n)``), so on FalkorDB each
fact or entity a read returns carries its 1,536-float embedding, which
``get_entity_edge_from_record`` / ``get_entity_node_from_record`` then pop
from the map unread. The embeddings are most of what such a read sends and
parses, float by float, on the event loop: a memory search's time grows with
the facts it returns, not with the facts it scans.

The rewrite returns the same map with the embedding set to null, which
graphiti pops as before. A read that needs an embedding names it in a column
of its own (``e.fact_embedding AS fact_embedding``), which the rewrite leaves
alone.
"""

_PROJECTIONS = {
    "properties(e) AS attributes": "e {.*, fact_embedding: NULL} AS attributes",
    "properties(n) AS attributes": "n {.*, name_embedding: NULL} AS attributes",
}


def without_embeddings(cypher_query: str) -> str:
    """``cypher_query`` with graphiti's all-properties projections replaced by
    ones that return every property but the embedding."""
    for projection, lean in _PROJECTIONS.items():
        cypher_query = cypher_query.replace(projection, lean)
    return cypher_query
