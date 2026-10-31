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

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query

_RECORD_PROJECTIONS = (
    (
        get_entity_edge_return_query(GraphProvider.FALKORDB),
        "properties(e) AS attributes",
        "e {.*, fact_embedding: NULL} AS attributes",
    ),
    (
        get_entity_node_return_query(GraphProvider.FALKORDB),
        "properties(n) AS attributes",
        "n {.*, name_embedding: NULL} AS attributes",
    ),
)


def without_embeddings(cypher_query: str, cypher_code: str) -> str:
    """Replace only graphiti's complete record-builder projections in code."""
    if len(cypher_query) != len(cypher_code):
        return cypher_query

    replacements: list[tuple[int, int, str]] = []
    for record, projection, lean in _RECORD_PROJECTIONS:
        projection_offset = record.index(projection)
        search_from = 0
        while (record_start := cypher_code.find(record, search_from)) >= 0:
            projection_start = record_start + projection_offset
            replacements.append(
                (projection_start, projection_start + len(projection), lean)
            )
            search_from = record_start + len(record)

    for start, end, lean in sorted(
        replacements, key=lambda replacement: replacement[0], reverse=True
    ):
        cypher_query = cypher_query[:start] + lean + cypher_query[end:]
    return cypher_query
