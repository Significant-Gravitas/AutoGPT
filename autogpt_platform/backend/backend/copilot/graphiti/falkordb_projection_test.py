from unittest.mock import AsyncMock, MagicMock

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    node_bfs_search,
    node_fulltext_search,
    node_similarity_search,
)

from .falkordb_driver import AutoGPTFalkorDriver
from .falkordb_projection import without_embeddings

_LEAN_EDGE = "e {.*, fact_embedding: NULL} AS attributes"
_LEAN_NODE = "n {.*, name_embedding: NULL} AS attributes"
_GROUP = ["user_test"]
_VECTOR = [0.1] * EMBEDDING_DIM


class _EmptyResult:
    """A falkordb query result with no rows."""

    def __init__(self) -> None:
        self.header: list = []
        self.result_set: list = []


def test_graphiti_edge_projection_loses_the_embedding() -> None:
    projection = get_entity_edge_return_query(GraphProvider.FALKORDB)

    assert "properties(e) AS attributes" in projection, (
        "graphiti's edge projection changed: check falkordb_projection still"
        " leaves the embedding out"
    )
    assert _LEAN_EDGE in without_embeddings(projection)
    assert "properties(e)" not in without_embeddings(projection)


def test_graphiti_node_projection_loses_the_embedding() -> None:
    projection = get_entity_node_return_query(GraphProvider.FALKORDB)

    assert "properties(n) AS attributes" in projection, (
        "graphiti's node projection changed: check falkordb_projection still"
        " leaves the embedding out"
    )
    assert _LEAN_NODE in without_embeddings(projection)
    assert "properties(n)" not in without_embeddings(projection)


@pytest.mark.parametrize(
    "query",
    [
        "MATCH ()-[e:RELATES_TO]->() RETURN e.fact_embedding AS fact_embedding",
        "MATCH (n:Entity) RETURN n.name_embedding AS name_embedding",
        "MATCH ()-[e]->() RETURN properties(e) AS properties",
        "MATCH (n) RETURN labels(n) AS labels, properties(n) AS properties",
    ],
)
def test_a_read_that_names_its_columns_is_left_alone(query: str) -> None:
    assert without_embeddings(query) == query


@pytest.mark.asyncio
async def test_graphitis_searches_reach_falkordb_without_the_embedding() -> None:
    """Every graphiti search and uuid read memory recall and ingestion make
    goes to FalkorDB without the all-properties projection."""
    driver = AutoGPTFalkorDriver(falkor_db=MagicMock())
    ro_query = AsyncMock(return_value=_EmptyResult())
    driver.client.select_graph.return_value.ro_query = ro_query
    filters = SearchFilters()

    await edge_fulltext_search(driver, "trigger schedule", filters, _GROUP, 10)
    await edge_similarity_search(driver, _VECTOR, None, None, filters, _GROUP, 10)
    await edge_bfs_search(driver, ["origin"], 3, filters, _GROUP, 10)
    await node_fulltext_search(driver, "trigger schedule", filters, _GROUP, 10)
    await node_similarity_search(driver, _VECTOR, filters, _GROUP, 10)
    await node_bfs_search(driver, ["origin"], filters, 3, _GROUP, 10)
    await EntityEdge.get_by_uuids(driver, ["fact"])
    await EntityNode.get_by_uuids(driver, ["entity"])

    sent = [call.args[0] for call in ro_query.await_args_list]
    assert len(sent) == 8
    assert not [query for query in sent if "properties(e)" in query]
    assert not [query for query in sent if "properties(n)" in query]
    assert sum(_LEAN_EDGE in query for query in sent) == 4
    assert sum(_LEAN_NODE in query for query in sent) == 4
