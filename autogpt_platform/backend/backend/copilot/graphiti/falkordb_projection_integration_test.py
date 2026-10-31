"""graphiti's FalkorDB reads without the embeddings (``falkordb_projection.py``)
against a live FalkorDB: every search and uuid read memory recall and
ingestion make returns the same facts and entities, attributes included, as
with graphiti's own all-properties projection, and the embeddings no longer
come back.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/falkordb_projection_integration_test.py
"""

from typing import Any

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

from . import falkordb_driver
from .falkordb_driver import AutoGPTFalkorDriver
from .recall_integration_fixtures import ALICE, BOB, CAROL, ingest_facts

_VECTOR = [0.1] * EMBEDDING_DIM


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reads_return_the_same_facts_and_entities_without_the_embedding(
    scope_graph, stub_graphiti_client, mocker
) -> None:
    driver, scope = scope_graph
    _, edges = await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE, BOB, CAROL]
    )
    entities = await _entity_uuids(driver)

    lean = await _read_everything(
        driver, scope.group_id, list(edges.values()), entities
    )
    mocker.patch.object(
        falkordb_driver, "without_embeddings", lambda query, _code: query
    )
    full = await _read_everything(
        driver, scope.group_id, list(edges.values()), entities
    )

    assert lean == full
    for read in ("edge_similarity", "edge_bfs", "edges_by_uuid", "nodes_by_uuid"):
        assert lean[read], f"{read} found nothing, so the comparison proves nothing"
    assert {edge["uuid"] for edge in lean["edges_by_uuid"]} == set(edges.values())


@pytest.mark.integration
@pytest.mark.asyncio
async def test_the_embeddings_no_longer_come_back(
    scope_graph, stub_graphiti_client, mocker
) -> None:
    driver, scope = scope_graph
    await ingest_facts(driver, scope, stub_graphiti_client, [ALICE, BOB])

    lean_edges, lean_nodes = await _stored_embeddings(driver)
    mocker.patch.object(
        falkordb_driver, "without_embeddings", lambda query, _code: query
    )
    full_edges, full_nodes = await _stored_embeddings(driver)

    assert lean_edges and set(lean_edges) == set(full_edges)
    assert all(embedding is None for embedding in lean_edges.values())
    assert all(len(embedding) == EMBEDDING_DIM for embedding in full_edges.values())
    assert lean_nodes and set(lean_nodes) == set(full_nodes)
    assert all(embedding is None for embedding in lean_nodes.values())
    assert all(len(embedding) == EMBEDDING_DIM for embedding in full_nodes.values())


async def _read_everything(
    driver: AutoGPTFalkorDriver,
    group_id: str,
    edge_uuids: list[str],
    entity_uuids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Each graphiti read through the driver, its results as dumps by uuid."""
    filters, groups = SearchFilters(), [group_id]
    reads = {
        "edge_fulltext": await edge_fulltext_search(
            driver, "Alice Atlas Carol Borealis", filters, groups, 10
        ),
        "edge_similarity": await edge_similarity_search(
            driver, _VECTOR, None, None, filters, groups, 10, 0.0
        ),
        "edge_bfs": await edge_bfs_search(driver, entity_uuids, 3, filters, groups, 10),
        "node_fulltext": await node_fulltext_search(
            driver, "Alice Atlas Carol Borealis", filters, groups, 10
        ),
        "node_similarity": await node_similarity_search(
            driver, _VECTOR, filters, groups, 10, 0.0
        ),
        "node_bfs": await node_bfs_search(driver, entity_uuids, filters, 3, groups, 10),
        "edges_by_uuid": await EntityEdge.get_by_uuids(driver, edge_uuids),
        "nodes_by_uuid": await EntityNode.get_by_uuids(driver, entity_uuids),
    }
    return {
        name: sorted((item.model_dump() for item in found), key=lambda d: d["uuid"])
        for name, found in reads.items()
    }


async def _entity_uuids(driver: AutoGPTFalkorDriver) -> list[str]:
    records, _, _ = await driver.execute_query("MATCH (n:Entity) RETURN n.uuid AS uuid")
    return sorted(record["uuid"] for record in records)


async def _stored_embeddings(
    driver: AutoGPTFalkorDriver,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Each fact's and entity's embedding as graphiti's projection returns it
    inside ``attributes``."""
    facts, _, _ = await driver.execute_query(
        "MATCH ()-[e:RELATES_TO]->() RETURN "
        + get_entity_edge_return_query(GraphProvider.FALKORDB)
    )
    entities, _, _ = await driver.execute_query(
        "MATCH (n:Entity) RETURN "
        + get_entity_node_return_query(GraphProvider.FALKORDB)
    )
    return (
        {row["uuid"]: row["attributes"].get("fact_embedding") for row in facts},
        {row["uuid"]: row["attributes"].get("name_embedding") for row in entities},
    )
