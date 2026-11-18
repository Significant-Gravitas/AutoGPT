"""Check repeated-origin graphiti BFS reads against a live FalkorDB.

Walking each origin once preserves the complete facts. A truncating limit may
select different facts because Cypher does not guarantee order without
``ORDER BY``; each result must contain distinct records from the complete set.

Run with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/falkordb_origins_integration_test.py
"""

from typing import Any

import pytest
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import edge_bfs_search

from . import falkordb_driver
from .falkordb_driver import AutoGPTFalkorDriver
from .recall_integration_fixtures import ALICE, BOB, Fact, ingest_facts

_ALICE_LEADS: Fact = ("Alice", "Borealis", "Alice leads Borealis")
_ALICE_OWNS: Fact = ("Alice", "Cygnus", "Alice owns Cygnus")

_EDGE_BFS_WITH_AGGREGATE = """
    UNWIND $bfs_origin_node_uuids AS origin_uuid
    MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..3]->(:Entity)
    UNWIND relationships(path) AS rel
    WITH rel AS e, startNode(rel) AS n, endNode(rel) AS m
    WHERE type(e) = 'RELATES_TO'
    WITH e, n, m
    WITH e, n, m, count(*) AS origin_hits
    RETURN DISTINCT
        origin_hits AS origin_hits,
        e.uuid AS uuid,
        startNode(e).uuid AS source_node_uuid,
        endNode(e).uuid AS target_node_uuid,
        e.group_id AS group_id,
        e.created_at AS created_at,
        e.name AS name,
        e.fact AS fact,
        e.episodes AS episodes,
        e.expired_at AS expired_at,
        e.valid_at AS valid_at,
        e.invalid_at AS invalid_at,
        e.reference_time AS reference_time,
        properties(e) AS attributes
    LIMIT $limit
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_literal_distinct_text_preserves_repeats_before_limit(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    await ingest_facts(driver, scope, stub_graphiti_client, [ALICE])

    records, _, _ = await driver.execute_query(
        """
        UNWIND $bfs_origin_node_uuids AS origin
        RETURN origin AS value, "RETURN DISTINCT" AS marker
        LIMIT $limit
        """,
        bfs_origin_node_uuids=["hub", "hub", "other"],
        limit=2,
    )

    assert [record["value"] for record in records] == ["hub", "hub"]
    assert [record["marker"] for record in records] == [
        "RETURN DISTINCT",
        "RETURN DISTINCT",
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_changed_edge_bfs_aggregate_keeps_duplicate_origins(
    scope_graph, stub_graphiti_client
) -> None:
    driver, scope = scope_graph
    await ingest_facts(driver, scope, stub_graphiti_client, [ALICE])
    alice = (await _entities(driver))["Alice"]

    records, _, _ = await driver.execute_query(
        _EDGE_BFS_WITH_AGGREGATE,
        bfs_origin_node_uuids=[alice, alice],
        limit=10,
    )

    assert len(records) == 1
    assert records[0]["origin_hits"] == 2


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [10, 2])
async def test_walking_each_origin_once_preserves_facts_and_limit(
    scope_graph, stub_graphiti_client, mocker, limit: int
) -> None:
    driver, scope = scope_graph
    await ingest_facts(
        driver, scope, stub_graphiti_client, [ALICE, BOB, _ALICE_LEADS, _ALICE_OWNS]
    )
    entities = await _entities(driver)
    alice, bob = entities["Alice"], entities["Bob"]
    origins = [alice, bob, alice, alice, bob, alice]

    once = await _walk(driver, scope.group_id, origins, limit)
    mocker.patch.object(
        falkordb_driver, "distinct_origins", lambda cypher_query, params: params
    )
    every_repeat = await _walk(driver, scope.group_id, origins, limit)

    assert len(once) == len(every_repeat) == min(limit, 4)
    if limit >= 4:
        assert sorted(once, key=lambda edge: edge["uuid"]) == sorted(
            every_repeat, key=lambda edge: edge["uuid"]
        )
    else:
        full = await _walk(driver, scope.group_id, origins, 10)
        assert len(full) == 4
        for records in (once, every_repeat):
            assert len({edge["uuid"] for edge in records}) == limit
            assert all(edge in full for edge in records)


async def _walk(
    driver: AutoGPTFalkorDriver, group_id: str, origins: list[str], limit: int
) -> list[dict[str, Any]]:
    found = await edge_bfs_search(
        driver, origins, 3, SearchFilters(), [group_id], limit
    )
    return [edge.model_dump() for edge in found]


async def _entities(driver: AutoGPTFalkorDriver) -> dict[str, str]:
    records, _, _ = await driver.execute_query(
        "MATCH (n:Entity) RETURN n.name AS name, n.uuid AS uuid"
    )
    return {record["name"]: record["uuid"] for record in records}
