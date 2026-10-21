"""A graphiti BFS read whose origins repeat (``falkordb_origins.py``) against a
live FalkorDB: walking each origin once returns the same facts in the same
order as walking every repeat, under a limit that cuts the walk short too.

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


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [10, 2])
async def test_walking_each_origin_once_returns_the_same_facts_in_order(
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

    assert once == every_repeat
    assert len(once) == min(limit, 4)


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
