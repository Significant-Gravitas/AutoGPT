from unittest.mock import AsyncMock, MagicMock

import pytest
from graphiti_core.embedder.client import EMBEDDING_DIM
from graphiti_core.search.search import edge_search
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import edge_bfs_search

from .falkordb_driver import AutoGPTFalkorDriver
from .falkordb_origins import BFS_ORIGINS, distinct_origins

_GROUP = ["user_test"]
_EDGE_COLUMNS = (
    "uuid",
    "source_node_uuid",
    "target_node_uuid",
    "group_id",
    "created_at",
    "name",
    "fact",
    "episodes",
    "expired_at",
    "valid_at",
    "invalid_at",
    "reference_time",
    "attributes",
)


class _Result:
    """A falkordb query result."""

    def __init__(self, header: list, result_set: list) -> None:
        self.header = header
        self.result_set = result_set


def test_each_origin_is_kept_once_where_it_first_appears() -> None:
    params = {BFS_ORIGINS: ["hub", "a", "hub", "b", "a", "hub"], "limit": 40}

    assert distinct_origins(params) == {BFS_ORIGINS: ["hub", "a", "b"], "limit": 40}
    assert params[BFS_ORIGINS] == ["hub", "a", "hub", "b", "a", "hub"]


@pytest.mark.parametrize(
    "params", [{}, {"limit": 10}, {BFS_ORIGINS: None}, {BFS_ORIGINS: []}]
)
def test_a_read_without_origins_is_left_alone(params: dict) -> None:
    assert distinct_origins(params) == params


@pytest.mark.asyncio
async def test_a_bfs_read_reaches_falkordb_with_each_origin_once() -> None:
    driver, ro_query = _driver(lambda query: _Result([], []))

    await edge_bfs_search(
        driver, ["hub", "a", "hub", "hub"], 3, SearchFilters(), _GROUP, 40
    )

    params = ro_query.await_args.args[1]
    assert params[BFS_ORIGINS] == ["hub", "a"]
    assert params["limit"] == 40


@pytest.mark.asyncio
async def test_graphitis_search_walks_a_hub_once() -> None:
    """graphiti's edge search hands its second-stage BFS the source node of
    every fact its first stage found, repeats included; the BFS that reaches
    FalkorDB walks each once, in the order graphiti gave them."""
    facts = [("hub", "f1"), ("hub", "f2"), ("other", "f3"), ("hub", "f4")]
    driver, ro_query = _driver(
        lambda query: (
            _similarity(facts) if "vec.cosineDistance" in query else _Result([], [])
        )
    )
    config = EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.cosine_similarity, EdgeSearchMethod.bfs],
        reranker=EdgeReranker.rrf,
    )

    found, _ = await edge_search(
        driver,
        MagicMock(),
        "query",
        [0.1] * EMBEDDING_DIM,
        _GROUP,
        config,
        SearchFilters(),
    )

    assert {edge.uuid for edge in found} == {"f1", "f2", "f3", "f4"}
    bfs = [
        call
        for call in ro_query.await_args_list
        if "RELATES_TO|MENTIONS" in call.args[0]
    ]
    assert [call.args[1][BFS_ORIGINS] for call in bfs] == [["hub", "other"]]


def _driver(answer) -> tuple[AutoGPTFalkorDriver, AsyncMock]:
    driver = AutoGPTFalkorDriver(falkor_db=MagicMock())
    ro_query = AsyncMock(side_effect=lambda query, params: answer(query))
    driver.client.select_graph.return_value.ro_query = ro_query
    return driver, ro_query


def _similarity(facts: list[tuple[str, str]]) -> _Result:
    """A similarity read's rows: one fact per (source node, fact uuid)."""
    rows = [
        [
            uuid,
            source,
            f"target-{uuid}",
            _GROUP[0],
            "2026-10-21T00:00:00+00:00",
            "RELATES_TO",
            f"fact {uuid}",
            ["episode"],
            None,
            None,
            None,
            None,
            {},
        ]
        for source, uuid in facts
    ]
    return _Result([[1, column] for column in _EDGE_COLUMNS], rows)
