"""Fail-closed lifecycle filtering for shared Graphiti retrieval."""

from .config import graphiti_config

_ACTIVE_EDGE_QUERY = """
UNWIND $edge_ids AS edge_id
MATCH ()-[e:RELATES_TO {uuid: edge_id}]->()
WHERE e.group_id = $group_id
  AND e.status = 'active'
  AND e.expired_at IS NULL
RETURN e.uuid AS uuid
"""

_SCOPE_EDGE_QUERY = """
UNWIND $edge_ids AS edge_id
MATCH ()-[e:RELATES_TO {uuid: edge_id}]->()
WHERE e.group_id = $group_id
  AND e.scope = $scope
RETURN e.uuid AS uuid
"""


def active_shared_search_filter():
    from graphiti_core.search.search_filters import (
        ComparisonOperator,
        DateFilter,
        SearchFilters,
    )

    return SearchFilters(
        expired_at=[
            [
                DateFilter(
                    date=None,
                    comparison_operator=ComparisonOperator.is_null,
                )
            ]
        ]
    )


def _open_driver(group_id: str):
    from .falkordb_driver import AutoGPTFalkorDriver

    return AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
        build_indices=False,
    )


def _uuid(item) -> str | None:
    value = getattr(item, "uuid", None)
    return str(value) if value else None


async def filter_active_shared_edges(
    group_id: str, edges: list, *, driver=None
) -> list:
    """Return only ratified, non-retracted shared facts."""
    edge_ids = [uuid for edge in edges if (uuid := _uuid(edge))]
    if not edge_ids:
        return []
    active_edge_ids: set[str] = set()

    owns_driver = driver is None
    if driver is None:
        driver = _open_driver(group_id)
    try:
        result = await driver.execute_query(
            _ACTIVE_EDGE_QUERY, group_id=group_id, edge_ids=edge_ids
        )
        if result is not None:
            active_edge_ids = {str(row["uuid"]) for row in result[0]}
    finally:
        if owns_driver:
            await driver.close()

    return [edge for edge in edges if _uuid(edge) in active_edge_ids]


async def filter_edges_by_scope(
    group_id: str,
    edges: list,
    scope: str,
    *,
    driver=None,
) -> list:
    edge_ids = [uuid for edge in edges if (uuid := _uuid(edge))]
    if not edge_ids:
        return []
    matching_ids: set[str] = set()

    owns_driver = driver is None
    if driver is None:
        driver = _open_driver(group_id)
    try:
        result = await driver.execute_query(
            _SCOPE_EDGE_QUERY,
            group_id=group_id,
            edge_ids=edge_ids,
            scope=scope,
        )
        if result is not None:
            matching_ids = {str(row["uuid"]) for row in result[0]}
    finally:
        if owns_driver:
            await driver.close()

    return [edge for edge in edges if _uuid(edge) in matching_ids]
