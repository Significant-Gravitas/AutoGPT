import pytest

from .falkordb_origins import BFS_ORIGINS, distinct_origins

_DISTINCT_READ = """
    UNWIND $bfs_origin_node_uuids AS origin_uuid
    MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..3]->(:Entity)
    UNWIND relationships(path) AS rel
    WITH rel AS e, startNode(rel) AS n, endNode(rel) AS m
    WHERE type(e) = 'RELATES_TO'
    WITH e, n, m
    RETURN DISTINCT
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
        e {.*, fact_embedding: NULL} AS attributes
    LIMIT $limit
"""
_EVERY_ROW_READ = (
    "UNWIND $bfs_origin_node_uuids AS o MATCH (n {uuid: o}) RETURN n LIMIT 4"
)


def test_each_origin_is_kept_once_where_it_first_appears() -> None:
    marker = ["unchanged"]
    params = {
        BFS_ORIGINS: ["hub", "a", "hub", "b", "a", "hub"],
        "limit": 40,
        "marker": marker,
    }

    result = distinct_origins(_DISTINCT_READ, params)

    assert result == {
        BFS_ORIGINS: ["hub", "a", "b"],
        "limit": 40,
        "marker": marker,
    }
    assert result is not params
    assert result["marker"] is marker
    assert params[BFS_ORIGINS] == ["hub", "a", "hub", "b", "a", "hub"]


def test_a_read_that_returns_every_row_keeps_its_repeats() -> None:
    """Without ``DISTINCT`` the repeats are rows of their own, so fewer
    origins would change which rows survive the read's ``LIMIT``."""
    params = {BFS_ORIGINS: ["hub", "a", "hub"], "limit": 4}

    assert distinct_origins(_EVERY_ROW_READ, params) == params


def test_an_edge_bfs_subquery_with_later_origin_reuse_keeps_repeats() -> None:
    query = f"""
        CALL {{
            {_DISTINCT_READ}
        }}
        UNWIND $bfs_origin_node_uuids AS origin
        RETURN origin
    """
    params = {BFS_ORIGINS: ["hub", "hub", "other"], "limit": 2}

    assert distinct_origins(query, params) is params


@pytest.mark.parametrize(
    "query",
    [
        _DISTINCT_READ.replace(
            "WITH e, n, m\n    RETURN DISTINCT",
            "WITH e, n, m\n    WITH e, n, m, count(*) AS origin_hits\n"
            "    RETURN DISTINCT",
        ),
        _DISTINCT_READ.replace(
            "e.uuid AS uuid,", "count(*) AS origin_hits,\n        e.uuid AS uuid,"
        ),
        _DISTINCT_READ.replace("LIMIT $limit", "ORDER BY e.uuid LIMIT $limit"),
        _DISTINCT_READ.replace("'RELATES_TO'", "'MENTIONS'"),
        f"CALL {{{_DISTINCT_READ}}} RETURN 1",
        f"{_DISTINCT_READ} RETURN 1",
    ],
)
def test_changed_edge_bfs_shapes_keep_repeats(query: str) -> None:
    params = {BFS_ORIGINS: ["hub", "hub", "other"], "limit": 2}

    assert distinct_origins(query, params) is params


@pytest.mark.parametrize(
    "query",
    [
        "RETURN 'RETURN DISTINCT' AS marker",
        'RETURN "RETURN DISTINCT" AS marker',
        r"RETURN 'escaped \' RETURN DISTINCT' AS marker",
        r'RETURN "escaped \" RETURN DISTINCT" AS marker',
        "RETURN 1 AS `RETURN DISTINCT`",
        "RETURN 1 // RETURN DISTINCT",
        "RETURN 1 /* RETURN DISTINCT */",
        "UNWIND $bfs_origin_node_uuids AS origin RETURN DISTINCT origin",
        f'RETURN "{_DISTINCT_READ}" AS marker',
        f"/* {_DISTINCT_READ} */ RETURN 1",
    ],
)
def test_only_the_pinned_edge_bfs_de_duplicates_origins(query: str) -> None:
    params = {BFS_ORIGINS: ["hub", "hub", "other"], "limit": 2}

    assert distinct_origins(query, params) == params


@pytest.mark.parametrize(
    "params", [{}, {"limit": 10}, {BFS_ORIGINS: None}, {BFS_ORIGINS: []}]
)
def test_a_read_without_origins_is_left_alone(params: dict) -> None:
    assert distinct_origins(_DISTINCT_READ, params) == params
