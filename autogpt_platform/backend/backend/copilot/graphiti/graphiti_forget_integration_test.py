"""Integration tests for the P-1.3 demotion helpers, against live FalkorDB.

The unit-test sibling (``backend/copilot/tools/graphiti_forget_test.py``)
pins the Cypher strings and call signatures via mock drivers; those run
fast but don't catch Cypher that's syntactically valid yet semantically
wrong on FalkorDB (different graph engines have slightly different
behavior around relationship variable scoping, ``MATCH`` semantics with
property-only patterns, etc.).

This file is the regression net that catches those. For every P-1.3
behavior, seed a known graph, run the helper, query the resulting
state via raw Cypher, and assert.

The single most important test in the file is
``test_invalidate_entity_direct_neighbors_is_single_hop`` — it pins the
boundary that distinguishes our scoped cascade from the
runaway-demotion footgun.
"""

import pytest

from backend.copilot.tools.graphiti_forget import (
    _retract_edges,
    _soft_delete_edges,
    invalidate_entity_direct_neighbors,
    mark_edges_superseded,
)


async def _select_edge(driver, uuid: str) -> dict | None:
    """Return the first row of edge properties matching ``uuid`` (or None)."""
    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {uuid: $uuid}]-()
        RETURN e.expired_at AS expired_at,
               e.invalid_at AS invalid_at,
               e.status AS status,
               e.expiration_reason AS expiration_reason
        """,
        uuid=uuid,
    )
    return records[0] if records else None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retract_edges_sets_only_expired_at(seeded_graph) -> None:
    """Per Snodgrass — system retraction sets expired_at only."""
    driver, group_id = seeded_graph

    deleted, failed = await _retract_edges(driver, ["e1"], "test-user")
    assert deleted == ["e1"]
    assert failed == []

    row = await _select_edge(driver, "e1")
    assert row is not None
    assert row["expired_at"] is not None, "_retract_edges must set expired_at"
    assert row["invalid_at"] is None, (
        "_retract_edges must NOT set invalid_at — that's the contradiction-detector "
        "path. Conflating the two breaks the bi-temporal model."
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_soft_delete_edges_sets_both(seeded_graph) -> None:
    """Contradiction-detector path keeps the original both-timestamps behavior."""
    driver, group_id = seeded_graph

    deleted, _ = await _soft_delete_edges(driver, ["e1"], "test-user")
    assert deleted == ["e1"]

    row = await _select_edge(driver, "e1")
    assert row is not None
    assert row["expired_at"] is not None
    assert row["invalid_at"] is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mark_edges_superseded_writes_status_and_reason(
    seeded_graph,
) -> None:
    """`status` + `expiration_reason` survive on the durable edge for audit."""
    driver, group_id = seeded_graph

    deleted, failed = await mark_edges_superseded(
        driver,
        ["e1"],
        reason="stale_fact",
        new_status="superseded",
        user_id="test-user",
    )
    assert deleted == ["e1"]
    assert failed == []

    row = await _select_edge(driver, "e1")
    assert row is not None
    assert row["expired_at"] is not None
    assert row["status"] == "superseded"
    assert row["expiration_reason"] == "stale_fact"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mark_edges_superseded_supports_contradicted_status(
    seeded_graph,
) -> None:
    """`contradicted` is the other allowed status — used by P0.5 web fact-check."""
    driver, group_id = seeded_graph

    await mark_edges_superseded(
        driver,
        ["e2"],
        reason="web_contradicted:https://example.com",
        new_status="contradicted",
        user_id="test-user",
    )

    row = await _select_edge(driver, "e2")
    assert row is not None
    assert row["status"] == "contradicted"
    assert row["expiration_reason"].startswith("web_contradicted:")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalidate_entity_direct_neighbors_is_single_hop(
    clean_graph,
) -> None:
    """The 3-hop A→B→C→D test — the core P-1.3 / P0.3b guard.

    Build a chain A↔B↔C↔D, invalidate B. Edges directly attached to B
    (A↔B and B↔C) must be superseded. The remaining edge C↔D — one hop
    further out — must be **untouched**. The instinct to use
    ``[r:RELATES_TO*1..N]`` would propagate the cascade and destroy the
    tangential relationship; the bare ``[r:RELATES_TO]`` pattern in
    ``invalidate_entity_direct_neighbors`` is the discipline this test
    pins.
    """
    driver, group_id = clean_graph

    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'A', name: 'A', group_id: $gid}),
          (b:Entity {uuid: 'B', name: 'B', group_id: $gid}),
          (c:Entity {uuid: 'C', name: 'C', group_id: $gid}),
          (d:Entity {uuid: 'D', name: 'D', group_id: $gid}),
          (a)-[:RELATES_TO {uuid: 'AB', group_id: $gid, fact: 'A-B', status: 'active'}]->(b),
          (b)-[:RELATES_TO {uuid: 'BC', group_id: $gid, fact: 'B-C', status: 'active'}]->(c),
          (c)-[:RELATES_TO {uuid: 'CD', group_id: $gid, fact: 'C-D', status: 'active'}]->(d)
        """,
        gid=group_id,
    )

    demoted = await invalidate_entity_direct_neighbors(
        driver, group_id=group_id, entity_uuid="B", reason="dead_client"
    )

    assert set(demoted) == {
        "AB",
        "BC",
    }, f"Expected edges directly attached to B (AB, BC) to be demoted; got {demoted}"

    # CD must be untouched — that's the boundary contract.
    cd = await _select_edge(driver, "CD")
    assert cd is not None
    assert cd["expired_at"] is None, (
        "CD is two hops from B and must NOT be demoted. If this fires, the "
        "runaway-demotion guard has regressed — most likely someone introduced "
        "a variable-length pattern (*1..N) into the Cypher."
    )
    assert cd["status"] != "superseded"

    # Sanity: AB and BC should be superseded with the right reason.
    for edge_uuid in ("AB", "BC"):
        row = await _select_edge(driver, edge_uuid)
        assert row is not None
        assert row["expired_at"] is not None
        assert row["status"] == "superseded"
        assert row["expiration_reason"] == "dead_client"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalidate_entity_direct_neighbors_handles_both_edge_directions(
    clean_graph,
) -> None:
    """The query uses an undirected ``-[r:RELATES_TO]-`` pattern so both
    inbound and outbound edges are caught. Pin that.
    """
    driver, group_id = clean_graph

    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'A', name: 'A', group_id: $gid}),
          (b:Entity {uuid: 'B', name: 'B', group_id: $gid}),
          (c:Entity {uuid: 'C', name: 'C', group_id: $gid}),
          (a)-[:RELATES_TO {uuid: 'AB', group_id: $gid, fact: 'a→b', status: 'active'}]->(b),
          (c)-[:RELATES_TO {uuid: 'CB', group_id: $gid, fact: 'c→b', status: 'active'}]->(b)
        """,
        gid=group_id,
    )

    demoted = await invalidate_entity_direct_neighbors(
        driver, group_id=group_id, entity_uuid="B", reason="x"
    )
    assert set(demoted) == {"AB", "CB"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalidate_entity_does_not_affect_other_users(
    clean_graph,
) -> None:
    """``group_id`` scoping in the MATCH must isolate per-user graphs.

    Build the same entity-uuid in two different group_ids. Invalidate
    one; the other must be untouched.
    """
    driver, group_id = clean_graph
    other_group = group_id + "_other"

    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'shared', name: 'Shared', group_id: $g1}),
          (b:Entity {uuid: 'other',  name: 'Other',  group_id: $g1}),
          (a2:Entity {uuid: 'shared', name: 'Shared', group_id: $g2}),
          (b2:Entity {uuid: 'other',  name: 'Other',  group_id: $g2}),
          (a)-[:RELATES_TO {uuid: 'e_self', group_id: $g1, fact: 'self', status: 'active'}]->(b),
          (a2)-[:RELATES_TO {uuid: 'e_other', group_id: $g2, fact: 'other', status: 'active'}]->(b2)
        """,
        g1=group_id,
        g2=other_group,
    )

    demoted = await invalidate_entity_direct_neighbors(
        driver, group_id=group_id, entity_uuid="shared", reason="test"
    )
    assert demoted == ["e_self"]

    other_row = await _select_edge(driver, "e_other")
    assert other_row is not None
    assert other_row["expired_at"] is None, "other user's edge must not be touched"
