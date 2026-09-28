"""Integration tests for the dream's single-hop entity invalidation
(``guarded_writes.invalidate_entity_direct_neighbors``), against live FalkorDB.

The unit-test sibling (``guarded_writes_test.py``) pins the Cypher strings and
call signatures via mock drivers; those run fast but don't catch Cypher
that's syntactically valid yet semantically wrong on FalkorDB (different
graph engines have slightly different behavior around relationship variable
scoping, ``MATCH`` semantics with property-only patterns, etc.). What the
recall guard in the same statement spares is
``recall_guard_integration_test.py``.

This file is the regression net that catches those. For every P-1.3
behavior, seed a known graph, run the helper, query the resulting
state via raw Cypher, and assert.

The single most important test in the file is
``test_invalidate_entity_direct_neighbors_is_single_hop`` — it pins the
boundary that distinguishes our scoped cascade from the
runaway-demotion footgun.
"""

import pytest

from .guarded_writes import invalidate_entity_direct_neighbors


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


# The user-forget retraction (``expired_at`` + ``status='retracted'``,
# never ``invalid_at``) is pinned live in ``recall_integration_test.py``.


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

    demoted = (
        await invalidate_entity_direct_neighbors(
            driver, group_id=group_id, entity_uuid="B", reason="dead_client"
        )
    ).changed

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

    demoted = (
        await invalidate_entity_direct_neighbors(
            driver, group_id=group_id, entity_uuid="B", reason="x"
        )
    ).changed
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

    demoted = (
        await invalidate_entity_direct_neighbors(
            driver, group_id=group_id, entity_uuid="shared", reason="test"
        )
    ).changed
    assert demoted == ["e_self"]

    other_row = await _select_edge(driver, "e_other")
    assert other_row is not None
    assert other_row["expired_at"] is None, "other user's edge must not be touched"
