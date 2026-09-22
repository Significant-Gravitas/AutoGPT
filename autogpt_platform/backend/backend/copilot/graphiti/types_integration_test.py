"""Integration tests for the custom edge-type schema (P-1.1).

Validates the schema contract that ``MemoryFact`` declares: when a
``:RELATES_TO`` edge carries the new properties (``status``,
``confidence``, ``source_kind``, ``scope``, ``provenance``,
``web_verified_at``, ``ratified_at``, ``expiration_reason``), all of
them round-trip cleanly through FalkorDB + the AutoGPTFalkorDriver,
and Cypher queries can filter on them natively.

What this does NOT test: end-to-end ``add_episode`` calls that exercise
the LLM-driven entity extractor. That path requires a live LLM key and
is covered by separate ``requires_llm``-marked tests when those are
added. The schema-roundtrip test below is enough to catch the audit
finding that motivated P-1.1 — metadata stranded inside
``:Episodic.content`` JSON. After this test passes we know the
properties at least *can* live on the durable edge.
"""

from datetime import datetime, timezone

import pytest

from .client import derive_group_id
from .config import graphiti_config
from .falkordb_driver import AutoGPTFalkorDriver


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hyphenated_uuid_group_id_round_trips_through_cypher(
    falkordb_available: bool,
) -> None:
    """P-1.6 regression: derive_group_id intentionally preserves hyphens
    (UUIDs contain them), and older Graphiti versions mangled hyphens in
    Cypher property keys (upstream issue #1483). This test pins that a
    real hyphenated UUID survives a write + read as a ``group_id`` property
    so we notice immediately if a future driver/version regresses.

    Uses its own driver — the ``clean_graph`` fixture mints hyphen-free
    test ids on purpose, which would mask the bug we are guarding.
    """
    user_id = "3237579d-a31a-4bb8-ab56-d6e8f7cd0244"
    group_id = derive_group_id(user_id)
    assert "-" in group_id, "derive_group_id must preserve hyphens for this regression"

    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
    )
    try:
        await driver.execute_query(
            "CREATE (n:Entity {uuid: 'probe', name: 'Alice', group_id: $gid})",
            gid=group_id,
        )
        records, _, _ = await driver.execute_query(
            "MATCH (n:Entity {group_id: $gid}) RETURN n.uuid AS uuid, n.group_id AS gid",
            gid=group_id,
        )
        assert len(records) == 1
        assert records[0]["uuid"] == "probe"
        assert records[0]["gid"] == group_id
    finally:
        try:
            await driver.execute_query("MATCH (n) DETACH DELETE n")
        finally:
            await driver.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_fact_properties_roundtrip(clean_graph) -> None:
    """Every MemoryFact property survives a write + read through Cypher."""
    driver, group_id = clean_graph

    # FalkorDB doesn't implement Cypher's literal-string ``datetime(...)``
    # constructor (errors with ``Unknown function 'datetime'``). Pass ISO
    # strings as parameters and write them straight onto the edge — the
    # property's wire type is irrelevant to this roundtrip test, which
    # only asserts the value reads back as non-null.
    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'a', name: 'Alice', group_id: $gid}),
          (t:Entity {uuid: 't', name: 'Atlas', group_id: $gid}),
          (a)-[:RELATES_TO {
              uuid: 'e1',
              group_id: $gid,
              fact: 'Alice works on Atlas',
              name: 'works_on',
              status: 'active',
              confidence: 0.85,
              source_kind: 'user_asserted',
              scope: 'real:global',
              provenance: 'session:abc#msg:42',
              web_verified_at: $web_verified_at,
              ratified_at: $ratified_at,
              expiration_reason: null
          }]->(t)
        """,
        gid=group_id,
        web_verified_at="2026-05-13T03:00:00Z",
        ratified_at="2026-05-14T03:00:00Z",
    )

    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {uuid: 'e1'}]->()
        RETURN e.status AS status,
               e.confidence AS confidence,
               e.source_kind AS source_kind,
               e.scope AS scope,
               e.provenance AS provenance,
               e.web_verified_at AS web_verified_at,
               e.ratified_at AS ratified_at,
               e.expiration_reason AS expiration_reason
        """,
    )

    assert len(records) == 1
    row = records[0]
    assert row["status"] == "active"
    assert row["confidence"] == pytest.approx(0.85)
    assert row["source_kind"] == "user_asserted"
    assert row["scope"] == "real:global"
    assert row["provenance"] == "session:abc#msg:42"
    assert row["web_verified_at"] is not None
    assert row["ratified_at"] is not None
    assert row["expiration_reason"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_status_filter_is_cypher_native(seeded_graph) -> None:
    """`WHERE e.status = 'active'` is the whole reason for P-1.1.

    Before custom edge types landed, the filter had to parse the
    `:Episodic.content` JSON blob in Python. After P-1.1, it lives on
    the edge property and Cypher does the work. This test pins that
    contract: seed two active edges, demote one, assert the active
    filter returns exactly one.
    """
    driver, group_id = seeded_graph

    # Demote e1. FalkorDB doesn't implement Cypher's no-arg ``datetime()``;
    # generate the timestamp in Python and bind it as a parameter (same
    # workaround the seeded_graph fixture, dream/fetch.py, and
    # tools/graphiti_forget.py use against the same backend).
    await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {uuid: 'e1'}]->()
        SET e.status = 'superseded',
            e.expired_at = $now,
            e.expiration_reason = 'integration-test-demotion'
        """,
        now=datetime.now(timezone.utc).isoformat(),
    )

    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {group_id: $gid}]->()
        WHERE e.status = 'active'
        RETURN e.uuid AS uuid
        """,
        gid=group_id,
    )
    surviving = {r["uuid"] for r in records}
    assert surviving == {"e2"}, f"expected only e2 to remain active, got {surviving}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_scope_filter_is_cypher_native(clean_graph) -> None:
    """Same shape as status filter, but for ``scope`` — proves the audit's
    "push scope into edges" recommendation works once custom types are wired.
    """
    driver, group_id = clean_graph

    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'a', name: 'Alice', group_id: $gid}),
          (b:Entity {uuid: 'b', name: 'Bob',   group_id: $gid}),
          (a)-[:RELATES_TO {uuid: 'g1', group_id: $gid, fact: 'global', scope: 'real:global'}]->(b),
          (a)-[:RELATES_TO {uuid: 'p1', group_id: $gid, fact: 'project',   scope: 'project:atlas'}]->(b),
          (a)-[:RELATES_TO {uuid: 'p2', group_id: $gid, fact: 'other',     scope: 'project:beacon'}]->(b)
        """,
        gid=group_id,
    )

    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {group_id: $gid}]->()
        WHERE e.scope = 'project:atlas' OR e.scope = 'real:global'
        RETURN e.uuid AS uuid
        """,
        gid=group_id,
    )
    assert {r["uuid"] for r in records} == {"g1", "p1"}
