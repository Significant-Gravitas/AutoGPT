"""Integration test for the P-1.2 backfill migration.

The migration script
(``backend/copilot/graphiti/migrations/backfill_edge_props.py``) stamps
``status``/``source_kind``/``scope`` on every existing ``:RELATES_TO``
edge that lacks them, recovering ``source_kind``/``scope`` from the
originating ``MemoryEnvelope`` JSON that survives in
``:Episodic.content``. This file pins that contract against live
FalkorDB:

- Pre-migration edges without metadata get the defaults.
- Edges sourced from an envelope episode recover the envelope's real
  ``source_kind``/``scope`` (the #13389 mislabeling fix).
- The migration is idempotent — re-running is a no-op.
- Edges that already have ``status`` set are not overwritten (e.g. an
  already-superseded edge stays superseded).
"""

import pytest

from .memory_model import MemoryEnvelope, SourceKind
from .migrations.backfill_edge_props import backfill_graph, backfill_one_user


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_adds_active_status_to_legacy_edges(
    clean_graph,
) -> None:
    """Edges created before P-1.1 have no ``status`` property; backfill
    sets them to 'active'. Run against the fixture driver directly
    (the higher-level ``backfill_one_user`` walks user_id → group_id,
    which we sidestep here since the fixture mints its own group_id).
    """
    driver, group_id = clean_graph

    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'a', name: 'Alice', group_id: $gid}),
          (b:Entity {uuid: 'b', name: 'Atlas', group_id: $gid}),
          (a)-[:RELATES_TO {uuid: 'legacy-1', group_id: $gid, fact: 'pre-P-1'}]->(b),
          (a)-[:RELATES_TO {uuid: 'legacy-2', group_id: $gid, fact: 'pre-P-1 too'}]->(b)
        """,
        gid=group_id,
    )

    # Sanity: no edge has status yet.
    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        WHERE e.status IS NULL
        RETURN count(e) AS n
        """
    )
    assert records[0]["n"] == 2

    assert await backfill_graph(driver) == 2

    # Every edge now has status='active' and the historical defaults.
    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        RETURN e.status AS status, e.source_kind AS source_kind, e.scope AS scope
        """
    )
    assert {r["status"] for r in records} == {"active"}
    assert {r["source_kind"] for r in records} == {"user_asserted"}
    assert {r["scope"] for r in records} == {"real:global"}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_recovers_envelope_source_kind_and_scope(
    clean_graph,
) -> None:
    """The #13389 mislabeling fix: an edge whose originating envelope was
    ``assistant_derived``/project-scoped must recover those values from
    the surviving ``:Episodic.content`` JSON — NOT be branded
    ``user_asserted``/``real:global``, which would grant a dream-derived
    fact the user-level trust ratification keys on."""
    driver, group_id = clean_graph

    dream_body = MemoryEnvelope(
        content="Atlas deploys are gated on the smoke suite",
        source_kind=SourceKind.assistant_derived,
        scope="project:atlas",
    ).model_dump_json()

    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'a', name: 'Alice', group_id: $gid}),
          (b:Entity {uuid: 'b', name: 'Atlas', group_id: $gid}),
          (ep:Episodic {uuid: 'ep-dream', group_id: $gid, content: $dream_body}),
          (conv:Episodic {uuid: 'ep-conv', group_id: $gid,
                          content: 'User: hello\\nAssistant: hi'}),
          (a)-[:RELATES_TO {uuid: 'dream-edge', group_id: $gid,
                            fact: 'deploy gate', episodes: ['ep-dream']}]->(b),
          (a)-[:RELATES_TO {uuid: 'merged-edge', group_id: $gid,
                            fact: 'merged fact',
                            episodes: ['ep-dream', 'ep-conv']}]->(b)
        """,
        gid=group_id,
        dream_body=dream_body,
    )

    assert await backfill_graph(driver) == 2

    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        RETURN e.uuid AS uuid, e.source_kind AS source_kind, e.scope AS scope
        """
    )
    by_uuid = {r["uuid"]: r for r in records}
    # Sole-sourced by the dream envelope → envelope values recovered.
    assert by_uuid["dream-edge"]["source_kind"] == "assistant_derived"
    assert by_uuid["dream-edge"]["scope"] == "project:atlas"
    # Merged with a conversation episode → most-trusted source wins.
    assert by_uuid["merged-edge"]["source_kind"] == "user_asserted"
    assert by_uuid["merged-edge"]["scope"] == "real:global"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_is_idempotent(clean_graph) -> None:
    """Re-running the migration is a no-op."""
    driver, group_id = clean_graph

    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'a', group_id: $gid}),
          (b:Entity {uuid: 'b', group_id: $gid}),
          (a)-[:RELATES_TO {uuid: 'e1', group_id: $gid, fact: 'fact'}]->(b)
        """,
        gid=group_id,
    )

    assert await backfill_graph(driver) == 1
    assert await backfill_graph(driver) == 0, "second run must update zero edges"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_does_not_overwrite_existing_status(
    clean_graph,
) -> None:
    """If an edge already has ``status`` set to something other than
    'active' (e.g. an edge that was demoted before the backfill ran),
    the migration must NOT clobber it.

    A demoted edge is still touched by the backfill: ``mark_edges_superseded``
    only writes ``status``/``expired_at``/``expiration_reason``, so its
    ``source_kind``/``scope`` are NULL and legitimately get the defaults.
    The coalesce guards ``status`` so the existing 'superseded' value
    survives.
    """
    driver, group_id = clean_graph

    await driver.execute_query(
        """
        CREATE
          (a:Entity {uuid: 'a', group_id: $gid}),
          (b:Entity {uuid: 'b', group_id: $gid}),
          (a)-[:RELATES_TO {
              uuid: 'already-superseded',
              group_id: $gid,
              fact: 'old fact',
              status: 'superseded',
              expiration_reason: 'previous_pass'
          }]->(b),
          (a)-[:RELATES_TO {
              uuid: 'legacy-active',
              group_id: $gid,
              fact: 'new fact'
          }]->(b)
        """,
        gid=group_id,
    )

    # Both edges match the WHERE clause: the legacy edge needs all three
    # defaults, and the superseded edge still needs source_kind/scope.
    assert await backfill_graph(driver) == 2

    # Verify the superseded edge kept its status while gaining the other
    # backfilled defaults.
    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {uuid: 'already-superseded'}]->()
        RETURN e.status AS status,
               e.expiration_reason AS reason,
               e.source_kind AS source_kind,
               e.scope AS scope
        """
    )
    assert records[0]["status"] == "superseded"
    assert records[0]["reason"] == "previous_pass"
    assert records[0]["source_kind"] == "user_asserted"
    assert records[0]["scope"] == "real:global"

    # And the legacy edge got 'active'.
    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {uuid: 'legacy-active'}]->()
        RETURN e.status AS status
        """
    )
    assert records[0]["status"] == "active"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_one_user_skips_missing_graph(
    falkordb_available, monkeypatch
) -> None:
    """``backfill_one_user`` should not raise on a user_id whose
    FalkorDB database has never been written to (no graph exists yet).

    Returns 0 quietly. The migration is safe to point at every User row
    in Postgres, including users who have never used memory features.

    Depends on ``falkordb_available`` so the suite is skipped cleanly
    when no FalkorDB is reachable (matches the other tests in this
    file, which inherit the skip via ``clean_graph``). Without this,
    CI runs without docker-compose'd FalkorDB hit a
    ``ConnectionError: localhost:6380``.
    """
    # Construct a user_id whose derived group_id is valid but corresponds
    # to a database that doesn't exist. derive_group_id will accept any
    # [a-zA-Z0-9_-] string.
    nonexistent_user = "no-such-user-in-falkordb"

    updated = await backfill_one_user(nonexistent_user)
    assert updated == 0
