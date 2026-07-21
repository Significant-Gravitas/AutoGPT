"""Integration test for the P-1.2 backfill migration.

The migration script
(``backend/copilot/graphiti/migrations/backfill_edge_props.py``) sets
default ``status='active'`` on every existing ``:RELATES_TO`` edge that
lacks the property. This file pins that contract against live FalkorDB:

- Pre-migration edges without ``status`` are updated to ``status='active'``.
- The migration is idempotent — re-running is a no-op.
- Edges that already have ``status`` set are not overwritten (e.g. an
  already-superseded edge stays superseded).
"""

import pytest

from .migrations.backfill_edge_props import BACKFILL_QUERY, backfill_one_user


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_adds_active_status_to_legacy_edges(
    clean_graph,
) -> None:
    """Edges created before P-1.1 have no ``status`` property; backfill
    sets them to 'active'. Run the query directly against the driver
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

    records, _, _ = await driver.execute_query(BACKFILL_QUERY)
    assert records[0]["updated"] == 2

    # Every edge now has status='active'.
    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO]->()
        RETURN e.status AS status
        """
    )
    assert {r["status"] for r in records} == {"active"}


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

    records, _, _ = await driver.execute_query(BACKFILL_QUERY)
    assert records[0]["updated"] == 1

    records, _, _ = await driver.execute_query(BACKFILL_QUERY)
    assert records[0]["updated"] == 0, "second run must update zero edges"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_backfill_does_not_overwrite_existing_status(
    clean_graph,
) -> None:
    """If an edge already has ``status`` set to something other than
    'active' (e.g. an edge that was demoted before the backfill ran),
    the migration must NOT clobber it.
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

    records, _, _ = await driver.execute_query(BACKFILL_QUERY)
    assert records[0]["updated"] == 1, "only the unstatussed edge should be touched"

    # Verify the superseded edge kept its status.
    records, _, _ = await driver.execute_query(
        """
        MATCH ()-[e:RELATES_TO {uuid: 'already-superseded'}]->()
        RETURN e.status AS status, e.expiration_reason AS reason
        """
    )
    assert records[0]["status"] == "superseded"
    assert records[0]["reason"] == "previous_pass"

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
