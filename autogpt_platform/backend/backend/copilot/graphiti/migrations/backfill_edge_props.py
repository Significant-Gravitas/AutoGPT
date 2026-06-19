"""Backfill custom edge properties on existing ``:RELATES_TO`` edges.

Created 2026-05-12 alongside the introduction of custom Graphiti entity
and edge types (``graphiti/types.py``). Before custom types were wired
into ``add_episode``, the LLM extractor produced edges with only
``fact``/``name``/temporal-validity properties. The dream pass and
ratification loop want to filter edges by ``status`` natively in Cypher,
which requires the property to exist on every edge.

This script walks every per-user FalkorDB database and sets
``e.status = 'active'`` on every ``:RELATES_TO`` edge where ``status`` is
unset. It is idempotent — re-running is a no-op.

Usage:

    poetry run python -m \\
        backend.copilot.graphiti.migrations.backfill_edge_props

By default it processes every user. Pass ``--user-id <id>`` to backfill
one user (useful for canary / debugging).
"""

import argparse
import asyncio
import logging
import sys

from backend.copilot.graphiti.client import derive_group_id
from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver

logger = logging.getLogger(__name__)


# Backfill the three MemoryFact attributes that have meaningful,
# Cypher-filterable defaults (``status``, ``source_kind``, ``scope``).
# ``confidence`` and ``provenance`` legitimately default to NULL, so we
# leave them unset rather than inventing values. Each property is
# ``coalesce``-guarded so the query is idempotent AND a partial prior run
# (or a real value already set by ``mark_edges_superseded``) is never
# clobbered.
#
# ``status`` is temporal-aware, gated on ``expired_at`` ONLY: graphiti
# stamps ``expired_at = now()`` exactly when an edge is actually
# invalidated, so it's the reliable "already retired" signal — such an
# edge defaults to ``'superseded'`` rather than being mislabeled
# ``'active'``. We deliberately do NOT trigger on ``invalid_at``:
# ``invalid_at`` is parsed from LLM-extracted dates and can be
# FUTURE-dated (a fact true now but with a known end date), so a
# currently-valid edge would otherwise be wrongly marked superseded.
# (Pre-existing edges carry no demotion ``status`` because native graphiti
# invalidation only sets the temporal fields — exactly the 25 edges
# observed on the dev canary, all of which have ``expired_at`` set.)
BACKFILL_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.status IS NULL OR e.source_kind IS NULL OR e.scope IS NULL
SET e.source_kind = coalesce(e.source_kind, 'user_asserted'),
    e.scope = coalesce(e.scope, 'real:global'),
    e.status = coalesce(
        e.status,
        CASE WHEN e.expired_at IS NOT NULL THEN 'superseded' ELSE 'active' END
    )
RETURN count(e) AS updated
"""

# Page size for the Postgres User scan. We process users in fixed-size
# batches ordered by ``id`` so the migration's memory footprint stays
# constant even when the user base grows to millions of rows.
USER_BATCH_SIZE = 1000


async def backfill_one_user(user_id: str) -> int:
    """Set default status='active' on edges for one user. Returns count updated."""
    try:
        group_id = derive_group_id(user_id)
    except ValueError:
        logger.warning(
            "Skipping user %s — invalid for group_id derivation", user_id[:12]
        )
        return 0

    driver = AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=group_id,
    )
    try:
        # ``execute_query`` is typed as ``tuple | None`` upstream; coerce
        # to an empty tuple so pyright stops complaining about iterating
        # over a None and the runtime stays single-pass.
        result = await driver.execute_query(BACKFILL_QUERY)
        records = result[0] if result else []
        updated = records[0]["updated"] if records else 0
        if updated:
            logger.info("Backfilled %d edges for user %s", updated, user_id[:12])
        return updated
    except Exception:
        # User may have no graph yet (never used memory). Treat as no-op.
        logger.debug(
            "No graph found or query failed for user %s", user_id[:12], exc_info=True
        )
        return 0
    finally:
        await driver.close()


async def backfill_all_users() -> tuple[int, int]:
    """Walk every User row in Postgres and backfill their graph.

    Pages through ``User`` ordered by ``id`` so the migration's memory
    footprint stays bounded by ``USER_BATCH_SIZE`` regardless of how
    large the user table grows. Each page is fetched with
    ``id > last_seen_id`` (keyset / cursor pagination), which is
    O(log n) per page on the primary-key index — no growing OFFSET cost.

    Returns (users_processed, edges_updated).
    """
    from prisma import Prisma

    db = Prisma()
    await db.connect()
    total_users = 0
    total_edges = 0
    try:
        last_seen_id: str | None = None
        while True:
            where = {"id": {"gt": last_seen_id}} if last_seen_id else {}
            batch = await db.user.find_many(
                where=where,
                order={"id": "asc"},
                take=USER_BATCH_SIZE,
            )
            if not batch:
                break
            for user in batch:
                total_users += 1
                updated = await backfill_one_user(user.id)
                total_edges += updated
            last_seen_id = batch[-1].id
            if len(batch) < USER_BATCH_SIZE:
                break
    finally:
        await db.disconnect()

    return total_users, total_edges


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.user_id:
        updated = await backfill_one_user(args.user_id)
        print(f"backfilled {updated} edges for user {args.user_id[:12]}")
        return 0

    users, edges = await backfill_all_users()
    print(f"backfilled {edges} edges across {users} users")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user-id",
        help="Backfill a single user instead of all users. Useful for canary runs.",
    )
    sys.exit(asyncio.run(main(parser.parse_args())))
