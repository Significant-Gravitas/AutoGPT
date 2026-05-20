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


BACKFILL_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.status IS NULL
SET e.status = 'active'
RETURN count(e) AS updated
"""


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
        records, _, _ = await driver.execute_query(BACKFILL_QUERY)
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

    Returns (users_processed, edges_updated).
    """
    from prisma import Prisma

    db = Prisma()
    await db.connect()
    try:
        users = await db.user.find_many()
    finally:
        await db.disconnect()

    total_users = 0
    total_edges = 0
    for user in users:
        total_users += 1
        updated = await backfill_one_user(user.id)
        total_edges += updated

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
