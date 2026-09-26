"""Restamp the user forgets made before the recall policy.

A forget used to set ``expired_at`` on the fact and nothing else, and left
its episodes as they were. The recall policy still treats that shape as a
forget (``recall.legacy_forget_predicate``); this script makes those records
look like today's forgets, so that clause can be dropped once it has run
everywhere. It stamps ``redacted_at`` on every episode naming such an edge,
then gives the edge what ``recall_forget.retract`` writes:
``status='retracted'`` and ``expiration_reason='user_signal'``. Episodes go
first because a restamped edge no longer has the legacy shape, so a run cut
short between the two writes could not find its episodes again.

It walks every memory graph on the FalkorDB server, account (``user_*``) and
expert (``expert_*``) graphs alike, and is idempotent. Dry run by default: it
only counts. Pass ``--apply`` to write.

Usage:

    poetry run python -m \\
        backend.copilot.graphiti.migrations.backfill_legacy_forgets [--apply]

Pass ``--graph <name>`` to handle one graph (useful for a canary).
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

from pydantic import BaseModel

from backend.copilot.graphiti.config import graphiti_config
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.copilot.graphiti.memory_model import MemoryStatus
from backend.copilot.graphiti.recall import USER_FORGET_REASON, legacy_forget_predicate

logger = logging.getLogger(__name__)

# Memory graphs are named for their scope (``client.derive_memory_group_id``).
MEMORY_GRAPH_PREFIXES = ("user_", "expert_")


class LegacyForgets(BaseModel):
    """Legacy-forgotten edges, and the unredacted episodes naming them."""

    edges: int = 0
    episodes: int = 0


async def backfill_graph(driver: AutoGPTFalkorDriver, *, apply: bool) -> LegacyForgets:
    """Count one graph's legacy forgets and, with ``apply``, restamp them."""
    result = await driver.execute_query(COUNT_QUERY)
    records = result[0] if result else []
    found = LegacyForgets.model_validate(records[0]) if records else LegacyForgets()
    if not apply or not found.edges:
        return found
    await driver.execute_query(
        REDACT_EPISODES_QUERY, now=datetime.now(timezone.utc).isoformat()
    )
    await driver.execute_query(
        RETRACT_EDGES_QUERY,
        status=MemoryStatus.retracted.value,
        reason=USER_FORGET_REASON,
    )
    return found


async def backfill_all_graphs(
    *, apply: bool, graph: str | None = None
) -> LegacyForgets:
    """Backfill every memory graph on the server, or just ``graph``.

    A graph that fails is logged and skipped; a re-run picks it up.
    """
    names = [graph] if graph else await _memory_graph_names()
    totals = LegacyForgets()
    for name in names:
        driver = _graph_driver(name)
        try:
            found = await backfill_graph(driver, apply=apply)
        except Exception:
            logger.warning(f"Backfill failed for graph {name[:20]}", exc_info=True)
            continue
        finally:
            await driver.close()
        if found.edges:
            logger.info(f"{name[:20]}: {found.edges} edges, {found.episodes} episodes")
        totals.edges += found.edges
        totals.episodes += found.episodes
    return totals


async def _memory_graph_names() -> list[str]:
    driver = _graph_driver("default_db")
    try:
        names = await driver.client.list_graphs()
    finally:
        await driver.close()
    return sorted(name for name in names if name.startswith(MEMORY_GRAPH_PREFIXES))


def _graph_driver(database: str) -> AutoGPTFalkorDriver:
    """A driver on a graph known only by its name. ``open_driver`` needs a
    ``MemoryScope``, and an expert graph's name is a digest that no scope can
    be rebuilt from. Opening one creates no graph (see the driver)."""
    return AutoGPTFalkorDriver(
        host=graphiti_config.falkordb_host,
        port=graphiti_config.falkordb_port,
        password=graphiti_config.falkordb_password or None,
        database=database,
    )


_LEGACY = legacy_forget_predicate("e")

# Reads only, so a dry run never writes (nor creates) a graph.
COUNT_QUERY = f"""
OPTIONAL MATCH ()-[e:RELATES_TO]->()
WHERE {_LEGACY}
WITH collect(e.uuid) AS legacy
OPTIONAL MATCH (ep:Episodic)
WHERE ep.redacted_at IS NULL
  AND any(x IN coalesce(ep.entity_edges, []) WHERE x IN legacy)
RETURN size(legacy) AS edges, count(ep) AS episodes
"""

REDACT_EPISODES_QUERY = f"""
MATCH ()-[e:RELATES_TO]->()
WHERE {_LEGACY}
WITH collect(e.uuid) AS legacy
MATCH (ep:Episodic)
WHERE any(x IN coalesce(ep.entity_edges, []) WHERE x IN legacy)
SET ep.redacted_at = coalesce(ep.redacted_at, $now)
"""

RETRACT_EDGES_QUERY = f"""
MATCH ()-[e:RELATES_TO]->()
WHERE {_LEGACY}
SET e.status = $status, e.expiration_reason = $reason
"""


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    totals = await backfill_all_graphs(apply=args.apply, graph=args.graph)
    verb = "restamped" if args.apply else "would restamp (dry run)"
    print(f"{verb} {totals.edges} edges and {totals.episodes} episodes")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the changes.")
    parser.add_argument("--graph", help="Backfill one graph instead of all of them.")
    sys.exit(asyncio.run(main(parser.parse_args())))
