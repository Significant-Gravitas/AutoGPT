"""Restamp the user forgets made before the recall policy.

A forget used to set ``expired_at`` on the fact and nothing else, and left
its episodes as they were. The recall policy still treats that shape as a
forget (``recall.legacy_forget_predicate``); this script makes those records
look like today's forgets, so that clause can be dropped once it has run
everywhere. It hides what a forget hides with the forget's own code
(``recall_hide.py``): the fact's sentence moves to ``fact_redacted``, the
summaries and attributes of the entities it joins or its episodes mention
are cleared, and so are their communities' summaries, and every episode
naming the edge is stamped ``redacted_at``. Then it gives the edge what
``recall_forget.retract`` writes: ``forgotten_at`` (the old forget's
``expired_at``), ``status='retracted'`` and ``expiration_reason='user_signal'``.
The restamp goes last because a restamped edge no longer has the legacy
shape, so a run cut short before it could not find the edge again.

It walks every memory graph on the FalkorDB server, account (``user_*``) and
expert (``expert_*``) graphs alike, and is idempotent. Dry run by default: it
only counts. Pass ``--apply`` to write.

With ``--apply`` each graph is rewritten holding its write lock
(``scope_lock.py``), the lock the ingestion worker and every forget take, from
the count to the last restamp: an ingestion running meanwhile cannot save its
older copy of an edge over the restamp. A graph whose lock another writer
keeps past ``BACKFILL_LOCK_WAIT_SECONDS`` (60 s), or that cannot be locked
because Redis is unreachable, is skipped with nothing written and counted
busy; a graph whose backfill raises is counted failed (every step is
idempotent). If any graph was skipped either way the script exits 1: run it
again to pick those graphs up.

Usage:

    poetry run python -m \\
        backend.copilot.graphiti.migrations.backfill_legacy_forgets [--apply]

Pass ``--graph <name>`` to handle one graph (useful for a canary).

``--scan-markers`` instead reads, and writes nothing: per graph, the uuids of
the facts carrying a forget's marker (``forgotten_at``), never their text, and
among them those whose sentence no forget scrubbed (it still reads otherwise
than ``recall.FORGOTTEN_FACT``). The backfill scrubs before it stamps; a
forget stamps first, and scrubs after, so only a forget whose clean-up failed
leaves its marker on an unscrubbed sentence (forgetting again finishes it).
Every other unscrubbed marker was invented by graphiti's attribute extraction
before it was held to its declared fields (``recall_ingest.py``), and would
hide a fact nobody forgot. Before the recall policy is deployed no forget has
written a marker, so every marker found then was invented. It exits 1 when
any marker is unscrubbed.
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
from backend.copilot.graphiti.recall import (
    FORGOTTEN_FACT,
    USER_FORGET_REASON,
    legacy_forget_predicate,
)
from backend.copilot.graphiti.recall_hide import REDACT_EPISODES_QUERY, scrub
from backend.copilot.graphiti.scope_lock import LockState, graph_write_lock

logger = logging.getLogger(__name__)

# How long ``--apply`` waits for a graph's write lock, as long as the
# ingestion worker waits, before counting the graph busy.
BACKFILL_LOCK_WAIT_SECONDS = 60

# Memory graphs are named for their scope (``client.derive_memory_group_id``).
MEMORY_GRAPH_PREFIXES = ("user_", "expert_")


class LegacyForgets(BaseModel):
    """Legacy-forgotten edges and the unredacted episodes naming them, and
    the graphs to run again: ``busy`` (write lock not taken, nothing
    written) and ``failed``."""

    edges: int = 0
    episodes: int = 0
    busy: int = 0
    failed: int = 0


class _GraphForgets(LegacyForgets):
    uuids: list[str] = []


class ForgetMarkers(BaseModel):
    """One graph's facts carrying a forget's marker, and those among them
    whose sentence no forget scrubbed (``--scan-markers``)."""

    graph: str
    marked: list[str] = []
    unscrubbed: list[str] = []


async def backfill_graph(driver: AutoGPTFalkorDriver, *, apply: bool) -> LegacyForgets:
    """Count one graph's legacy forgets and, with ``apply``, restamp them
    holding the graph's write lock from the count to the last restamp;
    ``busy``, with nothing written, when the lock cannot be taken."""
    if not apply:
        return await _legacy_forgets(driver)
    graph, wait = driver.graph_name, BACKFILL_LOCK_WAIT_SECONDS
    async with graph_write_lock(graph, wait_seconds=wait) as lock:
        if lock is not LockState.HELD:
            logger.warning(f"Skipped graph {graph[:20]}: write lock {lock.value}")
            return LegacyForgets(busy=1)
        found = await _legacy_forgets(driver)
        if found.edges:
            await _restamp(driver, found.uuids)
        return found


async def _legacy_forgets(driver: AutoGPTFalkorDriver) -> _GraphForgets:
    result = await driver.execute_query(COUNT_QUERY)
    records = result[0] if result else []
    return _GraphForgets.model_validate(records[0]) if records else _GraphForgets()


async def _restamp(driver: AutoGPTFalkorDriver, uuids: list[str]) -> None:
    """Hide what a forget hides, then give the edges a forget's marker."""
    await scrub(driver, uuids)
    await driver.execute_query(
        REDACT_EPISODES_QUERY,
        uuids=uuids,
        now=datetime.now(timezone.utc).isoformat(),
    )
    await driver.execute_query(
        RETRACT_EDGES_QUERY,
        uuids=uuids,
        status=MemoryStatus.retracted.value,
        reason=USER_FORGET_REASON,
    )


async def backfill_all_graphs(
    *, apply: bool, graph: str | None = None
) -> LegacyForgets:
    """Backfill every memory graph on the server, or just ``graph``.

    A graph that is busy or fails is logged, skipped and counted; a re-run
    picks it up.
    """
    names = [graph] if graph else await _memory_graph_names()
    totals = LegacyForgets()
    for name in names:
        driver = _graph_driver(name)
        try:
            found = await backfill_graph(driver, apply=apply)
        except Exception:
            logger.warning(f"Backfill failed for graph {name[:20]}", exc_info=True)
            totals.failed += 1
            continue
        finally:
            await driver.close()
        if found.edges:
            logger.info(f"{name[:20]}: {found.edges} edges, {found.episodes} episodes")
        totals.edges += found.edges
        totals.episodes += found.episodes
        totals.busy += found.busy
    return totals


async def scan_markers(graph: str | None = None) -> list[ForgetMarkers]:
    """Every memory graph's facts carrying a forget's marker, or just
    ``graph``'s; reads only."""
    names = [graph] if graph else await _memory_graph_names()
    found: list[ForgetMarkers] = []
    for name in names:
        driver = _graph_driver(name)
        try:
            result = await driver.execute_query(
                MARKERS_QUERY, placeholder=FORGOTTEN_FACT
            )
        finally:
            await driver.close()
        rows = result[0] if result else []
        found.append(
            ForgetMarkers(
                graph=name,
                marked=[row["uuid"] for row in rows],
                unscrubbed=[row["uuid"] for row in rows if not row["scrubbed"]],
            )
        )
    return found


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
    be rebuilt from. Opening one creates no graph, and builds its client off
    the event loop at the first command (see the driver)."""
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
RETURN legacy AS uuids, size(legacy) AS edges, count(ep) AS episodes
"""

# The facts carrying a forget's marker, and whether their sentence was
# scrubbed. Reads only.
MARKERS_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.forgotten_at IS NOT NULL
RETURN e.uuid AS uuid, coalesce(e.fact = $placeholder, false) AS scrubbed
ORDER BY uuid
"""

# A legacy forget set ``expired_at`` when the user forgot, so that is when
# the restamped edge was forgotten.
RETRACT_EDGES_QUERY = f"""
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid IN $uuids AND {_LEGACY}
SET e.forgotten_at = coalesce(e.forgotten_at, e.expired_at),
    e.status = $status,
    e.expiration_reason = $reason
"""


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.scan_markers:
        return _report(await scan_markers(args.graph))
    totals = await backfill_all_graphs(apply=args.apply, graph=args.graph)
    verb = "restamped" if args.apply else "would restamp (dry run)"
    print(f"{verb} {totals.edges} edges and {totals.episodes} episodes")
    if not (totals.busy or totals.failed):
        return 0
    print(f"skipped {totals.busy} busy and {totals.failed} failed graphs: run again")
    return 1


def _report(found: list[ForgetMarkers]) -> int:
    """Print each graph's marked and unscrubbed facts by uuid; 1 when any
    marker is unscrubbed."""
    for graph in found:
        if graph.marked:
            print(
                f"{graph.graph}: {len(graph.marked)} marked, "
                f"{len(graph.unscrubbed)} unscrubbed {graph.unscrubbed}"
            )
    unscrubbed = sum(len(graph.unscrubbed) for graph in found)
    marked = sum(len(graph.marked) for graph in found)
    print(f"{marked} facts marked forgotten, {unscrubbed} of them unscrubbed")
    return 1 if unscrubbed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Write the changes, each graph holding its write lock.",
    )
    mode.add_argument(
        "--scan-markers",
        action="store_true",
        help="Only report the facts carrying a forget's marker, by uuid.",
    )
    parser.add_argument("--graph", help="Backfill one graph instead of all of them.")
    sys.exit(asyncio.run(main(parser.parse_args())))
