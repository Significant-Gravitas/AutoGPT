"""Record what dream facts written before derivation records rest on.

The ingestion worker records a dream write's citations on its episode and on
the facts only dream episodes state (``recall_derivation.py``); a forget's
cascade follows them (``recall_cascade.py``). An older dream write has them
only in its episode's ``source_description``, the first five of each kind
(a consolidation listed episodes, a proposal facts). This reads them back,
only in the shapes the dream wrote and only uuids the graph has, facts in
the episode's own scope (``legacy_citations.py``: a model's rationale could
forge a citation): every dream episode with no record (named ``dream_...``
or described ``dream-pass...``) gets one from its description, empty when
it lists nothing or its shape is ambiguous; then every fact with no record
whose source episodes all have one gets their union, as ingestion stamps
it. It reports the ambiguous descriptions, the citations it rejected, and
the dream facts it could not attribute (stamped with nothing cited, so no
forget reaches them); a citation past the first five is lost too, and so is
one of a fact a hard forget purged before this ran. It does not cascade on
its own: ``--cascade-existing-forgets`` also runs a forget's cascade from
every root the graph's forgets left, the facts a hard forget purged and the
episodes a forget hid included (``backfill_cascade.py``), which a dry run
only counts.

Dry run by default. ``--apply`` writes, each graph holding its write lock
(``scope_lock.py``) from its first read to its last write, in batches of
``BATCH_SIZE``, after completing the dream records a failed write left
pending in it (``recall_reconcile.py``). A graph locked past
``BACKFILL_LOCK_WAIT_SECONDS``, or not lockable because Redis is
unreachable, is skipped unwritten and counted busy; one whose backfill
raises, or whose cascade stops short, is counted failed. Either makes the
script exit 1: run it again. Every write is idempotent.

Usage:

    poetry run python -m \\
        backend.copilot.graphiti.migrations.backfill_derivations \\
        [--apply] [--graph <name>] [--cascade-existing-forgets]
"""

import argparse
import asyncio
import logging
import sys
from typing import Any

from pydantic import BaseModel

from backend.copilot.dream.citations import envelope_scope
from backend.copilot.graphiti.falkordb_driver import (
    AutoGPTFalkorDriver,
    open_graph_driver,
)
from backend.copilot.graphiti.graphs import list_graph_names
from backend.copilot.graphiti.recall_reconcile import reconcile
from backend.copilot.graphiti.scope_lock import LockState, graph_write_lock

from .backfill_cascade import cascade_existing_forgets
from .backfill_legacy_forgets import MEMORY_GRAPH_PREFIXES
from .backfill_pages import pages, write_rows
from .legacy_citations import LegacyCitations, checked, described_citations

logger = logging.getLogger(__name__)

BACKFILL_LOCK_WAIT_SECONDS = 60


class Derivations(BaseModel):
    """What the backfill found (with ``--apply``, wrote): pending dream
    records completed (``reconciled``), dream ``episodes`` given a record,
    of their descriptions those ``ambiguous``, the citations ``rejected``
    (not in the graph in the episode's scope), ``facts`` stamped, of them
    ``unattributed`` with nothing cited; the ``roots`` a cascade started
    from (forgotten or purged facts, hidden episodes) and the facts it
    retracted (``derived``); and the graphs to run again, ``busy`` and
    ``failed``."""

    reconciled: int = 0
    episodes: int = 0
    ambiguous: int = 0
    rejected: int = 0
    facts: int = 0
    unattributed: int = 0
    roots: int = 0
    derived: int = 0
    busy: int = 0
    failed: int = 0

    def plus(self, other: "Derivations") -> "Derivations":
        theirs = other.model_dump()
        return Derivations(**{k: v + theirs[k] for k, v in self.model_dump().items()})


async def backfill_graph(
    driver: AutoGPTFalkorDriver, *, apply: bool, cascade_forgets: bool
) -> Derivations:
    """Record one graph's derivations; with ``apply``, write them holding the
    graph's write lock, and cascade from its forgets if asked."""
    if not apply:
        return await _derive(driver, apply=False, cascade_forgets=cascade_forgets)
    graph, wait = driver.graph_name, BACKFILL_LOCK_WAIT_SECONDS
    async with graph_write_lock(graph, wait_seconds=wait) as lock:
        if lock is not LockState.HELD:
            logger.warning(f"Skipped graph {graph[:20]}: write lock {lock.value}")
            return Derivations(busy=1)
        pending = await reconcile(driver, graph)
        found = await _derive(driver, apply=True, cascade_forgets=cascade_forgets)
        found.reconciled = pending.completed
        if pending.incomplete():
            found.failed = 1
        return found


async def _derive(
    driver: AutoGPTFalkorDriver, *, apply: bool, cascade_forgets: bool
) -> Derivations:
    """Both steps, reading the whole graph before writing anything."""
    records: dict[str, _Record] = {}
    described: dict[str, LegacyCitations] = {}
    scopes: dict[str, str] = {}
    for row in await pages(driver, DREAM_EPISODES_QUERY):
        if row["facts"] is None:
            described[row["uuid"]] = described_citations(row["description"])
            scopes[row["uuid"]] = envelope_scope(row["content"])
        else:
            records[row["uuid"]] = _Record(
                uuid=row["uuid"], facts=row["facts"], episodes=row["episodes"] or []
            )
    verified = await checked(driver, described, scopes)
    new = [
        _Record(uuid=uuid, facts=cited.facts, episodes=cited.episodes)
        for uuid, cited in verified.cited.items()
    ]
    records |= {record.uuid: record for record in new}
    stamps = [
        stamp
        for row in await pages(driver, UNSTAMPED_FACTS_QUERY)
        if (stamp := _union(row["uuid"], row["episodes"], records)) is not None
    ]
    found = Derivations(
        episodes=len(new),
        ambiguous=sum(1 for cited in described.values() if cited.ambiguous),
        rejected=verified.rejected,
        facts=len(stamps),
        unattributed=sum(1 for stamp in stamps if not (stamp.facts or stamp.episodes)),
    )
    if apply:
        await write_rows(driver, RECORD_EPISODES_QUERY, _dumped(new))
        await write_rows(driver, STAMP_FACTS_QUERY, _dumped(stamps))
    if cascade_forgets:
        cascaded = await cascade_existing_forgets(driver, apply=apply)
        found.roots, found.derived = cascaded.roots, cascaded.derived
        found.failed = int(cascaded.failed)
    return found


class _Record(BaseModel):
    """A dream episode's record, or the stamp a fact gets from its sources'."""

    uuid: str
    facts: list[str]
    episodes: list[str]


def _union(
    uuid: str, sources: list[str], records: dict[str, _Record]
) -> _Record | None:
    """Fact ``uuid``'s stamp when every episode it names has a record, as
    ``recall_derivation.STAMP_FACTS_QUERY`` builds it; else None."""
    named = list(dict.fromkeys(sources))
    if not named or any(source not in records for source in named):
        return None
    facts = [cited for source in named for cited in records[source].facts]
    episodes = [cited for source in named for cited in records[source].episodes]
    return _Record(
        uuid=uuid,
        facts=list(dict.fromkeys(facts)),
        episodes=list(dict.fromkeys(episodes)),
    )


def _dumped(records: list[_Record]) -> list[dict[str, Any]]:
    return [record.model_dump() for record in records]


async def backfill_all_graphs(
    *, apply: bool, cascade_forgets: bool, graph: str | None = None
) -> Derivations:
    """Backfill every memory graph on the server, or just ``graph``; a graph
    that is busy or fails is logged, skipped and counted."""
    names = [graph] if graph else await _memory_graph_names()
    totals = Derivations()
    for name in names:
        driver = open_graph_driver(name)
        try:
            found = await backfill_graph(
                driver, apply=apply, cascade_forgets=cascade_forgets
            )
        except Exception:
            logger.warning(f"Backfill failed for graph {name[:20]}", exc_info=True)
            totals.failed += 1
            continue
        finally:
            await driver.close()
        totals = totals.plus(found)
    return totals


async def _memory_graph_names() -> list[str]:
    names = await list_graph_names()
    return sorted(name for name in names if name.startswith(MEMORY_GRAPH_PREFIXES))


# Reads page by uuid, so a dry run never writes (nor creates) a graph.
DREAM_EPISODES_QUERY = """
MATCH (ep:Episodic)
WHERE ep.uuid > $after
  AND ep.write_pending IS NULL
  AND (ep.derived_from_facts IS NOT NULL
       OR ep.name STARTS WITH 'dream_'
       OR ep.source_description STARTS WITH 'dream-pass')
RETURN ep.uuid AS uuid, ep.source_description AS description,
       ep.content AS content,
       ep.derived_from_facts AS facts, ep.derived_from_episodes AS episodes
ORDER BY uuid
LIMIT $limit
"""

# As ingestion stamps: never a fact a forget reached.
UNSTAMPED_FACTS_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid > $after
  AND e.derived_from_facts IS NULL
  AND e.forgotten_at IS NULL
RETURN e.uuid AS uuid, coalesce(e.episodes, []) AS episodes
ORDER BY uuid
LIMIT $limit
"""

# ``IS NULL`` keeps a record already written, the ingestion worker's
# complete one included.
RECORD_EPISODES_QUERY = """
UNWIND $rows AS row
MATCH (ep:Episodic {uuid: row.uuid})
WHERE ep.derived_from_facts IS NULL
SET ep.derived_from_facts = row.facts,
    ep.derived_from_episodes = row.episodes
"""

STAMP_FACTS_QUERY = """
UNWIND $rows AS row
MATCH ()-[e:RELATES_TO {uuid: row.uuid}]->()
WHERE e.derived_from_facts IS NULL
SET e.derived_from_facts = row.facts,
    e.derived_from_episodes = row.episodes
"""


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    totals = await backfill_all_graphs(
        apply=args.apply,
        cascade_forgets=args.cascade_existing_forgets,
        graph=args.graph,
    )
    verb = "recorded" if args.apply else "would record (dry run)"
    print(
        f"completed {totals.reconciled} pending dream records; "
        f"{verb} {totals.episodes} dream episodes and {totals.facts} facts; "
        f"{totals.unattributed} dream facts cite nothing to attribute; "
        f"{totals.ambiguous} descriptions ambiguous, {totals.rejected} "
        "citations not in the graph, or of a fact of another scope"
    )
    if args.cascade_existing_forgets:
        done = f"retracted {totals.derived} derived facts" if args.apply else "not run"
        print(f"cascade from {totals.roots} roots forgets left: {done}")
    if not (totals.busy or totals.failed):
        return 0
    print(f"skipped {totals.busy} busy and {totals.failed} failed graphs: run again")
    return 1


def parser() -> argparse.ArgumentParser:
    """The command line: a dry run unless ``--apply``."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--apply", action="store_true", help="Write, under each lock.")
    cli.add_argument("--graph", help="Backfill one graph instead of all of them.")
    cli.add_argument(
        "--cascade-existing-forgets",
        action="store_true",
        help="Also retract what the dream derived from facts already forgotten.",
    )
    return cli


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parser().parse_args())))
