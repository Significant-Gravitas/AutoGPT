"""List and resolve the dream citation markers left in the memory graphs.

A dream write's citation marker (``recall_derivation.py``) goes once its
writer has recorded and settled the write, or once
``recall_reconcile.reconcile`` completes it after the write landed or drops
it after the write was aborted. A marker whose write never landed and whose
writer is gone (it crashed, or its process was stopped) stays ``pending``:
for ``MARKER_EXPIRY_SECONDS`` (a day) a forget of anything it cites reports
``cleanup_error``; then it is marked ``expired`` and kept, never deleted for
its age alone.

This lists every marker, one JSON line each: its graph, state and age, the
uuid its write's episode goes under, whether graphiti saved that episode,
whether the write landed, and how many facts and episodes it cites.
``--resolve <uuid>`` (repeatable) and ``--resolve-expired`` resolve markers,
a dry run unless ``--apply``: one whose episode graphiti saved is completed
as reconcile completes a landed one (its citations recorded, the write
settled, then the marker deleted); any other is deleted with the episode its
writer placed, by a statement that first confirms no saved episode has its
uuid. Resolve a pending marker only once its writer is known to be gone: a
write that lands after its marker went is still recorded and settled by its
writer, but should that writer die before its record, nothing names its
facts' provenance. Each graph is resolved holding its write lock; a graph
locked past ``MARKERS_LOCK_WAIT_SECONDS``, or not lockable, is skipped and
counted busy, one that raises is counted failed, and either makes the
script exit 1.

Usage:

    poetry run python -m backend.copilot.graphiti.migrations.dream_markers \\
        [--graph <name>] [--resolve <marker uuid> ...] [--resolve-expired] \\
        [--apply]
"""

import argparse
import asyncio
import logging
import math
import sys
from datetime import datetime, timezone
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel, Field

from backend.copilot.graphiti.falkordb_driver import open_graph_driver
from backend.copilot.graphiti.graphs import list_graph_names
from backend.copilot.graphiti.recall_derivation import EXPIRED
from backend.copilot.graphiti.recall_reconcile import (
    LANDED,
    MARKERS_QUERY,
    age_seconds,
    complete,
    drop_unlanded,
)
from backend.copilot.graphiti.scope_lock import LockState, graph_write_lock

from .backfill_legacy_forgets import MEMORY_GRAPH_PREFIXES

logger = logging.getLogger(__name__)

MARKERS_LOCK_WAIT_SECONDS = 60
# Markers read from one graph: far more than a graph should ever hold.
LIST_LIMIT = 10_000


class Marker(BaseModel):
    """One marker as listed: whether graphiti ``saved`` its episode and
    whether its write ``landed`` (reconcile completes it), with how many
    facts and episodes it cites."""

    graph: str
    uuid: str
    state: str
    age_hours: float | None
    episode: str
    saved: bool
    landed: bool
    facts: int
    episodes: int


class Selection(BaseModel):
    """The markers to resolve: these ``uuids``, and every expired one when
    ``expired``."""

    uuids: set[str] = Field(default_factory=set)
    expired: bool = False

    def wanted(self, row: dict[str, Any]) -> bool:
        return row["uuid"] in self.uuids or (self.expired and row["state"] == EXPIRED)

    def any(self) -> bool:
        return bool(self.uuids) or self.expired


class Resolved(BaseModel):
    """Markers ``completed`` (their write landed) and ``deleted`` (it never
    did), or with a dry run that would be; ``kept`` (an episode under the
    uuid was saved since it was read, or the completion stopped short); and
    the graphs ``busy`` or ``failed``."""

    completed: int = 0
    deleted: int = 0
    kept: int = 0
    busy: int = 0
    failed: int = 0

    def plus(self, other: "Resolved") -> "Resolved":
        theirs = other.model_dump()
        return Resolved(**{k: v + theirs[k] for k, v in self.model_dump().items()})


async def list_markers(driver: GraphDriver, graph: str) -> list[Marker]:
    """Every marker in ``graph``, landed ones first."""
    now = datetime.now(timezone.utc)
    return [_listed(graph, row, now) for row in await _markers(driver, graph)]


async def resolve_graph(
    driver: GraphDriver, graph: str, selection: Selection, *, apply: bool
) -> Resolved:
    """Resolve ``graph``'s selected markers holding its write lock; a dry
    run counts what it would do."""
    if not apply:
        rows = [row for row in await _markers(driver, graph) if selection.wanted(row)]
        saved = sum(1 for row in rows if _saved(row))
        return Resolved(completed=saved, deleted=len(rows) - saved)
    async with graph_write_lock(graph, wait_seconds=MARKERS_LOCK_WAIT_SECONDS) as lock:
        if lock is not LockState.HELD:
            logger.warning(f"Skipped graph {graph[:20]}: write lock {lock.value}")
            return Resolved(busy=1)
        done = Resolved()
        for row in await _markers(driver, graph):
            if selection.wanted(row):
                await _resolve(driver, graph, row, done)
        return done


async def _resolve(
    driver: GraphDriver, graph: str, row: dict[str, Any], done: Resolved
) -> None:
    """Complete a marker whose episode graphiti saved, else delete it."""
    if _saved(row):
        if await complete(driver, graph, row):
            done.completed += 1
        else:
            done.kept += 1
    elif await drop_unlanded(driver, row["uuid"]):
        done.deleted += 1
    else:
        done.kept += 1


def _saved(row: dict[str, Any]) -> bool:
    return row["rank"] == LANDED or bool(row["saved"])


def _listed(graph: str, row: dict[str, Any], now: datetime) -> Marker:
    age = age_seconds(row["created_at"], now)
    return Marker(
        graph=graph,
        uuid=row["uuid"],
        state=row["state"],
        age_hours=None if math.isinf(age) else round(age / 3600, 1),
        episode=row["episode"],
        saved=bool(row["saved"]),
        landed=row["rank"] == LANDED,
        facts=len(row["facts"]),
        episodes=len(row["episodes"]),
    )


async def _markers(driver: GraphDriver, graph: str) -> list[dict[str, Any]]:
    result = await driver.execute_query(MARKERS_QUERY, group_id=graph, limit=LIST_LIMIT)
    return result[0] if result else []


async def run(
    selection: Selection, *, apply: bool, graph: str | None = None
) -> Resolved:
    """List every memory graph's markers (or ``graph``'s), printing one JSON
    line each, and resolve the selected ones."""
    names = [graph] if graph else await _memory_graph_names()
    totals = Resolved()
    for name in names:
        driver = open_graph_driver(name)
        try:
            for marker in await list_markers(driver, name):
                print(marker.model_dump_json())
            if selection.any():
                found = await resolve_graph(driver, name, selection, apply=apply)
                totals = totals.plus(found)
        except Exception:
            logger.warning(f"Markers failed for graph {name[:20]}", exc_info=True)
            totals.failed += 1
        finally:
            await driver.close()
    return totals


async def _memory_graph_names() -> list[str]:
    names = await list_graph_names()
    return sorted(name for name in names if name.startswith(MEMORY_GRAPH_PREFIXES))


async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    selection = Selection(uuids=set(args.resolve or []), expired=args.resolve_expired)
    totals = await run(selection, apply=args.apply, graph=args.graph)
    if selection.any():
        verb = "resolved" if args.apply else "would resolve (dry run)"
        print(
            f"{verb}: {totals.completed} completed, {totals.deleted} deleted; "
            f"{totals.kept} kept"
        )
    if not (totals.busy or totals.failed):
        return 0
    print(f"skipped {totals.busy} busy and {totals.failed} failed graphs: run again")
    return 1


def parser() -> argparse.ArgumentParser:
    """The command line: lists, and resolves only with ``--apply``."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--graph", help="Only this graph.")
    cli.add_argument(
        "--resolve", action="append", metavar="UUID", help="Resolve this marker."
    )
    cli.add_argument(
        "--resolve-expired", action="store_true", help="Resolve every expired marker."
    )
    cli.add_argument("--apply", action="store_true", help="Write, under each lock.")
    return cli


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parser().parse_args())))
