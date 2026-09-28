"""Recall stamps: the usage a recall leaves on the facts it returned, and the
test the dream's destructive writes make of it.

Warm context (``dream.ratification.try_ratify_on_hit``) and ``memory_search``
(``record_recall``) stamp every live fact (``RELATES_TO`` edge) they return,
in one batched write per call: ``recall_count`` counts the recalls,
``last_recalled_at`` is the latest and ``prev_recalled_at`` the one before
it. A recall within ``RECALL_DEDUPE_INTERVAL`` of the last one is the same
use and is not counted again, so warm context pulling a fact into every turn
of a conversation counts it once. An edge never stamped reads as never
recalled, so no edge needs a backfill. Each scope's hooks stamp its own graph,
an expert's included.

Recall history is evidence that a memory is relied on, never that it has gone
stale: the dream shows it to its model (``dream/prompts.py``), and each of its
destructive writes carries ``spared_by_recall`` in its own statement, so a
fact recalled within the protection window is left alone
(``RecallProtection``, ``dream/recall_guard.py``). Retrieval order ignores the
stamps.

A stamp takes no graph write lock (``scope_lock.py``) and never waits for one.
That is safe because it writes only the three usage properties, never the
fact's envelope (its text, ``status``, ``expired_at``, ``forgotten_at``,
``expiration_reason``), and only on an edge that is live when the write runs
(one FalkorDB query runs whole), so it can neither write over a forget nor
bring a forgotten fact back. An ingestion that read the edge before the stamp
saves its older copy after it (graphiti's ``SET e = edge``) and so puts the
older usage values back: one recall is lost, which leaves the dream pass no
more destructive than if that recall had never happened.

Every stamp time is written by ``stamp_time``: a UTC string of one width, so
stamps compare by time as plain strings in Cypher, in the dedupe below and in
``spared_by_recall``.
"""

import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel, ConfigDict

from .falkordb_driver import open_driver
from .recall import live_fact_predicate, record_hit
from .scope import MemoryScope

logger = logging.getLogger(__name__)

# Recalls closer together than this are one use: warm context re-reads the
# same facts turn after turn, and a day of that is one day of use.
RECALL_DEDUPE_INTERVAL = timedelta(hours=24)


class RecallStamp(BaseModel):
    """A fact's recall stamps as ``recall_stamp_columns`` reads them; ``None``
    where it has none (never recalled)."""

    uuid: str
    recall_count: int | None = None
    last_recalled_at: str | None = None
    prev_recalled_at: str | None = None


class RecallProtection(BaseModel):
    """What a destructive dream write leaves alone, tested in its own
    statement (``spared_by_recall``): a live fact last recalled at or after
    ``recalled_since``, unless ``override`` (the write says the fact is
    wrong, not stale), which never reaches ``cited``, the fact the write
    cites, so a fact cannot contradict itself. ``recalled_since=None``
    protects nothing."""

    model_config = ConfigDict(frozen=True)

    recalled_since: str | None = None
    override: bool = False
    cited: str | None = None

    def params(self) -> dict[str, str | bool | None]:
        """The statement parameters ``spared_by_recall`` reads."""
        return {
            "recalled_since": self.recalled_since,
            "override": self.override,
            "cited": self.cited,
        }


class _StampedRow(BaseModel):
    """The row ``_STAMP_QUERY`` returns."""

    stamped: int = 0


def stamp_time(moment: datetime) -> str:
    """A stamp time as it is written: UTC ISO 8601 with microseconds."""
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def parse_stamp(raw: str | None) -> datetime | None:
    """A stamp time as an aware datetime; ``None`` when missing or
    unreadable. A trailing ``Z`` and a naive value (read as UTC) pass."""
    if not raw:
        return None
    text = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def recall_stamp_columns(alias: str = "e") -> str:
    """The stamp columns ``RecallStamp`` holds, for a query binding the edge
    as *alias*. ``toString`` keeps a stamp time a string whatever type it was
    written as."""
    return (
        f"{alias}.recall_count AS recall_count, "
        f"toString({alias}.last_recalled_at) AS last_recalled_at, "
        f"toString({alias}.prev_recalled_at) AS prev_recalled_at"
    )


def spared_by_recall(alias: str) -> str:
    """The Cypher test ``RecallProtection`` puts in a destructive write's own
    statement: edge *alias* was last recalled at or after ``$recalled_since``
    and the write's ``$override`` does not reach it. Tested in the statement
    that writes, so a recall stamped before the statement runs is always
    seen, and one stamped after it finds the fact no longer live."""
    return (
        f"($recalled_since IS NOT NULL"
        f" AND {alias}.last_recalled_at IS NOT NULL"
        f" AND {alias}.last_recalled_at >= $recalled_since"
        f" AND NOT ($override AND ($cited IS NULL OR {alias}.uuid <> $cited)))"
    )


async def stamp_recalls(
    driver: GraphDriver, edge_uuids: Sequence[str], *, owner: str
) -> int:
    """Stamp one recall on each live fact among *edge_uuids*, in one query;
    how many were stamped (one recalled within the dedupe interval is not).
    Takes no graph write lock and waits for none (the module docstring says
    why that is safe). Never raises: a stamp that fails is logged and the
    recall goes on."""
    uuids = list(dict.fromkeys(edge_uuids))
    if not uuids:
        return 0
    now = datetime.now(timezone.utc)
    try:
        result = await driver.execute_query(
            _STAMP_QUERY,
            uuids=uuids,
            now=stamp_time(now),
            dedupe_cutoff=stamp_time(now - RECALL_DEDUPE_INTERVAL),
        )
        rows = result[0] if result else []
        return _StampedRow.model_validate(rows[0]).stamped if rows else 0
    except Exception:
        logger.warning(
            f"Recall stamp failed for {owner[:12]} ({len(uuids)} fact(s))",
            exc_info=True,
        )
        return 0


async def stamp_recalls_in_scope(scope: MemoryScope, edge_uuids: Sequence[str]) -> int:
    """``stamp_recalls`` on *scope*'s own graph, through a driver of its own.
    Never raises."""
    if not edge_uuids:
        return 0
    try:
        driver = open_driver(scope)
    except Exception:
        logger.warning(
            f"Recall stamp could not open {scope.owner_user_id[:12]}'s graph",
            exc_info=True,
        )
        return 0
    try:
        return await stamp_recalls(driver, edge_uuids, owner=scope.owner_user_id)
    finally:
        await _close(driver)


async def record_recall(scope: MemoryScope, edge_uuids: list[str]) -> None:
    """``memory_search``'s hit hook: count the hit for the ratification sweep
    (``recall.record_hit``), then stamp the recall. Never raises."""
    await record_hit(scope, edge_uuids)
    await stamp_recalls_in_scope(scope, edge_uuids)


async def _close(driver: GraphDriver) -> None:
    try:
        await driver.close()
    except Exception:
        logger.debug("Closing the recall stamp driver failed", exc_info=True)


# INVARIANT: ``e.last_recalled_at < $dedupe_cutoff`` compares strings, which
# orders by time only because ``stamp_time`` writes every stamp, one width
# and UTC. The edge lookup is graphiti's range index on ``RELATES_TO.uuid``.
_STAMP_QUERY = f"""
UNWIND $uuids AS target_uuid
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid = target_uuid AND {live_fact_predicate("e")}
  AND (e.last_recalled_at IS NULL OR e.last_recalled_at < $dedupe_cutoff)
SET e.recall_count = coalesce(e.recall_count, 0) + 1,
    e.prev_recalled_at = e.last_recalled_at,
    e.last_recalled_at = $now
RETURN count(e) AS stamped
"""
