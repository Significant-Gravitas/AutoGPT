"""Recall stamps: the usage a recall leaves on the facts it returned.

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
stale: the dream pass reads the stamps (``dream/fetch.py``) only to leave a
recently recalled fact alone (``dream/recall_guard.py``), and retrieval order
ignores them.

A stamp takes no graph write lock (``scope_lock.py``) and never waits for one.
That is safe because it writes only the three usage properties, never the
fact's envelope (its text, ``status``, ``expired_at``, ``forgotten_at``,
``expiration_reason``), and only on an edge that is live when the write runs
(one FalkorDB query runs whole), so it can neither write over a forget nor
bring a forgotten fact back. An ingestion that read the edge before the stamp
saves its older copy after it (graphiti's ``SET e = edge``) and so puts the
older usage values back: one recall is lost, which leaves the dream pass no
more destructive than if that recall had never happened.

Every stamp time is written by ``stamp_time``: UTC with microseconds, so all
stamps have one width and the dedupe's string comparison is chronological.
"""

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel

from .falkordb_driver import open_driver
from .recall import live_fact_predicate, record_hit
from .scope import MemoryScope

logger = logging.getLogger(__name__)

# Recalls closer together than this are one use: warm context re-reads the
# same facts turn after turn, and a day of that is one day of use.
RECALL_DEDUPE_INTERVAL = timedelta(hours=24)


class RecallStamp(BaseModel):
    """A fact's recall stamps; ``None`` where it has none (never recalled)."""

    uuid: str
    recall_count: int | None = None
    last_recalled_at: str | None = None
    prev_recalled_at: str | None = None


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
    """The stamp columns ``stamp_fields`` reads, for a query binding the edge
    as *alias*. ``toString`` keeps a stamp time a string whatever type it was
    written as."""
    return (
        f"{alias}.recall_count AS recall_count, "
        f"toString({alias}.last_recalled_at) AS last_recalled_at, "
        f"toString({alias}.prev_recalled_at) AS prev_recalled_at"
    )


def stamp_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """A row's stamp columns as ``RecallStamp`` holds them: a count that is
    not a number, or a time that is not a string, reads as absent."""
    return {
        "recall_count": _count(row.get("recall_count")),
        "last_recalled_at": _text(row.get("last_recalled_at")),
        "prev_recalled_at": _text(row.get("prev_recalled_at")),
    }


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
    except Exception:
        logger.warning(
            f"Recall stamp failed for {owner[:12]} ({len(uuids)} fact(s))",
            exc_info=True,
        )
        return 0
    rows = result[0] if result else []
    stamped = rows[0].get("stamped") if rows else None
    return stamped if isinstance(stamped, int) else 0


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


async def read_recall_stamps(
    driver: GraphDriver, group_id: str, uuids: Sequence[str]
) -> list[RecallStamp] | None:
    """The stamps of the live facts among *uuids*, read now; ``None`` when
    the read fails. A fact no longer live is left out."""
    wanted = list(dict.fromkeys(uuids))
    if not wanted:
        return []
    return await _read(
        driver, _READ_STAMPS_QUERY, what=group_id, uuids=wanted, group_id=group_id
    )


async def read_neighbour_stamps(
    driver: GraphDriver, group_id: str, entity_uuid: str
) -> list[RecallStamp] | None:
    """The stamps of every live fact on entity *entity_uuid*: the neighbours
    ``invalidate_entity_direct_neighbors`` would demote. ``None`` when the
    read fails."""
    return await _read(
        driver,
        _NEIGHBOUR_STAMPS_QUERY,
        what=entity_uuid,
        entity_uuid=entity_uuid,
        group_id=group_id,
    )


async def _read(
    driver: GraphDriver, query: str, *, what: str, **params: Any
) -> list[RecallStamp] | None:
    try:
        result = await driver.execute_query(query, **params)
    except Exception:
        logger.warning(f"Reading recall stamps for {what[:24]} failed", exc_info=True)
        return None
    rows = result[0] if result else []
    return [
        RecallStamp(uuid=str(row["uuid"]), **stamp_fields(row))
        for row in rows
        if row.get("uuid")
    ]


async def _close(driver: GraphDriver) -> None:
    try:
        await driver.close()
    except Exception:
        logger.debug("Closing the recall stamp driver failed", exc_info=True)


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


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

_READ_STAMPS_QUERY = f"""
UNWIND $uuids AS target_uuid
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid = target_uuid AND e.group_id = $group_id
  AND {live_fact_predicate("e")}
RETURN e.uuid AS uuid, {recall_stamp_columns("e")}
"""

_NEIGHBOUR_STAMPS_QUERY = f"""
MATCH (n:Entity {{uuid: $entity_uuid, group_id: $group_id}})-[e:RELATES_TO]-()
WHERE {live_fact_predicate("e")}
RETURN DISTINCT e.uuid AS uuid, {recall_stamp_columns("e")}
"""
