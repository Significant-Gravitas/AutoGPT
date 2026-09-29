"""Complete the dream citation markers a graph still holds.

The ingestion worker writes a dream write's citation marker before the write
and deletes it once the write's record has landed (``recall_derivation.py``).
A marker left behind means the record failed after the write (the worker
reported it as ``provenance_pending``), the graph write itself failed, or the
process died in between. ``reconcile`` completes each: it finds the episode
the marker names (by name: graphiti gives an episode its uuid inside
``add_episode``, so the marker cannot hold it), records the marker's
citations on it and stamps the facts that episode produced or merged into,
as ``record`` would have, then deletes the marker. A marker whose episode is
not in the graph is a write that never landed: deleted once it is older than
``MARKER_ORPHAN_SECONDS``, and left alone while younger, since without the
write lock (Redis unreachable) its write may still be under way.

Every forget runs it before anything else, holding the graph's write lock
(``recall_forget.py``), so its cascade always sees complete provenance; the
dream reaper runs it for the graphs a failed record was noted in
(``provenance_pending.py``), and the derivation backfill for every graph it
writes. Idempotent: a marker completed twice stamps the same record.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel

from .recall_derivation import DROP_MARKER_QUERY, MARKER_LABEL, STAMP_FACTS_QUERY

logger = logging.getLogger(__name__)

# How old a marker whose episode is not in the graph must be to count as a
# write that never landed (an add_episode takes minutes at most).
MARKER_ORPHAN_SECONDS = 3600
# Markers one call completes; more are left for the next.
RECONCILE_MAX_MARKERS = 200


class Reconciled(BaseModel):
    """One call's markers: ``completed`` (their record made), ``orphaned``
    (deleted, their write never landed), ``waiting`` (no episode yet, too
    young to call orphaned), and whether more were ``left`` than it took."""

    completed: int = 0
    orphaned: int = 0
    waiting: int = 0
    left: bool = False


async def reconcile(
    driver: GraphDriver, group_id: str, *, limit: int = RECONCILE_MAX_MARKERS
) -> Reconciled:
    """Complete up to ``limit`` of the markers graph ``group_id`` holds,
    oldest first. Raises when a query fails: what it completed stays."""
    markers = _rows(await driver.execute_query(PENDING_MARKERS_QUERY, limit=limit + 1))
    done = Reconciled(left=len(markers) > limit)
    now = datetime.now(timezone.utc)
    for marker in markers[:limit]:
        await _complete(driver, group_id, marker, now, done)
    if done.completed or done.orphaned:
        logger.info(
            f"Reconciled graph {group_id[:20]}: {done.completed} dream record(s) "
            f"completed, {done.orphaned} marker(s) of writes that never landed "
            "dropped"
        )
    return done


async def _complete(
    driver: GraphDriver,
    group_id: str,
    marker: dict[str, Any],
    now: datetime,
    done: Reconciled,
) -> None:
    """Record ``marker``'s citations on the episode it names and the facts
    that episode touched, then drop it; or drop an orphan's."""
    written = _rows(
        await driver.execute_query(
            RECORD_NAMED_EPISODES_QUERY,
            name=marker["name"],
            group_id=group_id,
            facts=marker["facts"] or [],
            episodes=marker["episodes"] or [],
        )
    )
    episodes = [row["uuid"] for row in written]
    if episodes:
        touched = _rows(
            await driver.execute_query(TOUCHED_FACTS_QUERY, episodes=episodes)
        )
        if touched:
            await driver.execute_query(
                STAMP_FACTS_QUERY,
                uuids=[row["uuid"] for row in touched],
                group_id=group_id,
            )
        done.completed += 1
    elif _age_seconds(marker["created_at"], now) < MARKER_ORPHAN_SECONDS:
        done.waiting += 1
        return
    else:
        done.orphaned += 1
    await driver.execute_query(DROP_MARKER_QUERY, uuid=marker["uuid"])


def _age_seconds(created_at: object, now: datetime) -> float:
    """How long ago a marker was written; unreadable reads as long ago."""
    try:
        written = datetime.fromisoformat(str(created_at))
    except ValueError:
        return float("inf")
    if written.tzinfo is None:
        written = written.replace(tzinfo=timezone.utc)
    return (now - written).total_seconds()


def _rows(result: Any) -> list[dict[str, Any]]:
    return result[0] if result else []


PENDING_MARKERS_QUERY = f"""
MATCH (m:{MARKER_LABEL})
RETURN m.uuid AS uuid, m.episode_name AS name,
       m.derived_from_facts AS facts, m.derived_from_episodes AS episodes,
       m.created_at AS created_at
ORDER BY created_at
LIMIT $limit
"""

# The dream episode a marker names; a dream episode's name
# (``dream_<pass>_<phase>_<n>``) is its pass's alone.
RECORD_NAMED_EPISODES_QUERY = """
MATCH (ep:Episodic {name: $name})
WHERE ep.group_id = $group_id OR ep.group_id IS NULL
SET ep.derived_from_facts = $facts,
    ep.derived_from_episodes = $episodes
RETURN ep.uuid AS uuid
"""

# The facts an episode produced or merged into: graphiti lists it among
# their ``episodes`` (an edge it only invalidated does not).
TOUCHED_FACTS_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE any(x IN coalesce(e.episodes, []) WHERE x IN $episodes)
RETURN e.uuid AS uuid
"""
