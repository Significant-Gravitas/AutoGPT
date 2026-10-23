"""Complete, or resolve, the dream citation markers a graph still holds.

The ingestion worker writes a dream write's citation marker before the
write, naming the uuid the write's episode is written under
(``marked_write.py``), and deletes it once the write is recorded and settled
(``recall_derivation.record``). A marker left behind is in one of these
states, and ``reconcile`` acts on each:

- its write landed: graphiti saved the episode and every fact the episode
  lists (the record failed after the write, or the writer died). It is
  completed as the writer would have: the marker's citations recorded on
  the episode, found by its uuid in its graph, and on the facts the episode
  produced or merged into, the write settled (``recall_landing.settle``),
  then the marker deleted;
- ``aborted``: the writer's ``add_episode`` raised. An episode graphiti
  saved, even in part, is completed as above. Otherwise the marker is
  deleted, with the episode its writer placed (``write_pending``, never
  saved), in one statement that first confirms no saved episode has its
  uuid (``drop_unlanded``);
- ``pending``, not landed: its write is under way, or its writer died
  before graphiti saved it. It is ``waiting`` for
  ``MARKER_EXPIRY_SECONDS``, and while it is, a forget of anything it cites
  reports ``cleanup_error`` (``in_flight``): its write could still land
  after the forget, and only its own settle would retract it. Past the
  bound it is marked ``expired`` and no longer holds forgets up, but it is
  never deleted for its age: should the write land after all, its writer
  records and settles it, and a marker kept is completed as soon as its
  episode is there;
- ``expired``: as pending, past the bound. An operator lists and resolves
  these with ``migrations/dream_markers.py``.

Markers whose write landed come first, then aborted, pending and expired
ones, so waiting writes never starve a landed one: at most
``RECONCILE_MAX_MARKERS`` a call, and ``left`` says a landed one remains.

Every forget runs ``reconcile`` before anything else, holding the graph's
write lock (``recall_forget.py``); the dream reaper runs it for the graphs a
failed record was noted in (``provenance_pending.py``), and the derivation
backfill for every graph it writes. Idempotent: a marker completed twice
stamps the same record.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel

from .recall_citations import Citations
from .recall_derivation import ABORTED, EXPIRED, MARKER_LABEL, PENDING, record

logger = logging.getLogger(__name__)

# How long a pending marker whose write has not landed holds up the forgets
# of what it cites before it is marked expired (an add_episode takes minutes
# at most while the lock holds; this bounds a writer that died).
MARKER_EXPIRY_SECONDS = 24 * 3600
# Markers one call looks at; more are left for the next.
RECONCILE_MAX_MARKERS = 200
# The order markers are taken in: landed (or aborted and saved), aborted and
# not saved, pending and not landed, expired.
LANDED, DROPPABLE, UNLANDED, KEPT = 0, 1, 2, 3


class Reconciled(BaseModel):
    """One call's markers: ``completed`` (recorded, settled, deleted),
    ``unsettled`` (landed, but the record or its settle failed: kept),
    ``dropped`` (aborted, no episode saved), ``waiting`` (could still land:
    pending within the bound, or aborted with an episode saved since it was
    read), ``expired``, and whether a landed one was ``left``."""

    completed: int = 0
    unsettled: int = 0
    dropped: int = 0
    waiting: int = 0
    expired: int = 0
    left: bool = False

    def incomplete(self) -> bool:
        """A write that landed may still be without its record."""
        return self.left or self.unsettled > 0


async def reconcile(
    driver: GraphDriver, group_id: str, *, limit: int = RECONCILE_MAX_MARKERS
) -> Reconciled:
    """Complete or resolve up to ``limit`` of the markers graph ``group_id``
    holds, landed ones first. Raises when a read fails: what it did stays."""
    markers = _rows(
        await driver.execute_query(MARKERS_QUERY, group_id=group_id, limit=limit + 1)
    )
    done = Reconciled(left=len(markers) > limit and markers[limit]["rank"] == LANDED)
    now = datetime.now(timezone.utc)
    for marker in markers[:limit]:
        await _resolve(driver, group_id, marker, now, done)
    if done.completed or done.dropped or done.unsettled:
        logger.info(
            f"Reconciled graph {group_id[:20]}: {done.completed} dream "
            f"record(s) completed, {done.unsettled} not, {done.dropped} "
            "aborted write(s) resolved"
        )
    return done


async def _resolve(
    driver: GraphDriver,
    group_id: str,
    marker: dict[str, Any],
    now: datetime,
    done: Reconciled,
) -> None:
    """Act on one marker as its state says (the module docstring)."""
    if marker["rank"] == LANDED:
        if await complete(driver, group_id, marker):
            done.completed += 1
        else:
            done.unsettled += 1
    elif marker["state"] == ABORTED:
        if await drop_unlanded(driver, marker["uuid"]):
            done.dropped += 1
        else:
            done.waiting += 1
    elif marker["state"] == EXPIRED:
        done.expired += 1
    elif age_seconds(marker["created_at"], now) < MARKER_EXPIRY_SECONDS:
        done.waiting += 1
    else:
        await driver.execute_query(
            EXPIRE_QUERY, uuid=marker["uuid"], state=EXPIRED, now=now.isoformat()
        )
        done.expired += 1


async def complete(driver: GraphDriver, group_id: str, marker: dict[str, Any]) -> bool:
    """Record, settle and delete ``marker`` (a ``MARKERS_QUERY`` row) whose
    write landed; False when that stopped short (the marker stays)."""
    touched = _rows(
        await driver.execute_query(TOUCHED_FACTS_QUERY, episodes=[marker["episode"]])
    )
    citations = Citations(
        fact_uuids=marker["facts"] or [], episode_uuids=marker["episodes"] or []
    )
    return await record(
        driver,
        group_id,
        marker["uuid"],
        marker["episode"],
        [row["uuid"] for row in touched],
        citations,
    )


async def drop_unlanded(driver: GraphDriver, marker: str) -> bool:
    """Delete ``marker`` and the episode its writer placed, when no saved
    episode has its uuid; True when it was deleted."""
    rows = _rows(await driver.execute_query(DROP_UNLANDED_QUERY, uuid=marker))
    return bool(rows and rows[0]["dropped"])


async def in_flight(driver: GraphDriver, facts: list[str], episodes: list[str]) -> int:
    """How many dream writes that could still land (``pending`` markers
    within ``MARKER_EXPIRY_SECONDS``) cite one of ``facts`` or
    ``episodes``."""
    if not (facts or episodes):
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=MARKER_EXPIRY_SECONDS)
    rows = _rows(
        await driver.execute_query(
            IN_FLIGHT_QUERY,
            facts=facts,
            episodes=episodes,
            state=PENDING,
            cutoff=cutoff.isoformat(),
        )
    )
    return int(rows[0]["count"]) if rows else 0


def age_seconds(created_at: object, now: datetime) -> float:
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


# Every marker, with the rank its state gives it: whether graphiti saved its
# episode (found by uuid in the graph, no longer ``write_pending``) and every
# fact that episode lists (graphiti saves the episode first, its facts last).
MARKERS_QUERY = f"""
MATCH (m:{MARKER_LABEL})
OPTIONAL MATCH (ep:Episodic {{uuid: m.episode_uuid}})
WHERE ep.group_id = $group_id
WITH m, ep, coalesce(ep.entity_edges, []) AS listed
UNWIND (CASE WHEN size(listed) = 0 THEN [null] ELSE listed END) AS listed_uuid
OPTIONAL MATCH ()-[e:RELATES_TO {{uuid: listed_uuid}}]->()
WITH m, ep, size(listed) AS wanted, count(e) AS found
WITH m, coalesce(m.state, '{PENDING}') AS state,
     ep IS NOT NULL AND ep.write_pending IS NULL AS saved,
     ep IS NOT NULL AND ep.write_pending IS NULL AND found = wanted AS landed
WITH m, state, saved, CASE
        WHEN landed OR (saved AND state = '{ABORTED}') THEN {LANDED}
        WHEN state = '{ABORTED}' THEN {DROPPABLE}
        WHEN state = '{PENDING}' THEN {UNLANDED}
        ELSE {KEPT} END AS rank
RETURN m.uuid AS uuid, m.episode_uuid AS episode,
       coalesce(m.derived_from_facts, []) AS facts,
       coalesce(m.derived_from_episodes, []) AS episodes,
       m.created_at AS created_at, state, saved, rank
ORDER BY rank, created_at
LIMIT $limit
"""

# A marker whose write graphiti never saved, with the episode its writer
# placed: only when every episode under its uuid is still ``write_pending``
# (or there is none), tested in the statement that deletes it.
DROP_UNLANDED_QUERY = f"""
MATCH (m:{MARKER_LABEL} {{uuid: $uuid}})
OPTIONAL MATCH (ep:Episodic {{uuid: m.episode_uuid}})
WITH m, collect(ep) AS placed
WHERE all(ep IN placed WHERE ep.write_pending IS NOT NULL)
FOREACH (ep IN placed | DETACH DELETE ep)
DELETE m
RETURN count(m) AS dropped
"""

EXPIRE_QUERY = f"""
MATCH (m:{MARKER_LABEL} {{uuid: $uuid}})
WHERE coalesce(m.state, '{PENDING}') = '{PENDING}'
SET m.state = $state, m.expired_at = $now
"""

IN_FLIGHT_QUERY = f"""
MATCH (m:{MARKER_LABEL})
WHERE coalesce(m.state, $state) = $state
  AND m.created_at > $cutoff
  AND (any(x IN coalesce(m.derived_from_facts, []) WHERE x IN $facts)
       OR any(x IN coalesce(m.derived_from_episodes, []) WHERE x IN $episodes))
RETURN count(m) AS count
"""

# The facts an episode produced or merged into: graphiti lists it among
# their ``episodes`` (an edge it only invalidated does not).
TOUCHED_FACTS_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE any(x IN coalesce(e.episodes, []) WHERE x IN $episodes)
RETURN e.uuid AS uuid
"""
