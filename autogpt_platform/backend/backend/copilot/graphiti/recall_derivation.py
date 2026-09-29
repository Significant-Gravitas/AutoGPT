"""What a dream write was derived from, recorded where a forget finds it.

A dream write cites the facts and episodes it rests on
(``dream/citations.py``). The ingestion worker holds the graph's write lock
for the whole write (``ingest._write_locked``), and inside it
(``marked_write.py``):

1. first, ``mark`` writes the write's complete citations to a
   ``DreamCitations`` marker in the same graph, ``pending``: the uuid the
   write's episode is given (drawn before the marker and handed to graphiti,
   so the marker names its episode exactly), the episode's name for people
   reading the graph (never used to find it), ``derived_from_facts``,
   ``derived_from_episodes``, the graph's ``group_id`` and when it was
   written; uuids and names only, never text. If the marker cannot be
   written the write is not made: it fails closed. So the provenance of
   every dream fact is in the graph before the fact is. Then the write is
   checked against the forgets made since the pass read the graph
   (``recall_citations.rests_on_a_forget``); one it drops has its marker
   deleted (``withdraw``). Marking first means a forget landing after that
   check, where the lock does not hold, finds the marker;
2. after ``add_episode``, ``record`` stores those citations as
   ``derived_from_facts`` and ``derived_from_episodes``:

   - on the dream's episode, found by its uuid in its graph. graphiti never
     saves an episode node again once ``add_episode`` has written it, so
     this is the lasting record of the write, and it marks the episode as
     the dream's;
   - on every fact the write produced or merged into whose source episodes
     (``episodes``) are all the dream's: the union of their records. A fact
     the write alone produced gets the write's citations. A fact it merged
     into that a user's own episode also states gets none: it has a source
     that no forget of what the dream cited reaches, and so it is never
     retracted for one;

   then settles the write (``recall_landing.settle``): a source it cites
   that a forget reached while it was being written (only where the lock
   did not hold) is cascaded from, which retracts the write's own facts,
   erasing them for a source a hard forget reached; and only then deletes
   the marker. A failure at any step leaves the marker, and the worker
   reports the write as ``provenance_pending``. A write whose
   ``add_episode`` raised marks its marker ``aborted`` (``abort``).

``recall_reconcile.reconcile`` completes or resolves whatever markers a
graph still holds, before every forget's cascade (``recall_forget.py``),
from the dream reaper (``provenance_pending.py``) and in the derivation
backfill.

graphiti rewrites a fact's attributes whenever its model merges a new
statement into it (``SET r = edge``), which drops the record. A later dream
write merging into a derived fact therefore rebuilds the union from the
episodes' records; a user's statement merging into one leaves it without a
record, as it should be once the user has said it themselves.

A forget retracts the live facts whose record names a fact it forgot or an
episode it hid, and hides the dream episodes whose record does
(``recall_cascade.py``).
"""

import logging
import uuid as uuidlib
from datetime import datetime, timezone

from graphiti_core.driver.driver import GraphDriver

from .recall_citations import Citations
from .recall_landing import settle

logger = logging.getLogger(__name__)

# The label of a pending record's marker, and the states it is in.
MARKER_LABEL = "DreamCitations"
PENDING = "pending"
ABORTED = "aborted"
EXPIRED = "expired"


async def mark(
    driver: GraphDriver,
    group_id: str,
    episode_uuid: str,
    episode_name: str,
    citations: Citations,
) -> str:
    """Write the ``pending`` marker of the dream write about to be made under
    ``episode_uuid``; the marker's uuid. Raises when it cannot be written:
    the caller must not make the write."""
    marker = str(uuidlib.uuid4())
    await driver.execute_query(
        MARK_QUERY,
        uuid=marker,
        group_id=group_id,
        episode=episode_uuid,
        name=episode_name,
        facts=list(dict.fromkeys(citations.fact_uuids)),
        episodes=list(dict.fromkeys(citations.episode_uuids)),
        state=PENDING,
        now=datetime.now(timezone.utc).isoformat(),
    )
    return marker


async def abort(driver: GraphDriver, marker: str) -> None:
    """Mark ``marker``'s write as one whose ``add_episode`` raised, for
    reconcile to resolve once it has found no episode under its uuid. Never
    raises: a marker left ``pending`` waits and then expires instead."""
    try:
        await driver.execute_query(
            ABORT_QUERY,
            uuid=marker,
            state=ABORTED,
            now=datetime.now(timezone.utc).isoformat(),
        )
    except Exception:
        logger.warning(f"Could not mark dream marker {marker} aborted", exc_info=True)


async def withdraw(driver: GraphDriver, marker: str) -> None:
    """Delete the marker of a write dropped before ``add_episode``; one that
    cannot be deleted is marked aborted, for reconcile to drop. Never
    raises."""
    try:
        await driver.execute_query(DROP_MARKER_QUERY, uuid=marker)
    except Exception:
        logger.warning(f"Could not withdraw dream marker {marker}", exc_info=True)
        await abort(driver, marker)


async def record(
    driver: GraphDriver,
    group_id: str,
    marker: str,
    episode_uuid: str,
    edge_uuids: list[str],
    citations: Citations,
) -> bool:
    """Record ``citations`` on the dream episode ``episode_uuid``, then on
    each of ``edge_uuids`` (the facts it produced or merged into) whose
    sources are all dream episodes, settle the write, then delete its
    ``marker`` (gone already is fine). False, the marker left for
    ``recall_reconcile.reconcile``, when a step failed."""
    try:
        await driver.execute_query(
            RECORD_EPISODE_QUERY,
            episode=episode_uuid,
            group_id=group_id,
            facts=list(dict.fromkeys(citations.fact_uuids)),
            episodes=list(dict.fromkeys(citations.episode_uuids)),
        )
        if edge_uuids:
            await driver.execute_query(
                STAMP_FACTS_QUERY, uuids=edge_uuids, group_id=group_id
            )
        if not await settle(driver, group_id, citations):
            return False
        await driver.execute_query(DROP_MARKER_QUERY, uuid=marker)
    except Exception:
        logger.warning(
            f"Failed to record what dream episode {episode_uuid} was derived "
            f"from in graph {group_id[:20]}; its marker stays for reconcile",
            exc_info=True,
        )
        return False
    return True


MARK_QUERY = f"""
CREATE (:{MARKER_LABEL} {{
    uuid: $uuid,
    group_id: $group_id,
    episode_uuid: $episode,
    episode_name: $name,
    derived_from_facts: $facts,
    derived_from_episodes: $episodes,
    state: $state,
    created_at: $now
}})
"""

ABORT_QUERY = f"""
MATCH (m:{MARKER_LABEL} {{uuid: $uuid}})
SET m.state = $state, m.aborted_at = $now
"""

DROP_MARKER_QUERY = f"""
MATCH (m:{MARKER_LABEL} {{uuid: $uuid}})
DELETE m
"""

RECORD_EPISODE_QUERY = """
MATCH (ep:Episodic {uuid: $episode})
WHERE ep.group_id = $group_id
SET ep.derived_from_facts = $facts,
    ep.derived_from_episodes = $episodes
"""

# A fact is stamped only when every episode it names is there and carries a
# record: a missing or unrecorded source (a user's episode) could have stated
# it, so it is left unstamped. ``reduce`` builds each union in source order,
# once each.
STAMP_FACTS_QUERY = """
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid IN $uuids
  AND (e.group_id = $group_id OR e.group_id IS NULL)
  AND e.forgotten_at IS NULL
MATCH (source:Episodic)
WHERE source.uuid IN coalesce(e.episodes, [])
WITH e, collect(DISTINCT source) AS sources
WITH e, sources, [s IN sources | s.uuid] AS found
WHERE all(x IN coalesce(e.episodes, []) WHERE x IN found)
  AND all(s IN sources WHERE s.derived_from_facts IS NOT NULL)
SET e.derived_from_facts = reduce(
        acc = [], s IN sources |
        acc + [x IN s.derived_from_facts WHERE NOT x IN acc]),
    e.derived_from_episodes = reduce(
        acc = [], s IN sources |
        acc + [x IN coalesce(s.derived_from_episodes, []) WHERE NOT x IN acc])
RETURN e.uuid AS uuid
"""
