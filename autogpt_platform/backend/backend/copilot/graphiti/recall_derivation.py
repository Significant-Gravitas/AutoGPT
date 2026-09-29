"""What a dream write was derived from, recorded where a forget finds it.

A dream write cites the facts and episodes it rests on
(``dream/citations.py``). The ingestion worker holds the graph's write lock
for the whole write (``ingest._write_locked``), and inside it:

1. before ``add_episode``, ``mark`` writes the write's complete citations to
   a ``DreamCitations`` marker in the same graph: the dream episode's name,
   ``derived_from_facts``, ``derived_from_episodes``, the graph's
   ``group_id`` and when it was written, uuids and names only, never text.
   If the marker cannot be written the write is not made: it fails closed.
   So the provenance of every dream fact is in the graph before the fact is;
2. after ``add_episode``, ``record`` stores those citations as
   ``derived_from_facts`` and ``derived_from_episodes``:

   - on the dream's episode. graphiti never saves an episode node again once
     ``add_episode`` has written it, so this is the lasting record of the
     write, and it marks the episode as the dream's;
   - on every fact the write produced or merged into whose source episodes
     (``episodes``) are all the dream's: the union of their records. A fact
     the write alone produced gets the write's citations. A fact it merged
     into that a user's own episode also states gets none: it has a source
     that no forget of what the dream cited reaches, and so it is never
     retracted for one;

   then deletes the marker: a marker only ever exists while a record is
   pending. A failure leaves it, and the worker reports the write as
   ``provenance_pending``.

``recall_reconcile.reconcile`` completes whatever markers a graph still
holds, and every forget runs it before its cascade (``recall_forget.py``),
so a forget always sees complete provenance; the dream reaper runs it for
the graphs whose record failed (``provenance_pending.py``).

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

logger = logging.getLogger(__name__)

# The label of a pending record's marker.
MARKER_LABEL = "DreamCitations"


async def mark(
    driver: GraphDriver, group_id: str, episode_name: str, citations: Citations
) -> str:
    """Write the marker of the dream write about to be made as
    ``episode_name``; its uuid. Raises when it cannot be written: the caller
    must not make the write."""
    marker = str(uuidlib.uuid4())
    await driver.execute_query(
        MARK_QUERY,
        uuid=marker,
        group_id=group_id,
        name=episode_name,
        facts=list(dict.fromkeys(citations.fact_uuids)),
        episodes=list(dict.fromkeys(citations.episode_uuids)),
        now=datetime.now(timezone.utc).isoformat(),
    )
    return marker


async def record(
    driver: GraphDriver,
    group_id: str,
    marker: str,
    episode_uuid: str,
    edge_uuids: list[str],
    citations: Citations,
) -> bool:
    """Record ``citations`` on the dream episode just written, then on each of
    ``edge_uuids`` (the facts it produced or merged into) whose sources are
    all dream episodes, then delete the write's ``marker``. False, the
    marker left for ``recall_reconcile.reconcile``, when a step failed."""
    try:
        await driver.execute_query(
            RECORD_EPISODE_QUERY,
            episode=episode_uuid,
            facts=list(dict.fromkeys(citations.fact_uuids)),
            episodes=list(dict.fromkeys(citations.episode_uuids)),
        )
        if edge_uuids:
            await driver.execute_query(
                STAMP_FACTS_QUERY, uuids=edge_uuids, group_id=group_id
            )
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
    episode_name: $name,
    derived_from_facts: $facts,
    derived_from_episodes: $episodes,
    created_at: $now
}})
"""

DROP_MARKER_QUERY = f"""
MATCH (m:{MARKER_LABEL} {{uuid: $uuid}})
DELETE m
"""

RECORD_EPISODE_QUERY = """
MATCH (ep:Episodic {uuid: $episode})
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
