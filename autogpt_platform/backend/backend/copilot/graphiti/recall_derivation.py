"""What a dream write was derived from, recorded where a forget finds it.

A dream write cites the facts and episodes it rests on
(``dream/citations.py``). Right after the ingestion worker writes it, still
holding the graph's write lock (``ingest._write_locked``), ``record`` stores
those citations as ``derived_from_facts`` and ``derived_from_episodes``:

- on the dream's episode. graphiti never saves an episode node again once
  ``add_episode`` has written it, so this is the lasting record of the write,
  and it marks the episode as the dream's;
- on every fact the write produced or merged into whose source episodes
  (``episodes``) are all the dream's: the union of their records. A fact the
  write alone produced gets the write's citations. A fact it merged into that
  a user's own episode also states gets none: it has a source that no forget
  of what the dream cited reaches, and so it is never retracted for one.

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

from graphiti_core.driver.driver import GraphDriver

from .recall_citations import Citations

logger = logging.getLogger(__name__)


async def record(
    driver: GraphDriver,
    group_id: str,
    episode_uuid: str,
    edge_uuids: list[str],
    citations: Citations,
) -> None:
    """Record ``citations`` on the dream episode just written, then on each of
    ``edge_uuids`` (the facts it produced or merged into) whose sources are
    all dream episodes. Best-effort: a failure is logged and the write stands
    without its record; ``migrations/backfill_derivations.py`` recovers what
    the episode's ``source_description`` lists."""
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
    except Exception:
        logger.warning(
            f"Failed to record what dream episode {episode_uuid} was derived "
            f"from in graph {group_id[:20]}",
            exc_info=True,
        )


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
