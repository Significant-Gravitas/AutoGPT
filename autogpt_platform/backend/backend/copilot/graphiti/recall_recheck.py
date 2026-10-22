"""The last graph read before recall shows what it found.

Warm context and ``memory_search`` read facts (``recall.search_facts``) and
recent episodes (``recall.recent_episodes``) side by side and render them
once both reads are done; the fact search can take seconds (warm context
reranks with a cross-encoder), and a forget can answer in the meantime. So
right before rendering they call ``recheck``, which checks every fact and
episode they are about to show again, by uuid, in one statement, and keeps
the facts still live and the episodes still recallable. Every item shown
passed that check, its own last graph read. A forget writes each fact's
marker before anything else, so a check that began after the marker
committed drops the fact and every episode citing it; a response whose
check began earlier can still carry them while it is delivered (rendering,
scheduling, network), and no later read will.
"""

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode

from . import recall
from .scope import MemoryScope


async def recheck(
    scope: MemoryScope,
    facts: list[EntityEdge],
    episodes: list[EpisodicNode],
    *,
    include_tentative: bool = True,
) -> tuple[list[EntityEdge], list[EpisodicNode]]:
    """``facts`` still live and ``episodes`` still recallable, each in its
    order: one statement on the driver the fact search used, so no forget
    lands between the check of one item and another's."""
    if not facts and not episodes:
        return facts, episodes
    client = await recall.get_graphiti_client(scope.group_id)
    live, recallable = await _still_shown(
        client.driver, facts, episodes, include_tentative
    )
    return (
        [fact for fact in facts if fact.uuid in live],
        [episode for episode in episodes if episode.uuid in recallable],
    )


async def _still_shown(
    driver: GraphDriver,
    facts: list[EntityEdge],
    episodes: list[EpisodicNode],
    include_tentative: bool,
) -> tuple[set[str], set[str]]:
    """The uuids of ``facts`` still live and of ``episodes`` still
    recallable."""
    result = await driver.execute_query(
        _recheck_query(include_tentative),
        fact_uuids=[fact.uuid for fact in facts],
        episode_uuids=[episode.uuid for episode in episodes],
    )
    rows = result[0] if result else []
    # No row reads as nothing found: nothing is shown unchecked.
    row = rows[0] if rows else {"facts": [], "episodes": []}
    return set(row["facts"]), set(row["episodes"])


def _recheck_query(include_tentative: bool) -> str:
    """The live-fact and recallable-episode tests on the given uuids, in one
    statement returning one row."""
    live = recall.live_fact_predicate("fact", include_tentative=include_tentative)
    return (
        recall.forgotten_facts_clause()
        + f"""
OPTIONAL MATCH ()-[fact:RELATES_TO]->()
WHERE fact.uuid IN $fact_uuids AND {live}
WITH forgotten, collect(DISTINCT fact.uuid) AS facts
OPTIONAL MATCH (episode:Episodic)
WHERE episode.uuid IN $episode_uuids
  AND {recall.recallable_episode_predicate("episode")}
RETURN facts, collect(DISTINCT episode.uuid) AS episodes
"""
    )
