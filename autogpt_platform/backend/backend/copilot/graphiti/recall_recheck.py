"""The last graph read before recall shows what it found.

Warm context and ``memory_search`` read facts (``recall.search_facts``) and
recent episodes (``recall.recent_episodes``) side by side and render them
once both reads are done; the fact search can take seconds (warm context
reranks with a cross-encoder), and a forget can answer in the meantime. So
right before rendering they call ``recheck``, which reads both lists again
by uuid and keeps only the facts still live and the episodes still
recallable. That bounds the stale window to the time between this read and
the response: a forget that answers after it can miss a response already
on its way, and nothing later.
"""

import asyncio

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EpisodicNode

from . import recall
from .scope import MemoryScope

# The recallable-episode test, on the given episodes only.
_RECALLABLE_NOW_QUERY = (
    recall.forgotten_facts_clause()
    + f"""
MATCH (e:Episodic)
WHERE e.uuid IN $uuids AND {recall.recallable_episode_predicate("e")}
RETURN e.uuid AS uuid
"""
)


async def recheck(
    scope: MemoryScope,
    facts: list[EntityEdge],
    episodes: list[EpisodicNode],
    *,
    include_tentative: bool = True,
) -> tuple[list[EntityEdge], list[EpisodicNode]]:
    """``facts`` still live and ``episodes`` still recallable, each in its
    order, read again by uuid on the driver the fact search used: the last
    graph read before they are shown."""
    if not facts and not episodes:
        return facts, episodes
    driver = (await recall.get_graphiti_client(scope.group_id)).driver
    return await asyncio.gather(
        recall.live_now(driver, facts, include_tentative=include_tentative),
        recallable_now(driver, episodes),
    )


async def recallable_now(
    driver: GraphDriver, episodes: list[EpisodicNode]
) -> list[EpisodicNode]:
    """``episodes`` still recallable, in order, read again by uuid in one
    query."""
    if not episodes:
        return episodes
    result = await driver.execute_query(
        _RECALLABLE_NOW_QUERY, uuids=[episode.uuid for episode in episodes]
    )
    kept = {row["uuid"] for row in (result[0] if result else [])}
    return [episode for episode in episodes if episode.uuid in kept]
