"""A dream's writes, checked against forgets right before they are written.

A dream pass reads the graph, asks its model for consolidated facts and
proposals, and queues them as episodes minutes or hours later
(``dream/apply.py``). A forget can answer in between, and graphiti, which
never merges a statement into a forgotten edge (``recall_ingest.py``), would
then save such a write as a new live fact: the forget undone without a word
from the user. So each dream write carries what it rests on (``Citations``)
and the ingestion worker checks it under the graph's write lock, right
before ``add_episode`` (``ingest._write_locked``). A forget that answered
before the check drops the write; one that comes later waits for the lock
and then reaches the facts it names, as after any write.

A write rests on a forget when a fact it cites is forgotten or gone (a hard
forget deletes it), or an episode it cites is no longer recallable or gone.
A write that cites nothing is taken to cite everything its pass read
(``apply._citations``), and its statement is also compared with the
sentence every forgotten fact keeps for audit, lower-cased and with its
whitespace collapsed, as graphiti's own exact-match dedup compares. It has
no endpoints until graphiti extracts it, so that comparison spans the
graph: it drops more than a same-endpoints one would, never less.
"""

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel, Field

from .recall import (
    forgotten_fact_predicate,
    forgotten_facts_clause,
    recallable_episode_predicate,
)


class Citations(BaseModel):
    """What a dream write rests on, for ``rests_on_a_forget``."""

    fact_uuids: list[str] = Field(default_factory=list)
    episode_uuids: list[str] = Field(default_factory=list)
    # Set on a write that cites nothing itself: compared with the sentences
    # forgotten facts keep for audit.
    statement: str | None = None


async def rests_on_a_forget(
    driver: GraphDriver, citations: Citations | None
) -> str | None:
    """Why the write must not be made, or None when nothing it rests on has
    been forgotten (or it is not a dream write: no ``citations``)."""
    if citations is None:
        return None
    reason = await _cites_a_forget(driver, citations)
    if reason is None and citations.statement is not None:
        reason = await _restates_a_forget(driver, citations.statement)
    return reason


async def _cites_a_forget(driver: GraphDriver, citations: Citations) -> str | None:
    if not citations.fact_uuids and not citations.episode_uuids:
        return None
    result = await driver.execute_query(
        _CITED_QUERY,
        fact_uuids=citations.fact_uuids,
        episode_uuids=citations.episode_uuids,
    )
    rows = result[0] if result else []
    # No row reads as nothing found: every citation counts as forgotten.
    row = rows[0] if rows else {"facts": [], "episodes": []}
    if set(citations.fact_uuids) - set(row["facts"]):
        return "cites a fact that is forgotten or gone"
    if set(citations.episode_uuids) - set(row["episodes"]):
        return "cites an episode that is hidden or gone"
    return None


async def _restates_a_forget(driver: GraphDriver, statement: str) -> str | None:
    result = await driver.execute_query(
        _RESTATED_QUERY, statement=statement, space=r"\s+"
    )
    return "restates a forgotten fact" if result and result[0] else None


def _normalized(text: str) -> str:
    """``text`` lower-cased, whitespace collapsed and trimmed, in Cypher."""
    return f"trim(string.replaceRegEx(toLower({text}), $space, ' '))"


# Every cited fact still there and not forgotten, and every cited episode
# still recallable; one row.
_CITED_QUERY = (
    forgotten_facts_clause()
    + f"""
OPTIONAL MATCH ()-[fact:RELATES_TO]->()
WHERE fact.uuid IN $fact_uuids AND NOT (fact.uuid IN forgotten)
WITH forgotten, collect(DISTINCT fact.uuid) AS facts
OPTIONAL MATCH (episode:Episodic)
WHERE episode.uuid IN $episode_uuids
  AND {recallable_episode_predicate("episode")}
RETURN facts, collect(DISTINCT episode.uuid) AS episodes
"""
)

# A forgotten fact whose kept sentence reads as ``$statement``.
_RESTATED_QUERY = f"""
MATCH ()-[e:RELATES_TO]->()
WHERE {forgotten_fact_predicate("e")}
  AND {_normalized("coalesce(e.fact_redacted, e.fact)")} = {_normalized("$statement")}
RETURN e.uuid AS uuid
LIMIT 1
"""
