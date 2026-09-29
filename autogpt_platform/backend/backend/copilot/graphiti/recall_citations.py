"""A dream's writes, checked against forgets right before they are written.

A dream pass reads the graph, asks its model for consolidated facts and
proposals, and queues them as episodes minutes or hours later
(``dream/apply.py``). A forget can answer in between, and graphiti, whose
model-decided dedup never merges a statement into a forgotten edge
(``recall_ingest.py``; its exact-text match only lists an episode on one
when the statement itself reads ``[forgotten]``), would then save such a
write as a new live fact: the forget undone without a word from the user.
So each dream write carries what it rests on (``Citations``) and the
ingestion worker checks it under the graph's write lock, right before
``add_episode`` (``ingest._write_locked``). A forget that answered before
the check drops the write; one that comes later waits for the lock and then
reaches the facts it names, as after any write.

A write rests on a forget when a fact it cites is forgotten or gone (a hard
forget deletes it, a forget's cascade retracts it), or an episode it cites is
no longer recallable or gone; and when a derived fact it cites that is no
longer live (superseded, contradicted) rests on one of those, as far up the
records as derived facts no longer live go (``recall_sources.ancestry``,
bounded like the cascade; a walk stopped by its bound drops the write too).
A forget's cascade walks through such a fact to what rests on it, and the
write would rest on it the same way, carrying that content.

The ``statement`` comparison is kept as defence in depth. apply no longer
sends a write that cites nothing: it drops one before it is queued
(``dream/citations.py``), so nothing in the codebase sets ``statement``
today. Should a write carrying only a ``statement`` reach the worker, it is
compared with the sentence every forgotten fact keeps for audit, lower-cased
and with its whitespace collapsed, as graphiti's own exact-match dedup
compares. It has no endpoints until graphiti extracts it, so that comparison
spans the graph: it drops more than a same-endpoints one would, never less.
A sentence a hard forget erased (``recall_erase.py``) is no longer there to
compare with.

Once written, what a dream write cites is recorded where a later forget
finds it (``recall_derivation.py``).
"""

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel, Field

from .recall import (
    forgotten_fact_predicate,
    forgotten_facts_clause,
    recallable_episode_predicate,
)
from .recall_sources import ancestry, fact_states


class Citations(BaseModel):
    """What a dream write rests on, for ``rests_on_a_forget``."""

    fact_uuids: list[str] = Field(default_factory=list)
    episode_uuids: list[str] = Field(default_factory=list)
    # Set on a write that cites nothing itself: compared with the sentences
    # forgotten facts keep for audit. apply sends none; kept as defence in
    # depth (see the module docstring).
    statement: str | None = None


async def rests_on_a_forget(
    driver: GraphDriver, citations: Citations | None
) -> str | None:
    """Why the write must not be made, or None when nothing it rests on has
    been forgotten (or it is not a dream write: no ``citations``)."""
    if citations is None:
        return None
    reason = await _cites_a_forget(driver, citations)
    if reason is None and citations.fact_uuids:
        reason = await _rests_through_derived(driver, citations.fact_uuids)
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


async def _rests_through_derived(driver: GraphDriver, facts: list[str]) -> str | None:
    """Why a write citing ``facts`` rests on a forget through the derived
    facts no longer live among them, else None."""
    walk = await ancestry(driver, list((await fact_states(driver, facts)).values()))
    if walk.reached:
        return "cites a derived fact no longer live that rests on a forget"
    if walk.unfinished:
        return "cites derived facts no longer live deeper than the check follows"
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
