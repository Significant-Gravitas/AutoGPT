"""Whether a forget reached what a dream write cites: directly, or through
the derived facts no longer live that it cites.

A cited fact a forget reached is forgotten, retracted by a forget's cascade
too (its reason names the root the user forgot), or gone: a hard forget
purged it (the pass read it, so it existed). A hard forget reached a fact
that is gone, one it is still purging (``hard_forgotten_at``, stamped when
it retracts the fact) and a derived fact its cascade erased (blank audit
copy). A cited episode a forget reached is hidden, and a tombstone, one
that is gone, or one hidden for a fact a hard forget reached, was emptied
by a hard forget or soon will be. Each leads to the root the user forgot
(``Reach``), hard when a hard forget reached it.

A forget's cascade walks down through a derived fact that is no longer
live (superseded, contradicted) without retracting it, and retracts what
rests on it (``recall_cascade.py``). A write citing such a fact rests on
its sources just as those do, and carries their content. So ``ancestry``
walks up from each cited derived fact no longer live, through the records
(``derived_from_facts``, ``derived_from_episodes``), level by level, as far
as derived facts no longer live go: the cascade's walk-through rule,
upward. A derived fact carries a record and no user's episode states it
(every episode it names carries a record). A source a forget reached ends
its branch, reached; a live fact, a user's fact or a recallable episode
ends it clean. Bounded like the cascade: at most ``CASCADE_MAX_ROUNDS``
levels up and ``CASCADE_MAX_ITEMS`` facts read. A walk stopped by either
with facts left to read is ``unfinished``, and its callers fail closed: the
check before a write drops it (``recall_citations.rests_on_a_forget``), the
settle after one lands retracts it (``recall_landing.settle``).
"""

from typing import Any

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel, Field

from .recall import (
    forgotten_fact_predicate,
    forgotten_facts_clause,
    live_fact_predicate,
    recallable_episode_predicate,
)
from .recall_cascade import CASCADE_MAX_ITEMS, CASCADE_MAX_ROUNDS
from .recall_cascade_walk import DERIVED_FROM_FORGOTTEN

_PREFIX = f"{DERIVED_FROM_FORGOTTEN}:"


class Reach(BaseModel):
    """Where a source a forget reached leads: the ``root`` the user forgot,
    and whether a ``hard`` forget reached it."""

    root: str
    hard: bool


class Ancestry(BaseModel):
    """Each cited derived fact no longer live that rests on a source a forget
    reached, and where that source leads; ``unfinished`` when the walk
    stopped at its bound with facts left to read."""

    reached: dict[str, Reach] = Field(default_factory=dict)
    unfinished: bool = False

    def add(self, cited: str, reach: Reach) -> None:
        """File ``reach`` for ``cited``, a hard one over a soft one."""
        known = self.reached.get(cited)
        if known is None or (reach.hard and not known.hard):
            self.reached[cited] = reach


async def fact_states(driver: GraphDriver, uuids: list[str]) -> dict[str, Any]:
    """``FACT_STATES_QUERY``'s row for each of ``uuids`` still in the graph."""
    return {row["uuid"]: row for row in await _read(driver, FACT_STATES_QUERY, uuids)}


def walkable(row: dict[str, Any]) -> bool:
    """A derived fact no longer live and not forgotten: the cascade walks
    through one, and a write citing one rests on what it rests on."""
    return bool(row["derived"]) and not (row["live"] or row["forgotten"])


def fact_reach(uuid: str, row: dict[str, Any] | None) -> Reach | None:
    """Where fact ``uuid`` leads when a forget reached it (its ``row`` None:
    gone, purged by a hard forget), else None."""
    if row is None:
        return Reach(root=uuid, hard=True)
    if not row["forgotten"]:
        return None
    return Reach(root=_root(uuid, row), hard=bool(row["hard"]))


async def reached_episodes(driver: GraphDriver, uuids: list[str]) -> dict[str, Reach]:
    """Each of ``uuids`` a forget reached (hidden, or gone), and where it
    leads: the root it was hidden for (the first fact in its
    ``redacted_for``, or that fact's root), else itself."""
    rows = await _read(driver, EPISODE_STATES_QUERY, uuids)
    hidden = [row for row in rows if row["hidden"]]
    firsts = [row["hidden_for"][0] for row in hidden if row["hidden_for"]]
    facts = await fact_states(driver, firsts)
    found = {row["uuid"] for row in rows}
    reached = {
        uuid: Reach(root=uuid, hard=True)
        for uuid in dict.fromkeys(uuids)
        if uuid not in found
    }
    reached.update({row["uuid"]: _episode_reach(row, facts) for row in hidden})
    return reached


async def ancestry(driver: GraphDriver, cited: list[dict[str, Any]]) -> Ancestry:
    """Walk up from the walkable facts among ``cited`` (``fact_states``
    rows) to the sources a forget reached (the module docstring). Each
    source is read once, for the first cited fact that reaches it: either
    caller needs one, since the check drops the whole write, and the
    settle's cascade from any cited fact reaches every fact the write made,
    whose record names them all."""
    found = Ancestry()
    frontier = [row for row in cited if walkable(row)]
    origin = {row["uuid"]: row["uuid"] for row in frontier}
    seen = set(origin)
    read = 0
    for _ in range(CASCADE_MAX_ROUNDS):
        if not frontier:
            return found
        parents = _named(frontier, origin, "facts", seen)
        episodes = _named(frontier, origin, "episodes", set())
        read += len(parents)
        if read > CASCADE_MAX_ITEMS:
            found.unfinished = True
            return found
        states = await fact_states(driver, list(parents))
        for uuid, cited_uuid in parents.items():
            if (reach := fact_reach(uuid, states.get(uuid))) is not None:
                found.add(cited_uuid, reach)
        for uuid, reach in (await reached_episodes(driver, list(episodes))).items():
            found.add(episodes[uuid], reach)
        seen.update(parents)
        frontier = [states[u] for u in parents if u in states and walkable(states[u])]
        origin.update({row["uuid"]: parents[row["uuid"]] for row in frontier})
    found.unfinished = bool(frontier)
    return found


def _named(
    frontier: list[dict[str, Any]], origin: dict[str, str], key: str, seen: set[str]
) -> dict[str, str]:
    """The sources ``frontier``'s records name under ``key`` (not ``seen``),
    each mapped to the cited fact it was reached from."""
    named: dict[str, str] = {}
    for row in frontier:
        for uuid in row[key]:
            if uuid not in seen and uuid not in named:
                named[uuid] = origin[row["uuid"]]
    return named


def _episode_reach(row: dict[str, Any], facts: dict[str, Any]) -> Reach:
    """Where a hidden episode leads; hard when it is a tombstone or a hard
    forget reached the fact it was hidden for."""
    first = row["hidden_for"][0] if row["hidden_for"] else None
    root = _root(first, facts[first]) if first in facts else first or row["uuid"]
    purged = first is not None and (first not in facts or bool(facts[first]["hard"]))
    return Reach(root=root, hard=bool(row["hard"]) or purged)


def _root(uuid: str, fact: dict[str, Any]) -> str:
    """The root a forgotten fact's retraction names, else the fact itself."""
    reason = fact.get("reason") or ""
    return reason.removeprefix(_PREFIX) if reason.startswith(_PREFIX) else uuid


async def _read(
    driver: GraphDriver, query: str, uuids: list[str]
) -> list[dict[str, Any]]:
    """``query``'s rows for ``uuids``, once each; none read for none."""
    if not uuids:
        return []
    result = await driver.execute_query(query, uuids=list(dict.fromkeys(uuids)))
    return result[0] if result else []


# Each of ``$uuids`` still in the graph: whether a forget reached it, and a
# hard one (purging it, or its cascade erased it), its reason (a cascade's
# names the root), whether it is live, whether it is derived (it carries a
# record and no user's episode states it: every episode it names carries
# one), and what its record names.
FACT_STATES_QUERY = f"""
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid IN $uuids
OPTIONAL MATCH (stated:Episodic)
WHERE stated.uuid IN coalesce(e.episodes, [])
  AND stated.derived_from_facts IS NULL
WITH e, count(stated) AS independent
RETURN e.uuid AS uuid, {forgotten_fact_predicate("e")} AS forgotten,
       e.hard_forgotten_at IS NOT NULL
           OR coalesce(e.fact_redacted, '-') = '' AS hard,
       e.expiration_reason AS reason, {live_fact_predicate("e")} AS live,
       e.derived_from_facts IS NOT NULL AND independent = 0 AS derived,
       coalesce(e.derived_from_facts, []) AS facts,
       coalesce(e.derived_from_episodes, []) AS episodes
"""

# Each of ``$uuids`` still in the graph, whether the recall policy hides it,
# whether a hard forget emptied it, and the forgotten facts it was hidden for.
EPISODE_STATES_QUERY = (
    forgotten_facts_clause()
    + f"""
MATCH (ep:Episodic)
WHERE ep.uuid IN $uuids
RETURN ep.uuid AS uuid, NOT ({recallable_episode_predicate("ep")}) AS hidden,
       ep.hard_deleted_at IS NOT NULL AS hard,
       coalesce(ep.redacted_for, []) AS hidden_for
"""
)
