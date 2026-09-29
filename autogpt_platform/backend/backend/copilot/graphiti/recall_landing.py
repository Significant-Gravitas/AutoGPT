"""A dream write that lands after a forget reached what it cites.

The graph's write lock keeps a forget from landing between a dream write's
check before ``add_episode`` (``recall_citations.rests_on_a_forget``) and its
record. Where the lock does not hold (Redis unreachable, or a holder that
lost its lease: ``scope_lock.py``), a forget can land in between, and its
cascade cannot reach the write's facts, which carry no record yet. So
whoever completes a write's citation marker, its writer right after the
record or ``recall_reconcile.reconcile``, settles the write first:
``settle`` looks up every source it cites and, when a forget reached one
since the pass read the graph, runs a forget's cascade from it
(``recall_cascade.py``). The write's facts, now recorded, are retracted and
its dream episode hidden, erased when a hard forget reached the source.
Only then is the marker cleared. The writer settles from the citations it holds, so it
does even when its marker is already gone.

A cited fact a forget reached is forgotten, retracted by a forget's cascade
too (its reason names the root the user forgot), or gone: a hard forget
purged it (the pass read it, so it existed). A hard forget reached a fact
that is gone, one it is still purging (``hard_forgotten_at``, stamped when
it retracts the fact) and a derived fact its cascade erased (blank audit
copy). A cited episode a forget reached is hidden, and a tombstone, one
that is gone, or one hidden for a fact a hard forget reached, was emptied
by a hard forget or soon will be. The cascade from each names the root the
user forgot, and erases for whatever a hard forget reached.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel, Field

from .memory_model import ForgetResult
from .recall import (
    forgotten_fact_predicate,
    forgotten_facts_clause,
    recallable_episode_predicate,
)
from .recall_cascade import cascade
from .recall_cascade_walk import DERIVED_FROM_FORGOTTEN
from .recall_citations import Citations, rests_on_a_forget

logger = logging.getLogger(__name__)

_PREFIX = f"{DERIVED_FROM_FORGOTTEN}:"


class Reached(BaseModel):
    """The sources a write cites that a forget reached, each mapped to the
    root its retractions name: facts, and hidden episodes (``seeds``), to
    cascade from softly or, for what a hard forget reached, erasing
    (``hard``)."""

    soft: dict[str, str] = Field(default_factory=dict)
    hard: dict[str, str] = Field(default_factory=dict)
    soft_seeds: dict[str, str] = Field(default_factory=dict)
    hard_seeds: dict[str, str] = Field(default_factory=dict)

    def any(self) -> bool:
        return bool(self.soft or self.hard or self.soft_seeds or self.hard_seeds)


async def settle(driver: GraphDriver, group_id: str, citations: Citations) -> bool:
    """Cascade from each source in ``citations`` a forget reached since the
    pass read the graph, so what rests on it (the write just recorded) is
    retracted; True when nothing is left to do, False when a cascade stopped
    short and the write's marker must stay. Raises when a read fails."""
    if await rests_on_a_forget(driver, citations) is None:
        return True
    reached = await reached_sources(driver, citations)
    if not reached.any():
        return True
    result = ForgetResult()
    now = datetime.now(timezone.utc).isoformat()
    for facts, seeds, erase in (
        (reached.hard, reached.hard_seeds, True),
        (reached.soft, reached.soft_seeds, False),
    ):
        if facts or seeds:
            await cascade(
                driver,
                group_id,
                list(facts),
                now,
                result,
                erase=erase,
                seeds=seeds,
                named=facts,
            )
    if result.failures:
        logger.warning(
            f"Could not settle a dream write in graph {group_id[:20]}: its "
            "cascade stopped short; its marker stays for reconcile"
        )
        return False
    logger.info(
        f"Settled a dream write in graph {group_id[:20]} that landed after a "
        f"forget of what it cites: {len(result.derived)} fact(s) retracted"
    )
    return True


async def reached_sources(driver: GraphDriver, citations: Citations) -> Reached:
    """The sources in ``citations`` a forget reached, as ``Reached``."""
    episodes = await _read(driver, CITED_EPISODES_QUERY, citations.episode_uuids)
    hidden = [row for row in episodes if row["hidden"]]
    hidden_for = [row["hidden_for"][0] for row in hidden if row["hidden_for"]]
    lookups = [*citations.fact_uuids, *hidden_for]
    facts = {
        row["uuid"]: row for row in await _read(driver, CITED_FACTS_QUERY, lookups)
    }
    reached = Reached()
    for uuid in dict.fromkeys(citations.fact_uuids):
        if uuid not in facts:
            reached.hard[uuid] = uuid
        elif facts[uuid]["forgotten"]:
            hard = facts[uuid]["hard"]
            (reached.hard if hard else reached.soft)[uuid] = _root(uuid, facts[uuid])
    found = {row["uuid"] for row in episodes}
    for uuid in dict.fromkeys(citations.episode_uuids):
        if uuid not in found:
            reached.hard_seeds[uuid] = uuid
    for row in hidden:
        _seed(reached, row, facts)
    return reached


def _seed(reached: Reached, row: dict[str, Any], facts: dict[str, Any]) -> None:
    """File a hidden cited episode under the root it was hidden for (the
    first fact in its ``redacted_for``, or that fact's root), else itself;
    to erase from when it is a tombstone or a hard forget reached that
    fact."""
    first = row["hidden_for"][0] if row["hidden_for"] else None
    name = _root(first, facts[first]) if first in facts else first or row["uuid"]
    purged = first is not None and (first not in facts or facts[first]["hard"])
    hard = row["hard"] or purged
    (reached.hard_seeds if hard else reached.soft_seeds)[row["uuid"]] = name


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


# Each of ``$uuids`` still in the graph, whether a forget reached it, and a
# hard one (purging it, or its cascade erased it), and its reason (a
# cascade's names the root).
CITED_FACTS_QUERY = f"""
MATCH ()-[e:RELATES_TO]->()
WHERE e.uuid IN $uuids
RETURN e.uuid AS uuid, {forgotten_fact_predicate("e")} AS forgotten,
       e.hard_forgotten_at IS NOT NULL
           OR coalesce(e.fact_redacted, '-') = '' AS hard,
       e.expiration_reason AS reason
"""

# Each of ``$uuids`` still in the graph, whether the recall policy hides it,
# whether a hard forget emptied it, and the forgotten facts it was hidden for.
CITED_EPISODES_QUERY = (
    forgotten_facts_clause()
    + f"""
MATCH (ep:Episodic)
WHERE ep.uuid IN $uuids
RETURN ep.uuid AS uuid, NOT ({recallable_episode_predicate("ep")}) AS hidden,
       ep.hard_deleted_at IS NOT NULL AS hard,
       coalesce(ep.redacted_for, []) AS hidden_for
"""
)
