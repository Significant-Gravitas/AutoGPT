"""A dream write that lands after a forget reached what it cites.

The graph's write lock keeps a forget from landing between a dream write's
check before ``add_episode`` (``recall_citations.rests_on_a_forget``) and its
record. Where the lock does not hold (Redis unreachable, or a holder that
lost its lease: ``scope_lock.py``), a forget can land in between, and its
cascade cannot reach the write's facts, which carry no record yet. So
whoever completes a write's citation marker, its writer right after the
record or ``recall_reconcile.reconcile``, settles the write first:
``settle`` looks up every source it cites and, when a forget reached one
since the pass read the graph, directly or up the derived facts no longer
live that it cites (``recall_sources.py``), runs a forget's cascade from it
(``recall_cascade.py``). The write's facts, now recorded, are retracted and
its dream episode hidden, erased when a hard forget reached the source.
Only then is the marker cleared. The writer settles from the citations it
holds, so it does even when its marker is already gone.

A cited derived fact no longer live is cascaded from itself, under the root
the source it rests on names: the cascade leaves it as it is and retracts
what rests on it, the write's facts among them. When the walk up from one
stopped at its bound (``recall_sources.ancestry``), the settle fails closed
and cascades from it under its own name.
"""

import logging
from datetime import datetime, timezone

from graphiti_core.driver.driver import GraphDriver
from pydantic import BaseModel, Field

from .memory_model import ForgetResult
from .recall_cascade import cascade
from .recall_citations import Citations
from .recall_sources import (
    Reach,
    ancestry,
    fact_reach,
    fact_states,
    reached_episodes,
    walkable,
)

logger = logging.getLogger(__name__)


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

    def fact(self, uuid: str, reach: Reach) -> None:
        (self.hard if reach.hard else self.soft)[uuid] = reach.root

    def seed(self, uuid: str, reach: Reach) -> None:
        (self.hard_seeds if reach.hard else self.soft_seeds)[uuid] = reach.root


async def settle(driver: GraphDriver, group_id: str, citations: Citations) -> bool:
    """Cascade from each source in ``citations`` a forget reached since the
    pass read the graph, so what rests on it (the write just recorded) is
    retracted; True when nothing is left to do, False when a cascade stopped
    short and the write's marker must stay. Raises when a read fails."""
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
    """The sources in ``citations`` a forget reached, directly or through
    the derived facts no longer live among them, as ``Reached``."""
    reached = Reached()
    facts = await fact_states(driver, citations.fact_uuids)
    for uuid in dict.fromkeys(citations.fact_uuids):
        if (reach := fact_reach(uuid, facts.get(uuid))) is not None:
            reached.fact(uuid, reach)
    walk = await ancestry(driver, list(facts.values()))
    for uuid, reach in walk.reached.items():
        reached.fact(uuid, reach)
    if walk.unfinished:
        for row in facts.values():
            if walkable(row) and row["uuid"] not in walk.reached:
                reached.fact(row["uuid"], Reach(root=row["uuid"], hard=False))
    episodes = await reached_episodes(driver, citations.episode_uuids)
    for uuid, reach in episodes.items():
        reached.seed(uuid, reach)
    return reached
