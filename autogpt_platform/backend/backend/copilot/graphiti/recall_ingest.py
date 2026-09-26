"""How ingestion repairs what graphiti's ``add_episode`` does to a forget
(``graphiti/AGENTS.md`` lists what stays out of reach).

graphiti works from what it read when the episode started. It resolves each
new fact against every edge between the same entities, forgotten ones
included: its model can call the fact a duplicate of a forgotten edge (the
episode is appended to it and its attributes rewritten, the forget's fields
with them) or say it contradicts one (``invalid_at`` is stamped). And it
saves the edges and entities it read (``SET r = edge``, ``SET n = node``),
so a forget that landed while it ran is overwritten by the older copy.

So ``ingest._add_episode`` snapshots every forgotten edge before
``add_episode`` and calls ``keep_forgotten`` after it, which also reads the
forget stash (``recall_stash.py``, forgets from any process), decides what to
repair (``recall_ingest_plan.py``) and does it here:

- a forget that landed while the episode ran is applied again as a whole
  (``recall_forget.forget_edges``); a hard one whose edge stays gone still
  has its endpoints scrubbed;
- a forgotten edge graphiti changed is put back (``recall_restore.py``);
- whether the episode's statement is covered by the forget depends on when
  it was said: said before the forget, it is one of the forgotten fact's
  sources and is hidden with it, and an edge graphiti made from it that says
  the forgotten sentence again is forgotten too; said after, it states the
  fact again, is taken off the forgotten edge and gets a new live edge
  (``recall_restate.py``);
- an episode that only cites a forgotten edge as contradicted stops citing it.

None of it ever fails the write: every failure is logged, and a restore that
failed twice is left in the stash for the next ingestion or forget.
"""

import logging
from collections.abc import Awaitable
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import AddEpisodeResults
from graphiti_core.nodes import EpisodicNode

from .memory_model import ForgetResult
from .recall_forget import forget_edges
from .recall_hide import Hiding, hide, scrub_entities
from .recall_ingest_plan import IngestRun, Plan, Reapply, make_plan
from .recall_restate import restate
from .recall_restore import ForgottenEdge, restore
from .recall_stash import read_forgets

logger = logging.getLogger(__name__)


async def keep_forgotten(
    client: Graphiti,
    run: IngestRun,
    before: dict[str, ForgottenEdge],
    result: AddEpisodeResults,
) -> None:
    """Keep every forget through the ``add_episode`` that produced
    ``result`` (see the module docstring)."""
    stashed = await read_forgets(run.group_id)
    if not before and not stashed:
        return
    try:
        plan = await make_plan(client.driver, run, before, stashed, result)
    except Exception:
        logger.warning("Reading forgotten facts after ingestion failed", exc_info=True)
        return
    await carry_out(client, run, plan, result)


async def carry_out(
    client: Graphiti, run: IngestRun, plan: Plan, result: AddEpisodeResults
) -> None:
    """Each step on its own, so one failing leaves the others done."""
    driver = client.driver
    new_edges = await _restate(client, run, plan, result)
    await _step(_repoint(driver, result.episode, plan.unlink, new_edges))
    for spec in plan.restores:
        await restore(driver, run.group_id, spec)
    now = datetime.now(timezone.utc).isoformat()
    covered = Hiding(
        uuids=[spec.uuid for spec in plan.covered],
        recovered=[
            [spec.uuid, spec.audit["fact_redacted"], spec.audit["name_redacted"]]
            for spec in plan.covered
        ],
    )
    await _step(hide(driver, run.group_id, covered, now, ForgetResult()))
    if plan.scrub:
        await _step(scrub_entities(driver, [], plan.scrub))
    for (hard, reason), uuids in _grouped(plan.reapply).items():
        await forget_edges(driver, run.group_id, uuids, hard=hard, reason=reason)
    result.edges = [
        edge for edge in result.edges if edge.uuid not in plan.forgotten
    ] + new_edges


def _grouped(entries: list[Reapply]) -> dict[tuple[bool, str], list[str]]:
    grouped: dict[tuple[bool, str], list[str]] = {}
    for entry in entries:
        grouped.setdefault((entry.hard, entry.reason), []).append(entry.uuid)
    return grouped


async def _restate(
    client: Graphiti, run: IngestRun, plan: Plan, result: AddEpisodeResults
) -> list[EntityEdge]:
    if not plan.merged:
        return []
    merged = {(spec.source or "", spec.target or "") for spec in plan.merged}
    try:
        return await restate(client, result, merged, run.previous, run.instructions)
    except Exception:
        logger.warning("Restating facts taught again failed", exc_info=True)
        return []


async def _repoint(
    driver: GraphDriver,
    episode: EpisodicNode,
    unlink: list[str],
    added: list[EntityEdge],
) -> None:
    """Point the episode at its new live edges and away from the forgotten
    ones that would otherwise hide it."""
    if not unlink and not added:
        return
    await driver.execute_query(
        _REPOINT_QUERY,
        uuid=episode.uuid,
        unlink=unlink,
        added=[edge.uuid for edge in added],
    )


async def _step(work: Awaitable[object]) -> None:
    try:
        await work
    except Exception:
        logger.warning("A forget repair step failed after ingestion", exc_info=True)


_REPOINT_QUERY = """
MATCH (ep:Episodic {uuid: $uuid})
SET ep.entity_edges =
    [x IN coalesce(ep.entity_edges, []) WHERE NOT x IN $unlink] + $added
"""
