"""A dream pass's destructive stage: its demotions and entity invalidations,
and the account of what they did.

Demotions targeting a fact the pass never read are dropped first: the model
may invent a uuid, or be steered to one. The rest are written grouped by
status and reason, in the order the pass proposed them, then each entity
invalidation. Usage data plays no part in which of them are attempted.

Every write carries the recall guard in its own statement
(``graphiti/guarded_writes.py``, with the protection ``recall_guard.py``
builds for its reason): a live fact the user recalled within the protection
window is left alone unless the write's reason overrides it, and the
statement returns the facts it changed and those it spared. Nothing is read
beforehand to decide. ``protected_demotions`` counts the distinct facts
protection kept live through the pass: spared by a write and changed by no
later one. A fact spared twice (a duplicated demotion, or a demotion and an
invalidation) counts once; a fact spared by one write and then changed by a
later one whose reason overrides the guard counts only as changed. Each
operation's own summary still records what its write did. A write that
fails is logged and changes nothing, as before recall stamps existed.

Entity invalidation single-hop demotes every live edge around the entity,
the most destructive op in the pass, so it stays behind its own LD flag for
staged rollout, independent of the dream pass being enabled.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver, open_driver
from backend.copilot.graphiti.guarded_writes import (
    WriteOutcome,
    invalidate_entity_direct_neighbors,
    supersede_unless_recalled,
)
from backend.copilot.graphiti.scope import MemoryScope
from backend.util.feature_flag import Flag, is_feature_enabled

from .batch_submit import read_input_bundle
from .recall_guard import DemotionGuard
from .schemas import (
    DemotionSummary,
    DreamDemotion,
    DreamOperations,
    EntityInvalidation,
    EntityInvalidationSummary,
)

logger = logging.getLogger(__name__)

DemotionStatus = Literal["superseded", "contradicted"]


class DemotionResults(BaseModel):
    """What the destructive stage did, one summary per operation."""

    demotions: list[DemotionSummary] = Field(default_factory=list)
    entity_invalidations: list[EntityInvalidationSummary] = Field(default_factory=list)

    @property
    def demoted(self) -> int:
        return sum(d.applied for d in self.demotions)

    @property
    def failed(self) -> int:
        return sum(not d.applied and not d.protected for d in self.demotions)

    @property
    def entity_edges(self) -> int:
        return sum(len(s.edges_touched) for s in self.entity_invalidations)

    @property
    def protected(self) -> int:
        """The distinct facts protection kept live: spared by a write of the
        stage and changed by no later one. A fact is changed at most once, so
        one both spared and changed was changed after it was spared."""
        spared = {d.edge_uuid for d in self.demotions if d.protected} | {
            uuid for s in self.entity_invalidations for uuid in s.edges_protected
        }
        changed = {d.edge_uuid for d in self.demotions if d.applied} | {
            uuid for s in self.entity_invalidations for uuid in s.edges_touched
        }
        return len(spared - changed)


async def apply_demotions(
    scope: MemoryScope,
    pass_id: str,
    ops: DreamOperations,
    known_fact_uuids: set[str] | None,
) -> DemotionResults:
    """Write *ops*' demotions and, when their flag is on, its entity
    invalidations, each under the recall guard; what they did.

    ``known_fact_uuids`` are the facts the pass read (``None``: look up the
    input bundle the batch path persisted, see ``_known_demotions``); they
    are also the facts a contradiction may cite."""
    demotions, known = await _known_demotions(pass_id, ops.demotions, known_fact_uuids)
    invalidations = await _enabled_invalidations(scope, ops.entity_invalidations)
    if not demotions and not invalidations:
        return DemotionResults()
    guard = DemotionGuard.at(datetime.now(timezone.utc), known)
    driver = open_driver(scope)
    try:
        return DemotionResults(
            demotions=await _demote(driver, scope, demotions, guard),
            entity_invalidations=[
                await _invalidate(driver, scope, inv, guard) for inv in invalidations
            ],
        )
    finally:
        await driver.close()


async def _demote(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    demotions: list[DreamDemotion],
    guard: DemotionGuard,
) -> list[DemotionSummary]:
    """One guarded write per (status, reason) bucket, one summary per
    demotion: the k-th demotion of a bucket gets the k-th outcome."""
    buckets: dict[tuple[DemotionStatus, str], list[str]] = {}
    for d in demotions:
        buckets.setdefault((d.new_status, d.reason), []).append(d.edge_uuid)
    outcomes = {
        (status, reason): iter(
            await supersede_unless_recalled(
                driver,
                uuids,
                reason=reason,
                new_status=status,
                # Defense-in-depth: the driver is already opened against the
                # scope's own graph, but the group_id predicate keeps a
                # wrong-driver caller from touching another scope's edges.
                group_id=scope.group_id,
                protection=guard.protection(reason),
                user_id=scope.owner_user_id,
            )
        )
        for (status, reason), uuids in buckets.items()
    }
    return [_summary(d, next(outcomes[(d.new_status, d.reason)])) for d in demotions]


def _summary(demotion: DreamDemotion, outcome: WriteOutcome) -> DemotionSummary:
    return DemotionSummary(
        edge_uuid=demotion.edge_uuid,
        reason=demotion.reason,
        new_status=demotion.new_status,
        applied=outcome is WriteOutcome.CHANGED,
        protected=outcome is WriteOutcome.SPARED,
    )


async def _invalidate(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    inv: EntityInvalidation,
    guard: DemotionGuard,
) -> EntityInvalidationSummary:
    """One entity's single-hop invalidation, its protected neighbours left
    alone by the write itself."""
    writes = await invalidate_entity_direct_neighbors(
        driver,
        group_id=scope.group_id,
        entity_uuid=inv.entity_uuid,
        reason=inv.reason,
        protection=guard.protection(inv.reason),
    )
    return EntityInvalidationSummary(
        entity_uuid=inv.entity_uuid,
        reason=inv.reason,
        edges_touched=writes.changed,
        edges_protected=writes.spared,
    )


async def _enabled_invalidations(
    scope: MemoryScope, invalidations: list[EntityInvalidation]
) -> list[EntityInvalidation]:
    """*invalidations* while ``DREAM_PASS_INVALIDATE_ENTITY`` is on for the
    owner, none otherwise; the flag is not read when there are none."""
    if invalidations and await is_feature_enabled(
        Flag.DREAM_PASS_INVALIDATE_ENTITY, scope.owner_user_id
    ):
        return invalidations
    return []


async def _known_demotions(
    pass_id: str,
    demotions: list[DreamDemotion],
    known_fact_uuids: set[str] | None,
) -> tuple[list[DreamDemotion], frozenset[str]]:
    """Code-level pre-flight for LLM-proposed demotion targets, and the facts
    the pass read (empty when unknown).

    The sanitize prompt tells the model only ``known_fact_uuids`` are
    valid demotion targets, but prompt text isn't enforcement — a
    hallucinated or injected uuid would otherwise reach Cypher and
    could demote edges the dream pass never fetched. Both the sync
    orchestrator and the batch callback converge on
    ``apply_operations``, so this is the one chokepoint that covers
    both paths.

    Both routes pass ``known_fact_uuids`` from their ``DreamInput``; a caller
    that passes none falls back to the input bundle persisted at submit time.
    If neither source exists (bundle expired/corrupted, or the Redis
    read itself fails) we keep the demotions rather than zeroing the
    pass — the same fail-open posture as the clamp's
    unknown-fact-count fallback — and log that validation was skipped;
    a contradiction can then cite nothing, so it overrides no protection.
    The Redis error MUST NOT propagate: by the time apply runs on the
    batch path the at-most-once apply gate is already claimed, so an
    exception here would permanently lose the dream (a retry hits the
    "duplicate" branch and skips apply entirely).

    Entity invalidations are NOT filtered here: the input bundle
    carries no entity-uuid allowlist (``FactRow.source``/``target``
    are entity *names*), so there is nothing to validate against.
    """
    if not demotions or known_fact_uuids is not None:
        known = frozenset(known_fact_uuids or ())
        return _only_known(pass_id, demotions, known), known
    try:
        bundle = await read_input_bundle(pass_id)
    except Exception as exc:
        logger.warning(
            f"Dream pass {pass_id}: input bundle read failed ({exc}) — failing "
            f"open and skipping known-fact validation for {len(demotions)} "
            "demotion(s)"
        )
        return demotions, frozenset()
    if bundle is None:
        logger.warning(
            f"Dream pass {pass_id}: no input bundle available — skipping "
            f"known-fact validation for {len(demotions)} demotion(s)"
        )
        return demotions, frozenset()
    known = frozenset(bundle.known_fact_uuids)
    return _only_known(pass_id, demotions, known), known


def _only_known(
    pass_id: str, demotions: list[DreamDemotion], known: frozenset[str]
) -> list[DreamDemotion]:
    kept = [d for d in demotions if d.edge_uuid in known]
    dropped = len(demotions) - len(kept)
    if dropped:
        logger.warning(
            f"Dream pass {pass_id}: dropped {dropped} demotion(s) targeting "
            "edge uuids outside the pass's known_fact_uuids (prompt-only "
            "constraint violated by the model)"
        )
    return kept
