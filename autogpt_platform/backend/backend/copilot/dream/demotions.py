"""A dream pass's destructive stage: its demotions and entity invalidations,
and the account of what they did.

Demotions targeting a fact the pass never read are dropped first: the model
may invent a uuid, or be steered to one. The rest are written in
``(status, reason)`` buckets, in the order each bucket first appears and in
proposal order within it, then each entity invalidation. Usage data plays
no part in which of them are attempted, or in their order.

Every write carries the recall guard in its own statement
(``graphiti/guarded_writes.py``, with the protection ``recall_guard.py``
builds for its reason): a live fact the user recalled within the protection
window is left alone unless the write's reason overrides it, and the
statement reports the facts it changed and those it spared. Nothing is read
beforehand to decide.

A write that raises has an unknown outcome. It may have committed before
its acknowledgement was lost, it may never have arrived, or it may still be
queued on the server and land after the pass returns. It is counted in
``indeterminate``, and its facts are neither counted as changed nor assumed
untouched.

``protected_demotions`` counts the distinct facts an acknowledged write
spared that one final read, after every acknowledged write, finds live
(``_count_kept_live``). It is a snapshot at that read: a later forget,
another pass, or a write of this pass still in flight can retire a counted
fact afterwards. The read writes nothing and changes no write. A fact spared
twice (a duplicated demotion, or a demotion and an invalidation) counts
once, and a fact spared and then changed by a later write whose reason
overrides the guard is not counted once that change is visible. Each
operation's own summary still records what its write reported.

``accounting_complete`` is True only when that read answered and no write of
the pass has an unknown outcome. Otherwise the count is provisional: the
read's count if it answered, else spared minus acknowledged changes.

Entity invalidation single-hop demotes every live edge around the entity,
with no degree cap, the most destructive op in the pass, so it stays behind
its own LD flag for staged rollout, independent of the dream pass being
enabled.
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
    live_fact_uuids,
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
    """What the destructive stage did: one summary per operation, as each
    write reported it, and the facts protection kept live."""

    demotions: list[DemotionSummary] = Field(default_factory=list)
    entity_invalidations: list[EntityInvalidationSummary] = Field(default_factory=list)
    # Distinct facts an acknowledged write spared that the final read, after
    # every acknowledged write, found live (``_count_kept_live``).
    protected: int = 0
    # False when that read failed or any write's outcome is unknown: the
    # counts are then provisional.
    accounting_complete: bool = True

    @property
    def demoted(self) -> int:
        return sum(d.applied for d in self.demotions)

    @property
    def failed(self) -> int:
        """Demotions whose statement ran and matched no live fact."""
        return sum(
            not (d.applied or d.protected or d.indeterminate) for d in self.demotions
        )

    @property
    def entity_edges(self) -> int:
        return sum(len(s.edges_touched) for s in self.entity_invalidations)

    @property
    def indeterminate(self) -> int:
        """Writes that raised: each may have committed, never arrived, or
        still be queued on the server."""
        return sum(d.indeterminate for d in self.demotions) + sum(
            s.indeterminate for s in self.entity_invalidations
        )

    def spared_uuids(self) -> set[str]:
        """The distinct facts an acknowledged write spared."""
        return {d.edge_uuid for d in self.demotions if d.protected} | {
            uuid for s in self.entity_invalidations for uuid in s.edges_protected
        }

    def changed_uuids(self) -> set[str]:
        """The distinct facts an acknowledged write changed."""
        return {d.edge_uuid for d in self.demotions if d.applied} | {
            uuid for s in self.entity_invalidations for uuid in s.edges_touched
        }


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
        results = DemotionResults(
            demotions=await _demote(driver, scope, demotions, guard),
            entity_invalidations=[
                await _invalidate(driver, scope, inv, guard) for inv in invalidations
            ],
        )
        return await _count_kept_live(driver, scope, pass_id, results)
    finally:
        await driver.close()


async def _count_kept_live(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    pass_id: str,
    results: DemotionResults,
) -> DemotionResults:
    """*results* with the protected count set: of the facts an acknowledged
    write spared, those live at one read made after every acknowledged write,
    in one statement over just those facts. Accounting only: the read writes
    nothing and changes no write.

    The count is a snapshot at that read, and complete only when the read
    answered and no write of the pass has an unknown outcome. A write that
    raised may still be queued on the server, and the read (``RO_QUERY``,
    which does not wait for queued writes) can run before it lands. So an
    unknown write leaves the count provisional, as does a failed read, whose
    count falls back to spared minus acknowledged changes."""
    settled = results.indeterminate == 0
    spared = results.spared_uuids()
    if not spared:
        return results.model_copy(update={"accounting_complete": settled})
    try:
        live = await live_fact_uuids(driver, scope.group_id, sorted(spared))
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: the liveness read of {len(spared)} spared "
            "fact(s) failed; protected_demotions is provisional",
            exc_info=True,
        )
        provisional = len(spared - results.changed_uuids())
        return results.model_copy(
            update={"protected": provisional, "accounting_complete": False}
        )
    return results.model_copy(
        update={"protected": len(spared & live), "accounting_complete": settled}
    )


async def _demote(
    driver: AutoGPTFalkorDriver,
    scope: MemoryScope,
    demotions: list[DreamDemotion],
    guard: DemotionGuard,
) -> list[DemotionSummary]:
    """One writer call per (status, reason) bucket, which makes one guarded
    statement per edge, and one summary per demotion: the k-th demotion of a
    bucket gets the k-th outcome."""
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
        indeterminate=outcome is WriteOutcome.UNKNOWN,
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
        indeterminate=writes.unknown,
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
