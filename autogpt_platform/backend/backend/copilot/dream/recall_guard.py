"""Recall history protects a memory from the dream pass; it never condemns one.

A demotion the sanitizer proposes is dropped when the user recalled its fact
within ``Config.dream_demotion_protect_days`` (the recall stamps,
``graphiti/recall_stamp.py``), unless its reason says the fact is wrong rather
than stale: another fact this pass read contradicts it
(``contradicted_by:<uuid>``), or the user retracted it (``user_signal``).
Usage disproves staleness, not wrongness. Every reason is written by the
model, which reads content an attacker may control, so a contradiction
overrides only when it cites a fact this pass fetched other than the one it
demotes; any other reason, ``web_contradicted:`` included, stays blocked.

The guard runs where a demotion can still be stopped: at clamp time on the
pass's input (``clamp.clamp_pass_operations``), so a protected demotion takes
no slot of the cap, and in apply on stamps read from the graph right before
the demotions are written, under the pass's lease, since a batch pass's input
is hours old by then. An entity invalidation demotes the entity's neighbours,
and a protected neighbour is left alone the same way. Each demotion dropped is
counted in the pass's ``protected_demotions``.

It only ever removes demotions: whatever the stamps say, a pass demotes no
more than it would with no recall history at all. No recalls means nothing,
since the user may simply have been away, and nothing here or in the prompts
prefers demoting a fact nobody recalled. A read of the stamps that fails
leaves the demotions to what the pass's input said.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable
from datetime import datetime, timedelta, timezone

from graphiti_core.driver.driver import GraphDriver

from backend.copilot.graphiti.recall import USER_FORGET_REASON
from backend.copilot.graphiti.recall_stamp import (
    RecallStamp,
    parse_stamp,
    read_neighbour_stamps,
    read_recall_stamps,
)
from backend.util.settings import Settings

from .schemas import DreamDemotion, EntityInvalidation

logger = logging.getLogger(__name__)

# The reasons that say a fact is wrong, not stale: the user's own retraction
# (the reason a forget records) and a contradiction citing a fact the pass read.
USER_RETRACTION_REASON = USER_FORGET_REASON
CONTRADICTION_PREFIX = "contradicted_by:"


def demotion_protect_window() -> timedelta:
    """How recent a recall protects a fact
    (``Config.dream_demotion_protect_days``); zero protects nothing."""
    return timedelta(days=Settings().config.dream_demotion_protect_days)


def protected_uuids(
    stamps: Iterable[RecallStamp], *, now: datetime, window: timedelta
) -> set[str]:
    """The facts among *stamps* last recalled within *window* before *now*."""
    if window <= timedelta(0):
        return set()
    cutoff = now - window
    return {
        stamp.uuid
        for stamp in stamps
        if (last := parse_stamp(stamp.last_recalled_at)) is not None and last >= cutoff
    }


def overrides_protection(
    reason: str, target_uuid: str, citable: Collection[str]
) -> bool:
    """Whether *reason* demotes *target_uuid* for being wrong rather than
    stale: the user's retraction, or a contradiction citing a fact in
    *citable* (the uuids the pass fetched) other than *target_uuid* itself,
    which is listed to the model and so is the one citation an injected
    reason could always make."""
    if reason == USER_RETRACTION_REASON:
        return True
    if not reason.startswith(CONTRADICTION_PREFIX):
        return False
    cited = reason.removeprefix(CONTRADICTION_PREFIX).strip()
    return cited != target_uuid and cited in citable


def drop_protected(
    demotions: list[DreamDemotion],
    protected: Collection[str],
    citable: Collection[str],
    *,
    where: str,
) -> tuple[list[DreamDemotion], int]:
    """*demotions* without those of a *protected* fact whose reason does not
    override the protection, and how many that was."""
    kept = [
        d
        for d in demotions
        if d.edge_uuid not in protected
        or overrides_protection(d.reason, d.edge_uuid, citable)
    ]
    dropped = len(demotions) - len(kept)
    if dropped:
        logger.info(
            f"{where}: dropped {dropped} demotion(s) of facts recalled within "
            "the protection window"
        )
    return kept, dropped


def guard_at_clamp(
    demotions: list[DreamDemotion],
    facts: Iterable[RecallStamp],
    citable: Collection[str],
    *,
    now: datetime | None = None,
) -> tuple[list[DreamDemotion], int]:
    """The clamp-time guard, on the stamps the pass's input carries."""
    if not demotions:
        return demotions, 0
    protected = protected_uuids(
        facts,
        now=now or datetime.now(timezone.utc),
        window=demotion_protect_window(),
    )
    return drop_protected(demotions, protected, citable, where="Dream clamp")


async def guard_at_apply(
    driver: GraphDriver,
    group_id: str,
    pass_id: str,
    demotions: list[DreamDemotion],
    citable: Collection[str],
) -> tuple[list[DreamDemotion], int]:
    """The apply-time guard: the targets' stamps read from the graph now, so
    a fact recalled after the pass gathered its input is protected too. A
    read that fails keeps *demotions* as the clamp-time guard left them."""
    window = demotion_protect_window()
    targets = [
        d.edge_uuid
        for d in demotions
        if not overrides_protection(d.reason, d.edge_uuid, citable)
    ]
    if not targets or window <= timedelta(0):
        return demotions, 0
    stamps = await read_recall_stamps(driver, group_id, targets)
    if stamps is None:
        logger.warning(
            f"Dream pass {pass_id}: recall stamps could not be read again; its "
            f"{len(demotions)} demotion(s) go ahead on the stamps it gathered"
        )
        return demotions, 0
    protected = protected_uuids(stamps, now=datetime.now(timezone.utc), window=window)
    return drop_protected(
        demotions, protected, citable, where=f"Dream pass {pass_id} apply"
    )


async def protected_neighbours(
    driver: GraphDriver,
    group_id: str,
    pass_id: str,
    invalidation: EntityInvalidation,
    citable: Collection[str],
) -> set[str]:
    """The neighbours of *invalidation*'s entity it must leave alone: recalled
    within the window, read from the graph now, unless its reason overrides
    the protection. A read that fails protects none, as before the stamps."""
    window = demotion_protect_window()
    if window <= timedelta(0):
        return set()
    stamps = await read_neighbour_stamps(driver, group_id, invalidation.entity_uuid)
    if stamps is None:
        logger.warning(
            f"Dream pass {pass_id}: recall stamps around entity "
            f"{invalidation.entity_uuid} could not be read; its invalidation "
            "goes ahead unguarded"
        )
        return set()
    recent = protected_uuids(stamps, now=datetime.now(timezone.utc), window=window)
    protected = {
        uuid
        for uuid in recent
        if not overrides_protection(invalidation.reason, uuid, citable)
    }
    if protected:
        logger.info(
            f"Dream pass {pass_id}: entity {invalidation.entity_uuid}'s "
            f"invalidation leaves {len(protected)} recently recalled fact(s) alone"
        )
    return protected
