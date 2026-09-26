"""Hard limits on what one dream pass may write, whatever the sanitizer
emitted.

The phase 3 prompt asks for these caps, but the model can still over-emit, so
both routes clamp its ``DreamOperations`` in code before apply runs: the sync
orchestrator and the batch callbacks call ``clamp_operations``. On the way it
drops "transient intent" (questions captured as facts) and collapses
near-duplicate writes (``dedup.py``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TypeVar

from .dedup import dedupe_near_duplicate_writes
from .prompts import MAX_DEMOTIONS_PER_PASS, MAX_PROPOSALS_PER_PASS, MAX_WRITES_PER_PASS
from .schemas import ConsolidatedFact, DreamOperations, ProposedFinding

logger = logging.getLogger(__name__)


# Entity invalidation is the most destructive op the sanitizer can emit:
# each one single-hop demotes EVERY :RELATES_TO edge on the entity, so a
# hub entity multiplies the blast radius far past the demotion caps. We
# cap the *count* of invalidations per pass here; the per-entity edge
# blast radius is bounded by the ``DREAM_PASS_INVALIDATE_ENTITY`` LD flag
# (staged rollout, off by default) plus the single-hop-only guarantee in
# ``invalidate_entity_direct_neighbors`` (no multi-hop propagation).
# Deliberately NOT degree-aware — fetching entity degrees is async graph
# work that doesn't belong in this sync clamp.
MAX_ENTITY_INVALIDATIONS_PER_PASS = 2


# High-precision filter for "transient intent" facts — content that
# records what the user is ASKING/wants to KNOW rather than a durable
# fact about them. The sanitize prompt is told to drop these, but
# prompt-only sanitization leaks them (#13388: "User is asking how
# Kubernetes works", "User is interested in knowing which PRs are open"),
# so this is a deterministic belt-and-suspenders gate.
#
# Deliberately NARROW to knowledge-seeking intent. We do NOT match goals
# like "user wants to create/build X" — those are legitimate durable
# memories. Generic world-knowledge pollution ("Kubernetes uses pods…")
# is left to the sanitize prompt: it needs LLM judgment (is the subject
# the user?) that a regex can't do without false-positives.
#
# Interrogative complements are the sharp edge — several verbs are durable
# aspirations on their own but transient questions once they take a
# question word:
#   * ``asking`` — "asking FOR weekly reports" / "asking the agent TO
#     monitor PRs" are durable requests (semantically like "wants X"), so
#     only interrogative ``asking HOW/WHAT/…`` counts as transient.
#   * ``learn``/``understand``/``find out`` — "wants to learn Spanish",
#     "wants to understand distributed systems", "wants to find out about
#     new markets" are durable skill/aspiration GOALS; they only read as
#     transient with an interrogative ("wants to learn HOW X works").
#   * ``curious`` — "curious ABOUT X" is transient, but "curious by nature"
#     is a durable personality trait, so a ``curious about`` complement is
#     required (mirroring ``confused about``/``unsure about``).
# ``know`` stays complement-free: "wants to know X" is transient curiosity
# regardless of phrasing (there is no durable "wants to know" aspiration —
# that role is served by ``learn``).
# Known limitation (nice-to-have, low frequency): a standing notification
# preference phrased "wants to know when X happens" is dropped; separating
# it from a one-off "wants to know when the deploy is" isn't reliably
# regex-able, so it's left to the sanitize prompt + human review.
#
# Subject scope: the gate deliberately anchors on the generic ``user``
# subject only. Name-phrased transient intent ("Nick is asking how the
# auth flow works") is intentionally NOT matched here — broadening the
# subject to arbitrary proper nouns would risk false-positives on
# non-user entities ("Kubernetes is asking for more nodes"), so
# name-first phrasing is left to the sanitize prompt's LLM judgment. The
# leading auxiliary allows perfect-progressive forms ("has been asking").
_TRANSIENT_INTENT_RE = re.compile(
    r"^(the\s+)?user\s+(?:(?:is|has|was)\s+(?:been\s+)?)?"
    r"(asking\s+(how|what|why|whether|if|when|where|who|which|about)\b"
    r"|wondering\b"
    r"|curious\s+about\b"
    r"|confused\s+about\b"
    r"|unsure\s+about\b"
    r"|trying\s+to\s+understand\b"
    r"|interested\s+in\s+(knowing|understanding)\b"
    r"|interested\s+in\s+learning\s+(how|what|why|whether|if|when|where)\b"
    r"|wants?\s+to\s+know\b"
    r"|wants?\s+to\s+(understand|find\s+out)\s+(how|what|why|whether|if|when|where)\b"
    r"|wants?\s+to\s+learn\s+(how|what|why|whether|if|when|where)\b"
    r"|asked\s+(how|what|why|whether|if|when|where|about)\b)",
    re.IGNORECASE,
)


def _is_transient_intent(content: str) -> bool:
    """True when ``content`` reads as a question/knowledge-seeking intent
    rather than a durable fact about the user."""
    return bool(_TRANSIENT_INTENT_RE.match(content.strip()))


_ContentItem = TypeVar("_ContentItem", ConsolidatedFact, ProposedFinding)


def _drop_transient_intent(
    items: Sequence[_ContentItem],
) -> tuple[list[_ContentItem], int]:
    """Filter ConsolidatedFact / ProposedFinding items whose ``.content``
    is a transient intent. Returns (kept, dropped_count)."""
    kept = [it for it in items if not _is_transient_intent(it.content)]
    return kept, len(items) - len(kept)


def clamp_operations(
    ops: DreamOperations,
    active_fact_count: int,
    known_fact_uuids: set[str] | None = None,
) -> DreamOperations:
    """Hard-trim oversized phase 3 outputs.

    Phase 3's prompt asks for these caps but the model can still
    over-emit. The orchestrator enforces them in code so apply.py
    never writes more than the policy allows.

    Demotions carry a second ceiling — 5% of the active fact set — so a
    single pass can never wipe a meaningful fraction of a user's memory
    even if the absolute ``MAX_DEMOTIONS_PER_PASS`` cap would allow it.
    The 5% ceiling floors at 1 when there is at least one active fact:
    small graphs (< 20 facts) would otherwise round to a cap of 0 and
    never get even a single contradicted fact demoted.
    ``active_fact_count < 0`` means the count is unknown (the batch path
    lost its persisted input bundle); fall back to the absolute cap only
    rather than silently dropping every demotion.

    When ``known_fact_uuids`` is provided, demotions targeting uuids
    outside it are dropped BEFORE the cap slice — otherwise a
    hallucinated uuid at the head of the model's list consumes a cap
    slot (the entire budget on a floor-of-1 small graph) and displaces
    a valid demotion that apply.py would have accepted. ``None`` skips
    the pre-filter; apply.py's idempotent known-uuid filter remains the
    security chokepoint either way.

    Entity invalidations are count-capped at
    ``MAX_ENTITY_INVALIDATIONS_PER_PASS``; see the constant's comment
    for why the per-entity edge blast radius is bounded elsewhere (LD
    flag + single-hop guarantee), not here.
    """
    demotions = ops.demotions
    if known_fact_uuids is not None:
        demotions = [d for d in demotions if d.edge_uuid in known_fact_uuids]
        dropped = len(ops.demotions) - len(demotions)
        if dropped:
            logger.warning(
                "Dream clamp: dropped %d demotion(s) targeting edge uuids "
                "outside known_fact_uuids before applying the demotion cap",
                dropped,
            )
    demotion_cap = MAX_DEMOTIONS_PER_PASS
    if active_fact_count == 0:
        demotion_cap = 0
    elif active_fact_count > 0:
        demotion_cap = min(MAX_DEMOTIONS_PER_PASS, max(1, active_fact_count * 5 // 100))

    # Drop transient-intent pollution ("user is asking…") before the cap
    # slice so a leaked question never displaces a real fact (#13388).
    writes, w_intent_dropped = _drop_transient_intent(ops.writes)
    proposals, p_intent_dropped = _drop_transient_intent(ops.proposals)
    if w_intent_dropped or p_intent_dropped:
        logger.info(
            "Dream clamp: dropped %d transient-intent write(s) and %d "
            "proposal(s) (questions captured as facts)",
            w_intent_dropped,
            p_intent_dropped,
        )
    # Collapse intra-pass near-duplicate writes before the cap slice so the
    # cap counts distinct facts, not paraphrases of one (#13387).
    writes, w_dup_dropped = dedupe_near_duplicate_writes(writes)
    if w_dup_dropped:
        logger.info(
            "Dream clamp: collapsed %d near-duplicate write(s) into their "
            "canonical (longest) phrasing",
            w_dup_dropped,
        )
    return DreamOperations(
        writes=writes[:MAX_WRITES_PER_PASS],
        proposals=proposals[:MAX_PROPOSALS_PER_PASS],
        demotions=demotions[:demotion_cap],
        entity_invalidations=ops.entity_invalidations[
            :MAX_ENTITY_INVALIDATIONS_PER_PASS
        ],
        summary_for_user=ops.summary_for_user,
    )
