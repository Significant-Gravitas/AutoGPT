"""Promote dream-pass rules into (and demote them out of) learned notes.

The dream pass is the consolidation step of the durable-corrections loop:
Graphiti captures a correction as a ``memory_kind=rule`` memory, the nightly
pass decides which of those rules have held up, and the surviving ones become
``ExpertLearnedNote`` rows that the prompt builder renders as
``<what_ive_learned>``.

"Held up" is deliberately deterministic rather than another LLM judgement —
phase 3 already gated these proposals, so this layer only applies two
mechanical filters on top:

  * ``confidence >= LEARNED_NOTE_MIN_CONFIDENCE`` — the sanitizer's own rating.
  * at least ``LEARNED_NOTE_MIN_CITATIONS`` distinct source uuids — a rule
    corroborated by one passing remark is not yet a standing instruction.

Demotion is the mirror: when Graphiti's temporal model invalidates the edge a
note was promoted from (a demotion, or an entity invalidation that swept it
up), the note is archived so the prompt stops asserting it.

Both directions run behind ``Flag.EXPERT_LEARNED_NOTES`` and are best-effort:
a failure here must never fail a dream pass that already wrote memory.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from backend.api.features.experts.learned_notes_db import LearnedNoteCandidate
from backend.api.features.experts.models import LEARNED_NOTE_TEXT_MAX_LENGTH
from backend.copilot.graphiti.memory_model import MemoryKind
from backend.data.db_accessors import expert_learned_notes_db
from backend.util.feature_flag import Flag, is_feature_enabled

from .schemas import DreamOperations, DreamOperationsSnapshot

logger = logging.getLogger(__name__)

# The sanitizer's self-rated confidence floor for a proposal to graduate from
# "tentative finding in the graph" to "sentence in every future prompt". Set
# well above the write threshold: a wrong learned note is far more expensive
# than a wrong graph edge, because the model reads it on every single turn.
LEARNED_NOTE_MIN_CONFIDENCE = 0.85

# Distinct source episode/fact uuids a rule must cite. One citation means the
# user said it once; two means the pass saw it corroborated.
LEARNED_NOTE_MIN_CITATIONS = 2

# Per-pass promotion ceiling. The active set is capped at 20 (see
# ``MAX_ACTIVE_LEARNED_NOTES``); without a per-pass cap a single enthusiastic
# night could churn the entire block.
MAX_PROMOTIONS_PER_PASS = 5

# Near-duplicate threshold for "this note already says that". Mirrors the
# orchestrator's intra-pass write dedupe intent but is kept local: the
# orchestrator imports this module, so borrowing its private helpers would
# make the import cycle.
_DEDUP_JACCARD_THRESHOLD = 0.7
_DEDUP_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "for",
        "on",
        "in",
        "at",
        "by",
        "with",
        "that",
        "this",
        "it",
        "as",
        "be",
        "has",
        "have",
        "had",
        "s",
        "user",
        "users",
        "always",
        "should",
    }
)


def _content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _DEDUP_STOPWORDS
    )


def _is_near_duplicate(a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    union = len(a | b)
    return bool(union) and len(a & b) / union >= _DEDUP_JACCARD_THRESHOLD


def select_rule_candidates(
    ops: DreamOperations, snapshot: DreamOperationsSnapshot | None
) -> list[LearnedNoteCandidate]:
    """Rules from this pass that are stable enough to become learned notes.

    ``snapshot`` supplies the durable Graphiti edge uuid apply.py minted for
    each proposal; a proposal whose edge uuid is unknown is still promotable
    (the note is simply not linkable back to an edge), because losing the note
    is worse than losing its provenance.
    """
    edge_uuids = _edge_uuids_by_content(snapshot)
    candidates: list[LearnedNoteCandidate] = []
    for proposal in ops.proposals:
        if proposal.memory_kind != MemoryKind.rule:
            continue
        if proposal.confidence < LEARNED_NOTE_MIN_CONFIDENCE:
            continue
        citations = {*proposal.source_episode_uuids, *proposal.source_fact_uuids}
        if len(citations) < LEARNED_NOTE_MIN_CITATIONS:
            continue
        text = proposal.content.strip()
        if not text:
            continue
        candidates.append(
            LearnedNoteCandidate(
                text=text[:LEARNED_NOTE_TEXT_MAX_LENGTH],
                source_rule_id=edge_uuids.get(proposal.content),
            )
        )
    return candidates


def dedupe_candidates(
    candidates: list[LearnedNoteCandidate], existing_texts: list[str]
) -> list[LearnedNoteCandidate]:
    """Drop candidates that restate an existing note or an earlier candidate.

    The same standing instruction gets re-derived on consecutive nights (the
    evidence for it is still in the 14-day window), so without this every pass
    would push the previous notes out of the active cap with paraphrases of
    themselves.
    """
    seen = [_content_tokens(text) for text in existing_texts]
    kept: list[LearnedNoteCandidate] = []
    for candidate in candidates:
        tokens = _content_tokens(candidate.text)
        if any(_is_near_duplicate(tokens, other) for other in seen):
            continue
        seen.append(tokens)
        kept.append(candidate)
    return kept


def invalidated_rule_ids(
    ops: DreamOperations, snapshot: DreamOperationsSnapshot | None
) -> list[str]:
    """Graphiti edge uuids this pass retired, in note-archival order.

    Reads the snapshot when apply.py produced one so only demotions that
    actually touched an edge count; falls back to the requested operations
    otherwise. Archiving a note for an edge that no longer exists is harmless
    (``archive_notes_for_rules`` matches nothing), so the fallback is safe.
    """
    if snapshot is None:
        return [d.edge_uuid for d in ops.demotions]
    retired = [d.edge_uuid for d in snapshot.demotions if d.applied]
    for invalidation in snapshot.entity_invalidations:
        retired.extend(invalidation.edges_touched)
    return list(dict.fromkeys(retired))


async def promote_pass_learned_notes(
    user_id: str,
    expert_id: str | None,
    ops: DreamOperations,
    apply_stats: Mapping[str, object],
) -> None:
    """Run promotion + demotion for one applied dream pass.

    Never raises: the pass has already written to Graphiti and Postgres by the
    time this runs, so a learned-notes failure must degrade to "no notes
    changed tonight" rather than turning a successful pass into an error.
    """
    try:
        if not await is_feature_enabled(
            Flag.EXPERT_LEARNED_NOTES, user_id, default=False
        ):
            return
        raw_snapshot = apply_stats.get("snapshot")
        snapshot = (
            raw_snapshot if isinstance(raw_snapshot, DreamOperationsSnapshot) else None
        )
        raw_session_id = apply_stats.get("session_id")
        session_id = raw_session_id if isinstance(raw_session_id, str) else None

        notes_db = expert_learned_notes_db()

        archived = await notes_db.archive_notes_for_rules(
            user_id, expert_id, invalidated_rule_ids(ops, snapshot)
        )

        # Demote first, then read: a note archived above must not count as an
        # existing note that blocks a fresh promotion of the same rule.
        existing = await notes_db.list_learned_notes(user_id, expert_id)
        deduped = dedupe_candidates(
            select_rule_candidates(ops, snapshot), [note.text for note in existing]
        )[:MAX_PROMOTIONS_PER_PASS]
        candidates = [
            c.model_copy(update={"source_session_id": session_id}) for c in deduped
        ]

        created = await notes_db.promote_learned_notes(user_id, expert_id, candidates)
        if created or archived:
            logger.info(
                "Dream learned notes for user %s: promoted %d, archived %d",
                user_id[:12],
                len(created),
                archived,
            )
    except Exception:
        logger.warning(
            "Learned-note promotion failed for user %s — pass unaffected",
            user_id[:12],
            exc_info=True,
        )


def _edge_uuids_by_content(
    snapshot: DreamOperationsSnapshot | None,
) -> dict[str, str]:
    if snapshot is None:
        return {}
    return {
        written.content: written.edge_uuid
        for written in snapshot.proposals
        if written.edge_uuid
    }
