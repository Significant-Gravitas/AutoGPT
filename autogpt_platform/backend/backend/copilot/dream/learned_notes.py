import logging
from collections.abc import Mapping

from backend.api.features.experts.learned_notes_db import (
    LearnedNoteCandidate,
    is_equivalent_learning,
)
from backend.api.features.experts.models import LEARNED_NOTE_TEXT_MAX_LENGTH
from backend.copilot.graphiti.memory_model import MemoryKind
from backend.data.db_accessors import expert_learned_notes_db
from backend.util.feature_flag import Flag, is_feature_enabled

from .schemas import DreamOperations, DreamOperationsSnapshot

logger = logging.getLogger(__name__)

LEARNED_NOTE_MIN_CONFIDENCE = 0.85
MAX_PROMOTIONS_PER_PASS = 5


def select_rule_candidates(
    ops: DreamOperations, snapshot: DreamOperationsSnapshot | None
) -> list[LearnedNoteCandidate]:
    edge_uuids = _edge_uuids_by_content(snapshot)
    candidates: list[LearnedNoteCandidate] = []
    for proposal in ops.proposals:
        if proposal.memory_kind != MemoryKind.rule:
            continue
        if proposal.confidence < LEARNED_NOTE_MIN_CONFIDENCE:
            continue
        citations = {*proposal.source_episode_uuids, *proposal.source_fact_uuids}
        if not citations:
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
    seen = list(existing_texts)
    kept: list[LearnedNoteCandidate] = []
    for candidate in candidates:
        if any(is_equivalent_learning(candidate.text, text) for text in seen):
            continue
        seen.append(candidate.text)
        kept.append(candidate)
    return kept


def invalidated_rule_ids(
    ops: DreamOperations, snapshot: DreamOperationsSnapshot | None
) -> list[str]:
    if snapshot is None:
        return [demotion.edge_uuid for demotion in ops.demotions]
    retired = [
        demotion.edge_uuid for demotion in snapshot.demotions if demotion.applied
    ]
    for invalidation in snapshot.entity_invalidations:
        retired.extend(invalidation.edges_touched)
    return list(dict.fromkeys(retired))


async def promote_pass_learned_notes(
    user_id: str,
    expert_id: str | None,
    ops: DreamOperations,
    apply_stats: Mapping[str, object],
) -> None:
    if expert_id is None:
        return
    try:
        if not await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False):
            return
        raw_snapshot = apply_stats.get("snapshot")
        snapshot = (
            raw_snapshot if isinstance(raw_snapshot, DreamOperationsSnapshot) else None
        )
        raw_session_id = apply_stats.get("session_id")
        source_session_id = raw_session_id if isinstance(raw_session_id, str) else None

        notes_db = expert_learned_notes_db()
        await notes_db.archive_notes_for_rules(
            user_id,
            expert_id,
            invalidated_rule_ids(ops, snapshot),
        )
        existing = await notes_db.list_learned_notes(user_id, expert_id)
        candidates = dedupe_candidates(
            select_rule_candidates(ops, snapshot),
            [note.text for note in existing],
        )[:MAX_PROMOTIONS_PER_PASS]
        await notes_db.promote_learned_notes(
            user_id,
            expert_id,
            [
                candidate.model_copy(update={"source_session_id": source_session_id})
                for candidate in candidates
            ],
        )
    except Exception:
        logger.warning(
            "Expert learned-note promotion failed; dream pass remains successful",
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
