"""Persistence for expert "what I've learned" notes.

The notes are machine-curated: the nightly dream pass promotes stable,
high-confidence Graphiti rules into this table, and the prompt builder renders
the ACTIVE set as ``<what_ive_learned>``. Nothing here ever touches the
user-authored ``identity`` / ``boundaries`` columns.

Every function is owner-scoped by ``userId`` — an expert id alone is never
enough to read or write a note.
"""

import logging

import prisma.enums
import prisma.models
import prisma.types
from pydantic import BaseModel

from backend.api.features.experts.models import (
    LEARNED_NOTE_TEXT_MAX_LENGTH,
    MAX_ACTIVE_LEARNED_NOTES,
    ExpertLearnedNote,
)
from backend.util.exceptions import ExpertNotFoundError

logger = logging.getLogger(__name__)


class LearnedNoteCandidate(BaseModel):
    """A rule the dream pass wants promoted into a learned note."""

    text: str
    source_rule_id: str | None = None
    source_session_id: str | None = None


def _to_model(row: prisma.models.ExpertLearnedNote) -> ExpertLearnedNote:
    return ExpertLearnedNote(
        id=row.id,
        expert_id=row.expertId,
        text=row.text,
        learned_at=row.learnedAt,
        source_session_id=row.sourceSessionId,
        source_rule_id=row.sourceRuleId,
        status=(
            "archived"
            if row.status == prisma.enums.ExpertLearnedNoteStatus.ARCHIVED
            else "active"
        ),
    )


async def list_learned_notes(
    user_id: str,
    expert_id: str | None = None,
    *,
    include_archived: bool = False,
    limit: int = MAX_ACTIVE_LEARNED_NOTES,
) -> list[ExpertLearnedNote]:
    """Notes for one scope, newest first.

    ``expert_id=None`` is the base AutoPilot scope, not "any expert" — the
    filter is an explicit null match so one expert's notes can never leak into
    another's prompt.
    """
    where: prisma.types.ExpertLearnedNoteWhereInput = {
        "userId": user_id,
        "expertId": expert_id,
    }
    if not include_archived:
        where["status"] = prisma.enums.ExpertLearnedNoteStatus.ACTIVE
    rows = await prisma.models.ExpertLearnedNote.prisma().find_many(
        where=where,
        order={"learnedAt": "desc"},
        take=limit,
    )
    return [_to_model(row) for row in rows]


async def archive_learned_note(
    user_id: str, note_id: str, expert_id: str | None = None
) -> ExpertLearnedNote:
    """Soft-delete one note and return it, so the caller can invalidate the
    Graphiti rule it came from.

    Raises :class:`ExpertNotFoundError` when the note is not an active note of
    this owner in this scope — an already-archived note is not re-archived, so
    a double click cannot re-fire the Graphiti invalidation.
    """
    updated = await prisma.models.ExpertLearnedNote.prisma().update_many(
        where={
            "id": note_id,
            "userId": user_id,
            "expertId": expert_id,
            "status": prisma.enums.ExpertLearnedNoteStatus.ACTIVE,
        },
        data={"status": prisma.enums.ExpertLearnedNoteStatus.ARCHIVED},
    )
    if updated == 0:
        raise ExpertNotFoundError(note_id)
    row = await prisma.models.ExpertLearnedNote.prisma().find_first(
        where={"id": note_id, "userId": user_id}
    )
    if row is None:
        raise ExpertNotFoundError(note_id)
    return _to_model(row)


async def archive_notes_for_rules(
    user_id: str, expert_id: str | None, rule_ids: list[str]
) -> int:
    """Archive every active note whose source rule was invalidated upstream.

    Called by the dream pass when Graphiti's temporal model demoted the edge a
    note was promoted from: the note has to stop being injected, but stays for
    audit rather than being deleted.
    """
    if not rule_ids:
        return 0
    return await prisma.models.ExpertLearnedNote.prisma().update_many(
        where={
            "userId": user_id,
            "expertId": expert_id,
            "sourceRuleId": {"in": rule_ids},
            "status": prisma.enums.ExpertLearnedNoteStatus.ACTIVE,
        },
        data={"status": prisma.enums.ExpertLearnedNoteStatus.ARCHIVED},
    )


async def promote_learned_notes(
    user_id: str,
    expert_id: str | None,
    candidates: list[LearnedNoteCandidate],
) -> list[ExpertLearnedNote]:
    """Insert new notes, then trim the ACTIVE set back to the cap.

    Candidates are expected to be text-deduped against the existing notes by
    the caller (``dream.learned_notes``); the ``source_rule_id`` filter here is
    the idempotency backstop that makes a re-dispatched dream pass a no-op.
    """
    if not candidates:
        return []

    known_rule_ids = await _existing_rule_ids(
        user_id, expert_id, [c.source_rule_id for c in candidates if c.source_rule_id]
    )
    fresh = [
        c
        for c in candidates
        if c.source_rule_id is None or c.source_rule_id not in known_rule_ids
    ]
    if not fresh:
        return []

    created = [
        _to_model(
            await prisma.models.ExpertLearnedNote.prisma().create(
                data={
                    "userId": user_id,
                    "expertId": expert_id,
                    "text": candidate.text[:LEARNED_NOTE_TEXT_MAX_LENGTH],
                    "sourceRuleId": candidate.source_rule_id,
                    "sourceSessionId": candidate.source_session_id,
                }
            )
        )
        for candidate in fresh
    ]
    await _trim_to_cap(user_id, expert_id)
    return created


async def _existing_rule_ids(
    user_id: str, expert_id: str | None, rule_ids: list[str]
) -> set[str]:
    if not rule_ids:
        return set()
    rows = await prisma.models.ExpertLearnedNote.prisma().find_many(
        where={
            "userId": user_id,
            "expertId": expert_id,
            "sourceRuleId": {"in": rule_ids},
        }
    )
    return {row.sourceRuleId for row in rows if row.sourceRuleId}


async def _trim_to_cap(user_id: str, expert_id: str | None) -> None:
    """Archive the oldest ACTIVE notes past ``MAX_ACTIVE_LEARNED_NOTES``.

    The prompt block is only worth having while it stays small, so the cap is
    enforced on write rather than on read — a stale reader can never see more
    notes than the prompt does.
    """
    stale = await prisma.models.ExpertLearnedNote.prisma().find_many(
        where={
            "userId": user_id,
            "expertId": expert_id,
            "status": prisma.enums.ExpertLearnedNoteStatus.ACTIVE,
        },
        order={"learnedAt": "desc"},
        skip=MAX_ACTIVE_LEARNED_NOTES,
    )
    if not stale:
        return
    archived = await prisma.models.ExpertLearnedNote.prisma().update_many(
        where={"id": {"in": [row.id for row in stale]}, "userId": user_id},
        data={"status": prisma.enums.ExpertLearnedNoteStatus.ARCHIVED},
    )
    logger.info(
        "Archived %d learned note(s) past the active cap for user %s",
        archived,
        user_id[:12],
    )
