import hashlib
import logging
import re

import prisma.enums
import prisma.errors
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
        "user",
        "users",
        "always",
        "should",
    }
)


class LearnedNoteCandidate(BaseModel):
    text: str
    source_rule_id: str | None = None
    source_session_id: str | None = None


def content_tokens(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _DEDUP_STOPWORDS
    )


def is_equivalent_learning(first: str, second: str) -> bool:
    left = content_tokens(first)
    right = content_tokens(second)
    if not left or not right:
        return first.strip().casefold() == second.strip().casefold()
    return len(left & right) / len(left | right) >= _DEDUP_JACCARD_THRESHOLD


def _dedupe_key(text: str) -> str:
    normalized = " ".join(sorted(content_tokens(text))) or text.strip().casefold()
    return hashlib.sha256(normalized.encode()).hexdigest()


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
    expert_id: str,
    *,
    include_archived: bool = False,
    limit: int = MAX_ACTIVE_LEARNED_NOTES,
) -> list[ExpertLearnedNote]:
    where = {
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
    user_id: str, note_id: str, expert_id: str
) -> ExpertLearnedNote:
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
        where={"id": note_id, "userId": user_id, "expertId": expert_id}
    )
    if row is None:
        raise ExpertNotFoundError(note_id)
    return _to_model(row)


async def archive_notes_for_rules(
    user_id: str, expert_id: str, rule_ids: list[str]
) -> int:
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
    expert_id: str,
    candidates: list[LearnedNoteCandidate],
) -> list[ExpertLearnedNote]:
    if not candidates:
        return []
    if (
        await prisma.models.Expert.prisma().count(
            where={
                "id": expert_id,
                "ownerUserId": user_id,
                "isTemplate": False,
                "isArchived": False,
            }
        )
        == 0
    ):
        raise ExpertNotFoundError(expert_id)

    existing = await list_learned_notes(user_id, expert_id)
    seen_texts = [note.text for note in existing]
    seen_rule_ids = {
        note.source_rule_id for note in existing if note.source_rule_id is not None
    }
    created: list[ExpertLearnedNote] = []
    for candidate in candidates:
        text = candidate.text.strip()[:LEARNED_NOTE_TEXT_MAX_LENGTH]
        if not text or candidate.source_rule_id in seen_rule_ids:
            continue
        if any(is_equivalent_learning(text, known) for known in seen_texts):
            continue
        try:
            row = await prisma.models.ExpertLearnedNote.prisma().create(
                data={
                    "userId": user_id,
                    "expertId": expert_id,
                    "text": text,
                    "dedupeKey": _dedupe_key(text),
                    "sourceRuleId": candidate.source_rule_id,
                    "sourceSessionId": candidate.source_session_id,
                }
            )
        except prisma.errors.UniqueViolationError:
            continue
        created.append(_to_model(row))
        seen_texts.append(text)
        if candidate.source_rule_id:
            seen_rule_ids.add(candidate.source_rule_id)
    await _trim_to_cap(user_id, expert_id)
    return created


async def _trim_to_cap(user_id: str, expert_id: str) -> None:
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
    await prisma.models.ExpertLearnedNote.prisma().update_many(
        where={"id": {"in": [row.id for row in stale]}, "userId": user_id},
        data={"status": prisma.enums.ExpertLearnedNoteStatus.ARCHIVED},
    )
