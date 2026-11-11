from typing import cast

import pydantic
from prisma.models import FeedbackPrompt

from backend.data.feedback_prompts import FeedbackPromptKind

MAX_NOTE_CHARS = 2000
MAX_REASON_CHARS = 64


class ClaimFeedbackPromptRequest(pydantic.BaseModel):
    kind: FeedbackPromptKind


class ClaimFeedbackPromptResponse(pydantic.BaseModel):
    id: str | None = pydantic.Field(
        description="The showing's id, or null when the prompt must not appear.",
    )


class UpdateFeedbackPromptRequest(pydantic.BaseModel):
    reason: str | None = pydantic.Field(default=None, max_length=MAX_REASON_CHARS)
    note: str | None = pydantic.Field(default=None, max_length=MAX_NOTE_CHARS)
    score: int | None = pydantic.Field(default=None, ge=0, le=10)
    dismissed: bool = False

    @pydantic.field_validator("note")
    @classmethod
    def blank_note_is_none(cls, note: str | None) -> str | None:
        if note is None:
            return None
        return note.strip() or None

    @property
    def has_answer(self) -> bool:
        """Whether this carries anything but a dismissal."""
        return (
            self.reason is not None or self.note is not None or self.score is not None
        )


class FeedbackPromptState(pydantic.BaseModel):
    id: str
    kind: FeedbackPromptKind
    answered: bool
    dismissed: bool
    reason: str | None
    score: int | None

    @classmethod
    def from_row(cls, row: FeedbackPrompt) -> "FeedbackPromptState":
        return cls(
            id=row.id,
            kind=cast(FeedbackPromptKind, row.kind),
            answered=row.answeredAt is not None,
            dismissed=row.dismissedAt is not None,
            reason=row.reason,
            score=row.score,
        )
