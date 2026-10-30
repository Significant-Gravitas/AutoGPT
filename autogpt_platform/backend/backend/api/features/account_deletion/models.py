"""Request and response bodies for self-serve account deletion (SECRT-2877)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FEEDBACK_TEXT_MAX_CHARS = 1000

DeletionSource = Literal["settings", "paywall"]
ExitReason = Literal[
    "privacy",
    "cost",
    "not_using",
    "missing_capability",
    "limits",
    "other",
    "skipped",
]
Alternative = Literal["manage_apps", "disconnect_all", "billing", "keep_account"]


class AccountDeletionRequestBody(BaseModel):
    confirm_email: str = Field(max_length=320)
    source: DeletionSource


class AccountDeletionResponse(BaseModel):
    """Whether the account is closed and waiting to be erased. While it is,
    ``erase_after`` says when the erasure may start: until then its owner can
    keep the account."""

    status: Literal["open", "scheduled"]
    erase_after: datetime | None = None


class DeletionFeedbackRequest(BaseModel):
    """One open of the Delete account dialog (with the surface it was opened
    from), one answer to "Why are you leaving?", or one click on the
    alternative it offered. Nothing here identifies the person; the route adds
    no identity either."""

    kind: Literal["opened", "answer", "alternative_clicked"]
    reason: ExitReason | None = None
    text: str | None = None
    alternative: Alternative | None = None
    surface: DeletionSource | None = None

    @field_validator("text")
    @classmethod
    def _trim_and_cap(cls, text: str | None) -> str | None:
        if text is None:
            return None
        trimmed = text.strip()[:FEEDBACK_TEXT_MAX_CHARS].strip()
        return trimmed or None

    @model_validator(mode="after")
    def _match_kind(self) -> "DeletionFeedbackRequest":
        if self.kind == "opened":
            if self.surface is None:
                raise ValueError("opened needs a surface")
            if self.reason or self.text or self.alternative:
                raise ValueError("an open carries only its surface")
            return self
        if self.surface is not None:
            raise ValueError("only an open carries a surface")
        if self.kind == "alternative_clicked" and self.alternative is None:
            raise ValueError("alternative_clicked needs an alternative")
        if self.kind == "answer" and self.alternative is not None:
            raise ValueError("an answer carries no alternative")
        return self
