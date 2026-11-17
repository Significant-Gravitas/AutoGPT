"""The copy of one account email, which the auth email templates lay out."""

from pydantic import BaseModel


class AuthEmailContent(BaseModel):
    subject: str
    eyebrow: str
    headline: str
    intro: str
    preheader: str
    # The button label. Without one the email has no button, no link and no
    # "If you didn't request this" line.
    action: str | None = None
    reason: str
    # Must match the links the frontend issues.
    expiry: str | None = None
    # What the recipient may not expect from following the link. Without a
    # link it ends the body paragraph instead.
    note: str | None = None
    # Close with the shared inbox (which also takes replies) instead of
    # Discord, for an email about the person's own account.
    close_with_inbox: bool = False
    # (lead, rest) progress steps, for the templates that show them.
    steps: list[tuple[str, str]] = []
