"""Whether an account may exist in MailerLite at all.

Someone who refused marketing (on the signup page; later, by unsubscribing from
an email or in settings) is never written to MailerLite: no subscriber, no
group, no fields, and no "opted out" placeholder either. Every MailerLite write
upserts the subscriber, a group removal or a field update included (see
`mailerlite.py`), so every path that queues an audience change asks
`marketing_allowed` first, and the backfills leave such accounts out of their
plan. Billing and account emails are service mail and are not affected.
"""

import logging
from datetime import datetime
from typing import Protocol

from backend.data.notifications import AudienceAction
from backend.notifications.mailerlite import pseudonym

logger = logging.getLogger(__name__)

# Never a stored value, so a missing attribute cannot pass for an unset one.
_MISSING = object()


class MarketingConsent(Protocol):
    @property
    def marketing_opt_out_at(self) -> datetime | None: ...


class MarketingContact(MarketingConsent, Protocol):
    """An account an audience change would be queued for."""

    @property
    def id(self) -> str: ...

    @property
    def email(self) -> str: ...


def marketing_allowed(user: MarketingConsent) -> bool:
    # Shared-cache entries can outlive a rolling deploy. A user pickled by the
    # previous version has no `marketing_opt_out_at`, so read it defensively
    # until that cache entry expires. Its consent is unknown, so this fails
    # closed without raising: only the MailerLite change is skipped, never the
    # caller's own work (a trial notice), and the backfills catch it up.
    return getattr(user, "marketing_opt_out_at", _MISSING) is None


def audience_change_allowed(user: MarketingContact, action: AudienceAction) -> bool:
    """Whether `action` may be queued for this account. A refusal is logged."""
    if marketing_allowed(user):
        return True
    log_opted_out_skip(user.email, action.value)
    return False


def log_opted_out_skip(email: str, what: str) -> None:
    """At debug level and by pseudonym, so the address never reaches a log."""
    logger.debug(
        "Skipping MailerLite %s for %s: opted out of marketing", what, pseudonym(email)
    )
