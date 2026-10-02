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

from backend.notifications.mailerlite import _pseudonym

logger = logging.getLogger(__name__)


class MarketingConsent(Protocol):
    @property
    def marketing_opt_out_at(self) -> datetime | None: ...


def marketing_allowed(user: MarketingConsent) -> bool:
    return user.marketing_opt_out_at is None


def log_opted_out_skip(email: str, what: str) -> None:
    """At debug level and by pseudonym, so the address never reaches a log."""
    logger.debug(
        "Skipping MailerLite %s for %s: opted out of marketing", what, _pseudonym(email)
    )
