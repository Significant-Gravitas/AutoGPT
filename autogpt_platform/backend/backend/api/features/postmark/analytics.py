"""PostHog events for what Postmark reports about an email we sent.

The webhook handler finds the user the email went to and calls
:func:`track_email_event`. The recipient address itself never reaches PostHog:
the user id is the distinct id, and a clicked link keeps only its scheme, host
and path, with id- and token-like path segments replaced by ``:id``.
"""

import logging
import re
from datetime import datetime
from urllib.parse import urlsplit

from backend.util.posthog_events import PostHogEvent
from backend.util.product_analytics import track

from .models import (
    PostmarkBounceWebhook,
    PostmarkClickWebhook,
    PostmarkDeliveryWebhook,
    PostmarkOpenWebhook,
)

logger = logging.getLogger(__name__)

EmailWebhook = (
    PostmarkDeliveryWebhook
    | PostmarkOpenWebhook
    | PostmarkClickWebhook
    | PostmarkBounceWebhook
)

ID_SEGMENT = ":id"

# A path segment this long with a digit or a capital in it is an id, a code or
# a token (a UUID, a reset token), not a word or a slug.
_ID_LIKE_RE = re.compile(r"(?=.*[\dA-Z]).{16,}")
_ID_LIKE_MIN_ANY = 32


def track_email_event(
    user_id: str,
    event: PostHogEvent,
    webhook: EmailWebhook,
    *,
    occurred_at: str,
    properties: dict[str, object] | None = None,
) -> None:
    """Send *event* for one Postmark webhook, at the time Postmark reports.

    Keyed on the message, the record type and that time, so a webhook Postmark
    redelivers is merged with the first one rather than counted twice.
    """
    track(
        user_id,
        event,
        {
            # The tag the email was sent with: its notification type.
            "email_type": webhook.Tag or None,
            "message_stream": webhook.MessageStream,
            "message_id": webhook.MessageID,
            **(properties or {}),
        },
        dedup_key=f"{webhook.MessageID}:{webhook.RecordType}:{occurred_at}",
        timestamp=parse_postmark_time(occurred_at),
    )


def parse_postmark_time(value: str) -> datetime | None:
    """Postmark's ISO 8601 times (``2019-11-05T16:33:54.9070259Z``), or None
    when one can't be read, so the event takes the time it was received."""
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning("Unreadable Postmark timestamp %r", value)
        return None


def safe_link(url: str) -> str | None:
    """The clicked link without anything that could identify a person or
    carry a secret: no query string, fragment, credentials or ids.

    Only web links keep a host and path; any other scheme keeps just its
    name, since a ``mailto:`` link's path is an email address.
    """
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return f"{scheme}:" if scheme else None
    if not host:
        return None
    netloc = f"{host}:{port}" if port else host
    path = "/".join(
        ID_SEGMENT if _is_id_like(segment) else segment
        for segment in parts.path.split("/")
    )
    return f"{scheme}://{netloc}{path}"


def _is_id_like(segment: str) -> bool:
    return len(segment) >= _ID_LIKE_MIN_ANY or bool(_ID_LIKE_RE.fullmatch(segment))
