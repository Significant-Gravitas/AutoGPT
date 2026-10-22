"""PostHog events for what Postmark reports about an email we sent.

The webhook handler finds the user the email went to and calls
:func:`track_email_event`. The recipient address itself never reaches PostHog:
the user id is the distinct id, and a clicked link keeps only its scheme, host
and path, with any path segment that isn't a plain lower-case word replaced
by ``:id``.
"""

import hashlib
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

# The path segments a link keeps: lower-case words and slugs. Anything else
# (an email address, a capitalised or percent-encoded token) becomes ``:id``.
_PLAIN_SEGMENT_RE = re.compile(r"[a-z0-9._-]*")
# A plain segment this long with a digit in it is an id or a code (a UUID),
# and one longer still is a token whatever it holds.
_ID_LIKE_RE = re.compile(r"(?=.*\d).{16,}")
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

    Keyed on the message, the record type and that time, plus a hash of the
    link for a click, so a webhook Postmark redelivers is merged with the
    first one rather than counted twice.
    """
    dedup_key = f"{webhook.MessageID}:{webhook.RecordType}:{occurred_at}"
    if isinstance(webhook, PostmarkClickWebhook):
        link_hash = hashlib.sha256(webhook.OriginalLink.encode()).hexdigest()
        dedup_key = f"{dedup_key}:{link_hash[:16]}"
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
        dedup_key=dedup_key,
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
    carry a secret: no query string, fragment, credentials, ids or any path
    segment that isn't a plain lower-case word.

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
        segment if _is_plain(segment) else ID_SEGMENT
        for segment in parts.path.split("/")
    )
    return f"{scheme}://{netloc}{path}"


def _is_plain(segment: str) -> bool:
    return (
        bool(_PLAIN_SEGMENT_RE.fullmatch(segment))
        and len(segment) < _ID_LIKE_MIN_ANY
        and not _ID_LIKE_RE.fullmatch(segment)
    )
