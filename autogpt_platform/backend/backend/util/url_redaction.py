"""Masking for credentials that travel in request URLs.

The browser opens its WebSocket with the user's session JWT in ``?token=``,
email links carry one-click tokens in the query, and some path segments are
bearer secrets themselves: whoever holds one can post to a webhook, claim a
platform link, read a shared chat or run, or accept an org invitation. uvicorn
logs the path and query of every request and Sentry attaches both to every
event, so anything that records a request URL masks it here first.
"""

import logging
import re

REDACTED = "[REDACTED]"

_CREDENTIAL_PATH_SEGMENT = re.compile(
    r"(/api/(?:integrations/(?:[^/?#\s]+/)?webhooks|platform-linking/(?:user-)?tokens"
    r"|public/shared(?:/chats)?|invitations)/)[^/?#\s\"']+"
)
# The query runs to the end of its URL, which may sit inside a longer line.
_QUERY_STRING = re.compile(r"\?([^\s\"'#]+)")
_QUERY_VALUE = re.compile(r"=[^&]*")


def redact_url(text: str) -> str:
    """Mask the credentials in every request URL or path within ``text``.

    Every query value is masked and its key kept, so ``/ws?token=abc&page=2``
    becomes ``/ws?token=[REDACTED]&page=[REDACTED]``, and so is every path
    segment that is a credential in its own right.
    """
    text = _CREDENTIAL_PATH_SEGMENT.sub(lambda match: match[1] + REDACTED, text)
    return _QUERY_STRING.sub(lambda match: "?" + redact_query_string(match[1]), text)


def redact_query_string(query: str) -> str:
    """Mask every value in a bare query string such as ``token=abc&page=2``."""
    return _QUERY_VALUE.sub("=" + REDACTED, query)


class RequestURLRedactionFilter(logging.Filter):
    """Masks request URLs in a log record before any handler sees it.

    uvicorn passes the request target as a format argument, so the record is
    rendered first, and replaced only when masking changed it. Records with
    nothing to mask keep their format string and arguments.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            # A malformed record fails again, and is reported, in the handler;
            # raising here would instead fail the request that logged it.
            return True
        redacted = redact_url(message)
        if redacted != message:
            record.msg, record.args = redacted, ()
        return True
