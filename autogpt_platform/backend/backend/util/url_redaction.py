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

# The segment after each of these prefixes is a credential. /api/invitations/
# also has static routes (/pending), so there only an invitation's own count.
_CREDENTIAL_PATH_SEGMENT = re.compile(
    r"(/api/(?:integrations/(?:[^/?#\s]+/)?webhooks|platform-linking/(?:user-)?tokens"
    r"|public/shared(?:/chats)?|invitations(?=/[^/?#\s]+/(?:accept|decline)))/)"
    r"[^/?#\s]+"
)
# A request target can't contain whitespace, so in a longer message its query
# runs to the next whitespace. Quotes don't end it: a raw one can be in a value.
_QUERY_IN_TEXT = re.compile(r"\?([^\s#]+)")
_QUERY_VALUE = re.compile(r"=[^&]*")


def redact_url(url: str) -> str:
    """Mask the credentials in a URL or request target.

    Every query value is masked and its key kept, so ``/ws?token=abc&page=2``
    becomes ``/ws?token=[REDACTED]&page=[REDACTED]``, and so is every path
    segment that is a credential in its own right.
    """
    target, question_mark, rest = url.partition("?")
    query, hash_mark, fragment = rest.partition("#")
    return (
        _redact_path(target)
        + question_mark
        + _redact_query(query)
        + hash_mark
        + fragment
    )


def redact_urls_in_text(text: str) -> str:
    """Mask the credentials in every request URL inside a longer message."""
    return _QUERY_IN_TEXT.sub(
        lambda match: "?" + _redact_query(match[1]), _redact_path(text)
    )


def _redact_path(text: str) -> str:
    return _CREDENTIAL_PATH_SEGMENT.sub(lambda match: match[1] + REDACTED, text)


def _redact_query(query: str) -> str:
    return _QUERY_VALUE.sub("=" + REDACTED, query)


class RequestURLRedactionFilter(logging.Filter):
    """Masks request URLs in a log record before any handler sees it.

    uvicorn passes the request target as a format argument, which is masked
    as a whole URL. A message that arrives already formatted is scanned for
    URLs instead.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple) and record.args:
            record.args = tuple(
                redact_url(arg) if isinstance(arg, str) else arg for arg in record.args
            )
        elif not record.args and isinstance(record.msg, str):
            record.msg = redact_urls_in_text(record.msg)
        return True
