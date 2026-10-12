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
from collections.abc import Mapping

REDACTED = "[REDACTED]"

# The segment after each of these prefixes is a credential, even on a route
# that doesn't exist. /api/invitations/pending is the one static route among
# them, so it alone is kept. The match is zero-width so that overlapping
# prefixes, which a crafted path or Host header can stack, are all found.
_CREDENTIAL_PATH_SEGMENT = re.compile(
    r"(?=/api/(?:integrations/(?:[^/?\s]+/)?webhooks|platform-linking/(?:user-)?tokens"
    r"|public/shared(?:/chats)?|invitations(?!/pending(?:[/?\s\"']|$)))/([^/?\s]+))"
)
# A request target can't contain whitespace, so in a longer message its query
# runs to the next whitespace. Nothing else ends it: a raw quote can sit in a
# value, and uvicorn's WebSocket scope keeps even a raw # in the query.
_QUERY_IN_TEXT = re.compile(r"\?(\S+)")
_QUERY_VALUE = re.compile(r"=[^&]*")
_MENTIONS_URL = re.compile(r"\?|/api/")


def redact_url(url: str) -> str:
    """Mask the credentials in a URL or request target.

    Every query value is masked and its key kept, so ``/ws?token=abc&page=2``
    becomes ``/ws?token=[REDACTED]&page=[REDACTED]``, and so is every path
    segment that is a credential in its own right. Paths are masked across the
    whole string, after the query: Sentry builds a URL from the Host header,
    which can hold a ``?`` of its own.
    """
    target, question_mark, query = url.partition("?")
    return _redact_path(target + question_mark + _redact_query(query))


def redact_urls_in_text(text: str) -> str:
    """Mask the credentials in every request URL inside a longer message."""
    return _redact_path(
        _QUERY_IN_TEXT.sub(lambda match: "?" + _redact_query(match[1]), text)
    )


def _redact_path(text: str) -> str:
    # Built once from the spans, so a URL stacked with prefixes stays linear.
    parts: list[str] = []
    done = 0
    for match in _CREDENTIAL_PATH_SEGMENT.finditer(text):
        start, end = match.span(1)
        if end > done:
            parts += [text[done : max(start, done)], REDACTED]
            done = end
    return "".join(parts) + text[done:]


def _redact_query(query: str) -> str:
    return _QUERY_VALUE.sub("=" + REDACTED, query)


class RequestURLRedactionFilter(logging.Filter):
    """Masks request URLs in a log record before any handler sees it.

    Each argument is masked as a whole URL, which is how uvicorn passes the
    request target. A format string that mentions a URL itself, or a message
    that arrives already formatted, is rendered and scanned instead.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, Mapping):
            record.args = {
                key: _redact_argument(value) for key, value in record.args.items()
            }
        elif isinstance(record.args, tuple):
            record.args = tuple(_redact_argument(arg) for arg in record.args)
        if isinstance(record.msg, str) and _MENTIONS_URL.search(record.msg):
            try:
                message = record.getMessage()
            except Exception:
                # A malformed record fails again, and is reported, in the
                # handler; raising here would fail the request that logged it.
                return True
            masked = redact_urls_in_text(message)
            if masked != message:
                record.msg, record.args = masked, ()
        return True


def _redact_argument(argument: object) -> object:
    try:
        text = argument if isinstance(argument, str) else str(argument)
    except Exception:
        return argument  # the handler hits, and reports, the same failure
    masked = redact_url(text)
    return argument if masked == text else masked
