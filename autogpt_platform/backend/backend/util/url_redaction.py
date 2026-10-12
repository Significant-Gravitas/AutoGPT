"""Masking for credentials that travel in request URLs and headers.

The browser opens its WebSocket with the user's session JWT in ``?token=``,
email links carry one-click tokens in the query, and some path segments are
bearer secrets themselves: whoever holds one can post to a webhook, claim a
platform link, read a shared chat or run, or accept an org invitation. uvicorn
logs the path and query of every request and Sentry attaches both to every
event, so anything that records a request URL masks it here first. Outgoing
URLs carry credentials too: an API key in ``?key=``, or a Telegram bot's token
in the path of every call to its API.
"""

import logging
import re
from collections.abc import Mapping

REDACTED = "[REDACTED]"

# The segment after each of these prefixes is a credential, even on a route
# that doesn't exist. /api/invitations/pending is the one static route among
# them, so it alone is kept. Telegram's segment is "bot" and the token. The
# match is zero-width so that overlapping prefixes, which a crafted path or
# Host header can stack, are all found.
_CREDENTIAL_PATH_SEGMENT = re.compile(
    r"(?=(?:/api/(?:integrations/(?:[^/?\s]+/)?webhooks|platform-linking/(?:user-)?tokens"
    r"|public/shared(?:/chats)?|invitations(?!/pending(?:[/?\s\"']|$)))/"
    r"|api\.telegram\.org/(?:file/)?bot)([^/?\s]+))"
)
# A request target can't contain whitespace, so in a longer message its query
# runs to the next whitespace. Nothing else ends it: a raw quote can sit in a
# value, and uvicorn's WebSocket scope keeps even a raw # in the query.
_QUERY_IN_TEXT = re.compile(r"\?(\S+)")
_QUERY_VALUE = re.compile(r"=[^&]*")
_TOKEN_IN_QUERY = re.compile(r"[?&]token=([^&\s]*)")
# Every URL redact_urls_in_text masks holds one of these, so a string without
# any is left as it is: Sentry runs it on every string of every event.
_MENTIONS_URL = re.compile(r"\?|/api/|api\.telegram\.org/")
# A shorter value can't be a credential the platform issued, and masking it
# everywhere it appears would mangle the rest of the event.
_MIN_CREDENTIAL_LENGTH = 16
# A credential shown by its prefix must be long enough that the rest can't be
# guessed: a UUID keeps 90 random bits once its first 8 characters are shown.
_SHOWN_PREFIX_LENGTH = 8
_MIN_PREFIXED_CREDENTIAL_LENGTH = 32


def redact_url(url: str) -> str:
    """Mask the credentials in a URL or request target.

    Every query value is masked and its key kept, so ``/ws?token=abc&page=2``
    becomes ``/ws?token=[REDACTED]&page=[REDACTED]``, and so is every path
    segment that is a credential in its own right. Paths are masked across the
    whole string, after the query: Sentry builds a URL from the Host header,
    which can hold a ``?`` of its own.
    """
    target, question_mark, query = url.partition("?")
    return _redact_path(target + question_mark + redact_query_string(query))


def redact_urls_in_text(text: str) -> str:
    """Mask the credentials in every request URL inside a longer message."""
    if not _MENTIONS_URL.search(text):
        return text
    return _redact_path(
        _QUERY_IN_TEXT.sub(lambda match: "?" + redact_query_string(match[1]), text)
    )


def redact_query_string(query: str) -> str:
    """Mask every value in a query string, without its ``?``, and keep the keys."""
    return _QUERY_VALUE.sub("=" + REDACTED, query)


def credentials_in_url(url: str) -> set[str]:
    """The credentials a request URL carries: each credential path segment, and
    the value of ``token`` in its query.

    A route hands them on as it likes, as an argument, a local or a database
    query's ``where``, so whatever records what it did masks them by value.
    Like ``redact_url``, this reads paths across the whole URL.
    """
    found = {match[1] for match in _CREDENTIAL_PATH_SEGMENT.finditer(url)}
    found.update(_TOKEN_IN_QUERY.findall(url))
    return {value for value in found if len(value) >= _MIN_CREDENTIAL_LENGTH}


def mask_credential(credential: str) -> str:
    """Shorten a credential that is also its record's ID, like a webhook's, to a
    prefix that still finds the record but can't be used in its place."""
    if len(credential) < _MIN_PREFIXED_CREDENTIAL_LENGTH:
        return REDACTED
    return credential[:_SHOWN_PREFIX_LENGTH] + "..."


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


_CREDENTIAL_HEADER = re.compile(
    r"(?i)auth|cookie|credential|jwt|key|password|secret|session|signature|token"
    r"|sec-websocket-protocol"
)
# At DEBUG, websockets logs every handshake header it reads as ``< name: value``
# and every one it writes as ``> name: value``.
_HEADER_LINES = frozenset({"< %s: %s", "> %s: %s"})


def is_credential_header(name: str) -> bool:
    """Whether a header's value is a credential: the session cookie and
    ``Authorization``, an API key, or a webhook's signature or secret token,
    whatever the header is called."""
    return bool(_CREDENTIAL_HEADER.search(name))


class CredentialHeaderFilter(logging.Filter):
    """Masks the value of every credential header websockets logs, such as the
    session cookie and ``Authorization`` of each WebSocket handshake."""

    def filter(self, record: logging.LogRecord) -> bool:
        if (
            isinstance(record.msg, str)
            and record.msg in _HEADER_LINES
            and isinstance(record.args, tuple)
            and len(record.args) == 2
            and isinstance(record.args[0], str)
            and is_credential_header(record.args[0])
        ):
            record.args = (record.args[0], REDACTED)
        return True
