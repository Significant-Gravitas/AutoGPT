"""Masking for credentials that travel in request URLs and headers.

The browser opens its WebSocket with the user's session JWT in ``?token=``,
email links carry one-click tokens in the query, and some path segments are
bearer secrets themselves: whoever holds one can post to a webhook, claim a
platform link, read a shared chat or run, or accept an org invitation. uvicorn
logs the path and query of every request and Sentry attaches both to every
event, so anything that records a request URL masks it here first. Outgoing
URLs carry credentials too: an API key in ``?key=``, or a Telegram bot's token
in the path of every call to its API. Headers carry the session cookie,
``Authorization`` and a webhook provider's own secret token.
"""

import itertools
import logging
import re
from collections.abc import Iterable, Mapping
from urllib.parse import unquote

REDACTED = "[REDACTED]"

# The segment after each of these prefixes is a credential, even on a route
# that doesn't exist. /api/invitations/pending is the one static route among
# them, so it alone is kept, and so is a route's template, such as
# {share_token}, which names a credential without being one; /chats is matched
# possessively, so a template after it doesn't make "chats" the credential.
# Telegram's segment is "bot" and the token. The match is zero-width so that
# overlapping prefixes, which a crafted path or Host header can stack, are all
# found.
_CREDENTIAL_PATH_SEGMENT = re.compile(
    r"(?=(?:/api/(?:integrations/(?:[^/?\s]+/)?webhooks|platform-linking/(?:user-)?tokens"
    r"|public/shared(?:/chats)?+|invitations(?!/pending(?:[/?\s\"']|$)))/"
    r"|api\.telegram\.org/(?:file/)?bot)"
    r"(?!\{\w+\}(?:[/?\s\"']|$))([^/?\s]+))"
)
# A request target can't contain whitespace, so in a longer message its query
# runs to the next whitespace. Nothing else ends it: a raw quote can sit in a
# value, and uvicorn's WebSocket scope keeps even a raw # in the query.
_QUERY_IN_TEXT = re.compile(r"\?(\S+)")
_QUERY_VALUE = re.compile(r"=[^&]*")
# Every URL redact_urls_in_text masks holds one of these, so a string without
# any is left as it is: Sentry's scrubber runs it on every string of an event.
_MENTIONS_URL = re.compile(r"\?|/api/|api\.telegram\.org/")
# A query parameter's name, wherever a "?" or "&" starts one: a "?" in a value,
# or in the Host header Sentry builds a URL from, starts another as a server
# would read it.
_QUERY_NAME = re.compile(r"[?&]([^=&?]*)=")
# How many values named like a credential one URL is read for.
_MAX_QUERY_CREDENTIALS = 16
# Names of headers, cookies and query parameters whose values are credentials.
_CREDENTIAL_NAME = re.compile(
    r"(?i)auth|code|cookie|credential|jwt|key|password|secret|sess|sid|sig|state"
    r"|token|sec-websocket-protocol"
)
# A shorter value can't be a credential the platform issued or a provider's
# key, and masking it everywhere it turns up would mangle the rest of an event.
MIN_CREDENTIAL_LENGTH = 16
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
    each query value under a credential's name, such as ``token`` or ``key``.

    A route hands them on as it likes, as an argument, a local or a database
    query's ``where``, so whatever records what it did masks them by value.
    Like ``redact_url``, this reads paths across the whole URL.
    """
    found = {match[1] for match in _CREDENTIAL_PATH_SEGMENT.finditer(url)}
    named = (
        match
        for match in _QUERY_NAME.finditer(url)
        if _CREDENTIAL_NAME.search(match[1])
    )
    for match in itertools.islice(named, _MAX_QUERY_CREDENTIALS):
        # A value runs to the next "&", as redact_url masks it.
        end = url.find("&", match.end())
        value = url[match.end() : end if end >= 0 else len(url)]
        found.update((value, unquote(value)))
    return {value for value in found if len(value) >= MIN_CREDENTIAL_LENGTH}


def credentials_in_headers(headers: Iterable[tuple[str, str]]) -> set[str]:
    """The credentials request headers carry: each credential header's value,
    the token after its scheme, such as ``Bearer``, and each credential cookie."""
    found: set[str] = set()
    for name, value in headers:
        if name.lower() == "cookie":
            for cookie in value.split(";"):
                cookie_name, _, cookie_value = cookie.strip().partition("=")
                if _CREDENTIAL_NAME.search(cookie_name):
                    cookie_value = cookie_value.strip('"')
                    found.update((cookie_value, unquote(cookie_value)))
        elif is_credential_header(name):
            found.update((value, value.partition(" ")[2].strip()))
    return {value for value in found if len(value) >= MIN_CREDENTIAL_LENGTH}


def is_credential_header(name: str) -> bool:
    """Whether a header's value is a credential: the session cookie and
    ``Authorization``, an API key, or a webhook's signature or secret token,
    whatever the header is called."""
    return bool(_CREDENTIAL_NAME.search(name))


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


_TRACEBACK_FORMATTER = logging.Formatter()


class RequestURLRedactionFilter(logging.Filter):
    """Masks request URLs in a log record before any handler sees it.

    Each argument is masked as a whole URL, which is how uvicorn passes the
    request target. A format string that mentions a URL itself, or a message
    that arrives already formatted, is rendered and scanned instead, and so is
    the record's traceback, whose exception can quote an outgoing request's URL.
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
        if record.exc_info and not record.exc_text:
            # Rendered as every handler's formatter renders it, which then
            # uses this text instead of rendering the traceback again.
            record.exc_text = redact_urls_in_text(
                _TRACEBACK_FORMATTER.formatException(record.exc_info)
            )
        return True


def _redact_argument(argument: object) -> object:
    try:
        text = argument if isinstance(argument, str) else str(argument)
    except Exception:
        return argument  # the handler hits, and reports, the same failure
    masked = redact_url(text)
    return argument if masked == text else masked


# At DEBUG, websockets logs every handshake header it reads as ``< name: value``
# and every one it writes as ``> name: value``.
_HEADER_LINES = frozenset({"< %s: %s", "> %s: %s"})


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
