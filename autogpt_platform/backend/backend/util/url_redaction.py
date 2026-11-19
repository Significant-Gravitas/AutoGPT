"""Masking for credentials that travel in request URLs and headers.

The browser opens its WebSocket with the user's session JWT in ``?token=``,
email links carry one-click tokens in the query, and some path segments are
bearer secrets themselves: whoever holds one can post to a webhook, claim a
platform link, read a shared chat or run, or accept an org invitation. uvicorn
logs the path and query of every request and Sentry attaches both to every
event, so anything that records a request URL masks it here first. Outgoing
URLs carry credentials too: an API key in ``?key=``, a Telegram bot's token in
the path of every call to its API, or a user and password in a proxy's or a
database's URL. Headers carry the session cookie, ``Authorization`` and a
webhook provider's own secret token.

The credentials of the request being handled are kept here too, by
``backend.util.request_credentials``, so that a log line or an error report
masks them by value wherever the code handling the request quotes them. A copy
derived from one (sliced, re-encoded, case-folded) is not recognised.
"""

import itertools
import logging
import re
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from http.cookies import _unquote as _unquote_cookie
from urllib.parse import unquote, unquote_plus

REDACTED = "[REDACTED]"
# Set once per request in its own task, longest first, and never reset: an
# error is logged and reported after the request's middleware has returned.
_request_credentials: ContextVar[tuple[str, ...]] = ContextVar(
    "request_credentials", default=()
)

# The segment after each of these prefixes is a credential, even on a route
# that doesn't exist. /api/invitations/pending is the one static route among
# them, so it alone is kept, and so is a route's template, such as
# {share_token}, which names a credential without being one; /chats is matched
# possessively, so a template after it doesn't make "chats" the credential.
# Telegram's segment is "bot" and the token, under its host in any case and
# with or without its default port. The match is zero-width so that
# overlapping prefixes, which a crafted path or Host header can stack, are all
# found.
_CREDENTIAL_PATH_SEGMENT = re.compile(
    r"(?=(?:/api/(?:integrations/(?:[^/?\s]+/)?webhooks|platform-linking/(?:user-)?tokens"
    r"|public/shared(?:/chats)?+|invitations(?!/pending(?:[/?\s\"']|$)))/"
    r"|(?i:api\.telegram\.org)(?::443)?/(?:file/)?bot)"
    r"(?!\{\w+\}(?:[/?\s\"']|$))([^/?\s]+))"
)
# A query in a longer message starts at a "?" right after a path: a run of the
# characters a path can hold (RFC 3986's, brackets for an IPv6 host, and the
# braces, bars, carets, backslashes and backticks a client can send raw), with a
# "/" in it (a URL's "//" included). So code a traceback or an error quotes,
# such as a regex's "(?=", isn't taken for one, though code with a "/" just
# before its "(?=" loses the rest of that word. A query with no path before it,
# or after a quote, which ends a quoted URL, isn't read: the price of keeping
# code intact. A request target can't contain whitespace, so the query runs to
# the next whitespace. Nothing else ends it: a raw quote can sit in a value,
# and uvicorn's WebSocket scope keeps even a raw # in the query. The run is
# read once, from its start and without backtracking: a client sends the path
# an access log quotes, and trying it from each "/" takes time growing with the
# square of its length.
_PATH_CHARACTERS = r"\w\-.~%!$&'()*+,;=:@\[\]{}|^\\`"
_QUERY_IN_TEXT = re.compile(
    rf"(?<![{_PATH_CHARACTERS}/])"
    rf"([{_PATH_CHARACTERS}]*+/[{_PATH_CHARACTERS}/]*+)\?(\S+)"
)
_QUERY_VALUE = re.compile(r"=[^&]*")
# The user and password in a URL's authority. They end at its last "@", as a
# URL parser reads a password that holds one, and the authority ends before any
# "/", "?" or "#", so an "@" in a path or query isn't taken for one.
_USERINFO = re.compile(r"(?<=://)[^/?#\s]+@")
# Every URL redact_urls_in_text masks holds one of these, so a string without
# any is left as it is: Sentry's scrubber runs it on every string of an event.
_MENTIONS_URL = re.compile(r"\?|/api/|(?i:api\.telegram\.org)(?::443)?/|://[^/?#\s@]*@")
# A query parameter's name, wherever a "?" or "&" starts one: a "?" in a value,
# or in the Host header Sentry builds a URL from, starts another as a server
# would read it.
_QUERY_NAME = re.compile(r"[?&]([^=&?]*)=")
# How many values named like a credential one URL is read for.
_MAX_QUERY_CREDENTIALS = 16
# Names of headers, cookies and query parameters whose values are credentials,
# matched anywhere in a name, so glued names such as `PHPSESSID` and
# `clientsecret`, and header names as Sentry shows an ASGI scope's bytes
# (`b'cookie'`), are masked. The accepted cost: a name that only contains one of
# these words, such as `author` or `design`, has its value masked too.
_CREDENTIAL_NAME = re.compile(
    r"(?i)auth|code|cookie|credential|csrf|jwt|key|password|secret|sess|sid|sig"
    r"|state|token|sec-websocket-protocol|content-mac"
)
# A shorter value can't be a credential the platform issued or a provider's
# key, and masking it everywhere it turns up would mangle the rest of an event.
MIN_CREDENTIAL_LENGTH = 16
# Past this many places of the request's credentials in one text, mask_known
# masks the text whole.
_MAX_MERGED_MATCHES = 1024
# A credential shown by its prefix must be long enough that the rest can't be
# guessed: a UUID keeps 90 random bits once its first 8 characters are shown.
_SHOWN_PREFIX_LENGTH = 8
_MIN_PREFIXED_CREDENTIAL_LENGTH = 32


def keep_request_credentials(credentials: tuple[str, ...]) -> None:
    """Keep ``credentials``, longest first, as those of the request the current
    task handles."""
    _request_credentials.set(credentials)


def request_credentials() -> tuple[str, ...]:
    """The credentials of the request being handled, longest first."""
    return _request_credentials.get()


def mask_known(text: str, credentials: Iterable[str]) -> str:
    """``text`` with every place any of ``credentials``, longest first, occurs
    masked. Each is found without overlapping itself, as ``str.replace`` finds
    it, and places where different ones overlap are masked as one, so none is
    left partly shown by another's mask. A text where different ones occur in
    more than ``_MAX_MERGED_MATCHES`` places is masked whole: only crafted text
    holds that many, and the work stays linear in ``text`` either way."""
    present = [
        credential for credential in credentials if credential and credential in text
    ]
    spans: list[tuple[int, int]] = []
    if len(present) > 1:
        for credential in present:
            start = text.find(credential)
            while start != -1 and len(spans) <= _MAX_MERGED_MATCHES:
                stop = start + len(credential)
                spans.append((start, stop))
                start = text.find(credential, stop)
    if len(spans) > _MAX_MERGED_MATCHES:
        return REDACTED
    if len(present) < 2:
        for credential in present:
            text = text.replace(credential, REDACTED)
        return text
    spans.sort()
    merged = [list(spans[0])]
    for start, stop in spans[1:]:
        if start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    parts: list[str] = []
    shown = 0
    for start, stop in merged:
        parts += [text[shown:start], REDACTED]
        shown = stop
    parts.append(text[shown:])
    return "".join(parts)


def redact_url(url: str) -> str:
    """Mask the credentials in a URL or request target.

    Every query value is masked and its key kept, so ``/ws?token=abc&page=2``
    becomes ``/ws?token=[REDACTED]&page=[REDACTED]``, and so is every path
    segment that is a credential in its own right. Paths are masked across the
    whole string, after the query: Sentry builds a URL from the Host header,
    which can hold a ``?`` of its own.
    """
    target, question_mark, query = url.partition("?")
    target = _USERINFO.sub(REDACTED + "@", target)
    return _redact_path(target + question_mark + redact_query_string(query))


def redact_urls_in_text(text: str) -> str:
    """Mask the credentials in every request URL inside a longer message."""
    if not _MENTIONS_URL.search(text):
        return text
    text = _QUERY_IN_TEXT.sub(
        lambda match: f"{match[1]}?{redact_query_string(match[2])}", text
    )
    return _redact_path(_USERINFO.sub(REDACTED + "@", text))


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
    # Named and valued as the route reads them, form-decoded, and as sent. A
    # value runs to the next "&", as redact_url masks it, and one too short to
    # be a credential takes none of the slots: a query can repeat a name with
    # blanks before the value a route reads.
    values = (
        url[match.end() : _end_of_value(url, match.end())]
        for match in _QUERY_NAME.finditer(url)
        if _CREDENTIAL_NAME.search(unquote_plus(match[1]))
    )
    long_enough = (value for value in values if len(value) >= MIN_CREDENTIAL_LENGTH)
    for value in itertools.islice(long_enough, _MAX_QUERY_CREDENTIALS):
        found.update((value, unquote(value), unquote_plus(value)))
    return {value for value in found if len(value) >= MIN_CREDENTIAL_LENGTH}


def _end_of_value(url: str, start: int) -> int:
    end = url.find("&", start)
    return end if end >= 0 else len(url)


def credentials_in_headers(headers: Iterable[tuple[str, str]]) -> set[str]:
    """The credentials request headers carry: each credential header's value,
    the token after its scheme, such as ``Bearer``, and each credential cookie."""
    found: set[str] = set()
    for name, value in headers:
        if name.lower() == "cookie":
            for cookie in value.split(";"):
                cookie_name, _, cookie_value = cookie.strip().partition("=")
                if _CREDENTIAL_NAME.search(cookie_name):
                    # As sent, and as Starlette reads it: trimmed, unquoted.
                    read = _unquote_cookie(cookie_value.strip())
                    unquoted = cookie_value.strip('"')
                    found.update((unquoted, unquote(unquoted), read, unquote(read)))
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
    """Masks request URLs in a log record before any handler sees it, and the
    credentials of the request being handled wherever the record quotes them.

    Each argument is masked as a whole URL, which is how uvicorn passes the
    request target. A format string that mentions a URL or a credential itself,
    or a message that arrives already formatted, is rendered and scanned
    instead, and so is the record's traceback, whose exception can quote an
    outgoing request's URL or a header's value.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        credentials = request_credentials()
        # A string argument is masked in place; any other, such as an exception
        # or a number, where the rendered record shows it, so its format (%d,
        # %r) still applies to it.
        render = not isinstance(record.msg, str) and _masks(record.msg, credentials)
        if isinstance(record.args, Mapping):
            named = {
                key: _masked_argument(argument, credentials)
                for key, argument in record.args.items()
            }
            record.args = {key: argument for key, (argument, _) in named.items()}
            render = render or any(masks for _, masks in named.values())
        elif isinstance(record.args, tuple):
            listed = [_masked_argument(arg, credentials) for arg in record.args]
            record.args = tuple(argument for argument, _ in listed)
            render = render or any(masks for _, masks in listed)
        if render or (
            isinstance(record.msg, str)
            and (
                _MENTIONS_URL.search(record.msg)
                or any(credential in record.msg for credential in credentials)
            )
        ):
            try:
                message = record.getMessage()
            except Exception as failure:
                # The handler would fail on it again and report its arguments
                # and the failure's text, which can hold what this filter
                # couldn't check, so it gets a notice instead: the format,
                # masked, and the failure's type. Raising here would fail the
                # request that logged it.
                template = record.msg if isinstance(record.msg, str) else ""
                shown = _masked_text(template, credentials)
                record.msg = (
                    "Not logged: a record that couldn't be formatted "
                    f"({type(failure).__name__}): {shown}"
                )
                record.args = ()
            else:
                masked = _masked_text(message, credentials)
                if masked != message:
                    record.msg, record.args = masked, ()
        if record.exc_info and not record.exc_text:
            # Rendered as every handler's formatter renders it, which then
            # uses this text instead of rendering the traceback again.
            traceback = _TRACEBACK_FORMATTER.formatException(record.exc_info)
            record.exc_text = _masked_text(traceback, credentials)
        return True


def _masked_argument(
    argument: object, credentials: tuple[str, ...]
) -> tuple[object, bool]:
    """A string argument masked in place, and for any other, whether its text
    has something to mask, which is then masked in the rendered record."""
    if isinstance(argument, str):
        return _masked_url(argument, credentials), False
    return argument, _masks(argument, credentials)


def _masked_url(url: str, credentials: tuple[str, ...]) -> str:
    """``url`` masked as a whole URL, as uvicorn passes a request target, then
    for ``credentials``."""
    masked = mask_known(redact_url(url), credentials)
    return url if masked == url else masked


def _masked_text(text: str, credentials: tuple[str, ...]) -> str:
    """``text`` with the credentials of any URL it quotes masked, then
    ``credentials``: in that order, so a request value that is part of a URL,
    such as its host, can't hide the URL's credentials from their masking."""
    return mask_known(redact_urls_in_text(text), credentials)


def _masks(value: object, credentials: tuple[str, ...]) -> bool:
    """Whether ``value``, as ``%s`` or ``%r`` shows it, has anything to mask.
    A value that either can't show can't be checked, so it counts as one."""
    shown: list[str] = []
    for show in (str, repr):
        try:
            shown.append(show(value))
        except Exception:
            return True
    return any(_masked_text(text, credentials) != text for text in dict.fromkeys(shown))


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
