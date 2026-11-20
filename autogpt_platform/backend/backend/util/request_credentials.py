"""The credentials a request carries, kept for the error reports raised while it
is handled.

Sentry's SDK masks the session cookie and ``Authorization`` in an event's
request data, but not the copies a route makes of them, such as a header's
value in a variable of its own, which the event carries with every frame's
variables. So each service's app keeps the credentials in a request's URL and
headers as it arrives, and the scrubber in ``backend.util.metrics`` masks each
of them wherever it turns up in an event raised while handling it, as the log
filters in ``backend.util.url_redaction`` do in a log line.
"""

from starlette.applications import Starlette
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.util.url_redaction import (
    credentials_in_headers,
    credentials_in_url,
    keep_request_credentials,
)

# Each is searched for in every string of an event, so their number is capped,
# and so is their length: no credential the app issues comes near it, and a
# frame variable is read as far as twice the longest (metrics._masked_head). A
# longer value is deliberately not masked by value, only where its shape or its
# field's name shows it.
MAX_REQUEST_CREDENTIALS = 16
MAX_CREDENTIAL_LENGTH = 4096


def remember_request_credentials(app: Starlette) -> None:
    """Keep each request's credentials while ``app`` handles it."""
    app.add_middleware(RequestCredentialsMiddleware)


def select_credentials(*groups: set[str]) -> tuple[str, ...]:
    """At most ``MAX_REQUEST_CREDENTIALS``, none longer than
    ``MAX_CREDENTIAL_LENGTH``, taken from ``groups`` in order, so that values of
    a later group, such as a URL's, never crowd out an earlier one's, such as
    the headers'. Returned longest first, so that one holding another is masked
    whole."""
    chosen: list[str] = []
    for group in groups:
        room = MAX_REQUEST_CREDENTIALS - len(chosen)
        fresh = sorted(
            (
                value
                for value in group.difference(chosen)
                if len(value) <= MAX_CREDENTIAL_LENGTH
            ),
            key=len,
            reverse=True,
        )
        chosen += fresh[:room]
    return tuple(sorted(chosen, key=len, reverse=True))


class RequestCredentialsMiddleware:
    """Keeps each HTTP request's and WebSocket connection's credentials while
    the app it wraps handles it. The value is never reset: an error is reported
    after the middleware returns, and a task the request starts inherits it,
    which can only mask more of what that task reports."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            # Never reset: an error is reported after the request's middleware
            # has returned, and the value lives in the request's own task.
            keep_request_credentials(_credentials_in_scope(scope))
        await self.app(scope, receive, send)


def _credentials_in_scope(scope: Scope) -> tuple[str, ...]:
    headers = [
        (name.decode("latin-1"), value.decode("latin-1"))
        for name, value in scope.get("headers", ())
    ]
    query = scope.get("query_string", b"").decode("latin-1")
    target = f"{scope.get('path', '')}?{query}"
    # The session cookie, Authorization and a provider's secret first: the
    # query's values are the ones a caller can send any number of.
    return select_credentials(
        credentials_in_headers(headers), credentials_in_url(target)
    )
