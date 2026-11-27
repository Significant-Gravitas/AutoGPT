import re

from autogpt_libs.auth.dependencies import IMPERSONATION_HEADER_NAME
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from backend.util import posthog_client

# The browser's PostHog session id, sent only with analytics consent
# (frontend `posthog-session-header.ts`).
POSTHOG_SESSION_HEADER_NAME = "X-PostHog-Session-Id"

# PostHog's session ids are UUIDv7s; anything else is dropped, not forwarded.
_SESSION_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
    re.IGNORECASE,
)


class AnalyticsContextMiddleware:
    """Tell the PostHog client who a request's events are for
    (`posthog_client.capture`): whether an admin is acting as the user, and
    which browser session the request came from.

    The impersonation header is read as it is: the auth dependencies refuse it
    from anyone but an admin, and a request that names it without being one
    can only mark its own events.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        impersonated = posthog_client.set_impersonated(
            bool(headers.get(IMPERSONATION_HEADER_NAME, "").strip())
        )
        session_id = posthog_client.set_session_id(
            _valid_session_id(headers.get(POSTHOG_SESSION_HEADER_NAME))
        )
        try:
            await self.app(scope, receive, send)
        finally:
            posthog_client.reset_session_id(session_id)
            posthog_client.reset_impersonated(impersonated)


def _valid_session_id(value: str | None) -> str | None:
    value = (value or "").strip()
    return value if _SESSION_ID.fullmatch(value) else None
