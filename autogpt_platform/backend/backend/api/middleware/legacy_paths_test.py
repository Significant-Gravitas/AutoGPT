"""The internal API still answers at the paths it had before /api/internal."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Match
from starlette.types import Receive, Scope, Send

from backend.api import rest_api
from backend.api.middleware.legacy_paths import (
    LegacyAPIPathMiddleware,
    internal_api_segments,
)


@pytest.mark.parametrize(
    "method, path",
    [
        # Stripe, Postmark and MailerLite call the webhook URLs their dashboards hold.
        ("POST", "/api/credits/stripe_webhook"),
        ("POST", "/api/email/"),
        ("POST", "/api/email/mailerlite/webhook"),
        # Every sent email's List-Unsubscribe header.
        ("POST", "/api/email/unsubscribe"),
        # Registered with each provider when the webhook was created.
        ("POST", "/api/integrations/github/webhooks/w-1/ingress"),
        # Third-party apps signed in through our OAuth provider.
        ("POST", "/api/oauth/token"),
        ("POST", "/api/oauth/introspect"),
        ("POST", "/api/oauth/revoke"),
        # Stored in marketplace listings and chat transcripts.
        ("GET", "/api/store/media/u-1/images/a.png"),
        ("GET", "/api/store/submissions/media/u-1/images/a.png"),
        ("GET", "/api/desktop-preview"),
        ("GET", "/api/workspace/files/f-1/download"),
    ],
)
async def test_a_url_held_outside_the_app_still_reaches_its_route(method, path):
    scope = await _rewritten({"type": "http", "method": method, "path": path})

    matched = [
        route.path
        for route in rest_api.app.routes
        if route.matches(scope)[0] == Match.FULL
    ]
    assert matched and matched[0].startswith("/api/internal/"), (path, matched)


@pytest.mark.parametrize(
    "path",
    [
        "/api/internal/library/agents",
        # Mounted at their own /api paths, outside the internal API.
        "/api/copilot-webhooks/slack/events",
        "/api/v2/graphs",
        "/api/v2/mcp",
        "/.well-known/oauth-protected-resource/api/v2/mcp",
        "/external-api/v1/blocks",
        "/health",
    ],
)
async def test_paths_outside_the_internal_api_are_left_alone(path):
    scope = await _rewritten({"type": "http", "method": "GET", "path": path})

    assert scope["path"] == path


async def test_a_versioned_path_is_never_read_as_an_internal_one():
    """`/api/v2` stays the public API's even if an internal route is named `v2`."""
    seen: list[Scope] = []

    async def record(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope)

    middleware = LegacyAPIPathMiddleware(record, segments=frozenset({"v2"}))
    await middleware(
        {"type": "http", "method": "GET", "path": "/api/v2/mcp"},
        _never_receive,
        _never_send,
    )

    assert seen[0]["path"] == "/api/v2/mcp"


def test_an_old_path_keeps_its_parameters_and_query():
    app = FastAPI()

    @app.get("/api/internal/things/{thing_id}")
    def read(thing_id: str, x: str):
        return {"thing_id": thing_id, "x": x}

    app.add_middleware(LegacyAPIPathMiddleware, segments=internal_api_segments(app))

    response = TestClient(app).get("/api/things/7?x=1")

    assert response.json() == {"thing_id": "7", "x": "1"}


async def _rewritten(scope: Scope) -> Scope:
    """`scope` as the routes see it, through the middleware the app has installed."""
    seen: list[Scope] = []

    async def record(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope)

    # Outermost, so the other middleware read the rewritten path too.
    installed = rest_api.app.user_middleware[0]
    assert installed.cls is LegacyAPIPathMiddleware
    await installed.cls(record, **installed.kwargs)(scope, _never_receive, _never_send)
    return seen[0]


async def _never_receive():
    raise AssertionError("the rewrite reads no body")


async def _never_send(message):
    raise AssertionError("the rewrite sends nothing")
