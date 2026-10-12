"""Credentials a request carries stay out of the logs and Sentry, wherever the
code that handles it passes them on.

A route that raises sends Sentry the local variables of every frame; a request
body that fails validation is quoted whole in pydantic's errors; a webhook's ID
is the credential in its ingress URL; and an outgoing request's URL can hold an
API key. These run the real servers, handlers and SDK client, as
request_url_logging_test.py does, and check what is left behind.
"""

import io
import json
import random
import string
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest
import sentry_sdk
from fastapi import FastAPI
from prisma.models import IntegrationWebhook

from backend.api import rest_api
from backend.data.model import APIKeyCredentials
from backend.integrations.webhooks.github import GithubWebhooksManager
from backend.util import metrics
from backend.util.logging_test_helpers import CapturingTransport, serve
from backend.util.service import AppService
from backend.util.url_redaction import REDACTED

SHARE_TOKEN = "5ec2e7ab-0000-4000-8000-00000000c0de"
WEBHOOK_ID = "0b5e55ed-0000-4000-8000-00000000beef"
UNKNOWN_WEBHOOK_ID = "0dd5ba11-0000-4000-8000-00000000f00d"
COMPASS_WEBHOOK_ID = "c0e1a55e-0000-4000-8000-00000000cafe"
SIGNING_SECRET = "signing-SECRET345"
REFRESH_TOKEN = "refresh-SECRET678"
CLIENT_SECRET = "client-SECRET901"
API_KEY = "AIzaSECRET2345"
BOT_TOKEN = "123456:SECRET-bot-6789"


async def test_sentry_gets_no_credential_from_a_failing_routes_variables(
    sentry: CapturingTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with (
        serve(rest_api.AgentServer(), monkeypatch) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        # No database is connected, so each route raises with the credential
        # in its own variables and in the query it was building.
        shared = await client.get(f"/api/public/shared/{SHARE_TOKEN}")
        ingress = await client.post(
            f"/api/integrations/github/webhooks/{WEBHOOK_ID}/ingress", json={}
        )
    assert (shared.status_code, ingress.status_code) == (500, 500)

    assert SHARE_TOKEN not in sentry.payload
    assert WEBHOOK_ID not in sentry.payload
    # The frames are still sent, with the credentials masked.
    route_variables = {
        frame["function"]: frame.get("vars", {})
        for event in sentry.events
        for exception in (event.get("exception") or {}).get("values", [])
        for frame in exception["stacktrace"]["frames"]
    }
    assert route_variables["get_shared_execution"]["share_token"] == "[Filtered]"
    assert route_variables["webhook_ingress_generic"]["webhook_id"] == "[Filtered]"
    # A credential under a name that tells nothing is found by its value.
    assert route_variables["find_unique"]["where"]["id"] == f"'{REDACTED}'"
    # Error events keep their route's name, which names its credential.
    assert "/api/public/shared/{share_token}" in {
        event.get("transaction") for event in sentry.events
    }


async def test_a_rejected_request_body_stays_out_of_the_logs(
    service_output: io.StringIO,
    sentry: CapturingTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with (
        serve(rest_api.AgentServer(), monkeypatch) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        # Without a client_id the body matches neither grant, so pydantic
        # reports each missing field with the whole body as its input.
        response = await client.post(
            "/api/oauth/token",
            json={
                "grant_type": "refresh_token",
                "refresh_token": REFRESH_TOKEN,
                "client_secret": CLIENT_SECRET,
            },
        )
    assert response.status_code == 422

    logs = service_output.getvalue()
    for credential in (REFRESH_TOKEN, CLIENT_SECRET):
        assert credential not in logs
        assert credential not in sentry.payload
    reason = "body.TokenRequestByRefreshToken.client_id (missing)"
    assert "Validation failed for POST /api/oauth/token: " in logs
    assert reason in logs
    # Sentry gets the same description, instead of a message filtered whole.
    assert reason in sentry.payload


async def test_a_service_logs_no_input_that_failed_validation(
    service_output: io.StringIO,
) -> None:
    app = FastAPI()

    @app.post("/store_credentials")
    async def store_credentials() -> None:
        # The provider is missing, so the error's input is the whole object.
        APIKeyCredentials.model_validate({"api_key": API_KEY, "title": "Gemini"})

    AppService._register_exception_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://service"
    ) as client:
        response = await client.post("/store_credentials")
    assert response.status_code == 400

    logs = service_output.getvalue()
    assert API_KEY not in logs
    assert (
        "POST /store_credentials failed: 1 validation error: provider (missing) "
        "for APIKeyCredentials"
    ) in logs


@pytest.mark.parametrize("log_level", ["DEBUG"])
async def test_webhook_ingress_logs_its_webhook_ids_only_by_prefix(
    service_output: io.StringIO,
    sentry: CapturingTransport,
    monkeypatch: pytest.MonkeyPatch,
    log_level: str,
) -> None:
    webhooks = {
        WEBHOOK_ID: _webhook_row(WEBHOOK_ID, "github"),
        COMPASS_WEBHOOK_ID: _webhook_row(COMPASS_WEBHOOK_ID, "compass"),
    }
    monkeypatch.setattr(IntegrationWebhook, "prisma", lambda: _WebhookTable(webhooks))
    monkeypatch.setattr(
        GithubWebhooksManager,
        "verify_signature",
        AsyncMock(side_effect=RuntimeError("malformed signature")),
    )
    async with (
        serve(rest_api.AgentServer(), monkeypatch) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        responses = [
            (
                await client.post(f"/api/integrations/github/webhooks/{id}/ingress")
            ).json()
            for id in (UNKNOWN_WEBHOOK_ID, COMPASS_WEBHOOK_ID, WEBHOOK_ID)
        ]

    logs = service_output.getvalue()
    for credential in (UNKNOWN_WEBHOOK_ID, COMPASS_WEBHOOK_ID, WEBHOOK_ID):
        assert credential not in logs
        assert credential not in sentry.payload
    assert SIGNING_SECRET not in logs
    # A webhook registered for another provider reads as one that doesn't exist.
    assert responses == [
        {"detail": f"Webhook #{UNKNOWN_WEBHOOK_ID[:8]}... not found"},
        {"detail": f"Webhook #{COMPASS_WEBHOOK_ID[:8]}... not found"},
        {"detail": "Invalid webhook signature"},
    ]
    for line in (
        f"Received github webhook ingress for ID {WEBHOOK_ID[:8]}...",
        f"unknown webhook #{UNKNOWN_WEBHOOK_ID[:8]}...",
        f"Webhook #{COMPASS_WEBHOOK_ID[:8]}... provider mismatch",
        f"Webhook #{WEBHOOK_ID[:8]}...: {{'user_id': 'user-1', 'provider'",
        f"Signature verification failed for webhook #{WEBHOOK_ID[:8]}... (github)",
    ):
        assert line in logs


async def test_sentry_gets_no_credentials_from_outgoing_request_urls(
    sentry: CapturingTransport,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        # What Google answers a key it doesn't know.
        return httpx.Response(400, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        with sentry_sdk.start_transaction(name="outgoing"):
            gemini = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": API_KEY},
            )
            await client.post(f"https://api.telegram.org/bot{BOT_TOKEN}/getMe")
        try:
            gemini.raise_for_status()
        except httpx.HTTPStatusError as error:
            sentry_sdk.capture_exception(error)

    assert API_KEY not in sentry.payload
    assert BOT_TOKEN not in sentry.payload
    # The SDK records each request in a breadcrumb, and in a span of the
    # transaction it ran in; both keep the URL and the query's keys.
    [error] = [event for event in sentry.events if event.get("exception")]
    breadcrumbs = [crumb["data"] for crumb in error["breadcrumbs"]["values"]]
    [transaction] = [e for e in sentry.events if e.get("type") == "transaction"]
    spans = [span for span in transaction["spans"] if span["op"] == "http.client"]
    expected = [
        ("https://generativelanguage.googleapis.com/v1beta/models", f"key={REDACTED}"),
        (f"https://api.telegram.org/bot{REDACTED}/getMe", ""),
    ]
    for recorded in (breadcrumbs[-2:], [span["data"] for span in spans]):
        assert [(data["url"], data["http.query"]) for data in recorded] == expected
    assert (
        spans[1]["description"] == f"POST https://api.telegram.org/bot{REDACTED}/getMe"
    )
    assert f"/v1beta/models?key={REDACTED}" in error["exception"]["values"][0]["value"]


@pytest.mark.parametrize("seed", range(5))
def test_no_part_of_an_error_event_keeps_a_credential_from_its_request_url(
    seed: int,
) -> None:
    """A route can hand a credential from its URL on to anything, so it is
    masked by value: here in random places of an event, under random URLs with
    crafted Host headers and prefixes stacked in front of the real one."""
    rng = random.Random(seed)
    for _ in range(200):
        secret = "".join(rng.choices(_URL_SAFE, k=rng.randint(16, 48)))
        route = rng.choice(_CREDENTIAL_ROUTES)
        path = rng.choice(_STACKED_PREFIXES) + route.replace("{}", secret)
        query = f"token={secret}" if rng.random() < 0.3 else ""
        if query and rng.random() < 0.5:
            query = f"q={rng.choice(['', '?', 'a=b', '/api/'])}&{query}"
        host = rng.choice(["backend", "evil?", "evil?/api/invitations", "h#"])
        holders = [
            repr({"id": secret}),
            f"Looked up {secret!r} and found nothing",
            f"https://{host}{path}",
            [rng.choice(["x", "", secret]), secret],
        ]
        event = {
            "request": {"url": f"http://{host}{path}", "query_string": query},
            "transaction": route.replace("{}", "{share_token}"),
            "transaction_info": {"source": "route"},
            "exception": {
                "values": [
                    {
                        "value": rng.choice(holders[:2]),
                        "stacktrace": {
                            "frames": [{"vars": {"where": rng.choice(holders)}}]
                        },
                    }
                ]
            },
            "breadcrumbs": {"values": [{"message": rng.choice(holders[1:3])}]},
        }
        scrubbed = metrics._before_send(event, {})
        assert scrubbed is not None
        assert secret not in json.dumps(scrubbed), (host, path, query)
        assert scrubbed["transaction"] == route.replace("{}", "{share_token}")


_URL_SAFE = string.ascii_letters + string.digits + "-_"
_CREDENTIAL_ROUTES = [
    "/api/integrations/github/webhooks/{}/ingress",
    "/api/integrations/webhooks/{}/ping",
    "/api/platform-linking/tokens/{}/confirm",
    "/api/public/shared/{}",
    "/api/public/shared/chats/{}/messages",
    "/api/invitations/{}/accept",
]
_STACKED_PREFIXES = ["", "", "/api/public/shared", "/api/invitations", "/backend"]


def _webhook_row(webhook_id: str, provider: str) -> IntegrationWebhook:
    return IntegrationWebhook(
        id=webhook_id,
        createdAt=datetime.now(timezone.utc),
        userId="user-1",
        provider=provider,
        credentialsId="",
        webhookType="repo",
        resource="owner/repo",
        events=["push"],
        config=json.dumps({"secret": SIGNING_SECRET}),
        secret=SIGNING_SECRET,
        providerWebhookId="1",
        AgentNodes=[],
        AgentPresets=[],
        visibility="PRIVATE",
    )


class _WebhookTable:
    """What ``IntegrationWebhook.prisma()`` reads, without a database."""

    def __init__(self, rows: dict[str, IntegrationWebhook]) -> None:
        self.rows = rows

    async def find_unique(self, where: dict, include: object = None):
        return self.rows.get(where["id"])
