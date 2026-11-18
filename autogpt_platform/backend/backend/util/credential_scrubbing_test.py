"""Credentials a request carries stay out of the logs and Sentry, wherever the
code that handles it passes them on.

A route that raises sends Sentry the local variables of every frame; a value
that fails validation is quoted in pydantic's message; a webhook's ID is the
credential in its ingress URL; and an outgoing request's URL can hold an API
key. These run the real servers, handlers, logging filters and SDK client, as
request_url_logging_test.py does, and check what is left behind.
"""

import io
import json
import random
import string
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, Literal
from unittest.mock import AsyncMock

import httpx
import pytest
import sentry_sdk
from fastapi import FastAPI, Request, WebSocket
from prisma.errors import ClientNotConnectedError
from prisma.models import AgentGraphExecution, IntegrationWebhook
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from backend.api import rest_api, ws_api
from backend.api.conn_manager import ConnectionManager
from backend.api.features.integrations import router as integrations_router
from backend.api.features.library import triggers
from backend.api.utils.exceptions import add_exception_handlers
from backend.blocks.generic_webhook._webhook import (
    GenericWebhooksManager,
    GenericWebhookType,
)
from backend.blocks.generic_webhook.triggers import GenericWebhookTriggerBlock
from backend.blocks.github.triggers import GithubPullRequestTriggerBlock
from backend.data import integrations as integrations_db
from backend.data import redis_client
from backend.data.model import APIKeyCredentials
from backend.integrations.webhooks import _base as webhooks_base
from backend.integrations.webhooks import utils as webhooks_utils
from backend.integrations.webhooks.github import GithubWebhooksManager
from backend.integrations.webhooks.utils import (
    loggable_event_type,
    setup_webhook_for_block,
)
from backend.util import metrics
from backend.util.logging_test_helpers import CapturingTransport
from backend.util.request_credentials import (
    MAX_CREDENTIAL_LENGTH,
    MAX_REQUEST_CREDENTIALS,
    RequestCredentialsMiddleware,
    remember_request_credentials,
    select_credentials,
)
from backend.util.serve_app_test import serve, serve_app
from backend.util.service import AppService
from backend.util.url_redaction import REDACTED, request_credentials

SHARE_TOKEN = "5ec2e7ab-31d4-4c7e-9a52-6f08b3d1c0de"
WEBHOOK_ID = "0b5e55ed-8a1f-4e27-b9c3-d4527e19beef"
UNKNOWN_WEBHOOK_ID = "0dd5ba11-c6e2-49d8-a7f1-3b95c82af00d"
COMPASS_WEBHOOK_ID = "c0e1a55e-5b3d-4f86-8e04-a1d97c63cafe"
SIGNING_SECRET = "sg7Qw2Lx9Rk4Tz1V"
API_KEY = "AIzaSyB4kQ9xR2mW7pL5vT8nJ3hF6cD1gZ0eY"
BOT_TOKEN = "7120458312:AAHt6Yq3Wn8Rk2Zp5Lx9Vc4Jm7Bs1Gd0Fu"
TELEGRAM_SECRET_TOKEN = "tg9Kd3Pw7Xq2Mz5Rb8Lf1Hs4"
SESSION_COOKIE = "Zk3pQ9vT2mW8xR4sL7yN1bH6cJ0dF5gA.aB3cD4eF5gH6iJ7kL8mN9"
BEARER_TOKEN = "bt5Hn8Qr2Wx6Lk9Pz3Vm7Jc4"
TRIGGER_SECRET = "tr4Gv8Nk2Yw6Qs9Lp3Xd7Mh1"
QUERY_TOKEN = "qt3Lm8Vx2Kp7Rw4Nz9Hb5Jd6"
_SESSION_COOKIE_NAME = "__Secure-better-auth.session_token"
# A stored secret Telegram's verifier can't compare, so it fails with a
# traceback while its frame holds the header's value.
_UNCOMPARABLE_SECRET = "sécret"
_GEMINI_MODELS = "https://generativelanguage.googleapis.com/v1beta/models"


class Limits(BaseModel):
    per_key: dict[str, int]


class _Secret(BaseModel):
    type: Literal["secret"]
    value: str


class _Note(BaseModel):
    type: Literal["note"]
    text: str


class Filed(BaseModel):
    item: Annotated[_Secret | _Note, Field(discriminator="type")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class LookupFailedLoudly(RuntimeError):
    pass


class Checked(BaseModel):
    token: str

    @field_validator("token")
    @classmethod
    def _recognised(cls, token: str) -> str:
        raise ValueError(f"unrecognised token {token!r}")


# --- 1. Frame locals ---------------------------------------------------------


async def test_sentry_gets_no_credential_from_a_failing_routes_variables(
    sentry: CapturingTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(AgentGraphExecution, "prisma", lambda: _UnreachableTable())
    webhook = _webhook_row(WEBHOOK_ID, "telegram", secret=_UNCOMPARABLE_SECRET)
    monkeypatch.setattr(
        IntegrationWebhook, "prisma", lambda: _WebhookTable({WEBHOOK_ID: webhook})
    )
    async with (
        serve(rest_api.AgentServer(), monkeypatch) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        shared = await client.get(f"/api/public/shared/{SHARE_TOKEN}")
        # Telegram signs each delivery with a secret token in a header of its
        # own, which Sentry's own scrubbing doesn't know, and its verifier keeps
        # it in a variable of its own. Every frame holding the request's ASGI
        # scope holds the session cookie too.
        ingress = await client.post(
            f"/api/integrations/telegram/webhooks/{WEBHOOK_ID}/ingress",
            json={},
            headers={
                "X-Telegram-Bot-Api-Secret-Token": TELEGRAM_SECRET_TOKEN,
                "Cookie": f"__Secure-better-auth.session_token={SESSION_COOKIE}",
            },
        )
    assert (shared.status_code, ingress.status_code) == (500, 403)

    # A webhook's ID is shown by its first 8 characters, as the logs show it.
    for credential in (
        SHARE_TOKEN,
        WEBHOOK_ID[8:],
        TELEGRAM_SECRET_TOKEN,
        SESSION_COOKIE,
    ):
        assert not _leaks(credential, sentry.payload)
    # The frames are still sent, with only the credentials masked.
    variables = _frame_variables(sentry.events)
    assert variables["get_shared_execution"]["share_token"] == "[Filtered]"
    assert variables["webhook_ingress_generic"]["webhook_id"] == "[Filtered]"
    assert variables["verify_signature"]["secret_header"] == f"'{REDACTED}'"
    # Error events keep their route's name, which names its credential.
    assert "/api/public/shared/{share_token}" in {
        event.get("transaction") for event in sentry.events
    }
    # Errors and transactions both carry the request's headers.
    header = "x-telegram-bot-api-secret-token"
    assert {
        (event.get("type", "error"), event["request"]["headers"][header])
        for event in sentry.events
        if header in (event.get("request") or {}).get("headers", {})
    } == {("error", "[Filtered]"), ("transaction", "[Filtered]")}


async def test_a_credential_header_copied_into_any_variable_is_masked(
    sentry: CapturingTransport,
) -> None:
    """Whatever a route names the variable it keeps a credential header in."""
    long_body = "x" * 1008 + f"key: {API_KEY} and more" + "y" * 2000

    async def lookup(request: Request) -> None:
        cookie_lookup = dict(request.cookies)
        header_value = request.headers["x-api-key"]
        body = long_body.replace(API_KEY, header_value)
        raw = body.encode()
        raise RuntimeError(f"No match among {len(cookie_lookup)} for {len(raw)}")

    async with _behind_the_rest_middleware("/lookup", lookup) as client:
        response = await client.get(
            "/lookup",
            headers={
                "Cookie": f"__Secure-better-auth.session_token={SESSION_COOKIE}",
                "X-API-Key": API_KEY,
            },
        )
    assert response.status_code == 500

    for credential in (SESSION_COOKIE, API_KEY):
        assert not _leaks(credential, sentry.payload)
    variables = _frame_variables(sentry.events)["lookup"]
    assert variables["cookie_lookup"] == {
        "__Secure-better-auth.session_token": f"'{REDACTED}'"
    }
    assert variables["header_value"] == f"'{REDACTED}'"
    # A long variable is cut where the SDK is told to cut it, and a credential
    # across the cut is masked before it, not left half there.
    for name in ("body", "raw"):
        assert variables[name].endswith(f"... ({len(long_body):,} long)")
        assert not _leaks(API_KEY, variables[name])
        assert f"key: {REDACTED}" in variables[name]


def test_a_value_longer_than_any_credential_is_not_kept() -> None:
    """Every kept credential is searched for in each string of an event, and
    sets how far a long frame variable is read."""
    too_long = "t" * (MAX_CREDENTIAL_LENGTH + 1)
    longest = "l" * MAX_CREDENTIAL_LENGTH
    assert select_credentials({too_long, API_KEY}, {longest}) == (longest, API_KEY)


async def test_a_request_keeps_at_most_its_cap_of_credentials_headers_first() -> None:
    kept: list[str] = []

    async def lookup(request: Request) -> None:
        kept.extend(request_credentials())

    tokens = {f"token{i}": f"query-credential-{i:02d}-{'q' * 20}" for i in range(40)}
    async with _behind_the_rest_middleware("/lookup", lookup) as client:
        await client.get("/lookup", params=tokens, headers={"X-API-Key": API_KEY})

    assert len(kept) == MAX_REQUEST_CREDENTIALS
    assert API_KEY in kept


def test_every_service_app_keeps_its_requests_credentials() -> None:
    """One that doesn't sends the copies its routes make of them unmasked."""

    class Service(AppService):
        @classmethod
        def get_port(cls) -> int:
            return 0

    for app in (rest_api.app, ws_api.app, Service().build_fastapi_app()):
        middleware = {entry.cls for entry in app.user_middleware}
        assert RequestCredentialsMiddleware in middleware, app


async def test_a_query_full_of_tokens_leaves_room_for_the_headers(
    sentry: CapturingTransport,
) -> None:
    """A caller can send any number of credential-named query values, longer
    than any header's, and the header's credential is still masked."""
    query = {f"token{index}": f"{index:02d}" + "q" * 46 for index in range(20)}

    async def lookup(request: Request) -> None:
        header_value = request.headers["x-api-key"]
        raise RuntimeError(f"No match for {len(header_value)}")

    async with _behind_the_rest_middleware("/lookup", lookup) as client:
        await client.get("/lookup", params=query, headers={"X-API-Key": API_KEY})

    assert not _leaks(API_KEY, sentry.payload)
    assert _frame_variables(sentry.events)["lookup"]["header_value"] == f"'{REDACTED}'"


# --- 2. Internal RPC and REST errors ------------------------------------------


async def test_a_service_logs_no_input_that_failed_validation(
    service_output: io.StringIO, sentry: CapturingTransport
) -> None:
    app = FastAPI()

    @app.post("/store_credentials")
    async def store_credentials() -> None:
        # The provider is missing, so pydantic quotes the whole object.
        APIKeyCredentials.model_validate({"api_key": API_KEY, "title": "Gemini"})

    AppService._register_exception_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://service"
    ) as client:
        response = await client.post("/store_credentials")
    assert response.status_code == 400

    logs = service_output.getvalue()
    assert not _leaks(API_KEY, logs)
    assert (
        "POST /store_credentials failed: 1 validation error for APIKeyCredentials: "
        "[{'type': 'missing', 'loc': ['provider'], 'msg': 'Field required'}]"
    ) in logs
    # The traceback is kept: it says where validation failed.
    assert "in store_credentials" in logs
    assert "APIKeyCredentials.model_validate(" in logs


@pytest.mark.parametrize(
    "add_handlers",
    [AppService._register_exception_handlers, add_exception_handlers],
    ids=["service", "rest"],
)
async def test_an_error_logs_no_key_from_an_outgoing_url(
    service_output: io.StringIO,
    sentry: CapturingTransport,
    add_handlers: Callable[[FastAPI], None],
) -> None:
    app = FastAPI()

    @app.post("/list_models")
    async def list_models() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(_bad_request)) as c:
            response = await c.get(_GEMINI_MODELS, params={"key": API_KEY})
            response.raise_for_status()

    add_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://service",
    ) as client:
        response = await client.post("/list_models")
    assert response.status_code == 500

    logs = service_output.getvalue()
    assert not _leaks(API_KEY, logs)
    assert not _leaks(API_KEY, sentry.payload)
    # The error reached Sentry, masked the same way.
    assert f"models?key={REDACTED}" in sentry.payload
    # In the message and in the traceback's, whose frames are kept.
    assert logs.count(f"{_GEMINI_MODELS}?key={REDACTED}") >= 2
    assert "response.raise_for_status()" in logs


async def test_a_rejected_mappings_keys_stay_out_of_the_log(
    service_output: io.StringIO,
) -> None:
    """A validation error's location names fields, list indices and the keys of
    a mapping, which are the client's own data. The response keeps them."""
    app = FastAPI()

    @app.post("/limits")
    async def set_limits(limits: Limits) -> None:
        return None

    add_exception_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://backend"
    ) as client:
        response = await client.post(
            "/limits", json={"per_key": {API_KEY: "many", "daily": 5}}
        )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "per_key", API_KEY]

    logs = service_output.getvalue()
    assert not _leaks(API_KEY, logs)
    assert "'loc': ['body', 'per_key', '*']" in logs


async def test_a_rejected_tag_stays_out_of_the_log(service_output: io.StringIO) -> None:
    """pydantic quotes a union's rejected tag in its message, and the input it
    names is the whole object around it."""
    app = FastAPI()

    @app.post("/filed")
    async def file_item(filed: Filed) -> None:
        return None

    add_exception_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://backend"
    ) as client:
        response = await client.post("/filed", json={"item": {"type": API_KEY}})
    assert response.status_code == 422

    logs = service_output.getvalue()
    assert not _leaks(API_KEY, logs)
    assert f"Input tag '{REDACTED}' found using 'type'" in logs
    assert "'loc': ['body', 'item']" in logs


@pytest.mark.parametrize(
    "item",
    [
        pytest.param({"type": "hunter2"}, id="short"),
        pytest.param(
            {**{f"field{i}": "x" for i in range(300)}, "type": API_KEY},
            id="past-what-is-read",
        ),
    ],
)
async def test_a_rejected_tag_stays_out_of_the_log_whatever_its_length_or_place(
    service_output: io.StringIO, item: dict[str, str]
) -> None:
    app = FastAPI()

    @app.post("/filed")
    async def file_item(filed: Filed) -> None:
        return None

    add_exception_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://backend"
    ) as client:
        response = await client.post("/filed", json={"item": item})
    # The response keeps pydantic's own message.
    assert item["type"] in response.json()["detail"][0]["msg"]

    logs = service_output.getvalue()
    assert item["type"] not in logs
    assert (
        f"Input tag '{REDACTED}' found using 'type' does not match any of the "
        "expected tags: 'secret', 'note'"
    ) in logs


async def test_a_validators_message_stays_out_of_the_log(
    service_output: io.StringIO,
) -> None:
    """A validator can quote what was sent in any form, here escaped by
    ``!r``, which no search for the value as sent would find."""
    app = FastAPI()

    @app.post("/checked")
    async def check(body: Checked) -> None:
        return None

    add_exception_handlers(app)
    token = f"first line\n{SESSION_COOKIE}"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://backend"
    ) as client:
        response = await client.post("/checked", json={"token": token})
    # The response keeps the validator's message.
    assert "unrecognised token" in response.json()["detail"][0]["msg"]

    logs = service_output.getvalue()
    assert not _leaks(SESSION_COOKIE, logs)
    assert "'loc': ['body', 'token']" in logs
    assert f"'msg': 'Value error, {REDACTED}'" in logs


async def test_a_key_refused_as_extra_stays_out_of_the_log(
    service_output: io.StringIO,
) -> None:
    """A key a model refuses is the client's own, even one that another model
    declares as a field, as `Limits` does `per_key`."""
    app = FastAPI()

    @app.post("/strict")
    async def strict(body: Strict) -> None:
        return None

    add_exception_handlers(app)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://backend"
    ) as client:
        response = await client.post("/strict", json={"name": "n", "per_key": 1})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "per_key"]

    logs = service_output.getvalue()
    assert "'loc': ['body', '*']" in logs
    assert "per_key" not in logs


# --- 3. Webhook logs ---------------------------------------------------------


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
        integrations_router, "is_user_refused", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        GithubWebhooksManager,
        "verify_signature",
        # The first delivery's signature is malformed, and the next one's good.
        AsyncMock(side_effect=[RuntimeError("malformed signature"), None]),
    )

    # A delivery that gets through is published on a channel of its webhook's,
    # which the event bus logs when Redis is down. Each attempt fails afresh, as
    # a connection does: an exception raised again keeps the frames, and so the
    # variables, of every request that raised it before.
    async def redis_down() -> None:
        raise ConnectionError("Redis is down")

    monkeypatch.setattr(redis_client, "get_redis_async", redis_down)
    async with (
        serve(rest_api.AgentServer(), monkeypatch) as host,
        httpx.AsyncClient(base_url=f"http://{host}") as client,
    ):
        rejected = [
            await client.post(f"/api/integrations/github/webhooks/{id}/ingress")
            for id in (UNKNOWN_WEBHOOK_ID, COMPASS_WEBHOOK_ID, WEBHOOK_ID)
        ]
        delivered = await client.post(
            f"/api/integrations/compass/webhooks/{COMPASS_WEBHOOK_ID}/ingress",
            json={"transcript": "hello"},
        )
        # GitHub's event type ends with the delivery's action, whatever it says.
        described = await client.post(
            f"/api/integrations/github/webhooks/{WEBHOOK_ID}/ingress",
            headers={"X-GitHub-Event": "issues"},
            json={"action": "left at the door, code 4417"},
        )

    logs = service_output.getvalue()
    # Both failures reach Sentry, so what it is sent can be checked too.
    for reported in (
        f"Signature verification failed for webhook #{WEBHOOK_ID[:8]}... (github)",
        f"Failed to publish event to Redis channel {COMPASS_WEBHOOK_ID[:8]}.../",
    ):
        assert reported in sentry.payload
    for credential in (UNKNOWN_WEBHOOK_ID, COMPASS_WEBHOOK_ID, WEBHOOK_ID):
        # Logs show a webhook's ID by its first 8 characters.
        assert not _leaks(credential[8:], logs)
        assert not _leaks(credential[8:], sentry.payload)
    assert not _leaks(SIGNING_SECRET, logs)
    assert "left at the door" not in logs
    # A webhook registered for another provider reads as one that doesn't exist.
    assert [response.json() for response in rejected] == [
        {"detail": f"Webhook #{UNKNOWN_WEBHOOK_ID[:8]}... not found"},
        {"detail": f"Webhook #{COMPASS_WEBHOOK_ID[:8]}... not found"},
        {"detail": "Invalid webhook signature"},
    ]
    assert delivered.status_code == described.status_code == 200
    for line in (
        f"Received github webhook ingress for ID {WEBHOOK_ID[:8]}...",
        f"unknown webhook #{UNKNOWN_WEBHOOK_ID[:8]}...",
        f"Webhook #{COMPASS_WEBHOOK_ID[:8]}... provider mismatch",
        f"Webhook #{WEBHOOK_ID[:8]}...: {{'user_id': 'user-1', 'provider'",
        f"Signature verification failed for webhook #{WEBHOOK_ID[:8]}... (github)",
        f"Failed to publish event to Redis channel {COMPASS_WEBHOOK_ID[:8]}.../",
        f"Publishing transcription event for webhook #{COMPASS_WEBHOOK_ID[:8]}...",
        "Validated github repo [unlisted] event",
        f"Publishing [unlisted] event for webhook #{WEBHOOK_ID[:8]}...",
        f"Redis channel {WEBHOOK_ID[:8]}.../[unlisted]",
    ):
        assert line in logs


async def test_pruning_a_webhook_logs_its_id_only_by_prefix(
    service_output: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(IntegrationWebhook, "prisma", lambda: _WebhookTable({}))

    await triggers._prune_dangling_webhook("user-1", WEBHOOK_ID)

    logs = service_output.getvalue()
    assert not _leaks(WEBHOOK_ID[8:], logs)
    assert (
        f"Best-effort prune of webhook #{WEBHOOK_ID[:8]}... failed: "
        f"Webhook #{WEBHOOK_ID[:8]}... not found"
    ) in logs


@pytest.mark.parametrize("log_level", ["DEBUG"])
async def test_setting_up_a_webhook_logs_no_secret_or_ingress_credential(
    service_output: io.StringIO, monkeypatch: pytest.MonkeyPatch, log_level: str
) -> None:
    async def stored(webhook: integrations_db.Webhook) -> integrations_db.Webhook:
        return webhook

    monkeypatch.setattr(
        integrations_db, "find_webhook_by_graph_and_props", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(integrations_db, "create_webhook", stored)
    for config in (webhooks_base.app_config, webhooks_utils.app_config):
        monkeypatch.setattr(config, "platform_base_url", "https://platform.example")

    webhook, feedback = await setup_webhook_for_block(
        "user-1",
        GenericWebhookTriggerBlock(),
        # A stored config can carry a key the block doesn't declare.
        {
            "secret_token": TRIGGER_SECRET,
            "constants": {"team": "blue"},
            f"note_{API_KEY}": "x",
        },
        for_graph_id="graph-1",
    )
    assert feedback is None and webhook is not None
    # Registering a manual webhook only logs where it receives deliveries.
    await GenericWebhooksManager()._register_webhook(
        APIKeyCredentials(provider="x", api_key=SecretStr(API_KEY), title="x"),
        GenericWebhookType.PLAIN,
        "",
        [],
        webhook.url,
        webhook.secret,
    )

    # A resource is built from the trigger's inputs, such as a GitHub repo,
    # which can hold whatever the user typed.
    repo = API_KEY
    unset, reason = await setup_webhook_for_block(
        "user-1",
        GithubPullRequestTriggerBlock(),
        {"repo": repo, "events": {"opened": True}},
        for_graph_id="graph-1",
    )
    assert (
        unset is None and reason == "Cannot set up github webhook without credentials"
    )

    logs = service_output.getvalue()
    for secret in (TRIGGER_SECRET, webhook.secret, webhook.id[8:], API_KEY):
        assert not _leaks(secret, logs)
    for line in (
        "Setting up webhook for block #8fa8c167-2002-47ce-aba8-97572fc5d387 with "
        "inputs ['constants', 'secret_token']",
        f"Acquired webhook #{webhook.id[:8]}... (generic_webhook plain)",
        "Manual webhook registered with ingress URL: "
        + webhook.url.replace(webhook.id, REDACTED),
        "Constructed resource string from {repo}",
    ):
        assert line in logs


@pytest.mark.parametrize("log_level", ["DEBUG"])
def test_the_event_bus_logs_an_events_size_not_the_event(
    service_output: io.StringIO, log_level: str
) -> None:
    bus = integrations_db.WebhookEventBus()
    channel = f"{WEBHOOK_ID}/push"
    event = integrations_db.WebhookEvent(
        provider="github",
        webhook_id=WEBHOOK_ID,
        event_type="push",
        payload={"token": API_KEY},
    )

    message, _ = bus._serialize_message(event, channel)
    assert bus._deserialize_message({"type": "smessage", "data": message}, channel)
    # One this release can't read, such as another release's.
    unreadable = message.replace('"event_type":"push"', '"event_type":1')
    assert not bus._deserialize_message(
        {"type": "smessage", "data": unreadable}, channel
    )

    logs = service_output.getvalue()
    assert not _leaks(API_KEY, logs)
    assert not _leaks(WEBHOOK_ID[8:], logs)
    for line in (
        f"[webhooks/{WEBHOOK_ID[:8]}.../push] Publishing an event to Redis "
        f"({len(message)} chars)",
        f"[{WEBHOOK_ID[:8]}.../push] Consuming an event from Redis",
        f"Failed to parse an event from Redis [{WEBHOOK_ID[:8]}.../push]: "
        "1 validation error for ",
        "'loc': ['payload', 'event_type']",
    ):
        assert line in logs


@pytest.mark.parametrize(
    "event_type, shown",
    [
        ("push", "push"),
        ("pull_request.opened", "pull_request.opened"),
        ("order.SHIPPED", "order.SHIPPED"),
        ("*", "*"),
        ("order.left at the door, code 4417", "[unlisted]"),
        ("order.x\nERROR a forged line", "[unlisted]"),
        ("order." + "x" * 64, "[unlisted]"),
        (f"order.{WEBHOOK_ID}", f"order.{WEBHOOK_ID[:8]}..."),
    ],
)
def test_webhook_logs_show_an_event_type_only_if_shaped_like_one(
    event_type: str, shown: str
) -> None:
    """Most event types come from the delivery, such as a Slant3D order's
    status, and the logs show only those that read as one."""
    assert loggable_event_type(event_type, WEBHOOK_ID) == shown
    channel = f"{WEBHOOK_ID}/{event_type}"
    assert integrations_db.WebhookEventBus()._loggable_key(channel) == (
        f"{WEBHOOK_ID[:8]}.../{shown}"
    )


async def test_a_wildcard_channel_names_its_webhook_only_by_prefix() -> None:
    bus = integrations_db.WebhookEventBus()
    event = integrations_db.WebhookEvent(
        provider="github", webhook_id=WEBHOOK_ID, event_type="*", payload={}
    )

    with pytest.raises(ValueError) as refused:
        await bus.publish_event(event, f"{WEBHOOK_ID}/*")

    assert not _leaks(WEBHOOK_ID[8:], str(refused.value))
    assert f"{WEBHOOK_ID[:8]}.../*" in str(refused.value)


# --- 4. WebSocket handshake headers -------------------------------------------


@pytest.mark.parametrize("log_level", ["DEBUG"])
async def test_websocket_handshake_headers_stay_out_of_the_debug_log(
    service_output: io.StringIO, monkeypatch: pytest.MonkeyPatch, log_level: str
) -> None:
    """At DEBUG, websockets logs each handshake's target and headers: the
    server's on uvicorn's error log, the client's on its own."""
    monkeypatch.setattr(
        ConnectionManager, "_start_notification_subscription", AsyncMock()
    )
    monkeypatch.setattr(ws_api, "_connection_manager", None)
    monkeypatch.setattr(
        ws_api, "authenticate_websocket", AsyncMock(return_value="user-1")
    )
    async with serve(ws_api.WebsocketServer(), monkeypatch) as host:
        async with connect(
            f"ws://{host}/ws?token={BEARER_TOKEN}",
            additional_headers={
                "Cookie": f"session={SESSION_COOKIE}",
                "Authorization": f"Bearer {BEARER_TOKEN}",
            },
        ):
            pass

    logs = service_output.getvalue()
    for credential in (SESSION_COOKIE, BEARER_TOKEN):
        assert not _leaks(credential, logs)
    # The client logs what it sends, and the server what uvicorn hands it, with
    # the header names lowercased. Other headers are kept.
    for line in (
        f"> GET /ws?token={REDACTED} HTTP/1.1",
        f"> Cookie: {REDACTED}",
        f"> Authorization: {REDACTED}",
        f"< GET /ws?token={REDACTED} HTTP/1.1",
        f"< cookie: {REDACTED}",
        f"< authorization: {REDACTED}",
        f"< host: {host}",
    ):
        assert line in logs


# --- 5. Outgoing request URLs -------------------------------------------------


async def test_sentry_gets_no_credentials_from_outgoing_request_urls(
    sentry: CapturingTransport,
) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(_bad_request)) as client:
        with sentry_sdk.start_transaction(name="outgoing"):
            gemini = await client.get(_GEMINI_MODELS, params={"key": API_KEY})
            await client.post(f"https://api.telegram.org/bot{BOT_TOKEN}/getMe")
        try:
            gemini.raise_for_status()
        except httpx.HTTPStatusError as error:
            sentry_sdk.capture_exception(error)

    assert not _leaks(API_KEY, sentry.payload)
    assert not _leaks(BOT_TOKEN, sentry.payload)
    # The SDK records each request in a breadcrumb, and in a span of the
    # transaction it ran in; both keep the URL and the query's keys.
    [error] = [event for event in sentry.events if event.get("exception")]
    breadcrumbs = [crumb["data"] for crumb in error["breadcrumbs"]["values"]]
    [transaction] = [e for e in sentry.events if e.get("type") == "transaction"]
    spans = [span for span in transaction["spans"] if span["op"] == "http.client"]
    expected = [
        (_GEMINI_MODELS, f"key={REDACTED}"),
        (f"https://api.telegram.org/bot{REDACTED}/getMe", ""),
    ]
    for recorded in (breadcrumbs[-2:], [span["data"] for span in spans]):
        assert [(data["url"], data["http.query"]) for data in recorded] == expected
    assert (
        spans[1]["description"] == f"POST https://api.telegram.org/bot{REDACTED}/getMe"
    )
    assert f"/v1beta/models?key={REDACTED}" in error["exception"]["values"][0]["value"]


async def test_a_value_naming_the_event_is_not_taken_for_a_credential(
    sentry: CapturingTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any caller can send any value as a token; masking the event's own names
    with it would move their errors out of the alerts and issues they belong to."""
    # Pinned, so the test doesn't depend on a configured name being as long as
    # a credential.
    environment = "app:pinned-environment-name"
    monkeypatch.setitem(sentry_sdk.get_client().options, "environment", environment)
    user_id = "user-7a2c9e41b3d5f608"

    async def lookup_that_fails_loudly(request: Request) -> None:
        sentry_sdk.set_user({"id": user_id})
        sent = request.query_params["token"]
        raise LookupFailedLoudly(f"No match for a {len(sent)}-character token")

    async with _behind_the_rest_middleware("/lookup", lookup_that_fails_loudly) as c:
        await c.get(
            "/lookup",
            params={
                "token": environment,
                "key": "lookup_that_fails_loudly",
                "code": "LookupFailedLoudly",
                "state": user_id,
            },
            headers={"X-API-Key": API_KEY},
        )

    [error] = [event for event in sentry.events if event.get("exception")]
    assert error["environment"] == environment
    assert error["user"]["id"] == user_id
    assert error["exception"]["values"][-1]["type"] == "LookupFailedLoudly"
    variables = _frame_variables([error])
    # Where else the value appears, it is masked as the credential it was sent as.
    assert variables["lookup_that_fails_loudly"]["sent"] == f"'{REDACTED}'"
    # A credential the same request carries is still masked.
    assert not _leaks(API_KEY, sentry.payload)


# --- 6. Credentials in keys ---------------------------------------------------


async def test_a_credential_used_as_a_key_is_masked(
    sentry: CapturingTransport,
) -> None:
    """In a mapping a route keeps, and in an outgoing request's query."""

    async def find_run(request: Request) -> None:
        share_token = request.path_params["share_token"]
        runs_by_token = {share_token: "run-1", "fallback": "run-0"}
        async with httpx.AsyncClient(transport=httpx.MockTransport(_bad_request)) as c:
            await c.get("https://api.example.com/runs", params={share_token: ""})
        raise RuntimeError(f"{len(runs_by_token)} runs and none shared")

    async with _behind_the_rest_middleware(
        "/api/public/shared/{share_token}", find_run
    ) as client:
        response = await client.get(f"/api/public/shared/{SHARE_TOKEN}")
    assert response.status_code == 500

    assert not _leaks(SHARE_TOKEN, sentry.payload)
    variables = _frame_variables(sentry.events)["find_run"]
    assert variables["runs_by_token"] == {REDACTED: "'run-1'", "fallback": "'run-0'"}
    [error] = [event for event in sentry.events if event.get("exception")]
    [crumb] = [
        crumb["data"]
        for crumb in error["breadcrumbs"]["values"]
        if crumb.get("type") == "http"
    ]
    assert crumb["http.query"] == f"{REDACTED}={REDACTED}"


def test_a_credential_key_past_the_budget_is_masked() -> None:
    """Past the budget values are filtered, but keys are kept for the event's
    shape, so they are still checked for the request's credentials."""
    event = {
        "request": {"url": f"http://backend/api/public/shared/{SHARE_TOKEN}"},
        "extra": {
            # Leaves less of the budget than the key takes, once the URL is read.
            "filler": "x" * (metrics._SENTRY_SCRUB_BUDGET - 50),
            "runs": {SHARE_TOKEN: "run-1"},
        },
    }

    scrubbed = metrics._before_send(event, {})

    assert scrubbed is not None
    assert scrubbed["extra"]["filler"] != "[Filtered]"
    assert not _leaks(SHARE_TOKEN, json.dumps(scrubbed))
    assert list(scrubbed["extra"]["runs"]) == [REDACTED]


def test_keys_masked_alike_are_all_kept() -> None:
    query = f"token={SESSION_COOKIE}"
    event = {
        "request": {
            "url": f"http://backend/api/public/shared/{SHARE_TOKEN}",
            "query_string": query,
        },
        "extra": {"by": {SHARE_TOKEN: 1, SESSION_COOKIE: 2, REDACTED: 3}},
    }

    scrubbed = metrics._before_send(event, {})

    assert scrubbed is not None
    assert scrubbed["extra"]["by"] == {
        f"{REDACTED}*": 1,
        f"{REDACTED}**": 2,
        REDACTED: 3,
    }


def test_a_secret_field_renamed_for_its_name_is_still_filtered() -> None:
    """A key named like a secret field is filtered by that name even when the
    request also sent the name as a credential, which renames the key."""
    event = {
        "request": {"url": "http://backend/lookup?token=codex_access_token"},
        "extra": {"codex": {"codex_access_token": "plain words, not token-shaped"}},
    }

    scrubbed = metrics._before_send(event, {})

    assert scrubbed is not None
    assert scrubbed["extra"]["codex"] == {REDACTED: "[Filtered]"}


async def test_the_sdks_record_of_a_cut_value_keeps_no_credential_key(
    sentry: CapturingTransport,
) -> None:
    """The SDK records each value it cuts under ``_meta``, at the value's path,
    keys and all."""

    async def find_run(request: Request) -> None:
        share_token = request.path_params["share_token"]
        sentry_sdk.set_extra(
            "runs", {share_token: "x" * 20_000, "fallback": "y" * 20_000}
        )
        raise RuntimeError("no run is shared")

    async with _behind_the_rest_middleware(
        "/api/public/shared/{share_token}", find_run
    ) as client:
        await client.get(f"/api/public/shared/{SHARE_TOKEN}")

    assert not _leaks(SHARE_TOKEN, sentry.payload)
    [error] = [event for event in sentry.events if event.get("exception")]
    # The record of what was cut elsewhere is kept.
    assert list(error["_meta"]["extra"]["runs"]) == ["fallback"]


async def test_an_error_log_masks_the_requests_credentials_wherever_quoted(
    service_output: io.StringIO,
) -> None:
    """The REST server's error handlers and uvicorn's own error log mask the
    request's credentials by value, outside any URL too, as Sentry does."""
    app = FastAPI()

    @app.get("/api/public/shared/{share_token}")
    async def shared(request: Request, share_token: str) -> None:
        raise RuntimeError(f"bad cookie {request.headers['authorization']}")

    @app.get("/api/public/shared/{share_token}/runs")
    async def runs(request: Request, share_token: str) -> None:
        cookie = request.cookies[_SESSION_COOKIE_NAME]
        raise ValueError(f"no run for {share_token} with {cookie}")

    @app.websocket("/ws")
    async def socket(websocket: WebSocket) -> None:
        raise RuntimeError(
            f"refused {websocket.query_params['token']} "
            f"{websocket.headers['authorization']} "
            f"{websocket.cookies[_SESSION_COOKIE_NAME]}"
        )

    add_exception_handlers(app)
    remember_request_credentials(app)
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}",
        "Cookie": f"{_SESSION_COOKIE_NAME}={SESSION_COOKIE}",
    }
    async with (
        serve_app(app, log_config=None, ws="websockets-sansio") as host,
        httpx.AsyncClient(base_url=f"http://{host}", headers=headers) as client,
    ):
        failed = await client.get(f"/api/public/shared/{SHARE_TOKEN}")
        refused = await client.get(f"/api/public/shared/{SHARE_TOKEN}/runs")
        with pytest.raises(InvalidStatus):
            async with connect(
                f"ws://{host}/ws?token={QUERY_TOKEN}", additional_headers=headers
            ):
                pass
    assert (failed.status_code, refused.status_code) == (500, 400)

    logs = service_output.getvalue()
    for credential in (BEARER_TOKEN, SESSION_COOKIE, SHARE_TOKEN, QUERY_TOKEN):
        assert not _leaks(credential, logs)
    for line in (
        f"failed. Investigate and resolve the underlying issue: bad cookie {REDACTED}",
        f"RuntimeError: bad cookie {REDACTED}",
        f"failed with 400: no run for {REDACTED} with {REDACTED}",
        "Exception in ASGI application",
        f"RuntimeError: refused {REDACTED} {REDACTED} {REDACTED}",
    ):
        assert line in logs


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
            {secret: "found"},
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
            "breadcrumbs": {
                "values": [
                    {"message": rng.choice(holders[1:3])},
                    {"data": {"http.query": f"{rng.choice(['', 'a&'])}{secret}"}},
                ]
            },
        }
        transaction = {
            "type": "transaction",
            "request": dict(event["request"]),
            "transaction": event["transaction"],
            "transaction_info": event["transaction_info"],
            # A Redis command names its key, such as a channel named for a
            # credential, and an outgoing request records its URL.
            "spans": [
                {"op": "db.redis", "description": f"SPUBLISH {rng.choice(holders)!r}"},
                {"op": "http.client", "data": {"url": rng.choice(holders)}},
            ],
        }
        for scrubbed in (
            metrics._before_send(event, {}),
            metrics._before_send_transaction(transaction, {}),
        ):
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


@asynccontextmanager
async def _behind_the_rest_middleware(
    path: str, route: Callable
) -> AsyncIterator[httpx.AsyncClient]:
    """A client for ``route``, served behind the REST server's own middleware
    stack as each of its routes is, with no handler of its own for the error."""
    app = FastAPI()
    app.user_middleware = list(rest_api.app.user_middleware)
    app.add_api_route(path, route)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://backend",
    ) as client:
        yield client


def _leaks(credential: str, text: str) -> bool:
    """Whether 8 characters in a row of ``credential`` are in ``text``. pydantic
    cuts a long value in the middle and the SDK cuts one at its end, so a check
    for the whole value passes on what they leave."""
    return any(credential[i : i + 8] in text for i in range(len(credential) - 7))


def _bad_request(request: httpx.Request) -> httpx.Response:
    """What Google answers a key it doesn't know."""
    return httpx.Response(400, request=request)


def _frame_variables(events: list[dict]) -> dict[str, dict]:
    return {
        frame["function"]: frame.get("vars", {})
        for event in events
        for exception in (event.get("exception") or {}).get("values", [])
        for frame in exception["stacktrace"]["frames"]
    }


def _webhook_row(
    webhook_id: str, provider: str, secret: str = SIGNING_SECRET
) -> IntegrationWebhook:
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
        secret=secret,
        providerWebhookId="1",
        AgentNodes=[],
        AgentPresets=[],
        visibility="PRIVATE",
    )


class _UnreachableTable:
    """What a table reads with the database down."""

    async def find_first(self, where: dict, **kwargs: object) -> None:
        raise ClientNotConnectedError()

    async def find_unique(self, where: dict, **kwargs: object) -> None:
        raise ClientNotConnectedError()


class _WebhookTable:
    """What ``IntegrationWebhook.prisma()`` reads, without a database."""

    def __init__(self, rows: dict[str, IntegrationWebhook]) -> None:
        self.rows = rows

    async def find_unique(self, where: dict, include: object = None):
        return self.rows.get(where["id"])
