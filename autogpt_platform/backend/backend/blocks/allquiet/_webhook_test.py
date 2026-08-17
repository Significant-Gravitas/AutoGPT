"""Tests for All Quiet webhook signature verification."""

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from starlette.requests import Request

from backend.blocks.allquiet._webhook import (
    AllQuietWebhooksManager,
    AllQuietWebhookType,
)

SECRET = "s3cr3t-signing-key"
BODY = b'{"id":"81cd20be","title":"RAM above 60%"}'


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _sign(body: bytes, timestamp: str, secret: str = SECRET) -> str:
    return base64.b64encode(
        hmac.new(
            secret.encode("utf-8"),
            msg=f"{timestamp}:".encode("utf-8") + body,
            digestmod=hashlib.sha256,
        ).digest()
    ).decode("utf-8")


def _request(headers: dict[str, str], body: bytes = BODY) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def _webhook(
    *, node_inputs: list[dict] | None = None, preset_inputs: list[dict] | None = None
):
    return SimpleNamespace(
        id="webhook-1",
        triggered_nodes=[
            SimpleNamespace(input_default=inputs) for inputs in node_inputs or []
        ],
        triggered_presets=[
            SimpleNamespace(inputs=inputs) for inputs in preset_inputs or []
        ],
    )


def _signed_webhook(secret: str = SECRET):
    return _webhook(node_inputs=[{"signing_secret": secret}])


class TestManagerIsConcrete:
    def test_can_be_instantiated(self):
        # Regression: BaseWebhooksManager declares validate_payload abstract, so
        # omitting it still imports and registers fine and only explodes at
        # trigger-setup time with "Can't instantiate abstract class
        # AllQuietWebhooksManager without an implementation for abstract method
        # 'validate_payload'". The block test harness never instantiates the
        # manager, so nothing else catches this.
        AllQuietWebhooksManager()

    def test_implements_every_abstract_method(self):
        assert not getattr(AllQuietWebhooksManager, "__abstractmethods__", frozenset())


class TestValidatePayload:
    async def test_returns_the_body_and_the_incident_event_type(self):
        body = {"id": "inc-1", "title": "RAM above 60%"}
        request = _request({}, body=json.dumps(body).encode())

        payload, event_type = await AllQuietWebhooksManager.validate_payload(
            _webhook(), request
        )

        assert payload == body
        assert event_type == AllQuietWebhookType.INCIDENT

    async def test_rejects_a_body_that_is_not_json(self):
        request = _request({}, body=b"not json at all")

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.validate_payload(_webhook(), request)
        assert exc.value.status_code == 400

    async def test_rejects_a_json_body_that_is_not_an_object(self):
        # A malformed Handlebars template can render a bare array or string.
        request = _request({}, body=b'["not", "an", "object"]')

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.validate_payload(_webhook(), request)
        assert exc.value.status_code == 400
        assert "JSON object" in exc.value.detail


class TestUnsignedWebhooks:
    async def test_accepts_anything_when_no_secret_is_configured(self):
        # Signing is opt-in in All Quiet; the URL is the only credential.
        await AllQuietWebhooksManager.verify_signature(_webhook(), _request({}))

    async def test_treats_a_blank_secret_as_unconfigured(self):
        webhook = _webhook(node_inputs=[{"signing_secret": "   "}])

        await AllQuietWebhooksManager.verify_signature(webhook, _request({}))


class TestSignedWebhooks:
    async def test_accepts_a_valid_allquiet_format_signature(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )

        await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)

    async def test_accepts_a_valid_aws_format_signature(self):
        timestamp = _now()
        request = _request(
            {
                "x-amzn-event-signature": _sign(BODY, timestamp),
                "x-amzn-event-timestamp": timestamp,
            }
        )

        await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)

    async def test_rejects_a_signature_made_with_the_wrong_secret(self):
        timestamp = _now()
        request = _request(
            {
                "x-aq-signature": _sign(BODY, timestamp, secret="not-the-secret"),
                "x-aq-timestamp": timestamp,
            }
        )

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert exc.value.status_code == 403

    async def test_rejects_a_tampered_body(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp},
            body=b'{"id":"attacker-controlled"}',
        )

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert exc.value.status_code == 403

    async def test_rejects_a_signature_bound_to_a_different_timestamp(self):
        # The timestamp is part of the signed material, so swapping it must fail.
        request = _request(
            {
                "x-aq-signature": _sign(BODY, "2020-01-01T00:00:00.000Z"),
                "x-aq-timestamp": _now(),
            }
        )

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert exc.value.status_code == 403

    async def test_rejects_an_unsigned_request_when_a_secret_is_set(self):
        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(
                _signed_webhook(), _request({})
            )

        assert exc.value.status_code == 403
        assert "x-aq-signature" in exc.value.detail

    async def test_rejects_a_signature_with_no_timestamp_header(self):
        timestamp = _now()
        request = _request({"x-aq-signature": _sign(BODY, timestamp)})

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert exc.value.status_code == 403


class TestReplayProtection:
    async def test_rejects_a_stale_but_correctly_signed_delivery(self):
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%S.000Z"
        )
        request = _request({"x-aq-signature": _sign(BODY, old), "x-aq-timestamp": old})

        with pytest.raises(HTTPException) as exc:
            await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)
        assert "window" in exc.value.detail

    async def test_allows_an_unparseable_timestamp_that_signs_correctly(self):
        # The signature already covers the timestamp, so an unfamiliar format
        # must not lock out genuine deliveries.
        odd = "17/12/2023 11:51:08"
        request = _request({"x-aq-signature": _sign(BODY, odd), "x-aq-timestamp": odd})

        await AllQuietWebhooksManager.verify_signature(_signed_webhook(), request)


class TestSecretDiscovery:
    async def test_reads_the_secret_from_a_preset(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )
        webhook = _webhook(preset_inputs=[{"signing_secret": SECRET}])

        await AllQuietWebhooksManager.verify_signature(webhook, request)

    async def test_accepts_a_secret_stored_as_a_secretstr(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )
        webhook = _webhook(node_inputs=[{"signing_secret": SecretStr(SECRET)}])

        await AllQuietWebhooksManager.verify_signature(webhook, request)

    async def test_ignores_nodes_without_a_secret(self):
        timestamp = _now()
        request = _request(
            {"x-aq-signature": _sign(BODY, timestamp), "x-aq-timestamp": timestamp}
        )
        webhook = _webhook(node_inputs=[{}, {"signing_secret": SECRET}])

        await AllQuietWebhooksManager.verify_signature(webhook, request)
