"""Tests for RMFG webhook registration, signature checks and event mapping."""

import hashlib
import hmac
import json
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException, Request
from pydantic import SecretStr

from backend.blocks.rmfg._webhook import (
    EVENT_TYPES,
    SIGNING_SECRET_KEY,
    TIMESTAMP_TOLERANCE_SECONDS,
    RMFGWebhooksManager,
    RMFGWebhookType,
    _parse_timestamp,
)
from backend.data import integrations
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.providers import ProviderName

SIGNING_SECRET = "whsec_rmfg_test"  # pragma: allowlist secret
PAYLOAD = json.dumps(
    {
        "id": "evt_1",
        "type": "order.status_changed",
        "created_at": "2026-09-08T16:00:00Z",
        "data": {"id": "ord_1", "object": "order", "status": "shipped"},
    }
).encode()


def make_webhook(config: dict | None = None) -> integrations.Webhook:
    return integrations.Webhook(
        id="webhook_123",
        user_id="user_123",
        provider=ProviderName("rmfg"),
        credentials_id="creds_123",
        webhook_type=RMFGWebhookType.ACCOUNT,
        resource="",
        events=["order_status_changed"],
        config={SIGNING_SECRET_KEY: SIGNING_SECRET} if config is None else config,
        secret="platform-secret-not-used-by-rmfg",
        provider_webhook_id="whe_123",
    )


def make_request(headers: dict[str, str], body: bytes = PAYLOAD) -> Request:
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/ingress",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )
    request._body = body
    return request


def sign(timestamp: str, body: bytes = PAYLOAD, secret: str = SIGNING_SECRET) -> str:
    digest = hmac.new(
        secret.encode(),
        msg=f"{timestamp}.".encode() + body,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return f"v1={digest}"


def signed_request(timestamp: str | None = None, **overrides: str) -> Request:
    timestamp = timestamp or str(int(time.time()))
    headers = {
        "X-RMFG-Timestamp": timestamp,
        "X-RMFG-Signature": sign(timestamp),
        **overrides,
    }
    return make_request(headers)


def api_key_credentials() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="creds_123",
        provider="rmfg",
        api_key=SecretStr("rmfg_test_key"),
        title="RMFG API key",
        expires_at=None,
    )


class _FakeResponse:
    def __init__(self, status: int, payload: Any = None):
        self.status = status
        self._payload = payload

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


class TestVerifySignature:
    async def test_accepts_a_valid_signature(self):
        await RMFGWebhooksManager.verify_signature(make_webhook(), signed_request())

    async def test_accepts_iso_timestamps(self):
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        await RMFGWebhooksManager.verify_signature(
            make_webhook(), signed_request(timestamp=now)
        )

    async def test_accepts_any_v1_entry_in_a_list(self):
        timestamp = str(int(time.time()))
        request = make_request(
            {
                "X-RMFG-Timestamp": timestamp,
                "X-RMFG-Signature": f"v1=deadbeef,{sign(timestamp)}",
            }
        )
        await RMFGWebhooksManager.verify_signature(make_webhook(), request)

    async def test_rejects_a_missing_signature_header(self):
        request = make_request({"X-RMFG-Timestamp": str(int(time.time()))})
        with pytest.raises(HTTPException) as exc_info:
            await RMFGWebhooksManager.verify_signature(make_webhook(), request)
        assert exc_info.value.status_code == 403

    async def test_rejects_a_wrong_secret(self):
        timestamp = str(int(time.time()))
        request = make_request(
            {
                "X-RMFG-Timestamp": timestamp,
                "X-RMFG-Signature": sign(timestamp, secret="someone-else"),
            }
        )
        with pytest.raises(HTTPException) as exc_info:
            await RMFGWebhooksManager.verify_signature(make_webhook(), request)
        assert exc_info.value.status_code == 403
        assert "mismatch" in exc_info.value.detail

    async def test_rejects_a_tampered_body(self):
        timestamp = str(int(time.time()))
        request = make_request(
            {"X-RMFG-Timestamp": timestamp, "X-RMFG-Signature": sign(timestamp)},
            body=PAYLOAD + b" ",
        )
        with pytest.raises(HTTPException):
            await RMFGWebhooksManager.verify_signature(make_webhook(), request)

    async def test_rejects_a_stale_timestamp(self):
        old = str(int(time.time()) - TIMESTAMP_TOLERANCE_SECONDS - 60)
        with pytest.raises(HTTPException) as exc_info:
            await RMFGWebhooksManager.verify_signature(
                make_webhook(), signed_request(timestamp=old)
            )
        assert "outside the window" in exc_info.value.detail

    async def test_rejects_an_unparseable_timestamp(self):
        with pytest.raises(HTTPException) as exc_info:
            await RMFGWebhooksManager.verify_signature(
                make_webhook(), signed_request(timestamp="yesterday")
            )
        assert exc_info.value.status_code == 403

    async def test_non_ascii_signature_is_a_403_not_a_500(self):
        request = make_request(
            {"X-RMFG-Timestamp": str(int(time.time())), "X-RMFG-Signature": "v1=é"}
        )
        with pytest.raises(HTTPException) as exc_info:
            await RMFGWebhooksManager.verify_signature(make_webhook(), request)
        assert exc_info.value.status_code == 403

    async def test_missing_stored_secret_is_a_server_error(self):
        with pytest.raises(HTTPException) as exc_info:
            await RMFGWebhooksManager.verify_signature(
                make_webhook(config={}), signed_request()
            )
        assert exc_info.value.status_code == 500


class TestParseTimestamp:
    def test_epoch_seconds(self):
        assert _parse_timestamp("1700000000") == 1700000000.0

    def test_iso_with_z(self):
        assert _parse_timestamp("1970-01-01T00:00:10Z") == 10.0

    def test_naive_iso_is_treated_as_utc(self):
        assert _parse_timestamp("1970-01-01T00:00:10") == 10.0

    def test_garbage_is_none(self):
        assert _parse_timestamp("soon") is None


class TestValidatePayload:
    async def test_maps_the_rmfg_type_to_the_filter_key(self):
        payload, event_type = await RMFGWebhooksManager.validate_payload(
            make_webhook(), make_request({}), None
        )
        assert payload["id"] == "evt_1"
        # The trigger block filters on its own field names, so the platform
        # must see the filter key rather than the dotted API type.
        assert event_type == "order_status_changed"

    async def test_unknown_types_pass_through_unmatched(self):
        body = json.dumps({"id": "evt_2", "type": "something.new"}).encode()
        _, event_type = await RMFGWebhooksManager.validate_payload(
            make_webhook(), make_request({}, body=body), None
        )
        assert event_type == "something.new"

    async def test_missing_type_is_a_400(self):
        body = json.dumps({"id": "evt_3"}).encode()
        with pytest.raises(HTTPException) as exc_info:
            await RMFGWebhooksManager.validate_payload(
                make_webhook(), make_request({}, body=body), None
            )
        assert exc_info.value.status_code == 400

    async def test_non_object_body_is_a_400(self):
        with pytest.raises(HTTPException) as exc_info:
            await RMFGWebhooksManager.validate_payload(
                make_webhook(), make_request({}, body=b"[1, 2]"), None
            )
        assert exc_info.value.status_code == 400


class TestEventMapping:
    def test_every_filter_key_is_a_valid_identifier(self):
        # Filter keys become fields on the trigger block's EventsFilter model.
        for key in EVENT_TYPES:
            assert key.isidentifier(), key

    def test_covers_every_documented_event(self):
        assert set(EVENT_TYPES.values()) == {
            "design.ready",
            "design.failed",
            "dfm_report.production_files.ready",
            "dfm_report.production_files.failed",
            "quote.ready",
            "quote.failed",
            "cart.checked_out",
            "order.status_changed",
        }


class TestRegistration:
    async def test_registers_dotted_events_and_stores_the_secret(self):
        response = _FakeResponse(201, {"id": "whe_1", "secret": "whsec_new"})
        with patch("backend.blocks.rmfg._webhook.Requests") as requests_cls:
            requests_cls.return_value.post = AsyncMock(return_value=response)

            endpoint_id, config = await RMFGWebhooksManager()._register_webhook(
                api_key_credentials(),
                RMFGWebhookType.ACCOUNT,
                "",
                ["order_status_changed", "cart_checked_out"],
                "https://platform.example/ingress",
                "platform-secret",
            )

            post = requests_cls.return_value.post
            url = post.await_args.args[0]
            kwargs = post.await_args.kwargs

        assert endpoint_id == "whe_1"
        assert config == {SIGNING_SECRET_KEY: "whsec_new"}
        assert url == "https://api.rmfg.com/v1/webhook-endpoints"
        assert kwargs["json"]["url"] == "https://platform.example/ingress"
        assert kwargs["json"]["events"] == ["order.status_changed", "cart.checked_out"]
        assert kwargs["headers"]["Authorization"] == "Bearer rmfg_test_key"

    async def test_refuses_unknown_events_before_calling_out(self):
        with patch("backend.blocks.rmfg._webhook.Requests") as requests_cls:
            with pytest.raises(ValueError, match="Unknown RMFG events: nope"):
                await RMFGWebhooksManager()._register_webhook(
                    api_key_credentials(),
                    RMFGWebhookType.ACCOUNT,
                    "",
                    ["nope"],
                    "https://platform.example/ingress",
                    "s",
                )
            requests_cls.return_value.post.assert_not_called()

    async def test_refuses_a_registration_without_a_secret(self):
        response = _FakeResponse(201, {"id": "whe_1"})
        with patch("backend.blocks.rmfg._webhook.Requests") as requests_cls:
            requests_cls.return_value.post = AsyncMock(return_value=response)
            with pytest.raises(ValueError, match="signing secret"):
                await RMFGWebhooksManager()._register_webhook(
                    api_key_credentials(),
                    RMFGWebhookType.ACCOUNT,
                    "",
                    ["design_ready"],
                    "https://platform.example/ingress",
                    "s",
                )

    async def test_surfaces_the_api_error_message(self):
        response = _FakeResponse(
            403, {"error": {"type": "permission_error", "message": "scope missing"}}
        )
        with patch("backend.blocks.rmfg._webhook.Requests") as requests_cls:
            requests_cls.return_value.post = AsyncMock(return_value=response)
            with pytest.raises(ValueError, match="scope missing"):
                await RMFGWebhooksManager()._register_webhook(
                    api_key_credentials(),
                    RMFGWebhookType.ACCOUNT,
                    "",
                    ["design_ready"],
                    "https://platform.example/ingress",
                    "s",
                )

    async def test_requires_an_api_key(self):
        oauth = OAuth2Credentials(
            id="creds_123",
            provider="rmfg",
            title="RMFG OAuth",
            access_token=SecretStr("token"),
            scopes=[],
        )
        with pytest.raises(ValueError, match="API key"):
            await RMFGWebhooksManager()._register_webhook(
                oauth, RMFGWebhookType.ACCOUNT, "", ["design_ready"], "https://x", "s"
            )


class TestDeregistration:
    async def test_deletes_the_endpoint(self):
        with patch("backend.blocks.rmfg._webhook.Requests") as requests_cls:
            delete = AsyncMock(return_value=_FakeResponse(204))
            requests_cls.return_value.delete = delete

            await RMFGWebhooksManager()._deregister_webhook(
                make_webhook(), api_key_credentials()
            )

            assert delete.await_args.args[0] == (
                "https://api.rmfg.com/v1/webhook-endpoints/whe_123"
            )

    async def test_an_already_deleted_endpoint_is_fine(self):
        with patch("backend.blocks.rmfg._webhook.Requests") as requests_cls:
            requests_cls.return_value.delete = AsyncMock(
                return_value=_FakeResponse(404, {"error": {"message": "gone"}})
            )
            await RMFGWebhooksManager()._deregister_webhook(
                make_webhook(), api_key_credentials()
            )

    async def test_other_failures_are_logged_not_raised(self):
        with patch("backend.blocks.rmfg._webhook.Requests") as requests_cls:
            requests_cls.return_value.delete = AsyncMock(
                return_value=_FakeResponse(500, {"error": {"message": "boom"}})
            )
            await RMFGWebhooksManager()._deregister_webhook(
                make_webhook(), api_key_credentials()
            )
