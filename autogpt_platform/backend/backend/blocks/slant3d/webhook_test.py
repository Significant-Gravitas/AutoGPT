import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException, Request

from backend.blocks.slant3d._api import TEST_CREDENTIALS
from backend.data.integrations import Webhook
from backend.integrations.webhooks.slant3d import Slant3DWebhooksManager

SECRET = "s" * 64
ENDPOINT = "https://example.com/slant3d/webhook"
PAYLOAD = {
    "event_type": "order.shipped",
    "service": "order-service",
    "platform_id": "platform-1",
    "data": {
        "order": {
            "public_id": "SLANT_123",
            "status": "SHIPPED",
            "tracking_number": "track-1",
        }
    },
}


def webhook(version=2):
    return Webhook.model_construct(
        id="webhook-1",
        resource="platform-1",
        secret=SECRET,
        config={"api_version": version, "endpoint": ENDPOINT},
    )


def request_for(payload, *, timestamp=None, secret=SECRET, signed=True):
    body = json.dumps(payload).encode()
    timestamp = timestamp or str(int(time.time() * 1000))
    digest = hmac.new(
        secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256
    ).hexdigest()
    headers = (
        [
            (b"x-webhook-timestamp", timestamp.encode()),
            (b"x-webhook-signature-256", f"sha256={digest}".encode()),
        ]
        if signed
        else []
    )
    return Request(
        {"type": "http", "headers": headers},
        receive=AsyncMock(
            return_value={"type": "http.request", "body": body, "more_body": False}
        ),
    )


async def test_registration_configures_platform_with_signing_secret():
    manager = Slant3DWebhooksManager()
    with patch.object(
        manager,
        "_platform_request",
        AsyncMock(return_value={"data": {"webhookURL": ""}}),
    ) as request:
        provider_id, config = await manager._register_webhook(
            TEST_CREDENTIALS,
            "orders",
            "platform-1",
            ["order.shipped"],
            ENDPOINT,
            SECRET,
        )
    assert provider_id == "platform-1"
    assert config["api_version"] == 2
    assert config["endpoint"] == ENDPOINT
    assert request.await_args_list[1].args[:2] == ("PATCH", "platform-1")
    assert request.await_args_list[1].kwargs["json"] == {
        "webhookURL": ENDPOINT,
        "webhookSecret": SECRET,
    }


async def test_registration_does_not_overwrite_another_subscription():
    manager = Slant3DWebhooksManager()
    with patch.object(
        manager,
        "_platform_request",
        AsyncMock(return_value={"data": {"webhookURL": "https://another.example.com"}}),
    ) as request:
        with pytest.raises(ValueError, match="different webhook URL"):
            await manager._register_webhook(
                TEST_CREDENTIALS, "orders", "platform-1", [], ENDPOINT, SECRET
            )
    request.assert_awaited_once()


async def test_registration_requires_platform():
    with pytest.raises(ValueError, match="platform_id"):
        await Slant3DWebhooksManager()._register_webhook(
            TEST_CREDENTIALS, "orders", "", [], ENDPOINT, SECRET
        )


async def test_platform_request_uses_v2_bearer_auth():
    response = Mock(ok=True)
    response.json.return_value = {"success": True, "data": {"id": "platform-1"}}
    with patch("backend.integrations.webhooks.slant3d.Requests") as requests:
        requests.return_value.request = AsyncMock(return_value=response)
        await Slant3DWebhooksManager()._platform_request(
            "GET", "platform-1", TEST_CREDENTIALS
        )
    requests.return_value.request.assert_awaited_once_with(
        "GET",
        "https://slant3dapi.com/v2/api/platforms/platform-1",
        headers={
            "Authorization": f"Bearer {TEST_CREDENTIALS.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        },
    )


async def test_signature_and_v2_payload_normalization():
    request = request_for(PAYLOAD)
    await Slant3DWebhooksManager.verify_signature(webhook(), request)
    payload, event = await Slant3DWebhooksManager.validate_payload(
        webhook(), request, None
    )
    assert event == "order.shipped"
    assert payload["orderId"] == "SLANT_123"
    assert payload["trackingNumber"] == "track-1"
    assert payload["carrierCode"] == ""
    assert payload["data"] == PAYLOAD["data"]


@pytest.mark.parametrize("timestamp_offset", [-301_000, 301_000])
async def test_expired_and_future_signatures_are_rejected(timestamp_offset):
    request = request_for(
        PAYLOAD, timestamp=str(int(time.time() * 1000) + timestamp_offset)
    )
    with pytest.raises(HTTPException) as error:
        await Slant3DWebhooksManager.verify_signature(webhook(), request)
    assert error.value.status_code == 403


@pytest.mark.parametrize(
    "options", [{"signed": False}, {"secret": "wrong"}, {"timestamp": "invalid"}]
)
async def test_missing_invalid_or_wrong_signatures_are_rejected(options):
    with pytest.raises(HTTPException) as error:
        await Slant3DWebhooksManager.verify_signature(
            webhook(), request_for(PAYLOAD, **options)
        )
    assert error.value.status_code == 403


async def test_body_tampering_is_rejected():
    request = request_for(PAYLOAD)
    body = await request.body()
    tampered = Request(
        {"type": "http", "headers": request.scope["headers"]},
        receive=AsyncMock(
            return_value={
                "type": "http.request",
                "body": body + b" ",
                "more_body": False,
            }
        ),
    )
    with pytest.raises(HTTPException):
        await Slant3DWebhooksManager.verify_signature(webhook(), tampered)


async def test_payload_cannot_select_another_platform():
    with pytest.raises(ValueError, match="platform does not match"):
        await Slant3DWebhooksManager.validate_payload(
            webhook(), request_for({**PAYLOAD, "platform_id": "another-platform"}), None
        )


async def test_test_deliveries_do_not_trigger_order_events():
    _, event = await Slant3DWebhooksManager.validate_payload(
        webhook(), request_for({**PAYLOAD, "dummy": True}), None
    )
    assert event == "dummy"


async def test_non_order_notifications_can_be_filtered_without_parsing_an_order():
    payload = {
        **PAYLOAD,
        "event_type": "filament.updated",
        "data": {"filaments": {"added": ["filament-1"]}},
    }
    result, event = await Slant3DWebhooksManager.validate_payload(
        webhook(), request_for(payload), None
    )
    assert result == payload
    assert event == "filament.updated"


async def test_existing_legacy_subscriptions_remain_compatible():
    payload = {
        "orderId": "123",
        "status": "SHIPPED",
        "trackingNumber": "track-1",
        "carrierCode": "usps",
    }
    request = request_for(payload, signed=False)
    await Slant3DWebhooksManager.verify_signature(webhook(version=1), request)
    result, event = await Slant3DWebhooksManager.validate_payload(
        webhook(version=1), request, None
    )
    assert result == payload
    assert event == "order.shipped"


@pytest.mark.parametrize(
    "current_url,expected_calls", [(ENDPOINT, 2), ("https://another.example.com", 1)]
)
async def test_deregistration_only_clears_its_own_url(current_url, expected_calls):
    manager = Slant3DWebhooksManager()
    with patch.object(
        manager,
        "_platform_request",
        AsyncMock(return_value={"data": {"webhookURL": current_url}}),
    ) as request:
        await manager._deregister_webhook(webhook(), TEST_CREDENTIALS)
    assert request.await_count == expected_calls
    if expected_calls == 2:
        assert request.call_args.kwargs["json"] == {"webhookURL": ""}
