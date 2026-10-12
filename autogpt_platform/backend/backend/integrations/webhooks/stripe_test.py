import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from pydantic import SecretStr

from backend.data import integrations
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.providers import ProviderName
from backend.integrations.webhooks.stripe import (
    STRIPE_TIMESTAMP_TOLERANCE,
    StripeWebhooksManager,
    StripeWebhookType,
)

SIGNING_SECRET = "whsec_test_secret"
PAYLOAD = json.dumps({"type": "customer.subscription.created"}).encode()


def make_webhook(config: dict | None = None) -> integrations.Webhook:
    return integrations.Webhook(
        id="webhook_123",
        user_id="user_123",
        provider=ProviderName.STRIPE,
        credentials_id="creds_123",
        webhook_type=StripeWebhookType.ACCOUNT,
        resource="",
        events=["customer.subscription.created"],
        config={"signing_secret": SIGNING_SECRET} if config is None else config,
        secret="platform-secret-not-used-by-stripe",
        provider_webhook_id="we_123",
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


def sign(timestamp: int, body: bytes = PAYLOAD, secret: str = SIGNING_SECRET) -> str:
    signature = hmac.new(
        secret.encode(),
        msg=f"{timestamp}.{body.decode()}".encode(),
        digestmod=hashlib.sha256,
    ).hexdigest()
    return f"t={timestamp},v1={signature}"


def oauth2_credentials() -> OAuth2Credentials:
    return OAuth2Credentials(
        id="creds_123",
        provider="stripe",
        title="Stripe OAuth",
        access_token=SecretStr("token"),
        scopes=[],
    )


def api_key_credentials() -> APIKeyCredentials:
    return APIKeyCredentials(
        id="creds_123",
        provider="stripe",
        api_key=SecretStr("sk_test_123"),
        title="Stripe API key",
        expires_at=None,
    )


@pytest.mark.asyncio
async def test_verify_signature_accepts_valid_signature() -> None:
    request = make_request({"Stripe-Signature": sign(int(time.time()))})
    await StripeWebhooksManager.verify_signature(make_webhook(), request)


@pytest.mark.asyncio
async def test_verify_signature_rejects_tampered_payload() -> None:
    # Signature is computed over PAYLOAD but a different body is delivered
    request = make_request(
        {"Stripe-Signature": sign(int(time.time()))}, body=b'{"type": "spoofed"}'
    )
    with pytest.raises(HTTPException) as exc_info:
        await StripeWebhooksManager.verify_signature(make_webhook(), request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_signature_accepts_one_of_several_v1_signatures() -> None:
    """Stripe signs with every active secret during a secret rotation."""
    now = int(time.time())
    rotated = sign(now, secret="whsec_previous_secret")
    current = sign(now)
    header = f"{rotated},{current.split(',', 1)[1]}"
    await StripeWebhooksManager.verify_signature(
        make_webhook(), make_request({"Stripe-Signature": header})
    )


@pytest.mark.asyncio
async def test_verify_signature_handles_non_utf8_body() -> None:
    """Signing operates on raw bytes, so an undecodable body must not 500."""
    body = b"\xff\xfe not utf-8"
    now = int(time.time())
    signature = hmac.new(
        SIGNING_SECRET.encode(),
        msg=f"{now}.".encode() + body,
        digestmod=hashlib.sha256,
    ).hexdigest()
    request = make_request({"Stripe-Signature": f"t={now},v1={signature}"}, body=body)
    await StripeWebhooksManager.verify_signature(make_webhook(), request)


@pytest.mark.asyncio
async def test_verify_signature_rejects_wrong_secret() -> None:
    request = make_request(
        {"Stripe-Signature": sign(int(time.time()), secret="whsec_other_secret")}
    )
    with pytest.raises(HTTPException) as exc_info:
        await StripeWebhooksManager.verify_signature(make_webhook(), request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_signature_rejects_replayed_timestamp() -> None:
    stale = int(time.time()) - 3600
    request = make_request({"Stripe-Signature": sign(stale)})
    with pytest.raises(HTTPException) as exc_info:
        await StripeWebhooksManager.verify_signature(make_webhook(), request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sig_header",
    ["garbage", "t=notanumber,v1=abc", "v1=abc", "t=123"],
)
async def test_verify_signature_rejects_malformed_header(sig_header: str) -> None:
    request = make_request({"Stripe-Signature": sig_header})
    with pytest.raises(HTTPException) as exc_info:
        await StripeWebhooksManager.verify_signature(make_webhook(), request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_signature_rejects_non_ascii_signature() -> None:
    """Starlette decodes headers as latin-1, and compare_digest rejects
    non-ASCII str with a TypeError — which would 500 a public endpoint."""
    now = int(time.time())
    request = make_request({"Stripe-Signature": f"t={now},v1=ÿ"})
    with pytest.raises(HTTPException) as exc_info:
        await StripeWebhooksManager.verify_signature(make_webhook(), request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_signature_tolerates_whitespace_around_values() -> None:
    now = int(time.time())
    signature = sign(now).split(",", 1)[1].removeprefix("v1=")
    request = make_request({"Stripe-Signature": f"t= {now} , v1= {signature} "})
    await StripeWebhooksManager.verify_signature(make_webhook(), request)


@pytest.mark.asyncio
async def test_verify_signature_rejects_absurd_timestamp() -> None:
    """`int()` parses unbounded values, so the float subtraction overflows —
    a malformed header must still be a 403, not an unhandled 500."""
    request = make_request({"Stripe-Signature": f"t={'9' * 400},v1=abc"})
    with pytest.raises(HTTPException) as exc_info:
        await StripeWebhooksManager.verify_signature(make_webhook(), request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_signature_accepts_edge_of_tolerance_window() -> None:
    """Just inside the 5-minute window is still a valid delivery."""
    recent = int(time.time()) - (STRIPE_TIMESTAMP_TOLERANCE - 5)
    request = make_request({"Stripe-Signature": sign(recent)})
    await StripeWebhooksManager.verify_signature(make_webhook(), request)


@pytest.mark.asyncio
async def test_verify_signature_rejects_just_outside_tolerance_window() -> None:
    stale = int(time.time()) - (STRIPE_TIMESTAMP_TOLERANCE + 5)
    request = make_request({"Stripe-Signature": sign(stale)})
    with pytest.raises(HTTPException) as exc_info:
        await StripeWebhooksManager.verify_signature(make_webhook(), request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_signature_rejects_missing_header() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await StripeWebhooksManager.verify_signature(make_webhook(), make_request({}))
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_verify_signature_errors_without_signing_secret() -> None:
    request = make_request({"Stripe-Signature": sign(int(time.time()))})
    with pytest.raises(HTTPException) as exc_info:
        await StripeWebhooksManager.verify_signature(make_webhook(config={}), request)
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_validate_payload_returns_event_type() -> None:
    request = make_request({})
    payload, event_type = await StripeWebhooksManager.validate_payload(
        make_webhook(), request, None
    )
    assert event_type == "customer.subscription.created"
    assert payload == json.loads(PAYLOAD)


@pytest.mark.asyncio
async def test_validate_payload_rejects_payload_without_type() -> None:
    request = make_request({}, body=b"{}")
    with pytest.raises(HTTPException) as exc_info:
        await StripeWebhooksManager.validate_payload(make_webhook(), request, None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_register_webhook_form_encodes_events_and_stores_signing_secret() -> None:
    response = AsyncMock()
    response.ok = True
    response.json = lambda: {"id": "we_new", "secret": SIGNING_SECRET}

    with patch(
        "backend.integrations.webhooks.stripe.Requests.post",
        new=AsyncMock(return_value=response),
    ) as mock_post:
        webhook_id, config = await StripeWebhooksManager()._register_webhook(
            credentials=api_key_credentials(),
            webhook_type=StripeWebhookType.ACCOUNT,
            resource="",
            events=["customer.subscription.created", "customer.subscription.updated"],
            ingress_url="https://example.com/api/integrations/stripe/webhooks/1/ingress",
            secret="unused",
        )

    assert webhook_id == "we_new"
    assert config == {"signing_secret": SIGNING_SECRET}

    sent_body = mock_post.call_args.kwargs["data"]
    # The ingress URL must be percent-encoded, not interpolated raw
    assert "https%3A%2F%2Fexample.com" in sent_body
    assert sent_body.count("enabled_events") == 2


@pytest.mark.asyncio
async def test_register_webhook_surfaces_stripe_error_message() -> None:
    response = AsyncMock()
    response.ok = False
    response.status = 400
    response.json = lambda: {"error": {"message": "You have reached the maximum"}}

    with patch(
        "backend.integrations.webhooks.stripe.Requests.post",
        new=AsyncMock(return_value=response),
    ):
        with pytest.raises(ValueError, match="You have reached the maximum"):
            await StripeWebhooksManager()._register_webhook(
                credentials=api_key_credentials(),
                webhook_type=StripeWebhookType.ACCOUNT,
                resource="",
                events=["customer.subscription.created"],
                ingress_url="https://example.com/ingress",
                secret="unused",
            )


@pytest.mark.asyncio
async def test_register_webhook_error_falls_back_when_body_is_not_json() -> None:
    """A 5xx from a proxy in front of Stripe isn't JSON; don't mask it."""
    response = AsyncMock()
    response.ok = False
    response.status = 502
    response.json = MagicMock(side_effect=ValueError("not JSON"))

    with patch(
        "backend.integrations.webhooks.stripe.Requests.post",
        new=AsyncMock(return_value=response),
    ):
        with pytest.raises(ValueError, match="HTTP 502"):
            await StripeWebhooksManager()._register_webhook(
                credentials=api_key_credentials(),
                webhook_type=StripeWebhookType.ACCOUNT,
                resource="",
                events=["customer.subscription.created"],
                ingress_url="https://example.com/ingress",
                secret="unused",
            )


@pytest.mark.asyncio
async def test_register_webhook_requires_api_key_credentials() -> None:
    oauth_credentials = oauth2_credentials()
    with pytest.raises(ValueError):
        await StripeWebhooksManager()._register_webhook(
            credentials=oauth_credentials,
            webhook_type=StripeWebhookType.ACCOUNT,
            resource="",
            events=["customer.subscription.created"],
            ingress_url="https://example.com/ingress",
            secret="unused",
        )


@pytest.mark.asyncio
async def test_deregister_webhook_deletes_endpoint() -> None:
    response = AsyncMock()
    response.status = 200
    response.json = lambda: {}

    with patch(
        "backend.integrations.webhooks.stripe.Requests.delete",
        new=AsyncMock(return_value=response),
    ) as mock_delete:
        await StripeWebhooksManager()._deregister_webhook(
            make_webhook(), api_key_credentials()
        )

    assert mock_delete.call_args.args[0].endswith("/webhook_endpoints/we_123")


@pytest.mark.asyncio
async def test_deregister_webhook_warns_on_delete_failure(caplog) -> None:
    """A failed teardown must be visible: it leaves a live endpoint at Stripe."""
    response = AsyncMock()
    response.status = 500
    response.json = lambda: {"error": {"message": "Something went wrong"}}

    with patch(
        "backend.integrations.webhooks.stripe.Requests.delete",
        new=AsyncMock(return_value=response),
    ):
        await StripeWebhooksManager()._deregister_webhook(
            make_webhook(), api_key_credentials()
        )

    assert "Something went wrong" in caplog.text


@pytest.mark.asyncio
async def test_deregister_webhook_skips_non_api_key_credentials() -> None:
    oauth_credentials = oauth2_credentials()
    with patch(
        "backend.integrations.webhooks.stripe.Requests.delete",
        new=AsyncMock(),
    ) as mock_delete:
        await StripeWebhooksManager()._deregister_webhook(
            make_webhook(), oauth_credentials
        )

    mock_delete.assert_not_called()


@pytest.mark.asyncio
async def test_deregister_webhook_without_endpoint_id_makes_no_request() -> None:
    webhook = make_webhook()
    webhook.provider_webhook_id = ""

    with patch(
        "backend.integrations.webhooks.stripe.Requests.delete",
        new=AsyncMock(),
    ) as mock_delete:
        await StripeWebhooksManager()._deregister_webhook(
            webhook, api_key_credentials()
        )

    mock_delete.assert_not_called()


@pytest.mark.asyncio
async def test_register_webhook_rejects_success_body_without_secret() -> None:
    """A 200 with no signing secret would leave a webhook that can never verify."""
    response = AsyncMock()
    response.ok = True
    response.json = lambda: {"id": "we_new"}

    with patch(
        "backend.integrations.webhooks.stripe.Requests.post",
        new=AsyncMock(return_value=response),
    ):
        with pytest.raises(ValueError, match="no endpoint ID or signing secret"):
            await StripeWebhooksManager()._register_webhook(
                credentials=api_key_credentials(),
                webhook_type=StripeWebhookType.ACCOUNT,
                resource="",
                events=["customer.subscription.created"],
                ingress_url="https://example.com/ingress",
                secret="unused",
            )
