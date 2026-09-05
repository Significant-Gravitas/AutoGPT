"""Signed billing-card events must refresh trials, including repeated delivery."""

import hashlib
import hmac
import json
import time
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest

from backend.api.features import v1

app = fastapi.FastAPI()
app.include_router(v1.v1_router)
client = fastapi.testclient.TestClient(app, raise_server_exceptions=False)
TEST_SECRET = "whsec_trial_billing_test_only"


@pytest.fixture
def billing_refresh(mocker):
    mocker.patch.object(v1.settings.secrets, "stripe_webhook_secret", TEST_SECRET)
    mocker.patch.object(v1, "_claim_stripe_event", AsyncMock(return_value=True))
    return mocker.patch.object(v1, "sync_trials_for_billing_event", AsyncMock())


def signed_request(event_type, data=None, secret=TEST_SECRET):
    event = {
        "id": "evt_billing_test",
        "object": "event",
        "api_version": "2023-10-16",
        "type": event_type,
        "data": data or {"object": {"id": "cus_trial"}},
    }
    payload = json.dumps(event)
    timestamp = int(time.time())
    signature = hmac.new(
        secret.encode(), f"{timestamp}.{payload}".encode(), hashlib.sha256
    ).hexdigest()
    return client.post(
        "/credits/stripe_webhook",
        content=payload,
        headers={"stripe-signature": f"t={timestamp},v1={signature}"},
    )


@pytest.mark.parametrize(
    "event_type",
    [
        "customer.updated",
        "customer.deleted",
        "payment_method.attached",
        "payment_method.updated",
        "payment_method.automatically_updated",
        "payment_method.detached",
        "setup_intent.succeeded",
        "setup_intent.requires_action",
        "setup_intent.setup_failed",
        "setup_intent.canceled",
    ],
)
def test_signed_card_event_refreshes_trial(event_type, billing_refresh):
    if event_type.startswith("customer."):
        data = {"object": {"id": "cus_trial"}}
    elif event_type.startswith("setup_intent."):
        data = {"object": {"id": "seti_trial", "customer": "cus_trial"}}
    else:
        data = {
            "object": {"id": "pm_trial", "customer": None},
            "previous_attributes": {"customer": "cus_trial"},
        }
    response = signed_request(event_type, data)
    assert response.status_code == 200
    billing_refresh.assert_awaited_once_with(event_type, data)


def test_invalid_signature_cannot_refresh_trial(billing_refresh):
    response = signed_request("customer.updated", secret="wrong_secret")
    assert response.status_code == 400
    billing_refresh.assert_not_awaited()


def test_unconfigured_webhook_cannot_refresh_trial(billing_refresh, mocker):
    mocker.patch.object(v1.settings.secrets, "stripe_webhook_secret", "")
    response = signed_request("customer.updated")
    assert response.status_code == 503
    billing_refresh.assert_not_awaited()


def test_unrelated_event_does_not_refresh_trial(billing_refresh):
    assert signed_request("customer.tax_id.created").status_code == 200
    billing_refresh.assert_not_awaited()


def test_repeated_billing_event_always_refreshes_current_state(billing_refresh, mocker):
    claim = mocker.patch.object(
        v1, "_claim_stripe_event", AsyncMock(return_value=False)
    )
    assert signed_request("customer.updated").status_code == 200
    assert signed_request("customer.updated").status_code == 200
    assert billing_refresh.await_count == 2
    claim.assert_not_awaited()


def test_failed_billing_refresh_is_retryable(billing_refresh):
    billing_refresh.side_effect = [RuntimeError("temporary Stripe outage"), None]
    assert signed_request("customer.updated").status_code == 500
    assert signed_request("customer.updated").status_code == 200
    assert billing_refresh.await_count == 2
