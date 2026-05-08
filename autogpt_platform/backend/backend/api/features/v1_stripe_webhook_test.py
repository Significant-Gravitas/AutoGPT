"""Unit tests for Stripe webhook handler and subscription checkout helpers."""

from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest_mock
import stripe

from backend.data.credit import _list_and_expire_open_subscription_sessions

from .v1 import v1_router

app = fastapi.FastAPI()
app.include_router(v1_router)
client = fastapi.testclient.TestClient(app)


def _make_checkout_event(mode: str, sub_id: str | None = None) -> dict:
    return {
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "id": "cs_test_123",
                "mode": mode,
                "subscription": sub_id,
            }
        },
    }


def test_stripe_webhook_checkout_subscription_syncs_tier(
    mocker: pytest_mock.MockFixture,
) -> None:
    fake_sub = {
        "id": "sub_123",
        "customer": "cus_abc",
        "status": "active",
        "items": {"data": []},
        "metadata": {},
    }
    mocker.patch(
        "stripe.Webhook.construct_event",
        return_value=_make_checkout_event("subscription", "sub_123"),
    )
    mocker.patch(
        "backend.api.features.v1.settings.secrets.stripe_webhook_secret",
        new="whsec_test",
    )
    mocker.patch(
        "backend.api.features.v1.UserCredit.fulfill_checkout", new_callable=AsyncMock
    )
    mocker.patch(
        "backend.api.features.v1.run_in_threadpool",
        new_callable=AsyncMock,
        return_value=fake_sub,
    )
    mock_sync = mocker.patch(
        "backend.api.features.v1.sync_subscription_from_stripe", new_callable=AsyncMock
    )

    response = client.post(
        "/credits/stripe_webhook",
        content=b"{}",
        headers={"stripe-signature": "t=1,v1=sig"},
    )

    assert response.status_code == 200
    mock_sync.assert_called_once_with(fake_sub)


def test_stripe_webhook_checkout_payment_mode_does_not_sync_tier(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "stripe.Webhook.construct_event",
        return_value=_make_checkout_event("payment"),
    )
    mocker.patch(
        "backend.api.features.v1.settings.secrets.stripe_webhook_secret",
        new="whsec_test",
    )
    mocker.patch(
        "backend.api.features.v1.UserCredit.fulfill_checkout", new_callable=AsyncMock
    )
    mock_sync = mocker.patch(
        "backend.api.features.v1.sync_subscription_from_stripe", new_callable=AsyncMock
    )

    response = client.post(
        "/credits/stripe_webhook",
        content=b"{}",
        headers={"stripe-signature": "t=1,v1=sig"},
    )

    assert response.status_code == 200
    mock_sync.assert_not_called()


def test_stripe_webhook_checkout_subscription_stripe_error_does_not_break_webhook(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "stripe.Webhook.construct_event",
        return_value=_make_checkout_event("subscription", "sub_err"),
    )
    mocker.patch(
        "backend.api.features.v1.settings.secrets.stripe_webhook_secret",
        new="whsec_test",
    )
    mocker.patch(
        "backend.api.features.v1.UserCredit.fulfill_checkout", new_callable=AsyncMock
    )
    mocker.patch(
        "backend.api.features.v1.run_in_threadpool",
        new_callable=AsyncMock,
        side_effect=stripe.StripeError("network error"),
    )
    mock_sync = mocker.patch(
        "backend.api.features.v1.sync_subscription_from_stripe", new_callable=AsyncMock
    )

    response = client.post(
        "/credits/stripe_webhook",
        content=b"{}",
        headers={"stripe-signature": "t=1,v1=sig"},
    )

    assert response.status_code == 200
    mock_sync.assert_not_called()


def test_expire_open_subscription_sessions_called_on_checkout(
    mocker: pytest_mock.MockFixture,
) -> None:
    mock_sessions = MagicMock()
    mock_sessions.data = []
    mocker.patch("stripe.checkout.Session.list", return_value=mock_sessions)
    mock_expire = mocker.patch("stripe.checkout.Session.expire")

    _list_and_expire_open_subscription_sessions("cus_test")

    stripe.checkout.Session.list.assert_called_once_with(
        customer="cus_test", status="open", limit=20
    )
    mock_expire.assert_not_called()


def test_expire_open_subscription_sessions_expires_subscription_sessions(
    mocker: pytest_mock.MockFixture,
) -> None:
    sub_session = MagicMock()
    sub_session.id = "cs_sub_open"
    sub_session.mode = "subscription"
    payment_session = MagicMock()
    payment_session.id = "cs_pay_open"
    payment_session.mode = "payment"

    mock_sessions = MagicMock()
    mock_sessions.data = [sub_session, payment_session]
    mocker.patch("stripe.checkout.Session.list", return_value=mock_sessions)
    mock_expire = mocker.patch("stripe.checkout.Session.expire")

    _list_and_expire_open_subscription_sessions("cus_test")

    mock_expire.assert_called_once_with("cs_sub_open")
