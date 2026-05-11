"""Unit tests for Stripe webhook handler and subscription checkout helpers."""

from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest_mock
import stripe

from backend.data.credit import (
    _expire_open_subscription_sessions,
    reconcile_stripe_tier_for_user,
    sync_tier_from_checkout_session,
)

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


# ---------------------------------------------------------------------------
# Webhook endpoint tests — verify the handler calls the right helpers
# ---------------------------------------------------------------------------


def test_stripe_webhook_checkout_calls_sync_tier_helper(
    mocker: pytest_mock.MockFixture,
) -> None:
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
    mock_sync = mocker.patch(
        "backend.api.features.v1.sync_tier_from_checkout_session",
        new_callable=AsyncMock,
    )

    response = client.post(
        "/credits/stripe_webhook",
        content=b"{}",
        headers={"stripe-signature": "t=1,v1=sig"},
    )

    assert response.status_code == 200
    mock_sync.assert_called_once()


def test_stripe_webhook_checkout_propagates_sync_failure(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Tier sync failure must surface as 5xx so Stripe retries the webhook."""
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
        "backend.api.features.v1.sync_tier_from_checkout_session",
        new_callable=AsyncMock,
        side_effect=stripe.StripeError("network error"),
    )

    # raise_server_exceptions=False so the test client returns 500 instead of
    # re-raising, mirroring real ASGI behavior where Stripe sees the 5xx and retries.
    nonraising_client = fastapi.testclient.TestClient(
        app, raise_server_exceptions=False
    )
    response = nonraising_client.post(
        "/credits/stripe_webhook",
        content=b"{}",
        headers={"stripe-signature": "t=1,v1=sig"},
    )

    assert response.status_code >= 500


# ---------------------------------------------------------------------------
# sync_tier_from_checkout_session unit tests
# ---------------------------------------------------------------------------


async def test_sync_tier_subscription_mode_retrieves_and_syncs(
    mocker: pytest_mock.MockFixture,
) -> None:
    fake_sub = MagicMock()
    mocker.patch(
        "backend.data.credit.stripe.Subscription.retrieve_async",
        new_callable=AsyncMock,
        return_value=fake_sub,
    )
    mock_sync = mocker.patch(
        "backend.data.credit.sync_subscription_from_stripe", new_callable=AsyncMock
    )

    await sync_tier_from_checkout_session(
        {"mode": "subscription", "subscription": "sub_123"}
    )

    mock_sync.assert_called_once_with(dict(fake_sub))


async def test_sync_tier_payment_mode_is_noop(
    mocker: pytest_mock.MockFixture,
) -> None:
    mock_retrieve = mocker.patch(
        "backend.data.credit.stripe.Subscription.retrieve_async",
        new_callable=AsyncMock,
    )

    await sync_tier_from_checkout_session({"mode": "payment"})

    mock_retrieve.assert_not_called()


async def test_sync_tier_missing_sub_id_is_noop(
    mocker: pytest_mock.MockFixture,
) -> None:
    mock_retrieve = mocker.patch(
        "backend.data.credit.stripe.Subscription.retrieve_async",
        new_callable=AsyncMock,
    )

    await sync_tier_from_checkout_session(
        {"mode": "subscription", "subscription": None}
    )

    mock_retrieve.assert_not_called()


# ---------------------------------------------------------------------------
# _expire_open_subscription_sessions tests
# ---------------------------------------------------------------------------


async def test_expire_open_subscription_sessions_empty_list(
    mocker: pytest_mock.MockFixture,
) -> None:
    mock_sessions = MagicMock()
    mock_sessions.data = []
    mock_sessions.has_more = False
    mocker.patch(
        "stripe.checkout.Session.list_async",
        new_callable=AsyncMock,
        return_value=mock_sessions,
    )
    mock_expire = mocker.patch(
        "stripe.checkout.Session.expire_async", new_callable=AsyncMock
    )

    await _expire_open_subscription_sessions("cus_test")

    stripe.checkout.Session.list_async.assert_called_once_with(
        customer="cus_test", status="open", limit=100
    )
    mock_expire.assert_not_called()


async def test_expire_open_subscription_sessions_expires_only_subscription_sessions(
    mocker: pytest_mock.MockFixture,
) -> None:
    sub_session = MagicMock(id="cs_sub_open", mode="subscription")
    payment_session = MagicMock(id="cs_pay_open", mode="payment")

    mock_sessions = MagicMock()
    mock_sessions.data = [sub_session, payment_session]
    mock_sessions.has_more = False
    mocker.patch(
        "stripe.checkout.Session.list_async",
        new_callable=AsyncMock,
        return_value=mock_sessions,
    )
    mock_expire = mocker.patch(
        "stripe.checkout.Session.expire_async", new_callable=AsyncMock
    )

    await _expire_open_subscription_sessions("cus_test")

    mock_expire.assert_called_once_with("cs_sub_open")


async def test_expire_open_subscription_sessions_paginates(
    mocker: pytest_mock.MockFixture,
) -> None:
    page1 = MagicMock()
    page1.data = [MagicMock(id="cs_page1", mode="subscription")]
    page1.has_more = True
    page2 = MagicMock()
    page2.data = [MagicMock(id="cs_page2", mode="subscription")]
    page2.has_more = False

    mock_list = mocker.patch(
        "stripe.checkout.Session.list_async",
        new_callable=AsyncMock,
        side_effect=[page1, page2],
    )
    mock_expire = mocker.patch(
        "stripe.checkout.Session.expire_async", new_callable=AsyncMock
    )

    await _expire_open_subscription_sessions("cus_test")

    assert mock_list.call_count == 2
    mock_list.assert_any_call(customer="cus_test", status="open", limit=100)
    mock_list.assert_any_call(
        customer="cus_test", status="open", limit=100, starting_after="cs_page1"
    )
    assert mock_expire.call_count == 2


# ---------------------------------------------------------------------------
# reconcile_stripe_tier_for_user tests
# ---------------------------------------------------------------------------


async def test_reconcile_stripe_tier_no_customer_returns_false(
    mocker: pytest_mock.MockFixture,
) -> None:
    user = MagicMock(stripe_customer_id=None)
    mocker.patch(
        "backend.data.credit.get_user_by_id", new_callable=AsyncMock, return_value=user
    )
    mock_get_sub = mocker.patch(
        "backend.data.credit._get_active_subscription", new_callable=AsyncMock
    )

    assert await reconcile_stripe_tier_for_user("user_abc") is False
    mock_get_sub.assert_not_called()


async def test_reconcile_stripe_tier_no_active_sub_returns_false(
    mocker: pytest_mock.MockFixture,
) -> None:
    user = MagicMock(stripe_customer_id="cus_abc")
    mocker.patch(
        "backend.data.credit.get_user_by_id", new_callable=AsyncMock, return_value=user
    )
    mocker.patch(
        "backend.data.credit._get_active_subscription",
        new_callable=AsyncMock,
        return_value=None,
    )
    mock_sync = mocker.patch(
        "backend.data.credit.sync_subscription_from_stripe", new_callable=AsyncMock
    )

    assert await reconcile_stripe_tier_for_user("user_abc") is False
    mock_sync.assert_not_called()


async def test_reconcile_stripe_tier_stripe_error_returns_false(
    mocker: pytest_mock.MockFixture,
) -> None:
    user = MagicMock(stripe_customer_id="cus_abc")
    mocker.patch(
        "backend.data.credit.get_user_by_id", new_callable=AsyncMock, return_value=user
    )
    mocker.patch(
        "backend.data.credit._get_active_subscription",
        new_callable=AsyncMock,
        side_effect=stripe.StripeError("network error"),
    )
    mock_sync = mocker.patch(
        "backend.data.credit.sync_subscription_from_stripe", new_callable=AsyncMock
    )

    assert await reconcile_stripe_tier_for_user("user_abc") is False
    mock_sync.assert_not_called()


async def test_reconcile_stripe_tier_active_sub_syncs_and_returns_true(
    mocker: pytest_mock.MockFixture,
) -> None:
    user = MagicMock(stripe_customer_id="cus_abc")
    fake_sub = MagicMock()
    mocker.patch(
        "backend.data.credit.get_user_by_id", new_callable=AsyncMock, return_value=user
    )
    mocker.patch(
        "backend.data.credit._get_active_subscription",
        new_callable=AsyncMock,
        return_value=fake_sub,
    )
    mock_sync = mocker.patch(
        "backend.data.credit.sync_subscription_from_stripe", new_callable=AsyncMock
    )

    assert await reconcile_stripe_tier_for_user("user_abc") is True
    mock_sync.assert_called_once_with(dict(fake_sub))
