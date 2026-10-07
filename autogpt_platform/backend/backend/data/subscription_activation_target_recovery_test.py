"""Recover selected targets without confusing scheduled or later paid plans."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import stripe

from backend.data import subscription_activation_checkout as checkout
from backend.data import subscription_activation_checkout_fixtures as fixtures
from backend.data import subscription_activation_stripe as billing
from backend.data.subscription_activation_models import ActivationConfirmRequest

pytest_plugins = ("backend.data.subscription_trial_fixtures",)
max_attempt = fixtures.max_attempt
attempt = fixtures.attempt
boundaries = fixtures.boundaries
live_subscription = fixtures.live_subscription


@pytest.mark.asyncio
async def test_confirmed_max_retry_at_scheduled_boundary_reports_trial_ending(
    max_attempt, boundaries, live_subscription
):
    max_attempt.confirmed_at = datetime.now(UTC)
    live_subscription.trial_end = int(datetime.now(UTC).timestamp()) + 30
    result = await checkout.confirm_activation(
        max_attempt.user_id,
        max_attempt.id,
        ActivationConfirmRequest(confirmed=True, terms_token=max_attempt.terms.token),
    )
    assert result.status == "processing" and result.error_code == "trial_ending"
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("getter", ["current", "attempt"])
async def test_paid_original_plan_does_not_complete_selected_max(
    max_attempt, boundaries, live_subscription, getter
):
    max_attempt.confirmed_at = datetime.now(UTC)
    live_subscription.status = "active"
    live_subscription.latest_invoice = billing.BillingInvoice(
        id="in_pro", customer="cus_1", status="paid"
    )
    if getter == "current":
        response = await checkout.current_activation(max_attempt.user_id)
    else:
        response = await checkout.get_activation(max_attempt.user_id, max_attempt.id)
    assert response.status == "not_applicable" and response.error_code == "plan_changed"
    assert response.return_to == max_attempt.return_to
    assert not response.usage_reset and response.activation_id is None
    boundaries.reconcile_paid_activation.assert_not_awaited()
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", [{"price_id": "price_max_new"}, {"amount_due": 9900}, {"plan": "PRO"}]
)
async def test_changed_max_quote_requires_fresh_confirmation(
    max_attempt, boundaries, change
):
    boundaries.quote_terms.return_value = max_attempt.terms.model_copy(update=change)
    with pytest.raises(billing.ActivationUnavailable, match="charge changed"):
        await checkout.confirm_activation(
            max_attempt.user_id,
            max_attempt.id,
            ActivationConfirmRequest(
                confirmed=True, terms_token=max_attempt.terms.token
            ),
        )
    boundaries.save_confirmation.assert_not_awaited()
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_changed_original_trial_terms_do_not_restore_max_confirmation(
    max_attempt, boundaries, live_subscription
):
    live_subscription.items.data[0].price.unit_amount += 100
    response = await checkout.get_activation(max_attempt.user_id, max_attempt.id)
    assert response.status == "processing" and response.error_code == "terms_changed"
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_max_does_not_reconcile_original_trial_setup_invoice(
    max_attempt, boundaries, live_subscription
):
    max_attempt.confirmed_at = datetime.now(UTC)
    live_subscription.latest_invoice = billing.BillingInvoice(
        id="in_setup", customer="cus_1", status="paid"
    )
    await checkout.get_activation(max_attempt.user_id, max_attempt.id)
    boundaries.reconcile_paid_activation.assert_not_awaited()


@pytest.mark.asyncio
async def test_basic_trial_can_preview_and_recover_selected_max(
    max_attempt, boundaries, trial, live_subscription, monkeypatch
):
    trial.offer = trial.offer.model_copy(
        update={"tier": "BASIC", "price_id": "price_basic"}
    )
    max_attempt.terms.accepted_offer_token = trial.offer.token
    live_subscription.items.data[0].price.id = "price_basic"
    monkeypatch.setattr(billing, "owned_subscription", boundaries.owned_subscription)
    source, sub = await billing.conversion_trial(max_attempt.user_id)
    assert source.offer.tier == "BASIC" and sub.id == live_subscription.id
    result = await checkout.get_activation(max_attempt.user_id, max_attempt.id)
    assert result.status == "confirmation_required" and result.error_code is None
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("catalog_change_at", ["confirmation", "retry"])
async def test_saved_max_price_survives_catalog_change(
    max_attempt, boundaries, trial, live_subscription, monkeypatch, catalog_change_at
):
    async def read_stripe(fn, *args, **kwargs):
        if fn == stripe.Price.retrieve_async:
            return {
                "id": args[0],
                "unit_amount": 9000,
                "currency": "usd",
                "recurring": {"interval": "month", "interval_count": 1},
            }
        assert fn == stripe.Invoice.create_preview_async
        return {"customer": "cus_1", "currency": "usd", "amount_due": 8700}

    prices = AsyncMock(return_value="price_max")
    reads = AsyncMock(side_effect=read_stripe)
    monkeypatch.setattr(billing.credit, "get_subscription_price_id", prices)
    monkeypatch.setattr(billing, "stripe_call", reads)
    monkeypatch.setattr(checkout, "quote_terms", billing.quote_terms)
    max_attempt.terms = await billing.quote_terms(trial, live_subscription, "MAX")
    boundaries.save_confirmation.return_value = max_attempt.model_copy(
        update={"confirmed_at": datetime.now(UTC)}
    )
    boundaries.stripe_call.side_effect = stripe.APIConnectionError("unknown outcome")
    body = ActivationConfirmRequest(confirmed=True, terms_token=max_attempt.terms.token)
    if catalog_change_at == "retry":
        await checkout.confirm_activation(max_attempt.user_id, max_attempt.id, body)
        boundaries.get_attempt.return_value = boundaries.save_confirmation.return_value
    prices.return_value = "price_max_new"

    result = await checkout.confirm_activation(
        max_attempt.user_id, max_attempt.id, body
    )

    assert result.status == "processing" and result.error_code is None
    assert boundaries.stripe_call.await_count == (
        2 if catalog_change_at == "retry" else 1
    )
    for call in boundaries.stripe_call.await_args_list:
        assert call.kwargs["items"] == [
            {"id": "si_owned", "price": "price_max", "quantity": 1}
        ]
        assert call.kwargs["idempotency_key"] == f"pro-activation:{max_attempt.id}"
    assert all(
        call.args[1] == "price_max"
        for call in reads.await_args_list
        if call.args[0] == stripe.Price.retrieve_async
    )
    prices.assert_awaited_once()
    fresh = await billing.quote_terms(trial, live_subscription, "MAX")
    assert fresh.price_id == "price_max_new"
