"""Explicit confirmation and charge recovery contract."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import stripe
from pydantic import ValidationError

from backend.api.features.subscription_activation_routes import confirm_pro_activation
from backend.data import subscription_activation_checkout as checkout
from backend.data import subscription_activation_stripe as billing
from backend.data.subscription_activation_models import (
    ActivationConfirmRequest,
    ActivationNotFound,
    ActivationPreviewRequest,
    ActivationUnavailable,
    PaidActivationResult,
)

pytest_plugins = (
    "backend.data.subscription_trial_fixtures",
    "backend.data.subscription_activation_checkout_fixtures",
)


@pytest.mark.parametrize("confirmation", [False, 1, "true", None])
def test_conversion_requires_explicit_true_confirmation(confirmation):
    with pytest.raises(ValidationError):
        ActivationConfirmRequest(confirmed=confirmation, terms_token="a" * 64)


@pytest.mark.parametrize(
    "path", ["https://attacker.test", "//attacker.test", "/\\evil", "/x\n"]
)
def test_return_destination_stays_on_application_origin(path):
    with pytest.raises(ValidationError):
        ActivationPreviewRequest(return_to=path)


@pytest.mark.asyncio
async def test_preview_never_charges_and_preserves_destination(attempt, boundaries):
    response = await checkout.preview_activation(attempt.user_id, attempt.return_to)
    assert response.status == "confirmation_required"
    assert response.return_to == attempt.return_to
    assert response.terms.amount_due == 4200
    boundaries.stripe_call.assert_not_awaited()
    boundaries.save_confirmation.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_persists_intent_before_idempotent_existing_subscription_update(
    attempt, boundaries
):
    async def stripe_update(fn, subscription_id, **kwargs):
        boundaries.save_confirmation.assert_awaited_once()
        assert subscription_id == "sub_1"
        assert fn == stripe.Subscription.modify_async
        assert kwargs == {
            "trial_end": "now",
            "proration_behavior": "none",
            "payment_behavior": "allow_incomplete",
            "metadata": {"pro_activation_attempt_id": attempt.id},
            "idempotency_key": f"pro-activation:{attempt.id}",
        }

    boundaries.stripe_call.side_effect = stripe_update
    response = await checkout.confirm_activation(
        attempt.user_id,
        attempt.id,
        ActivationConfirmRequest(confirmed=True, terms_token=attempt.terms.token),
    )
    assert response.status == "processing"
    boundaries.stripe_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_changed_or_expired_terms_never_start_payment(attempt, boundaries):
    boundaries.quote_terms.return_value = attempt.terms.model_copy(
        update={"amount_due": 4201}
    )
    with pytest.raises(ActivationUnavailable, match="charge changed"):
        await checkout.confirm_activation(
            attempt.user_id,
            attempt.id,
            ActivationConfirmRequest(confirmed=True, terms_token=attempt.terms.token),
        )
    boundaries.save_confirmation.assert_not_awaited()
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_ownership_failure_does_not_reveal_attempt(attempt, boundaries):
    boundaries.get_attempt.return_value = None
    with pytest.raises(ActivationNotFound):
        await checkout.get_activation("other-user", attempt.id)
    boundaries.get_attempt.assert_awaited_once_with("other-user", attempt.id)
    boundaries.owned_subscription.assert_not_awaited()


@pytest.mark.asyncio
async def test_network_uncertainty_is_processing_and_reuses_key(attempt, boundaries):
    confirmed = attempt.model_copy(update={"confirmed_at": datetime.now(UTC)})
    boundaries.get_attempt.return_value = confirmed
    boundaries.stripe_call.side_effect = stripe.APIConnectionError("timeout")
    body = ActivationConfirmRequest(confirmed=True, terms_token=attempt.terms.token)
    for _ in range(2):
        assert (
            await checkout.confirm_activation(attempt.user_id, attempt.id, body)
        ).status == "processing"
    assert {
        call.kwargs["idempotency_key"]
        for call in boundaries.stripe_call.await_args_list
    } == {f"pro-activation:{attempt.id}"}
    boundaries.save_confirmation.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirmed_get_never_resubmits_charge(attempt, boundaries):
    boundaries.get_attempt.return_value = attempt.model_copy(
        update={"confirmed_at": datetime.now(UTC)}
    )
    assert (
        await checkout.get_activation(attempt.user_id, attempt.id)
    ).status == "processing"
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_near_scheduled_boundary_never_mutates_trial(
    attempt, boundaries, live_subscription
):
    live_subscription.trial_end = int(datetime.now(UTC).timestamp()) + 5
    result = await checkout.confirm_activation(
        attempt.user_id,
        attempt.id,
        ActivationConfirmRequest(confirmed=True, terms_token=attempt.terms.token),
    )
    assert result.status == "processing"
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("ready", [False, True])
async def test_paid_invoice_requires_authoritative_reset_completion(
    attempt, boundaries, live_subscription, ready
):
    live_subscription.status = "active"
    live_subscription.latest_invoice = billing.BillingInvoice(
        id="in_paid", customer="cus_1", status="paid", amount_remaining=0
    )
    boundaries.reconcile_paid_activation.return_value = (
        PaidActivationResult(
            invoice_id="in_paid", usage_reset=True, activation_id="generation-1"
        )
        if ready
        else None
    )
    response = await checkout.get_activation(attempt.user_id, attempt.id)
    assert response.status == ("ready" if ready else "processing")
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
async def test_paid_database_failure_remains_processing(
    attempt, boundaries, live_subscription
):
    live_subscription.status = "active"
    live_subscription.latest_invoice = billing.BillingInvoice(
        id="in_paid", customer="cus_1", status="paid"
    )
    boundaries.reconcile_paid_activation.side_effect = ConnectionError(
        "database unavailable"
    )
    assert (
        await checkout.get_activation(attempt.user_id, attempt.id)
    ).status == "processing"
    boundaries.stripe_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payment_status, expected",
    [
        ("requires_action", "action_required"),
        ("requires_payment_method", "payment_required"),
        ("processing", "processing"),
    ],
)
async def test_payment_and_authentication_required_never_activate(
    attempt, boundaries, live_subscription, payment_status, expected
):
    live_subscription.status = "past_due"
    live_subscription.latest_invoice = billing.BillingInvoice(
        id="in_open",
        customer="cus_1",
        status="open",
        payment_intent="pi_1",
        hosted_invoice_url="https://invoice.stripe.com/pay/in_open",
    )
    boundaries.stripe_call.return_value = SimpleNamespace(
        customer="cus_1", status=payment_status
    )
    response = await checkout.get_activation(attempt.user_id, attempt.id)
    assert response.status == expected
    assert response.hosted_invoice_url == "https://invoice.stripe.com/pay/in_open"
    boundaries.reconcile_paid_activation.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cost", [250000, 1000000])
async def test_partially_used_and_exhausted_trials_keep_accepted_offer(
    trial, live_subscription, monkeypatch, cost
):
    trial.cost_microdollars = cost
    call = AsyncMock(
        return_value={"customer": "cus_1", "currency": "usd", "amount_due": 4200}
    )
    monkeypatch.setattr(billing, "stripe_call", call)
    terms = await billing.quote_terms(trial, live_subscription)
    assert terms.accepted_offer_token == trial.offer.token
    assert terms.renewal_unit_amount == trial.offer.unit_amount
    assert terms.amount_due == 4200
    assert trial.cost_microdollars == cost


@pytest.mark.asyncio
async def test_lock_exit_failure_after_charge_is_recoverable_processing(
    attempt, boundaries, live_subscription, monkeypatch
):
    @asynccontextmanager
    async def failed_lock_exit(user_id):
        yield
        raise ConnectionError("The checkout lock transaction expired after payment")

    async def payment_succeeded(*args, **kwargs):
        live_subscription.status = "active"
        live_subscription.latest_invoice = billing.BillingInvoice(
            id="in_paid", customer="cus_1", status="paid"
        )

    monkeypatch.setattr(checkout, "subscription_checkout_lock", failed_lock_exit)
    boundaries.stripe_call.side_effect = payment_succeeded
    boundaries.reconcile_paid_activation.return_value = PaidActivationResult(
        invoice_id="in_paid", usage_reset=True, activation_id="generation-1"
    )
    response = await confirm_pro_activation(
        attempt.id,
        ActivationConfirmRequest(confirmed=True, terms_token=attempt.terms.token),
        attempt.user_id,
    )
    assert response.status == "processing"
    assert response.id == attempt.id
    boundaries.stripe_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_current_recovers_new_paid_signup_after_canceled_conversion(
    attempt, boundaries, live_subscription, monkeypatch
):
    canceled = live_subscription.model_copy(update={"status": "canceled"})
    paid = live_subscription.model_copy(
        update={
            "id": "sub_new",
            "status": "active",
            "latest_invoice": billing.BillingInvoice(
                id="in_new", customer="cus_1", status="paid"
            ),
        }
    )
    boundaries.owned_subscription.side_effect = [canceled, paid]
    boundaries.stripe_call.return_value = SimpleNamespace(data=[paid], has_more=False)
    boundaries.reconcile_paid_activation.return_value = PaidActivationResult(
        invoice_id="in_new", usage_reset=True, activation_id="generation-1"
    )
    monkeypatch.setattr(
        checkout.User,
        "prisma",
        lambda: MagicMock(
            find_unique_or_raise=AsyncMock(
                return_value=SimpleNamespace(stripeCustomerId="cus_1")
            )
        ),
    )
    monkeypatch.setattr(
        checkout, "activation_return_to", AsyncMock(return_value="/chat/new")
    )
    response = await checkout.current_activation(attempt.user_id)
    assert response.status == "ready"
    assert response.invoice_id == "in_new" and response.return_to == "/chat/new"
    assert boundaries.stripe_call.await_count == 1
    assert boundaries.stripe_call.await_args.args[0].__name__ == "list_async"
