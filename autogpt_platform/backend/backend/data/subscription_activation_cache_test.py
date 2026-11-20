"""Trial activation mutations expire shared Stripe billing lookups."""

from unittest.mock import AsyncMock

import pytest
import stripe

from backend.data import credit
from backend.data import subscription_activation_checkout as checkout
from backend.data import subscription_activation_checkout_fixtures as fixtures
from backend.data import subscription_trial_fixtures as trial_fixtures
from backend.data.subscription_activation_models import ActivationConfirmRequest

trial = trial_fixtures.trial
attempt = fixtures.attempt
live_subscription = fixtures.live_subscription
boundaries = fixtures.boundaries


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_response", [False, True])
async def test_confirm_invalidates_shared_lookups_even_when_stripe_response_is_lost(
    attempt, boundaries, monkeypatch, lost_response
):
    invalidated = AsyncMock()
    monkeypatch.setattr(credit, "invalidate_billing_lookups", invalidated)
    if lost_response:
        boundaries.stripe_call.side_effect = stripe.APIConnectionError("lost response")
    response = await checkout.confirm_activation(
        attempt.user_id,
        attempt.id,
        ActivationConfirmRequest(confirmed=True, terms_token=attempt.terms.token),
    )
    assert response.status == "processing"
    boundaries.stripe_call.assert_awaited_once()
    invalidated.assert_awaited_once_with(attempt.customer_id)


@pytest.mark.asyncio
async def test_status_polling_does_not_mutate_or_invalidate_billing_lookups(
    attempt, boundaries, monkeypatch
):
    invalidated = AsyncMock()
    monkeypatch.setattr(credit, "invalidate_billing_lookups", invalidated)
    await checkout.get_activation(attempt.user_id, attempt.id)
    boundaries.stripe_call.assert_not_awaited()
    invalidated.assert_not_awaited()
