"""Unit tests for automation pause/resume on subscription tier transitions."""

from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.data.credit import _handle_tier_transition_automations

_MODULE = "backend.data.credit"


@pytest.mark.asyncio
async def test_paid_to_no_tier_pauses_automations():
    with (
        patch(f"{_MODULE}.pause_automations_for_payment_lapse", new=AsyncMock()) as p,
        patch(
            f"{_MODULE}.resume_automations_after_payment_restored", new=AsyncMock()
        ) as r,
    ):
        await _handle_tier_transition_automations(
            "user-1", SubscriptionTier.PRO, SubscriptionTier.NO_TIER
        )
    p.assert_awaited_once_with("user-1")
    r.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_tier_to_paid_resumes_automations():
    with (
        patch(f"{_MODULE}.pause_automations_for_payment_lapse", new=AsyncMock()) as p,
        patch(
            f"{_MODULE}.resume_automations_after_payment_restored", new=AsyncMock()
        ) as r,
    ):
        await _handle_tier_transition_automations(
            "user-1", SubscriptionTier.NO_TIER, SubscriptionTier.BASIC
        )
    r.assert_awaited_once_with("user-1")
    p.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "previous,new",
    [
        (SubscriptionTier.PRO, SubscriptionTier.BASIC),
        (SubscriptionTier.PRO, SubscriptionTier.PRO),
        (None, SubscriptionTier.NO_TIER),
        (SubscriptionTier.NO_TIER, SubscriptionTier.NO_TIER),
    ],
)
async def test_other_transitions_do_nothing(previous, new):
    with (
        patch(f"{_MODULE}.pause_automations_for_payment_lapse", new=AsyncMock()) as p,
        patch(
            f"{_MODULE}.resume_automations_after_payment_restored", new=AsyncMock()
        ) as r,
    ):
        await _handle_tier_transition_automations("user-1", previous, new)
    p.assert_not_awaited()
    r.assert_not_awaited()


@pytest.mark.asyncio
async def test_pause_failure_is_swallowed():
    """A scheduler outage must not propagate into the Stripe webhook path."""
    with (
        patch(
            f"{_MODULE}.pause_automations_for_payment_lapse",
            new=AsyncMock(side_effect=RuntimeError("scheduler down")),
        ),
        patch(f"{_MODULE}.resume_automations_after_payment_restored", new=AsyncMock()),
    ):
        await _handle_tier_transition_automations(
            "user-1", SubscriptionTier.PRO, SubscriptionTier.NO_TIER
        )
