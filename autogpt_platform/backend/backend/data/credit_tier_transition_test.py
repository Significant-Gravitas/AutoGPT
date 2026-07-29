"""Unit tests for automation pause/resume on subscription tier transitions."""

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import SubscriptionTier

from backend.data.credit import _handle_tier_transition_automations

# Patched on automation_pause (not credit): the transition hook imports these
# locally from that module to avoid a circular import at load time.
_MODULE = "backend.data.automation_pause"


@contextmanager
def _mock_automation_ops(has_lapsed: bool = False):
    with (
        patch(f"{_MODULE}.pause_automations_for_payment_lapse", new=AsyncMock()) as p,
        patch(
            f"{_MODULE}.resume_automations_after_payment_restored", new=AsyncMock()
        ) as r,
        patch(
            f"{_MODULE}.has_payment_lapsed_automations",
            new=AsyncMock(return_value=has_lapsed),
        ) as h,
    ):
        yield p, r, h


@pytest.mark.asyncio
async def test_paid_to_no_tier_pauses_automations():
    with _mock_automation_ops() as (p, r, _):
        await _handle_tier_transition_automations(
            "user-1", SubscriptionTier.PRO, SubscriptionTier.NO_TIER
        )
    p.assert_awaited_once_with("user-1")
    r.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_tier_to_paid_resumes_automations():
    with _mock_automation_ops() as (p, r, _):
        await _handle_tier_transition_automations(
            "user-1", SubscriptionTier.NO_TIER, SubscriptionTier.BASIC
        )
    r.assert_awaited_once_with("user-1")
    p.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_tier_paid_self_heals_when_automations_still_lapsed():
    """A same-tier paid webhook retry re-attempts resume while payment-lapsed
    automations remain, repairing a resume that partially failed earlier."""
    with _mock_automation_ops(has_lapsed=True) as (p, r, _):
        await _handle_tier_transition_automations(
            "user-1", SubscriptionTier.PRO, SubscriptionTier.PRO
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
    # No payment-lapsed automations remain, so a same-tier paid retry is a no-op.
    with _mock_automation_ops(has_lapsed=False) as (p, r, _):
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
        patch(
            f"{_MODULE}.has_payment_lapsed_automations",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.data.credit.discord_send_alert", new=AsyncMock()
        ) as discord_alert,
    ):
        await _handle_tier_transition_automations(
            "user-1", SubscriptionTier.PRO, SubscriptionTier.NO_TIER
        )
    # The except block's purpose is to page ops loudly, not just swallow.
    discord_alert.assert_awaited_once()
