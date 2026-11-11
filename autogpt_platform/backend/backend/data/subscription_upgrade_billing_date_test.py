"""The Max upgrade review once an upgrade starts a fresh billing month: the
charge reviewed is the charge confirmed, minutes apart."""

from unittest.mock import AsyncMock

import pytest

from backend.data import subscription_upgrade as billing
from backend.data.subscription_upgrade_models import (
    SubscriptionUpgradePreviewRequest,
    UpgradePreviewUnavailable,
)

pytest_plugins = ("backend.data.subscription_upgrade_fixtures",)

NOW = 1791374400  # Wed 7 Oct 2026 12:00:00 UTC
MONTH_LATER = 1794052800  # Sat 7 Nov 2026 12:00:00 UTC
YEAR_LATER = 1822910400  # Thu 7 Oct 2027 12:00:00 UTC
REQUEST = SubscriptionUpgradePreviewRequest(tier="MAX")


@pytest.fixture(autouse=True)
def graph_cleanup():
    """No graph/database resources are used by these Stripe boundary tests."""
    yield


@pytest.fixture
def live(setup, monkeypatch):
    """From launch: Stripe's preview charges the full new price and dates the
    new period from the moment each preview is asked for."""
    switch = AsyncMock(return_value=True)
    monkeypatch.setattr(billing, "billing_clock_live", switch)
    asked = {"count": 0}

    def preview(**_):
        asked["count"] += 1
        started = NOW + 240 * (asked["count"] - 1)  # four minutes later each time
        return {
            **setup.invoice,
            "lines": {
                "data": [
                    {
                        "description": "1 x Max (at $320.00 / month)",
                        "amount": 32000,
                        "proration": False,
                        "price": {"id": "price_max"},
                        "period": {"start": started, "end": started + 2678400},
                    },
                    {
                        "description": "Unused time on Pro",
                        "amount": -4677,
                        "proration": True,
                        "price": {"id": "price_pro"},
                        "period": {"start": NOW, "end": NOW + 2500000},
                    },
                ]
            },
        }

    setup.preview.side_effect = preview
    return switch


async def test_the_review_asks_stripe_for_a_fresh_billing_month(setup, live):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)

    assert setup.preview.await_args.kwargs["subscription_details"] == {
        "items": [{"id": "si_owned", "price": "price_max", "quantity": 1}],
        "proration_behavior": "always_invoice",
        "proration_date": NOW,
        "billing_cycle_anchor": "now",
    }
    # The plan renews a month from the review, not at the old period's end.
    assert preview.period_end == MONTH_LATER
    assert [(line.amount, line.proration) for line in preview.lines] == [
        (32000, False),
        (-4677, True),
    ]
    assert (preview.lines[0].period_start, preview.lines[0].period_end) == (
        NOW,
        MONTH_LATER,
    )


async def test_confirming_minutes_later_matches_the_review(setup, live):
    """Stripe dates the new period from each call, so the re-check sees a
    later period than the review did; the charge is the same."""
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)

    quote = await billing.validate_subscription_upgrade_preview(
        "user-1", REQUEST, preview.preview_id, preview.proration_date
    )

    assert quote.preview == preview
    assert quote.preview.resets_billing_date is True
    assert setup.preview.await_count == 2


async def test_a_changed_charge_still_needs_a_new_review(setup, live):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    setup.invoice["amount_due"] += 100

    with pytest.raises(UpgradePreviewUnavailable, match="billing details changed"):
        await billing.validate_subscription_upgrade_preview(
            "user-1", REQUEST, preview.preview_id, preview.proration_date
        )


async def test_a_review_from_before_launch_cannot_be_confirmed_after_it(setup, live):
    """It showed a prorated charge in the old period; the charge would now be
    the full new price."""
    live.return_value = False
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    live.return_value = True

    with pytest.raises(UpgradePreviewUnavailable, match="billing details changed"):
        await billing.validate_subscription_upgrade_preview(
            "user-1", REQUEST, preview.preview_id, preview.proration_date
        )


async def test_a_yearly_upgrade_renews_a_year_from_the_review(setup, live):
    setup.price["recurring"] = {"interval": "year"}
    yearly = SubscriptionUpgradePreviewRequest(tier="MAX", billing_cycle="yearly")

    preview = await billing.preview_subscription_upgrade("user-1", yearly)

    assert preview.period_end == YEAR_LATER


async def test_before_launch_the_review_keeps_the_billing_period(setup, live):
    live.return_value = False

    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)

    details = setup.preview.await_args.kwargs["subscription_details"]
    assert "billing_cycle_anchor" not in details
    assert preview.period_end == NOW + 2500000
