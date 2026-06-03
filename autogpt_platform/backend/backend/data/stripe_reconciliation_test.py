"""Tests for the periodic Stripe → DB tier reconciliation sweep."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock
from prisma.enums import SubscriptionTier, SubscriptionTierSource

from backend.data.stripe_reconciliation import (
    ReconciliationSummary,
    _record_subscription,
    reconcile_all_stripe_tiers,
)


def _sub(customer: str, price_id: str) -> dict:
    return {
        "id": f"sub_{customer}",
        "customer": customer,
        "items": {"data": [{"price": {"id": price_id}}]},
    }


def _candidate(
    user_id: str, customer_id: str | None, tier: SubscriptionTier
) -> MagicMock:
    user = MagicMock()
    user.id = user_id
    user.stripeCustomerId = customer_id
    user.subscriptionTier = tier
    return user


def _patch_stripe_pages(
    mocker: pytest_mock.MockFixture, by_status: dict[str, list[dict]]
) -> None:
    """Make stripe.Subscription.list return one page per status."""

    def _list(*, status: str, limit: int, starting_after: str | None = None):
        page = MagicMock()
        page.data = by_status.get(status, [])
        page.has_more = False
        return page

    mocker.patch(
        "backend.data.stripe_reconciliation.stripe.Subscription.list",
        side_effect=_list,
    )


@pytest.fixture(autouse=True)
def mock_alert(mocker: pytest_mock.MockFixture) -> AsyncMock:
    """Patch the Discord ops alert for every test so none hits the network."""
    return mocker.patch(
        "backend.data.stripe_reconciliation.alert_tier_reconciliation_discrepancy",
        new_callable=AsyncMock,
    )


@pytest.mark.asyncio
async def test_sweep_upgrades_downgrades_and_skips_unchanged(
    mocker: pytest_mock.MockFixture,
    mock_alert: AsyncMock,
) -> None:
    mocker.patch(
        "backend.data.stripe_reconciliation.build_price_to_tier_map",
        new_callable=AsyncMock,
        return_value={"price_pro": SubscriptionTier.PRO},
    )
    _patch_stripe_pages(
        mocker,
        {"active": [_sub("cus_keep", "price_pro"), _sub("cus_up", "price_pro")]},
    )
    candidates = [
        # already PRO with active PRO sub -> unchanged
        _candidate("u_keep", "cus_keep", SubscriptionTier.PRO),
        # NO_TIER with active PRO sub -> upgrade
        _candidate("u_up", "cus_up", SubscriptionTier.NO_TIER),
        # PRO with no active sub -> downgrade
        _candidate("u_down", "cus_gone", SubscriptionTier.PRO),
    ]
    mocker.patch(
        "backend.data.stripe_reconciliation.User.prisma",
        return_value=MagicMock(find_many=AsyncMock(return_value=candidates)),
    )
    set_tier = mocker.patch(
        "backend.data.stripe_reconciliation.set_subscription_tier",
        new_callable=AsyncMock,
    )

    summary = await reconcile_all_stripe_tiers()

    assert summary.upgrades == 1
    assert summary.downgrades == 1
    assert summary.unchanged == 1
    assert summary.errors == 0
    set_tier.assert_any_await(
        "u_up", SubscriptionTier.PRO, SubscriptionTierSource.STRIPE
    )
    set_tier.assert_any_await(
        "u_down", SubscriptionTier.NO_TIER, SubscriptionTierSource.STRIPE
    )
    # Each correction is recorded, and the sweep alerts ops exactly once (not
    # per-user) with the discrepancy counts.
    assert {d.direction for d in summary.discrepancies} == {"upgrade", "downgrade"}
    assert len(summary.discrepancies) == 2
    mock_alert.assert_awaited_once()
    alert_msg = mock_alert.await_args.args[0]
    assert "2 discrepancy" in alert_msg
    assert "1 downgrade" in alert_msg and "1 upgrade" in alert_msg


@pytest.mark.asyncio
async def test_sweep_no_discrepancies_does_not_alert(
    mocker: pytest_mock.MockFixture,
    mock_alert: AsyncMock,
) -> None:
    """Steady state (every candidate already correct) must stay silent."""
    mocker.patch(
        "backend.data.stripe_reconciliation.build_price_to_tier_map",
        new_callable=AsyncMock,
        return_value={"price_pro": SubscriptionTier.PRO},
    )
    _patch_stripe_pages(mocker, {"active": [_sub("cus_keep", "price_pro")]})
    mocker.patch(
        "backend.data.stripe_reconciliation.User.prisma",
        return_value=MagicMock(
            find_many=AsyncMock(
                return_value=[_candidate("u_keep", "cus_keep", SubscriptionTier.PRO)]
            )
        ),
    )
    mocker.patch(
        "backend.data.stripe_reconciliation.set_subscription_tier",
        new_callable=AsyncMock,
    )

    summary = await reconcile_all_stripe_tiers()

    assert summary.unchanged == 1
    assert summary.discrepancies == []
    mock_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_only_queries_stripe_sourced_users(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.data.stripe_reconciliation.build_price_to_tier_map",
        new_callable=AsyncMock,
        return_value={},
    )
    _patch_stripe_pages(mocker, {})
    find_many = AsyncMock(return_value=[])
    mocker.patch(
        "backend.data.stripe_reconciliation.User.prisma",
        return_value=MagicMock(find_many=find_many),
    )
    mocker.patch(
        "backend.data.stripe_reconciliation.set_subscription_tier",
        new_callable=AsyncMock,
    )

    await reconcile_all_stripe_tiers()

    find_many.assert_awaited_once_with(
        where={"subscriptionTierSource": SubscriptionTierSource.STRIPE},
    )


@pytest.mark.asyncio
async def test_sweep_counts_errors_without_aborting(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.data.stripe_reconciliation.build_price_to_tier_map",
        new_callable=AsyncMock,
        return_value={"price_pro": SubscriptionTier.PRO},
    )
    _patch_stripe_pages(mocker, {"active": [_sub("cus_ok", "price_pro")]})
    candidates = [
        _candidate("u_err", "cus_gone", SubscriptionTier.PRO),
        _candidate("u_ok", "cus_ok", SubscriptionTier.NO_TIER),
    ]
    mocker.patch(
        "backend.data.stripe_reconciliation.User.prisma",
        return_value=MagicMock(find_many=AsyncMock(return_value=candidates)),
    )
    mocker.patch(
        "backend.data.stripe_reconciliation.set_subscription_tier",
        new_callable=AsyncMock,
        side_effect=[RuntimeError("db down"), None],
    )

    summary = await reconcile_all_stripe_tiers()

    assert summary.errors == 1
    assert summary.upgrades == 1


def test_record_subscription_keeps_highest_tier() -> None:
    tiers: dict[str, SubscriptionTier] = {}
    price_to_tier = {
        "price_basic": SubscriptionTier.BASIC,
        "price_pro": SubscriptionTier.PRO,
    }
    _record_subscription(_sub("cus_x", "price_basic"), price_to_tier, tiers)
    _record_subscription(_sub("cus_x", "price_pro"), price_to_tier, tiers)
    assert tiers["cus_x"] == SubscriptionTier.PRO
    # A subsequent lower-tier sub must not clobber the higher tier already seen.
    _record_subscription(_sub("cus_x", "price_basic"), price_to_tier, tiers)
    assert tiers["cus_x"] == SubscriptionTier.PRO


def test_record_subscription_ignores_unknown_price() -> None:
    tiers: dict[str, SubscriptionTier] = {}
    _record_subscription(_sub("cus_y", "price_unknown"), {}, tiers)
    assert "cus_y" not in tiers


def test_summary_defaults_to_zero() -> None:
    assert ReconciliationSummary().model_dump() == {
        "stripe_active_subscriptions": 0,
        "candidate_users": 0,
        "upgrades": 0,
        "downgrades": 0,
        "unchanged": 0,
        "errors": 0,
        "pagination_capped": False,
        "discrepancies": [],
    }
