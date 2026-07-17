"""Tests for the periodic Stripe → DB tier reconciliation sweep."""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock
import stripe
from prisma.enums import SubscriptionTier

from backend.data.stripe_reconciliation import (
    ReconciliationSummary,
    _collect_status_page,
    _record_subscription,
    reconcile_all_stripe_tiers,
)


class _SubDict(dict):
    """A subscription mapping that also exposes ``.id`` as an attribute, mirroring
    Stripe's StripeObject (dict-like .get() AND attribute access on the id used
    by the pagination cursor ``subs.data[-1].id``)."""

    @property
    def id(self) -> str:
        return self["id"]


def _sub(customer: str, price_id: str) -> dict:
    return _SubDict(
        {
            "id": f"sub_{customer}",
            "customer": customer,
            "items": {"data": [{"price": {"id": price_id}}]},
        }
    )


def _candidate(
    user_id: str,
    customer_id: str | None,
    tier: SubscriptionTier,
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
    set_tier.assert_any_await("u_up", SubscriptionTier.PRO)
    set_tier.assert_any_await("u_down", SubscriptionTier.NO_TIER)
    # Each correction is recorded, and the sweep alerts ops exactly once (not
    # per-user) with the discrepancy counts.
    assert {d.direction for d in summary.discrepancies} == {"upgrade", "downgrade"}
    assert len(summary.discrepancies) == 2
    mock_alert.assert_awaited_once()
    alert_msg = mock_alert.await_args.args[0]
    # Count header up front.
    assert "reconciled 2 account" in alert_msg
    assert "1 upgraded" in alert_msg and "1 downgraded" in alert_msg
    # Full affected-user list at the end, each with its tier change.
    assert "Affected users:" in alert_msg
    assert "u_up" in alert_msg and "u_down" in alert_msg
    assert "NO_TIER → PRO (upgrade)" in alert_msg
    assert "PRO → NO_TIER (downgrade)" in alert_msg


@pytest.mark.asyncio
async def test_sweep_counts_paid_to_paid_downgrade_as_downgrade(
    mocker: pytest_mock.MockFixture,
    mock_alert: AsyncMock,
) -> None:
    """MAX -> PRO is a downgrade even though the target isn't NO_TIER."""
    mocker.patch(
        "backend.data.stripe_reconciliation.build_price_to_tier_map",
        new_callable=AsyncMock,
        return_value={"price_pro": SubscriptionTier.PRO},
    )
    _patch_stripe_pages(mocker, {"active": [_sub("cus_x", "price_pro")]})
    mocker.patch(
        "backend.data.stripe_reconciliation.User.prisma",
        return_value=MagicMock(
            find_many=AsyncMock(
                return_value=[_candidate("u_x", "cus_x", SubscriptionTier.MAX)]
            )
        ),
    )
    mocker.patch(
        "backend.data.stripe_reconciliation.set_subscription_tier",
        new_callable=AsyncMock,
    )

    summary = await reconcile_all_stripe_tiers()

    assert summary.downgrades == 1
    assert summary.upgrades == 0
    assert summary.discrepancies[0].direction == "downgrade"


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
async def test_sweep_queries_reconcilable_users(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Candidates mirror `_is_stripe_reconcilable`: every user with a Stripe
    customer that is not on ENTERPRISE."""
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
        where={
            "stripeCustomerId": {"not": None},
            "subscriptionTier": {"not": SubscriptionTier.ENTERPRISE},
        },
    )


@pytest.mark.asyncio
async def test_sweep_incomplete_map_skips_downgrades(
    mocker: pytest_mock.MockFixture,
    mock_alert: AsyncMock,
) -> None:
    """A failed/partial Stripe snapshot must never downgrade an absent user."""
    mocker.patch(
        "backend.data.stripe_reconciliation.build_price_to_tier_map",
        new_callable=AsyncMock,
        return_value={"price_pro": SubscriptionTier.PRO},
    )
    # Stripe list fails -> _collect_status_page returns capped -> map incomplete.
    mocker.patch(
        "backend.data.stripe_reconciliation.stripe.Subscription.list",
        side_effect=stripe.StripeError("rate limited"),
    )
    candidates = [_candidate("u_down", "cus_gone", SubscriptionTier.PRO)]
    mocker.patch(
        "backend.data.stripe_reconciliation.User.prisma",
        return_value=MagicMock(find_many=AsyncMock(return_value=candidates)),
    )
    set_tier = mocker.patch(
        "backend.data.stripe_reconciliation.set_subscription_tier",
        new_callable=AsyncMock,
    )

    summary = await reconcile_all_stripe_tiers()

    assert summary.pagination_capped is True
    assert summary.skipped_incomplete == 1
    assert summary.downgrades == 0
    set_tier.assert_not_called()
    mock_alert.assert_not_awaited()


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


def test_record_subscription_skips_missing_customer() -> None:
    tiers: dict[str, SubscriptionTier] = {}
    price_to_tier = {"price_pro": SubscriptionTier.PRO}
    # customer None and empty string are both ignored without crashing.
    _record_subscription(
        {
            "id": "s1",
            "customer": None,
            "items": {"data": [{"price": {"id": "price_pro"}}]},
        },
        price_to_tier,
        tiers,
    )
    _record_subscription(
        {
            "id": "s2",
            "customer": "",
            "items": {"data": [{"price": {"id": "price_pro"}}]},
        },
        price_to_tier,
        tiers,
    )
    assert tiers == {}


def test_record_subscription_skips_missing_items() -> None:
    tiers: dict[str, SubscriptionTier] = {}
    price_to_tier = {"price_pro": SubscriptionTier.PRO}
    # Missing 'items' entirely, and an empty items list, are both skipped.
    _record_subscription({"id": "s1", "customer": "cus_a"}, price_to_tier, tiers)
    _record_subscription(
        {"id": "s2", "customer": "cus_b", "items": {"data": []}}, price_to_tier, tiers
    )
    assert "cus_a" not in tiers
    assert "cus_b" not in tiers


def test_record_subscription_skips_missing_price_id() -> None:
    tiers: dict[str, SubscriptionTier] = {}
    price_to_tier = {"price_pro": SubscriptionTier.PRO}
    # Price object with no/None id resolves to no tier -> customer not added.
    _record_subscription(
        {"id": "s1", "customer": "cus_a", "items": {"data": [{"price": {}}]}},
        price_to_tier,
        tiers,
    )
    _record_subscription(
        {"id": "s2", "customer": "cus_b", "items": {"data": [{"price": {"id": ""}}]}},
        price_to_tier,
        tiers,
    )
    assert tiers == {}


def _page(data: list[dict], has_more: bool) -> MagicMock:
    page = MagicMock()
    page.data = data
    page.has_more = has_more
    return page


@pytest.mark.asyncio
async def test_collect_status_page_follows_pagination(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Multiple pages are followed via starting_after = last item id until
    has_more=False, accumulating every page's subscriptions into the map."""
    pages = [
        _page([_sub("cus_a", "price_basic")], has_more=True),
        _page([_sub("cus_b", "price_pro")], has_more=False),
    ]
    seen_starting_after: list[str | None] = []

    def _list(*, status: str, limit: int, starting_after: str | None = None):
        seen_starting_after.append(starting_after)
        return pages.pop(0)

    mocker.patch(
        "backend.data.stripe_reconciliation.stripe.Subscription.list",
        side_effect=_list,
    )
    price_to_tier = {
        "price_basic": SubscriptionTier.BASIC,
        "price_pro": SubscriptionTier.PRO,
    }
    tiers: dict[str, SubscriptionTier] = {}

    capped = await _collect_status_page("active", price_to_tier, tiers)

    assert capped is False
    assert tiers == {"cus_a": SubscriptionTier.BASIC, "cus_b": SubscriptionTier.PRO}
    # First call has no cursor; second passes the first page's last item id.
    assert seen_starting_after == [None, "sub_cus_a"]


@pytest.mark.asyncio
async def test_collect_status_page_highest_tier_wins_across_pages(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A customer appearing with different tiers on different pages resolves to
    the highest tier regardless of which page it appeared on first."""
    price_to_tier = {
        "price_basic": SubscriptionTier.BASIC,
        "price_pro": SubscriptionTier.PRO,
    }

    def _run(first_price: str, second_price: str) -> dict[str, SubscriptionTier]:
        pages = [
            _page([_sub("cus_c", first_price)], has_more=True),
            _page([_sub("cus_c2", second_price)], has_more=False),
        ]
        # Reuse the same customer id across both pages.
        pages[0].data[0]["customer"] = "cus_c"
        pages[1].data[0]["customer"] = "cus_c"
        return pages

    for first, second in (("price_basic", "price_pro"), ("price_pro", "price_basic")):
        pages = _run(first, second)
        mocker.patch(
            "backend.data.stripe_reconciliation.stripe.Subscription.list",
            side_effect=lambda *, status, limit, starting_after=None: pages.pop(0),
        )
        tiers: dict[str, SubscriptionTier] = {}
        await _collect_status_page("active", price_to_tier, tiers)
        assert tiers == {"cus_c": SubscriptionTier.PRO}


@pytest.mark.asyncio
async def test_collect_status_page_stops_at_cap(
    mocker: pytest_mock.MockFixture,
) -> None:
    """When Stripe never stops returning has_more=True, the loop bails at the
    page cap and reports capped=True rather than looping forever."""
    mocker.patch("backend.data.stripe_reconciliation._MAX_SUBSCRIPTION_PAGES", 3)
    call_count = {"n": 0}

    def _list(*, status: str, limit: int, starting_after: str | None = None):
        call_count["n"] += 1
        return _page([_sub(f"cus_{call_count['n']}", "price_pro")], has_more=True)

    mocker.patch(
        "backend.data.stripe_reconciliation.stripe.Subscription.list",
        side_effect=_list,
    )
    tiers: dict[str, SubscriptionTier] = {}

    capped = await _collect_status_page(
        "active", {"price_pro": SubscriptionTier.PRO}, tiers
    )

    assert capped is True
    # Stopped exactly at the (patched) cap, not beyond.
    assert call_count["n"] == 3


def test_summary_defaults_to_zero() -> None:
    assert ReconciliationSummary().model_dump() == {
        "stripe_active_subscriptions": 0,
        "candidate_users": 0,
        "upgrades": 0,
        "downgrades": 0,
        "unchanged": 0,
        "errors": 0,
        "skipped_incomplete": 0,
        "pagination_capped": False,
        "discrepancies": [],
    }
