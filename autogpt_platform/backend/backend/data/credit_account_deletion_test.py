"""Deleting an account ends its Stripe subscriptions now, and marks the cancel
so the webhooks it sets off keep the tier sync but skip analytics."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier
from pytest_mock import MockerFixture

from backend.data import credit

CHECKOUT = "backend.data.subscription_checkout"
CUSTOMER = "cus_123"
COMMENT = {"comment": credit.ACCOUNT_DELETION_CANCELLATION_COMMENT}
BILLABLE = ["active", "trialing"]
# Still open in Stripe, so still able to bill or dun the customer.
UNSETTLED = ["past_due", "unpaid", "paused", "incomplete"]
# Terminal: Stripe refuses to cancel these.
ENDED = ["incomplete_expired", "canceled"]


@pytest.fixture(autouse=True)
def checkouts(mocker: MockerFixture) -> MagicMock:
    """The customer's open Checkout sessions: none unless a test opens some."""
    parent = MagicMock()
    parent.attach_mock(
        mocker.patch(
            f"{CHECKOUT}.stripe.checkout.Session.list_async",
            new_callable=AsyncMock,
            return_value=_sessions(),
        ),
        "list",
    )
    parent.attach_mock(
        mocker.patch(
            f"{CHECKOUT}.stripe.checkout.Session.expire_async",
            new_callable=AsyncMock,
        ),
        "expire",
    )
    return parent


def _sessions(*sessions: dict) -> MagicMock:
    page = MagicMock()
    page.data = [stripe.checkout.Session.construct_from(s, "sk_test") for s in sessions]
    page.has_more = False
    return page


def _user(stripe_customer_id: str | None) -> MagicMock:
    user = MagicMock()
    user.stripe_customer_id = stripe_customer_id
    return user


def _page(*subscriptions: dict) -> MagicMock:
    page = MagicMock()
    page.data = [
        stripe.Subscription.construct_from(s, "sk_test") for s in subscriptions
    ]
    page.has_more = False
    return page


def _stripe_lists(active: MagicMock, trialing: MagicMock):
    pages = {"active": active, "trialing": trialing}

    def by_status(*_args, **kwargs):
        return pages.get(kwargs.get("status"), _page())

    return by_status


def _one_per_status(*statuses: str):
    """Holds one subscription in each status and, like Stripe, lists only the
    ones in the status a call asks for."""

    def by_status(*_args, **kwargs):
        status = kwargs.get("status")
        if status not in statuses:
            return _page()
        return _page({"id": f"sub_{status}", "status": status, "schedule": None})

    return by_status


def _listed_statuses(listed: MagicMock) -> list:
    return [call.kwargs.get("status") for call in listed.call_args_list]


def _touched(*stripe_methods: MagicMock) -> set[str]:
    return {call.args[0] for method in stripe_methods for call in method.call_args_list}


def _assert_nothing_invoiced(cancel: AsyncMock) -> None:
    for call in cancel.await_args_list:
        assert call.kwargs.get("invoice_now") is not True
        assert call.kwargs.get("prorate") is not True


@contextmanager
def _stripe(list_by_status) -> Iterator[tuple[MagicMock, AsyncMock, AsyncMock]]:
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_user(CUSTOMER),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            side_effect=list_by_status,
        ) as listed,
        patch(
            "backend.data.credit.stripe.Subscription.cancel_async",
            new_callable=AsyncMock,
        ) as cancel,
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as modify,
    ):
        yield listed, cancel, modify


async def _cancelled(sub_id: str, **params) -> stripe.Subscription:
    """Stripe echoes the cancel's comment back on the cancelled subscription."""
    details = {"reason": "cancellation_requested"}
    details.update(params.get("cancellation_details") or {})
    return stripe.Subscription.construct_from(
        {
            "id": sub_id,
            "customer": CUSTOMER,
            "status": "canceled",
            "metadata": {"trial_enrollment_id": "trial-1", "user_id": "user-1"},
            "cancellation_details": details,
        },
        "sk_test",
    )


@pytest.mark.asyncio
async def test_a_user_without_a_stripe_customer_cancels_nothing(checkouts: MagicMock):
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_user(None),
        ),
        patch("backend.data.credit.stripe.Subscription.list_async") as listed,
        patch("backend.data.credit.stripe.Subscription.cancel_async") as cancel,
        patch("backend.data.credit.stripe.Customer.create_async") as created,
    ):
        assert await credit.cancel_all_subscriptions_now("user-1") == 0

    listed.assert_not_called()
    cancel.assert_not_called()
    created.assert_not_called()
    checkouts.list.assert_not_called()


@pytest.mark.asyncio
async def test_every_open_checkout_expires_before_any_subscription_is_listed(
    checkouts: MagicMock,
):
    """A Checkout left open in another tab would otherwise subscribe and bill
    the closed account once finished, whether a plan, a trial or a top-up."""
    checkouts.list.return_value = _sessions(
        {"id": "cs_plan", "mode": "subscription"},
        {"id": "cs_top_up", "mode": "payment"},
    )
    with _stripe(_one_per_status("active")) as (listed, _cancel, _modify):
        checkouts.attach_mock(listed, "list_subscriptions")
        assert await credit.cancel_all_subscriptions_now("user-1") == 1

    checkouts.list.assert_awaited_once_with(customer=CUSTOMER, status="open", limit=100)
    assert [c.args[0] for c in checkouts.expire.await_args_list] == [
        "cs_plan",
        "cs_top_up",
    ]
    order = [name for name, _, _ in checkouts.mock_calls]
    last_expire = len(order) - 1 - order[::-1].index("expire")
    assert last_expire < order.index("list_subscriptions")


@pytest.mark.asyncio
async def test_paid_and_trial_subscriptions_end_now_without_an_invoice():
    active = _page({"id": "sub_paid", "schedule": None})
    trialing = _page({"id": "sub_trial", "schedule": None})
    comment = {"comment": credit.ACCOUNT_DELETION_CANCELLATION_COMMENT}
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_user(CUSTOMER),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            side_effect=_stripe_lists(active, trialing),
        ) as listed,
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as modify,
        patch(
            "backend.data.credit.stripe.Subscription.cancel_async",
            new_callable=AsyncMock,
        ) as cancel,
    ):
        assert await credit.cancel_all_subscriptions_now("user-1") == 2

    assert {call.kwargs["customer"] for call in listed.call_args_list} == {CUSTOMER}
    cancel.assert_any_await("sub_paid", cancellation_details=comment)
    cancel.assert_any_await(
        "sub_trial", invoice_now=False, prorate=False, cancellation_details=comment
    )
    assert cancel.await_count == 2
    for call in cancel.await_args_list:
        assert call.kwargs.get("invoice_now") is not True
        assert call.kwargs.get("prorate") is not True
    # Never the at-period-end road: the account is closing now.
    modify.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", UNSETTLED)
async def test_an_unsettled_plan_ends_now_with_the_comment_and_no_invoice(status):
    """Cancelling stops Stripe collecting the plan's open invoices, so what
    is still owed is written off; the cancel itself invoices nothing."""
    with _stripe(_one_per_status(status)) as (_listed, cancel, modify):
        assert await credit.cancel_all_subscriptions_now("user-1") == 1

    cancel.assert_awaited_once_with(f"sub_{status}", cancellation_details=COMMENT)
    _assert_nothing_invoiced(cancel)
    modify.assert_not_called()


@pytest.mark.asyncio
async def test_every_open_subscription_ends_and_ended_ones_are_never_touched():
    every_status = (*BILLABLE, *UNSETTLED, *ENDED)
    with _stripe(_one_per_status(*every_status)) as (listed, cancel, modify):
        assert await credit.cancel_all_subscriptions_now("user-1") == 6

    # Each open status is asked for once, by name: never "all", "ended" or
    # no status at all, which would bring back subscriptions Stripe can no
    # longer cancel.
    assert sorted(_listed_statuses(listed)) == sorted([*BILLABLE, *UNSETTLED])
    assert _touched(cancel, modify) == {f"sub_{s}" for s in (*BILLABLE, *UNSETTLED)}
    assert all(
        call.kwargs["cancellation_details"] == COMMENT
        for call in cancel.await_args_list
    )
    _assert_nothing_invoiced(cancel)
    modify.assert_not_called()


async def _cancel_plan_at_period_end() -> None:
    await credit.cancel_stripe_subscription("user-1")


async def _clean_up_after_upgrade() -> None:
    await credit._cleanup_stale_subscriptions(CUSTOMER, "sub_new")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "caller",
    [_cancel_plan_at_period_end, _clean_up_after_upgrade],
)
async def test_other_cancels_still_reach_only_active_and_trialing_plans(caller):
    """Only an account deletion ends a past-due or other unsettled plan."""
    every_status = (*BILLABLE, *UNSETTLED, *ENDED)
    with (
        _stripe(_one_per_status(*every_status)) as (listed, cancel, modify),
        patch("backend.data.credit.invalidate_billing_lookups", new_callable=AsyncMock),
    ):
        await caller()

    assert _listed_statuses(listed) == BILLABLE
    assert _touched(cancel, modify) == {"sub_active", "sub_trialing"}


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["list_async", "cancel_async"])
async def test_a_stripe_error_reaches_the_caller(failing):
    active = _page({"id": "sub_paid", "schedule": None})
    trialing = _page()
    stripe_methods = {
        "list_async": AsyncMock(side_effect=_stripe_lists(active, trialing)),
        "cancel_async": AsyncMock(),
    }
    stripe_methods[failing] = AsyncMock(side_effect=stripe.StripeError("down"))
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_user(CUSTOMER),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            stripe_methods["list_async"],
        ),
        patch(
            "backend.data.credit.stripe.Subscription.cancel_async",
            stripe_methods["cancel_async"],
        ),
    ):
        with pytest.raises(stripe.StripeError):
            await credit.cancel_all_subscriptions_now("user-1")


@pytest.mark.asyncio
async def test_an_enrolled_trial_still_syncs_its_tier_but_not_its_person():
    """The inline trial sync keeps the drop to NO_TIER, but must not refresh
    the PostHog person that the erasure deletes."""
    trialing = _page(
        {
            "id": "sub_trial",
            "schedule": None,
            "metadata": {"trial_enrollment_id": "trial-1", "user_id": "user-1"},
        }
    )
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_user(CUSTOMER),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            side_effect=_stripe_lists(_page(), trialing),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.cancel_async",
            side_effect=_cancelled,
        ),
        patch(
            "backend.data.credit._sync_subscription_tier_from_stripe",
            new_callable=AsyncMock,
        ) as tier_sync,
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as person_sync,
    ):
        assert await credit.cancel_all_subscriptions_now("user-1") == 1

    tier_sync.assert_awaited_once()
    assert tier_sync.await_args.args[0]["id"] == "sub_trial"
    person_sync.assert_not_called()


def _deleted_event(comment: str | None, status: str = "canceled") -> dict:
    details: dict = {"reason": "cancellation_requested"}
    if comment:
        details["comment"] = comment
    return {
        "id": "sub_paid",
        "customer": CUSTOMER,
        "status": status,
        "items": {"data": []},
        "cancellation_details": details,
    }


# A past-due, unpaid or paused plan ends "canceled" like a paid one; an
# incomplete one may end "incomplete_expired". All carry the same comment.
@pytest.mark.asyncio
@pytest.mark.parametrize("ended_status", ENDED)
async def test_the_deletion_webhook_drops_the_tier_without_touching_the_person(
    ended_status,
):
    user = MagicMock(
        id="user-1", subscriptionTier=SubscriptionTier.PRO, stripeCustomerId=CUSTOMER
    )
    event = _deleted_event(credit.ACCOUNT_DELETION_CANCELLATION_COMMENT, ended_status)
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=user)),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            return_value=_page(),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.retrieve_async",
            new_callable=AsyncMock,
            return_value=event,
        ),
        patch(
            "backend.data.credit.get_subscription_trial",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as set_tier,
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as person_sync,
    ):
        await credit.sync_subscription_from_stripe(event)

    set_tier.assert_awaited_once_with("user-1", SubscriptionTier.NO_TIER)
    person_sync.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "comment", [None, credit.REPLACED_PLAN_CANCELLATION_COMMENT, "other"]
)
async def test_any_other_cancel_still_refreshes_the_person(comment):
    with (
        patch(
            "backend.data.credit._sync_subscription_tier_from_stripe",
            new_callable=AsyncMock,
        ),
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as person_sync,
    ):
        await credit.sync_subscription_from_stripe(_deleted_event(comment))

    person_sync.assert_called_once_with(stripe_customer_id=CUSTOMER)


def test_only_the_account_deletion_comment_marks_a_deletion():
    assert credit.ACCOUNT_DELETION_CANCELLATION_COMMENT == "autogpt:account_deleted"
    assert credit.is_account_deletion_cancellation(
        _deleted_event(credit.ACCOUNT_DELETION_CANCELLATION_COMMENT)
    )
    assert not credit.is_account_deletion_cancellation(
        _deleted_event(credit.REPLACED_PLAN_CANCELLATION_COMMENT)
    )
    assert not credit.is_account_deletion_cancellation({"id": "sub_paid"})
    assert not credit.is_account_deletion_cancellation(
        {"id": "sub_paid", "cancellation_details": None}
    )
