"""The billing emails fire on exactly one signal each.

The failure modes these guard against are the classic ones: greeting a
returning customer like a stranger, emailing on every Stripe retry, sending the
cancellation confirmation weeks late off `subscription.deleted`, and treating
every `subscription.updated` as a cancellation.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import NotificationType

from backend.notifications import lifecycle

CUSTOMER = "cus_1"
PLAN_PATCH = "backend.notifications.lifecycle.plan_from_subscription"
INVOICE_PLAN_PATCH = "backend.notifications.lifecycle.plan_from_invoice"


class _User:
    def __init__(self, welcome_sent_at=None):
        self.id = "user-1"
        self.email = "sam@example.com"
        self.name = "Sam Carter"
        self.welcomeEmailSentAt = welcome_sent_at


def _subscription(**over) -> dict:
    sub = {
        "id": "sub_1",
        "customer": CUSTOMER,
        "current_period_end": 1789200000,
        "cancel_at_period_end": False,
        "ended_at": 1789200000,
        "items": {"data": [{"price": {"id": "price_1", "unit_amount": 5000,
                                      "currency": "usd",
                                      "recurring": {"interval": "month"}}}]},
    }
    sub.update(over)
    return sub


def _invoice(**over) -> dict:
    invoice = {
        "id": "in_1",
        "customer": CUSTOMER,
        "amount_due": 5000,
        "currency": "usd",
        "attempt_count": 1,
        "next_payment_attempt": 1789200000,
        "period_end": 1789200000,
        "lines": {"data": [{"price": {"id": "price_1", "unit_amount": 5000,
                                      "currency": "usd",
                                      "recurring": {"interval": "month"}}}]},
    }
    invoice.update(over)
    return invoice


def _context(user, claim=True):
    from backend.data.notifications import SubscriptionPlan

    plan = SubscriptionPlan(
        name="Pro",
        cycle="monthly",
        cycle_noun="month",
        label="Pro — monthly",
        price_display="$50.00 / month",
    )
    return [
        patch(
            "backend.notifications.lifecycle._user_for", AsyncMock(return_value=user)
        ),
        patch(PLAN_PATCH, AsyncMock(return_value=plan)),
        patch(INVOICE_PLAN_PATCH, AsyncMock(return_value=plan)),
        patch(
            "backend.notifications.lifecycle.claim_once",
            AsyncMock(return_value=claim),
        ),
        patch(
            "backend.notifications.lifecycle.queue_notification_async", AsyncMock()
        ),
        patch("backend.notifications.lifecycle.queue_audience_change", AsyncMock()),
    ]


async def _run(coro_factory, user, claim=True):
    patches = _context(user, claim)
    started = [p.start() for p in patches]
    try:
        await coro_factory()
        return {"notify": started[4], "audience": started[5]}
    finally:
        for p in patches:
            p.stop()


@pytest.mark.asyncio
async def test_first_subscription_gets_the_welcome_and_the_tour():
    user = _User(welcome_sent_at=None)
    with patch(
        "backend.notifications.lifecycle._claim_welcome",
        AsyncMock(return_value=True),
    ):
        calls = await _run(
            lambda: lifecycle.on_checkout_completed({"customer": CUSTOMER},
                                                    _subscription()),
            user,
        )
    queued = calls["notify"].await_args.args[0]
    assert queued.type is NotificationType.SUBSCRIPTION_WELCOME
    assert queued.data.user_name == "Sam"
    calls["audience"].assert_awaited_once()
    assert calls["audience"].await_args.args[0].action.value == "enroll_tour"


@pytest.mark.asyncio
async def test_a_returning_customer_is_not_greeted_like_a_stranger():
    user = _User(welcome_sent_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    calls = await _run(
        lambda: lifecycle.on_checkout_completed({"customer": CUSTOMER},
                                                _subscription()),
        user,
    )
    calls["notify"].assert_not_awaited()
    assert calls["audience"].await_args.args[0].action.value == "add_changelog"


@pytest.mark.asyncio
async def test_only_the_first_failed_charge_emails():
    calls = await _run(
        lambda: lifecycle.on_payment_failed(_invoice(attempt_count=1)), _User()
    )
    assert calls["notify"].await_args.args[0].type is NotificationType.PAYMENT_FAILED

    calls = await _run(
        lambda: lifecycle.on_payment_failed(_invoice(attempt_count=3)), _User()
    )
    calls["notify"].assert_not_awaited()


@pytest.mark.asyncio
async def test_no_retries_left_is_the_final_notice_not_another_heads_up():
    calls = await _run(
        lambda: lifecycle.on_payment_failed(_invoice(next_payment_attempt=None)),
        _User(),
    )
    queued = calls["notify"].await_args.args[0]
    assert queued.type is NotificationType.PAYMENT_FINAL_NOTICE


@pytest.mark.asyncio
async def test_a_replayed_webhook_does_not_send_twice():
    calls = await _run(
        lambda: lifecycle.on_payment_failed(_invoice()), _User(), claim=False
    )
    calls["notify"].assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_fires_on_the_flip_not_on_every_update():
    calls = await _run(
        lambda: lifecycle.on_subscription_updated(
            _subscription(cancel_at_period_end=True),
            {"cancel_at_period_end": False},
        ),
        _User(),
    )
    queued = calls["notify"].await_args.args[0]
    assert queued.type is NotificationType.SUBSCRIPTION_CANCELLED

    # An unrelated update carries no cancel_at_period_end in previous_attributes.
    calls = await _run(
        lambda: lifecycle.on_subscription_updated(
            _subscription(), {"default_payment_method": "pm_old"}
        ),
        _User(),
    )
    calls["notify"].assert_not_awaited()


@pytest.mark.asyncio
async def test_resuming_closes_the_loop_on_the_cancellation_email():
    calls = await _run(
        lambda: lifecycle.on_subscription_updated(
            _subscription(cancel_at_period_end=False),
            {"cancel_at_period_end": True},
        ),
        _User(),
    )
    assert calls["notify"].await_args.args[0].type is NotificationType.SUBSCRIPTION_RESUMED


@pytest.mark.asyncio
async def test_the_ended_email_branches_on_which_road_they_took():
    calls = await _run(
        lambda: lifecycle.on_subscription_deleted(
            _subscription(cancellation_details={"reason": "payment_failed"})
        ),
        _User(),
    )
    queued = calls["notify"].await_args.args[0]
    assert queued.type is NotificationType.SUBSCRIPTION_ENDED
    assert queued.data.due_to_payment is True
    # Churned users get win-back only, never the monthly update.
    assert calls["audience"].await_args.args[0].action.value == "remove_changelog"

    calls = await _run(
        lambda: lifecycle.on_subscription_deleted(
            _subscription(cancellation_details={"reason": "cancellation_requested"})
        ),
        _User(),
    )
    assert calls["notify"].await_args.args[0].data.due_to_payment is False


def test_the_platform_does_not_listen_for_trials():
    assert not hasattr(lifecycle, "on_trial_will_end")
    assert not any(
        name.endswith("TRIAL_ENDING") for name in dir(NotificationType)
    )
