"""Restarting the billing period after a late payment, against a fake Stripe.

The fake keeps the state that matters here: each subscription's period and
latest invoice, credit notes, the customer's balance and what the card was
charged. Resetting the anchor works as Stripe documents it for classic
billing: the new period starts now and is invoiced at once, the customer's
balance pays what it can and the card the rest. Idempotency keys replay for
24 hours, so a test can also deliver an event after they have expired.
"""

import asyncio
from contextlib import ExitStack, asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from dateutil.relativedelta import relativedelta
from prisma.enums import SubscriptionTier

from backend.data.subscription_billing_restart import (
    RESTARTED_FROM_INVOICE_KEY,
    RestartOutcome,
    restart_period_after_late_payment,
)
from backend.data.subscription_billing_restart_invoice import line_period_start

CUSTOMER = "cus_1"
SUB = "sub_1"
PRICE = 5000
DAY = 86400
HOUR = 3600


def _ts(year: int, month: int, day: int, hour: int = 12) -> int:
    return int(datetime(year, month, day, hour, tzinfo=UTC).timestamp())


class FakeStripe:
    def __init__(self, now: int) -> None:
        self.now = now
        self.subscriptions: dict[str, dict] = {}
        self.invoices: dict[str, dict] = {}
        self.credit_notes: list[dict] = []
        self.balance = 0
        self.card_charges: list[int] = []
        self.idempotent: dict[str, object] = {}
        self.fail_next_modify = 0
        self.card_declines = False
        self.modify_started = asyncio.Event()
        self.hold_modify: asyncio.Event | None = None
        self.preview_total: int | None = None
        self.lock_calls: list[tuple] = []
        self.tier: dict = {}
        self.rows: dict[str, MagicMock] = {}

    def add_subscription(
        self, period_start: int, interval: str = "month", **extra
    ) -> dict:
        sub = {
            "id": SUB,
            "customer": CUSTOMER,
            "status": "active",
            "created": period_start - 30 * DAY,
            "interval": interval,
            "billing_cycle_anchor": period_start,
            "current_period_start": period_start,
            "current_period_end": self._advance(period_start, period_start, 1),
            "latest_invoice": None,
            "schedule": None,
            "pending_update": None,
            **extra,
        }
        self.subscriptions[SUB] = sub
        return sub

    def add_invoice(
        self,
        invoice_id: str,
        *,
        paid_at: int | None,
        billing_reason: str = "subscription_cycle",
        total: int = PRICE,
        sub_id: str = SUB,
        attempt_count: int = 1,
    ) -> None:
        self.invoices[invoice_id] = {
            "id": invoice_id,
            "customer": CUSTOMER,
            "subscription": sub_id,
            "billing_reason": billing_reason,
            "status": "paid" if paid_at else "open",
            "total": total,
            "attempt_count": attempt_count,
            "status_transitions": {"paid_at": paid_at},
            "lines": {
                "data": [
                    {
                        "type": "subscription",
                        "proration": False,
                        "period": {
                            "start": self.subscriptions.get(sub_id, {}).get(
                                "current_period_start", paid_at
                            )
                        },
                    }
                ],
                "has_more": False,
            },
        }
        if paid_at:
            self.card_charges.append(total)
        if sub_id in self.subscriptions:
            self.subscriptions[sub_id]["latest_invoice"] = invoice_id

    def pay(self, invoice_id: str, paid_at: int) -> None:
        invoice = self.invoices[invoice_id]
        invoice.update(status="paid", status_transitions={"paid_at": paid_at})
        self.card_charges.append(invoice["total"])

    def _advance(self, anchor: int, start: int, periods: int) -> int:
        interval = self.subscriptions.get(SUB, {}).get("interval", "month")
        anchor_dt = datetime.fromtimestamp(anchor, UTC)
        step = relativedelta(years=1) if interval == "year" else relativedelta(months=1)
        n = 1
        while True:
            candidate = int((anchor_dt + step * n).timestamp())
            if candidate > start:
                return candidate
            n += 1

    def _replay(self, key: str | None):
        if key is None:
            return None
        return self.idempotent.get(key)

    async def retrieve_invoice(self, invoice_id: str, **_):
        return dict(self.invoices[invoice_id])

    async def retrieve_subscription(self, sub_id: str, **_):
        return dict(self.subscriptions[sub_id])

    async def list_credit_notes(self, invoice: str, limit: int):
        page = MagicMock()
        page.data = [dict(n) for n in self.credit_notes if n["invoice"] == invoice]
        page.has_more = False
        return page

    async def create_credit_note(self, idempotency_key=None, **params):
        if (replayed := self._replay(idempotency_key)) is not None:
            return replayed
        note = {
            "id": f"cn_{len(self.credit_notes) + 1}",
            "invoice": params["invoice"],
            "status": "issued",
            "amount": params["amount"],
            "metadata": params.get("metadata") or {},
        }
        assert params["credit_amount"] == params["amount"]
        self.credit_notes.append(note)
        self.balance -= params["credit_amount"]
        if idempotency_key:
            self.idempotent[idempotency_key] = note
        return note

    async def modify_subscription(self, sub_id: str, idempotency_key=None, **params):
        self.modify_started.set()
        if self.hold_modify is not None:
            await self.hold_modify.wait()
        if (replayed := self._replay(idempotency_key)) is not None:
            return replayed
        if self.fail_next_modify:
            self.fail_next_modify -= 1
            raise stripe.APIConnectionError("modify failed")
        assert params == {
            "billing_cycle_anchor": "now",
            "proration_behavior": "none",
            "payment_behavior": "error_if_incomplete",
        }
        sub = self.subscriptions[sub_id]
        total = self.new_period_total(sub_id)
        from_balance = min(total, -self.balance)
        card = total - from_balance
        if card and self.card_declines:
            raise stripe.CardError("declined", None, "card_declined")
        self.balance += from_balance
        if card:
            self.card_charges.append(card)
        invoice_id = f"in_reset_{len(self.invoices)}"
        self.invoices[invoice_id] = {
            "id": invoice_id,
            "customer": CUSTOMER,
            "subscription": sub_id,
            "billing_reason": "subscription_update",
            "status": "paid",
            "total": total,
            "status_transitions": {"paid_at": self.now},
        }
        sub.update(
            billing_cycle_anchor=self.now,
            current_period_start=self.now,
            current_period_end=self._advance(self.now, self.now, 1),
            latest_invoice=invoice_id,
        )
        result = dict(sub)
        if idempotency_key:
            self.idempotent[idempotency_key] = result
        return result

    def new_period_total(self, sub_id: str) -> int:
        sub = self.subscriptions[sub_id]
        return PRICE * (12 if sub["interval"] == "year" else 1)

    async def preview_invoice(self, subscription: str, subscription_details: dict):
        assert subscription_details == {
            "billing_cycle_anchor": "now",
            "proration_behavior": "none",
        }
        total = self.new_period_total(subscription)
        if self.preview_total is not None:
            total = self.preview_total
        return {"total": total, "amount_due": max(0, total + self.balance)}

    async def list_subscriptions(self, customer: str, status: str, limit: int):
        page = MagicMock()
        page.data = [
            dict(s)
            for s in self.subscriptions.values()
            if s["customer"] == customer and s["status"] == status
        ]
        page.has_more = False
        return page

    def renewals(self, count: int) -> list[str]:
        sub = self.subscriptions[SUB]
        dates, start = [], sub["current_period_start"]
        for _ in range(count):
            start = self._advance(sub["billing_cycle_anchor"], start, 1)
            dates.append(datetime.fromtimestamp(start, UTC).date().isoformat())
        return dates


@pytest.fixture
def fake():
    fake = FakeStripe(now=_ts(2026, 3, 9))
    tier = {"value": SubscriptionTier.PRO}
    lock = asyncio.Lock()

    @asynccontextmanager
    async def fake_transaction(timeout=None):
        async with lock:
            tx = MagicMock()
            tx.execute_raw = AsyncMock(
                side_effect=lambda *args: fake.lock_calls.append(args)
            )
            yield tx

    async def find_user(where):
        return MagicMock(id="user-1", subscriptionTier=tier["value"])

    rows: dict[str, MagicMock] = {}

    async def upsert(where, data):
        key = where["stripeInvoiceId"]
        if key not in rows:
            rows[key] = MagicMock(
                **data["create"], stripeCreditNoteId=None, restartedAt=None
            )
        return rows[key]

    async def update(where, data):
        row = rows[where["stripeInvoiceId"]]
        for field, value in data.items():
            setattr(row, field, value)
        return row

    module = "backend.data.subscription_billing_restart"
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                f"{module}.SubscriptionBillingRestart.prisma",
                return_value=MagicMock(
                    upsert=AsyncMock(side_effect=upsert),
                    update=AsyncMock(side_effect=update),
                ),
            )
        )
        stack.enter_context(patch(f"{module}.transaction", fake_transaction))
        stack.enter_context(
            patch(
                f"{module}.User.prisma",
                return_value=MagicMock(find_first=AsyncMock(side_effect=find_user)),
            )
        )
        stack.enter_context(
            patch(f"{module}.invalidate_billing_lookups", new_callable=AsyncMock)
        )
        for target, method in (
            ("stripe.Invoice.retrieve_async", fake.retrieve_invoice),
            ("stripe.Subscription.retrieve_async", fake.retrieve_subscription),
            ("stripe.Subscription.modify_async", fake.modify_subscription),
            ("stripe.Subscription.list_async", fake.list_subscriptions),
            ("stripe.CreditNote.list_async", fake.list_credit_notes),
            ("stripe.CreditNote.create_async", fake.create_credit_note),
            ("stripe.Invoice.create_preview_async", fake.preview_invoice),
        ):
            stack.enter_context(patch(target, side_effect=method))
        fake.tier = tier
        fake.rows = rows
        yield fake


def _fail_on_3rd_pay_on_9th(fake: FakeStripe) -> None:
    fake.add_subscription(period_start=_ts(2026, 3, 3), status="active")
    fake.add_invoice("in_march", paid_at=_ts(2026, 3, 9))


@pytest.mark.asyncio
async def test_due_on_the_3rd_paid_on_the_9th_renews_on_the_9th_and_is_charged_once(
    fake,
):
    _fail_on_3rd_pay_on_9th(fake)

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.RESTARTED
    )

    sub = fake.subscriptions[SUB]
    assert sub["current_period_start"] == _ts(2026, 3, 9)
    assert sub["current_period_end"] == _ts(2026, 4, 9)
    assert fake.renewals(3) == ["2026-04-09", "2026-05-09", "2026-06-09"]
    # The late payment is the only money taken: the new period's invoice was
    # paid from the credited balance, nothing is left over.
    assert fake.card_charges == [PRICE]
    assert fake.balance == 0
    assert [n["metadata"] for n in fake.credit_notes] == [
        {RESTARTED_FROM_INVOICE_KEY: "in_march"}
    ]
    assert fake.invoices[sub["latest_invoice"]]["status"] == "paid"


@pytest.mark.asyncio
async def test_a_renewal_paid_on_its_first_attempt_keeps_its_date(fake):
    fake.add_subscription(period_start=_ts(2026, 3, 3))
    fake.add_invoice("in_march", paid_at=_ts(2026, 3, 3) + HOUR)

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NOT_LATE
    )
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 3, 3)
    assert fake.credit_notes == []


@pytest.mark.asyncio
async def test_a_retry_soon_after_a_failed_first_attempt_also_restarts(fake):
    fake.now = _ts(2026, 3, 3) + 2 * HOUR
    fake.add_subscription(period_start=_ts(2026, 3, 3))
    fake.add_invoice("in_march", paid_at=fake.now, attempt_count=2)

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.RESTARTED
    )
    assert fake.subscriptions[SUB]["current_period_start"] == fake.now


@pytest.mark.asyncio
async def test_a_recorded_restart_is_never_repeated(fake):
    _fail_on_3rd_pay_on_9th(fake)
    await restart_period_after_late_payment("in_march")
    assert fake.rows["in_march"].restartedAt is not None
    assert fake.rows["in_march"].stripeCreditNoteId == "cn_1"

    # Even if Stripe's subscription read back as it was before the restart
    # (a stale replica, a replayed object), the recorded restart holds.
    fake.subscriptions[SUB].update(
        latest_invoice="in_march", current_period_start=_ts(2026, 3, 3)
    )
    fake.idempotent.clear()
    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NOT_APPLICABLE
    )
    assert len(fake.credit_notes) == 1
    assert fake.card_charges == [PRICE]


@pytest.mark.asyncio
async def test_pending_bank_approval_restarts_only_once_the_payment_is_confirmed(
    fake,
):
    fake.add_subscription(period_start=_ts(2026, 3, 3), status="past_due")
    fake.add_invoice("in_march", paid_at=None)

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NOT_APPLICABLE
    )
    assert fake.credit_notes == []

    # The bank confirms on the 12th; the period runs from then.
    fake.now = _ts(2026, 3, 12)
    fake.pay("in_march", paid_at=_ts(2026, 3, 12))
    fake.subscriptions[SUB]["status"] = "active"
    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.RESTARTED
    )
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 3, 12)
    assert fake.card_charges == [PRICE]


@pytest.mark.asyncio
async def test_a_delayed_notification_still_restarts_once_and_never_charges_more(
    fake,
):
    _fail_on_3rd_pay_on_9th(fake)
    # Stripe's event arrives two days after the payment.
    fake.now = _ts(2026, 3, 11)

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.RESTARTED
    )
    # Stripe anchors a classic subscription only at "now": the two days the
    # notification was held up are free, never charged.
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 3, 11)
    assert fake.card_charges == [PRICE]
    assert fake.balance == 0


@pytest.mark.asyncio
async def test_repeated_events_move_the_renewal_date_once(fake):
    _fail_on_3rd_pay_on_9th(fake)
    await restart_period_after_late_payment("in_march")
    restarted = dict(fake.subscriptions[SUB])

    # Stripe's idempotency keys have expired by the time this redelivery lands.
    fake.idempotent.clear()
    fake.now = _ts(2026, 3, 14)
    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NOT_APPLICABLE
    )
    assert fake.subscriptions[SUB] == restarted
    assert len(fake.credit_notes) == 1
    assert fake.card_charges == [PRICE]


@pytest.mark.asyncio
async def test_simultaneous_events_for_one_payment_restart_once(fake):
    _fail_on_3rd_pay_on_9th(fake)
    fake.hold_modify = asyncio.Event()

    # invoice.paid, invoice.payment_succeeded and invoice_payment.paid arrive
    # together; the first holds Stripe's update open while the others wait.
    tasks = [
        asyncio.create_task(restart_period_after_late_payment("in_march"))
        for _ in range(3)
    ]
    await asyncio.wait_for(fake.modify_started.wait(), timeout=10)
    fake.hold_modify.set()
    outcomes = await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)

    assert sorted(outcomes) == sorted(
        [RestartOutcome.RESTARTED] + [RestartOutcome.NOT_APPLICABLE] * 2
    )
    assert len(fake.credit_notes) == 1
    assert fake.card_charges == [PRICE]
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 3, 9)


@pytest.mark.asyncio
async def test_a_failure_between_credit_and_reset_is_finished_by_the_retry(fake):
    _fail_on_3rd_pay_on_9th(fake)
    fake.fail_next_modify = 1

    with pytest.raises(stripe.APIConnectionError):
        await restart_period_after_late_payment("in_march")
    assert len(fake.credit_notes) == 1
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 3, 3)

    fake.idempotent.clear()
    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.RESTARTED
    )
    assert len(fake.credit_notes) == 1
    assert fake.balance == 0
    assert fake.card_charges == [PRICE]


@pytest.mark.asyncio
async def test_an_old_payment_after_a_newer_period_began_moves_nothing(fake):
    # The March bill was left open and April renewed on time; paying March
    # now must not shift April's period.
    fake.add_subscription(period_start=_ts(2026, 4, 9))
    fake.add_invoice("in_march", paid_at=_ts(2026, 4, 20))
    fake.add_invoice("in_april", paid_at=_ts(2026, 4, 9) + HOUR)

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NOT_APPLICABLE
    )
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 4, 9)
    assert fake.credit_notes == []


@pytest.mark.asyncio
async def test_an_old_plan_revived_beside_a_newer_plan_is_not_restarted(fake):
    _fail_on_3rd_pay_on_9th(fake)
    fake.subscriptions["sub_new"] = {
        "id": "sub_new",
        "customer": CUSTOMER,
        "status": "active",
        "created": fake.subscriptions[SUB]["created"] + DAY,
    }

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NOT_APPLICABLE
    )
    assert fake.credit_notes == []


@pytest.mark.asyncio
async def test_annual_plan_restarts_for_a_full_year_from_the_payment(fake):
    fake.now = _ts(2028, 2, 29)
    fake.add_subscription(period_start=_ts(2028, 2, 20), interval="year")
    fake.add_invoice("in_year", paid_at=_ts(2028, 2, 29), total=PRICE * 12)

    assert await restart_period_after_late_payment("in_year") == (
        RestartOutcome.RESTARTED
    )
    # February 29 renews on February 28 in years without one.
    assert fake.renewals(2) == ["2029-02-28", "2030-02-28"]
    assert fake.card_charges == [PRICE * 12]
    assert fake.balance == 0


@pytest.mark.asyncio
async def test_month_end_payment_keeps_the_last_day_rule(fake):
    fake.now = _ts(2026, 1, 31)
    fake.add_subscription(period_start=_ts(2026, 1, 20))
    fake.add_invoice("in_jan", paid_at=_ts(2026, 1, 31))

    await restart_period_after_late_payment("in_jan")

    assert fake.renewals(3) == ["2026-02-28", "2026-03-31", "2026-04-30"]


@pytest.mark.asyncio
async def test_wallet_payment_restarts_the_period_like_a_card(fake):
    fake.add_subscription(period_start=_ts(2026, 3, 3))
    fake.add_invoice("in_march", paid_at=_ts(2026, 3, 9))
    fake.card_charges.clear()  # paid out of band from the wallet

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.RESTARTED
    )
    assert fake.card_charges == []
    assert fake.balance == 0


@pytest.mark.asyncio
async def test_expired_trial_paid_later_starts_its_first_month_from_the_payment(
    fake,
):
    trial_end = _ts(2026, 3, 3)
    fake.add_subscription(period_start=trial_end, trial_end=trial_end)
    fake.add_invoice("in_first", paid_at=_ts(2026, 3, 9))

    assert await restart_period_after_late_payment("in_first") == (
        RestartOutcome.RESTARTED
    )
    sub = fake.subscriptions[SUB]
    assert sub["current_period_end"] == _ts(2026, 4, 9)
    # No trial is added to move the date: the subscription stays a paid one.
    assert sub["trial_end"] == trial_end
    assert sub["status"] == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["subscription_update", "manual"])
async def test_upgrades_and_one_off_invoices_never_move_the_renewal(fake, reason):
    fake.add_subscription(period_start=_ts(2026, 3, 3))
    fake.add_invoice("in_other", paid_at=_ts(2026, 3, 20), billing_reason=reason)

    assert await restart_period_after_late_payment("in_other") == (
        RestartOutcome.NOT_APPLICABLE
    )
    assert fake.credit_notes == []


@pytest.mark.asyncio
async def test_enterprise_account_is_left_alone(fake):
    _fail_on_3rd_pay_on_9th(fake)
    fake.tier["value"] = SubscriptionTier.ENTERPRISE

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NOT_APPLICABLE
    )
    assert fake.credit_notes == []


@pytest.mark.asyncio
async def test_a_scheduled_change_is_left_for_a_person(fake):
    fake.add_subscription(period_start=_ts(2026, 3, 3), schedule="sub_sched_1")
    fake.add_invoice("in_march", paid_at=_ts(2026, 3, 9))

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NEEDS_MANUAL_FIX
    )
    assert fake.credit_notes == []


@pytest.mark.asyncio
async def test_an_invoice_already_credited_otherwise_is_not_credited_again(fake):
    _fail_on_3rd_pay_on_9th(fake)
    fake.credit_notes.append(
        {"id": "cn_refund", "invoice": "in_march", "status": "issued", "metadata": {}}
    )

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NEEDS_MANUAL_FIX
    )
    assert len(fake.credit_notes) == 1
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 3, 3)


@pytest.mark.asyncio
async def test_a_late_invoice_cheaper_than_the_new_period_is_never_topped_up(fake):
    """A one-time promotion paid half the late bill; resetting would charge
    the card the other half, which the customer never agreed to."""
    _fail_on_3rd_pay_on_9th(fake)
    fake.invoices["in_march"]["total"] = PRICE // 2

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NEEDS_MANUAL_FIX
    )
    assert fake.card_charges == [PRICE]
    assert fake.credit_notes == []
    assert fake.balance == 0
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 3, 3)


@pytest.mark.asyncio
async def test_a_price_change_after_the_check_is_refused_not_charged(fake):
    _fail_on_3rd_pay_on_9th(fake)
    fake.invoices["in_march"]["total"] = PRICE - 1000
    fake.preview_total = PRICE - 1000  # the price rose after the preview
    fake.card_declines = True

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NEEDS_MANUAL_FIX
    )
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 3, 3)
    assert fake.subscriptions[SUB]["status"] == "active"


@pytest.mark.asyncio
async def test_each_restart_holds_the_subscriptions_advisory_lock(fake):
    _fail_on_3rd_pay_on_9th(fake)

    await restart_period_after_late_payment("in_march")

    assert fake.lock_calls == [
        (
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"billing-restart:{SUB}",
        )
    ]


@pytest.mark.asyncio
async def test_a_balance_the_customer_owes_is_never_charged_by_the_reset(fake):
    _fail_on_3rd_pay_on_9th(fake)
    fake.balance = 700  # owed from earlier, Stripe adds it to the next invoice

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NEEDS_MANUAL_FIX
    )
    assert fake.credit_notes == []
    assert fake.card_charges == [PRICE]
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 3, 3)


@pytest.mark.asyncio
async def test_an_invoice_billing_more_than_the_plan_is_left_for_a_person(fake):
    """Crediting a one-off line too would leave it on the balance for a
    later invoice, since the reset bills only the plan."""
    _fail_on_3rd_pay_on_9th(fake)
    fake.invoices["in_march"]["lines"]["data"].append(
        {"type": "invoiceitem", "proration": False, "period": {"start": 1}}
    )

    assert await restart_period_after_late_payment("in_march") == (
        RestartOutcome.NEEDS_MANUAL_FIX
    )
    assert fake.credit_notes == []


@pytest.mark.asyncio
async def test_a_plan_changed_after_the_credit_is_not_reset(fake):
    _fail_on_3rd_pay_on_9th(fake)
    original = fake.create_credit_note

    async def credit_then_upgrade(**params):
        note = await original(**params)
        fake.subscriptions[SUB]["latest_invoice"] = "in_upgrade"
        return note

    fake.create_credit_note = credit_then_upgrade
    with patch("stripe.CreditNote.create_async", side_effect=credit_then_upgrade):
        outcome = await restart_period_after_late_payment("in_march")

    assert outcome == RestartOutcome.NEEDS_MANUAL_FIX
    assert fake.subscriptions[SUB]["current_period_start"] == _ts(2026, 3, 3)
    assert fake.card_charges == [PRICE]


def test_the_period_start_is_read_from_the_recurring_plan_line_only():
    invoice = {
        "lines": {
            "data": [
                {"type": "invoiceitem", "proration": False, "period": {"start": 5}},
                {"type": "subscription", "proration": True, "period": {"start": 7}},
                {"type": "subscription", "proration": False, "period": {"start": 9}},
            ]
        }
    }
    basil = {
        "lines": {
            "data": [
                {
                    "parent": {
                        "type": "subscription_item_details",
                        "subscription_item_details": {"proration": False},
                    },
                    "period": {"start": 11},
                }
            ]
        }
    }

    assert line_period_start(invoice) == 9
    assert line_period_start(basil) == 11
