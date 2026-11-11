"""Real Postgres: a usage reset is paid for and granted together or not at
all, once per key, and a card payment grants nothing until Stripe says it is
paid, then exactly once."""

import asyncio
import time
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import stripe
from prisma.enums import SubscriptionTier
from prisma.errors import UniqueViolationError
from prisma.models import CreditTransaction, UsageReset

from backend.copilot.usage_clock import clock_from_state
from backend.data import usage_clock_integration_fixtures as fixtures
from backend.data import usage_reset, usage_reset_card
from backend.data.credit import UserCredit
from backend.data.pro_activation import get_usage_activation_state
from backend.data.usage_clock import save_billing_clock
from backend.data.usage_reset import (
    buy_reset_with_credits,
    credit_transaction_key,
    find_reset_purchase,
    list_reset_purchases,
)
from backend.data.usage_reset_card import (
    CheckoutUnsettled,
    fulfill_card_reset,
    start_card_reset,
)
from backend.util.exceptions import InsufficientBalanceError

billing_user = fixtures.billing_user
pytestmark = fixtures.pytestmark

DAY = 86_400
PRICE = 1_250
# The stored period is compared with the database's clock: it began two days
# ago, so the first weekly window (7.5 days) has days left.
START = int(time.time()) - 2 * DAY
NOW = START + 2 * DAY
WEEK_END = START + 7 * DAY + 12 * 3600
PERIOD = {
    "id": "sub_reset",
    "customer": "cus_x",
    "current_period_start": START,
    "current_period_end": START + 30 * DAY,
    "billing_cycle_anchor": START,
    "created": START - 90 * DAY,
}


@pytest.fixture(autouse=True)
def announced(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """The Redis announcement of a grant, which these tests don't need."""
    mark = AsyncMock()
    monkeypatch.setattr(usage_reset, "mark_reset", mark)
    return mark


async def _buy(user_id: str, key: str = "key-0001", sequence: int = 1):
    return await buy_reset_with_credits(
        user_id,
        key,
        tier=SubscriptionTier.PRO,
        price_credits=PRICE,
        week_start=START,
        week_end=WEEK_END,
        sequence=sequence,
        now=NOW,
    )


async def _balance(user_id: str) -> int:
    return await UserCredit().get_credits(user_id)


async def _charges(user_id: str) -> list[CreditTransaction]:
    return await CreditTransaction.prisma().find_many(
        where={"userId": user_id, "transactionKey": {"startsWith": "USAGE-RESET-"}}
    )


async def _resets_now(user_id: str) -> tuple[int | None, int]:
    state = await get_usage_activation_state(user_id)
    return state.reset_week_start, state.reset_count


@pytest.mark.asyncio
async def test_buying_with_credits_takes_them_and_resets_the_window(
    billing_user, announced: AsyncMock
):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")

    bought = await _buy(billing_user)

    assert bought.charged and bought.balance == 3_750
    purchase = bought.purchase
    assert (purchase.status, purchase.paid_with) == ("granted", "credits")
    assert (purchase.week_start, purchase.week_end) == (START, WEEK_END)
    assert (purchase.sequence, purchase.granted_at) == (1, NOW)
    assert (purchase.price_credits, purchase.tier) == (PRICE, "PRO")
    assert await _balance(billing_user) == 3_750
    charge = (await _charges(billing_user))[0]
    assert charge.transactionKey == credit_transaction_key("key-0001")
    assert charge.amount == -PRICE
    # The usage state every limit check reads now carries the reset.
    assert await _resets_now(billing_user) == (START, 1)
    announced.assert_awaited_once_with(billing_user)


@pytest.mark.asyncio
async def test_the_same_key_buys_once_however_often_it_is_sent(billing_user):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")

    first = await _buy(billing_user)
    again = await _buy(billing_user)
    together = await asyncio.gather(*(_buy(billing_user) for _ in range(5)))

    assert first.charged and not again.charged
    assert not any(purchase.charged for purchase in together)
    assert {p.purchase.id for p in [again, *together]} == {first.purchase.id}
    assert await _balance(billing_user) == 3_750
    assert len(await _charges(billing_user)) == 1
    assert await _resets_now(billing_user) == (START, 1)


@pytest.mark.asyncio
async def test_a_retry_after_the_balance_was_spent_gets_the_purchase_back(
    billing_user,
):
    """The first attempt took the last credits; its retry finds too few and
    must still be answered with the purchase, not with a refusal."""
    await UserCredit().grant_credits(billing_user, PRICE, "test balance")

    first = await _buy(billing_user)
    again = await _buy(billing_user)

    assert first.charged and not again.charged
    assert again.purchase.id == first.purchase.id and again.balance == 0


@pytest.mark.asyncio
async def test_too_few_credits_buy_nothing(billing_user):
    await UserCredit().grant_credits(billing_user, PRICE - 1, "test balance")

    with pytest.raises(InsufficientBalanceError):
        await _buy(billing_user)

    assert await _balance(billing_user) == PRICE - 1
    assert await _charges(billing_user) == []
    assert await find_reset_purchase(billing_user, "key-0001") is None
    assert await _resets_now(billing_user) == (None, 0)


@pytest.mark.asyncio
async def test_two_purchases_racing_for_the_same_reset_charge_once(billing_user):
    """Two tabs, two keys, one limit reached: only one of them is the
    window's next reset. The other is refused and keeps its credits."""
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")

    results = await asyncio.gather(
        _buy(billing_user, "key-tab-a"),
        _buy(billing_user, "key-tab-b"),
        return_exceptions=True,
    )

    refused = [r for r in results if isinstance(r, UniqueViolationError)]
    assert len(refused) == 1
    assert await _balance(billing_user) == 3_750
    assert len(await _charges(billing_user)) == 1
    assert await _resets_now(billing_user) == (START, 1)


@pytest.mark.asyncio
async def test_a_second_reset_in_the_same_window_counts_as_the_second(billing_user):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")

    await _buy(billing_user, "key-first")
    second = await _buy(billing_user, "key-second", sequence=2)

    assert second.charged and second.purchase.sequence == 2
    assert await _balance(billing_user) == 2_500
    assert await _resets_now(billing_user) == (START, 2)
    latest = await list_reset_purchases(billing_user)
    assert [p.sequence for p in latest] == [2, 1]


@pytest.mark.asyncio
async def test_only_the_latest_windows_resets_are_in_the_state(billing_user):
    await UserCredit().grant_credits(billing_user, 5_000, "test balance")
    await _buy(billing_user, "key-this-week")
    await buy_reset_with_credits(
        billing_user,
        "key-next-week",
        tier=SubscriptionTier.PRO,
        price_credits=PRICE,
        week_start=WEEK_END,
        week_end=WEEK_END + 7 * DAY + 12 * 3600,
        sequence=1,
        now=WEEK_END + 60,
    )

    assert await _resets_now(billing_user) == (WEEK_END, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "price,week_end,sequence",
    [(0, WEEK_END, 1), (PRICE, START, 1), (PRICE, WEEK_END, 0)],
)
async def test_a_reset_needs_a_price_a_window_and_a_number(
    billing_user, price: int, week_end: int, sequence: int
):
    with pytest.raises(ValueError):
        await buy_reset_with_credits(
            billing_user,
            "key-0001",
            tier=SubscriptionTier.PRO,
            price_credits=price,
            week_start=START,
            week_end=week_end,
            sequence=sequence,
            now=NOW,
        )


class Stripe:
    """Stripe's side of a Checkout: the sessions opened, and what Stripe
    says about each when asked."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sessions: dict[str, dict] = {}
        self.created: list[dict] = []
        monkeypatch.setattr(usage_reset_card, "stripe_call", self._call)
        monkeypatch.setattr(
            usage_reset_card,
            "get_stripe_customer_id",
            AsyncMock(return_value="cus_test"),
        )

    async def _call(self, function, *args, **kwargs):
        if args:
            return self.sessions[args[0]]
        session = {
            "id": f"cs_test_{uuid4().hex}",
            "url": f"https://checkout.stripe.test/{len(self.created)}",
            "status": "open",
            "mode": kwargs["mode"],
            "payment_status": "unpaid",
            "metadata": kwargs["metadata"],
            "amount_subtotal": kwargs["line_items"][0]["price_data"]["unit_amount"],
            "payment_intent": None,
        }
        self.created.append(kwargs)
        self.sessions[session["id"]] = session
        return _Session(session)

    def pays(self, session_id: str) -> None:
        self.sessions[session_id].update(
            status="complete",
            payment_status="paid",
            payment_intent=f"pi_test_{uuid4().hex}",
        )

    def expires(self, session_id: str) -> None:
        self.sessions[session_id].update(status="expired", url=None)


class _Session(dict):
    @property
    def id(self) -> str:
        return self["id"]

    @property
    def url(self) -> str | None:
        return self["url"]


@pytest.fixture
def stripe_side(monkeypatch: pytest.MonkeyPatch) -> Stripe:
    monkeypatch.setattr(usage_reset_card, "announce_reset", AsyncMock())
    return Stripe(monkeypatch)


async def _open(user_id: str, key: str = "key-card-1", now: int = NOW):
    return await start_card_reset(
        user_id, key, tier=SubscriptionTier.PRO, price_credits=PRICE, now=now
    )


@pytest.mark.asyncio
async def test_opening_a_checkout_grants_nothing(billing_user, stripe_side: Stripe):
    await save_billing_clock(billing_user, PERIOD)

    checkout = await _open(billing_user)

    assert checkout.url == "https://checkout.stripe.test/0"
    assert (checkout.purchase.status, checkout.purchase.paid_with) == (
        "pending",
        "card",
    )
    assert checkout.purchase.week_start is None
    assert await _resets_now(billing_user) == (None, 0)
    asked = stripe_side.created[0]
    # The whole price, by card, for this user and this purchase only.
    assert asked["mode"] == "payment" and asked["payment_method_types"] == ["card"]
    assert asked["line_items"][0]["price_data"]["unit_amount"] == PRICE
    assert asked["line_items"][0]["quantity"] == 1
    assert asked["metadata"]["user_id"] == billing_user
    assert asked["metadata"]["purchase_key"] == "key-card-1"
    assert asked["customer"] == "cus_test"
    # It can only be paid while the weekly window it was opened in lasts.
    assert asked["expires_at"] == NOW + usage_reset_card.CHECKOUT_LIFETIME_SECONDS
    assert "session_id={CHECKOUT_SESSION_ID}" in asked["success_url"]


@pytest.mark.asyncio
async def test_an_unpaid_checkout_is_never_granted(billing_user, stripe_side: Stripe):
    """Abandoned, declined, or stuck in 3-D Secure: Stripe doesn't call the
    session paid, so nothing is reset, however often it is asked about."""
    await save_billing_clock(billing_user, PERIOD)
    checkout = await _open(billing_user)
    session_id = checkout.purchase.checkout_session_id
    assert session_id

    assert await fulfill_card_reset(session_id, NOW + 60) is None
    stripe_side.expires(session_id)
    assert await fulfill_card_reset(session_id, NOW + 3600) is None

    assert await _resets_now(billing_user) == (None, 0)
    assert (await list_reset_purchases(billing_user))[0].status == "pending"


@pytest.mark.asyncio
async def test_a_paid_checkout_is_granted_once_however_often_stripe_says_so(
    billing_user, stripe_side: Stripe
):
    await save_billing_clock(billing_user, PERIOD)
    checkout = await _open(billing_user)
    session_id = checkout.purchase.checkout_session_id
    assert session_id
    stripe_side.pays(session_id)

    grants = await asyncio.gather(
        *(fulfill_card_reset(session_id, NOW + 90) for _ in range(6))
    )
    late = await fulfill_card_reset(session_id, NOW + 5_000)

    assert all(grant is not None for grant in grants)
    assert sum(1 for grant in grants if grant and grant.granted_now) == 1
    assert late is not None and not late.granted_now
    purchase = late.purchase
    assert (purchase.status, purchase.sequence) == ("granted", 1)
    assert (purchase.week_start, purchase.week_end) == (START, WEEK_END)
    assert purchase.granted_at == NOW + 90
    assert (
        purchase.payment_intent_id == stripe_side.sessions[session_id]["payment_intent"]
    )
    assert await _resets_now(billing_user) == (START, 1)
    assert await UsageReset.prisma().count(where={"userId": billing_user}) == 1
    # No credits moved: the card paid the whole price.
    assert await _charges(billing_user) == []


@pytest.mark.asyncio
async def test_a_session_that_is_not_a_reset_or_not_the_buyers_is_ignored(
    billing_user, stripe_side: Stripe
):
    await save_billing_clock(billing_user, PERIOD)
    checkout = await _open(billing_user)
    session_id = checkout.purchase.checkout_session_id
    assert session_id
    stripe_side.pays(session_id)

    # A top-up's or a subscription's session: no purchase has it.
    assert await fulfill_card_reset("cs_test_something_else", NOW) is None
    # Someone else returning with this session id.
    assert await fulfill_card_reset(session_id, NOW, user_id="another-user") is None
    assert await _resets_now(billing_user) == (None, 0)

    own = await fulfill_card_reset(session_id, NOW, user_id=billing_user)
    assert own is not None and own.granted_now


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"amount_subtotal": PRICE - 1},
        {"mode": "subscription"},
        {
            "metadata": {
                "usage_reset": "1",
                "user_id": "x",
                "purchase_key": "key-card-1",
            }
        },
        {"payment_status": "unpaid"},
    ],
)
async def test_a_session_that_does_not_pay_for_this_purchase_grants_nothing(
    billing_user, stripe_side: Stripe, change: dict
):
    await save_billing_clock(billing_user, PERIOD)
    checkout = await _open(billing_user)
    session_id = checkout.purchase.checkout_session_id
    assert session_id
    stripe_side.pays(session_id)
    stripe_side.sessions[session_id].update(change)

    assert await fulfill_card_reset(session_id, NOW) is None
    assert await _resets_now(billing_user) == (None, 0)


@pytest.mark.asyncio
async def test_a_payment_confirmed_after_the_window_ended_resets_the_new_one(
    billing_user, stripe_side: Stripe
):
    """Stripe's confirmation arrived late: the reset goes to the window the
    customer is in when it is granted, since that is when they get it."""
    await save_billing_clock(billing_user, PERIOD)
    checkout = await _open(billing_user)
    session_id = checkout.purchase.checkout_session_id
    assert session_id
    stripe_side.pays(session_id)

    grant = await fulfill_card_reset(session_id, WEEK_END + 120)

    assert grant is not None and grant.granted_now
    state = await get_usage_activation_state(billing_user)
    clock = clock_from_state(state, WEEK_END + 120)
    assert clock is not None and clock.windows.week_start == WEEK_END
    assert grant.purchase.week_start == WEEK_END
    assert grant.purchase.week_end == clock.windows.week_end
    assert await _resets_now(billing_user) == (WEEK_END, 1)


@pytest.mark.asyncio
async def test_a_card_reset_after_a_credit_reset_is_the_windows_second(
    billing_user, stripe_side: Stripe
):
    await save_billing_clock(billing_user, PERIOD)
    await UserCredit().grant_credits(billing_user, PRICE, "test balance")
    await _buy(billing_user, "key-credits")
    checkout = await _open(billing_user)
    session_id = checkout.purchase.checkout_session_id
    assert session_id
    stripe_side.pays(session_id)

    grant = await fulfill_card_reset(session_id, NOW + 300)

    assert grant is not None and grant.purchase.sequence == 2
    assert await _resets_now(billing_user) == (START, 2)


@pytest.mark.asyncio
async def test_asking_again_returns_the_open_checkout_not_a_second_one(
    billing_user, stripe_side: Stripe
):
    await save_billing_clock(billing_user, PERIOD)

    first = await _open(billing_user)
    again = await _open(billing_user)

    assert again.url == first.url
    assert again.purchase.id == first.purchase.id
    assert len(stripe_side.created) == 1


@pytest.mark.asyncio
async def test_an_expired_checkout_is_replaced_on_the_same_purchase(
    billing_user, stripe_side: Stripe
):
    await save_billing_clock(billing_user, PERIOD)
    first = await _open(billing_user)
    assert first.purchase.checkout_session_id
    stripe_side.expires(first.purchase.checkout_session_id)

    again = await _open(billing_user, now=NOW + 3_600)

    assert again.url == "https://checkout.stripe.test/1"
    assert again.purchase.id == first.purchase.id
    assert again.purchase.checkout_session_id != first.purchase.checkout_session_id
    assert await UsageReset.prisma().count(where={"userId": billing_user}) == 1
    # The expired session can't grant anything any more.
    assert (
        await fulfill_card_reset(first.purchase.checkout_session_id, NOW + 3_700)
        is None
    )


@pytest.mark.asyncio
async def test_two_requests_replacing_an_expired_checkout_end_on_one_session(
    billing_user, stripe_side: Stripe
):
    """Each opens a new session, but the purchase can point at one: both
    customers-to-be are shown that one, so the other can never be paid and
    leave a payment with no purchase."""
    await save_billing_clock(billing_user, PERIOD)
    first = await _open(billing_user)
    assert first.purchase.checkout_session_id
    stripe_side.expires(first.purchase.checkout_session_id)

    one, other = await asyncio.gather(
        _open(billing_user, now=NOW + 3_600), _open(billing_user, now=NOW + 3_600)
    )

    row = await UsageReset.prisma().find_first_or_raise(where={"userId": billing_user})
    attached = stripe_side.sessions[row.stripeCheckoutSessionId or ""]
    assert one.url == other.url == attached["url"]
    assert one.purchase.checkout_session_id == row.stripeCheckoutSessionId
    assert other.purchase.checkout_session_id == row.stripeCheckoutSessionId
    # Paying it grants the reset.
    stripe_side.pays(row.stripeCheckoutSessionId or "")
    grant = await fulfill_card_reset(row.stripeCheckoutSessionId or "", NOW + 3_700)
    assert grant is not None and grant.granted_now


@pytest.mark.asyncio
async def test_a_checkout_that_could_still_be_paid_is_never_replaced(
    billing_user, stripe_side: Stripe
):
    """Completed but not what the purchase asked for, or in a state Stripe
    hasn't settled: no new session takes its place, so a payment on it can't
    end up without a purchase."""
    await save_billing_clock(billing_user, PERIOD)
    first = await _open(billing_user)
    session_id = first.purchase.checkout_session_id
    assert session_id
    stripe_side.sessions[session_id].update(status="complete", url=None)

    with pytest.raises(CheckoutUnsettled):
        await _open(billing_user, now=NOW + 60)

    assert len(stripe_side.created) == 1
    row = await UsageReset.prisma().find_first_or_raise(where={"userId": billing_user})
    assert (row.stripeCheckoutSessionId, row.status) == (session_id, "pending")
    # Once Stripe reports it paid, the same retry grants it.
    stripe_side.pays(session_id)
    settled = await _open(billing_user, now=NOW + 120)
    assert settled.url is None and settled.purchase.status == "granted"


@pytest.mark.asyncio
async def test_a_key_credits_already_paid_for_opens_no_checkout_to_pay_again(
    billing_user, stripe_side: Stripe
):
    await save_billing_clock(billing_user, PERIOD)
    await UserCredit().grant_credits(billing_user, PRICE, "test balance")
    await _buy(billing_user, "key-both-ways")

    checkout = await _open(billing_user, "key-both-ways")

    assert checkout.url is None
    assert (checkout.purchase.status, checkout.purchase.paid_with) == (
        "granted",
        "credits",
    )
    assert stripe_side.created == []


@pytest.mark.asyncio
async def test_asking_again_after_paying_grants_instead_of_charging_again(
    billing_user, stripe_side: Stripe
):
    """The customer paid, the webhook is late, and the app retries the
    purchase: it is granted from the payment made, with no second checkout."""
    await save_billing_clock(billing_user, PERIOD)
    first = await _open(billing_user)
    assert first.purchase.checkout_session_id
    stripe_side.pays(first.purchase.checkout_session_id)

    again = await _open(billing_user, now=NOW + 30)

    assert again.url is None and again.purchase.status == "granted"
    assert len(stripe_side.created) == 1
    assert await _resets_now(billing_user) == (START, 1)


@pytest.mark.asyncio
async def test_a_payment_without_a_billing_clock_is_not_silently_dropped(
    billing_user, stripe_side: Stripe
):
    checkout = await _open(billing_user)
    session_id = checkout.purchase.checkout_session_id
    assert session_id
    stripe_side.pays(session_id)

    # No Stripe period is stored, but the account's own monthly clock is
    # there, so the paid reset is still granted.
    grant = await fulfill_card_reset(session_id, NOW)

    assert grant is not None and grant.purchase.status == "granted"


@pytest.mark.asyncio
async def test_a_refund_of_a_reset_payment_takes_no_credits(
    billing_user, stripe_side: Stripe
):
    await save_billing_clock(billing_user, PERIOD)
    await UserCredit().grant_credits(billing_user, 2_000, "test balance")
    checkout = await _open(billing_user)
    session_id = checkout.purchase.checkout_session_id
    assert session_id
    stripe_side.pays(session_id)
    grant = await fulfill_card_reset(session_id, NOW)
    assert grant is not None and grant.purchase.payment_intent_id

    refund = stripe.Refund.construct_from(
        {
            "id": "re_test_reset",
            "amount": PRICE,
            "payment_intent": grant.purchase.payment_intent_id,
        },
        "sk_test_none",
    )

    assert await UserCredit()._find_top_up(refund) is None
    assert await _balance(billing_user) == 2_000
