"""The fake Stripe, spoken to through the Stripe SDK the platform uses.

The platform calls Stripe with the SDK's async methods, and checks webhooks
with ``stripe.Webhook.construct_event``, so these do too: a journey's
checkout only works if the SDK reads what the fake sends back.
"""

import asyncio
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import stripe
from scripted_model.fake_stripe.events import API_VERSION, Events
from scripted_model.fake_stripe.fake import FakeStripe
from scripted_model.server import ScriptedModelServer

KEY = "sk_test_scripted_model"  # pragma: allowlist secret
SECRET = "whsec_scripted_model_tests"  # pragma: allowlist secret
APP = "http://app.example"
CREDITS = {"name": "AutoGPT Platform Credits"}
CATALOG = (
    Path(__file__).resolve().parents[1] / "scripted_model/fake_stripe/catalog.json"
)
PRO = "price_journeys_pro_monthly"
PRO_YEARLY = "price_journeys_pro_yearly"
MAX = "price_journeys_max_monthly"
MONTH_S = 28 * 24 * 3600


class Receiver(ThreadingHTTPServer):
    """The platform's webhook route, as far as checking Stripe's signature."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ReceiverHandler)
        self.events: list[stripe.Event] = []


class _ReceiverHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        payload = self.rfile.read(int(self.headers["Content-Length"]))
        receiver = self.server
        assert isinstance(receiver, Receiver)
        event = stripe.Webhook.construct_event(
            payload, self.headers["Stripe-Signature"], SECRET
        )
        receiver.events.append(event)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


class Stack:
    def __init__(self, url: str, receiver: Receiver, fake: FakeStripe) -> None:
        self.url = url
        self.receiver = receiver
        self.fake = fake

    def control(self, method: str, path: str, body: Any = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.url}/__stripe__/{path}", data=data, method=method
        )
        with urllib.request.urlopen(request, timeout=10) as reply:
            return json.loads(reply.read())

    def pay(self, session: stripe.checkout.Session) -> tuple[int, str]:
        """Clicks Pay on the session's page, as a browser posts its form."""
        return self.submit(f"/stripe/checkout/{session.id}/pay")

    def submit(self, path: str) -> tuple[int, str]:
        """Posts a page's form, as a browser does, and returns its redirect."""
        request = urllib.request.Request(f"{self.url}{path}", data=b"", method="POST")
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get("Location", "")
        raise AssertionError(f"posting {path} should redirect")

    def page(self, path: str) -> str:
        with urllib.request.urlopen(f"{self.url}{path}", timeout=10) as page:
            return page.read().decode()

    def received(self, object_id: str) -> list[str]:
        """The types of the events the platform got about an object, in order."""
        return [
            event.type
            for event in self.receiver.events
            if event.data.object["id"] == object_id
        ]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


@pytest.fixture
def stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Stack]:
    receiver = Receiver()
    server = ScriptedModelServer(("127.0.0.1", 0), tmp_path)
    url = "http://127.0.0.1:%d" % server.server_address[1]
    host, port = receiver.server_address[:2]
    server.stripe = FakeStripe(url, f"http://{host!s}:{port}/webhook", SECRET)
    threads = [
        threading.Thread(target=s.serve_forever, daemon=True)
        for s in (server, receiver)
    ]
    for thread in threads:
        thread.start()
    # The platform's settings: STRIPE_API_KEY and STRIPE_API_BASE.
    monkeypatch.setattr(stripe, "api_key", KEY)
    monkeypatch.setattr(stripe, "api_base", f"{url}/stripe")
    monkeypatch.setattr(stripe, "max_network_retries", 0)
    # A client of this test's own, closed with its event loop.
    monkeypatch.setattr(stripe, "default_http_client", None)
    try:
        yield Stack(url, receiver, server.stripe)
    finally:
        # Its webhooks still queued go out to the receiver, then its sender
        # stops: no thread outlives the test.
        server.stripe.close()
        for s in (server, receiver):
            s.shutdown()
            s.server_close()


def test_the_sdk_buys_credits_on_the_fake_checkout(stack: Stack) -> None:
    async def journey() -> None:
        # The idempotency key makes a retried create the same customer.
        customer, again = [
            await stripe.Customer.create_async(
                email="buyer@example.com",
                metadata={"user_id": "user-1"},
                idempotency_key="customer-create-user-1",
            )
            for _ in range(2)
        ]
        assert again.id == customer.id
        session = await stripe.checkout.Session.create_async(
            customer=customer.id,
            mode="payment",
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": "AutoGPT Platform Credits"},
                        "unit_amount": 500,
                    },
                    "quantity": 1,
                }
            ],
            payment_intent_data={"setup_future_usage": "off_session"},
            success_url=f"{APP}/billing?topup=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{APP}/billing?topup=cancel",
        )
        assert session.url and session.payment_status == "unpaid"
        with urllib.request.urlopen(session.url, timeout=10) as page:
            html = page.read().decode()
        assert "$5.00" in html and ">Pay<" in html

        status, location = await asyncio.to_thread(stack.pay, session)
        assert (status, location) == (
            303,
            f"{APP}/billing?topup=success&session_id={session.id}",
        )
        # Delivered before the redirect, and the platform's check accepts it.
        [event] = stack.receiver.events
        assert event.type == "checkout.session.completed"
        assert event.data.object["id"] == session.id
        assert event.data.object["payment_status"] == "paid"

        paid = await stripe.checkout.Session.retrieve_async(
            session.id, expand=["payment_intent"]
        )
        assert paid.status == "complete" and paid.payment_status == "paid"
        assert isinstance(paid.payment_intent, stripe.PaymentIntent)
        assert paid.payment_intent.amount == 500
        open_sessions = await stripe.checkout.Session.list_async(
            customer=customer.id, status="open"
        )
        cards = await stripe.PaymentMethod.list_async(customer=customer.id, type="card")
        plans = await stripe.Subscription.list_async(customer=customer.id)
        assert open_sessions.data == [] and plans.data == []
        assert [card.card.last4 for card in cards.data] == ["4242"]
        portal = await stripe.billing_portal.Session.create_async(
            customer=customer.id, return_url=f"{APP}/billing"
        )
        assert portal.url.startswith(f"{stack.url}/stripe/billing_portal/")

    # asyncio.run is fine here: the backend's rule against it doesn't cover
    # this package, and each run closes the SDK client it made.
    asyncio.run(_with_closed_client(journey()))


def test_a_journey_holds_and_redelivers_webhooks(stack: Stack) -> None:
    async def checkout() -> stripe.checkout.Session:
        customer = await stripe.Customer.create_async(email="late@example.com")
        return await stripe.checkout.Session.create_async(
            customer=customer.id,
            mode="payment",
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": CREDITS,
                        "unit_amount": 900,
                    }
                }
            ],
            success_url=f"{APP}/billing",
        )

    session = asyncio.run(_with_closed_client(checkout()))
    # A typo'd id fails at once instead of leaving the webhook unheld.
    with pytest.raises(urllib.error.HTTPError) as unknown:
        stack.control("POST", "hold?object=cs_test_typo")
    assert unknown.value.code == 404
    stack.control("POST", f"hold?object={session.id}")
    assert stack.pay(session)[0] == 303
    [held] = stack.control("GET", f"events?object={session.id}")["events"]
    assert held["held"] and held["deliveries"] == []
    assert stack.receiver.events == []

    # Delivered late, then again as Stripe retries: one event id.
    for _ in range(2):
        delivery = stack.control("POST", f"events/{held['id']}/deliver")["delivery"]
        assert delivery["status"] == 200
    # Another event type about the same session gets an id of its own.
    other = stack.control(
        "POST",
        "events",
        {"type": "checkout.session.async_payment_succeeded", "object": session.id},
    )
    assert [e.id for e in stack.receiver.events] == [
        held["id"],
        held["id"],
        other["id"],
    ]
    assert other["deliveries"][0]["status"] == 200
    assert other["event"]["data"]["object"]["payment_status"] == "paid"


def test_the_fake_models_the_api_version_the_platform_s_sdk_pins() -> None:
    """Stripe never changes what a pinned API version sends or accepts, so the
    fake can only fall behind when the platform's SDK moves to a newer version.
    This fails on that bump until the fake is checked against the new version's
    changelog and brought up to date."""
    assert API_VERSION == stripe.api_version


def test_the_fake_refuses_what_stripe_would(stack: Stack) -> None:
    async def refused() -> None:
        with pytest.raises(stripe.AuthenticationError):
            await stripe.Customer.create_async(api_key="sk_live_not_for_a_fake")
        with pytest.raises(stripe.InvalidRequestError, match="No such customer"):
            await stripe.Customer.retrieve_async("cus_missing")
        with pytest.raises(stripe.InvalidRequestError, match="No such checkout"):
            await stripe.checkout.Session.list_async(starting_after="cs_test_none")
        # A call the fake doesn't model fails as Stripe fails an unknown URL.
        with pytest.raises(stripe.InvalidRequestError, match="Unrecognized"):
            await stripe.Refund.create_async(payment_intent="pi_missing")
        # An inline price needs a product and an amount. Stripe keeps no
        # result for a request it refused for its parameters, so a corrected
        # retry with the same idempotency key goes through.
        refusals = [
            ({"currency": "usd", "unit_amount": 500}, "`product_data`"),
            ({"currency": "usd", "product_data": CREDITS}, "unit_amount"),
        ]
        for price, refusal in refusals:
            with pytest.raises(stripe.InvalidRequestError, match=refusal):
                await stripe.checkout.Session.create_async(
                    mode="payment",
                    line_items=[{"price_data": price}],
                    success_url=f"{APP}/billing",
                    idempotency_key="checkout-1",
                )
        session = await stripe.checkout.Session.create_async(
            mode="payment",
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": CREDITS,
                        "unit_amount": 0,
                    }
                }
            ],
            success_url=f"{APP}/billing",
            idempotency_key="checkout-1",
        )
        assert session.status == "open" and session.amount_total == 0

    cursor = stack.control("GET", "requests/cursor")["cursor"]
    asyncio.run(_with_closed_client(refused()))
    unserved = stack.control("GET", f"requests?unserved=true&after={cursor}")
    assert [(r["method"], r["path"]) for r in unserved["requests"]] == [
        ("POST", "/v1/refunds")
    ]
    # A journey's cursor leaves out what came before it.
    cursor = stack.control("GET", "requests/cursor")["cursor"]
    assert stack.control("GET", f"requests?unserved=true&after={cursor}") == {
        "requests": []
    }


def test_a_decimal_price_is_totalled_before_it_is_rounded(stack: Stack) -> None:
    """Stripe multiplies a ``unit_amount_decimal`` by the quantity, then
    rounds the line total up to a whole minor unit."""

    async def totals() -> list[int]:
        sessions = [
            await stripe.checkout.Session.create_async(
                mode="payment",
                line_items=[
                    {
                        "price_data": {
                            "currency": "usd",
                            "product_data": CREDITS,
                            "unit_amount_decimal": unit,
                        },
                        "quantity": quantity,
                    }
                ],
                success_url=f"{APP}/billing",
            )
            for unit, quantity in (("0.5", 1000), ("0.05", 30))
        ]
        return [session.amount_total for session in sessions]

    assert asyncio.run(_with_closed_client(totals())) == [500, 2]


async def _with_closed_client(work: Any) -> Any:
    """Runs ``work``, then closes the SDK's client and drops it, so the next
    event loop makes its own (the fixture restores the original after)."""
    try:
        return await work
    finally:
        if stripe.default_http_client is not None:
            await stripe.default_http_client.close_async()
            stripe.default_http_client = None


def test_the_catalog_s_plans_are_for_sale_under_their_ids(stack: Stack) -> None:
    """The journeys stack builds the platform's price flag from catalog.json,
    so every price in it must be one the fake sells, as it says."""
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))

    async def prices() -> list[stripe.Price]:
        return [
            await stripe.Price.retrieve_async(entry["id"])
            for entry in catalog["prices"]
        ]

    sold = asyncio.run(_with_closed_client(prices()))
    assert [
        (p.id, p.lookup_key, p.unit_amount, p.recurring.interval, p.type) for p in sold
    ] == [
        (e["id"], e["lookup_key"], e["unit_amount"], e["interval"], "recurring")
        for e in catalog["prices"]
    ]
    assert {e["lookup_key"] for e in catalog["prices"]} >= {
        "PRO",
        "PRO_YEARLY",
        "MAX",
        "MAX_YEARLY",
    }


def test_the_sdk_subscribes_on_the_fake_checkout(stack: Stack) -> None:
    async def checkout() -> stripe.checkout.Session:
        customer = await stripe.Customer.create_async(email="plan@example.com")
        return await stripe.checkout.Session.create_async(
            **_plan_checkout(customer.id, PRO, "user-1")
        )

    session = asyncio.run(_with_closed_client(checkout()))
    assert session.mode == "subscription" and session.subscription is None
    page = stack.page(f"/stripe/checkout/{session.id}")
    assert "AutoGPT Pro" in page and "$23.00 / month" in page
    assert ">Subscribe<" in page and "Total: <strong>$23.00" in page

    started = time.time()
    status, location = stack.pay(session)
    assert (status, location) == (
        303,
        f"{APP}/settings/billing?subscription=success&session_id={session.id}",
    )
    # Delivered, in Stripe's order, before the redirect.
    paid = stack.control("GET", f"objects/{session.id}")
    sub_id, invoice_id = paid["subscription"], paid["invoice"]
    assert stack.received(sub_id) == ["customer.subscription.created"]
    assert stack.received(invoice_id) == ["invoice.paid", "invoice.payment_succeeded"]
    assert [event.type for event in stack.receiver.events][-1] == (
        "checkout.session.completed"
    )

    async def read() -> None:
        sub = await stripe.Subscription.retrieve_async(sub_id)
        assert sub.status == "active" and sub.latest_invoice == invoice_id
        assert sub.metadata["user_id"] == "user-1"
        [item] = sub["items"].data
        assert (item.price.id, item.quantity) == (PRO, 1)
        assert started - 5 <= sub.current_period_start <= time.time()
        assert sub.current_period_end - sub.current_period_start >= MONTH_S
        assert not sub.cancel_at_period_end and sub.cancel_at is None
        invoice = await stripe.Invoice.retrieve_async(invoice_id)
        assert (invoice.status, invoice.amount_paid, invoice.amount_remaining) == (
            "paid",
            2300,
            0,
        )
        assert invoice.billing_reason == "subscription_create"
        assert invoice.subscription == sub_id and invoice.status_transitions.paid_at
        [line] = invoice.lines.data
        assert line.price is not None
        assert (line.type, line.subscription, line.price.id, line.amount) == (
            "subscription",
            sub_id,
            PRO,
            2300,
        )
        assert line.period.end == sub.current_period_end and not line.proration
        assert isinstance(sub.customer, str)
        live = await stripe.Subscription.list_async(
            customer=sub.customer, status="active"
        )
        assert [s.id for s in live.data] == [sub_id]
        invoices = await stripe.Invoice.list_async(customer=sub.customer, status="paid")
        assert [i.id for i in invoices.data] == [invoice_id]
        cards = await stripe.PaymentMethod.list_async(
            customer=sub.customer, type="card"
        )
        assert [(c.id, c.allow_redisplay) for c in cards.data] == [
            (sub.default_payment_method, "limited")
        ]

    asyncio.run(_with_closed_client(read()))


def test_a_subscription_checkout_takes_one_billing_interval(stack: Stack) -> None:
    """Stripe bills a subscription's prices on one cycle: a checkout mixing
    monthly and yearly prices is refused, in either order, and nothing is
    stored. Prices that share an interval still go together."""

    def plans(customer_id: str, *prices: str) -> dict[str, Any]:
        return {
            **_plan_checkout(customer_id, PRO, "user-1"),
            "line_items": [{"price": price, "quantity": 1} for price in prices],
        }

    async def checkouts() -> stripe.checkout.Session:
        customer = await stripe.Customer.create_async(email="user-1@example.com")
        for prices in ((PRO, PRO_YEARLY), (PRO_YEARLY, PRO)):
            with pytest.raises(stripe.InvalidRequestError, match="billing interval"):
                await stripe.checkout.Session.create_async(
                    **plans(customer.id, *prices)
                )
        stored = await stripe.checkout.Session.list_async(customer=customer.id)
        assert stored.data == []
        return await stripe.checkout.Session.create_async(
            **plans(customer.id, PRO, MAX)
        )

    both = asyncio.run(_with_closed_client(checkouts()))
    assert both.status == "open" and both.amount_total == 2300 + 9700


def test_a_plan_set_to_cancel_at_the_period_end_and_back(stack: Stack) -> None:
    """``Subscription.modify`` answers first, then the webhook follows, with
    what changed in ``previous_attributes``."""
    sub_id = subscribe(stack)["subscription"]

    async def cancel_then_resume() -> list[stripe.Subscription]:
        return [
            await stripe.Subscription.modify_async(sub_id, cancel_at_period_end=flag)
            for flag in (True, True, False)
        ]

    cancelled, unchanged, resumed = asyncio.run(
        _with_closed_client(cancel_then_resume())
    )
    assert cancelled.status == "active" and cancelled.cancel_at_period_end
    assert cancelled.cancel_at == cancelled.current_period_end
    assert cancelled.canceled_at and unchanged.canceled_at == cancelled.canceled_at
    assert cancelled.cancellation_details.reason == "cancellation_requested"
    assert not resumed.cancel_at_period_end and resumed.cancel_at is None
    # Setting what was already set changed nothing, so sent nothing.
    updates = _wait_for(
        lambda: _events(stack, sub_id, "customer.subscription.updated"),
        lambda events: len(events) >= 2 and all(e["deliveries"] for e in events),
    )
    assert [e["event"]["data"]["previous_attributes"] for e in updates] == [
        {
            "cancel_at": None,
            "cancel_at_period_end": False,
            "canceled_at": None,
            "cancellation_details": {"comment": None, "feedback": None, "reason": None},
        },
        {
            "cancel_at": cancelled.current_period_end,
            "cancel_at_period_end": True,
            "canceled_at": cancelled.canceled_at,
            "cancellation_details": {
                "comment": None,
                "feedback": None,
                "reason": "cancellation_requested",
            },
        },
    ]
    assert [d["status"] for e in updates for d in e["deliveries"]] == [200, 200]


def test_one_update_that_cancels_and_comments_reports_the_details_before_it(
    stack: Stack,
) -> None:
    """``previous_attributes`` holds what changed as it was before the
    update, even when the update changes it twice (the cancel sets a reason,
    then the comment is added)."""
    sub_id = subscribe(stack)["subscription"]
    before = stack.control("GET", f"objects/{sub_id}")["cancellation_details"]

    async def cancel() -> stripe.Subscription:
        return await stripe.Subscription.modify_async(
            sub_id,
            cancel_at_period_end=True,
            cancellation_details={"comment": "too pricey"},
        )

    cancelled = asyncio.run(_with_closed_client(cancel()))
    assert cancelled.cancellation_details.reason == "cancellation_requested"
    assert cancelled.cancellation_details.comment == "too pricey"
    [update] = _events(stack, sub_id, "customer.subscription.updated")
    previous = update["event"]["data"]["previous_attributes"]
    assert previous["cancellation_details"] == before
    assert previous["cancel_at_period_end"] is False


def test_changes_to_one_subscription_are_announced_in_the_order_they_were_made(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change made first is announced first, even when another change to the
    same subscription is made between its write and its webhook: the platform
    takes the last event it hears as Stripe's latest state."""
    sub_id = subscribe(stack)["subscription"]
    fake = stack.fake
    write = _PausesAfterFirstWrite(fake.store.update)
    monkeypatch.setattr(fake.store, "update", write)
    cancel, resume = [
        threading.Thread(
            target=fake.subscriptions.update,
            args=(sub_id, {"cancel_at_period_end": cancels}),
            kwargs={"wait": False},
        )
        for cancels in (True, False)
    ]
    cancel.start()
    try:
        assert write.paused.wait(5), "the cancel was never written"
        # Code that records a change's event apart from its write lets the
        # resume be written and announced now, ahead of the cancel.
        resume.start()
        resume.join(10)
    finally:
        write.release.set()
        cancel.join(10)
        if resume.ident is not None:
            resume.join(10)
    assert not cancel.is_alive() and not resume.is_alive()
    updates = _events(stack, sub_id, "customer.subscription.updated")
    announced = [u["event"]["data"]["object"]["cancel_at_period_end"] for u in updates]
    assert announced == [True, False]
    assert stack.control("GET", f"objects/{sub_id}")["cancel_at_period_end"] is False


def test_a_period_ends_in_a_renewal_or_with_the_cancelled_plan(stack: Stack) -> None:
    session = subscribe(stack)
    sub_id = session["subscription"]
    before = stack.control("GET", f"objects/{sub_id}")

    renewed = stack.control("POST", f"subscriptions/{sub_id}/end_period")
    sub = renewed["subscription"]
    assert (
        sub["status"] == "active" and sub["latest_invoice"] != before["latest_invoice"]
    )
    assert before["current_period_start"] <= sub["current_period_start"] <= time.time()
    assert sub["current_period_end"] - sub["current_period_start"] >= MONTH_S
    assert [e["type"] for e in renewed["events"]] == [
        "customer.subscription.updated",
        "invoice.paid",
        "invoice.payment_succeeded",
    ]
    assert all(e["deliveries"][0]["status"] == 200 for e in renewed["events"])
    invoice = stack.control("GET", f"objects/{sub['latest_invoice']}")
    assert (invoice["billing_reason"], invoice["status"], invoice["amount_paid"]) == (
        "subscription_cycle",
        "paid",
        2300,
    )
    assert invoice["number"].endswith("-0002")

    # Cancelled in the billing portal, as a customer does it: the webhook is
    # in before the page answers.
    portal = asyncio.run(_with_closed_client(_portal(sub["customer"])))
    page = stack.page(f"/stripe/billing_portal/{portal.id}")
    assert "AutoGPT Pro" in page and "$23.00 per month" in page and "renews on" in page
    action = f"/stripe/billing_portal/{portal.id}/subscriptions/{sub_id}/cancel"
    assert stack.submit(action) == (303, f"/stripe/billing_portal/{portal.id}")
    assert stack.received(sub_id)[-1] == "customer.subscription.updated"
    page = stack.page(f"/stripe/billing_portal/{portal.id}")
    assert "will be canceled on" in page and ">Renew plan<" in page

    ended = stack.control("POST", f"subscriptions/{sub_id}/end_period")
    assert [e["type"] for e in ended["events"]] == ["customer.subscription.deleted"]
    assert ended["events"][0]["deliveries"][0]["status"] == 200
    gone = ended["subscription"]
    assert gone["status"] == "canceled" and gone["ended_at"]
    assert gone["cancellation_details"]["reason"] == "cancellation_requested"
    assert "You have no plans." in stack.page(f"/stripe/billing_portal/{portal.id}")
    with pytest.raises(urllib.error.HTTPError) as again:
        stack.control("POST", f"subscriptions/{sub_id}/end_period")
    assert again.value.code == 409

    async def after() -> None:
        assert (
            await stripe.Subscription.list_async(customer=gone["customer"])
        ).data == []
        all_ = await stripe.Subscription.list_async(
            customer=gone["customer"], status="all"
        )
        assert [s.id for s in all_.data] == [sub_id]
        with pytest.raises(stripe.InvalidRequestError, match="canceled subscription"):
            await stripe.Subscription.modify_async(sub_id, cancel_at_period_end=False)

    asyncio.run(_with_closed_client(after()))


def test_a_subscription_cancelled_now(stack: Stack) -> None:
    sub_id = subscribe(stack)["subscription"]

    async def cancel() -> stripe.Subscription:
        canceled = await stripe.Subscription.cancel_async(
            sub_id, cancellation_details={"comment": "autogpt:account_deleted"}
        )
        with pytest.raises(stripe.InvalidRequestError, match="already canceled"):
            await stripe.Subscription.cancel_async(sub_id)
        return canceled

    canceled = asyncio.run(_with_closed_client(cancel()))
    assert canceled.status == "canceled" and canceled.ended_at
    assert canceled.cancellation_details.comment == "autogpt:account_deleted"
    [deleted] = _wait_for(
        lambda: _events(stack, sub_id, "customer.subscription.deleted"),
        lambda events: bool(events and events[0]["deliveries"]),
    )
    assert deleted["deliveries"][0]["status"] == 200
    assert deleted["event"]["data"]["object"]["status"] == "canceled"


def test_a_held_checkout_holds_the_plan_s_events_too(stack: Stack) -> None:
    async def checkout() -> stripe.checkout.Session:
        customer = await stripe.Customer.create_async(email="late@example.com")
        return await stripe.checkout.Session.create_async(
            **_plan_checkout(customer.id, PRO, "user-2")
        )

    session = asyncio.run(_with_closed_client(checkout()))
    stack.control("POST", f"hold?object={session.id}")
    assert stack.pay(session)[0] == 303
    customer = stack.control("GET", f"objects/{session.id}")["customer"]
    held = stack.control("GET", f"events?customer={customer}")["events"]
    assert [(e["type"], e["held"], e["deliveries"]) for e in held] == [
        ("customer.subscription.created", True, []),
        ("invoice.paid", True, []),
        ("invoice.payment_succeeded", True, []),
        ("checkout.session.completed", True, []),
    ]
    assert stack.receiver.events == []
    for event in held:
        stack.control("POST", f"events/{event['id']}/deliver")
    assert [e.type for e in stack.receiver.events] == [e["type"] for e in held]


def test_what_the_fake_doesn_t_model_is_refused_and_journaled(stack: Stack) -> None:
    sub_id = subscribe(stack)["subscription"]
    customer = stack.control("GET", f"objects/{sub_id}")["customer"]
    cursor = stack.control("GET", "requests/cursor")["cursor"]

    async def unmodelled() -> None:
        sub = await stripe.Subscription.retrieve_async(sub_id)
        item = sub["items"].data[0].id
        with pytest.raises(stripe.InvalidRequestError, match="doesn't model"):
            await stripe.Subscription.modify_async(
                sub_id,
                items=[{"id": item, "price": MAX}],
                proration_behavior="always_invoice",
            )
        with pytest.raises(stripe.InvalidRequestError, match="doesn't model"):
            await stripe.Subscription.cancel_async(sub_id, invoice_now=True)
        with pytest.raises(stripe.InvalidRequestError, match="doesn't model"):
            await stripe.checkout.Session.create_async(
                mode="subscription",
                line_items=[{"price": PRO, "quantity": 1}],
                success_url=f"{APP}/billing",
            )
        # A trial would start a plan the fake bills at once.
        trial = _plan_checkout(customer, PRO, "user-1")
        trial["subscription_data"] = {
            **trial["subscription_data"],
            "trial_period_days": 7,
        }
        with pytest.raises(stripe.InvalidRequestError, match="trial_period_days"):
            await stripe.checkout.Session.create_async(**trial)
        # Stripe's own refusal, which the fake models.
        with pytest.raises(stripe.InvalidRequestError, match="recurring price"):
            await stripe.checkout.Session.create_async(
                mode="payment",
                line_items=[{"price": PRO, "quantity": 1}],
                success_url=f"{APP}/billing",
            )
        # Nothing changed.
        same = await stripe.Subscription.retrieve_async(sub_id)
        assert same["items"].data[0].price.id == PRO and same.status == "active"

    asyncio.run(_with_closed_client(unmodelled()))
    unserved = stack.control("GET", f"requests?unserved=true&after={cursor}")
    assert [(r["method"], r["path"]) for r in unserved["requests"]] == [
        ("POST", f"/v1/subscriptions/{sub_id}"),
        ("DELETE", f"/v1/subscriptions/{sub_id}"),
        ("POST", "/v1/checkout/sessions"),
        ("POST", "/v1/checkout/sessions"),
    ]


def test_a_delivery_cut_short_is_recorded_and_the_next_one_still_goes_out() -> None:
    """A webhook answer that ends before its Content-Length is a failed
    delivery, and the deliveries after it still go out."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ShortBody)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    events = Events("http://%s:%d/webhook" % server.server_address[:2], SECRET)
    try:
        customer = {"id": "cus_short", "object": "customer"}
        records: list[dict[str, Any]] = []

        def emit_two() -> None:
            for event_type in ("customer.updated", "customer.deleted"):
                records.append(events.emit(event_type, customer))

        sender = threading.Thread(target=emit_two, daemon=True)
        sender.start()
        sender.join(timeout=30)
        assert not sender.is_alive(), "an emit waited forever: deliveries stopped"
        assert [r["deliveries"][0]["status"] for r in records] == [None, None]
        assert all("IncompleteRead" in r["deliveries"][0]["error"] for r in records)
    finally:
        events.close()
        server.shutdown()
        server.server_close()


def test_events_go_out_in_the_order_they_are_recorded(
    stack: Stack, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An event held between being recorded and being queued can't be
    overtaken: one recorded after it waits its turn."""
    host, port = stack.receiver.server_address[:2]
    events = Events(f"http://{host!s}:{port}/webhook", SECRET)
    held = _FirstPutWaits()
    monkeypatch.setattr(events, "_queue", held)
    first, second = [
        threading.Thread(
            target=events.emit,
            args=("customer.updated", {"id": customer_id, "object": "customer"}),
            kwargs={"wait": False},
        )
        for customer_id in ("cus_first", "cus_second")
    ]
    first.start()
    try:
        assert held.paused.wait(5)
        second.start()
        # Code that queues an event apart from recording it lets the second
        # one in now; correct code holds it until the first is queued.
        held.overtaken.wait(1)
    finally:
        held.release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    try:
        received = _wait_for(
            lambda: [e.data.object["id"] for e in stack.receiver.events],
            lambda ids: len(ids) >= 2,
        )
        assert received == ["cus_first", "cus_second"]
        assert [r["object"] for r in events.find()] == ["cus_first", "cus_second"]
    finally:
        events.close()


def test_closing_never_leaves_two_senders_delivering_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event emitted while ``close`` waits for the sender goes out from
    that sender, after the ones queued before it, and the sender then stops:
    no second sender ever delivers beside it."""
    before = set(threading.enumerate())
    receiver = _GatedReceiver()
    threading.Thread(target=receiver.serve_forever, daemon=True).start()
    events = Events("http://%s:%d/webhook" % receiver.server_address[:2], SECRET)
    queued = _NotesStop()
    monkeypatch.setattr(events, "_queue", queued)
    closing = threading.Thread(target=events.close)
    try:
        events.emit("customer.updated", _customer(1), wait=False)
        assert receiver.first_arrived.wait(5), "the first delivery never arrived"
        events.emit("customer.updated", _customer(2), wait=False)
        closing.start()
        assert queued.stopped.wait(5), "close() never queued its stop"
        events.emit("customer.updated", _customer(3), wait=False)
        # A second sender started now would send cus_2 while cus_1 is open.
        assert not receiver.overlapped.wait(1), "two deliveries were open at once"
        receiver.release.set()
        # close() stops waiting after its timeout, which a paused runner can
        # outlast: wait for the deliveries and the sender themselves.
        delivered = _wait_for(
            lambda: list(receiver.received), lambda ids: len(ids) >= 3, timeout=30
        )
        assert delivered == ["cus_1", "cus_2", "cus_3"]
        _wait_for(lambda: _senders(before), lambda alive: not alive, timeout=30)
        closing.join(30)
        assert not closing.is_alive(), "close() never returned"
    finally:
        receiver.release.set()
        if closing.ident is None:
            # Stopped before close() was asked for: the sender still waits
            # for its stop.
            events.close()
        else:
            closing.join(30)
        receiver.shutdown()
        receiver.server_close()


def _customer(number: int) -> dict[str, Any]:
    return {"id": f"cus_{number}", "object": "customer"}


def _senders(before: set[threading.Thread]) -> list[threading.Thread]:
    """The webhook senders started since ``before`` that are still alive."""
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "fake-stripe-webhooks" and thread not in before
    ]


class _ShortBody(BaseHTTPRequestHandler):
    """Answers with fewer bytes than its Content-Length says, then hangs up."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        self.wfile.write(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort")
        self.close_connection = True

    def log_message(self, format: str, *args: Any) -> None:
        pass


class _FirstPutWaits:
    """The events' queue, whose first ``put`` waits for ``release``. A put
    that comes in meanwhile sets ``overtaken``."""

    def __init__(self) -> None:
        self._inner: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self.paused = threading.Event()
        self.release = threading.Event()
        self.overtaken = threading.Event()

    def put(self, item: Any) -> None:
        if not self.paused.is_set():
            self.paused.set()
            # However long a slow runner takes: the test always releases it.
            self.release.wait()
        elif not self.release.is_set():
            self.overtaken.set()
        self._inner.put(item)

    def get(self) -> Any:
        return self._inner.get()

    def empty(self) -> bool:
        return self._inner.empty()


class _PausesAfterFirstWrite:
    """``Store.update``, whose first call waits for ``release`` once it has
    written."""

    def __init__(self, update: Callable[..., Any]) -> None:
        self._update = update
        self.paused = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self._update(*args, **kwargs)
        if not self.paused.is_set():
            self.paused.set()
            # However long a slow runner takes: the test always releases it.
            self.release.wait()
        return result


class _NotesStop:
    """The events' queue, setting ``stopped`` once ``close`` queues its stop."""

    def __init__(self) -> None:
        self._inner: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self.stopped = threading.Event()

    def put(self, item: Any) -> None:
        self._inner.put(item)
        if item is None:
            self.stopped.set()

    def get(self) -> Any:
        return self._inner.get()

    def empty(self) -> bool:
        return self._inner.empty()


class _GatedReceiver(ThreadingHTTPServer):
    """The webhook route. It holds the first delivery until ``release`` and
    sets ``overlapped`` when a delivery arrives while another is still open."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _GatedHandler)
        self.lock = threading.Lock()
        self.open = 0
        self.received: list[str] = []
        self.first_arrived = threading.Event()
        self.release = threading.Event()
        self.overlapped = threading.Event()


class _GatedHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        receiver = self.server
        assert isinstance(receiver, _GatedReceiver)
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with receiver.lock:
            receiver.open += 1
            if receiver.open > 1:
                receiver.overlapped.set()
            first = not receiver.received
            receiver.received.append(payload["data"]["object"]["id"])
        if first:
            receiver.first_arrived.set()
            receiver.release.wait(30)
        with receiver.lock:
            receiver.open -= 1
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        pass


def subscribe(stack: Stack, price: str = PRO, user_id: str = "user-1") -> Any:
    """A customer who bought ``price`` on the fake's checkout, as the platform
    sells a plan. Returns the completed session as the fake holds it."""

    async def checkout() -> stripe.checkout.Session:
        customer = await stripe.Customer.create_async(email=f"{user_id}@example.com")
        return await stripe.checkout.Session.create_async(
            **_plan_checkout(customer.id, price, user_id)
        )

    session = asyncio.run(_with_closed_client(checkout()))
    assert stack.pay(session)[0] == 303
    return stack.control("GET", f"objects/{session.id}")


def _plan_checkout(customer_id: str, price: str, user_id: str) -> dict[str, Any]:
    """What ``create_subscription_checkout`` sends."""
    return {
        "customer": customer_id,
        "mode": "subscription",
        "line_items": [{"price": price, "quantity": 1}],
        "success_url": f"{APP}/settings/billing?subscription=success"
        "&session_id={CHECKOUT_SESSION_ID}",
        "cancel_url": f"{APP}/settings/billing?subscription=cancelled",
        "subscription_data": {
            "metadata": {"user_id": user_id, "tier": "PRO", "billing_cycle": "monthly"}
        },
        "allow_promotion_codes": True,
        "automatic_tax": {"enabled": True},
        "billing_address_collection": "auto",
        "customer_update": {"address": "auto"},
        "saved_payment_method_options": {
            "payment_method_save": "enabled",
            "allow_redisplay_filters": ["always", "limited"],
        },
        "metadata": {},
    }


async def _portal(customer_id: str) -> stripe.billing_portal.Session:
    return await stripe.billing_portal.Session.create_async(
        customer=customer_id, return_url=f"{APP}/settings/billing"
    )


def _events(stack: Stack, object_id: str, event_type: str) -> list[dict[str, Any]]:
    events = stack.control("GET", f"events?object={object_id}")["events"]
    return [event for event in events if event["type"] == event_type]


def _wait_for(
    read: Callable[[], Any], done: Callable[[Any], bool], timeout: float = 10
) -> Any:
    """``read()`` once ``done`` accepts it: a webhook the API's answer came
    before arrives on its own. Fails with what it last read after
    ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        value = read()
        if done(value):
            return value
        if time.monotonic() > deadline:
            raise AssertionError(
                f"still waiting after {timeout} s; last read {value!r}"
            )
        time.sleep(0.05)
