"""The fake Stripe, spoken to through the Stripe SDK the platform uses.

The platform calls Stripe with the SDK's async methods, and checks webhooks
with ``stripe.Webhook.construct_event``, so these do too: a journey's
checkout only works if the SDK reads what the fake sends back.
"""

import asyncio
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import stripe
from scripted_model.fake_stripe.fake import FakeStripe
from scripted_model.server import ScriptedModelServer

KEY = "sk_test_scripted_model"  # pragma: allowlist secret
SECRET = "whsec_scripted_model_tests"  # pragma: allowlist secret
APP = "http://app.example"
CREDITS = {"name": "AutoGPT Platform Credits"}


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
    def __init__(self, url: str, receiver: Receiver) -> None:
        self.url = url
        self.receiver = receiver

    def control(self, method: str, path: str, body: Any = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.url}/__stripe__/{path}", data=data, method=method
        )
        with urllib.request.urlopen(request, timeout=10) as reply:
            return json.loads(reply.read())

    def pay(self, session: stripe.checkout.Session) -> tuple[int, str]:
        """Clicks Pay on the session's page, as a browser posts its form."""
        request = urllib.request.Request(
            f"{self.url}/stripe/checkout/{session.id}/pay", data=b"", method="POST"
        )
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            return error.code, error.headers.get("Location", "")
        raise AssertionError("paying should redirect")


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
        yield Stack(url, receiver)
    finally:
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
    try:
        return await work
    finally:
        if stripe.default_http_client is not None:
            await stripe.default_http_client.close_async()
