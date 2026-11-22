"""The hosted checkout page a session's ``url`` opens, and what paying does.

Paying completes the session the way Stripe does for a card that needs no
extra step: a payment intent (with the card, saved to the customer when the
session asked to keep it), the session ``complete`` and ``paid``, then the
``checkout.session.completed`` webhook, and last the redirect to
``success_url``. The webhook goes out before the redirect, so a journey
knows the platform has heard about the payment by the time the user is back,
unless it held the session's events (``Events.hold``).
"""

import html

from . import resources
from .events import Events
from .responses import Response
from .store import Store, StripeObject


class SessionClosed(Exception):
    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


class Checkout:
    def __init__(self, store: Store, events: Events) -> None:
        self._store = store
        self._events = events

    def handle(self, method: str, path: str) -> Response:
        match method, path.strip("/").split("/"):
            case "GET", ["stripe", "checkout", session_id]:
                return self._page(session_id)
            case "POST", ["stripe", "checkout", session_id, "pay"]:
                return self._pay(session_id)
            case "GET", ["stripe", "billing_portal", portal_id]:
                return self._portal(portal_id)
        return Response.html(
            404, "Not found", "<p>The fake Stripe has no such page.</p>"
        )

    def _page(self, session_id: str) -> Response:
        session = self._store.get(session_id, "checkout.session")
        if session is None:
            return Response.html(404, "Checkout", "<p>No such checkout session.</p>")
        if session["status"] != "open":
            text = f"<p>This checkout is {html.escape(session['status'])}.</p>"
            return Response.html(200, "Checkout", text)
        buyer = self._customer(session)
        items = "".join(
            f"<li>{html.escape(str(item['description']))} &times; {item['quantity']}:"
            f" {_money(item['amount_total'], item['currency'])}</li>"
            for item in session["_line_items"]
        )
        email = html.escape(str((buyer or {}).get("email") or ""))
        cancel = html.escape(str(session.get("cancel_url") or ""), quote=True)
        body = (
            "<p>Test mode: the fake Stripe charges nothing.</p>"
            f"<ul>{items}</ul>"
            f"<p>Total: <strong>{_money(session['amount_total'], session['currency'])}"
            "</strong></p>"
            f"<p>Paying as {email}</p>"
            f'<form method="post" action="/stripe/checkout/{html.escape(session_id)}/pay">'
            '<button type="submit">Pay</button></form>'
            f'<p><a href="{cancel}">Cancel</a></p>'
        )
        return Response.html(200, "Checkout", body)

    def _pay(self, session_id: str) -> Response:
        session = self._store.get(session_id, "checkout.session")
        if session is None:
            return Response.html(404, "Checkout", "<p>No such checkout session.</p>")
        if session["mode"] != "payment":
            text = "<p>The fake Stripe only completes payment-mode checkouts.</p>"
            return Response.html(501, "Checkout", text)
        if session["status"] != "open":
            text = f"<p>This checkout is {html.escape(session['status'])}.</p>"
            return Response.html(409, "Checkout", text)
        saved = bool(session["_payment_intent_data"].get("setup_future_usage"))
        card = resources.card(session["customer"] if saved else None)
        intent = resources.payment_intent(session, card["id"])
        details = resources.customer_details(self._customer(session))

        def complete(current: StripeObject) -> None:
            if current["status"] != "open":
                raise SessionClosed(current["status"])
            current.update(
                status="complete",
                payment_status="paid",
                payment_intent=intent["id"],
                customer_details=details,
                url=None,
            )

        try:
            # The card and the payment exist only if this pay closed the
            # session: two payers racing for it leave nothing behind.
            completed = self._store.update(
                session_id, complete, also_add=(card, intent)
            )
        except SessionClosed as closed:
            text = f"<p>This checkout is {html.escape(closed.status)}.</p>"
            return Response.html(409, "Checkout", text)
        if completed is None:
            return Response.html(404, "Checkout", "<p>No such checkout session.</p>")
        # Outside Store.update, always: the webhook goes out synchronously, and
        # the platform's handler calls back into the fake (Session.retrieve)
        # on another of the server's threads, which needs the store's lock.
        self._events.emit("checkout.session.completed", completed)
        success = str(completed["success_url"])
        return Response.redirect(success.replace("{CHECKOUT_SESSION_ID}", session_id))

    def _portal(self, portal_id: str) -> Response:
        portal = self._store.get(portal_id, "billing_portal.session")
        if portal is None:
            return Response.html(
                404, "Billing", "<p>No such billing portal session.</p>"
            )
        back = html.escape(str(portal.get("return_url") or ""), quote=True)
        text = f'<p>The fake Stripe has no billing portal.</p><p><a href="{back}">Back</a></p>'
        return Response.html(200, "Billing", text)

    def _customer(self, session: StripeObject) -> StripeObject | None:
        customer_id = session.get("customer")
        return self._store.get(customer_id, "customer") if customer_id else None


def _money(amount: int, currency: str | None) -> str:
    """The amount for the page; the currency is the request's, so escaped."""
    if (currency or "usd").lower() == "usd":
        return f"${amount / 100:,.2f}"
    return f"{amount / 100:,.2f} {html.escape(str(currency).upper())}"
