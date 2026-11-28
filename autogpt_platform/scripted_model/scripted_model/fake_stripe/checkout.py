"""The hosted checkout page a session's ``url`` opens, and what paying does.

Paying completes the session the way Stripe does for a card that needs no
extra step, then sends the webhooks, and last redirects to ``success_url``.
The webhooks go out before the redirect, so a journey knows the platform has
heard about the payment by the time the user is back, unless it held the
session's events (``Events.hold``).

- **Pay** (payment mode): a payment intent with the card, saved to the
  customer when the session asked to keep it; the session ``complete`` and
  ``paid``; ``checkout.session.completed``.
- **Subscribe** (subscription mode): the card saved to the customer, an
  active subscription and its first invoice, paid with the card
  (``Subscriptions.start``); ``customer.subscription.created``,
  ``invoice.paid`` and ``invoice.payment_succeeded``, then
  ``checkout.session.completed``.
"""

import html
from collections.abc import Callable

from . import resources
from .errors import ApiError
from .events import Events
from .responses import Response
from .store import Store, StripeObject
from .subscriptions import Subscriptions


class SessionClosed(Exception):
    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


class Checkout:
    def __init__(
        self, store: Store, events: Events, subscriptions: Subscriptions
    ) -> None:
        self._store = store
        self._events = events
        self._subscriptions = subscriptions

    def handle(self, method: str, path: str) -> Response:
        match method, path.strip("/").split("/"):
            case "GET", ["stripe", "checkout", session_id]:
                return self._page(session_id)
            case "POST", ["stripe", "checkout", session_id, "pay"]:
                return self._pay(session_id)
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
            f" {_money(item['amount_total'], item['currency'])}"
            f"{' / ' + html.escape(str(item['interval'])) if item.get('interval') else ''}</li>"
            for item in session["_line_items"]
        )
        email = html.escape(str((buyer or {}).get("email") or ""))
        cancel = html.escape(str(session.get("cancel_url") or ""), quote=True)
        action = "Subscribe" if session["mode"] == "subscription" else "Pay"
        body = (
            "<p>Test mode: the fake Stripe charges nothing.</p>"
            f"<ul>{items}</ul>"
            f"<p>Total: <strong>{_money(session['amount_total'], session['currency'])}"
            "</strong></p>"
            f"<p>Paying as {email}</p>"
            f'<form method="post" action="/stripe/checkout/{html.escape(session_id)}/pay">'
            f'<button type="submit">{action}</button></form>'
            f'<p><a href="{cancel}">Cancel</a></p>'
        )
        return Response.html(200, "Checkout", body)

    def _pay(self, session_id: str) -> Response:
        session = self._store.get(session_id, "checkout.session")
        if session is None:
            return Response.html(404, "Checkout", "<p>No such checkout session.</p>")
        if session["mode"] not in ("payment", "subscription"):
            text = (
                "<p>The fake Stripe completes payment and subscription checkouts.</p>"
            )
            return Response.html(501, "Checkout", text)
        if session["status"] != "open":
            return _closed(session["status"])
        if session["mode"] == "subscription":
            return self._subscribe(session)
        saved = bool(session["_payment_intent_data"].get("setup_future_usage"))
        card = resources.card(session["customer"] if saved else None)
        intent = resources.payment_intent(session, card["id"])
        details = self._details(session)

        def complete(current: StripeObject) -> None:
            _require_open(current)
            current.update(
                status="complete",
                payment_status="paid",
                payment_intent=intent["id"],
                customer_details=details,
                url=None,
            )

        # The card and the payment exist only if this pay closed the session:
        # two payers racing for it leave nothing behind.
        return self._complete(session_id, complete, (card, intent))

    def _subscribe(self, session: StripeObject) -> Response:
        # A subscription keeps the card it charges, which Checkout saves to the
        # customer for the payments to come.
        card = resources.card(session["customer"], allow_redisplay="limited")
        try:
            subscription, invoice, payment = self._subscriptions.start(session, card)
        except ApiError as error:
            message = html.escape(str(error))
            return Response.html(error.status, "Checkout", f"<p>{message}</p>")
        details = self._details(session)

        def complete(current: StripeObject) -> None:
            _require_open(current)
            self._subscriptions.number_invoice(invoice)
            current.update(
                status="complete",
                payment_status="paid",
                subscription=subscription["id"],
                invoice=invoice["id"],
                customer_details=details,
                url=None,
            )

        def announce(completed: StripeObject) -> None:
            self._subscriptions.announce_start(completed, subscription, invoice)

        return self._complete(
            session["id"],
            complete,
            (card, subscription, invoice, payment),
            before_completed=announce,
        )

    def _complete(
        self,
        session_id: str,
        complete: Callable[[StripeObject], None],
        also_add: tuple[StripeObject, ...],
        before_completed: Callable[[StripeObject], None] | None = None,
    ) -> Response:
        """Closes the session with ``complete`` under the store's lock, adding
        ``also_add`` only if this call closed it, then sends the webhooks."""
        try:
            completed = self._store.update(session_id, complete, also_add=also_add)
        except SessionClosed as closed:
            return _closed(closed.status)
        if completed is None:
            return Response.html(404, "Checkout", "<p>No such checkout session.</p>")
        # Outside Store.update, always: each webhook waits for the platform,
        # whose handler calls back into the fake (Session.retrieve) on another
        # of the server's threads, which needs the store's lock.
        if before_completed is not None:
            before_completed(completed)
        self._events.emit("checkout.session.completed", completed)
        success = str(completed["success_url"])
        return Response.redirect(success.replace("{CHECKOUT_SESSION_ID}", session_id))

    def _details(self, session: StripeObject) -> dict:
        return resources.customer_details(self._customer(session))

    def _customer(self, session: StripeObject) -> StripeObject | None:
        customer_id = session.get("customer")
        return self._store.get(customer_id, "customer") if customer_id else None


def _require_open(session: StripeObject) -> None:
    if session["status"] != "open":
        raise SessionClosed(session["status"])


def _closed(status: str) -> Response:
    text = f"<p>This checkout is {html.escape(status)}.</p>"
    return Response.html(409, "Checkout", text)


def _money(amount: int, currency: str | None) -> str:
    """The amount for the page; the currency is the request's, so escaped."""
    return html.escape(resources.money(amount, currency))
