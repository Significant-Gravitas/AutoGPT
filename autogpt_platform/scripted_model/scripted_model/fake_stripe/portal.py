"""The customer's own pages: the billing portal and hosted invoices.

The platform sends a customer to a billing portal session's ``url`` to manage
their plan, and links each invoice's ``hosted_invoice_url``. The portal lists
the customer's plans that haven't ended. **Cancel plan** sets one to cancel at
the end of its period and **Renew plan** takes that back, as Stripe's portal
does when it cancels at the period's end. Each sends
``customer.subscription.updated`` before the page answers. **Back** returns to
the session's ``return_url``.
"""

import html
from datetime import datetime, timezone

from . import catalog, resources
from .errors import ApiError
from .responses import Response
from .store import Store, StripeObject
from .subscriptions import ENDED, Subscriptions


class Portal:
    def __init__(self, store: Store, subscriptions: Subscriptions) -> None:
        self._store = store
        self._subscriptions = subscriptions

    def handle(self, method: str, path: str) -> Response:
        match method, path.strip("/").split("/"):
            case "GET", ["stripe", "billing_portal", portal_id]:
                return self._page(portal_id)
            case "POST", [
                "stripe",
                "billing_portal",
                portal_id,
                "subscriptions",
                subscription_id,
                ("cancel" | "renew") as action,
            ]:
                return self._change(portal_id, subscription_id, action)
            case "GET", ["stripe", "invoices", invoice_id]:
                return self._invoice(invoice_id)
        return Response.html(
            404, "Not found", "<p>The fake Stripe has no such page.</p>"
        )

    def _page(self, portal_id: str) -> Response:
        portal = self._store.get(portal_id, "billing_portal.session")
        if portal is None:
            return _missing("billing portal session")
        plans = self._store.find(
            "subscription",
            lambda sub: sub["customer"] == portal["customer"]
            and sub["status"] not in ENDED,
        )
        sections = "".join(self._plan(portal_id, plan) for plan in plans)
        back = html.escape(str(portal.get("return_url") or ""), quote=True)
        body = (
            "<p>Test mode: the fake Stripe's billing portal.</p>"
            f"{sections or '<p>You have no plans.</p>'}"
            f'<p><a href="{back}">Back</a></p>'
        )
        return Response.html(200, "Billing", body)

    def _plan(self, portal_id: str, plan: StripeObject) -> str:
        price = plan["items"]["data"][0]["price"]
        name = html.escape(catalog.product_name(self._store, price))
        cost = resources.money(int(price["unit_amount"]), price["currency"])
        interval = price["recurring"]["interval"]
        ends = _date(plan["current_period_end"])
        if plan["cancel_at_period_end"]:
            state, action, label = (
                f"Your plan will be canceled on {ends}.",
                "renew",
                "Renew plan",
            )
        else:
            state, action, label = (
                f"Your plan renews on {ends}.",
                "cancel",
                "Cancel plan",
            )
        target = (
            f"{resources.PORTAL_PAGE}/{html.escape(portal_id)}/subscriptions/"
            f"{html.escape(plan['id'])}/{action}"
        )
        return (
            f'<section aria-label="{name}"><h2>{name}</h2>'
            f"<p>{html.escape(cost)} per {html.escape(interval)}</p>"
            f"<p>{html.escape(state)}</p>"
            f'<form method="post" action="{target}">'
            f'<button type="submit">{label}</button></form></section>'
        )

    def _change(self, portal_id: str, subscription_id: str, action: str) -> Response:
        portal = self._store.get(portal_id, "billing_portal.session")
        if portal is None:
            return _missing("billing portal session")
        plan = self._store.get(subscription_id, "subscription")
        if plan is None or plan["customer"] != portal["customer"]:
            return _missing("plan")
        try:
            self._subscriptions.update(
                subscription_id, {"cancel_at_period_end": action == "cancel"}, wait=True
            )
        except ApiError as error:
            message = html.escape(str(error))
            return Response.html(error.status, "Billing", f"<p>{message}</p>")
        return Response.redirect(f"{resources.PORTAL_PAGE}/{portal_id}")

    def _invoice(self, invoice_id: str) -> Response:
        invoice = self._store.get(invoice_id, "invoice")
        if invoice is None:
            return _missing("invoice")
        lines = "".join(
            f"<li>{html.escape(str(line['description']))}:"
            f" {html.escape(resources.money(line['amount'], line['currency']))}</li>"
            for line in invoice["lines"]["data"]
        )
        total = resources.money(invoice["total"], invoice["currency"])
        body = (
            f"<p>Invoice {html.escape(str(invoice['number']))}</p>"
            f"<ul>{lines}</ul>"
            f"<p>Total: <strong>{html.escape(total)}</strong></p>"
            f"<p>Status: {html.escape(str(invoice['status']))}</p>"
        )
        return Response.html(200, "Invoice", body)


def _date(timestamp: int) -> str:
    moment = datetime.fromtimestamp(timestamp, timezone.utc)
    return f"{moment:%B} {moment.day}, {moment.year}"


def _missing(what: str) -> Response:
    return Response.html(404, "Billing", f"<p>No such {what}.</p>")
