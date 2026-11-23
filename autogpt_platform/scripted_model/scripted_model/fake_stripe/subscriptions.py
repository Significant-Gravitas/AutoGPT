"""What a subscription goes through in the fake Stripe.

- A paid subscription-mode checkout starts it (``start``): active, its first
  period paid by an invoice charged to the card the checkout took.
- The platform changes it through the API (``update``: whether it cancels at
  the period's end, its metadata) or ends it now (``cancel``). The billing
  portal's buttons cancel it at the period's end or renew it (``update``).
- A journey ends its current period (``end_period``, a control). It is
  cancelled if it was set to cancel then, and renewed with a new paid invoice
  otherwise.

Each change sends the events Stripe sends for it, recorded in the store step
that makes the change, so two changes made at once are announced in the order
they were made. A price change (``items``), a trial or a schedule isn't
modelled: such a call is refused and journaled as unserved (``NotModelled``).
"""

import calendar
import copy
from datetime import datetime, timezone
from typing import Any, Callable

from . import catalog, invoices, resources
from .errors import ApiError, NotModelled
from .events import Events
from .store import Store, StripeObject, now, public

Params = dict[str, Any]

UPDATE_PARAMS = frozenset(
    {"cancel_at_period_end", "metadata", "cancellation_details", "expand"}
)
CANCEL_PARAMS = frozenset({"invoice_now", "prorate", "cancellation_details", "expand"})
# What a cancelled subscription still takes, as Stripe allows.
ENDED_UPDATE_PARAMS = frozenset({"metadata", "cancellation_details", "expand"})
ENDED = ("canceled", "incomplete_expired")


class Subscriptions:
    def __init__(self, store: Store, events: Events, public_url: str) -> None:
        self._store = store
        self._events = events
        self._public_url = public_url

    def start(
        self, session: StripeObject, card: StripeObject
    ) -> tuple[StripeObject, StripeObject, StripeObject]:
        """The subscription paying ``session`` starts, its first invoice and the
        payment. None is stored yet: the checkout stores them as it completes,
        so only the payer who closes the session starts one."""
        prices = [
            (self._price(str(item["price"])), int(item["quantity"]))
            for item in session["_line_items"]
        ]
        start = now()
        interval = prices[0][0]["recurring"]["interval"]
        subscription = resources.subscription(
            session, prices, card["id"], start, period_end(start, interval)
        )
        invoice, payment = self._invoice(subscription, "subscription_create", start)
        subscription["latest_invoice"] = invoice["id"]
        return subscription, invoice, payment

    def announce_start(
        self, session: StripeObject, subscription: StripeObject, invoice: StripeObject
    ) -> None:
        """What Stripe tells the platform once a checkout has started a
        subscription, before the checkout itself completes. A held checkout
        holds these too."""
        self._events.hold_with(session["id"], subscription["id"], invoice["id"])
        self._events.emit("customer.subscription.created", subscription)
        self._events.emit("invoice.paid", invoice)
        self._events.emit("invoice.payment_succeeded", invoice)

    def update(
        self, subscription_id: str, params: Params, *, wait: bool
    ) -> StripeObject:
        """``POST /v1/subscriptions/<id>`` (``wait=False``), or the billing
        portal's cancel and renew."""
        unmodelled = sorted(set(params) - UPDATE_PARAMS)
        if unmodelled:
            raise NotModelled(f"changing a subscription's {', '.join(unmodelled)}")
        previous: Params = {}

        def change(subscription: StripeObject) -> None:
            if subscription["status"] in ENDED and set(params) - ENDED_UPDATE_PARAMS:
                raise ApiError(
                    400,
                    "A canceled subscription can only update its"
                    " cancellation_details and metadata.",
                )
            if "cancel_at_period_end" in params:
                cancels = _boolean(
                    params["cancel_at_period_end"], "cancel_at_period_end"
                )
                _cancel_at_period_end(subscription, cancels, previous)
            if "metadata" in params:
                _merge(subscription, "metadata", params["metadata"], previous)
            if "cancellation_details" in params:
                _merge_details(subscription, params["cancellation_details"], previous)

        def announce(updated: StripeObject) -> list[str]:
            if not previous:
                return []
            return [
                self._events.record(
                    "customer.subscription.updated",
                    updated,
                    previous_attributes=previous,
                )
            ]

        updated, events = self._update(subscription_id, change, announce=announce)
        if wait:
            self._events.wait(*events)
        return public(updated)

    def cancel(
        self, subscription_id: str, params: Params, *, wait: bool
    ) -> StripeObject:
        """``DELETE /v1/subscriptions/<id>``: ends it now, without a final
        invoice."""
        unmodelled = sorted(set(params) - CANCEL_PARAMS)
        if unmodelled:
            raise NotModelled(f"cancelling a subscription with {', '.join(unmodelled)}")
        for flag in ("invoice_now", "prorate"):
            if flag in params and _boolean(params[flag], flag):
                raise NotModelled(f"a final invoice on cancelling ({flag}=true)")
        details = params.get("cancellation_details") or {}

        def change(subscription: StripeObject) -> None:
            if subscription["status"] in ENDED:
                raise ApiError(
                    400, f"This subscription is already {subscription['status']}."
                )
            moment = now()
            subscription.update(
                status="canceled",
                canceled_at=moment,
                ended_at=moment,
                cancellation_details={
                    "comment": details.get("comment"),
                    "feedback": details.get("feedback"),
                    "reason": "cancellation_requested",
                },
            )

        canceled, events = self._update(
            subscription_id,
            change,
            announce=lambda canceled: [
                self._events.record("customer.subscription.deleted", canceled)
            ],
        )
        if wait:
            self._events.wait(*events)
        return public(canceled)

    def end_period(self, subscription_id: str) -> tuple[StripeObject, list[Params]]:
        """Ends the current period now, as if the clock had reached it, and
        returns the subscription and the events it sent, delivered."""
        subscription = self._require(subscription_id)
        if subscription["status"] != "active":
            raise ApiError(
                409,
                f"{subscription_id} is {subscription['status']}: only an active"
                " subscription has a period to end.",
            )
        if subscription["cancel_at_period_end"]:
            return self._end(subscription_id)
        return self._renew(subscription)

    def _end(self, subscription_id: str) -> tuple[StripeObject, list[Params]]:
        moment = now()

        def change(subscription: StripeObject) -> None:
            _require_active(subscription)
            if not subscription["cancel_at_period_end"]:
                raise ApiError(409, "It was renewed while ending; end it again.")
            subscription.update(
                status="canceled",
                cancel_at=moment,
                current_period_end=moment,
                ended_at=moment,
            )

        ended, events = self._update(
            subscription_id,
            change,
            announce=lambda ended: [
                self._events.record("customer.subscription.deleted", ended)
            ],
        )
        return public(ended), self._delivered(events)

    def _renew(self, current: StripeObject) -> tuple[StripeObject, list[Params]]:
        moment = now()
        interval = current["items"]["data"][0]["price"]["recurring"]["interval"]
        renewed = copy.deepcopy(current)
        renewed.update(
            current_period_start=moment, current_period_end=period_end(moment, interval)
        )
        invoice, payment = self._invoice(renewed, "subscription_cycle", moment)
        previous = {
            key: current[key]
            for key in ("current_period_end", "current_period_start", "latest_invoice")
        }

        def change(subscription: StripeObject) -> None:
            _require_active(subscription)
            if subscription["cancel_at_period_end"]:
                raise ApiError(
                    409, "It was set to cancel while renewing; end it again."
                )
            self.number_invoice(invoice)
            subscription.update(
                current_period_start=renewed["current_period_start"],
                current_period_end=renewed["current_period_end"],
                latest_invoice=invoice["id"],
            )

        def announce(updated: StripeObject) -> list[str]:
            return [
                self._events.record(
                    "customer.subscription.updated",
                    updated,
                    previous_attributes=previous,
                ),
                self._events.record("invoice.paid", invoice),
                self._events.record("invoice.payment_succeeded", invoice),
            ]

        updated, events = self._update(
            current["id"], change, also_add=(invoice, payment), announce=announce
        )
        return public(updated), self._delivered(events)

    def number_invoice(self, invoice: StripeObject) -> None:
        """Gives an invoice the next number in its customer's sequence. Call
        it inside the ``Store.update`` that adds the invoice, so two invoices
        made at once can't take the same number."""
        customer = invoice["customer"]
        taken = self._store.find("invoice", lambda other: other["customer"] == customer)
        invoice["number"] = invoices.invoice_number(customer, len(taken) + 1)

    def _invoice(
        self, subscription: StripeObject, billing_reason: str, created: int
    ) -> tuple[StripeObject, StripeObject]:
        describe = {
            item["price"]["id"]: catalog.product_name(self._store, item["price"])
            for item in subscription["items"]["data"]
        }
        return invoices.subscription_invoice(
            subscription,
            billing_reason,
            created,
            describe=describe,
            public_url=self._public_url,
        )

    def _price(self, price_id: str) -> StripeObject:
        price = self._store.get(price_id, "price")
        if price is None:
            raise ApiError(404, f"No such price: '{price_id}'", code="resource_missing")
        return price

    def _require(self, subscription_id: str) -> StripeObject:
        subscription = self._store.get(subscription_id, "subscription")
        if subscription is None:
            raise ApiError(
                404,
                f"No such subscription: '{subscription_id}'",
                code="resource_missing",
                param="id",
            )
        return subscription

    def _update(
        self,
        subscription_id: str,
        change: Callable[[StripeObject], None],
        also_add: tuple[StripeObject, ...] = (),
        announce: Callable[[StripeObject], list[str]] | None = None,
    ) -> tuple[StripeObject, list[str]]:
        """Applies ``change``, and ``announce`` records the events it raises
        for the result in the same store step. Returns the subscription and
        those events' ids, to wait for outside the store's lock."""
        self._require(subscription_id)
        recorded: list[str] = []

        def committed(updated: StripeObject) -> None:
            if announce is not None:
                recorded.extend(announce(updated))

        updated = self._store.update(subscription_id, change, also_add, committed)
        if updated is None:
            raise ApiError(404, f"No such subscription: '{subscription_id}'")
        return updated, recorded

    def _delivered(self, event_ids: list[str]) -> list[Params]:
        """These events, once each has been delivered."""
        self._events.wait(*event_ids)
        return [self._events.get(event_id) or {} for event_id in event_ids]


def period_end(start: int, interval: str) -> int:
    """One billing period after ``start``: a calendar month or year later, on
    the same day where that month has it (31 January + 1 month is 28 or 29
    February), as Stripe bills."""
    if interval not in ("month", "year"):
        raise ValueError(f"the fake Stripe bills monthly or yearly, not per {interval}")
    moment = datetime.fromtimestamp(start, timezone.utc)
    index = moment.month - 1 + (12 if interval == "year" else 1)
    year, month = moment.year + index // 12, index % 12 + 1
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return int(moment.replace(year=year, month=month, day=day).timestamp())


def _cancel_at_period_end(
    subscription: StripeObject, cancels: bool, previous: Params
) -> None:
    if cancels == bool(subscription["cancel_at_period_end"]):
        return
    for key in (
        "cancel_at",
        "cancel_at_period_end",
        "canceled_at",
        "cancellation_details",
    ):
        previous[key] = copy.deepcopy(subscription[key])
    if cancels:
        subscription.update(
            cancel_at=subscription["current_period_end"],
            cancel_at_period_end=True,
            canceled_at=now(),
            cancellation_details={
                "comment": None,
                "feedback": None,
                "reason": "cancellation_requested",
            },
        )
    else:
        subscription.update(
            cancel_at=None,
            cancel_at_period_end=False,
            canceled_at=None,
            cancellation_details=resources.not_cancelled(),
        )


def _merge(subscription: StripeObject, key: str, values: Any, previous: Params) -> None:
    """Sets the given keys of a hash; an empty value unsets one, as Stripe's
    metadata does."""
    if not isinstance(values, dict):
        raise ApiError(400, f"Invalid hash: {key}.", param=key)
    merged = {**subscription[key], **values}
    merged = {name: value for name, value in merged.items() if value != ""}
    if merged != subscription[key]:
        previous[key] = copy.deepcopy(subscription[key])
        subscription[key] = merged


def _merge_details(subscription: StripeObject, values: Any, previous: Params) -> None:
    """``cancellation_details``: only the customer's comment and feedback can
    be set; Stripe sets the reason."""
    if not isinstance(values, dict) or set(values) - {"comment", "feedback"}:
        raise ApiError(
            400,
            "cancellation_details takes comment and feedback.",
            param="cancellation_details",
        )
    details = {**subscription["cancellation_details"], **values}
    if details != subscription["cancellation_details"]:
        # A cancel in the same update may have set them already: what the
        # event reports is what they were before the update.
        previous.setdefault(
            "cancellation_details", copy.deepcopy(subscription["cancellation_details"])
        )
        subscription["cancellation_details"] = details


def _require_active(subscription: StripeObject) -> None:
    if subscription["status"] != "active":
        raise ApiError(409, f"{subscription['id']} is {subscription['status']} now.")


def _boolean(value: Any, param: str) -> bool:
    """A boolean as the API takes one: ``true`` or ``false``, in any case
    (the Python SDK form-encodes ``True``)."""
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("true", "false"):
        return str(value).lower() == "true"
    raise ApiError(400, f"Invalid boolean: {value}", param=param)
