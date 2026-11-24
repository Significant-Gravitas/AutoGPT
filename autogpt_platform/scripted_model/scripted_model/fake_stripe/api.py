"""The Stripe REST API the platform calls, served from the fake's store.

Only the calls the platform's billing pages and webhooks make are modelled
(README.md lists them). Anything else gets Stripe's 404 for an unknown URL,
and ``handle`` reports it unserved so the fake can journal it: a journey then
sees the platform reach for something the fake doesn't model.
"""

import json
import re
import threading
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Any

from . import catalog, resources
from .errors import ApiError, NotModelled
from .forms import as_int
from .line_items import product_name, unit_price
from .store import Store, StripeObject, public
from .subscriptions import Subscriptions

Params = dict[str, Any]
Handler = Callable[[re.Match[str], Params], StripeObject]


class StripeApi:
    def __init__(
        self, store: Store, public_url: str, subscriptions: Subscriptions
    ) -> None:
        self._store = store
        self._public_url = public_url
        self._subscriptions = subscriptions
        self._replies: dict[str, tuple[str, int, StripeObject]] = {}
        self._replies_lock = threading.Lock()
        id_ = r"(?P<id>[^/]+)"
        routes: list[tuple[str, str, Handler]] = [
            ("POST", "/v1/customers", self._create_customer),
            ("GET", "/v1/customers", self._lister("customer", "email")),
            ("GET", f"/v1/customers/{id_}", self._retriever("customer")),
            ("POST", f"/v1/customers/{id_}", self._update_customer),
            ("POST", "/v1/checkout/sessions", self._create_checkout_session),
            (
                "GET",
                "/v1/checkout/sessions",
                self._lister(
                    "checkout.session",
                    "customer",
                    "status",
                    "payment_intent",
                    "subscription",
                ),
            ),
            (
                "GET",
                f"/v1/checkout/sessions/{id_}",
                self._retriever("checkout.session"),
            ),
            ("POST", f"/v1/checkout/sessions/{id_}/expire", self._expire_session),
            ("GET", f"/v1/payment_intents/{id_}", self._retriever("payment_intent")),
            (
                "GET",
                "/v1/payment_methods",
                self._lister("payment_method", "customer", "type"),
            ),
            ("GET", "/v1/subscriptions", self._list_subscriptions),
            ("GET", f"/v1/subscriptions/{id_}", self._retriever("subscription")),
            # Never wait for their webhooks: see events.py.
            (
                "POST",
                f"/v1/subscriptions/{id_}",
                lambda match, params: self._subscriptions.update(
                    match["id"], params, wait=False
                ),
            ),
            (
                "DELETE",
                f"/v1/subscriptions/{id_}",
                lambda match, params: self._subscriptions.cancel(
                    match["id"], params, wait=False
                ),
            ),
            (
                "GET",
                "/v1/invoices",
                self._lister("invoice", "customer", "subscription", "status"),
            ),
            ("GET", f"/v1/invoices/{id_}", self._retriever("invoice")),
            ("POST", "/v1/billing_portal/sessions", self._create_portal_session),
            ("GET", f"/v1/prices/{id_}", self._retriever("price")),
        ]
        self._routes = [(m, re.compile(p), h) for m, p, h in routes]

    def handle(
        self, method: str, path: str, params: Params, headers: Mapping[str, str]
    ) -> tuple[int, StripeObject, bool]:
        """``(status, body, served)``; ``served`` is False for a URL the fake
        doesn't model, or parameters it doesn't (``NotModelled``). ``path``
        starts at ``/v1``; header names are lowercase."""
        if not _test_key(headers.get("authorization") or ""):
            error = ApiError(
                401,
                "Invalid API Key provided: the fake Stripe takes test keys only"
                " (sk_test_...).",
            )
            return error.status, error.body, True
        for route_method, pattern, handler in self._routes:
            match = pattern.fullmatch(path)
            if match and route_method == method:
                key = headers.get("idempotency-key") if method == "POST" else None
                return self._idempotent(key, path, params, match, handler)
        error = ApiError(404, f"Unrecognized request URL ({method}: {path}).")
        return error.status, error.body, False

    def _idempotent(
        self,
        key: str | None,
        path: str,
        params: Params,
        match: re.Match[str],
        handler: Handler,
    ) -> tuple[int, StripeObject, bool]:
        """A retried POST with the same Idempotency-Key gets the first answer,
        as Stripe gives it (Customer.create relies on this)."""
        if key is None:
            return self._run(match, params, handler)
        fingerprint = json.dumps([path, params], sort_keys=True)
        # The lock is held across the handler, so a retry waits for the first
        # answer. No handler may make an outbound call under it: one that
        # called back into the fake would wait on itself.
        with self._replies_lock:
            seen = self._replies.get(key)
            if seen is None:
                status, body, served = self._run(match, params, handler)
                # Stripe keeps no result for a request it refused for its
                # parameters, and the fake's only errors are such refusals.
                if status < 400:
                    self._replies[key] = (fingerprint, status, body)
                return status, body, served
        if seen[0] != fingerprint:
            error = ApiError(
                400,
                "Keys for idempotent requests can only be used with the same"
                " parameters they were first used with.",
                kind="idempotency_error",
            )
            return error.status, error.body, True
        return seen[1], seen[2], True

    def _run(
        self, match: re.Match[str], params: Params, handler: Handler
    ) -> tuple[int, StripeObject, bool]:
        try:
            body = handler(match, params)
        except NotModelled as error:
            return error.status, error.body, False
        except ApiError as error:
            return error.status, error.body, True
        for path in _expansions(params.get("expand")):
            self._expand(body, path.split("."))
        return 200, body, True

    def _expand(self, node: Any, parts: list[str]) -> None:
        """Replaces the id at ``parts`` (``data.subscription`` in a list) with
        its object, as ``expand[]`` asks."""
        if not parts or not isinstance(node, dict):
            return
        head, rest = parts[0], parts[1:]
        if head == "data" and isinstance(node.get("data"), list):
            for item in node["data"]:
                self._expand(item, rest)
            return
        value = node.get(head)
        if isinstance(value, str) and (found := self._store.get(value)):
            node[head] = value = public(found)
        self._expand(value, rest)

    def _require(self, object_id: str, kind: str, param: str = "id") -> StripeObject:
        obj = self._store.get(object_id, kind)
        if obj is None:
            raise ApiError(
                404,
                f"No such {kind}: '{object_id}'",
                code="resource_missing",
                param=param,
            )
        return obj

    def _retriever(self, kind: str) -> Handler:
        def retrieve(match: re.Match[str], params: Params) -> StripeObject:
            return public(self._require(match["id"], kind))

        return retrieve

    def _lister(self, kind: str, *filters: str) -> Handler:
        def list_(match: re.Match[str], params: Params) -> StripeObject:
            found = self._store.find(
                kind,
                lambda obj: all(
                    params.get(f) is None or obj.get(f) == params[f] for f in filters
                ),
            )
            return self._page(kind, match.string, found, params)

        return list_

    def _list_subscriptions(self, match: re.Match[str], params: Params) -> StripeObject:
        """Stripe leaves canceled subscriptions out unless ``status`` asks."""
        wanted = params.get("status")
        found = self._store.find(
            "subscription",
            lambda sub: (params.get("customer") in (None, sub.get("customer")))
            and (
                wanted == "all"
                or sub.get("status") == wanted
                or (wanted is None and sub.get("status") != "canceled")
            ),
        )
        return self._page("subscription", match.string, found, params)

    def _create_customer(self, match: re.Match[str], params: Params) -> StripeObject:
        return public(self._store.add(resources.customer(params)))

    def _update_customer(self, match: re.Match[str], params: Params) -> StripeObject:
        self._require(match["id"], "customer")

        def change(customer: StripeObject) -> None:
            for key, value in params.items():
                if key == "expand":
                    continue
                if isinstance(value, dict) and isinstance(customer.get(key), dict):
                    merged = {**customer[key], **value}
                    # An empty value unsets a metadata key.
                    customer[key] = {k: v for k, v in merged.items() if v != ""}
                else:
                    customer[key] = value

        return public(self._store.update(match["id"], change) or {})

    def _create_checkout_session(
        self, match: re.Match[str], params: Params
    ) -> StripeObject:
        if params.get("customer"):
            self._require(params["customer"], "customer", "customer")
        mode = params.get("mode")
        if mode not in ("payment", "subscription", "setup"):
            raise ApiError(400, "Invalid mode: must be payment, subscription or setup.")
        if mode == "subscription" and not params.get("customer"):
            raise NotModelled("a subscription checkout without a customer")
        # Only the subscription's metadata is modelled: a trial or anything
        # else there would start a plan the fake bills differently.
        subscription_data = params.get("subscription_data")
        unmodelled = (
            sorted(set(subscription_data) - {"metadata"})
            if isinstance(subscription_data, dict)
            else []
        )
        if unmodelled:
            fields = ", ".join(f"subscription_data[{name}]" for name in unmodelled)
            raise NotModelled(f"a checkout with {fields}")
        if not params.get("success_url"):
            raise ApiError(400, "Missing required param: success_url.")
        items = [
            self._line_item(index, item, mode)
            for index, item in enumerate(params.get("line_items") or [])
        ]
        if params["mode"] != "setup" and not items:
            raise ApiError(400, "Missing required param: line_items.")
        if mode == "subscription":
            _require_one_interval(items)
        session = resources.checkout_session(params, items, self._public_url)
        return public(self._store.add(session))

    def _line_item(self, index: int, item: Any, mode: str) -> dict[str, Any]:
        """A checkout's line: an inline price for a payment, a recurring price
        for a subscription, as Stripe takes them."""
        where = f"line_items[{index}]"
        if not isinstance(item, dict):
            raise ApiError(400, "Invalid line item.", param=where)
        if isinstance(price_data := item.get("price_data"), dict):
            if mode == "subscription":
                raise NotModelled("an inline price (price_data) for a subscription")
            return resources.line_item(
                product_name(price_data, f"{where}[price_data]"),
                unit_price(price_data, f"{where}[price_data]"),
                item.get("quantity"),
                price_data.get("currency") or "usd",
            )
        price = self._require(str(item.get("price")), "price", f"{where}[price]")
        recurring = price.get("type") == "recurring"
        if recurring and mode == "payment":
            raise ApiError(
                400,
                "You specified `payment` mode but passed a recurring price. Either"
                " switch to `subscription` mode or use only one-time prices.",
                param=f"{where}[price]",
            )
        if not recurring and mode == "subscription":
            raise NotModelled("a one-time price on a subscription's first invoice")
        return resources.line_item(
            catalog.product_name(self._store, price),
            Decimal(as_int(price.get("unit_amount"))),
            item.get("quantity"),
            price.get("currency") or "usd",
            price,
        )

    def _expire_session(self, match: re.Match[str], params: Params) -> StripeObject:
        self._require(match["id"], "checkout.session")

        def expire(session: StripeObject) -> None:
            if session["status"] != "open":
                raise ApiError(
                    400,
                    "Only Checkout Sessions with a status of open can be expired;"
                    f" this one is {session['status']}.",
                )
            session.update(status="expired", url=None)

        return public(self._store.update(match["id"], expire) or {})

    def _page(
        self, kind: str, url: str, found: list[StripeObject], params: Params
    ) -> StripeObject:
        """One page of a list, newest first: ``limit`` (10) after
        ``starting_after``, which must name a ``kind`` Stripe holds."""
        items = [public(obj) for obj in found]
        after = params.get("starting_after")
        if after:
            ids = [item["id"] for item in items]
            if after in ids:
                items = items[ids.index(after) + 1 :]
            elif self._store.get(str(after), kind) is None:
                raise ApiError(
                    404,
                    f"No such {kind}: '{after}'",
                    code="resource_missing",
                    param="starting_after",
                )
            else:
                items = []
        limit = max(1, min(as_int(params.get("limit"), 10), 100))
        return {
            "object": "list",
            "data": items[:limit],
            "has_more": len(items) > limit,
            "url": url,
        }

    def _create_portal_session(
        self, match: re.Match[str], params: Params
    ) -> StripeObject:
        self._require(str(params.get("customer")), "customer", "customer")
        session = resources.portal_session(params, self._public_url)
        return public(self._store.add(session))


def _require_one_interval(items: list[dict[str, Any]]) -> None:
    """Stripe bills a subscription's prices on one cycle, so they must recur
    alike. Mixing intervals takes flexible billing, which the API version the
    fake models doesn't have. (The catalogue's prices all recur every one
    month or year.)"""
    intervals = sorted({str(item["interval"]) for item in items})
    if len(intervals) > 1:
        raise ApiError(
            400,
            "All of a subscription's prices must have the same billing"
            f" interval; these recur by {' and '.join(intervals)}.",
            param="line_items",
        )


def _test_key(authorization: str) -> bool:
    scheme, _, key = authorization.partition(" ")
    return scheme.lower() == "bearer" and key.strip().startswith(
        ("sk_test_", "rk_test_")
    )


def _expansions(expand: Any) -> list[str]:
    if isinstance(expand, list):
        return [str(path) for path in expand]
    return [str(expand)] if expand else []
