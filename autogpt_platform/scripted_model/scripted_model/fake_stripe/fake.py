"""The fake Stripe a journeys stack points the platform at.

- ``/stripe/v1/...``: the REST API, for the platform's Stripe SDK
  (``STRIPE_API_BASE=http://scripted_model:4010/stripe``).
- ``/stripe/checkout/<id>``: the hosted checkout page a browser opens.
- ``/stripe/billing_portal/<id>``, ``/stripe/invoices/<id>``: the customer's
  billing portal and hosted invoices (``portal.py``).
- ``/__stripe__/...``: the journeys' controls (``control.py``).

It starts with the plans in ``catalog.json`` for sale.

The browser reaches the fake through a host port and the platform through
the compose network, so pages it hands out use ``FAKE_STRIPE_PUBLIC_URL``.
"""

import logging
import os
import threading
from collections import deque
from collections.abc import Mapping
from typing import Any

from . import catalog, control
from .api import StripeApi
from .checkout import Checkout
from .events import Events
from .forms import decode
from .portal import Portal
from .responses import Response
from .store import Store
from .subscriptions import Subscriptions

logger = logging.getLogger("scripted_model.fake_stripe")

API_PREFIX = "/stripe/v1/"
CHECKOUT_PREFIX = "/stripe/checkout/"
CUSTOMER_PAGE_PREFIXES = ("/stripe/billing_portal/", "/stripe/invoices/")
CONTROL_PREFIX = "/__stripe__/"
# Enough for a journeys run; the oldest requests go first. Unserved requests
# are also kept apart and never dropped, so a journey always sees them.
JOURNAL_LIMIT = 2000


def serves(path: str) -> bool:
    return path.startswith(("/stripe/", CONTROL_PREFIX))


class FakeStripe:
    def __init__(
        self, public_url: str, webhook_url: str = "", webhook_secret: str = ""
    ) -> None:
        public_url = public_url.rstrip("/")
        self.store = Store()
        catalog.seed(self.store)
        self.events = Events(webhook_url, webhook_secret)
        self.subscriptions = Subscriptions(self.store, self.events, public_url)
        self.api = StripeApi(self.store, public_url, self.subscriptions)
        self.checkout = Checkout(self.store, self.events, self.subscriptions)
        self.portal = Portal(self.store, self.subscriptions)
        self._requests: deque[dict[str, Any]] = deque(maxlen=JOURNAL_LIMIT)
        self._unserved: list[dict[str, Any]] = []
        self._last_seq = 0
        self._requests_lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "FakeStripe":
        return cls(
            public_url=os.environ.get(
                "FAKE_STRIPE_PUBLIC_URL", "http://localhost:4010"
            ),
            webhook_url=os.environ.get("FAKE_STRIPE_WEBHOOK_URL", ""),
            webhook_secret=os.environ.get("FAKE_STRIPE_WEBHOOK_SECRET", ""),
        )

    def handle(
        self,
        method: str,
        path: str,
        query: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> Response:
        if path.startswith(API_PREFIX):
            return self._api(method, path.removeprefix("/stripe"), query, body, headers)
        if path.startswith(CHECKOUT_PREFIX):
            return self.checkout.handle(method, path)
        if path.startswith(CUSTOMER_PAGE_PREFIXES):
            return self.portal.handle(method, path)
        if path.startswith(CONTROL_PREFIX):
            return control.handle(self, method, path, query, body)
        return Response.json(404, {"error": {"message": f"nothing serves {path}"}})

    def close(self) -> None:
        """Stops sending webhooks once those already queued are out."""
        self.events.close()

    def requests(
        self, unserved_only: bool = False, after: int = 0
    ) -> list[dict[str, Any]]:
        """The platform's API requests numbered above ``after``, oldest first."""
        with self._requests_lock:
            journal = self._unserved if unserved_only else self._requests
            return [dict(r) for r in journal if r["seq"] > after]

    def last_request(self) -> int:
        """The number of the latest API request: a journey's cursor, to look
        only at requests made after it."""
        with self._requests_lock:
            return self._last_seq

    def _api(
        self,
        method: str,
        path: str,
        query: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> Response:
        encoded = body.decode(errors="replace") if method == "POST" else query
        status, payload, served = self.api.handle(
            method, path, decode(encoded), headers
        )
        with self._requests_lock:
            self._last_seq += 1
            entry = {
                "seq": self._last_seq,
                "method": method,
                "path": path,
                "status": status,
                "served": served,
            }
            self._requests.append(entry)
            if not served:
                self._unserved.append(entry)
        log = logger.info if served else logger.warning
        log(
            "stripe %s %s -> %s%s",
            method,
            path,
            status,
            "" if served else " (unserved)",
        )
        return Response.json(status, payload)
