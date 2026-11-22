"""The fake Stripe a journeys stack points the platform at.

- ``/stripe/v1/...``: the REST API, for the platform's Stripe SDK
  (``STRIPE_API_BASE=http://scripted_model:4010/stripe``).
- ``/stripe/checkout/<id>``: the hosted checkout page a browser opens.
- ``/__stripe__/...``: the journeys' controls (``control.py``).

The browser reaches the fake through a host port and the platform through
the compose network, so pages it hands out use ``FAKE_STRIPE_PUBLIC_URL``.
"""

import logging
import os
import threading
from collections import deque
from collections.abc import Mapping
from typing import Any

from . import control
from .api import StripeApi
from .checkout import Checkout
from .events import Events
from .forms import decode
from .responses import Response
from .store import Store

logger = logging.getLogger("scripted_model.fake_stripe")

API_PREFIX = "/stripe/v1/"
PAGE_PREFIXES = ("/stripe/checkout/", "/stripe/billing_portal/")
CONTROL_PREFIX = "/__stripe__/"
# Enough for a journeys run; the oldest requests go first.
JOURNAL_LIMIT = 2000


def serves(path: str) -> bool:
    return path.startswith(("/stripe/", CONTROL_PREFIX))


class FakeStripe:
    def __init__(
        self, public_url: str, webhook_url: str = "", webhook_secret: str = ""
    ) -> None:
        self.store = Store()
        self.events = Events(webhook_url, webhook_secret)
        self.api = StripeApi(self.store, public_url.rstrip("/"))
        self.checkout = Checkout(self.store, self.events)
        self._requests: deque[dict[str, Any]] = deque(maxlen=JOURNAL_LIMIT)
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
        if path.startswith(PAGE_PREFIXES):
            return self.checkout.handle(method, path)
        if path.startswith(CONTROL_PREFIX):
            return control.handle(self, method, path, query, body)
        return Response.json(404, {"error": {"message": f"nothing serves {path}"}})

    def requests(self, unserved_only: bool = False) -> list[dict[str, Any]]:
        """The platform's API requests, oldest first."""
        with self._requests_lock:
            return [
                dict(r) for r in self._requests if not (unserved_only and r["served"])
            ]

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
            self._requests.append(
                {"method": method, "path": path, "status": status, "served": served}
            )
        log = logger.info if served else logger.warning
        log(
            "stripe %s %s -> %s%s",
            method,
            path,
            status,
            "" if served else " (unserved)",
        )
        return Response.json(status, payload)
